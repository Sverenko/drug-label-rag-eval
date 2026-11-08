"""Every Cohere call in the project goes through CachedCohere.

- Responses are cached in .cache/cohere.sqlite, keyed by a hash of the endpoint, model,
  parameters and input, so running anything a second time makes no API calls.
  Embeddings are cached per text, so re-chunking only re-embeds the texts that changed.
- Each cached response keeps the latency measured when the call was made, so results
  report real API latency even when they're rebuilt from the cache.
- Every attempt goes into a call ledger in the same database. It keeps the running total
  against the trial key's monthly calls (`python -m labelrag.usage`), and a run stops
  before it would go over.
- Calls are throttled to the trial key's per-minute limits. 429s, 5xx errors and network
  failures are retried with backoff here, where every retry is counted; the SDK's own
  retries are off because they'd spend calls without anyone seeing.
"""

import hashlib
import itertools
import json
import sqlite3
import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import cohere
import httpx
import numpy as np
from cohere.types import AssistantChatMessageV2, Document, SystemChatMessageV2, UserChatMessageV2

from labelrag import config

KEY_VERSION = 1  # bump to invalidate every cache entry if the stored format changes
EMBED_BATCH = 96  # most texts Cohere accepts in one embed call
MAX_ATTEMPTS = 4
MAX_BACKOFF_S = 60.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    input_type TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    vector BLOB NOT NULL,
    latency_ms REAL NOT NULL,
    batch_size INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS responses (
    key TEXT PRIMARY KEY,
    endpoint TEXT NOT NULL,
    model TEXT NOT NULL,
    request TEXT NOT NULL,
    response TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    model TEXT NOT NULL,
    inputs INTEGER NOT NULL,
    outcome TEXT NOT NULL,
    latency_ms REAL,
    billed TEXT,
    note TEXT
);
"""

MESSAGE_TYPES = {
    "system": SystemChatMessageV2,
    "user": UserChatMessageV2,
    "assistant": AssistantChatMessageV2,
}


class BudgetError(RuntimeError):
    """A call would go over the monthly budget or this run's cap on new calls."""


@dataclass
class EmbedResult:
    vectors: np.ndarray  # (n, dimension) float32, in input order
    latency_ms: list[float]  # per text: latency of the call that embedded it
    cached: list[bool]  # per text: True if it came from the cache this time


@dataclass
class RerankResult:
    order: list[int]  # document indexes, most relevant first
    scores: list[float]  # relevance score for each index in `order`
    latency_ms: float
    cached: bool


@dataclass
class ChatResult:
    text: str
    citations: list[dict[str, Any]]  # {"start", "end", "text", "sources": [document ids]}
    finish_reason: str | None
    latency_ms: float
    cached: bool


def utc_now() -> datetime:
    return datetime.now(UTC)


def stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S")


def month_bounds(moment: datetime) -> tuple[str, str]:
    """Start and end stamps of the calendar month (UTC) that contains `moment`."""
    start = moment.astimezone(UTC).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if start.month == 12:
        end = start.replace(year=start.year + 1, month=1)
    else:
        end = start.replace(month=start.month + 1)
    return stamp(start), stamp(end)


def cache_key(endpoint: str, model: str, params: dict[str, Any], payload: Any) -> str:
    """Hash of everything that determines a response."""
    params = {k: v for k, v in params.items() if v is not None}
    blob = json.dumps(
        {
            "v": KEY_VERSION,
            "endpoint": endpoint,
            "model": model,
            "params": params,
            "input": payload,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def billed(resp: Any) -> dict[str, float]:
    """Billed units from a response: `meta` for embed and rerank, `usage` for chat."""
    units = getattr(getattr(resp, "meta", None), "billed_units", None)
    if units is None:
        units = getattr(getattr(resp, "usage", None), "billed_units", None)
    if units is None or not hasattr(units, "model_dump"):
        return {}
    return {k: v for k, v in units.model_dump().items() if v}


def dump(resp: Any) -> Any:
    """The full response as JSON, kept for later digging. None if it can't be dumped."""
    try:
        return resp.model_dump(mode="json", exclude_none=True)
    except Exception:
        return None


def retryable(err: Exception) -> bool:
    status = getattr(err, "status_code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    return isinstance(err, httpx.TransportError)  # timeouts and dropped connections


def backoff(err: Exception, attempt: int) -> float:
    """Seconds to wait before the next attempt: Retry-After if the server sent it, else doubling."""
    headers = getattr(err, "headers", None) or {}
    retry_after = headers.get("retry-after") or headers.get("Retry-After")
    try:
        return min(float(retry_after), MAX_BACKOFF_S)
    except (TypeError, ValueError):
        return min(2.0**attempt, MAX_BACKOFF_S)


class RateLimiter:
    """Sliding window: at most `limit` units of work in any `window` seconds."""

    def __init__(
        self,
        limit: int,
        window: float,
        clock: Callable[[], float],
        sleep: Callable[[float], None],
    ):
        self.limit, self.window, self.clock, self.sleep = limit, window, clock, sleep
        self.events: deque[tuple[float, int]] = deque()

    def acquire(self, weight: int = 1) -> None:
        weight = min(weight, self.limit)
        while True:
            now = self.clock()
            while self.events and self.events[0][0] <= now - self.window:
                self.events.popleft()
            if sum(w for _, w in self.events) + weight <= self.limit:
                self.events.append((now, weight))
                return
            self.sleep(self.events[0][0] + self.window - now)


class Store:
    """The response cache and the call ledger, in one SQLite file."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)

    def get_embeddings(self, keys: list[str]) -> dict[str, tuple[np.ndarray, float]]:
        found = {}
        for chunk in itertools.batched(dict.fromkeys(keys), 500):
            marks = ",".join("?" * len(chunk))
            rows = self.db.execute(
                f"SELECT key, vector, latency_ms FROM embeddings WHERE key IN ({marks})", chunk
            )
            for key, blob, latency_ms in rows:
                found[key] = (np.frombuffer(blob, dtype=np.float32), latency_ms)
        return found

    def put_embeddings(self, rows: list[tuple]) -> None:
        with self.db:
            self.db.executemany(
                "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows
            )

    def get_response(self, key: str) -> tuple[dict[str, Any], float] | None:
        row = self.db.execute(
            "SELECT response, latency_ms FROM responses WHERE key = ?", (key,)
        ).fetchone()
        return (json.loads(row[0]), row[1]) if row else None

    def put_response(self, key, endpoint, model, request, response, latency_ms, at) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    key,
                    endpoint,
                    model,
                    json.dumps(request, ensure_ascii=False),
                    json.dumps(response, ensure_ascii=False),
                    latency_ms,
                    at,
                ),
            )

    def log_call(
        self, at, endpoint, model, inputs, outcome, latency_ms=None, billed=None, note=None
    ):
        with self.db:
            self.db.execute(
                "INSERT INTO calls (at, endpoint, model, inputs, outcome, latency_ms, billed, note)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (at, endpoint, model, inputs, outcome, latency_ms, json.dumps(billed or {}), note),
            )

    def month_calls(self, moment: datetime) -> int:
        start, end = month_bounds(moment)
        query = "SELECT COUNT(*) FROM calls WHERE at >= ? AND at < ?"
        return self.db.execute(query, (start, end)).fetchone()[0]

    def month_breakdown(self, moment: datetime) -> list[tuple[str, str, str, int]]:
        start, end = month_bounds(moment)
        return self.db.execute(
            "SELECT endpoint, model, outcome, COUNT(*) FROM calls WHERE at >= ? AND at < ?"
            " GROUP BY endpoint, model, outcome ORDER BY COUNT(*) DESC",
            (start, end),
        ).fetchall()

    def cache_sizes(self) -> tuple[int, int]:
        embeddings = self.db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        responses = self.db.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
        return embeddings, responses


def to_sdk_message(message: dict[str, str]) -> Any:
    return MESSAGE_TYPES[message["role"]](content=message["content"])


def to_sdk_document(doc: dict[str, Any]) -> Document:
    return Document(id=doc["id"], data={k: v for k, v in doc.items() if k != "id"})


def chat_data(resp: Any) -> dict[str, Any]:
    content = resp.message.content or []
    citations = [
        {
            "start": c.start,
            "end": c.end,
            "text": c.text,
            "sources": [s.id for s in c.sources or [] if getattr(s, "id", None)],
        }
        for c in resp.message.citations or []
    ]
    return {
        "text": "".join(c.text for c in content if getattr(c, "type", None) == "text"),
        "citations": citations,
        "finish_reason": resp.finish_reason,
        "raw": dump(resp),
    }


class CachedCohere:
    """Cohere's embed, rerank and chat, behind the cache, the ledger, throttling and retries."""

    def __init__(
        self,
        sdk: Any = None,
        *,
        db_path: Path | None = None,
        read_cache: bool = True,
        monthly_budget: int | None = None,
        max_new_calls: int | None = None,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = utc_now,
    ):
        self._sdk = sdk
        self.store = Store(db_path or config.CACHE_DB)
        self.read_cache = read_cache
        self.monthly_budget = (
            config.MONTHLY_CALL_BUDGET if monthly_budget is None else monthly_budget
        )
        self.max_new_calls = max_new_calls
        self.clock, self.sleep, self.now = clock, sleep, now
        self.calls: Counter[str] = Counter()  # live attempts this run, by endpoint
        self.from_cache: Counter[str] = Counter()  # texts for embed, requests otherwise
        self._limiters: dict[str, RateLimiter] = {}

    @property
    def sdk(self) -> Any:
        """The real Cohere client, created on the first live call, so cached runs need no key."""
        if self._sdk is None:
            self._sdk = cohere.ClientV2(api_key=config.api_key(), max_retries=0)
        return self._sdk

    def embed(
        self, texts: list[str], *, model: str, input_type: str, dimension: int | None = None
    ) -> EmbedResult:
        dimension = dimension or config.EMBED_DIMENSION
        params = {"input_type": input_type, "dimension": dimension, "embedding_type": "float"}
        keys = [cache_key("embed", model, params, text) for text in texts]
        found = self.store.get_embeddings(keys) if self.read_cache else {}
        todo = {k: t for k, t in zip(keys, texts, strict=True) if k not in found}
        for batch in itertools.batched(todo.items(), EMBED_BATCH):
            batch_texts = [text for _, text in batch]
            resp, latency_ms = self._call(
                "embed",
                model,
                len(batch_texts),
                partial(
                    self.sdk.embed,
                    model=model,
                    input_type=input_type,
                    texts=batch_texts,
                    embedding_types=["float"],
                    output_dimension=dimension,
                ),
            )
            vectors = np.asarray(resp.embeddings.float_, dtype=np.float32)
            if vectors.shape != (len(batch), dimension):
                raise ValueError(
                    f"expected {(len(batch), dimension)} embeddings, got {vectors.shape}"
                )
            at = stamp(self.now())
            pairs = list(zip((key for key, _ in batch), vectors, strict=True))
            self.store.put_embeddings(
                [
                    (key, model, input_type, dimension, v.tobytes(), latency_ms, len(batch), at)
                    for key, v in pairs
                ]
            )
            found.update({key: (v, latency_ms) for key, v in pairs})
        cached = [key not in todo for key in keys]
        self.from_cache["embed"] += sum(cached)
        if keys:
            vectors = np.stack([found[key][0] for key in keys])
        else:
            vectors = np.empty((0, dimension), dtype=np.float32)
        return EmbedResult(vectors, [found[key][1] for key in keys], cached)

    def rerank(
        self, query: str, documents: list[str], *, model: str, top_n: int | None = None
    ) -> RerankResult:
        request = {"query": query, "documents": documents}
        key = cache_key("rerank", model, {"top_n": top_n}, request)
        hit = self.store.get_response(key) if self.read_cache else None
        if hit:
            data, latency_ms = hit
            self.from_cache["rerank"] += 1
        else:
            extra = {} if top_n is None else {"top_n": top_n}
            resp, latency_ms = self._call(
                "rerank",
                model,
                len(documents),
                partial(self.sdk.rerank, model=model, query=query, documents=documents, **extra),
            )
            data = {
                "results": [{"index": r.index, "score": r.relevance_score} for r in resp.results],
                "raw": dump(resp),
            }
            self.store.put_response(
                key, "rerank", model, {**request, **extra}, data, latency_ms, stamp(self.now())
            )
        results = data["results"]
        return RerankResult(
            [r["index"] for r in results],
            [r["score"] for r in results],
            latency_ms,
            hit is not None,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        documents: list[dict[str, Any]] | None = None,
        **params: Any,
    ) -> ChatResult:
        """Messages are {"role", "content"} dicts; documents are {"id", "text", ...} dicts."""
        request = {"messages": messages, "documents": documents or []}
        key = cache_key("chat", model, params, request)
        hit = self.store.get_response(key) if self.read_cache else None
        if hit:
            data, latency_ms = hit
            self.from_cache["chat"] += 1
        else:
            kwargs = dict(params)
            if documents:
                kwargs["documents"] = [to_sdk_document(d) for d in documents]
            resp, latency_ms = self._call(
                "chat",
                model,
                1,
                partial(
                    self.sdk.chat,
                    model=model,
                    messages=[to_sdk_message(m) for m in messages],
                    **kwargs,
                ),
            )
            data = chat_data(resp)
            self.store.put_response(
                key, "chat", model, {**request, **params}, data, latency_ms, stamp(self.now())
            )
        return ChatResult(
            data["text"], data["citations"], data["finish_reason"], latency_ms, hit is not None
        )

    def summary(self) -> str:
        made = sum(self.calls.values())
        detail = ", ".join(f"{endpoint} {n}" for endpoint, n in sorted(self.calls.items()))
        hits = [
            f"{n:,} {noun if n == 1 else noun + 's'}"
            for endpoint, noun in (
                ("embed", "embedded text"),
                ("rerank", "rerank"),
                ("chat", "chat answer"),
            )
            if (n := self.from_cache[endpoint])
        ]
        used = self.store.month_calls(self.now())
        return "\n".join(
            [
                f"Cohere API calls this run: {made}" + (f" ({detail})" if detail else ""),
                f"From the cache: {', '.join(hits) or 'nothing'}",
                f"Calls this month: {used:,} of {self.monthly_budget:,}"
                " (call ledger, calendar month in UTC)",
            ]
        )

    def _call(
        self, endpoint: str, model: str, inputs: int, request: Callable[[], Any]
    ) -> tuple[Any, float]:
        """One live call: budget check, throttle, retries with backoff, a ledger row per attempt."""
        limiter, weight = self._limiter(endpoint, model, inputs)
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self._check_budget()
            limiter.acquire(weight)
            start = self.clock()
            try:
                resp = request()
            except Exception as err:
                latency_ms = (self.clock() - start) * 1000
                status = getattr(err, "status_code", None)
                outcome = "rate_limited" if status == 429 else "error"
                self._log(endpoint, model, inputs, outcome, latency_ms)
                if not retryable(err) or attempt == MAX_ATTEMPTS:
                    raise
                self.sleep(backoff(err, attempt))
                continue
            latency_ms = (self.clock() - start) * 1000
            self._log(endpoint, model, inputs, "ok", latency_ms, billed(resp))
            return resp, latency_ms
        raise AssertionError("unreachable")

    def _log(self, endpoint, model, inputs, outcome, latency_ms, billed_units=None) -> None:
        self.calls[endpoint] += 1
        self.store.log_call(
            stamp(self.now()), endpoint, model, inputs, outcome, round(latency_ms, 1), billed_units
        )

    def _check_budget(self) -> None:
        made = sum(self.calls.values())
        if self.max_new_calls is not None and made >= self.max_new_calls:
            raise BudgetError(f"This run is capped at {self.max_new_calls} new API calls.")
        used = self.store.month_calls(self.now())
        if used >= self.monthly_budget:
            raise BudgetError(
                f"The call ledger shows {used} calls this month, and the budget is "
                f"{self.monthly_budget}. See `python -m labelrag.usage`."
            )

    def _limiter(self, endpoint: str, model: str, inputs: int) -> tuple[RateLimiter, int]:
        if endpoint == "embed":
            name, limit, weight = "embed", config.EMBED_INPUTS_PER_MINUTE, inputs
        elif endpoint == "rerank":
            name, limit, weight = "rerank", config.RERANK_CALLS_PER_MINUTE, 1
        else:
            name, limit, weight = f"chat {model}", config.CHAT_CALLS_PER_MINUTE, 1
        if name not in self._limiters:
            self._limiters[name] = RateLimiter(limit, 60.0, self.clock, self.sleep)
        return self._limiters[name], weight
