from datetime import UTC, datetime
from types import SimpleNamespace as NS

import numpy as np
import pytest

from labelrag import config, usage
from labelrag.client import BudgetError, CachedCohere

NOW = datetime(2026, 11, 8, 12, 0, tzinfo=UTC)
QUESTION = [{"role": "user", "content": "Q?"}]


class FakeClock:
    """A clock that only moves when something sleeps or a fake call takes time."""

    def __init__(self):
        self.t = 1000.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.t += seconds


class ApiFailure(Exception):
    """Shaped like cohere's ApiError: a status code and response headers."""

    def __init__(self, status_code, headers=None):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        self.headers = headers or {}


class FakeSDK:
    """Stands in for cohere.ClientV2. Each call takes 250 ms on the fake clock."""

    def __init__(self, clock, failures=()):
        self.clock = clock
        self.failures = list(failures)
        self.requests: list[tuple[str, dict]] = []
        self.started: list[float] = []

    def _begin(self, endpoint, **kwargs):
        self.requests.append((endpoint, kwargs))
        self.started.append(self.clock.t)
        self.clock.t += 0.25
        if self.failures:
            raise self.failures.pop(0)

    def embed(self, *, model, input_type, texts, embedding_types, output_dimension):
        self._begin("embed", model=model, input_type=input_type, texts=list(texts))
        padding = [0.0] * (output_dimension - 2)
        rows = [[float(len(t)), float(sum(map(ord, t)))] + padding for t in texts]
        return NS(embeddings=NS(float_=rows), meta=None)

    def rerank(self, *, model, query, documents, top_n=None):
        self._begin("rerank", model=model, query=query, documents=list(documents), top_n=top_n)
        order = sorted(range(len(documents)), key=lambda i: -len(documents[i]))[:top_n]
        results = [NS(index=i, relevance_score=1 / (rank + 1)) for rank, i in enumerate(order)]
        return NS(results=results, meta=None)

    def chat(self, *, model, messages, documents=None, **params):
        self._begin("chat", model=model, messages=messages, documents=documents, **params)
        sources = [NS(id=documents[0].id)] if documents else []
        citation = NS(start=4, end=10, text="answer", sources=sources)
        message = NS(content=[NS(type="text", text="The answer.")], citations=[citation])
        return NS(message=message, finish_reason="COMPLETE", usage=None)


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def make(tmp_path, clock):
    """Build a client on a temp database. Clients built in one test share that database."""

    def build(sdk=None, **kwargs):
        sdk = sdk if sdk is not None else FakeSDK(clock)
        client = CachedCohere(
            sdk,
            db_path=tmp_path / "cache.sqlite",
            clock=clock,
            sleep=clock.sleep,
            now=lambda: NOW,
            **kwargs,
        )
        return client, sdk

    return build


def embed(client, texts, **kwargs):
    kwargs = {"model": "embed-pro", "input_type": "search_document", "dimension": 8} | kwargs
    return client.embed(texts, **kwargs)


def test_repeated_calls_make_no_new_requests(make):
    client, sdk = make()
    first = embed(client, ["a", "bb", "ccc"])
    again = embed(client, ["a", "bb", "ccc"])
    for _ in range(2):
        client.rerank("q", ["x", "yy"], model="rerank-pro")
        client.chat(QUESTION, model="chat")
    assert [endpoint for endpoint, _ in sdk.requests] == ["embed", "rerank", "chat"]
    assert np.array_equal(first.vectors, again.vectors)
    assert first.cached == [False, False, False]
    assert again.cached == [True, True, True]


def test_the_cache_survives_a_new_client(make):
    client, _ = make()
    embed(client, ["a"])
    client.chat(QUESTION, model="chat")
    later, later_sdk = make()  # as if in a new process
    assert embed(later, ["a"]).cached == [True]
    assert later.chat(QUESTION, model="chat").cached
    assert later_sdk.requests == []


def test_cached_runs_need_no_api_key(make, tmp_path, monkeypatch):
    client, _ = make()
    embed(client, ["a"])
    monkeypatch.delenv("CO_API_KEY", raising=False)
    monkeypatch.setattr(config, "ENV_FILE", tmp_path / "absent.env")
    keyless = CachedCohere(db_path=tmp_path / "cache.sqlite")  # no fake: a live call would fail
    assert embed(keyless, ["a"]).cached == [True]


def test_only_uncached_texts_are_sent_in_batches_of_96(make):
    client, sdk = make()
    embed(client, ["t0", "t1"])
    result = embed(client, [f"t{i}" for i in range(200)])
    assert [len(kwargs["texts"]) for _, kwargs in sdk.requests] == [2, 96, 96, 6]
    assert result.vectors.shape == (200, 8)
    assert result.cached[:2] == [True, True]
    assert not any(result.cached[2:])


def test_cache_key_covers_model_parameters_and_input(make):
    client, sdk = make()
    embed(client, ["a"])
    embed(client, ["a"], model="embed-fast")
    embed(client, ["a"], input_type="search_query")
    embed(client, ["a"], dimension=16)
    client.rerank("q", ["x", "yy"], model="rerank-pro")
    client.rerank("q", ["x", "yy"], model="rerank-pro", top_n=1)
    client.rerank("q", ["x", "zz"], model="rerank-pro")
    assert len(sdk.requests) == 7


def test_cached_entries_keep_the_latency_measured_at_call_time(make, clock):
    client, _ = make()
    fresh = [
        embed(client, ["q"]).latency_ms[0],
        client.rerank("q", ["x"], model="rerank-pro").latency_ms,
        client.chat(QUESTION, model="chat").latency_ms,
    ]
    clock.t += 3600
    later, _ = make()
    cached = [
        embed(later, ["q"]).latency_ms[0],
        later.rerank("q", ["x"], model="rerank-pro").latency_ms,
        later.chat(QUESTION, model="chat").latency_ms,
    ]
    assert fresh == pytest.approx([250, 250, 250])
    assert cached == pytest.approx([250, 250, 250])


def test_chat_returns_text_and_cited_document_ids(make):
    client, sdk = make()
    docs = [{"id": "doc-1", "text": "Some label text.", "title": "Dosage"}]
    result = client.chat(QUESTION, model="chat", documents=docs)
    assert result.text == "The answer."
    assert result.citations[0]["sources"] == ["doc-1"]
    sent = sdk.requests[0][1]["documents"][0]
    assert (sent.id, sent.data) == ("doc-1", {"text": "Some label text.", "title": "Dosage"})


def test_429s_back_off_and_every_attempt_is_logged(make, clock):
    client, _ = make(FakeSDK(clock, failures=[ApiFailure(429), ApiFailure(429)]))
    assert client.rerank("q", ["x"], model="rerank-pro").order == [0]
    assert clock.slept == [2.0, 4.0]
    assert client.calls["rerank"] == 3
    rows = client.store.db.execute("SELECT outcome FROM calls ORDER BY id")
    assert [outcome for (outcome,) in rows] == ["rate_limited", "rate_limited", "ok"]


def test_retry_after_header_sets_the_wait(make, clock):
    client, _ = make(FakeSDK(clock, failures=[ApiFailure(429, {"retry-after": "7"})]))
    client.rerank("q", ["x"], model="rerank-pro")
    assert clock.slept == [7.0]


def test_client_errors_are_not_retried(make, clock):
    client, sdk = make(FakeSDK(clock, failures=[ApiFailure(400)]))
    with pytest.raises(ApiFailure):
        client.rerank("q", ["x"], model="rerank-pro")
    assert len(sdk.requests) == 1
    assert clock.slept == []


def test_server_errors_give_up_after_four_attempts(make, clock):
    client, sdk = make(FakeSDK(clock, failures=[ApiFailure(503)] * 4))
    with pytest.raises(ApiFailure):
        client.rerank("q", ["x"], model="rerank-pro")
    assert len(sdk.requests) == 4
    assert clock.slept == [2.0, 4.0, 8.0]


def test_rerank_stays_under_10_calls_a_minute(make):
    client, sdk = make()
    for i in range(25):
        client.rerank(f"q{i}", ["x"], model="rerank-pro")
    starts = sdk.started
    assert len(starts) == 25
    assert all(b - a >= 60 for a, b in zip(starts, starts[10:], strict=False))


def test_embed_inputs_per_minute_are_limited(make, monkeypatch):
    monkeypatch.setattr(config, "EMBED_INPUTS_PER_MINUTE", 100)
    client, sdk = make()
    embed(client, [f"t{i}" for i in range(192)])
    assert len(sdk.requests) == 2
    assert sdk.started[1] - sdk.started[0] >= 60


def test_monthly_budget_stops_calls_but_not_cache_hits(make):
    client, sdk = make(monthly_budget=2)
    embed(client, ["a"])
    embed(client, ["b"])
    with pytest.raises(BudgetError):
        embed(client, ["c"])
    assert len(sdk.requests) == 2
    assert embed(client, ["a"]).cached == [True]


def test_a_run_can_cap_its_new_calls(make):
    client, sdk = make(max_new_calls=1)
    embed(client, ["a"])
    with pytest.raises(BudgetError):
        embed(client, ["b"])
    assert len(sdk.requests) == 1


def test_month_total_counts_only_this_calendar_month(make):
    client, _ = make()
    for at in ("2026-10-31T23:59:59", "2026-11-01T00:00:00", "2026-12-01T00:00:00"):
        client.store.log_call(at, "other", "-", 0, "ok")
    assert client.store.month_calls(NOW) == 1


def test_skipping_the_cache_still_counts_every_call(make):
    client, sdk = make(read_cache=False)
    embed(client, ["a"])
    embed(client, ["a"])
    assert len(sdk.requests) == 2
    assert client.store.month_calls(NOW) == 2


def test_summary_reports_the_run_and_the_month(make):
    client, _ = make()
    embed(client, ["a", "b"])
    embed(client, ["a", "b"])
    client.rerank("q", ["x"], model="rerank-pro")
    client.rerank("q", ["x"], model="rerank-pro")
    text = client.summary()
    assert "Cohere API calls this run: 2 (embed 1, rerank 1)" in text
    assert "From the cache: 2 embedded texts, 1 rerank" in text
    assert "Calls this month: 2 of 1,000" in text


def test_usage_command_records_and_reports_calls(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "CACHE_DB", tmp_path / "cache.sqlite")
    assert usage.main(["--add", "7", "--note", "before the ledger"]) == 0
    assert "7 of 1,000 calls" in capsys.readouterr().out
