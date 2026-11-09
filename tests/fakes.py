"""Test doubles shared by the test modules: a fake clock and a fake Cohere client."""

import re
import zlib
from types import SimpleNamespace as NS

from labelrag import answer


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

    def __init__(self, status_code, headers=None, body=None):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        self.headers = headers or {}
        self.body = body


NO_ANSWER = {"error_type": "NO_VALID_RESPONSE_GENERATED", "message": "No valid response generated."}


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def word_vector(text: str, dimension: int) -> list[float]:
    """A bag-of-words vector, so texts that share words point the same way."""
    vector = [0.0] * dimension
    for word in words(text):
        vector[zlib.crc32(word.encode()) % dimension] += 1.0
    if not any(vector):
        vector[0] = 1.0
    return vector


def overlap(query: str, document: str) -> float:
    """Share of the query's words that appear in the document."""
    asked = set(words(query))
    return len(asked & set(words(document))) / (len(asked) or 1)


class FakeSDK:
    """Stands in for cohere.ClientV2. Each call takes 250 ms on the fake clock. Embeddings
    are bag-of-words vectors and rerank scores are word overlap, so retrieval behaves
    sensibly in tests. Set `answer` and `cite` to shape the chat reply."""

    def __init__(self, clock, failures=()):
        self.clock = clock
        self.failures = list(failures)
        self.requests: list[tuple[str, dict]] = []
        self.started: list[float] = []
        self.answer = "The answer."
        self.cite = True

    def _begin(self, endpoint, **kwargs):
        self.requests.append((endpoint, kwargs))
        self.started.append(self.clock.t)
        self.clock.t += 0.25
        if self.failures:
            raise self.failures.pop(0)

    def embed(self, *, model, input_type, texts, embedding_types, output_dimension):
        self._begin("embed", model=model, input_type=input_type, texts=list(texts))
        rows = [word_vector(t, output_dimension) for t in texts]
        return NS(embeddings=NS(float_=rows), meta=None)

    def rerank(self, *, model, query, documents, top_n=None):
        self._begin("rerank", model=model, query=query, documents=list(documents), top_n=top_n)
        scores = [overlap(query, d) for d in documents]
        order = sorted(range(len(documents)), key=lambda i: (-scores[i], i))[:top_n]
        return NS(results=[NS(index=i, relevance_score=scores[i]) for i in order], meta=None)

    def chat(self, *, model, messages, documents=None, **params):
        self._begin("chat", model=model, messages=messages, documents=documents, **params)
        text = self.answer(messages, documents) if callable(self.answer) else self.answer
        citations = []
        if documents and self.cite:
            source = NS(id=documents[0].id)
            citations = [NS(start=0, end=len(text), text=text, sources=[source])]
        message = NS(content=[NS(type="text", text=text)], citations=citations)
        return NS(message=message, finish_reason="COMPLETE", usage=None)


def reply(messages, documents):
    """Fake Command for the eval tests: declines price questions, otherwise answers with
    every chunk's text."""
    if "cost" in messages[-1].content:
        return answer.DECLINE
    return " ".join(d.data["text"] for d in documents)
