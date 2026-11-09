"""Retrieve chunks for a question: the 50 nearest by embedding, then reranked.

    python -m labelrag.retrieve "What is the starting dose of Avapro?"

Retriever.retrieve() embeds the question with Embed Fast (input_type search_query), takes
the 50 chunks whose Embed Pro vectors are closest by cosine similarity, and has Rerank
Pro put all 50 in order. It returns both orders with their scores, so the eval scores
embedding-only and reranked retrieval from one retrieval, plus the time each stage took.
Generation uses the top 5 of either order.
"""

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from labelrag import chunk, config, index
from labelrag.client import CachedCohere

POOL = 50  # candidates from vector search; rerank orders all of them
TOP = 5  # chunks that go to generation


@dataclass
class Retrieval:
    question: str
    embed_ids: list[str]  # the nearest chunks, closest first
    embed_scores: list[float]  # cosine similarity
    rerank_ids: list[str]  # the same chunks, in rerank's order
    rerank_scores: list[float]  # rerank relevance scores
    timings_ms: dict[str, float]  # embed_query, vector_search, rerank
    cached: dict[str, bool]  # embed_query, rerank: True if served from the cache

    def top(self, order: str, k: int = TOP) -> list[str]:
        """The first k chunk IDs of the "embed" or "rerank" order."""
        if order not in ("embed", "rerank"):
            raise ValueError(f"order must be 'embed' or 'rerank', not {order!r}")
        return (self.rerank_ids if order == "rerank" else self.embed_ids)[:k]


class Retriever:
    def __init__(
        self,
        client: CachedCohere,
        idx: index.Index,
        chunks: dict[str, dict[str, Any]],
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.client, self.index, self.chunks, self.clock = client, idx, chunks, clock

    @classmethod
    def load(
        cls, client: CachedCohere | None = None, index_path: Path | None = None
    ) -> "Retriever":
        """The saved index (data/index/ unless given) and chunks, checked by index.load()."""
        idx = index.load(index_path)
        chunks = {c["chunk_id"]: c for c in chunk.load_chunks()}
        return cls(client or CachedCohere(), idx, chunks)

    def retrieve(
        self,
        question: str,
        pool: int = POOL,
        query_model: str | None = None,
        rerank_model: str | None = None,
        rerank: bool = True,
        with_header: bool = True,
    ) -> Retrieval:
        """The pool nearest by embedding, then reranked unless rerank is False (the reranked
        order is then empty). Rerank reads each chunk with its header unless with_header
        is False."""
        query = self.client.embed(
            [question],
            model=query_model or config.EMBED_QUERY_MODEL,
            input_type="search_query",
            dimension=config.EMBED_DIMENSION,
        )
        vector = query.vectors[0] / np.linalg.norm(query.vectors[0])

        start = self.clock()
        scores = self.index.vectors @ vector
        k = min(pool, len(scores))
        nearest = np.argpartition(-scores, k - 1)[:k] if k < len(scores) else np.arange(k)
        nearest = nearest[np.argsort(-scores[nearest], kind="stable")]
        search_ms = (self.clock() - start) * 1000

        embed_ids = [self.index.chunk_ids[i] for i in nearest]
        embed_scores = [float(scores[i]) for i in nearest]
        if not rerank:
            return Retrieval(
                question=question,
                embed_ids=embed_ids,
                embed_scores=embed_scores,
                rerank_ids=[],
                rerank_scores=[],
                timings_ms={"embed_query": query.latency_ms[0], "vector_search": search_ms},
                cached={"embed_query": query.cached[0]},
            )
        documents = [chunk.embed_text(self.chunks[cid], with_header) for cid in embed_ids]
        reranked = self.client.rerank(
            question,
            documents,
            model=rerank_model or config.RERANK_MODEL,
            top_n=len(documents),
        )
        return Retrieval(
            question=question,
            embed_ids=embed_ids,
            embed_scores=embed_scores,
            rerank_ids=[embed_ids[i] for i in reranked.order],
            rerank_scores=reranked.scores,
            timings_ms={
                "embed_query": query.latency_ms[0],
                "vector_search": search_ms,
                "rerank": reranked.latency_ms,
            },
            cached={"embed_query": query.cached[0], "rerank": reranked.cached},
        )


def describe(c: dict[str, Any]) -> str:
    return f"{c['header']} [{c['part']}/{c['parts']}]"


def print_retrieval(result: Retrieval, chunks: dict[str, dict[str, Any]], k: int = TOP) -> None:
    embed_rank = {cid: i for i, cid in enumerate(result.embed_ids, start=1)}
    rerank_rank = {cid: i for i, cid in enumerate(result.rerank_ids, start=1)}
    pool = len(result.embed_ids)
    print(f"Embedding only (top {k} of {pool}):")
    for i, (cid, score) in enumerate(
        zip(result.embed_ids[:k], result.embed_scores, strict=False), 1
    ):
        print(f"  {i}. {score:.3f}  {describe(chunks[cid])}  (rerank #{rerank_rank[cid]})")
    print(f"Reranked (top {k} of {pool}):")
    for i, (cid, score) in enumerate(
        zip(result.rerank_ids[:k], result.rerank_scores, strict=False), 1
    ):
        print(f"  {i}. {score:.3f}  {describe(chunks[cid])}  (embedding #{embed_rank[cid]})")
    timings = ", ".join(
        f"{stage.replace('_', ' ')} {ms:.0f} ms" + (" (cached)" if result.cached.get(stage) else "")
        for stage, ms in result.timings_ms.items()
    )
    print(f"Timings: {timings}")


def main(argv: list[str] | None = None, client: CachedCohere | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Retrieve chunks for a question; show both top 5s."
    )
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=TOP, help="how many to show from each order")
    args = parser.parse_args(argv)
    client = client or CachedCohere(max_new_calls=10)
    retriever = Retriever.load(client)
    print_retrieval(retriever.retrieve(args.question), retriever.chunks, args.k)
    print()
    print(client.summary())
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
