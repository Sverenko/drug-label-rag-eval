"""Answer a question from retrieved chunks, with citations, or decline.

    python -m labelrag.answer "What is the starting dose of Avapro?"
    python -m labelrag.answer "..." --order embed     # use the embedding-only top 5

Command A+ gets the system prompt from prompts/<version>.md, the question, and the top 5
chunks as documents whose IDs are chunk IDs, so every citation points at a chunk. When
the chunks don't answer the question, the prompt asks for one fixed sentence (DECLINE),
which makes declines easy to detect.

Calls run at temperature 0 with a fixed seed, so two prompt versions or two retrieval
orders are compared on their merits rather than on sampling luck. The prompt's text is
part of the cache key, so a new version never reuses an old version's answers.
"""

import argparse
import re
import sys
from dataclasses import dataclass, field
from typing import Any

from labelrag import config
from labelrag.client import CachedCohere
from labelrag.retrieve import TOP, Retriever, describe, print_retrieval

# The prompt in use. answer_v1 was the baseline; v2 and v3 came from its graded failures, and
# each run in results/ records the version it used.
PROMPT_VERSION = "answer_v3"
DECLINE = "The provided label excerpts don't answer this question."
TEMPERATURE = 0.0
SEED = 7


@dataclass
class Answer:
    text: str
    citations: list[dict[str, Any]]  # {"start", "end", "text", "sources": [chunk IDs]}
    declined: bool
    document_ids: list[str]  # the chunks sent, in order
    latency_ms: float
    cached: bool
    prompt_version: str = PROMPT_VERSION
    cited_ids: list[str] = field(default_factory=list)  # every chunk cited, first use first
    error: str | None = None  # Cohere's message when no answer could be generated


def load_prompt(version: str = PROMPT_VERSION) -> str:
    return (config.PROMPTS_DIR / f"{version}.md").read_text(encoding="utf-8").strip()


def plain(text: str) -> str:
    """Lowercase, straight apostrophes and single spaces, for comparing sentences."""
    return re.sub(r"\s+", " ", text.replace("’", "'")).strip().lower()


def is_decline(text: str) -> bool:
    return plain(DECLINE).rstrip(".") in plain(text)


def answer(
    client: CachedCohere,
    question: str,
    chunks: list[dict[str, Any]],
    model: str | None = None,
    prompt_version: str = PROMPT_VERSION,
) -> Answer:
    """Ask Command to answer `question` from `chunks`, citing them by chunk ID."""
    result = client.chat(
        [
            {"role": "system", "content": load_prompt(prompt_version)},
            {"role": "user", "content": question},
        ],
        model=model or config.CHAT_MODEL,
        documents=[{"id": c["chunk_id"], "title": c["header"], "text": c["text"]} for c in chunks],
        temperature=TEMPERATURE,
        seed=SEED,
    )
    cited = list(dict.fromkeys(s for c in result.citations for s in c["sources"]))
    return Answer(
        text=result.text,
        citations=result.citations,
        declined=is_decline(result.text),
        document_ids=[c["chunk_id"] for c in chunks],
        latency_ms=result.latency_ms,
        cached=result.cached,
        prompt_version=prompt_version,
        cited_ids=cited,
        error=result.error,
    )


def main(argv: list[str] | None = None, client: CachedCohere | None = None) -> int:
    parser = argparse.ArgumentParser(description="Answer a question from the label corpus.")
    parser.add_argument("question")
    parser.add_argument(
        "--order",
        choices=["rerank", "embed"],
        default="rerank",
        help="answer from the reranked top 5 (default) or the embedding-only top 5",
    )
    parser.add_argument("-k", type=int, default=TOP, help="how many chunks to send")
    args = parser.parse_args(argv)
    client = client or CachedCohere(max_new_calls=10)
    retriever = Retriever.load(client)
    retrieval = retriever.retrieve(args.question)
    print_retrieval(retrieval, retriever.chunks, args.k)
    chunks = [retriever.chunks[cid] for cid in retrieval.top(args.order, args.k)]
    result = answer(client, args.question, chunks)
    print(
        f"\nAnswer from the {args.order} top {args.k}" + (" (declined)" if result.declined else "")
    )
    print(f"  {result.text}")
    for citation in result.citations:
        sources = "; ".join(describe(retriever.chunks[s]) for s in citation["sources"])
        print(f"  cites {citation['text']!r} -> {sources}")
    cached = " (cached)" if result.cached else ""
    print(f"Generate: {result.latency_ms:.0f} ms{cached}, prompt {result.prompt_version}")
    print()
    print(client.summary())
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
