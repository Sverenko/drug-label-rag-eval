"""Check the trial key against every model the pipeline uses, one call each.

Costs 6 API calls: embed three short documents with Embed Pro, embed one question with
Embed Pro and with Embed Fast, rerank with Rerank Pro and Rerank Fast, and ask Command
with the documents attached. It skips the cache on purpose, so every check is a live call,
and each call is recorded in the call ledger. Run it with `python -m labelrag.smoke`.

The documents are short paraphrases of public drug-label text, written for this check.
"""

import sys
from functools import partial

import numpy as np

from labelrag import config
from labelrag.client import CachedCohere

DOCS = {
    "metformin-dosage": (
        "Metformin hydrochloride tablets: start with 500 mg twice a day or 850 mg once a day, "
        "given with meals. The maximum recommended daily dose is 2,550 mg in adults."
    ),
    "metformin-contraindications": (
        "Metformin is contraindicated in severe renal impairment (eGFR below 30 mL/min/1.73 m2), "
        "in hypersensitivity to metformin, and in acute or chronic metabolic acidosis."
    ),
    "atorvastatin-dosage": (
        "Atorvastatin calcium tablets: the dosage range is 10 mg to 80 mg once a day, "
        "taken at any time of day, with or without food."
    ),
}
QUESTION = "What is the maximum daily dose of metformin?"
EXPECTED = "metformin-dosage"


def attempt(label, fn):
    """Run one check's call and print its latency. Returns None if the call failed."""
    try:
        result = fn()
    except Exception as err:  # keep going, so one run shows every problem
        print(f"  {label:<40} FAILED: {err}")
        return None
    latency = result.latency_ms[0] if isinstance(result.latency_ms, list) else result.latency_ms
    print(f"  {label:<40} {latency:>6.0f} ms")
    return result


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def main() -> int:
    sys.stdout.reconfigure(errors="replace")
    client = CachedCohere(read_cache=False)
    ids, texts = list(DOCS), list(DOCS.values())
    checks: dict[str, bool] = {}

    print("Embed")
    vectors = {}
    for name, model, input_type, inputs in [
        ("documents", config.EMBED_DOC_MODEL, "search_document", texts),
        ("question, Pro", config.EMBED_DOC_MODEL, "search_query", [QUESTION]),
        ("question, Fast", config.EMBED_QUERY_MODEL, "search_query", [QUESTION]),
    ]:
        result = attempt(
            f"{model}, {name}", partial(client.embed, inputs, model=model, input_type=input_type)
        )
        if result is not None:
            vectors[name] = result.vectors
    dims = sorted({v.shape[1] for v in vectors.values()})
    print(f"    dimensions returned: {dims}")
    dims_ok = len(vectors) == 3 and dims == [config.EMBED_DIMENSION]
    checks[f"all three embed calls return {config.EMBED_DIMENSION} dimensions"] = dims_ok
    if len(vectors) == 3:
        docs = unit_rows(vectors["documents"])
        for name in ("question, Pro", "question, Fast"):
            sims = docs @ unit_rows(vectors[name])[0]
            order = np.argsort(-sims)
            ranked = ", ".join(f"{ids[i]} {sims[i]:.3f}" for i in order)
            print(f"    {name} vs Pro documents: {ranked}")
            checks[f"{name} vector finds the dosage text first"] = ids[order[0]] == EXPECTED
        pro_q = unit_rows(vectors["question, Pro"])[0]
        fast_q = unit_rows(vectors["question, Fast"])[0]
        print(f"    same question, Pro vs Fast vector: cosine {float(pro_q @ fast_q):.3f}")

    print("Rerank")
    for model in (config.RERANK_MODEL, config.RERANK_MODEL_FAST):
        result = attempt(
            model, partial(client.rerank, QUESTION, texts, model=model, top_n=len(texts))
        )
        if result is None:
            checks[f"{model} answers"] = False
            continue
        pairs = zip(result.order, result.scores, strict=True)
        print("    " + ", ".join(f"{ids[i]} {score:.3f}" for i, score in pairs))
        checks[f"{model} ranks the dosage text first"] = ids[result.order[0]] == EXPECTED

    print("Chat")
    result = attempt(
        config.CHAT_MODEL,
        partial(
            client.chat,
            [{"role": "user", "content": QUESTION}],
            model=config.CHAT_MODEL,
            documents=[{"id": i, "text": t} for i, t in DOCS.items()],
        ),
    )
    if result is None:
        checks[f"{config.CHAT_MODEL} answers"] = False
    else:
        cited = [source for c in result.citations for source in c["sources"]]
        print(f"    finish reason: {result.finish_reason}")
        print(f"    answer: {result.text}")
        for c in result.citations:
            print(f"    cites {c['sources']} for {c['text']!r}")
        checks[f"{config.CHAT_MODEL} answers"] = bool(result.text.strip())
        checks["citations point at the document IDs sent"] = bool(cited) and set(cited) <= set(ids)
        checks["citations include the dosage text"] = EXPECTED in cited

    print("\nChecks")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print()
    print(client.summary())
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
