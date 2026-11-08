"""Check the trial key against every model the pipeline uses, one call each.

Costs 6 API calls: embed three short documents with Embed Pro, embed one question with
Embed Pro and with Embed Fast, rerank with Rerank Pro and Rerank Fast, and ask Command
with the documents attached. Run it with `python -m labelrag.smoke`.

The documents are short paraphrases of public drug-label text, written for this check.
"""

import sys
import time
from functools import partial

import cohere
import numpy as np
from cohere.types import Document, UserChatMessageV2

from labelrag import config

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

calls_made = 0


def billed(resp) -> str:
    units = getattr(getattr(resp, "meta", None), "billed_units", None)
    if units is None:
        units = getattr(getattr(resp, "usage", None), "billed_units", None)
    if units is None:
        return ""
    data = units.model_dump() if hasattr(units, "model_dump") else units.dict()
    return ", ".join(f"{k}={v:g}" for k, v in data.items() if v)


def call(label, fn):
    """Make one API call, print its latency and billed units, and return the response."""
    global calls_made
    calls_made += 1
    start = time.perf_counter()
    try:
        resp = fn()
    except Exception as err:  # keep going, so one run shows every problem
        print(f"  {label:<40} FAILED: {err}")
        return None
    ms = (time.perf_counter() - start) * 1000
    print(f"  {label:<40} {ms:>6.0f} ms  {billed(resp)}")
    return resp


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def main() -> int:
    sys.stdout.reconfigure(errors="replace")
    # Retries would spend extra calls from the monthly budget, so a failure stays a failure.
    co = cohere.ClientV2(api_key=config.api_key(), max_retries=0)
    ids, texts = list(DOCS), list(DOCS.values())
    checks: dict[str, bool] = {}

    print("Embed")
    vectors = {}
    for name, model, input_type, inputs in [
        ("documents", config.EMBED_DOC_MODEL, "search_document", texts),
        ("question, Pro", config.EMBED_DOC_MODEL, "search_query", [QUESTION]),
        ("question, Fast", config.EMBED_QUERY_MODEL, "search_query", [QUESTION]),
    ]:
        resp = call(
            f"{model}, {name}",
            partial(
                co.embed,
                model=model,
                input_type=input_type,
                texts=inputs,
                embedding_types=["float"],
                output_dimension=config.EMBED_DIMENSION,
            ),
        )
        if resp is not None:
            vectors[name] = np.asarray(resp.embeddings.float_, dtype=np.float32)
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
        resp = call(
            model,
            partial(co.rerank, model=model, query=QUESTION, documents=texts, top_n=len(texts)),
        )
        if resp is None:
            checks[f"{model} answers"] = False
            continue
        print("    " + ", ".join(f"{ids[r.index]} {r.relevance_score:.3f}" for r in resp.results))
        checks[f"{model} ranks the dosage text first"] = ids[resp.results[0].index] == EXPECTED

    print("Chat")
    resp = call(
        config.CHAT_MODEL,
        partial(
            co.chat,
            model=config.CHAT_MODEL,
            messages=[UserChatMessageV2(content=QUESTION)],
            documents=[Document(id=i, data={"text": t}) for i, t in DOCS.items()],
        ),
    )
    if resp is None:
        checks[f"{config.CHAT_MODEL} answers"] = False
    else:
        content = resp.message.content or []
        answer = "".join(c.text for c in content if getattr(c, "type", None) == "text")
        citations = resp.message.citations or []
        cited = [s.id for c in citations for s in (c.sources or [])]
        print(f"    finish reason: {resp.finish_reason}")
        print(f"    answer: {answer}")
        for c in citations:
            print(f"    cites {[s.id for s in c.sources or []]} for {c.text!r}")
        checks[f"{config.CHAT_MODEL} answers"] = bool(answer.strip())
        checks["citations point at the document IDs sent"] = bool(cited) and set(cited) <= set(ids)
        checks["citations include the dosage text"] = EXPECTED in cited

    print("\nChecks")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\nAPI calls made: {calls_made}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
