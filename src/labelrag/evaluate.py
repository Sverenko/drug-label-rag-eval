"""Run the eval: every question through both retrieval orders and both answers.

    python -m labelrag.evaluate --run baseline           # all questions
    python -m labelrag.evaluate --run check --limit 2    # the first two, as a check

One retrieval per question gives the embedding-only and the reranked order of the same 50
chunks, and Command answers twice, once from each top 5. Results go to results/<run>/:

- rows.jsonl: one row per question and configuration (embed, rerank) with the retrieved
  chunk IDs and scores, the rank of the first gold chunk, the answer and its citations,
  the declined flag, the automatic grades, and the stage timings.
- summary.json and summary.md: the metrics as counts and percentages, the questions where
  the two configurations disagree, McNemar's test on each pair, latency percentiles, and
  results by tag.
- grades.csv: one row per distinct answer to grade by hand. Re-runs keep the hand grades,
  and once any are filled in they replace the key-fact grade in the summary, which then
  reports how often the two agree.

Automatic grades check the question's key facts, copied from the indexed text, against
the answer, the cited chunks and the five chunks sent. Citations are judged this way
reliably, because the chunk text is the text the facts came from; answers paraphrase, so
the key-fact grade for an answer is a rough signal until hand grades exist.

Every call goes through the cache, so re-running a finished run makes no API calls and
gives the same numbers. Latency comes from the original calls, except vector search,
which runs locally and is timed afresh on every run.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

from labelrag import answer as answering
from labelrag import config, questions
from labelrag.client import CachedCohere, stamp, utc_now
from labelrag.retrieve import TOP, Retriever

CONFIGS = ("embed", "rerank")
CONFIG_NAMES = {"embed": "Embed only", "rerank": "Embed + Rerank"}
GRADE_COLUMNS = [
    "id",
    "configs",
    "answerable",
    "question",
    "reference_answer",
    "key_facts",
    "answer",
    "citations",
    "auto_key_facts",
    "hand_correct",
    "comment",
]
YES, NO = {"y", "yes", "1", "true", "correct"}, {"n", "no", "0", "false", "wrong"}


def norm(text: str) -> str:
    """Text for key-fact matching: lowercase, plain punctuation, no markdown emphasis, no
    thousands separators ("2,550" matches "2550"), single spaces."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'}))
    text = re.sub(r"[‐-―]", "-", text).replace("*", "")
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)
    return re.sub(r"\s+", " ", text).strip()


def facts_found(facts: list[str], text: str) -> int:
    haystack = norm(text)
    return sum(norm(fact) in haystack for fact in facts)


def load_questions(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or config.QUESTIONS_FILE
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def gold_set(q: dict[str, Any]) -> set[tuple[str, str]]:
    return {(g["set_id"], g["section"]) for g in q.get("gold") or []}


def gold_rank(ids: list[str], chunks: dict[str, dict[str, Any]], gold: set) -> int | None:
    """1-based rank of the first chunk from a gold label section, or None."""
    for rank, cid in enumerate(ids, start=1):
        if (chunks[cid]["set_id"], chunks[cid]["section"]) in gold:
            return rank
    return None


def auto_grades(q: dict[str, Any], ans: answering.Answer, chunks: dict[str, Any]) -> dict[str, Any]:
    if not q["answerable"]:
        return {}
    facts = q["key_facts"]
    in_answer = facts_found(facts, ans.text)
    cited = " ".join(chunks[cid]["text"] for cid in ans.cited_ids)
    context = " ".join(chunks[cid]["text"] for cid in ans.document_ids)
    return {
        "key_facts": len(facts),
        "key_facts_in_answer": in_answer,
        "answer_has_key_facts": not ans.declined and in_answer == len(facts),
        "citations_contain_answer": (
            None if ans.declined else facts_found(facts, cited) == len(facts)
        ),
        "context_contains_answer": facts_found(facts, context) == len(facts),
    }


def run_questions(
    qs: list[dict[str, Any]], retriever: Retriever, client: CachedCohere, progress=print
) -> list[dict[str, Any]]:
    rows = []
    for q in qs:
        gold = gold_set(q)
        retrieval = retriever.retrieve(q["question"])
        notes = []
        for name in CONFIGS:
            ids = retrieval.rerank_ids if name == "rerank" else retrieval.embed_ids
            scores = retrieval.rerank_scores if name == "rerank" else retrieval.embed_scores
            ans = answering.answer(client, q["question"], [retriever.chunks[c] for c in ids[:TOP]])
            timings = {
                "embed_query": retrieval.timings_ms["embed_query"],
                "vector_search": retrieval.timings_ms["vector_search"],
            }
            if name == "rerank":
                timings["rerank"] = retrieval.timings_ms["rerank"]
            timings["generate"] = ans.latency_ms
            timings["end_to_end"] = sum(timings.values())
            rank = gold_rank(ids, retriever.chunks, gold) if gold else None
            rows.append(
                {
                    "id": q["id"],
                    "config": name,
                    "answerable": q["answerable"],
                    "tags": q.get("tags") or [],
                    "question": q["question"],
                    "retrieved": ids,
                    "scores": scores,
                    "gold_rank": rank,
                    "context_ids": ans.document_ids,
                    "answer": ans.text,
                    "declined": ans.declined,
                    "citations": ans.citations,
                    "cited_ids": ans.cited_ids,
                    "grades": auto_grades(q, ans, retriever.chunks),
                    "timings_ms": timings,
                    "cached": {**retrieval.cached, "generate": ans.cached},
                    "prompt_version": ans.prompt_version,
                }
            )
            where = f"gold #{rank}" if rank else ("gold missed" if gold else "unanswerable")
            notes.append(f"{name}: {where}{', declined' if ans.declined else ''}")
        progress(f"  {q['id']:<4} " + " | ".join(notes))
    return rows


def read_hand_grades(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Hand grades from a grades.csv, keyed by question ID and answer text."""
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {
            (row["id"], row["answer"]): {
                "hand_correct": row.get("hand_correct", ""),
                "comment": row.get("comment", ""),
            }
            for row in csv.DictReader(f)
        }


def hand_verdict(value: str) -> bool | None:
    value = (value or "").strip().lower()
    return True if value in YES else False if value in NO else None


def answer_correct(row: dict[str, Any], hand: dict) -> tuple[bool, str]:
    """A hand grade if there is one, otherwise the key-fact grade."""
    verdict = hand_verdict(hand.get((row["id"], row["answer"]), {}).get("hand_correct", ""))
    if verdict is not None:
        return verdict, "hand"
    return bool(row["grades"].get("answer_has_key_facts")), "key_facts"


def rate(flags: list[bool]) -> dict[str, Any]:
    n, count = len(flags), sum(bool(f) for f in flags)
    return {"count": count, "n": n, "rate": count / n if n else None}


def mcnemar(embed: list[bool], rerank: list[bool]) -> dict[str, Any]:
    """Exact two-sided McNemar test on paired outcomes."""
    only_embed = sum(a and not b for a, b in zip(embed, rerank, strict=True))
    only_rerank = sum(b and not a for a, b in zip(embed, rerank, strict=True))
    n = only_embed + only_rerank
    tail = sum(math.comb(n, i) for i in range(min(only_embed, only_rerank) + 1))
    p = 1.0 if n == 0 else min(1.0, 2 * tail / 2**n)
    return {"only_embed": only_embed, "only_rerank": only_rerank, "p_value": p}


def kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    """Cohen's kappa for two raters' yes/no grades."""
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    yes_a, yes_b = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    expected = yes_a * yes_b + (1 - yes_a) * (1 - yes_b)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "n": 0}
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "n": len(values),
    }


def summarize(
    rows: list[dict[str, Any]], qs: list[dict[str, Any]], meta: dict[str, Any], hand: dict
) -> dict[str, Any]:
    by = {(r["id"], r["config"]): r for r in rows}
    answerable = [q["id"] for q in qs if q["answerable"]]
    unanswerable = [q["id"] for q in qs if not q["answerable"]]

    def ranked_within(k: int):
        return lambda r: r["gold_rank"] is not None and r["gold_rank"] <= k

    def correct(r):
        return answer_correct(r, hand)[0]

    flags: dict[str, dict[str, list[bool]]] = {}
    metrics: dict[str, Any] = {}
    sources = {answer_correct(by[(i, c)], hand)[1] for i in answerable for c in CONFIGS}
    for c in CONFIGS:
        rs = [by[(i, c)] for i in answerable]
        flags[c] = {
            "hit@1": [ranked_within(1)(r) for r in rs],
            "hit@5": [ranked_within(5)(r) for r in rs],
            "answer_correct": [correct(r) for r in rs],
        }
        metrics[c] = {
            "hit@1": rate(flags[c]["hit@1"]),
            "hit@5": rate(flags[c]["hit@5"]),
            "hit@50": rate([r["gold_rank"] is not None for r in rs]),
            "answer_correct": rate(flags[c]["answer_correct"]),
            "answer_has_key_facts": rate([r["grades"]["answer_has_key_facts"] for r in rs]),
            "citations_contain_answer": rate(
                [r["grades"]["citations_contain_answer"] for r in rs if not r["declined"]]
            ),
            "uncited_answers": rate([not r["citations"] for r in rs if not r["declined"]]),
            "context_contains_answer": rate([r["grades"]["context_contains_answer"] for r in rs]),
            "answerable_declined": rate([r["declined"] for r in rs]),
            "unanswerable_declined": rate([by[(i, c)]["declined"] for i in unanswerable]),
        }

    def differ(metric: str) -> dict[str, list[str]]:
        e, r = flags["embed"][metric], flags["rerank"][metric]
        return {
            "only_embed": [i for i, a, b in zip(answerable, e, r, strict=True) if a and not b],
            "only_rerank": [i for i, a, b in zip(answerable, e, r, strict=True) if b and not a],
        }

    rerank_rows = [by[(i, "rerank")] for i in answerable + unanswerable]
    embed_rows = [by[(i, "embed")] for i in answerable + unanswerable]
    latency = {
        stage: percentiles([r["timings_ms"][stage] for r in rerank_rows])
        for stage in ("embed_query", "vector_search", "rerank", "generate", "end_to_end")
    }
    latency["end_to_end_embed_only"] = percentiles(
        [r["timings_ms"]["end_to_end"] for r in embed_rows]
    )

    tags = sorted({t for q in qs if q["answerable"] for t in q.get("tags") or []})
    by_tag = {}
    for tag in tags + ["(untagged)"]:
        ids = [
            q["id"]
            for q in qs
            if q["answerable"]
            and (tag in (q.get("tags") or []) or (tag == "(untagged)" and not q.get("tags")))
        ]
        by_tag[tag] = {
            c: {
                "hit@5": rate([ranked_within(5)(by[(i, c)]) for i in ids]),
                "answer_correct": rate([correct(by[(i, c)]) for i in ids]),
            }
            for c in CONFIGS
        }

    graded = [
        (answer_correct(r, hand)[0], bool(r["grades"]["answer_has_key_facts"]))
        for r in rows
        if r["answerable"] and answer_correct(r, hand)[1] == "hand"
    ]
    return {
        "run": meta,
        "questions": {
            "total": len(qs),
            "answerable": len(answerable),
            "unanswerable": len(unanswerable),
        },
        "answer_grades_from": sorted(sources),
        "metrics": metrics,
        "mcnemar": {
            m: mcnemar(flags["embed"][m], flags["rerank"][m])
            for m in ("hit@1", "hit@5", "answer_correct")
        },
        "disagreements": {m: differ(m) for m in ("hit@5", "answer_correct")},
        "latency_ms": latency,
        "by_tag": by_tag,
        "hand_vs_key_facts": {
            "graded": len(graded),
            "agree": sum(a == b for a, b in graded),
            "kappa": kappa(graded),
        },
    }


def fmt(r: dict[str, Any]) -> str:
    if not r["n"]:
        return "-"
    return f"{r['count']}/{r['n']} ({100 * r['rate']:.0f}%)"


def markdown(summary: dict[str, Any]) -> str:
    m, run, n = summary["metrics"], summary["run"], summary["questions"]
    models = run["models"]
    grade_source = " and ".join(summary["answer_grades_from"]).replace("key_facts", "key facts")

    def table_row(label: str, key: str, with_rerank: bool = True) -> str:
        rerank = fmt(m["rerank"][key]) if with_rerank else "—"
        return f"| {label} | {fmt(m['embed'][key])} | {rerank} |"

    lines = [
        f"# Eval run: {run['name']}",
        "",
        f"{run['finished']} UTC. {n['total']} questions ({n['answerable']} answerable, "
        f"{n['unanswerable']} not). Prompt `{run['prompt_version']}`; models "
        f"`{models['embed_documents']}`, `{models['embed_queries']}`, `{models['rerank']}`, "
        f"`{models['chat']}`.",
        "",
        "| | Embed only | Embed + Rerank |",
        "|---|---|---|",
        table_row("Right section in the top 5 (hit@5)", "hit@5"),
        table_row("Right section ranked first (hit@1)", "hit@1"),
        table_row("Right section anywhere in the top 50", "hit@50", with_rerank=False),
        table_row(f"Answer correct ({grade_source})", "answer_correct"),
        table_row("Citations that contain the answer", "citations_contain_answer"),
        table_row("Unanswerable questions declined", "unanswerable_declined"),
        table_row("Answerable questions wrongly declined", "answerable_declined"),
        "",
        "Citations that contain the answer counts the answers that weren't declined, and asks "
        "whether the cited chunks contain every key fact; an answer that cites nothing counts "
        f"as no. Answers that cited nothing: {fmt(m['embed']['uncited_answers'])} with "
        f"embedding only, {fmt(m['rerank']['uncited_answers'])} with rerank. The five chunks "
        f"sent held every key fact for {fmt(m['embed']['context_contains_answer'])} of "
        f"answerable questions with embedding only and "
        f"{fmt(m['rerank']['context_contains_answer'])} with rerank.",
        "",
        "McNemar's exact test, embed only vs embed + rerank, on the same questions:",
        "",
        "| Metric | Only embed right | Only rerank right | p |",
        "|---|---|---|---|",
    ]
    for metric, test in summary["mcnemar"].items():
        lines.append(
            f"| {metric} | {test['only_embed']} | {test['only_rerank']} | {test['p_value']:.3f} |"
        )
    lines += [
        "",
        "Latency per stage (ms), embed + rerank pipeline:",
        "",
        "| Stage | p50 | p95 |",
        "|---|---|---|",
    ]
    names = {
        "embed_query": "Embed query",
        "vector_search": "Vector search",
        "rerank": "Rerank",
        "generate": "Generate answer",
        "end_to_end": "End to end",
        "end_to_end_embed_only": "End to end, embed only",
    }
    for stage, label in names.items():
        p = summary["latency_ms"][stage]
        if p["n"]:
            lines.append(f"| {label} | {p['p50']:,.0f} | {p['p95']:,.0f} |")

    def ids(found: list[str]) -> str:
        return ", ".join(found) or "none"

    lines += ["", "Where the two configurations disagree:", ""]
    for metric, d in summary["disagreements"].items():
        label = metric.replace("_", " ")
        lines.append(
            f"- {label}, only embed: {ids(d['only_embed'])}; only rerank: {ids(d['only_rerank'])}"
        )
    lines += [
        "",
        "By tag (answerable questions):",
        "",
        "| Tag | n | hit@5 embed | hit@5 rerank | correct embed | correct rerank |",
        "|---|---|---|---|---|---|",
    ]
    for tag, r in summary["by_tag"].items():
        e, k = r["embed"], r["rerank"]
        if e["hit@5"]["n"]:
            counts = [e["hit@5"], k["hit@5"], e["answer_correct"], k["answer_correct"]]
            cells = " | ".join(str(c["count"]) for c in counts)
            lines.append(f"| {tag} | {e['hit@5']['n']} | {cells} |")
    agreement = summary["hand_vs_key_facts"]
    if agreement["graded"]:
        lines += [
            "",
            f"Hand grades vs key-fact grades: {agreement['agree']}/{agreement['graded']} agree, "
            f"Cohen's kappa {agreement['kappa']:.2f}.",
        ]
    return "\n".join(lines) + "\n"


def write_grades(
    rows: list[dict[str, Any]], qs: list[dict[str, Any]], path: Path, hand: dict
) -> int:
    """One row per distinct answer to each question, keeping hand grades already entered."""
    by_q = {q["id"]: q for q in qs}
    distinct: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        key = (r["id"], r["answer"])
        if key in distinct:
            distinct[key]["configs"].append(r["config"])
            continue
        q = by_q[r["id"]]
        grades = r["grades"]
        distinct[key] = {
            "id": r["id"],
            "configs": [r["config"]],
            "answerable": "yes" if r["answerable"] else "no",
            "question": q["question"],
            "reference_answer": q.get("answer", ""),
            "key_facts": " | ".join(q.get("key_facts") or []),
            "answer": r["answer"],
            "citations": " | ".join(c["text"] for c in r["citations"]),
            "auto_key_facts": (
                f"{grades['key_facts_in_answer']}/{grades['key_facts']}"
                if grades
                else "declined"
                if r["declined"]
                else "answered"
            ),
            **{"hand_correct": "", "comment": ""},
            **hand.get(key, {}),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GRADE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in distinct.values():
            writer.writerow({**row, "configs": " ".join(row["configs"])})
    return len(distinct)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None, client: CachedCohere | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run every eval question through the pipeline.")
    parser.add_argument(
        "--run", default="baseline", help="name of the run; results go to results/<run>/"
    )
    parser.add_argument("--limit", type=int, help="only the first N questions, for a quick check")
    parser.add_argument(
        "--max-calls",
        type=int,
        default=260,
        help="stop before making more than this many new API calls",
    )
    args = parser.parse_args(argv)

    report = questions.validate()
    if report.errors:
        print("The question file has errors; fix them first (python -m labelrag.questions):")
        for error in report.errors:
            print(f"  {error}")
        return 1
    qs = load_questions()[: args.limit] if args.limit else load_questions()
    client = client or CachedCohere(max_new_calls=args.max_calls)
    retriever = Retriever.load(client)
    print(f"Running {len(qs)} questions, two configurations each:")
    rows = run_questions(qs, retriever, client)

    run_dir = config.RESULTS_DIR / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "rows.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    hand = read_hand_grades(run_dir / "grades.csv")
    meta = {
        "name": args.run,
        "finished": stamp(utc_now()),
        "prompt_version": answering.PROMPT_VERSION,
        "models": {
            "embed_documents": config.EMBED_DOC_MODEL,
            "embed_queries": config.EMBED_QUERY_MODEL,
            "rerank": config.RERANK_MODEL,
            "chat": config.CHAT_MODEL,
        },
        "questions_sha256": digest(config.QUESTIONS_FILE),
        "index_built_at": retriever.index.meta.get("built_at"),
        "new_api_calls": dict(client.calls),
    }
    summary = summarize(rows, qs, meta, hand)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    text = markdown(summary)
    (run_dir / "summary.md").write_text(text, encoding="utf-8")
    sheet = write_grades(rows, qs, run_dir / "grades.csv", hand)
    print()
    print(text)
    files = "rows.jsonl, summary.json, summary.md, grades.csv"
    print(f"Wrote results/{args.run}/: {files} ({sheet} distinct answers to grade)")
    print(client.summary())
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
