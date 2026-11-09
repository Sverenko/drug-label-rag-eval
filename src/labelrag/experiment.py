"""Retrieval experiments: change one setting and compare hit@k and latency with the baseline.

    python -m labelrag.experiment query_pro --query-model embed-v5.0-pro --no-rerank
    python -m labelrag.experiment rerank_fast --rerank-model rerank-v5.0-fast
    python -m labelrag.index --no-header         # once: builds data/index-no-header/
    python -m labelrag.experiment no_header --no-header

An experiment retrieves for every answerable question again with one setting changed, and
generates no answers. It compares hit@1, hit@5 and hit@50 with the baseline run's
retrieval of the same questions, using McNemar's exact test, for the embedding order and,
unless --no-rerank, the reranked order, and reports the latency of each stage. Results go
to results/experiments/<name>/: rows.jsonl, summary.json and summary.md.

Calls go through the cache like everything else, so only what changes costs anything: the
no-header experiment, for one, reuses the cached query embeddings. Latency for a cached
call is the latency of the original call.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from labelrag import config, evaluate, index
from labelrag.client import CachedCohere, stamp, utc_now
from labelrag.retrieve import POOL, Retriever

STAGES = {"embed_query": "Query embedding", "vector_search": "Vector search", "rerank": "Rerank"}


def baseline_retrieval(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Per answerable question in an eval run: the gold rank in each order, and the timings."""
    lines = (run_dir / "rows.jsonl").read_text(encoding="utf-8").splitlines()
    out: dict[str, dict[str, Any]] = {}
    for r in (json.loads(line) for line in lines if line.strip()):
        if not r["answerable"]:
            continue
        mine = out.setdefault(r["id"], {"timings_ms": {}})
        mine[f"{r['config']}_rank"] = r["gold_rank"]
        for stage in ("embed_query", "vector_search", "rerank"):
            if r["timings_ms"].get(stage) is not None:
                mine["timings_ms"][stage] = r["timings_ms"][stage]
    return out


def run_experiment(
    retriever: Retriever,
    qs: list[dict[str, Any]],
    query_model: str,
    rerank_model: str,
    rerank: bool,
    with_header: bool,
    progress=print,
) -> list[dict[str, Any]]:
    rows = []
    for q in qs:
        r = retriever.retrieve(
            q["question"],
            query_model=query_model,
            rerank_model=rerank_model,
            rerank=rerank,
            with_header=with_header,
        )
        gold = evaluate.gold_set(q)
        row = {
            "id": q["id"],
            "tags": q.get("tags") or [],
            "embed_rank": evaluate.gold_rank(r.embed_ids, retriever.chunks, gold),
            "rerank_rank": evaluate.gold_rank(r.rerank_ids, retriever.chunks, gold)
            if rerank
            else None,
            "embed_ids": r.embed_ids,
            "rerank_ids": r.rerank_ids,
            "timings_ms": r.timings_ms,
            "cached": r.cached,
        }
        rows.append(row)
        ranks = [f"{order} #{row[f'{order}_rank'] or '-'}" for order in ("embed", "rerank")]
        progress(f"  {q['id']:<4} " + (", ".join(ranks) if rerank else ranks[0]))
    return rows


def within(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def summarize(
    name: str, rows: list[dict[str, Any]], base: dict, settings: dict[str, Any]
) -> dict[str, Any]:
    ids = [r["id"] for r in rows]
    orders = ["embed", "rerank"] if settings["rerank"] else ["embed"]
    hits, tests = {}, {}
    for order in orders:
        # Rerank reorders the same 50, so hit@50 is the same for both orders.
        for k in (1, 5, POOL) if order == "embed" else (1, 5):
            theirs = [within(base[i][f"{order}_rank"], k) for i in ids]
            mine = [within(r[f"{order}_rank"], k) for r in rows]
            key = f"{order} hit@{k}"
            hits[key] = {"baseline": evaluate.rate(theirs), name: evaluate.rate(mine)}
            if k in (1, 5):
                pairs = list(zip(ids, theirs, mine, strict=True))
                tests[key] = {
                    "only_baseline": [i for i, a, b in pairs if a and not b],
                    "only_experiment": [i for i, a, b in pairs if b and not a],
                    "p_value": evaluate.mcnemar(theirs, mine)["p_value"],
                }
    latency = {}
    for stage in STAGES:
        if stage == "rerank" and not settings["rerank"]:
            continue
        latency[stage] = {
            "baseline": evaluate.percentiles([base[i]["timings_ms"].get(stage) for i in ids]),
            name: evaluate.percentiles([r["timings_ms"].get(stage) for r in rows]),
        }
    return {
        "name": name,
        "finished": stamp(utc_now()),
        "settings": settings,
        "questions": len(rows),
        "hits": hits,
        "mcnemar": tests,
        "latency_ms": latency,
    }


def describe(settings: dict[str, Any]) -> str:
    changes = []
    if settings["query_model"] != config.EMBED_QUERY_MODEL:
        changes.append(
            f"query embeddings from `{config.EMBED_QUERY_MODEL}` to `{settings['query_model']}`"
        )
    if settings["rerank"] and settings["rerank_model"] != config.RERANK_MODEL:
        changes.append(f"rerank from `{config.RERANK_MODEL}` to `{settings['rerank_model']}`")
    if not settings["with_header"]:
        changes.append(
            "chunks embedded and reranked without their drug-and-section header "
            "(a separate index, data/index-no-header/)"
        )
    text = "Changed: " + ("; ".join(changes) if changes else "nothing") + "."
    if not settings["rerank"]:
        text += " Rerank skipped: embedding order only."
    return text


def markdown(summary: dict[str, Any]) -> str:
    name, settings = summary["name"], summary["settings"]
    labels = {1: "right section first (hit@1)", 5: "in the top 5 (hit@5)", POOL: "in the 50"}
    lines = [
        f"# Retrieval experiment: {name}",
        "",
        f"{describe(settings)} {summary['questions']} answerable questions, compared with "
        f"the retrieval in results/{settings['baseline']}/. No answers are generated.",
        "",
        f"| | {settings['baseline']} | {name} |",
        "|---|---|---|",
    ]
    for key, cells in summary["hits"].items():
        order, k = key.split(" hit@")
        prefix = "Embedding order" if order == "embed" else "Reranked"
        values = [evaluate.fmt(cell) for cell in cells.values()]
        lines.append(f"| {prefix}: {labels[int(k)]} | " + " | ".join(values) + " |")
    for stage, cells in summary["latency_ms"].items():
        values = [f"{p['p50']:,.0f} / {p['p95']:,.0f}" if p["n"] else "-" for p in cells.values()]
        lines.append(f"| {STAGES[stage]}, p50 / p95 ms | " + " | ".join(values) + " |")
    lines += [
        "",
        f"McNemar's exact test, {settings['baseline']} vs {name}, on the same questions:",
        "",
        f"| Metric | Only {settings['baseline']} right | Only {name} right | p |",
        "|---|---|---|---|",
    ]
    for key, test in summary["mcnemar"].items():
        only_a = ", ".join(test["only_baseline"]) or "none"
        only_b = ", ".join(test["only_experiment"]) or "none"
        lines.append(f"| {key} | {only_a} | {only_b} | {test['p_value']:.3f} |")
    lines += [
        "",
        f"Latency in {settings['baseline']} comes from that run's calls and in {name} from "
        "this one's; a call served from the cache keeps the latency of the original call. "
        "All are single calls from the same machine.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None, client: CachedCohere | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Retrieve every answerable question with one setting changed, without "
        "answers, and compare hit@k and latency with the baseline."
    )
    parser.add_argument("name", help="results go to results/experiments/<name>/")
    parser.add_argument("--query-model", default=config.EMBED_QUERY_MODEL)
    parser.add_argument("--rerank-model", default=config.RERANK_MODEL)
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="search data/index-no-header/ and rerank the chunks without their header",
    )
    parser.add_argument("--no-rerank", action="store_true", help="embedding order only")
    parser.add_argument("--baseline", default="baseline", help="the eval run to compare with")
    parser.add_argument(
        "--max-calls",
        type=int,
        default=130,
        help="stop before making more than this many new API calls",
    )
    args = parser.parse_args(argv)
    base_dir = config.RESULTS_DIR / args.baseline
    if not (base_dir / "rows.jsonl").exists():
        print(f"No eval run in {base_dir} to compare with: run python -m labelrag.evaluate.")
        return 1
    base = baseline_retrieval(base_dir)
    qs = [q for q in evaluate.load_questions() if q["answerable"] and q["id"] in base]
    settings = {
        "query_model": args.query_model,
        "rerank_model": args.rerank_model,
        "rerank": not args.no_rerank,
        "with_header": not args.no_header,
        "baseline": args.baseline,
    }
    client = client or CachedCohere(max_new_calls=args.max_calls)
    retriever = Retriever.load(client, index.index_dir(not args.no_header))
    print(f"Retrieving {len(qs)} questions for {args.name}:")
    rows = run_experiment(
        retriever, qs, args.query_model, args.rerank_model, not args.no_rerank, not args.no_header
    )
    out = config.RESULTS_DIR / "experiments" / args.name
    out.mkdir(parents=True, exist_ok=True)
    with (out / "rows.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    summary = summarize(args.name, rows, base, settings)
    summary["new_api_calls"] = dict(client.calls)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    text = markdown(summary)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print()
    print(text)
    print(client.summary())
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
