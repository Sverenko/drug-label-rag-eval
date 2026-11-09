import csv
import json
from pathlib import Path

import pytest
from fakes import NO_ANSWER, ApiFailure, FakeClock, FakeSDK

from labelrag import answer, chunk, config, evaluate, fetch, index
from labelrag.client import CachedCohere

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample"


def reply(messages, documents):
    """Fake Command: declines price questions, otherwise answers with every chunk's text."""
    if "cost" in messages[-1].content:
        return answer.DECLINE
    return " ".join(d.data["text"] for d in documents)


@pytest.fixture
def world(tmp_path, monkeypatch):
    """The two sample labels as a corpus, a fake client, and three questions about them."""
    rows, chunks = [], []
    for name in ("avapro", "avalide"):
        label = json.loads((SAMPLE / f"{name}.json").read_text(encoding="utf-8"))
        row = fetch.manifest_row(label, {"Blood pressure (ARBs)"})
        rows.append(row)
        chunks.extend(chunk.chunk_label(label, row))
    sections = tmp_path / "sections.jsonl"
    sections.write_text("".join(json.dumps(c) + "\n" for c in chunks), encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    fetch.write_manifest(rows, manifest)

    clock = FakeClock()
    sdk = FakeSDK(clock)
    sdk.answer = reply
    client = CachedCohere(sdk, db_path=tmp_path / "cache.sqlite", clock=clock, sleep=clock.sleep)
    meta = {
        "model": config.EMBED_DOC_MODEL,
        "dimension": config.EMBED_DIMENSION,
        "sections_sha256": index.digest(sections),
    }
    index.save(
        tmp_path / "index", index.build(client, chunks), [c["chunk_id"] for c in chunks], meta
    )

    def section_text(brand, section):
        return " ".join(
            c["text"] for c in chunks if c["brand_names"] == brand and c["section"] == section
        )

    def gold(brand, section):
        set_id = next(r["set_id"] for r in rows if r["brand_name"] == brand)
        return [{"set_id": set_id, "section": section}]

    qs = [
        {
            "id": "q01",
            "question": "What is the Avalide dosage and administration?",
            "answerable": True,
            "gold": gold("Avalide", "dosage_and_administration"),
            "answer": "...",
            "key_facts": [section_text("Avalide", "dosage_and_administration")[30:70]],
            "tags": ["near_duplicate"],  # Avapro, also irbesartan, is the sibling
        },
        {
            "id": "q02",
            "question": "What does the Avapro overdosage section say?",
            "answerable": True,
            "gold": gold("Avapro", "overdosage"),
            "answer": "...",
            "key_facts": [section_text("Avapro", "overdosage")[20:60]],
            "tags": ["full_text"],
        },
        {"id": "u01", "question": "How much does Avapro cost?", "answerable": False},
    ]
    questions_file = tmp_path / "questions.jsonl"
    questions_file.write_text("".join(json.dumps(q) + "\n" for q in qs), encoding="utf-8")

    monkeypatch.setattr(config, "MANIFEST", manifest)
    monkeypatch.setattr(config, "SECTIONS_FILE", sections)
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(config, "QUESTIONS_FILE", questions_file)
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    sdk.requests.clear()
    return client, sdk, {c["chunk_id"]: c for c in chunks}, tmp_path / "results" / "test"


def run(world, *extra):
    client = world[0]
    assert evaluate.main(["--run", "test", *extra], client=client) == 0
    out = world[3]
    rows = [
        json.loads(line) for line in (out / "rows.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    return rows, summary


def test_each_question_runs_in_both_configurations(world):
    rows, summary = run(world)
    assert [(r["id"], r["config"]) for r in rows] == [
        (q, c) for q in ("q01", "q02", "u01") for c in ("embed", "rerank")
    ]
    _, sdk, chunks, _ = world
    endpoints = [endpoint for endpoint, _ in sdk.requests]
    assert endpoints.count("rerank") == 3 and endpoints.count("chat") == 6
    for r in rows:
        assert len(r["context_ids"]) == 5 and r["context_ids"] == r["retrieved"][:5]
        if r["answerable"]:
            assert r["gold_rank"] is None or r["gold_rank"] >= 1
        else:
            assert r["gold_rank"] is None
    assert summary["questions"] == {"total": 3, "answerable": 2, "unanswerable": 1}


def test_gold_rank_is_the_first_chunk_from_a_gold_section(world):
    rows, _ = run(world)
    chunks = world[2]
    qs = {q["id"]: q for q in evaluate.load_questions()}
    for r in rows:
        if r["answerable"]:
            expected = evaluate.gold_rank(r["retrieved"], chunks, evaluate.gold_set(qs[r["id"]]))
            assert r["gold_rank"] == expected


def test_automatic_grades_follow_the_key_facts(world):
    rows, _ = run(world)
    chunks = world[2]
    qs = {q["id"]: q for q in evaluate.load_questions()}
    for r in rows:
        if not r["answerable"]:
            assert r["declined"] and r["grades"] == {}
            continue
        facts = qs[r["id"]]["key_facts"]
        context = " ".join(chunks[c]["text"] for c in r["context_ids"])
        first = chunks[r["context_ids"][0]]["text"]  # the fake cites the first chunk
        in_context = evaluate.facts_found(facts, context) == len(facts)
        assert r["grades"]["context_contains_answer"] == in_context
        assert r["grades"]["answer_has_key_facts"] == in_context  # the fake answers with all five
        assert r["grades"]["citations_contain_answer"] == (
            evaluate.facts_found(facts, first) == len(facts)
        )


def test_summary_counts_match_the_rows(world):
    rows, summary = run(world)
    for c in ("embed", "rerank"):
        mine = [r for r in rows if r["config"] == c and r["answerable"]]
        hits = sum(r["gold_rank"] is not None and r["gold_rank"] <= 5 for r in mine)
        assert summary["metrics"][c]["hit@5"] == {"count": hits, "n": 2, "rate": hits / 2}
        assert summary["metrics"][c]["unanswerable_declined"]["count"] == 1
    assert set(summary["by_tag"]) == {"full_text", "near_duplicate", "(untagged)"}
    assert "| Right section in the top 5 (hit@5) |" in (world[3] / "summary.md").read_text(
        encoding="utf-8"
    )


def test_sibling_ranks_measure_near_duplicate_confusion(world):
    rows, summary = run(world)
    chunks = world[2]
    qs = {q["id"]: q for q in evaluate.load_questions()}
    rows_csv = list(csv.DictReader(config.MANIFEST.open(newline="", encoding="utf-8")))
    ingredients = {
        r["set_id"]: evaluate.questions.ingredient_words(r["generic_name"]) for r in rows_csv
    }
    for r in rows:
        if r["answerable"]:
            gold = evaluate.gold_set(qs[r["id"]])
            assert r["sibling_rank"] == evaluate.sibling_rank(
                r["retrieved"], chunks, gold, ingredients
            )
    q01 = [r for r in rows if r["id"] == "q01"]
    sibling = chunks[q01[0]["retrieved"][q01[0]["sibling_rank"] - 1]]
    assert sibling["brand_names"] == "Avapro"
    assert sibling["section"] == "dosage_and_administration"
    for c in ("embed", "rerank"):
        assert summary["metrics"][c]["sibling_above_gold_in_top5"]["n"] == 1  # q01 only


def test_rerunning_makes_no_calls_and_gives_the_same_numbers(world):
    _, first = run(world)
    calls = len(world[1].requests)
    _, again = run(world)
    assert len(world[1].requests) == calls
    assert again["metrics"] == first["metrics"]
    for stage in ("embed_query", "rerank", "generate"):  # from the cache
        assert again["latency_ms"][stage] == first["latency_ms"][stage]
    # Vector search runs locally and is timed afresh, so only end to end can move, slightly.
    end = again["latency_ms"]["end_to_end"]["p50"]
    assert end == pytest.approx(first["latency_ms"]["end_to_end"]["p50"], abs=5)


def test_limit_runs_only_the_first_questions(world):
    rows, summary = run(world, "--limit", "1")
    assert {r["id"] for r in rows} == {"q01"}
    assert summary["questions"]["total"] == 1


def grade_sheet(world, **grades_by_id):
    """Fill in grades.csv: {question ID: {column: value}}."""
    sheet = world[3] / "grades.csv"
    with sheet.open(newline="", encoding="utf-8") as f:
        graded = list(csv.DictReader(f))
    for row in graded:
        row.update(grades_by_id.get(row["id"], {}))
    with sheet.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=evaluate.GRADE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(graded)
    return sheet


def test_both_graders_survive_a_rerun_and_hand_grades_take_over(world):
    run(world)
    grades = {
        "hand_correct": "n",
        "comment": "too vague",
        "claude_correct": "partial",
        "claude_comment": "half right",
    }
    sheet = grade_sheet(world, q01=grades)
    rows, summary = run(world)
    with sheet.open(newline="", encoding="utf-8") as f:
        kept = [row for row in csv.DictReader(f) if row["id"] == "q01"]
    assert kept and all({k: row[k] for k in grades} == grades for row in kept)
    assert "hand" in summary["answer_grades_from"] and summary["claude_grades"]
    for m in summary["metrics"].values():
        assert m["answer_correct"]["count"] <= 1  # q01 is now wrong in both
        assert m["answer_partial_claude"]["count"] == 1  # and partial, which counts as wrong
        assert m["answer_correct_claude"]["count"] <= 1
    graders = summary["graders"]
    assert graders["hand_vs_key_facts"]["graded"] == len(kept)  # distinct answers, as graded
    assert graders["hand_vs_claude"]["graded"] == len(kept)
    assert graders["hand_vs_claude"]["agree"] == 0  # n vs partial
    assert graders["hand_vs_claude_strict"]["agree"] == len(kept)  # both count as wrong
    assert "Answer correct (Claude's grades)" in (world[3] / "summary.md").read_text(
        encoding="utf-8"
    )


def test_regrade_rescores_the_saved_rows_without_running_anything(world):
    run(world)
    calls = len(world[1].requests)
    rows_file = world[3] / "rows.jsonl"
    before = rows_file.read_text(encoding="utf-8")
    grade_sheet(world, q01={"hand_correct": "y"}, q02={"hand_correct": "Partial"})
    assert evaluate.main(["--run", "test", "--regrade"]) == 0  # no client needed
    assert len(world[1].requests) == calls
    assert rows_file.read_text(encoding="utf-8") == before
    summary = json.loads((world[3] / "summary.json").read_text(encoding="utf-8"))
    assert summary["answer_grades_from"] == ["hand"]
    for m in summary["metrics"].values():
        assert m["answer_correct"]["count"] == 1 and m["answer_partial"]["count"] == 1


def test_regrade_needs_a_finished_run(world, capsys):
    assert evaluate.main(["--run", "missing", "--regrade"]) == 1
    assert "run it first" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Y", "y"),
        ("correct", "y"),
        (" no ", "n"),
        ("Partial", "partial"),
        ("indeterminate", "partial"),
        ("", None),
        (None, None),
        ("maybe", None),
    ],
)
def test_grade_values(value, expected):
    assert evaluate.grade(value) == expected


def test_a_failed_answer_is_recorded_and_the_run_goes_on(world):
    def no_answer_for_prices(messages, documents):
        if "cost" in messages[-1].content:
            raise ApiFailure(422, body=NO_ANSWER)
        return reply(messages, documents)

    world[1].answer = no_answer_for_prices
    rows, summary = run(world)
    failed = [r for r in rows if r["id"] == "u01"]
    assert all(r["error"] and r["answer"] == "" and not r["declined"] for r in failed)
    assert all(r["timings_ms"]["generate"] is None for r in failed)
    for c in ("embed", "rerank"):
        assert summary["metrics"][c]["failed_answers"]["count"] == 1
        assert summary["metrics"][c]["unanswerable_declined"]["count"] == 0
    assert summary["latency_ms"]["generate"]["n"] == 2  # the two answered questions
    calls = len(world[1].requests)
    run(world)
    assert len(world[1].requests) == calls  # the failure is cached too


def test_a_question_file_with_errors_stops_the_run(world, capsys):
    config.QUESTIONS_FILE.write_text('{"id": "q01"}\n', encoding="utf-8")
    assert evaluate.main(["--run", "test"], client=world[0]) == 1
    assert "has errors" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("embed", "rerank", "p"),
    [
        ([True] * 5, [True] * 5, 1.0),
        ([False] * 5, [True] * 5, 0.0625),
        ([True] + [False] * 9, [False] + [True] * 9, 2 * (1 + 10) / 1024),
    ],
)
def test_mcnemar_is_exact(embed, rerank, p):
    assert evaluate.mcnemar(embed, rerank)["p_value"] == pytest.approx(p)


def test_kappa():
    assert evaluate.kappa([(True, True), (False, False)]) == 1.0
    assert evaluate.kappa([(True, False), (False, True)]) == -1.0
    assert evaluate.kappa([]) is None
    # Three categories: observed 3/4, expected (2*1 + 1*1 + 1*2)/16, so kappa = 7/11.
    three = [("y", "y"), ("y", "partial"), ("n", "n"), ("partial", "partial")]
    assert evaluate.kappa(three) == pytest.approx(7 / 11)


@pytest.mark.parametrize(
    ("fact", "text"),
    [
        ("2,550 mg", "The maximum is **2550 mg** a day."),
        ("don't exceed", "Don’t  exceed it."),
        ("30-60", "an eGFR of 30–60"),
    ],
)
def test_key_fact_matching_ignores_formatting(fact, text):
    assert evaluate.facts_found([fact], text) == 1
