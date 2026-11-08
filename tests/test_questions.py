import json
from pathlib import Path

import pytest

from labelrag import chunk, config, fetch, questions

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample"
AVAPRO = "7885b2a8-be4e-48ab-8113-4e6ab791eb98"
AVALIDE = "0d91895e-c869-4b33-9b83-d55c1ad6317e"


@pytest.fixture
def corpus(tmp_path):
    """A manifest and a sections file built from the two sample labels."""
    rows, chunks = [], []
    for name in ("avapro", "avalide"):
        label = json.loads((SAMPLE / f"{name}.json").read_text(encoding="utf-8"))
        row = fetch.manifest_row(label, {"Blood pressure (ARBs)"})
        rows.append(row)
        chunks.extend(chunk.chunk_label(label, row))
    manifest = tmp_path / "manifest.csv"
    fetch.write_manifest(rows, manifest)
    sections = tmp_path / "sections.jsonl"
    sections.write_text("".join(json.dumps(c) + "\n" for c in chunks), encoding="utf-8")
    return manifest, sections


def check(tmp_path, corpus, *rows, with_corpus=True):
    path = tmp_path / "questions.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest, sections = corpus
    return questions.validate(path, manifest, sections if with_corpus else tmp_path / "none")


def answerable(qid="q01", **changes):
    question = {
        "id": qid,
        "question": "What is the recommended initial dose of Avapro?",
        "answerable": True,
        "gold": [{"set_id": AVAPRO, "section": "dosage_and_administration"}],
        "answer": "150 mg once daily.",
        "key_facts": ["recommended initial dose of AVAPRO is 150 mg once daily"],
    }
    return question | changes


METABOLISM = {
    "question": "How is irbesartan metabolized?",
    "answer": "By glucuronide conjugation and oxidation.",
    "key_facts": ["metabolized via glucuronide conjugation and oxidation"],
}
UNANSWERABLE = {"id": "u01", "question": "How much does Avapro cost?", "answerable": False}


def test_a_well_formed_file_passes(tmp_path, corpus):
    report = check(tmp_path, corpus, answerable(), UNANSWERABLE)
    assert report.errors == []
    assert report.warnings == []
    assert report.corpus_checked


def test_missing_fields_are_errors(tmp_path, corpus):
    no_facts = answerable()
    del no_facts["key_facts"]
    report = check(tmp_path, corpus, no_facts, {"id": "u02", "answerable": False})
    assert any("'key_facts'" in e for e in report.errors)
    assert any("missing 'question'" in e for e in report.errors)


def test_duplicate_ids_are_errors(tmp_path, corpus):
    report = check(tmp_path, corpus, answerable("q01"), answerable("q01"))
    assert "id 'q01' is used 2 times" in report.errors


def test_gold_locations_must_exist(tmp_path, corpus):
    report = check(
        tmp_path,
        corpus,
        answerable("q01", gold=[{"set_id": AVAPRO, "section": "drug_abuse_and_dependence"}]),
        answerable("q02", gold=[{"set_id": "no-such-label", "section": "overdosage"}]),
        answerable("q03", gold=[{"set_id": AVAPRO, "section": "dosage"}]),
    )
    assert len(report.errors) == 3
    assert "Avapro has no Drug Abuse and Dependence section" in report.errors[0]
    assert "isn't in the manifest" in report.errors[1]
    assert "'dosage' isn't a section key" in report.errors[2]


def test_a_question_can_have_several_gold_locations(tmp_path, corpus):
    both = [
        {"set_id": AVAPRO, "section": "clinical_pharmacology"},
        {"set_id": AVALIDE, "section": "clinical_pharmacology"},
    ]
    report = check(tmp_path, corpus, answerable(gold=both, **METABOLISM))
    assert report.errors == []
    assert report.warnings == []


def test_warns_when_the_key_facts_also_appear_in_another_label(tmp_path, corpus):
    only_avapro = [{"set_id": AVAPRO, "section": "clinical_pharmacology"}]
    report = check(tmp_path, corpus, answerable(gold=only_avapro, **METABOLISM))
    assert report.errors == []
    assert len(report.warnings) == 1
    assert "also appears in Avalide Clinical Pharmacology" in report.warnings[0]


def test_a_near_duplicate_tag_silences_only_the_other_labels(tmp_path, corpus):
    only_avapro = [{"set_id": AVAPRO, "section": "clinical_pharmacology"}]
    tagged = answerable(gold=only_avapro, tags=["near_duplicate"], **METABOLISM)
    assert check(tmp_path, corpus, tagged).warnings == []
    wrong_section = answerable(
        gold=[{"set_id": AVAPRO, "section": "adverse_reactions"}],
        tags=["near_duplicate"],
        key_facts=["150 mg once daily"],
    )
    elsewhere = [w for w in check(tmp_path, corpus, wrong_section).warnings if "also appears" in w]
    assert len(elsewhere) == 1
    assert "Avapro Dosage and Administration" in elsewhere[0]
    assert "Avalide" not in elsewhere[0]


def test_unrelated_labels_with_the_same_words_are_not_flagged():
    assert questions.ingredient_words("IRBESARTAN AND HYDROCHLOROTHIAZIDE") == {
        "irbesartan",
        "hydrochlorothiazide",
    }
    assert questions.ingredient_words("OLMESARTAN MEDOXOMIL") == {"olmesartan"}


def test_warns_when_a_key_fact_is_not_word_for_word(tmp_path, corpus):
    report = check(tmp_path, corpus, answerable(key_facts=["150 milligrams a day"]))
    assert any("doesn't appear word for word" in w for w in report.warnings)


def test_unanswerable_questions_cannot_have_gold(tmp_path, corpus):
    gold = [{"set_id": AVAPRO, "section": "overdosage"}]
    report = check(tmp_path, corpus, UNANSWERABLE | {"gold": gold})
    assert any("can't have gold locations" in e for e in report.errors)


def test_unknown_fields_and_tags_are_warnings(tmp_path, corpus):
    report = check(tmp_path, corpus, answerable(tags=["near_duplicates"], keyfacts=["x"]))
    assert report.errors == []
    assert any("unknown field 'keyfacts'" in w for w in report.warnings)
    assert any("unfamiliar tag 'near_duplicates'" in w for w in report.warnings)


def test_corpus_checks_wait_for_the_sections_file(tmp_path, corpus):
    missing_section = [{"set_id": AVAPRO, "section": "drug_abuse_and_dependence"}]
    report = check(tmp_path, corpus, answerable(gold=missing_section), with_corpus=False)
    assert report.errors == []
    assert not report.corpus_checked


def test_find_and_show_help_write_questions(corpus, monkeypatch, capsys):
    manifest, sections = corpus
    monkeypatch.setattr(config, "MANIFEST", manifest)
    monkeypatch.setattr(config, "SECTIONS_FILE", sections)
    assert questions.main(["find", "irbesartan"]) == 0
    found = capsys.readouterr().out
    assert AVAPRO in found and AVALIDE in found
    assert questions.DAILYMED.format(AVAPRO) in found
    assert questions.main(["show", "Avapro", "dosage_and_administration"]) == 0
    assert "150 mg once daily" in capsys.readouterr().out


def test_the_eval_questions_are_valid():
    if not config.QUESTIONS_FILE.exists():
        pytest.skip("eval/questions.jsonl hasn't been written yet")
    report = questions.validate()
    assert report.errors == []
