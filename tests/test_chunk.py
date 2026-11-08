import json
import re
from pathlib import Path

import pytest

from labelrag import chunk, config, fetch

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample"
EMBED_INPUT_TOKENS = 8192  # embed-v5.0 input limit, from Cohere's models endpoint


def load_sample(name):
    label = json.loads((SAMPLE / f"{name}.json").read_text(encoding="utf-8"))
    return label, fetch.manifest_row(label, {"Blood pressure (ARBs)"})


@pytest.fixture(params=["avapro", "avalide"])
def sample(request):
    """Avapro is irbesartan alone; Avalide is irbesartan with hydrochlorothiazide."""
    return load_sample(request.param)


def test_every_chunk_names_its_label_section_and_drug(sample):
    label, row = sample
    chunks = chunk.chunk_label(label, row)
    assert chunks
    for c in chunks:
        assert c["set_id"] == label["set_id"]
        assert c["section"] in chunk.SECTIONS
        assert c["brand_names"] and c["generic_name"]
        assert c["header"] == f"{c['brand_names']} ({c['generic_name']}): {c['section_title']}"
        assert c["text"].strip()
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)


def test_no_chunk_can_exceed_the_embed_input_limit(sample):
    label, row = sample
    for c in chunk.chunk_label(label, row):
        assert len(c["text"]) <= config.CHUNK_CHARS
        # A byte-level tokenizer never makes more tokens than there are bytes.
        assert len(chunk.embed_text(c).encode("utf-8")) <= EMBED_INPUT_TOKENS


def test_chunks_join_back_into_the_section_text(sample):
    label, row = sample
    chunks = chunk.chunk_label(label, row)
    for section in {c["section"] for c in chunks}:
        parts = [c for c in chunks if c["section"] == section]
        assert [c["part"] for c in parts] == list(range(1, len(parts) + 1))
        assert all(c["parts"] == len(parts) for c in parts)
        assert " ".join(c["text"] for c in parts) == chunk.clean(label[section])


def test_a_long_section_splits_at_its_subsection_headings():
    label, row = load_sample("avalide")
    parts = [c for c in chunk.chunk_label(label, row) if c["section"] == "clinical_pharmacology"]
    assert len(parts) > 1
    assert any(re.match(r"12\.\d+ [A-Z]", c["text"]) for c in parts[1:])


def test_a_combination_product_names_both_ingredients():
    label, row = load_sample("avalide")
    header = chunk.chunk_label(label, row)[0]["header"]
    assert "IRBESARTAN AND HYDROCHLOROTHIAZIDE" in header


def test_subsections_start_chunks_and_small_ones_pack_together():
    text = (
        "5 WARNINGS AND PRECAUTIONS 5.1 Alpha risk. "
        "5.2 Beta risk is described here in more detail. "
        "5.3 Gamma risk is described here in quite a lot more detail than the others."
    )
    assert chunk.split_section(text, limit=80) == [
        "5 WARNINGS AND PRECAUTIONS 5.1 Alpha risk.",
        "5.2 Beta risk is described here in more detail.",
        "5.3 Gamma risk is described here in quite a lot more detail than the others.",
    ]


def test_cross_references_are_not_split_points():
    text = "5 WARNINGS 5.1 Alpha. See Clinical Pharmacology (12.3) for more. " + "Filler. " * 20
    pieces = chunk.split_section(text.strip(), limit=80)
    assert not any(p.startswith("12.3") for p in pieces)


def test_text_without_subsections_splits_at_sentences_then_spaces():
    text = chunk.clean(["One short sentence. " * 3 + "x" * 50 + " " + "word " * 40])
    pieces = chunk.split_section(text, limit=60)
    assert all(len(p) <= 60 for p in pieces)
    assert " ".join(pieces) == text


def test_subsection_copies_tables_and_patient_guides_are_left_out():
    label = {
        "set_id": "s",
        "version": "1",
        "indications_and_usage": ["1 INDICATIONS AND USAGE Drug is indicated for hypertension."],
        "pregnancy": ["8.1 Pregnancy Risk Summary ..."],
        "adverse_reactions_table": ["<table><tr><td>Dizziness</td></tr></table>"],
        "spl_medguide": ["Medication Guide ..."],
        "package_label_principal_display_panel": ["NDC 0000-0000-00"],
    }
    row = {"set_id": "s", "brand_name": "Brand", "generic_name": "GENERIC", "theme": "Migraine"}
    assert [c["section"] for c in chunk.chunk_label(label, row)] == ["indications_and_usage"]


def test_header_lists_each_brand_once():
    label = {"set_id": "s", "indications_and_usage": ["1 INDICATIONS AND USAGE ..."]}
    row = {
        "set_id": "s",
        "brand_name": "OZEMPIC; RYBELSUS; Ozempic",
        "generic_name": "ORAL SEMAGLUTIDE",
        "theme": "Diabetes",
    }
    first = chunk.chunk_label(label, row)[0]
    assert first["header"] == "OZEMPIC / RYBELSUS (ORAL SEMAGLUTIDE): Indications and Usage"


def test_embed_text_can_leave_out_the_header():
    c = {"header": "Drug (DRUG): Overdosage", "text": "10 OVERDOSAGE ..."}
    assert chunk.embed_text(c) == "Drug (DRUG): Overdosage\n\n10 OVERDOSAGE ..."
    assert chunk.embed_text(c, with_header=False) == "10 OVERDOSAGE ..."


def test_a_label_must_match_its_manifest_row():
    label, row = load_sample("avapro")
    with pytest.raises(ValueError):
        chunk.chunk_label(label, {**row, "set_id": "someone-else"})


def test_the_command_writes_every_chunk(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    rows = []
    for name in ("avapro", "avalide"):
        label, row = load_sample(name)
        fetch.save_raw(label, raw)
        rows.append(row)
    fetch.write_manifest(rows, tmp_path / "manifest.csv")
    monkeypatch.setattr(config, "MANIFEST", tmp_path / "manifest.csv")
    monkeypatch.setattr(config, "RAW_DIR", raw)
    monkeypatch.setattr(config, "SECTIONS_FILE", tmp_path / "processed" / "sections.jsonl")
    assert chunk.main() == 0
    written = chunk.load_chunks(config.SECTIONS_FILE)
    expected = sum(len(chunk.chunk_label(*load_sample(n))) for n in ("avapro", "avalide"))
    assert len(written) == expected
