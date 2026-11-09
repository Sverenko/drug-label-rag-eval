"""Fixtures shared by the eval and experiment tests."""

import json
from pathlib import Path

import pytest
from fakes import FakeClock, FakeSDK, reply

from labelrag import chunk, config, fetch, index
from labelrag.client import CachedCohere

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample"


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
