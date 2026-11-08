import json
from pathlib import Path

import numpy as np
import pytest
from fakes import FakeClock, FakeSDK

from labelrag import chunk, config, fetch, index
from labelrag.client import CachedCohere

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample"


@pytest.fixture
def chunks():
    rows = []
    for name in ("avapro", "avalide"):
        label = json.loads((SAMPLE / f"{name}.json").read_text(encoding="utf-8"))
        rows.extend(chunk.chunk_label(label, fetch.manifest_row(label, {"Blood pressure (ARBs)"})))
    return rows


@pytest.fixture
def sections_file(tmp_path, chunks):
    path = tmp_path / "sections.jsonl"
    path.write_text("".join(json.dumps(c) + "\n" for c in chunks), encoding="utf-8")
    return path


def client_for(tmp_path):
    clock = FakeClock()
    sdk = FakeSDK(clock)
    db = tmp_path / "cache.sqlite"
    return CachedCohere(sdk, db_path=db, clock=clock, sleep=clock.sleep), sdk


def build_and_save(tmp_path, chunks, sections_file, with_header=True):
    client, sdk = client_for(tmp_path)
    vectors = index.build(client, chunks, with_header)
    meta = {
        "model": config.EMBED_DOC_MODEL,
        "dimension": config.EMBED_DIMENSION,
        "with_header": with_header,
        "sections_sha256": index.digest(sections_file),
    }
    index.save(tmp_path / "index", vectors, [c["chunk_id"] for c in chunks], meta)
    return sdk


def test_build_save_and_load(tmp_path, chunks, sections_file):
    sdk = build_and_save(tmp_path, chunks, sections_file)
    loaded = index.load(tmp_path / "index", sections_file)
    assert loaded.chunk_ids == [c["chunk_id"] for c in chunks]
    assert loaded.vectors.shape == (len(chunks), config.EMBED_DIMENSION)
    assert np.allclose(np.linalg.norm(loaded.vectors, axis=1), 1.0)
    first_request = sdk.requests[0][1]
    assert first_request["input_type"] == "search_document"
    assert first_request["texts"][0] == chunk.embed_text(chunks[0])


def test_rebuilding_from_the_cache_makes_no_calls(tmp_path, chunks, sections_file):
    build_and_save(tmp_path, chunks, sections_file)
    again = build_and_save(tmp_path, chunks, sections_file)
    assert again.requests == []


def test_no_header_embeds_the_text_alone(tmp_path, chunks, sections_file):
    sdk = build_and_save(tmp_path, chunks, sections_file, with_header=False)
    assert sdk.requests[0][1]["texts"][0] == chunks[0]["text"]


def test_load_refuses_another_model_or_dimension(tmp_path, chunks, sections_file, monkeypatch):
    build_and_save(tmp_path, chunks, sections_file)
    monkeypatch.setattr(config, "EMBED_DIMENSION", 1024)
    with pytest.raises(ValueError, match="Rebuild it"):
        index.load(tmp_path / "index", sections_file)


def test_load_refuses_an_index_older_than_the_chunks(tmp_path, chunks, sections_file):
    build_and_save(tmp_path, chunks, sections_file)
    sections_file.write_text(sections_file.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="chunks have changed"):
        index.load(tmp_path / "index", sections_file)


def test_load_explains_a_missing_index(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m labelrag.index"):
        index.load(tmp_path / "nothing-here")
