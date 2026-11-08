import json
from pathlib import Path

import pytest
from fakes import FakeClock, FakeSDK

from labelrag import chunk, config, fetch, index, retrieve
from labelrag.client import CachedCohere

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample"
QUESTION = "What is the recommended initial dose of Avalide for hypertension?"


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """A fake client, and an index of the two sample labels saved where the config points."""
    chunks = []
    for name in ("avapro", "avalide"):
        label = json.loads((SAMPLE / f"{name}.json").read_text(encoding="utf-8"))
        chunks.extend(
            chunk.chunk_label(label, fetch.manifest_row(label, {"Blood pressure (ARBs)"}))
        )
    sections = tmp_path / "sections.jsonl"
    sections.write_text("".join(json.dumps(c) + "\n" for c in chunks), encoding="utf-8")
    clock = FakeClock()
    sdk = FakeSDK(clock)
    client = CachedCohere(sdk, db_path=tmp_path / "cache.sqlite", clock=clock, sleep=clock.sleep)
    meta = {
        "model": config.EMBED_DOC_MODEL,
        "dimension": config.EMBED_DIMENSION,
        "sections_sha256": index.digest(sections),
    }
    vectors = index.build(client, chunks)
    index.save(tmp_path / "index", vectors, [c["chunk_id"] for c in chunks], meta)
    monkeypatch.setattr(config, "SECTIONS_FILE", sections)
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "index")
    sdk.requests.clear()
    return retrieve.Retriever.load(client), sdk, chunks


def test_both_orders_cover_the_same_pool(setup):
    retriever, _, chunks = setup
    result = retriever.retrieve(QUESTION)
    assert len(result.embed_ids) == min(retrieve.POOL, len(chunks))
    assert set(result.rerank_ids) == set(result.embed_ids)
    assert result.embed_scores == sorted(result.embed_scores, reverse=True)
    assert result.rerank_scores == sorted(result.rerank_scores, reverse=True)
    assert result.top("rerank") == result.rerank_ids[:5]
    assert result.top("embed", 3) == result.embed_ids[:3]


def test_query_uses_embed_fast_and_rerank_sees_the_headers(setup):
    retriever, sdk, _ = setup
    result = retriever.retrieve(QUESTION)
    (_, embed_call), (_, rerank_call) = sdk.requests
    assert embed_call["model"] == config.EMBED_QUERY_MODEL
    assert embed_call["input_type"] == "search_query"
    assert rerank_call["model"] == config.RERANK_MODEL
    assert rerank_call["top_n"] == len(result.embed_ids)
    first = retriever.chunks[result.embed_ids[0]]
    assert rerank_call["documents"][0] == chunk.embed_text(first)


def test_rerank_brings_the_dosage_section_up(setup):
    retriever, _, _ = setup
    result = retriever.retrieve(QUESTION)
    best = retriever.chunks[result.rerank_ids[0]]
    assert best["brand_names"] == "Avalide"
    assert best["section"] == "dosage_and_administration"


def test_timings_and_a_repeat_from_the_cache(setup):
    retriever, sdk, _ = setup
    first = retriever.retrieve(QUESTION)
    assert set(first.timings_ms) == {"embed_query", "vector_search", "rerank"}
    assert first.cached == {"embed_query": False, "rerank": False}
    calls = len(sdk.requests)
    again = retriever.retrieve(QUESTION)
    assert len(sdk.requests) == calls
    assert again.cached == {"embed_query": True, "rerank": True}
    assert (again.embed_ids, again.rerank_ids) == (first.embed_ids, first.rerank_ids)
    assert again.timings_ms["rerank"] == first.timings_ms["rerank"]


def test_a_pool_bigger_than_the_index_returns_everything(setup):
    retriever, _, chunks = setup
    result = retriever.retrieve(QUESTION, pool=10_000)
    assert len(result.embed_ids) == len(chunks)


def test_the_command_prints_both_top_fives(setup, capsys):
    retriever, _, _ = setup
    assert retrieve.main([QUESTION], client=retriever.client) == 0
    out = capsys.readouterr().out
    assert "Embedding only (top 5 of 50)" in out
    assert "Reranked (top 5 of 50)" in out
    assert "Timings: embed query" in out
