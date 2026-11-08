import json
from pathlib import Path

import pytest
from fakes import FakeClock, FakeSDK

from labelrag import answer, chunk, config, fetch, index
from labelrag.client import CachedCohere

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample"
QUESTION = "What is the recommended initial dose of Avalide for hypertension?"


@pytest.fixture
def chunks():
    rows = []
    for name in ("avapro", "avalide"):
        label = json.loads((SAMPLE / f"{name}.json").read_text(encoding="utf-8"))
        rows.extend(chunk.chunk_label(label, fetch.manifest_row(label, {"Blood pressure (ARBs)"})))
    return rows


@pytest.fixture
def fake(tmp_path):
    clock = FakeClock()
    sdk = FakeSDK(clock)
    client = CachedCohere(sdk, db_path=tmp_path / "cache.sqlite", clock=clock, sleep=clock.sleep)
    return client, sdk


def test_the_prompt_file_asks_for_the_exact_decline_sentence():
    assert answer.DECLINE in answer.load_prompt()


def test_sends_the_prompt_question_and_chunks_by_id(fake, chunks):
    client, sdk = fake
    sent = chunks[:5]
    answer.answer(client, QUESTION, sent)
    request = sdk.requests[0][1]
    system, user = request["messages"]
    assert system.content == answer.load_prompt()
    assert user.content == QUESTION
    assert [d.id for d in request["documents"]] == [c["chunk_id"] for c in sent]
    assert request["documents"][0].data == {"title": sent[0]["header"], "text": sent[0]["text"]}
    assert (request["temperature"], request["seed"]) == (answer.TEMPERATURE, answer.SEED)


def test_citations_point_at_chunk_ids(fake, chunks):
    client, _ = fake
    result = answer.answer(client, QUESTION, chunks[:5])
    assert result.cited_ids == [chunks[0]["chunk_id"]]
    assert result.citations[0]["sources"] == [chunks[0]["chunk_id"]]
    assert not result.declined


@pytest.mark.parametrize(
    "reply",
    [
        answer.DECLINE,
        "The provided label excerpts don’t answer this question.",
        "the provided label excerpts don't answer this question",
    ],
)
def test_declines_are_detected(fake, chunks, reply):
    client, sdk = fake
    sdk.answer = reply
    sdk.cite = False
    result = answer.answer(client, "How much does Avalide cost?", chunks[:5])
    assert result.declined
    assert result.citations == []


def test_answers_come_from_the_cache_the_second_time(fake, chunks):
    client, sdk = fake
    first = answer.answer(client, QUESTION, chunks[:5])
    again = answer.answer(client, QUESTION, chunks[:5])
    assert len(sdk.requests) == 1
    assert again.cached and again.text == first.text
    assert again.latency_ms == first.latency_ms


def test_a_different_prompt_is_a_different_request(fake, chunks, tmp_path, monkeypatch):
    client, sdk = fake
    answer.answer(client, QUESTION, chunks[:5])
    (tmp_path / "answer_v2.md").write_text("A different prompt.", encoding="utf-8")
    monkeypatch.setattr(config, "PROMPTS_DIR", tmp_path)
    answer.answer(client, QUESTION, chunks[:5], prompt_version="answer_v2")
    assert len(sdk.requests) == 2


def test_the_command_answers_from_the_reranked_top_five(
    fake, chunks, tmp_path, monkeypatch, capsys
):
    client, sdk = fake
    sections = tmp_path / "sections.jsonl"
    sections.write_text("".join(json.dumps(c) + "\n" for c in chunks), encoding="utf-8")
    meta = {
        "model": config.EMBED_DOC_MODEL,
        "dimension": config.EMBED_DIMENSION,
        "sections_sha256": index.digest(sections),
    }
    index.save(
        tmp_path / "index", index.build(client, chunks), [c["chunk_id"] for c in chunks], meta
    )
    monkeypatch.setattr(config, "SECTIONS_FILE", sections)
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "index")
    assert answer.main([QUESTION], client=client) == 0
    out = capsys.readouterr().out
    assert "Answer from the rerank top 5" in out
    assert "The answer." in out
    chat = [kwargs for endpoint, kwargs in sdk.requests if endpoint == "chat"][0]
    assert len(chat["documents"]) == 5
