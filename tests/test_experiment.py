import json

from labelrag import config, evaluate, experiment, index


def read(name):
    out = config.RESULTS_DIR / "experiments" / name
    lines = (out / "rows.jsonl").read_text(encoding="utf-8").splitlines()
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    return [json.loads(line) for line in lines], summary, (out / "summary.md").read_text("utf-8")


def test_a_query_model_experiment_only_embeds_the_questions_again(world):
    client, sdk = world[0], world[1]
    assert evaluate.main(["--run", "test"], client=client) == 0
    before = len(sdk.requests)
    args = ["pro", "--query-model", "embed-v5.0-pro", "--no-rerank", "--baseline", "test"]
    assert experiment.main(args, client=client) == 0
    new = sdk.requests[before:]
    assert [endpoint for endpoint, _ in new] == ["embed", "embed"]  # the answerable questions
    assert all(kwargs["model"] == "embed-v5.0-pro" for _, kwargs in new)

    rows, summary, text = read("pro")
    assert [r["id"] for r in rows] == ["q01", "q02"]
    assert all(r["rerank_rank"] is None and r["rerank_ids"] == [] for r in rows)
    assert set(summary["hits"]) == {"embed hit@1", "embed hit@5", "embed hit@50"}
    assert "rerank" not in summary["latency_ms"]
    # The fake's vectors don't depend on the model, so retrieval matches the baseline.
    assert summary["hits"]["embed hit@5"]["baseline"] == summary["hits"]["embed hit@5"]["pro"]
    assert "query embeddings from `embed-v5.0-fast` to `embed-v5.0-pro`" in text


def test_a_no_header_experiment_uses_its_own_index_and_bare_chunks(world, tmp_path, monkeypatch):
    client, sdk, chunks = world[0], world[1], list(world[2].values())
    assert evaluate.main(["--run", "test"], client=client) == 0
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    meta = {
        "model": config.EMBED_DOC_MODEL,
        "dimension": config.EMBED_DIMENSION,
        "sections_sha256": index.digest(config.SECTIONS_FILE),
    }
    vectors = index.build(client, chunks, with_header=False)
    index.save(index.index_dir(False), vectors, [c["chunk_id"] for c in chunks], meta)
    before = len(sdk.requests)
    assert experiment.main(["bare", "--no-header", "--baseline", "test"], client=client) == 0
    new = sdk.requests[before:]
    assert [endpoint for endpoint, _ in new] == ["rerank", "rerank"]  # queries from the cache
    headers = {c["header"] for c in chunks}
    for _, kwargs in new:
        assert not any(doc.startswith(h) for doc in kwargs["documents"] for h in headers)

    _, summary, text = read("bare")
    assert set(summary["hits"]) == {
        "embed hit@1",
        "embed hit@5",
        "embed hit@50",
        "rerank hit@1",
        "rerank hit@5",
    }
    assert "without their drug-and-section header" in text


def test_the_summary_compares_hits_on_the_same_questions():
    base = {
        "q1": {"embed_rank": 1, "timings_ms": {"embed_query": 80.0}},
        "q2": {"embed_rank": 7, "timings_ms": {"embed_query": 90.0}},
    }
    rows = [
        {"id": "q1", "embed_rank": 2, "timings_ms": {"embed_query": 100.0}},
        {"id": "q2", "embed_rank": 4, "timings_ms": {"embed_query": 120.0}},
    ]
    settings = {
        "query_model": "embed-v5.0-pro",
        "rerank_model": config.RERANK_MODEL,
        "rerank": False,
        "with_header": True,
        "baseline": "base",
    }
    summary = experiment.summarize("pro", rows, base, settings)
    assert summary["hits"]["embed hit@1"]["baseline"]["count"] == 1
    assert summary["hits"]["embed hit@1"]["pro"]["count"] == 0
    assert summary["mcnemar"]["embed hit@1"]["only_baseline"] == ["q1"]
    assert summary["mcnemar"]["embed hit@5"]["only_experiment"] == ["q2"]
    assert summary["latency_ms"]["embed_query"]["pro"]["p50"] == 110.0
    assert "| Query embedding, p50 / p95 ms | 85 / 90 | 110 / 119 |" in experiment.markdown(summary)


def test_an_experiment_needs_a_baseline_run(world, capsys):
    assert experiment.main(["x", "--baseline", "missing"], client=world[0]) == 1
    assert "No eval run" in capsys.readouterr().out
