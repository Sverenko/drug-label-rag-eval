import json

from labelrag import chart, config


def rows(embed_ranks, rerank_ranks):
    out = [{"answerable": True, "config": "embed", "gold_rank": r} for r in embed_ranks]
    out += [{"answerable": True, "config": "rerank", "gold_rank": r} for r in rerank_ranks]
    out += [{"answerable": False, "config": c, "gold_rank": None} for c in ("embed", "rerank")]
    return out


def test_hit_curves_count_each_gold_rank_against_every_k():
    curves = chart.hit_curves(rows([1, 3, None, 50], [1, 1, 2, None]))
    assert len(curves["embed"]) == 50
    assert (curves["embed"][0], curves["embed"][2], curves["embed"][49]) == (0.25, 0.5, 0.75)
    assert (curves["rerank"][0], curves["rerank"][1], curves["rerank"][49]) == (0.5, 0.75, 0.75)


def test_each_theme_draws_both_lines_in_its_own_colors(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    run = tmp_path / "results" / "final"
    run.mkdir(parents=True)
    lines = rows([1, 2, 6, 9], [1, 1, 2, 4])
    (run / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines), "utf-8")
    assert chart.main(["--run", "final", "--out", str(tmp_path / "docs")]) == 0
    for mode, theme in chart.THEMES.items():
        svg = (tmp_path / "docs" / f"hit-at-k-{mode}.svg").read_text(encoding="utf-8")
        assert svg.count("<polyline") == 2
        assert theme["embed"] in svg and theme["rerank"] in svg and theme["surface"] in svg
        assert "4 answerable questions" in svg and "<desc>" in svg


def test_a_missing_run_is_reported(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    assert chart.main(["--run", "nope", "--out", str(tmp_path)]) == 1
    assert "No rows" in capsys.readouterr().out
