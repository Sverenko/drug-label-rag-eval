"""Chart hit@k for k from 1 to 50, for the embedding order and the reranked order.

    python -m labelrag.chart                    # from results/prompt_v3
    python -m labelrag.chart --run baseline

Reads a run's rows.jsonl, where each answerable question has the rank of its first gold
chunk in each order, and writes two SVGs to docs/: one for light pages and one for dark,
which the README picks between with a <picture> element. It's plain SVG with no plotting
library, since the chart is two lines and some text. The colors are the first two slots
of a categorical palette checked for color-blind separation and contrast in both modes.
"""

import argparse
import json
import math
import sys
from html import escape
from pathlib import Path
from typing import Any

from labelrag import config

KS = range(1, 51)
TICKS = (1, 2, 3, 5, 10, 20, 50)
SERIES = {"embed": "Embedding order", "rerank": "Reranked"}
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "embed": "#2a78d6",
        "rerank": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19",
        "primary": "#f0efec",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "embed": "#3987e5",
        "rerank": "#d95926",
    },
}
WIDTH, HEIGHT = 680, 380
LEFT, RIGHT, TOP, BOTTOM = 52, 28, 92, 52
FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"


def hit_curves(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    """For each order, the share of answerable questions whose gold chunk is in the top k."""
    curves = {}
    for order in SERIES:
        ranks = [r["gold_rank"] for r in rows if r["answerable"] and r["config"] == order]
        curves[order] = [
            sum(rank is not None and rank <= k for rank in ranks) / len(ranks) for k in KS
        ]
    return curves


def svg(curves: dict[str, list[float]], n: int, theme: dict[str, str], source: str) -> str:
    plot_w, plot_h = WIDTH - LEFT - RIGHT, HEIGHT - TOP - BOTTOM
    lowest = min(min(values) for values in curves.values())
    floor = max(0.0, math.floor((lowest - 0.05) * 10) / 10)

    def x(k: float) -> float:
        return LEFT + math.log(k) / math.log(KS[-1]) * plot_w

    def y(value: float) -> float:
        return TOP + (1 - (value - floor) / (1 - floor)) * plot_h

    def text(px, py, body, fill, size=12, anchor="start", weight="normal"):
        return (
            f'<text x="{px:.1f}" y="{py:.1f}" fill="{fill}" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{escape(body)}</text>'
        )

    first = {order: values[0] for order, values in curves.items()}
    title = f"Right section within the top k, {n} answerable questions"
    desc = "; ".join(
        f"{SERIES[o]}: hit@1 {v[0]:.0%}, hit@5 {v[4]:.0%}, hit@10 {v[9]:.0%}, hit@50 {v[-1]:.0%}"
        for o, v in curves.items()
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" font-family="{FONT}" role="img">',
        f"<title>{escape(title)}</title>",
        f"<desc>{escape(desc)}. From {escape(source)}.</desc>",
        f'<rect width="{WIDTH}" height="{HEIGHT}" rx="8" fill="{theme["surface"]}"/>',
        text(LEFT, 30, title, theme["primary"], 15, weight="600"),
        text(LEFT, 50, f"Dots mark hit@1. From {source}; k on a log scale.", theme["secondary"]),
    ]
    legend_x = LEFT
    for order, label in SERIES.items():
        parts.append(
            f'<line x1="{legend_x}" y1="70" x2="{legend_x + 20}" y2="70" '
            f'stroke="{theme[order]}" stroke-width="2" stroke-linecap="round"/>'
        )
        parts.append(text(legend_x + 26, 74, label, theme["secondary"]))
        legend_x += 26 + 7 * len(label) + 28
    steps = round((1 - floor) * 10)
    for i in range(steps + 1):
        value = floor + i / 10
        parts.append(
            f'<line x1="{LEFT}" y1="{y(value):.1f}" x2="{WIDTH - RIGHT}" y2="{y(value):.1f}" '
            f'stroke="{theme["axis"] if i == 0 else theme["grid"]}" stroke-width="1"/>'
        )
        parts.append(text(LEFT - 8, y(value) + 4, f"{value:.0%}", theme["muted"], 11, "end"))
    for k in TICKS:
        parts.append(text(x(k), TOP + plot_h + 18, str(k), theme["muted"], 11, "middle"))
    parts.append(text(LEFT + plot_w / 2, HEIGHT - 12, "k", theme["muted"], 11, "middle"))
    for order, values in curves.items():
        points = " ".join(f"{x(k):.1f},{y(v):.1f}" for k, v in zip(KS, values, strict=True))
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{theme[order]}" '
            f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
        )
    # Mark hit@1, where the lines differ most. The legend names the series; the values sit
    # just below and right of each dot, clear of both lines as they climb.
    for order in first:
        cy = y(first[order])
        parts.append(
            f'<circle cx="{x(1):.1f}" cy="{cy:.1f}" r="4" fill="{theme[order]}" '
            f'stroke="{theme["surface"]}" stroke-width="2"/>'
        )
        parts.append(text(x(1) + 9, cy + 13, f"{first[order]:.0%}", theme["secondary"]))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chart hit@k for both retrieval orders.")
    parser.add_argument("--run", default="prompt_v3", help="the eval run in results/ to chart")
    parser.add_argument("--out", type=Path, default=config.ROOT / "docs")
    args = parser.parse_args(argv)
    path = config.RESULTS_DIR / args.run / "rows.jsonl"
    if not path.exists():
        print(f"No rows in {path}: run python -m labelrag.evaluate first.")
        return 1
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    curves = hit_curves(rows)
    n = sum(r["answerable"] for r in rows if r["config"] == "embed")
    args.out.mkdir(parents=True, exist_ok=True)
    for mode, theme in THEMES.items():
        out = args.out / f"hit-at-k-{mode}.svg"
        out.write_text(svg(curves, n, theme, f"results/{args.run}"), encoding="utf-8")
        print(f"Wrote {out}")
    for order, values in curves.items():
        print(f"  {SERIES[order]}: " + ", ".join(f"hit@{k} {values[k - 1]:.0%}" for k in TICKS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
