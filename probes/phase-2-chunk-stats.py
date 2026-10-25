"""Phase 2 inspection figures: how long are the chunks, and what are they made of?

Reads parsed/chunks/chunks.jsonl and writes to docs/figures/:
  phase2-chunk-tokens.png    histogram of chunk body length, with target and max marked
  phase2-chunk-stats.md      the same numbers as tables (the figure's table view)

Run:  uv run python probes/phase-2-chunk-stats.py [--config config.yaml]
"""

import argparse
import statistics
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rag.common.config import load_config  # noqa: E402
from rag.common.io import read_jsonl  # noqa: E402
from rag.parse.pipeline import CHUNKS, step_dir  # noqa: E402
from rag.parse.schemas import Chunk, ElementType  # noqa: E402

BIN_WIDTH = 16
SURFACE, TEXT, TEXT_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLOURS = {  # dataviz reference palette, categorical slots 1-4 in fixed order
    ElementType.PARAGRAPH: "#2a78d6",
    ElementType.TABLE: "#eb6834",
    ElementType.FIGURE_CAPTION: "#1baf7a",
    ElementType.ABSTRACT: "#eda100",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    chunks = read_jsonl(step_dir(config, "chunk") / CHUNKS, Chunk)
    c = config.parse.chunk
    out = config.paths.figures
    out.mkdir(parents=True, exist_ok=True)

    bins = list(range(0, c.max_tokens + BIN_WIDTH, BIN_WIDTH))
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    kinds = [k for k in COLOURS if any(ch.element_type is k for ch in chunks)]
    ax.hist(
        [[ch.token_count for ch in chunks if ch.element_type is k] for k in kinds],
        bins=bins,
        stacked=True,
        color=[COLOURS[k] for k in kinds],
        label=[str(k) for k in kinds],
        edgecolor=SURFACE,
        linewidth=1,
    )
    for value, name in ((c.target_tokens, "target"), (c.max_tokens, "max")):
        ax.axvline(value, color=TEXT_MUTED, linewidth=1, linestyle="--")
        ax.text(value + 4, ax.get_ylim()[1], name, color=TEXT_MUTED, fontsize=9, va="top")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Chunk length by element type", loc="left", color=TEXT, fontsize=12)
    ax.set_xlabel(f"body tokens ({c.tokenizer_model} tokenizer)", color=TEXT_MUTED, fontsize=9)
    ax.set_ylabel("chunks", color=TEXT_MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "phase2-chunk-tokens.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    per_paper = Counter(ch.record_id for ch in chunks)
    median, most = statistics.median(per_paper.values()), max(per_paper.values())
    lines = [
        "# Phase 2 chunk statistics",
        "",
        f"Chunks: {len(chunks)} from {len(per_paper)} papers "
        f"(median {median:.0f} per paper, max {most}).",
        f"Settings: target {c.target_tokens}, max {c.max_tokens}, "
        f"overlap {c.overlap_tokens} tokens.",
        "",
        "| Element type | Chunks | Median tokens | Max tokens |",
        "|---|---|---|---|",
    ]
    for k in kinds:
        tokens = [ch.token_count for ch in chunks if ch.element_type is k]
        lines.append(f"| {k} | {len(tokens)} | {statistics.median(tokens):.0f} | {max(tokens)} |")
    parsers = Counter(str(ch.parser) for ch in chunks)
    lines += ["", "| Parser | Chunks |", "|---|---|"]
    lines += [f"| {name} | {n} |" for name, n in parsers.most_common()]
    lines += ["", "| Token range | Chunks |", "|---|---|"]
    counts = Counter(min(ch.token_count // 64, c.max_tokens // 64) for ch in chunks)
    lines += [f"| {64 * b}-{64 * b + 63} | {counts[b]} |" for b in sorted(counts)]
    (out / "phase2-chunk-stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(chunks)} chunks; figure and tables in {out}")


if __name__ == "__main__":
    main()
