"""Phase 1 probe: what does each discovery route contribute, and where is coverage thin?

Reads the corpus manifest and writes to docs/figures/:
  phase1-accepted-by-year.png   accepted papers per year, stacked by record source
  phase1-discovery-routes.png   which combination of routes found each accepted paper
  phase1-coverage.md            the same numbers as tables (the figures' table view)

Counts papers *as decided*: linked copies and kept v1s (decided_by=inherited)
are left out, so one paper isn't counted once per manifestation.

Run:  uv run python probes/phase-1-corpus-coverage.py [--config config.yaml]
"""

import argparse
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt  # noqa: E402

from rag.collect.manifest import read_manifest  # noqa: E402
from rag.collect.schemas import DecidedBy, Decision, ManifestRecord, Source  # noqa: E402
from rag.common.config import load_config  # noqa: E402

# Reference palette (dataviz skill), first three categorical slots: validated
# as a set for colour-blind separation. Aqua is below 3:1 contrast on the
# surface, so the figures carry a legend and the .md file is the table view.
SOURCE_COLOURS = {Source.ARXIV: "#2a78d6", Source.ACL: "#eb6834", Source.S2: "#1baf7a"}
SOURCE_LABELS = {
    Source.ARXIV: "arXiv",
    Source.ACL: "ACL Anthology",
    Source.S2: "Semantic Scholar only",
}
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
ROUTE_NAMES = {
    "arxiv_query": "arXiv query",
    "snowball_ref": "snowball",
    "snowball_cit": "snowball",
    "acl_scan": "ACL scan",
    "anchor": "anchor",
}


def decided_accepted(records: list[ManifestRecord]) -> list[ManifestRecord]:
    return [
        r
        for r in records
        if r.relevance is Decision.ACCEPT and r.decided_by is not DecidedBy.INHERITED
    ]


def route_combo(record: ManifestRecord) -> str:
    routes = {ROUTE_NAMES.get(v.split(":", 1)[0], v.split(":", 1)[0]) for v in record.found_via}
    return " + ".join(sorted(routes))


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)


def plot_by_year(records: list[ManifestRecord], path: Path) -> dict[int, Counter]:
    by_year: dict[int, Counter] = {}
    for r in records:
        if r.year is not None:
            by_year.setdefault(r.year, Counter())[r.source] += 1
    years = sorted(by_year)
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=SURFACE)
    bottom = [0] * len(years)
    for source in (Source.ARXIV, Source.ACL, Source.S2):
        heights = [by_year[y][source] for y in years]
        # A 2px surface-coloured edge separates stacked segments and neighbours.
        ax.bar(
            years,
            heights,
            bottom=bottom,
            width=0.8,
            color=SOURCE_COLOURS[source],
            edgecolor=SURFACE,
            linewidth=1.5,
            label=SOURCE_LABELS[source],
        )
        bottom = [b + h for b, h in zip(bottom, heights, strict=True)]
    _style(ax)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Accepted papers per year, by record source", loc="left", color=TEXT, fontsize=12)
    ax.set_ylabel("papers", color=TEXT_MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return by_year


def plot_routes(records: list[ManifestRecord], path: Path) -> Counter:
    combos = Counter(route_combo(r) for r in records)
    ordered = combos.most_common()
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(ordered) + 1.2), facecolor=SURFACE)
    labels = [name for name, _ in ordered][::-1]
    values = [count for _, count in ordered][::-1]
    bars = ax.barh(
        labels,
        values,
        height=0.7,
        color=SOURCE_COLOURS[Source.ARXIV],
        edgecolor=SURFACE,
        linewidth=1.5,
    )
    # One series: no legend; values labelled at the bar ends in text ink.
    ax.bar_label(bars, padding=3, color=TEXT, fontsize=9)
    _style(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(
        "Which discovery routes found each accepted paper", loc="left", color=TEXT, fontsize=12
    )
    ax.set_xlabel("accepted papers", color=TEXT_MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return combos


def write_tables(by_year: dict[int, Counter], combos: Counter, total: int, path: Path) -> None:
    lines = [
        "# Phase 1 corpus coverage",
        "",
        f"Accepted papers (decided directly, excluding inherited copies): {total}",
        "",
        "## Accepted papers per year, by record source",
        "",
        "| Year | arXiv | ACL | S2 only | Total |",
        "|---|---|---|---|---|",
    ]
    for year in sorted(by_year):
        c = by_year[year]
        lines.append(
            f"| {year} | {c[Source.ARXIV]} | {c[Source.ACL]} | {c[Source.S2]} | {sum(c.values())} |"
        )
    lines += ["", "## Discovery-route combinations", "", "| Routes | Papers |", "|---|---|"]
    lines += [f"| {name} | {count} |" for name, count in combos.most_common()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    accepted = decided_accepted(read_manifest(config.paths.manifest))
    out = config.paths.figures
    out.mkdir(parents=True, exist_ok=True)
    by_year = plot_by_year(accepted, out / "phase1-accepted-by-year.png")
    combos = plot_routes(accepted, out / "phase1-discovery-routes.png")
    write_tables(by_year, combos, len(accepted), out / "phase1-coverage.md")
    print(f"{len(accepted)} accepted papers; figures and tables in {out}")


if __name__ == "__main__":
    main()
