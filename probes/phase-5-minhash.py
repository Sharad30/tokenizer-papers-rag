"""Phase 5 probe: how good is the MinHash estimate, and where does LSH draw the line?

Two questions, answered on the real corpus (full-text 5-word shingles of
each indexed paper, the same sets `rag dedup run` uses):

1. MinHash accuracy. For pairs of papers, compare the MinHash estimate of
   Jaccard with the exact value as the number of hash functions grows. The
   error should shrink like sqrt(J(1 - J) / num_perm).
2. The LSH S-curve. With b bands of r rows, a pair with Jaccard s becomes a
   candidate with probability 1 - (1 - s^r)^b. Plotted for the configured b
   and r, with the merge thresholds and the exact Jaccard of two groups of
   real pairs underneath:
     ID-linked        records the identifiers say are one paper (versions,
                      venue copies): what should merge
     same authors     pairs sharing at least half their authors but no
                      identifier: mostly follow-up papers, plus copies the
                      identifiers missed

No labels are used: the thresholds in config.yaml come from where these two
groups separate (D-040).

Writes docs/figures/phase5-minhash.png, phase5-scurve.png and the table view
phase5-minhash.md.

Run:  uv run python probes/phase-5-minhash.py
"""

import random
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt  # noqa: E402

from rag.common.config import load_config  # noqa: E402
from rag.dedup import pipeline  # noqa: E402
from rag.dedup.minhash import (  # noqa: E402
    MinHasher,
    candidate_probability,
    estimate_jaccard,
    exact_jaccard,
    shingle_hashes,
    shingles,
)
from rag.dedup.resolve import author_overlap, surnames  # noqa: E402
from rag.eval.identity import identity_groups  # noqa: E402

NUM_PERMS = [16, 32, 64, 128, 256, 512]
SAMPLE_PAIRS = 200
SEED = 5
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, TEXT, TEXT_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def main() -> None:
    config = load_config()
    cfg = config.dedup
    records = pipeline.indexed_records(config)
    chunks = pipeline.read_chunks(config)
    texts: dict[str, list[str]] = {}
    for chunk in chunks:
        texts.setdefault(chunk.record_id, []).append(chunk.text)
    sets = {
        rid: shingles(" ".join(parts), cfg.minhash.shingle_words) for rid, parts in texts.items()
    }
    records = [r for r in records if r.record_id in sets]

    groups = identity_groups(records)
    authors = {r.record_id: surnames(r.authors) for r in records}
    linked, same_authors = [], []
    for a, b in combinations(sorted(sets), 2):
        if groups[a] == groups[b]:
            linked.append((a, b))
        elif author_overlap(authors[a], authors[b]) >= cfg.author_overlap_min:
            same_authors.append((a, b))
    exact = {p: exact_jaccard(sets[p[0]], sets[p[1]]) for p in linked + same_authors}

    rng = random.Random(SEED)
    sample = rng.sample(linked, SAMPLE_PAIRS // 2) + rng.sample(same_authors, SAMPLE_PAIRS // 2)
    errors = accuracy(sample, sets, exact, cfg.seed)

    out = config.paths.figures
    plot_accuracy(errors, out / "phase5-minhash.png")
    linked_j = [exact[p] for p in linked]
    same_j = [exact[p] for p in same_authors]
    plot_scurve(cfg, linked_j, same_j, out / "phase5-scurve.png")
    write_table(cfg, errors, linked_j, same_j, out / "phase5-minhash.md")
    print((out / "phase5-minhash.md").read_text(encoding="utf-8"))


def accuracy(sample, sets, exact, seed: int) -> dict[int, tuple[float, float]]:
    """num_perm -> (mean absolute error, mean theoretical standard error)."""
    hashes = {rid: shingle_hashes(sets[rid]) for pair in sample for rid in pair}
    out = {}
    for n in NUM_PERMS:
        hasher = MinHasher(n, seed)
        sigs = {rid: hasher.signature(h) for rid, h in hashes.items()}
        errs, theory = [], []
        for a, b in sample:
            j = exact[(a, b)]
            errs.append(abs(estimate_jaccard(sigs[a], sigs[b]) - j))
            theory.append(np.sqrt(j * (1 - j) / n))
        out[n] = (float(np.mean(errs)), float(np.mean(theory)))
    return out


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=TEXT, fontsize=12)
    ax.set_xlabel(xlabel, color=TEXT_MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT_MUTED, fontsize=9)


def plot_accuracy(errors: dict[int, tuple[float, float]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ns = list(errors)
    ax.plot(
        ns,
        [errors[n][0] for n in ns],
        color=BLUE,
        linewidth=2,
        marker="o",
        markersize=8,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
        label="measured mean |estimate - exact|",
    )
    ax.plot(
        ns,
        [errors[n][1] for n in ns],
        color=TEXT_MUTED,
        linewidth=1.5,
        linestyle="--",
        label="theory: standard error sqrt(J(1-J)/num_perm)",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns, [str(n) for n in ns])
    _style(
        ax,
        f"MinHash error on {SAMPLE_PAIRS} real paper pairs",
        "hash functions (num_perm)",
        "mean absolute error in Jaccard",
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_scurve(cfg, linked_j: list[float], same_j: list[float], path: Path) -> None:
    bands, rows = cfg.minhash.bands, cfg.minhash.rows
    fig, (curve, strip) = plt.subplots(
        2,
        1,
        figsize=(8, 5.5),
        facecolor=SURFACE,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.3]},
    )
    for ax in (curve, strip):
        ax.set_facecolor(SURFACE)
    s = np.linspace(0, 1, 201)
    curve.plot(s, [candidate_probability(x, bands, rows) for x in s], color=BLUE, linewidth=2)
    for x, label in (
        (cfg.minhash.min_jaccard, f"text merge {cfg.minhash.min_jaccard}"),
        (cfg.title.review_min_jaccard, f"title+text {cfg.title.review_min_jaccard}"),
    ):
        for ax in (curve, strip):
            ax.axvline(x, color=TEXT_MUTED, linewidth=1, linestyle="--")
        curve.text(x, 1.02, f" {label}", color=TEXT_MUTED, fontsize=9, va="bottom")
    _style(curve, f"LSH candidate probability, {bands} bands x {rows} rows", "", "P(candidate)")
    curve.set_ylim(0, 1.08)

    jitter = np.random.default_rng(0)
    for row, (values, colour, label) in enumerate(
        (
            (linked_j, BLUE, f"ID-linked ({len(linked_j)})"),
            (
                [v for v in same_j if v > 0.01],
                ORANGE,
                f"same authors, J>0.01 ({sum(v > 0.01 for v in same_j)})",
            ),
        )
    ):
        strip.scatter(
            values,
            row + jitter.uniform(-0.2, 0.2, len(values)),
            s=10,
            color=colour,
            alpha=0.6,
            linewidth=0,
        )
        strip.text(1.0, row + 0.3, label, color=TEXT, fontsize=9, ha="right", va="bottom")
    strip.set_yticks([])
    strip.set_ylim(-0.5, 1.8)
    _style(strip, "", "exact Jaccard of 5-word shingles", "")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def write_table(cfg, errors, linked_j: list[float], same_j: list[float], path: Path) -> None:
    bands, rows = cfg.minhash.bands, cfg.minhash.rows
    lines = [
        "# Phase 5 probe: MinHash accuracy and the LSH S-curve",
        "",
        "Exact vs estimated Jaccard of full-text 5-word shingle sets. Figures:",
        "`phase5-minhash.png`, `phase5-scurve.png`.",
        "",
        f"## MinHash error ({SAMPLE_PAIRS} pairs: half ID-linked, half same-author)",
        "",
        "| num_perm | mean abs error | theory |",
        "|---|---|---|",
    ]
    lines += [f"| {n} | {err:.4f} | {th:.4f} |" for n, (err, th) in errors.items()]
    lines += [
        "",
        f"## Candidate probability, {bands} bands x {rows} rows",
        "",
        "| Jaccard | P(candidate) |",
        "|---|---|",
    ]
    lines += [
        f"| {s:.1f} | {candidate_probability(s, bands, rows):.3f} |"
        for s in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8)
    ]
    lines += [
        f"\nThe curve's steep part sits near (1/b)^(1/r) = {(1 / bands) ** (1 / rows):.2f}.",
        "",
        "## Real pairs by exact Jaccard",
        "",
        "| Jaccard at least | ID-linked pairs | same-author pairs, no shared ID |",
        "|---|---|---|",
    ]
    for t in (0.05, 0.1, 0.2, 0.3, 0.5, 0.8):
        lines.append(
            f"| {t} | {sum(v >= t for v in linked_j)} of {len(linked_j)} "
            f"| {sum(v >= t for v in same_j)} of {len(same_j)} |"
        )
    lines += [
        "",
        "- The measured error tracks the theoretical standard error (a mean absolute",
        "  error is about 0.8 of it for a normal estimate) and halves per 4x num_perm.",
        "- ID-linked pairs below 0.2 are heavily revised versions (a kept v1 vs its",
        "  final version): identifiers, not text, have to join those.",
        "- Same-author pairs with no shared ID above 0.5 are copies the identifiers",
        "  missed (an ACL paper and its arXiv preprint). Between 0.2 and 0.5 there",
        "  are revised copies and a few related papers, so text alone doesn't merge",
        "  there: a close title must agree. Below 0.2 they are follow-ups.",
        "",
        "Generated by `probes/phase-5-minhash.py`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
