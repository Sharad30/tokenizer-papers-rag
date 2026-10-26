"""Phase 7 probe: where should the retrieval gate's threshold sit?

The chat's first abstention layer (D-036) refuses to call the LLM when the
best dense cosine anywhere in the corpus is below a threshold. This probe
measures that best cosine for three groups of queries written for the probe
(NOT the eval set's gold questions, so the threshold isn't tuned on them):

  in-domain       answerable from tokenizer papers
  near-domain     on topic, but not answerable (made-up models, false premises)
  off-topic       nothing to do with tokenization

Expected: off-topic queries score clearly lower than in-domain ones, so a
threshold between them stops them for free. Near-domain queries score like
in-domain ones: words like "tokenizer" and "BPE" match real passages. The
gate can't catch them; the model's NOT_SUPPORTED has to. That's why there
are two layers.

Writes docs/figures/phase7-gate.png and its table view phase7-gate.md.

Run:  uv run python probes/phase-7-gate.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt  # noqa: E402

from rag.common.config import load_config  # noqa: E402
from rag.retrieve.retriever import Retriever  # noqa: E402

GROUPS: dict[str, list[str]] = {
    "in-domain": [
        "How does BPE-dropout regularize subword segmentation?",
        "What is the fertility of a tokenizer and why does it matter for multilingual models?",
        "How does the Unigram language model tokenizer choose a segmentation?",
        "What are the advantages of byte-level models like ByT5?",
        "How does SentencePiece treat whitespace?",
        "How does CANINE avoid an explicit tokenizer?",
        "Do tokenizers treat languages unfairly in terms of token counts?",
        "How does vocabulary size affect machine translation quality?",
        "What is WordPiece and how does it differ from BPE?",
        "How are numbers tokenized, and does it affect arithmetic in LLMs?",
        "What is tokenizer-free language modeling?",
        "How do subword units help translate rare words?",
    ],
    "near-domain": [
        "What vocabulary size does GPT-7 use?",
        "How many merge operations did the original BPE paper use for Klingon?",
        "What tokenizer does the Voyager probe's onboard language model use?",
        "Which tokenizer won the 2031 tokenization shared task?",
        "What BLEU score does ByT5 reach on Elvish-to-English translation?",
        "How does quantum tokenization reduce fertility to zero?",
        "Why did SentencePiece remove support for Unigram in version 9?",
        "What is the tokenizer of the Llama 9 model?",
        "How much faster is BPE than WordPiece on a 2040 smartphone?",
    ],
    "off-topic": [
        "How do I bake sourdough bread?",
        "What is the capital of Australia?",
        "Who won the 1998 football World Cup?",
        "How do I change a car tyre?",
        "What is the boiling point of water at high altitude?",
        "Recommend a good hiking trail in the Alps.",
        "How do vaccines train the immune system?",
        "What is the plot of Hamlet?",
        "How do I file my taxes?",
    ],
}
COLOURS = {"in-domain": "#2a78d6", "near-domain": "#eb6834", "off-topic": "#1baf7a"}
SURFACE, TEXT, TEXT_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def main() -> None:
    config = load_config()
    retriever = Retriever.load(config, {"dense"})
    scores = {
        group: [(q, retriever.search(q, k=1, mode="dense").top_scores["dense"]) for q in queries]
        for group, queries in GROUPS.items()
    }
    in_low = min(s for _, s in scores["in-domain"])
    off_high = max(s for _, s in scores["off-topic"])
    # Midway between the groups when they separate; otherwise there's no clean gate.
    suggested = round((in_low + off_high) / 2, 2) if off_high < in_low else None
    out = config.paths.figures
    plot(scores, suggested, config.chat.abstain.min_dense_cosine, out / "phase7-gate.png")
    configured = config.chat.abstain.min_dense_cosine
    write_table(scores, in_low, off_high, suggested, configured, out / "phase7-gate.md")
    print((out / "phase7-gate.md").read_text(encoding="utf-8"))


def plot(scores, midpoint: float | None, configured: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for row, (group, rows) in enumerate(scores.items()):
        values = [s for _, s in rows]
        ax.scatter(
            values,
            [row] * len(values),
            s=64,
            color=COLOURS[group],
            label=group,
            edgecolor=SURFACE,
            linewidth=2,
            zorder=3,
        )
    lines = [(configured, "-", f"gate in config {configured:.2f}", "right")]
    if midpoint is not None:
        lines.append((midpoint, "--", f"midpoint {midpoint:.2f}", "left"))
    for x, style, label, align in lines:
        ax.axvline(x, color=TEXT_MUTED, linewidth=1, linestyle=style)
        pad = " " if align == "left" else ""
        ax.text(
            x,
            len(scores) - 0.45,
            f"{pad}{label}{' ' if align == 'right' else ''}",
            color=TEXT_MUTED,
            fontsize=9,
            va="bottom",
            ha=align,
        )
    ax.set_yticks(range(len(scores)), list(scores))
    ax.set_ylim(-0.6, len(scores) - 0.2)
    ax.invert_yaxis()
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9, left=False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(
        "Best dense cosine in the corpus, per probe query", loc="left", color=TEXT, fontsize=12
    )
    ax.set_xlabel("bge-small cosine of the closest chunk", color=TEXT_MUTED, fontsize=9)
    ax.legend(
        frameon=False,
        fontsize=9,
        labelcolor=TEXT,
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def write_table(
    scores, in_low: float, off_high: float, suggested, configured: float, path: Path
) -> None:
    lines = [
        "# Phase 7 probe: the retrieval gate's threshold",
        "",
        "Best bge-small cosine anywhere in the corpus for 30 probe queries written for this",
        "probe (not eval gold). Figure: `phase7-gate.png`.",
        "",
        "| Group | Query | Best cosine |",
        "|---|---|---|",
    ]
    for group, rows in scores.items():
        for q, s in sorted(rows, key=lambda r: -r[1]):
            lines.append(f"| {group} | {q} | {s:.3f} |")
    lines += [
        "",
        f"- Lowest in-domain: {in_low:.3f}. Highest off-topic: {off_high:.3f}.",
    ]
    if suggested is None:
        lines.append("- The groups overlap: no threshold separates them cleanly.")
    else:
        lines.append(f"- They separate; the midpoint is {suggested:.2f}.")
    near = [s for _, s in scores["near-domain"]]
    lines.append(
        f"- Near-domain queries range {min(near):.3f}–{max(near):.3f}, mostly inside the"
        ' in-domain range: words like "tokenizer" and "BPE" match real passages.'
    )
    if suggested is not None:
        below = sum(s < suggested for s in near)
        lines.append(
            f"  {below} of {len(near)} fall below {suggested:.2f}; the gate can't stop the rest,"
            " so the model's NOT_SUPPORTED has to."
        )
    lines += [
        f"- Gate in config.yaml: {configured:.2f} (see the comment there for why).",
        "",
        "Generated by `probes/phase-7-gate.py`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
