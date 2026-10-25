"""Phase 2 probe: what happens to a chunk's embedding as it grows past the answer?

Takes one paragraph that answers a question, then embeds it inside more and
more of the paper's surrounding text, at a series of token lengths. Two
variants:

  answer first   the paragraph, then the text after it
  answer last    the text before it, then the paragraph

For each length, plots cosine similarity to the question. Expected:
- similarity falls as unrelated text dilutes the single embedding;
- past the model's limit (512 tokens) "answer first" goes flat: the model
  silently drops everything after token 512, so the extra text changes nothing;
- "answer last" collapses past 512: the answer itself is what gets dropped.

Reads a parsed paper from `rag parse` output. Writes to docs/figures/:
  phase2-dilution.png, phase2-dilution.md (the figure's table view)

Run:  uv run python probes/phase-2-dilution.py [--record-id ...] [--anchor ...] [--question ...]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt  # noqa: E402

from rag.common.config import load_config  # noqa: E402
from rag.parse.pipeline import safe_name, step_dir  # noqa: E402
from rag.parse.schemas import StructuredPaper  # noqa: E402
from rag.parse.tokens import HFTokenCounter  # noqa: E402

# bge-small-en-v1.5's model card: this instruction goes on queries only, never passages.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
LENGTHS = [128, 192, 256, 384, 448, 512, 576, 640, 768, 1024]
# Defaults: the BPE paper (ACL P16-1162, CC BY 4.0) and its algorithm paragraph (3.2).
DEFAULT_RECORD = "acl:P16-1162"
DEFAULT_ANCHOR = "initialize the symbol vocabulary with the character vocabulary"
DEFAULT_QUESTION = "How does BPE build its subword vocabulary by merging characters?"

COLOURS = {"answer first": "#2a78d6", "answer last": "#eb6834"}  # dataviz palette slots 1-2
SURFACE, TEXT, TEXT_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def paragraphs(paper: StructuredPaper) -> list[str]:
    texts = [" ".join(s.text for s in p.sentences) for p in paper.abstract]
    for section in paper.sections:
        texts += [" ".join(s.text for s in p.sentences) for p in section.paragraphs]
    return [t for t in texts if t]


def build_texts(
    paras: list[str], target: int, counter: HFTokenCounter
) -> dict[str, list[tuple[int, str]]]:
    """For each variant, (requested length, text cut to that many tokens)."""
    answer = paras[target]
    after = " ".join(paras[target + 1 :])
    before = " ".join(paras[:target])
    out: dict[str, list[tuple[int, str]]] = {"answer first": [], "answer last": []}
    for length in LENGTHS:
        first = counter.split(f"{answer} {after}", length)[0]
        out["answer first"].append((length, first))
        out["answer last"].append((length, last_tokens(f"{before} {answer}", length, counter)))
    return out


def last_tokens(text: str, length: int, counter: HFTokenCounter) -> str:
    """The last `length` tokens of text, cut from the original string."""
    skip = counter.count(text) - length
    if skip <= 0:
        return text
    head = counter.split(text, skip)[0]  # exactly the first `skip` tokens
    return text[len(head) :].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--record-id", default=DEFAULT_RECORD)
    parser.add_argument(
        "--anchor", default=DEFAULT_ANCHOR, help="phrase in the answering paragraph"
    )
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    config = load_config(args.config)
    model_name = config.parse.chunk.tokenizer_model
    paper_file = step_dir(config, "structure") / f"{safe_name(args.record_id)}.json"
    paper = StructuredPaper.model_validate_json(paper_file.read_text(encoding="utf-8"))
    paras = paragraphs(paper)
    target = next(i for i, p in enumerate(paras) if args.anchor.lower() in p.lower())
    counter = HFTokenCounter(model_name)
    model = SentenceTransformer(model_name)

    question = model.encode(args.question, prompt=QUERY_INSTRUCTION, normalize_embeddings=True)
    alone = model.encode(paras[target], normalize_embeddings=True)
    alone_tokens = counter.count(paras[target])
    results = {}
    for variant, texts in build_texts(paras, target, counter).items():
        vectors = model.encode([t for _, t in texts], normalize_embeddings=True)
        results[variant] = [
            (length, counter.count(text), float(vector @ question))
            for (length, text), vector in zip(texts, vectors, strict=True)
        ]

    out = config.paths.figures
    out.mkdir(parents=True, exist_ok=True)
    plot(
        results,
        alone_tokens,
        float(alone @ question),
        model.max_seq_length,
        out / "phase2-dilution.png",
    )
    write_table(
        results,
        args,
        alone_tokens,
        float(alone @ question),
        model.max_seq_length,
        out / "phase2-dilution.md",
    )
    print(f"Answer paragraph: {alone_tokens} tokens, similarity {float(alone @ question):.3f}")
    print(f"Figure and table in {out}")


def plot(results, alone_tokens: int, alone_sim: float, limit: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for variant, rows in results.items():
        ax.plot(
            [r[1] for r in rows],
            [r[2] for r in rows],
            color=COLOURS[variant],
            linewidth=2,
            marker="o",
            markersize=5,
            label=variant,
        )
    ax.scatter([alone_tokens], [alone_sim], color=TEXT, s=40, zorder=3, label="paragraph alone")
    ax.axvline(limit, color=TEXT_MUTED, linewidth=1, linestyle="--")
    ax.text(
        limit + 8,
        ax.get_ylim()[1],
        f"model limit: {limit} tokens",
        color=TEXT_MUTED,
        fontsize=9,
        va="top",
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(
        "Similarity to the question as the chunk grows", loc="left", color=TEXT, fontsize=12
    )
    ax.set_xlabel(
        "chunk length (tokens, before the model's truncation)", color=TEXT_MUTED, fontsize=9
    )
    ax.set_ylabel("cosine similarity", color=TEXT_MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def write_table(results, args, alone_tokens: int, alone_sim: float, limit: int, path: Path) -> None:
    lines = [
        "# Phase 2 probe: dilution and truncation",
        "",
        f'Paper: `{args.record_id}` · anchor: "{args.anchor}" · model limit: {limit} tokens',
        "",
        f"Question: {args.question}",
        "",
        f"Answer paragraph alone: {alone_tokens} tokens, similarity {alone_sim:.3f}",
        "",
        "| Tokens | Answer first | Answer last |",
        "|---|---|---|",
    ]
    first, last = results["answer first"], results["answer last"]
    for (_, n, s1), (_, _, s2) in zip(first, last, strict=True):
        lines.append(f"| {n} | {s1:.3f} | {s2:.3f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
