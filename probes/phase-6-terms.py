"""Phase 6 probe: do embedding models know tokenizer terms, and does keyword search still help?

Three questions, on the real corpus:

1. Term similarity. Cosine between pairs of terms per model: a name and its
   spelled-out form ("SentencePiece" / "sentence piece"), two different
   methods ("SentencePiece" / "BPE"), a term and its everyday sense
   ("fertility" / "birth rate"). A model that knows the field puts the
   first kind close and the others apart.
2. Exact-term queries. For queries naming a specific method or model, the
   share of each retriever's top 5 chunks that contain that exact name. BM25
   matches strings, so it should find them; dense models may drift to
   chunks about the general topic. This is the evidence for or against
   keeping keyword search next to vector search.
3. SPECTER2's ranking. Its model card ranks by L2 distance on raw vectors;
   the index normalises them so scoring is a dot product (D-028). On 2,000
   sampled chunks, how much do the top 10s of the two rankings overlap?

Not gold: terms and queries were written for this probe.
Writes docs/figures/phase6-terms.png and the table view phase6-terms.md.

Run:  uv run python probes/phase-6-terms.py   (needs every index in index.dense_models)
"""

import random
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt  # noqa: E402

from rag.common.config import load_config  # noqa: E402
from rag.retrieve.dense import Specter2Encoder, make_encoder  # noqa: E402
from rag.retrieve.retriever import Retriever  # noqa: E402
from rag.retrieve.text import chunk_text  # noqa: E402

PAIRS: list[tuple[str, str, str]] = [
    # (kind, a, b)
    ("same thing", "SentencePiece", "sentence piece"),
    ("same thing", "BPE", "byte pair encoding"),
    ("same thing", "ByT5", "byte-level T5"),
    ("same thing", "fertility", "tokens per word"),
    ("different methods", "SentencePiece", "BPE"),
    ("different methods", "WordPiece", "Unigram language model"),
    ("different methods", "BPE-dropout", "MaxMatch"),
    ("everyday sense", "fertility", "birth rate"),
    ("everyday sense", "tokenizer", "lexical analyzer"),
    ("everyday sense", "merges", "company mergers"),
]
EXACT: list[tuple[str, str]] = [
    # (query, exact term that should appear in the chunk)
    ("How does SentencePiece handle whitespace?", "SentencePiece"),
    ("What is BPE-dropout?", "BPE-dropout"),
    ("How does CANINE avoid a tokenizer?", "CANINE"),
    ("What does MorphPiece change about BPE?", "MorphPiece"),
    ("How fast is LinMaxMatch for WordPiece?", "MaxMatch"),
    ("What is the fertility of tokenizers on Hindi?", "Hindi"),
    ("How does ByT5 compare with mT5?", "ByT5"),
    ("What are glitch tokens such as SolidGoldMagikarp?", "SolidGoldMagikarp"),
    ("How does TokAlign map vocabularies between models?", "TokAlign"),
    ("What is Parity-aware BPE?", "Parity"),
]
SAMPLE_CHUNKS = 2000
SEED = 7
COLOURS = {"bge_small": "#2a78d6", "bge_base": "#1baf7a", "specter2": "#eb6834", "bm25": "#8a63d2"}
SURFACE, TEXT, TEXT_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def main() -> None:
    config = load_config()
    models = list(config.index.dense_models)
    similarity = {m: term_similarity(make_encoder(config.index.dense_models[m])) for m in models}
    exact = exact_term_rates(config, models)
    overlap = specter2_l2_vs_cosine(config) if "specter2" in models else None
    out = config.paths.figures
    plot(similarity, exact, out / "phase6-terms.png")
    write_table(similarity, exact, overlap, out / "phase6-terms.md")
    print((out / "phase6-terms.md").read_text(encoding="utf-8"))


def term_similarity(encoder) -> list[float]:
    """Cosine per pair, both sides encoded as documents (symmetric)."""
    terms = sorted({t for _, a, b in PAIRS for t in (a, b)})
    vectors = dict(zip(terms, encoder.encode_documents(terms), strict=True))
    return [float(vectors[a] @ vectors[b]) for _, a, b in PAIRS]


def exact_term_rates(config, models: list[str]) -> dict[str, list[float]]:
    """retriever -> share of the top 5 containing the query's exact term, per query."""
    out: dict[str, list[float]] = {}
    for model in models:
        retriever = Retriever.load(config, {"hybrid"}, dense=model)
        if isinstance(retriever.encoder, Specter2Encoder):
            retriever.encoder.encode_queries([q for q, _ in EXACT])
        if "bm25" not in out:
            out["bm25"] = [_rate(retriever, q, t, "bm25") for q, t in EXACT]
        out[model] = [_rate(retriever, q, t, "dense") for q, t in EXACT]
    return out


def _rate(retriever, query: str, term: str, mode: str) -> float:
    hits = retriever.retrieve(query, k=5, mode=mode)
    found = [term.lower() in chunk_text(h.chunk, True).lower() for h in hits]
    return sum(found) / len(found) if found else 0.0


def specter2_l2_vs_cosine(config) -> float:
    """Mean overlap of the top 10 by L2 on raw vectors and by cosine, over the EXACT queries."""
    from rag.dedup.pipeline import read_chunks

    settings = config.index.dense_models["specter2"]
    chunks = random.Random(SEED).sample(read_chunks(config), SAMPLE_CHUNKS)
    texts = [chunk_text(c, True, settings.header_separator) for c in chunks]
    raw = {}

    def keep_raw(packages, args):  # run the worker, keep its unnormalised output
        from rag.retrieve.dense import _run_isolated

        _run_isolated(packages, args)
        opts = dict(zip(args[::2], args[1::2], strict=False))
        raw[opts["--adapter"]] = np.load(opts["--output"])

    encoder = Specter2Encoder(settings, run=keep_raw)
    encoder.encode_documents(texts)
    docs = raw[settings.document_adapter]
    encoder.encode_queries([q for q, _ in EXACT])
    queries = raw[settings.query_adapter]
    overlaps = []
    unit = docs / np.linalg.norm(docs, axis=1, keepdims=True)
    for q in queries:
        by_l2 = set(np.argsort(np.linalg.norm(docs - q, axis=1))[:10])
        by_cos = set(np.argsort(-(unit @ (q / np.linalg.norm(q))))[:10])
        overlaps.append(len(by_l2 & by_cos) / 10)
    return float(np.mean(overlaps))


def plot(similarity: dict[str, list[float]], exact: dict[str, list[float]], path: Path) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE)
    labels = [f"{a} / {b}" for _, a, b in PAIRS]
    y = np.arange(len(PAIRS))
    width = 0.8 / len(similarity)
    for n, (model, values) in enumerate(similarity.items()):
        left.barh(
            y + n * width, values, height=width - 0.02, color=COLOURS.get(model, GRID), label=model
        )
    left.set_yticks(y + width * (len(similarity) - 1) / 2, labels, fontsize=8)
    left.invert_yaxis()
    left.set_title("Cosine between term pairs", loc="left", color=TEXT, fontsize=11)
    names = list(exact)
    right.bar(
        names, [np.mean(exact[n]) for n in names], color=[COLOURS.get(n, GRID) for n in names]
    )
    right.set_ylim(0, 1)
    right.set_title("Top 5 containing the query's exact term", loc="left", color=TEXT, fontsize=11)
    for ax in (left, right):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=TEXT_MUTED, labelsize=8)
        ax.grid(True, axis="x" if ax is left else "y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    left.legend(
        frameon=False,
        fontsize=8,
        labelcolor=TEXT,
        ncol=len(similarity),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def write_table(similarity, exact, overlap: float | None, path: Path) -> None:
    models = list(similarity)
    lines = [
        "# Phase 6 probe: tokenizer terms across embedding models",
        "",
        "Terms and queries written for this probe (not gold). Figure: `phase6-terms.png`.",
        "",
        "## Cosine between term pairs (both encoded as documents)",
        "",
        "| Kind | Pair | " + " | ".join(models) + " |",
        "|---|---|" + "---|" * len(models),
    ]
    for i, (kind, a, b) in enumerate(PAIRS):
        cells = " | ".join(f"{similarity[m][i]:.3f}" for m in models)
        lines.append(f"| {kind} | {a} / {b} | {cells} |")
    names = list(exact)
    lines += [
        "",
        "Raw cosines aren't comparable across models (each has its own range); compare the",
        "order within a column: are 'same thing' pairs above 'different' and 'everyday' ones?",
        "",
        "## Share of the top 5 containing the query's exact term",
        "",
        "| Query | Term | " + " | ".join(names) + " |",
        "|---|---|" + "---|" * len(names),
    ]
    for i, (query, term) in enumerate(EXACT):
        cells = " | ".join(f"{exact[n][i]:.1f}" for n in names)
        lines.append(f"| {query} | {term} | {cells} |")
    lines.append(
        "| **mean** | | " + " | ".join(f"**{np.mean(exact[n]):.2f}**" for n in names) + " |"
    )
    if overlap is not None:
        lines += [
            "",
            "## SPECTER2: L2 on raw vectors vs cosine on normalised ones",
            "",
            f"Mean top-10 overlap over the {len(EXACT)} queries on {SAMPLE_CHUNKS} sampled "
            f"chunks: {overlap:.2f} (1.0 = identical top 10s).",
        ]
    lines += ["", "Generated by `probes/phase-6-terms.py`."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
