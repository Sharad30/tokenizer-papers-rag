"""Phase 4 probe: BM25 by hand, its two knobs, and what RRF's k does.

1. BM25 by hand. One query, three chunks of the BPE paper. Every
   intermediate number of the Lucene BM25 formula is printed:
       idf(t)   = ln(1 + (N - df + 0.5) / (df + 0.5))
       tf part  = tf / (tf + k1 * (1 - b + b * dl / avgdl))
       score    = sum over query terms of idf * tf part
   and the total is checked against bm25s (the library the index uses).
2. The knobs. k1 (how fast repeated terms stop adding score) and b (how much
   long chunks are penalised) are swept on the three chunks, and on a
   synthetic pair (long chunk repeating a term vs short chunk with it once)
   where they flip the winner.
3. RRF's k. Two toy rankings fused with k = 1 and k = 60: with k = 1 a
   single first place counts as much as being third in both lists; with
   k = 60 agreement wins.

Writes docs/figures/phase4-bm25-rrf.md.

Run:  uv run python probes/phase-4-bm25-rrf.py [--record-id ...] [--query ...]
"""

import argparse
import math
from collections import Counter

from rag.common.config import load_config
from rag.common.io import read_jsonl
from rag.index.config import Bm25Config
from rag.parse.pipeline import CHUNKS, step_dir
from rag.parse.schemas import Chunk
from rag.retrieve.bm25 import Bm25Index
from rag.retrieve.fusion import rrf
from rag.retrieve.text import STOPWORDS, chunk_text, terms

DEFAULT_RECORD = "acl:P16-1162"  # the BPE paper, CC BY 4.0
DEFAULT_QUERY = "byte pair encoding merges frequent symbol pairs"
# Three chunks with different amounts of the query: the BPE algorithm section,
# the introduction's contribution list, and a results paragraph.
DEFAULT_CHUNKS = ["#0011", "#0003", "#0020"]
SWEEP = [(0.5, 0.75), (1.2, 0.75), (1.5, 0.75), (2.0, 0.75), (1.5, 0.0), (1.5, 1.0)]
# Synthetic pair where the knobs matter: a long chunk repeating the term vs a
# short chunk mentioning it once (filler words stand in for other content).
SYNTHETIC = {
    "long, term 4 times": ["unigram"] * 4 + ["filler"] * 100,
    "short, term once": ["unigram"] + ["filler"] * 5,
}
SYNTHETIC_SWEEP = [(0.5, 0.75), (1.5, 0.0), (1.5, 0.5), (1.5, 0.75), (1.5, 1.0), (3.0, 0.0)]
TOY = {"bm25": ["A", "B", "C", "D"], "dense": ["E", "F", "C", "G"]}


def by_hand(query: list[str], docs: list[list[str]], k1: float, b: float):
    """Per document: score and the per-term breakdown (term, tf, df, idf, tf part)."""
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / n
    df = {t: sum(t in d for d in docs) for t in set(query)}
    out = []
    for d in docs:
        counts = Counter(d)
        rows, total = [], 0.0
        for t in dict.fromkeys(query):  # each distinct term once, like bm25s
            if df[t] == 0:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            tf_part = counts[t] / (counts[t] + k1 * (1 - b + b * len(d) / avgdl))
            rows.append((t, counts[t], df[t], idf, tf_part))
            total += idf * tf_part
        out.append((total, rows, len(d)))
    return out, avgdl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record-id", default=DEFAULT_RECORD)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--chunks", nargs=3, default=DEFAULT_CHUNKS)
    args = parser.parse_args()

    config = load_config()
    base: Bm25Config = config.index.bm25
    stop = STOPWORDS[base.stopwords]
    wanted = [args.record_id + suffix for suffix in args.chunks]
    found = {
        c.chunk_id: c
        for c in read_jsonl(step_dir(config, "chunk") / CHUNKS, Chunk)
        if c.chunk_id in wanted
    }
    chunks = [found[i] for i in wanted]
    docs = [terms(chunk_text(c, base.include_header), stop, base.split_compounds) for c in chunks]
    query = terms(args.query, stop, base.split_compounds)

    lines = [
        "# Phase 4 probe: BM25 by hand and RRF's k",
        "",
        f'Query: "{args.query}" → terms {query}',
        "",
        "Chunks (a 3-chunk collection, so N = 3 and every idf is visible):",
        "",
    ]
    for c, d in zip(chunks, docs, strict=True):
        lines.append(
            f'- `{c.chunk_id}` ({" > ".join(c.section_path)}), {len(d)} terms: "{c.text[:90]}…"'
        )

    results, avgdl = by_hand(query, docs, base.k1, base.b)
    library = Bm25Index.build(chunks, base).scores(args.query)
    lines += ["", f"## 1. By hand (k1 = {base.k1}, b = {base.b}, avgdl = {avgdl:.1f})", ""]
    for c, (total, rows, dl), lib in zip(chunks, results, library, strict=True):
        lines += [
            f"**{c.chunk_id}** (dl = {dl})",
            "",
            "| term | tf | df | idf | tf part | idf × tf part |",
            "|---|---|---|---|---|---|",
        ]
        for t, tf, df, idf, part in rows:
            lines.append(f"| {t} | {tf} | {df} | {idf:.4f} | {part:.4f} | {idf * part:.4f} |")
        match = "matches" if math.isclose(total, float(lib), rel_tol=1e-5) else "DIFFERS from"
        lines += [
            "",
            f"Total by hand {total:.4f}; bm25s {float(lib):.4f}: {match} the library.",
            "",
        ]

    missing = [t for t in dict.fromkeys(query) if not any(t in d for d in docs)]
    if missing:
        lines += [
            f"Query terms in none of the chunks (no stemming, D-027): {missing}. The paper",
            'says "merge" and "merge operations", so "merges" scores nothing here.',
            "",
        ]

    lines += [
        "## 2. Sweeping k1 and b",
        "",
        "### On the three chunks",
        "",
        "| k1 | b | "
        + " | ".join(f"`{c.chunk_id.split('#')[1]}`" for c in chunks)
        + " | ranking |",
        "|---" * (len(chunks) + 3) + "|",
    ]
    rankings = set()
    for k1, b in SWEEP:
        scores, _ = by_hand(query, docs, k1, b)
        totals = [s for s, _, _ in scores]
        order = sorted(range(len(chunks)), key=lambda i: -totals[i])
        ranking = " > ".join(chunks[i].chunk_id.split("#")[1] for i in order)
        rankings.add(ranking)
        lines.append(
            f"| {k1} | {b} | " + " | ".join(f"{s:.3f}" for s in totals) + f" | {ranking} |"
        )
    lines.append("")
    if len(rankings) == 1:
        lines += [
            "The ranking never changes: one chunk matches far more query terms than the",
            "others, and k1 and b only rescale how much each match is worth. The knobs",
            "matter when chunks are close, as in the synthetic pair below.",
        ]
    else:
        lines.append(f"The ranking changes across settings: {sorted(rankings)}.")

    names = list(SYNTHETIC)
    syn_docs = list(SYNTHETIC.values())
    lines += [
        "",
        "### A long chunk repeating the term vs a short chunk mentioning it once",
        "",
        'Query: "unigram". Both chunks contain it, so idf is the same for both.',
        "",
        f"| k1 | b | {names[0]} | {names[1]} | wins |",
        "|---|---|---|---|---|",
    ]
    for k1, b in SYNTHETIC_SWEEP:
        scores, _ = by_hand(["unigram"], syn_docs, k1, b)
        a, s2 = scores[0][0], scores[1][0]
        lines.append(f"| {k1} | {b} | {a:.3f} | {s2:.3f} | {names[0] if a > s2 else names[1]} |")
    lines += [
        "",
        "- b sets how much a chunk's length counts against it: at b = 0 the four",
        "  repetitions win on raw counts; as b rises, the long chunk is discounted.",
        "- k1 sets how quickly repetitions saturate: small k1 makes the 4th mention",
        "  worth almost nothing beyond the 1st.",
        "",
        "## 3. RRF's k on a toy example",
        "",
        f"BM25 ranks {TOY['bm25']}, dense ranks {TOY['dense']}. C is 3rd in both; A and E",
        "are each 1st in one list only.",
        "",
        "| k | top 4 after fusion (score) |",
        "|---|---|",
    ]
    ids = sorted(set(TOY["bm25"]) | set(TOY["dense"]))
    as_ints = {name: [ids.index(x) for x in ranked] for name, ranked in TOY.items()}
    for k in (1, 60):
        fused = rrf(as_ints, k)[:4]
        lines.append(f"| {k} | " + ", ".join(f"{ids[i]} ({s:.4f})" for i, s in fused) + " |")
    lines += [
        "",
        "With k = 1, A's and E's single first places (1/2) tie C's two third places",
        "(1/4 + 1/4); ties go to the better best rank, so A and E come first. With k = 60",
        "the curve is flat: C's 2/63 beats A's 1/61, so agreement between retrievers wins.",
        "",
        "Generated by `probes/phase-4-bm25-rrf.py`.",
    ]
    out = config.paths.figures / "phase4-bm25-rrf.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
