"""Phase 3 probe: why gold is a span, not a chunk id.

Re-chunks one parsed paper in memory at several chunk sizes and follows a
few of its sentences. For each sentence and size it records which chunks
contain it, according to the same `map_span` the validator and the Phase 4
harness use. Expected:
- the chunk ids holding a sentence change with the chunk size;
- a chunk id recorded as gold at the default size (300) points at text
  without that sentence at other sizes: chunk-id gold silently goes stale;
- the span still maps at every size.

Reads a structured paper from `rag parse` output. Writes the table to
docs/figures/phase3-span-robustness.md (no plot: it's a table result).

Run:  uv run python probes/phase-3-span-robustness.py [--record-id ...] [--sentences 5]
"""

import argparse
from pathlib import Path

from rag.common.config import load_config
from rag.eval.schemas import GoldSpan
from rag.eval.spans import ChunkTexts, map_span
from rag.parse.chunker import Chunker
from rag.parse.pipeline import safe_name, step_dir
from rag.parse.schemas import Chunk, StructuredPaper
from rag.parse.tokens import HFTokenCounter

DEFAULT_RECORD = "acl:P16-1162"  # the BPE paper, CC BY 4.0
TARGETS = [100, 150, 300, 448]  # chunk target_tokens to compare; 300 is the config default
GOLD_TARGET = 300  # the size a chunk-id gold would have been labelled at
MIN_WORDS = 10  # skip short fragments when picking sentences
PREVIEW_CHARS = 60


def pick_sentences(paper: StructuredPaper, n: int) -> list[str]:
    """n sentences spread evenly through the body, so they land in different sections."""
    pool = [
        s.text
        for section in paper.sections
        for p in section.paragraphs
        if p.formula_id is None
        for s in p.sentences
        if len(s.text.split()) >= MIN_WORDS
    ]
    step = max(1, len(pool) // n)
    return pool[step // 2 :: step][:n]


def chunk_at(paper: StructuredPaper, target: int, chunker_args) -> list[Chunk]:
    config, counter = chunker_args
    sized = config.model_copy(update={"target_tokens": target})
    return Chunker(sized, counter).chunk(paper).chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--record-id", default=DEFAULT_RECORD)
    parser.add_argument("--sentences", type=int, default=5)
    args = parser.parse_args()

    config = load_config()
    chunk_config = config.parse.chunk
    paper_path = step_dir(config, "structure") / f"{safe_name(args.record_id)}.json"
    paper = StructuredPaper.model_validate_json(paper_path.read_text(encoding="utf-8"))
    counter = HFTokenCounter(chunk_config.tokenizer_model)

    by_target = {t: chunk_at(paper, t, (chunk_config, counter)) for t in TARGETS}
    texts = {t: ChunkTexts(chunks) for t, chunks in by_target.items()}
    text_of = {t: {c.chunk_id: c.text for c in chunks} for t, chunks in by_target.items()}
    sentences = pick_sentences(paper, args.sentences)

    rows = []
    for sentence in sentences:
        gold = GoldSpan(record_id=paper.record_id, span=sentence)
        mapped = {t: map_span(gold, texts[t]) for t in TARGETS}
        gold_ids = mapped[GOLD_TARGET]
        # Would the chunk ids labelled at size 300 still contain the sentence?
        id_gold_ok = {
            t: any(sentence in text_of[t].get(cid, "") for cid in gold_ids) for t in TARGETS
        }
        rows.append((sentence, mapped, id_gold_ok))

    out = config.paths.figures / "phase3-span-robustness.md"
    write_table(args.record_id, by_target, rows, out)
    print(out.read_text(encoding="utf-8"))


def short_ids(ids: list[str]) -> str:
    return ", ".join("#" + i.rsplit("#", 1)[1] for i in ids) or "none"


def write_table(record_id: str, by_target: dict[int, list[Chunk]], rows, path: Path) -> None:
    head = " | ".join(f"target {t}" for t in TARGETS)
    lines = [
        "# Phase 3 probe: span gold vs chunk-id gold under re-chunking",
        "",
        f"Paper `{record_id}`, re-chunked in memory at `target_tokens` = "
        f"{', '.join(map(str, TARGETS))} (other chunk settings from config.yaml).",
        "Chunks per size: "
        + ", ".join(f"{t}: {len(chunks)}" for t, chunks in by_target.items())
        + ".",
        "",
        "## Chunks containing each sentence (`map_span`)",
        "",
        f"| Sentence | {head} |",
        "|---" * (len(TARGETS) + 1) + "|",
    ]
    for sentence, mapped, _ in rows:
        cells = " | ".join(short_ids(mapped[t]) for t in TARGETS)
        lines.append(f"| {sentence[:PREVIEW_CHARS]}… | {cells} |")
    lines += [
        "",
        f"## Would chunk ids labelled at target {GOLD_TARGET} still be right?",
        "",
        f"| Sentence | {head} |",
        "|---" * (len(TARGETS) + 1) + "|",
    ]
    for sentence, _, ok in rows:
        cells = " | ".join("yes" if ok[t] else "**no**" for t in TARGETS)
        lines.append(f"| {sentence[:PREVIEW_CHARS]}… | {cells} |")
    span_ok = sum(all(mapped[t] for t in TARGETS) for _, mapped, _ in rows)
    id_ok = sum(all(ok.values()) for _, _, ok in rows)
    lines += [
        "",
        f"Span gold maps at every size for {span_ok}/{len(rows)} sentences; "
        f"chunk-id gold stays right at every size for {id_ok}/{len(rows)}.",
        "",
        "Generated by `probes/phase-3-span-robustness.py`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
