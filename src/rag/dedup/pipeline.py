"""`rag dedup run`: manifest + chunks -> works in data/works/.

works.jsonl        one Work per paper: members, canonical record, merge evidence
record_work.json   record_id -> work_id (what the Retriever loads)
review.md          every merge made by fuzzy evidence, and every refused one
meta.json          the chunks and manifest it was built from
stats.json
"""

import json
import logging
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from rag.collect.manifest import MANIFEST_JSON, read_manifest
from rag.collect.schemas import ManifestRecord
from rag.common.config import Config
from rag.common.io import file_sha256, read_jsonl, write_json, write_jsonl, write_text_atomic
from rag.common.provenance import git_commit
from rag.common.stats import stage_stats
from rag.dedup.config import DedupConfig
from rag.dedup.minhash import MinHasher, shingle_hashes, shingles
from rag.dedup.resolve import Resolution, resolve
from rag.dedup.schemas import Restatement, Work, WorksMeta
from rag.dedup.surveys import ReferenceResolver, find_restatements, find_surveys
from rag.parse.pipeline import CHUNKS, REFERENCES, step_dir
from rag.parse.schemas import Chunk, Reference
from rag.retrieve.dense import DenseIndex
from rag.retrieve.retriever import IndexUnavailable, dense_dir, read_meta

log = logging.getLogger(__name__)

WORKS = "works.jsonl"
RECORD_WORK = "record_work.json"
SURVEYS = "surveys.jsonl"
RESTATEMENTS = "restatements.jsonl"
REVIEW = "review.md"
META = "meta.json"


def indexed_records(config: Config) -> list[ManifestRecord]:
    return [r for r in read_manifest(config.paths.manifest) if r.included_in_index]


def read_chunks(config: Config, limit: int | None = None) -> list[Chunk]:
    """All chunks, or those of the first `limit` papers (the same rule as `rag index`)."""
    chunks = read_jsonl(step_dir(config, "chunk") / CHUNKS, Chunk)
    if limit is not None:
        keep = set(list(dict.fromkeys(c.record_id for c in chunks))[:limit])
        chunks = [c for c in chunks if c.record_id in keep]
    return chunks


def signatures(chunks: Sequence[Chunk], config: DedupConfig) -> dict[str, np.ndarray]:
    """record_id -> MinHash signature of its full text (chunk bodies, no headers)."""
    texts: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        texts[chunk.record_id].append(chunk.text)
    hasher = MinHasher(config.minhash.num_perm, config.seed)
    return {
        rid: hasher.signature(
            shingle_hashes(shingles(" ".join(parts), config.minhash.shingle_words))
        )
        for rid, parts in texts.items()
    }


def run(config: Config, limit: int | None) -> Resolution:
    out = config.paths.works
    out.mkdir(parents=True, exist_ok=True)
    with stage_stats(config, "dedup.resolve", out, limit) as stats:
        chunks = read_chunks(config, limit)
        present = {c.record_id for c in chunks}
        records = [r for r in indexed_records(config) if r.record_id in present]
        log.info("MinHash signatures for %d records", len(records))
        sigs = signatures(chunks, config.dedup)
        n_chunks = Counter(c.record_id for c in chunks)
        resolution = resolve(records, sigs, n_chunks, config.dedup)

        write_jsonl(out / WORKS, resolution.works)
        write_json(out / RECORD_WORK, resolution.record_work)
        write_text_atomic(out / REVIEW, review_markdown(resolution.works, records))
        write_json(
            out / META,
            WorksMeta(
                chunks_sha256=file_sha256(step_dir(config, "chunk") / CHUNKS),
                manifest_sha256=file_sha256(config.paths.manifest / MANIFEST_JSON),
                steps=config.dedup.steps,
                n_records=len(records),
                n_works=len(resolution.works),
                limit=limit,
                built_at=datetime.now(UTC),
                git_commit=git_commit(),
            ),
        )
        stats["counts"].update(resolution.counts)
        stats["counts"].update(_surveys(config, records, chunks, resolution.record_work, out))
        sizes = Counter(len(w.members) for w in resolution.works)
        stats["counts"].update({f"works.size_{n}": c for n, c in sorted(sizes.items())})
    log.info(
        "%d records -> %d works (%d with several records)",
        len(records),
        len(resolution.works),
        resolution.counts["works.multi_member"],
    )
    return resolution


def _surveys(
    config: Config,
    records: Sequence[ManifestRecord],
    chunks: Sequence[Chunk],
    record_work: dict[str, str],
    out: Path,
) -> dict[str, int]:
    cfg = config.dedup.surveys
    surveys = find_surveys(records, cfg)
    write_jsonl(out / SURVEYS, surveys)
    restatements: list[Restatement] = []
    try:
        # Needs the dense index over these exact chunks; it has the vectors.
        meta = read_meta(dense_dir(config))
        if meta.chunks_sha256 != file_sha256(step_dir(config, "chunk") / CHUNKS):
            raise IndexUnavailable("dense index is stale")
        dense = DenseIndex.load(dense_dir(config))
    except IndexUnavailable as exc:
        log.warning("No restatements (%s): run `rag index dense`, then `rag dedup run`", exc)
    else:
        references = read_jsonl(step_dir(config, "structure") / REFERENCES, Reference)
        restatements = find_restatements(
            surveys,
            chunks,
            references,
            ReferenceResolver(records, cfg.reference_title_min),
            record_work,
            dense.embeddings,
            dense.chunk_ids,
            cfg.restate_min,
        )
    write_jsonl(out / RESTATEMENTS, restatements)
    return {"surveys": len(surveys), "restatements": len(restatements)}


def review_markdown(works: Sequence[Work], records: Sequence[ManifestRecord]) -> str:
    """Every work that fuzzy evidence built or touched, for a human to check."""
    titles = {r.record_id: r.title for r in records}
    lines = [
        "# Fuzzy merges to review",
        "",
        "Works joined by title or text evidence (not identifiers), and edges the",
        "chain guard refused. Generated by `rag dedup run`; record false merges in",
        "DECISIONS.md (D-040).",
        "",
    ]
    for work in works:
        fuzzy = [e for e in work.edges if e.step != "ids"]
        if not fuzzy and not work.rejected:
            continue
        lines.append(f"## {work.work_id}")
        lines.append("")
        for rid in work.members:
            mark = " (canonical)" if rid == work.canonical else ""
            lines.append(f"- `{rid}`{mark}: {titles.get(rid, '?')}")
        for label, edges in (("merged", fuzzy), ("refused", work.rejected)):
            for e in edges:
                lines.append(
                    f"  - {label} by {e.step}/{e.reason}: `{e.a}` ~ `{e.b}` "
                    f"(title {e.title_ratio}, authors {e.author_overlap}, "
                    f"years {e.year_gap}, text {e.jaccard})"
                )
        lines.append("")
    return "\n".join(lines)


def read_record_work(out: Path) -> dict[str, str]:
    return json.loads((out / RECORD_WORK).read_text(encoding="utf-8"))
