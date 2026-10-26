"""`rag index`: build the BM25 and dense indexes over the Phase 2 chunks.

    bm25    chunks -> data/index/bm25/
    dense   chunks -> data/index/dense/<model>/   (embeddings.npy)

Each index folder gets a meta.json (settings + sha256 of chunks.jsonl) that
the retriever checks before use, and a stats.json.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path

from rag.common.config import Config
from rag.common.io import file_sha256, read_jsonl, write_json
from rag.common.provenance import git_commit
from rag.common.stats import stage_stats
from rag.parse.pipeline import CHUNKS, step_dir
from rag.parse.schemas import Chunk
from rag.retrieve.bm25 import Bm25Index
from rag.retrieve.dense import DenseIndex, Encoder, SentenceTransformerEncoder
from rag.retrieve.retriever import META, bm25_dir, dense_dir
from rag.retrieve.schemas import IndexMeta

log = logging.getLogger(__name__)


def _chunks(config: Config, limit: int | None) -> tuple[list[Chunk], str]:
    """The chunks to index (all, or those of the first `limit` papers) and the file's sha256."""
    path = step_dir(config, "chunk") / CHUNKS
    chunks = read_jsonl(path, Chunk)
    if limit is not None:
        keep = list(dict.fromkeys(c.record_id for c in chunks))[:limit]
        chunks = [c for c in chunks if c.record_id in set(keep)]
    return chunks, file_sha256(path)


def _write_meta(out: Path, kind, settings: dict, sha: str, n: int, limit: int | None) -> None:
    meta = IndexMeta(
        kind=kind,
        settings=settings,
        chunks_sha256=sha,
        n_chunks=n,
        limit=limit,
        built_at=datetime.now(UTC),
        git_commit=git_commit(),
    )
    write_json(out / META, meta)


def bm25(config: Config, limit: int | None = None) -> None:
    out = bm25_dir(config)
    chunks, sha = _chunks(config, limit)
    with stage_stats(config, "index.bm25", out, limit) as stats:
        index = Bm25Index.build(chunks, config.index.bm25)
        index.save(out)
        _write_meta(out, "bm25", config.index.bm25.model_dump(mode="json"), sha, len(chunks), limit)
        stats["counts"].update(
            chunks=len(chunks),
            papers=len({c.record_id for c in chunks}),
            vocabulary=len(index.model.vocab_dict),
        )
    log.info("BM25 index: %d chunks -> %s", len(chunks), out)


def dense(config: Config, limit: int | None = None, encoder: Encoder | None = None) -> None:
    out = dense_dir(config)
    chunks, sha = _chunks(config, limit)
    with stage_stats(config, "index.dense", out, limit) as stats:
        encoder = encoder or SentenceTransformerEncoder(config.index.dense, show_progress=True)
        index = DenseIndex.build(chunks, config.index.dense, encoder)
        index.save(out)
        _write_meta(
            out, "dense", config.index.dense.model_dump(mode="json"), sha, len(chunks), limit
        )
        stats["counts"].update(
            chunks=len(chunks),
            papers=len({c.record_id for c in chunks}),
            dimensions=int(index.embeddings.shape[1]),
        )
    log.info("Dense index: %s -> %s", index.embeddings.shape, out)


def run_all(config: Config, limit: int | None = None) -> None:
    bm25(config, limit)
    dense(config, limit)
