"""Schemas that cross the index/retrieve boundary."""

from datetime import datetime
from typing import Any, Literal

from rag.common.schemas import StrictModel
from rag.parse.schemas import Chunk


class IndexMeta(StrictModel):
    """Written next to each index: what it was built from and with which settings.

    The retriever compares it with the current chunks file and config, so a
    stale index fails loudly instead of returning chunk ids that no longer exist.
    """

    kind: Literal["bm25", "dense"]
    settings: dict[str, Any]  # the index's config section, as dumped
    chunks_sha256: str
    n_chunks: int
    limit: int | None  # papers, when built with --limit
    built_at: datetime
    git_commit: str


class Hit(StrictModel):
    chunk: Chunk
    rank: int  # 1-based, in the mode asked for
    score: float  # that mode's score: BM25, cosine, or RRF
    # Per retriever, when the chunk is in its candidate list: shown side by
    # side by `rag inspect query`.
    ranks: dict[str, int] = {}
    scores: dict[str, float] = {}


class Retrieval(StrictModel):
    hits: list[Hit]
    # Best score per retriever over every allowed chunk, not just the hits:
    # the abstention gate needs "how close is anything in the corpus" (D-036).
    top_scores: dict[str, float] = {}
