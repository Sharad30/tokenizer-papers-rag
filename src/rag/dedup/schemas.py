"""Schemas the dedup stage writes to data/works/."""

from datetime import datetime
from typing import Literal

from rag.common.schemas import StrictModel

Step = Literal["ids", "title", "minhash"]


class Edge(StrictModel):
    """Evidence that two records are the same paper, and which step found it."""

    a: str
    b: str
    step: Step
    reason: str  # "arxiv", "doi", "s2", "acl", "derived_from", "title", "text"
    title_ratio: float | None = None
    author_overlap: float | None = None
    year_gap: int | None = None
    jaccard: float | None = None  # MinHash estimate over full-text shingles


class Work(StrictModel):
    """One paper, however many records (versions, venue copies) it has."""

    work_id: str
    canonical: str  # the record whose chunks represent the work
    canonical_rule: str
    members: list[str]
    aliases: list[str]  # members other than the canonical one, kept for citation
    edges: list[Edge]  # the merges that built this work
    rejected: list[Edge]  # fuzzy edges the chain guard refused, touching this work


class Survey(StrictModel):
    record_id: str
    cues: list[str]  # the title words that flagged it


class Restatement(StrictModel):
    """A survey chunk that says what work W says, citing W."""

    chunk_id: str
    work_id: str
    cosine: float  # dense cosine to W's closest chunk
    ref_id: str  # the bibliography entry the citation points to
    matched_by: str  # how that entry was matched to W: arxiv, doi or title


class WorksMeta(StrictModel):
    """What the works were built from: the Retriever refuses stale works files."""

    chunks_sha256: str
    manifest_sha256: str
    steps: list[Step]
    n_records: int
    n_works: int
    limit: int | None
    built_at: datetime
    git_commit: str
