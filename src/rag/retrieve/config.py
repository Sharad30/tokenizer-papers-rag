"""Config models for retrieval (the `retrieve:` section of config.yaml)."""

from typing import Literal

from pydantic import Field

from rag.common.schemas import StrictModel

Mode = Literal["bm25", "dense", "hybrid"]


class DedupSettings(StrictModel):
    """Query-time duplicate handling (D-041); reads the works from `rag dedup run`."""

    enabled: bool  # false: Phase 4 behaviour, works not loaded
    canonical_only: bool  # hide non-canonical members' chunks
    per_work_cap: int = Field(ge=0)  # chunks per work in the results; 0 = no cap
    survey_demotion: bool
    mmr_lambda: float | None = Field(default=None, ge=0, le=1)  # None = off


class RetrieveConfig(StrictModel):
    default_mode: Mode
    default_k: int = Field(ge=1)
    # How deep each retriever's list goes into fusion (D-030).
    candidates: int = Field(ge=1)
    rrf_k: float = Field(gt=0)  # RRF's constant: larger = flatter rank weights
    dedup: DedupSettings
