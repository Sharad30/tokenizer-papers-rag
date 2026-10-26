"""Config models for retrieval (the `retrieve:` section of config.yaml)."""

from typing import Literal

from pydantic import Field

from rag.common.schemas import StrictModel

Mode = Literal["bm25", "dense", "hybrid"]


class RetrieveConfig(StrictModel):
    default_mode: Mode
    default_k: int = Field(ge=1)
    # How deep each retriever's list goes into fusion (D-030).
    candidates: int = Field(ge=1)
    rrf_k: float = Field(gt=0)  # RRF's constant: larger = flatter rank weights
