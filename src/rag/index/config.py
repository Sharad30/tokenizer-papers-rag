"""Config models for the indexes (the `index:` section of config.yaml)."""

from typing import Literal

from pydantic import Field

from rag.common.schemas import StrictModel


class Bm25Config(StrictModel):
    method: Literal["lucene", "robertson"]  # IDF/TF variant in bm25s (D-026)
    k1: float = Field(ge=0)  # term-frequency saturation
    b: float = Field(ge=0, le=1)  # document-length normalisation
    include_header: bool  # index "Title > Section" with the body (D-029)
    stopwords: Literal["en", "none"]
    split_compounds: bool  # "byte-level" also indexes "byte" and "level" (D-027)


class DenseConfig(StrictModel):
    model: str
    # Prepended to queries only. The model's sentence-transformers config has
    # no prompts, so encode_query would otherwise add nothing.
    query_prompt: str
    prepend_header: bool  # embed "header\ntext" (D-029)
    batch_size: int = Field(ge=1)
    device: str  # "cpu": CLAUDE.md keeps the project CPU-only


class IndexConfig(StrictModel):
    bm25: Bm25Config
    dense: DenseConfig
