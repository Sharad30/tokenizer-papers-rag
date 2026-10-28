"""Config models for the indexes (the `index:` section of config.yaml)."""

from typing import Literal

from pydantic import Field, model_validator

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
    # Fields below default to the Phase 4 behaviour; `settings` leaves defaults
    # out, so indexes built before they existed stay valid.
    kind: Literal["sentence_transformers", "specter2"] = "sentence_transformers"
    header_separator: str = "\n"  # between header and text; SPECTER2 expects [SEP]
    # SPECTER2 (D-043): adapters for queries and documents, and the packages of
    # the isolated environment it runs in (`adapters` pins an older transformers).
    query_adapter: str | None = None
    document_adapter: str | None = None
    max_length: int | None = None  # tokens; longer inputs are truncated (and counted)
    isolated_with: list[str] = []

    def settings(self) -> dict:
        """What an index records and is checked against."""
        return self.model_dump(mode="json", exclude_defaults=True)


class IndexConfig(StrictModel):
    bm25: Bm25Config
    dense_model: str  # which of dense_models the Retriever, chat and eval use
    # Every dense model compared in Phase 6, by name; each has its own index folder.
    dense_models: dict[str, DenseConfig]

    @model_validator(mode="after")
    def _deployed_is_known(self) -> "IndexConfig":
        if self.dense_model not in self.dense_models:
            raise ValueError(f"index.dense_model {self.dense_model!r} is not in dense_models")
        return self

    @property
    def dense(self) -> DenseConfig:
        return self.dense_models[self.dense_model]
