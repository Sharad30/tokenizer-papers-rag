"""Config models for the showcase UI (the `showcase:` section of config.yaml)."""

from pydantic import Field

from rag.common.schemas import StrictModel


class CollectShowcaseConfig(StrictModel):
    # Rejected abstracts aren't committed, so the snapshot keeps this many
    # per decision path to draw examples from on a fresh clone.
    pool_per_path: int = Field(ge=1)


class ParseShowcaseConfig(StrictModel):
    # Chunks kept per element type and per quality check, and references,
    # in the snapshot; also how many whole papers it carries.
    pool_size: int = Field(ge=1)
    snapshot_papers: int = Field(ge=1)
    token_bin_width: int = Field(ge=1)  # histogram bins of model tokens
    # A paragraph chunk with this many bibliography-entry patterns is flagged
    # as reference-list text that leaked into the body.
    bibliography_min_matches: int = Field(ge=1)


class DedupShowcaseConfig(StrictModel):
    # ID-only works and restatements kept in the snapshot (fuzzy merges are all kept).
    pool_size: int = Field(ge=1)


class ShowcaseConfig(StrictModel):
    seed: int  # first sample of examples; "Show others" moves on from it
    examples_shown: int = Field(ge=1)
    collect: CollectShowcaseConfig
    parse: ParseShowcaseConfig
    dedup: DedupShowcaseConfig
