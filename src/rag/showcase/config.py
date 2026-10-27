"""Config models for the showcase UI (the `showcase:` section of config.yaml)."""

from pydantic import Field

from rag.common.schemas import StrictModel


class CollectShowcaseConfig(StrictModel):
    # Rejected abstracts aren't committed, so the snapshot keeps this many
    # per decision path to draw examples from on a fresh clone.
    pool_per_path: int = Field(ge=1)


class ShowcaseConfig(StrictModel):
    seed: int  # first sample of examples; "Show others" moves on from it
    examples_shown: int = Field(ge=1)
    collect: CollectShowcaseConfig
