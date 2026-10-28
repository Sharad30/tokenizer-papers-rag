"""Config models for the dedup stage (the `dedup:` section of config.yaml)."""

from typing import Literal

from pydantic import Field, model_validator

from rag.common.schemas import StrictModel
from rag.dedup.schemas import Step


class TitleConfig(StrictModel):
    # rapidfuzz token_sort_ratio on normalised titles, 0-100. At or above
    # `min_ratio` (with shared authors) a title match is enough on its own;
    # between `review_ratio` and `min_ratio` it needs text evidence too, because
    # same-author follow-ups have near-identical titles (D-040).
    min_ratio: float = Field(ge=0, le=100)
    review_ratio: float = Field(ge=0, le=100)
    review_min_jaccard: float = Field(ge=0, le=1)
    year_gap_max: int = Field(ge=0)


class MinHashConfig(StrictModel):
    shingle_words: int = Field(ge=1)
    num_perm: int = Field(ge=1)
    bands: int = Field(ge=1)
    rows: int = Field(ge=1)
    min_jaccard: float = Field(ge=0, le=1)  # estimated Jaccard for a text-only merge

    @model_validator(mode="after")
    def _bands_fill_signature(self) -> "MinHashConfig":
        if self.bands * self.rows != self.num_perm:
            raise ValueError("minhash: bands * rows must equal num_perm")
        return self


class SurveyConfig(StrictModel):
    # Whole words in a title that make a paper a survey. Not S2's "Review"
    # publication type: it also tags ordinary research papers (D-041).
    title_cues: list[str]
    # Bibliography entry -> indexed record by title (fuzz.ratio, 0-100), when
    # it has no arXiv id or DOI.
    reference_title_min: float = Field(ge=0, le=100)
    # Dense cosine between a citing survey chunk and the cited work's closest chunk.
    restate_min: float = Field(ge=-1, le=1)


class DedupConfig(StrictModel):
    seed: int  # MinHash hash functions
    steps: list[Step]  # which resolver steps run (the ablation switches them off)
    # Share of the smaller author list whose surnames appear in the other one.
    # Every fuzzy merge needs it: text or title alone never merges two records.
    author_overlap_min: float = Field(ge=0, le=1)
    title: TitleConfig
    minhash: MinHashConfig
    # Transitive-chain guard: two parts of a cluster may join only if every pair
    # of ID-linked units across them has direct evidence: a merge edge, or
    # estimated text Jaccard at least this.
    cluster_floor: float = Field(ge=0, le=1)
    # Which member represents a work, by source; ties: latest arXiv version, then most chunks.
    canonical_order: list[Literal["acl", "arxiv", "s2"]]
    surveys: SurveyConfig

    @model_validator(mode="after")
    def _review_below_min(self) -> "DedupConfig":
        if self.title.review_ratio > self.title.min_ratio:
            raise ValueError("dedup.title: review_ratio must be <= min_ratio")
        return self
