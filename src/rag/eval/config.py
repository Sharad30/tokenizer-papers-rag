"""Config models for the eval set (the `eval:` section of config.yaml)."""

from pydantic import Field

from rag.common.schemas import StrictModel
from rag.dedup.schemas import Step
from rag.eval.schemas import QuestionType
from rag.retrieve.config import DedupSettings, Mode


class FindConfig(StrictModel):
    max_hits: int = Field(ge=1)  # `rag eval find` stops listing after this many


class BootstrapConfig(StrictModel):
    samples: int = Field(ge=100)
    seed: int


class Condition(StrictModel):
    """One row of an ablation table."""

    name: str
    label: str  # as shown in the table
    steps: list[Step] = []  # resolver steps that build the works
    dedup: DedupSettings  # query-time settings (retrieve.dedup)


class AblationConfig(StrictModel):
    queries_file: str  # under paths.probes
    mode: Mode | None = None  # default: retrieve.default_mode
    conditions: list[Condition] = Field(min_length=1)


class EvalConfig(StrictModel):
    questions_file: str  # under paths.eval
    pairs_file: str  # Sharad's labelled record pairs, under paths.eval (D-042)
    # PLAN's question mix, as minimums per type (D-025).
    mix_min: dict[QuestionType, int]
    duplicate_sensitive_min: int = Field(ge=0)
    conversations_min: int = Field(ge=0)
    # Longer spans are more likely to cross a sentence and so a chunk boundary.
    max_span_words: int = Field(ge=1)
    find: FindConfig
    # Phase 4 metrics (D-031).
    recall_k: int = Field(ge=1)
    mrr_cutoff: int = Field(ge=1)
    dup_k: int = Field(ge=1)
    bootstrap: BootstrapConfig
    ablations: dict[str, AblationConfig] = {}  # `rag eval ablate --name <key>`
