"""Config models for the eval set (the `eval:` section of config.yaml)."""

from pydantic import Field, model_validator

from rag.common.schemas import StrictModel
from rag.dedup.schemas import Step
from rag.eval.schemas import QuestionType
from rag.parse.schemas import ElementType
from rag.retrieve.config import DedupSettings, Mode


class FindConfig(StrictModel):
    max_hits: int = Field(ge=1)  # `rag eval find` stops listing after this many


class DraftConfig(StrictModel):
    """LLM-drafted questions turned into gold by hand (D-045)."""

    file: str  # under paths.eval
    title_min_ratio: float = Field(ge=0, le=100)  # fuzzy title fallback in check-papers
    max_candidates: int = Field(ge=1)  # passages `resolve` lists per gold entry
    excerpt_chars: int = Field(ge=1)  # shown either side of a match
    absent_max_excerpts: int = Field(ge=0)  # excerpts per term in the absent check
    short_term_chars: int = Field(ge=1)  # terms this short match whole words only


class BootstrapConfig(StrictModel):
    samples: int = Field(ge=100)
    seed: int


class SilverConfig(StrictModel):
    """`rag eval silver`: LLM-generated questions, never mixed with the gold set (D-044)."""

    file: str  # under paths.eval
    prompt: str  # file in paths.prompts
    n: int = Field(ge=1)  # papers sampled, one chunk each
    seed: int
    elements: list[ElementType]
    min_tokens: int = Field(ge=1)  # chunks shorter than this rarely hold a specific answer
    max_tokens: int = Field(ge=16)  # generation budget
    min_question_words: int = Field(ge=1)
    max_evidence_words: int = Field(ge=1)


class Condition(StrictModel):
    """One row of an ablation table. Unset fields mean the deployed configuration."""

    name: str
    label: str  # as shown in the table
    steps: list[Step] | None = None  # resolver steps that build the works (default: dedup.steps)
    dedup: DedupSettings | None = None  # query-time settings (default: retrieve.dedup)
    dense_model: str | None = None  # one of index.dense_models (default: index.dense_model)
    mode: Mode | None = None  # default: the ablation's mode
    rerank: bool = False  # cross-encoder over the fused candidates (retrieve.rerank)
    query_prompt: str | None = None  # override the dense model's query prompt ("" = none)


class AblationConfig(StrictModel):
    queries_file: str  # under paths.probes: probe queries (dup@k, latency), not gold
    mode: Mode | None = None  # default: retrieve.default_mode
    # Question sets scored for Recall/MRR, label -> file under paths.eval; missing
    # files are skipped. Default: {"gold": eval.questions_file}.
    question_files: dict[str, str] = {}
    conditions: list[Condition] = Field(min_length=1)
    # Condition pairs (a, b) to compare question by question, besides the top
    # two: the comparisons a decision rests on.
    pairs: list[tuple[str, str]] = []

    @model_validator(mode="after")
    def _pairs_name_conditions(self) -> "AblationConfig":
        names = {c.name for c in self.conditions}
        unknown = sorted({n for pair in self.pairs for n in pair} - names)
        if unknown:
            raise ValueError(f"pairs name unknown conditions: {unknown}")
        return self


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
    draft: DraftConfig
    # Phase 4 metrics (D-031).
    recall_k: int = Field(ge=1)
    mrr_cutoff: int = Field(ge=1)
    dup_k: int = Field(ge=1)
    bootstrap: BootstrapConfig
    silver: SilverConfig
    ablations: dict[str, AblationConfig] = {}  # `rag eval ablate --name <key>`
