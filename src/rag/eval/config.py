"""Config models for the eval set (the `eval:` section of config.yaml)."""

from pydantic import Field

from rag.common.schemas import StrictModel
from rag.eval.schemas import QuestionType


class FindConfig(StrictModel):
    max_hits: int = Field(ge=1)  # `rag eval find` stops listing after this many


class EvalConfig(StrictModel):
    questions_file: str  # under paths.eval
    # PLAN's question mix, as minimums per type (D-025).
    mix_min: dict[QuestionType, int]
    duplicate_sensitive_min: int = Field(ge=0)
    conversations_min: int = Field(ge=0)
    # Longer spans are more likely to cross a sentence and so a chunk boundary.
    max_span_words: int = Field(ge=1)
    find: FindConfig
