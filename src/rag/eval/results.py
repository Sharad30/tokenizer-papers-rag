"""The results file every `rag eval run` writes: enough to trace and reproduce each number."""

from datetime import datetime
from typing import Any

from rag.common.schemas import StrictModel
from rag.eval.schemas import QuestionType
from rag.retrieve.schemas import IndexMeta


class QuestionResult(StrictModel):
    question_id: str
    type: QuestionType
    variant: str  # "raw", "oracle" (follow-ups' oracle rewrite) or "system" (Phase 7)
    query: str | None  # None: variant not available yet (system rewrite before Phase 7)
    scored: bool  # False for unanswerable questions and missing variants
    metrics: dict[str, float] = {}  # "recall@5", "paper_recall@5", "mrr@10", "dup@5", "any_dup@5"
    # Highest dense cosine in the results: material for the Phase 7 abstention gate.
    top_dense_cosine: float | None = None
    top_chunk_ids: list[str] = []
    gold_unmapped: int = 0  # gold spans that map to no chunk (validate should have caught)


class MetricSummary(StrictModel):
    mean: float
    ci_low: float
    ci_high: float
    n: int


class ModeResult(StrictModel):
    # slice ("overall", "type:lookup", "followup:oracle", ...) -> metric -> summary
    aggregates: dict[str, dict[str, MetricSummary]]
    questions: list[QuestionResult]


class RunResult(StrictModel):
    run: str
    created_at: datetime
    git_commit: str
    config_hash: str
    config: dict[str, Any]
    questions_file: str
    questions_sha256: str
    n_questions: int
    index_meta: dict[str, IndexMeta]
    modes: dict[str, ModeResult]
