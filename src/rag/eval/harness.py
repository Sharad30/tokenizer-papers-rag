"""Score retrieval modes on the eval set, the same way every time.

For each question and query variant: retrieve the top results, decide which
of them contain gold spans, compute the metrics, then aggregate per slice
with bootstrap intervals. The retriever is passed in, so the harness scores
any configuration (and a fake one in tests).
"""

from collections import defaultdict
from collections.abc import Sequence
from typing import Protocol

from rag.eval.config import EvalConfig
from rag.eval.metrics import (
    bootstrap_ci,
    duplicate_rate_at_k,
    paper_recall_at_k,
    reciprocal_rank,
    span_recall_at_k,
)
from rag.eval.results import MetricSummary, ModeResult, QuestionResult
from rag.eval.schemas import GoldSpan, Question, QuestionType
from rag.eval.spans import ChunkTexts, map_span
from rag.retrieve.schemas import Hit


class Retrieves(Protocol):
    def retrieve(self, query: str, k: int | None = None, mode=None) -> list[Hit]: ...


class Gold:
    """A question's gold, resolved against the current chunks and identity groups."""

    def __init__(self, spans: list[set[str]], papers: set[str]) -> None:
        self.spans = spans  # per gold span: chunk ids containing it (any linked copy)
        self.papers = papers  # identity groups of the gold papers


def resolve_gold(gold: Sequence[GoldSpan], texts: ChunkTexts, groups: dict[str, str]) -> Gold:
    members: dict[str, list[str]] = defaultdict(list)
    for record_id, group in groups.items():
        members[group].append(record_id)
    spans = []
    for g in gold:
        group = groups.get(g.record_id, g.record_id)
        copies = members.get(group, [g.record_id])
        # A linked copy containing the same span is the same evidence (D-031).
        spans.append(
            {
                cid
                for rid in copies
                for cid in map_span(g.model_copy(update={"record_id": rid}), texts)
            }
        )
    return Gold(spans, {groups.get(g.record_id, g.record_id) for g in gold})


def variants(q: Question, rewrites: dict[str, str] | None = None) -> list[tuple[str, str | None]]:
    """The queries a question is scored with. Follow-ups three ways (PLAN Phase 4).

    "system" is the chat's own rewrite (Phase 7, `rag eval run --rewrite`);
    without it the variant stays unscored.
    """
    if q.type == QuestionType.FOLLOWUP:
        system = (rewrites or {}).get(q.id)
        return [("raw", q.question), ("oracle", q.oracle_rewrite), ("system", system)]
    return [("raw", q.question)]


def score_question(
    q: Question,
    variant: str,
    query: str | None,
    hits: list[Hit],
    gold: Gold,
    groups: dict[str, str],
    config: EvalConfig,
) -> QuestionResult:
    ranked = [h.chunk.chunk_id for h in hits]
    papers = [groups.get(h.chunk.record_id, h.chunk.record_id) for h in hits]
    dense = [h.scores["dense"] for h in hits if "dense" in h.scores]
    result = QuestionResult(
        question_id=q.id,
        type=q.type,
        variant=variant,
        query=query,
        scored=False,
        top_dense_cosine=max(dense) if dense else None,
        top_chunk_ids=ranked,
        gold_unmapped=sum(not s for s in gold.spans),
    )
    if query is None or not q.answerable:
        return result
    k, cutoff, dup_k = config.recall_k, config.mrr_cutoff, config.dup_k
    dup = duplicate_rate_at_k(papers, dup_k)
    result.scored = True
    result.metrics = {
        f"recall@{k}": span_recall_at_k(gold.spans, ranked, k),
        f"paper_recall@{k}": paper_recall_at_k(gold.papers, papers, k),
        f"mrr@{cutoff}": reciprocal_rank(gold.spans, ranked, cutoff),
        f"dup@{dup_k}": dup,
        f"any_dup@{dup_k}": float(dup > 0),
    }
    return result


def run_mode(
    retriever: Retrieves,
    mode: str,
    questions: list[Question],
    texts: ChunkTexts,
    groups: dict[str, str],
    config: EvalConfig,
    rewrites: dict[str, str] | None = None,
) -> ModeResult:
    depth = max(config.recall_k, config.mrr_cutoff, config.dup_k)
    rows = []
    for q in questions:
        gold = resolve_gold(q.gold, texts, groups)
        for variant, query in variants(q, rewrites):
            hits = retriever.retrieve(query, k=depth, mode=mode) if query else []
            rows.append(score_question(q, variant, query, hits, gold, groups, config))
    return ModeResult(aggregates=aggregate(rows, config), questions=rows)


def slices(row: QuestionResult) -> list[str]:
    """Which aggregate rows a scored result counts in.

    "overall" is what a user would type: every answerable question, follow-ups
    as asked (raw). Follow-ups also get a slice per variant, so a gap between
    raw and oracle measures what query rewriting could win.
    """
    if row.type == QuestionType.FOLLOWUP:
        named = [f"followup:{row.variant}"]
        return ["overall", f"type:{row.type}", *named] if row.variant == "raw" else named
    return ["overall", f"type:{row.type}"]


def aggregate(
    rows: list[QuestionResult], config: EvalConfig
) -> dict[str, dict[str, MetricSummary]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if not row.scored:
            continue
        for name in slices(row):
            for metric, value in row.metrics.items():
                grouped[name][metric].append(value)
    return {
        name: {metric: _summary(vals, config) for metric, vals in metrics.items()}
        for name, metrics in grouped.items()
    }


def _summary(values: list[float], config: EvalConfig) -> MetricSummary:
    mean, low, high = bootstrap_ci(values, config.bootstrap.samples, config.bootstrap.seed)
    return MetricSummary(mean=mean, ci_low=low, ci_high=high, n=len(values))
