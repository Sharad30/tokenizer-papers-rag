"""`rag eval answers`: run every eval question through the chat and keep everything (D-046).

Conversations run in order, each from a fresh history; a standalone question
always starts fresh too. The retrieval gate is switched off for the run, so
every question gets an answer and a recorded top cosine; the deployed gate
(and any other threshold) is then simulated by the report: a question
abstains when its top cosine is below the threshold or the model said
NOT_SUPPORTED. One LLM pass serves the whole threshold sweep.

Each answer is split into claims, one per sentence with the context labels
it cites (`chat.citations.claim_sentences`). Sharad labels them by hand
(`rag eval label`); this module never judges anything.
"""

from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime

from rag.chat.citations import claim_sentences
from rag.chat.engine import ChatEngine, TurnResult
from rag.common.schemas import StrictModel
from rag.eval.harness import resolve_gold
from rag.eval.schemas import Question
from rag.eval.spans import ChunkTexts


class Claim(StrictModel):
    index: int  # within the answer, from 0
    text: str  # the sentence as the model wrote it, citation markers included
    cites: list[str]  # context labels, e.g. ["C1", "C3"]; empty: an uncited claim


class ContextEntry(StrictModel):
    label: str  # "C1"
    chunk_id: str
    record_id: str
    title: str


class AnswerRecord(StrictModel):
    question_id: str
    type: str
    conversation_id: str | None = None
    turn: int | None = None
    answerable: bool
    unanswerable_kind: str | None = None
    expected_behavior: str | None = None
    user: str
    rewrite: str  # the query the chat retrieved with (equals `user` on turn 1)
    oracle_rewrite: str | None = None
    context: list[ContextEntry]
    top_dense_cosine: float | None
    model_abstained: bool  # the model answered NOT_SUPPORTED
    completion: str | None  # the model's raw text, [Cn] labels intact
    reply: str  # what the user saw (renumbered citations)
    cited: list[str]  # context labels the answer used, in order
    invalid_citations: list[str]
    uncited_sentences: int
    claims: list[Claim]
    # Share of the question's gold passages / papers present in the context;
    # None for questions without gold.
    gold_passages_in_context: float | None = None
    gold_papers_in_context: float | None = None
    seconds: dict[str, float] = {}


class AnswersMeta(StrictModel):
    created_at: datetime
    git_commit: str
    config_hash: str
    model: str
    prompts: dict[str, str]  # prompt file -> sha256
    questions_file: str
    questions_sha256: str
    n_questions: int
    deployed_threshold: float  # chat.abstain.min_dense_cosine, simulated by the report
    note: str = "retrieval gate off during the run; thresholds are simulated"


def conversations(questions: Sequence[Question]) -> list[list[Question]]:
    """Questions grouped as the chat must see them: each conversation in turn
    order, each standalone question alone, in the order they first appear."""
    groups: dict[str, list[Question]] = {}
    for q in questions:
        key = q.conversation_id or f"standalone:{q.id}"
        groups.setdefault(key, []).append(q)
    return [sorted(g, key=lambda q: q.turn or 0) for g in groups.values()]


def run(
    questions: Sequence[Question],
    engine: ChatEngine,
    texts: ChunkTexts,
    groups: dict[str, str],
    on_answer: Callable[[AnswerRecord], None] = lambda r: None,
) -> list[AnswerRecord]:
    out = []
    for conversation in conversations(questions):
        engine.reset()
        for q in conversation:
            record = to_record(q, engine.turn(q.question), texts, groups)
            on_answer(record)
            out.append(record)
    return out


def to_record(
    q: Question, result: TurnResult, texts: ChunkTexts, groups: dict[str, str]
) -> AnswerRecord:
    context = [
        ContextEntry(
            label=cc.label,
            chunk_id=cc.hit.chunk.chunk_id,
            record_id=cc.hit.chunk.record_id,
            title=cc.title,
        )
        for cc in result.context
    ]
    raw = result.completion.text if result.completion else None
    passages = papers = None
    if q.gold:
        gold = resolve_gold(q.gold, texts, groups)
        ids = {c.chunk_id for c in context}
        in_context = {groups.get(c.record_id, c.record_id) for c in context}
        passages = sum(bool(s & ids) for s in gold.spans) / len(gold.spans)
        papers = len(gold.papers & in_context) / len(gold.papers)
    return AnswerRecord(
        question_id=q.id,
        type=q.type,
        conversation_id=q.conversation_id,
        turn=q.turn,
        answerable=q.answerable,
        unanswerable_kind=q.unanswerable_kind,
        expected_behavior=q.expected_behavior,
        user=q.question,
        rewrite=result.rewrite.query,
        oracle_rewrite=q.oracle_rewrite,
        context=context,
        top_dense_cosine=result.top_dense_cosine,
        model_abstained=result.abstained == "model",
        completion=raw,
        reply=result.reply,
        cited=list(result.cited.used) if result.cited else [],
        invalid_citations=list(result.cited.invalid) if result.cited else [],
        uncited_sentences=result.cited.uncited_sentences if result.cited else 0,
        claims=claims(raw) if raw and result.abstained is None else [],
        gold_passages_in_context=passages,
        gold_papers_in_context=papers,
        seconds={k: round(v, 2) for k, v in result.seconds.items()},
    )


def claims(text: str) -> list[Claim]:
    return [
        Claim(index=i, text=sentence, cites=labels)
        for i, (sentence, labels) in enumerate(claim_sentences(text))
    ]


def meta(questions: Iterable[Question], deployed_threshold: float, **fields) -> AnswersMeta:
    return AnswersMeta(
        created_at=datetime.now(UTC),
        n_questions=len(list(questions)),
        deployed_threshold=deployed_threshold,
        **fields,
    )
