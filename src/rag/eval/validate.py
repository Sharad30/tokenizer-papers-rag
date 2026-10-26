"""`rag eval validate`: is the eval set usable as gold?

Per-question rules live in `schemas.Question`. This module adds the checks
that need the corpus or the whole file: papers exist and are indexed, every
span maps to a chunk, conversations are well formed, and the type mix meets
PLAN's minimums. Errors fail validation; warnings are things to look at.
"""

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from rag.eval.config import EvalConfig
from rag.eval.schemas import Question, QuestionType
from rag.eval.spans import ChunkTexts, elsewhere, map_span, normalise, suggest

# A sentence end followed by a new sentence: such a span may straddle two
# chunks, because chunks are cut between sentences (D-017).
_SENTENCE_BREAK = re.compile(r"[.!?][\"')\]]?\s+[A-Z]")


@dataclass
class Report:
    n_questions: int = 0
    missing_file: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    mix: list[tuple[str, int, int]] = field(default_factory=list)  # (what, have, minimum)

    @property
    def ok(self) -> bool:
        return not self.errors


def load_questions(path: Path) -> tuple[list[Question], list[str]]:
    """Every valid question, plus one error per bad line (with its line number)."""
    questions: list[Question] = []
    errors: list[str] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            questions.append(Question.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as exc:
            errors.append(f"line {n}: {_one_line(exc)}")
    return questions, errors


def validate_file(
    path: Path, texts: ChunkTexts, indexed: dict[str, bool], config: EvalConfig
) -> Report:
    if not path.is_file():
        report = Report(missing_file=True)
        report.mix = _mix([], config)
        return report
    questions, errors = load_questions(path)
    report = validate(questions, texts, indexed, config)
    report.errors[:0] = errors
    return report


def validate(
    questions: list[Question], texts: ChunkTexts, indexed: dict[str, bool], config: EvalConfig
) -> Report:
    """Cross-question and corpus checks. `indexed` maps record_id -> included_in_index."""
    report = Report(n_questions=len(questions))
    for qid, n in Counter(q.id for q in questions).items():
        if n > 1:
            report.errors.append(f"{qid}: id used {n} times")
    for q in questions:
        _check_gold(q, texts, indexed, config, report)
    _check_conversations(questions, report)
    report.mix = _mix(questions, config)
    for what, have, minimum in report.mix:
        if have < minimum:
            report.errors.append(f"mix: {what} has {have}, needs at least {minimum}")
    return report


def _check_gold(
    q: Question, texts: ChunkTexts, indexed: dict[str, bool], config: EvalConfig, report: Report
) -> None:
    for i, gold in enumerate(q.gold, start=1):
        where = f"{q.id} gold {i}"
        if gold.record_id not in indexed:
            report.errors.append(f"{where}: {gold.record_id} is not in the manifest")
            continue
        if not indexed[gold.record_id]:
            report.errors.append(f"{where}: {gold.record_id} is not in the index (no chunks)")
            continue
        chunk_ids = map_span(gold, texts)
        if not chunk_ids:
            hint = suggest(gold.span, texts.paper(gold.record_id))
            closest = (
                f"; closest: {hint.chunk_id} ({hint.similarity}) {hint.excerpt!r}" if hint else ""
            )
            report.errors.append(f"{where}: span not found in {gold.record_id}{closest}")
            continue
        if len(gold.span.split()) > config.max_span_words or _SENTENCE_BREAK.search(gold.span):
            report.warnings.append(
                f"{where}: long or multi-sentence span; it may straddle a chunk boundary"
            )
        if gold.section and not _section_matches(gold.section, chunk_ids, texts, gold.record_id):
            report.warnings.append(
                f"{where}: section {gold.section!r} doesn't match the chunk's section path"
            )
        others = elsewhere(gold.span, texts, exclude=gold.record_id)
        if others and not q.duplicate_sensitive:
            report.warnings.append(
                f"{where}: span also in {', '.join(others)}; duplicate_sensitive is false"
            )


def _section_matches(section: str, chunk_ids: list[str], texts: ChunkTexts, record: str) -> bool:
    wanted = normalise(section)
    ids = set(chunk_ids)
    return any(
        wanted in normalise(" > ".join(ic.chunk.section_path))
        for ic in texts.paper(record)
        if ic.chunk.chunk_id in ids
    )


def _check_conversations(questions: list[Question], report: Report) -> None:
    turns: dict[str, list[int]] = defaultdict(list)
    for q in questions:
        if q.conversation_id is not None and q.turn is not None:
            turns[q.conversation_id].append(q.turn)
    for conversation, seen in turns.items():
        if sorted(seen) != list(range(1, len(seen) + 1)):
            report.errors.append(
                f"conversation {conversation}: turns {sorted(seen)} are not 1..{len(seen)}"
            )


def _mix(questions: list[Question], config: EvalConfig) -> list[tuple[str, int, int]]:
    by_type = Counter(q.type for q in questions)
    rows = [(t.value, by_type[t], config.mix_min.get(t, 0)) for t in QuestionType]
    rows.append(
        (
            "duplicate_sensitive",
            sum(q.duplicate_sensitive for q in questions),
            config.duplicate_sensitive_min,
        )
    )
    conversations = {q.conversation_id for q in questions if q.conversation_id}
    rows.append(("conversations", len(conversations), config.conversations_min))
    return rows


def _one_line(exc: Exception) -> str:
    return " ".join(str(exc).split())


def render(report: Report, path: Path) -> str:
    lines: list[str] = []
    if report.missing_file:
        lines.append(f"No questions yet: {path} doesn't exist. Targets:")
    else:
        lines.append(f"{report.n_questions} questions in {path}")
    lines.append(f"  {'':<20} {'have':>5} {'min':>5}")
    for what, have, minimum in report.mix:
        mark = "" if have >= minimum else "  <"
        lines.append(f"  {what:<20} {have:>5} {minimum:>5}{mark}")
    for title, items in (("Errors", report.errors), ("Warnings", report.warnings)):
        if items:
            lines.append(f"\n{title} ({len(items)}):")
            lines.extend(f"  - {item}" for item in items)
    if not report.missing_file:
        lines.append("\nOK" if report.ok else "\nFAILED")
    return "\n".join(lines)
