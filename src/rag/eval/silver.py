"""`rag eval silver`: an LLM-generated question set, kept apart from the gold set (D-044).

Sharad's hand-written questions (eval/questions.jsonl) are the eval set.
Until they exist, this gives provisional numbers: sample one chunk per
paper, ask the local LLM for a question the chunk answers and the sentence
that answers it, keep the question only if that sentence really is in the
chunk. The gold for each question is that sentence.

Known biases, reported with every number:
- incomplete judgements: one gold chunk per question, though other papers
  may answer it too, so recall is underestimated for every system;
- single-paper lookups only: no multi-paper, follow-up or unanswerable
  questions;
- wording drawn from the passage favours BM25. Each question records the
  share of its content words found in the passage, so this can be checked.
"""

import json
import logging
import random
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from rag.chat.llm import LLM, LLMError, Message
from rag.chat.prompts import Prompt
from rag.common.io import write_json, write_jsonl
from rag.common.schemas import StrictModel
from rag.eval.config import SilverConfig
from rag.eval.schemas import GoldSpan, Origin, Question, QuestionType
from rag.eval.spans import ChunkTexts, map_span
from rag.parse.chunker import HEADER_SEPARATOR
from rag.parse.schemas import Chunk

log = logging.getLogger(__name__)

WORD = re.compile(r"[a-z][a-z0-9\-]{2,}")
STOPWORDS = frozenset(
    "the and for are with that this what which how does from into than then their there these "
    "those was were has have had been being its can could would should will about over under "
    "between when where why who whom whose not but any all each other such only also more most "
    "used use using based".split()
)
JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
# A question that points at the passage it came from can't be asked of a
# search engine: "the first listed metric", "according to the table".
SOURCE_REFERENCE = re.compile(
    r"\b(?:the passage|this (?:paper|study|work)|the authors|the table|the figure"
    r"|the results|according to|mentioned|listed|above|below)\b",
    re.IGNORECASE,
)


class SilverMeta(StrictModel):
    created_at: datetime
    model: str
    prompt_sha256: str
    seed: int
    sampled: int
    kept: int
    rejected: dict[str, int]  # reason -> count


def content_words(text: str) -> set[str]:
    return {w for w in WORD.findall(text.lower()) if w not in STOPWORDS}


def word_overlap(question: str, passage: str) -> float:
    """Share of the question's content words that appear in the passage."""
    words = content_words(question)
    return len(words & content_words(passage)) / len(words) if words else 0.0


def sample_chunks(
    chunks: Sequence[Chunk],
    canonical: set[str],
    exclude: set[str],
    config: SilverConfig,
) -> list[Chunk]:
    """One eligible chunk from each of `config.n` papers, reproducibly."""
    eligible: dict[str, list[Chunk]] = {}
    for c in chunks:
        if (
            c.record_id in canonical
            and c.record_id not in exclude
            and c.element_type in config.elements
            and c.token_count >= config.min_tokens
        ):
            eligible.setdefault(c.record_id, []).append(c)
    rng = random.Random(config.seed)
    papers = rng.sample(sorted(eligible), min(config.n, len(eligible)))
    return [rng.choice(eligible[p]) for p in papers]


def parse_reply(text: str) -> tuple[str, str] | None:
    """(question, evidence) from the model's JSON reply, or None."""
    match = JSON_OBJECT.search(text)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    question, evidence = data.get("question"), data.get("evidence")
    if not isinstance(question, str) or not isinstance(evidence, str):
        return None
    return question.strip(), evidence.strip()


def make_question(
    n: int, chunk: Chunk, reply: str, texts: ChunkTexts, config: SilverConfig
) -> tuple[Question | None, str]:
    """A silver question from one reply, or None and why it was rejected."""
    parsed = parse_reply(reply)
    if parsed is None:
        return None, "unparseable"
    question, evidence = parsed
    if not question.endswith("?") or len(question.split()) < config.min_question_words:
        return None, "not_a_question"
    if SOURCE_REFERENCE.search(question):
        return None, "refers_to_source"
    if not evidence or len(evidence.split()) > config.max_evidence_words:
        return None, "evidence_length"
    gold = GoldSpan(record_id=chunk.record_id, span=evidence)
    if chunk.chunk_id not in map_span(gold, texts):
        return None, "evidence_not_in_chunk"
    overlap = word_overlap(question, chunk.text)
    return (
        Question(
            id=f"s{n:03d}",
            type=QuestionType.LOOKUP,
            question=question,
            answerable=True,
            gold=[gold],
            origin=Origin.SILVER_LLM,
            notes=json.dumps({"chunk_id": chunk.chunk_id, "word_overlap": round(overlap, 3)}),
        ),
        "kept",
    )


def generate(
    chunks: Sequence[Chunk],
    llm: LLM,
    prompt: Prompt,
    texts: ChunkTexts,
    config: SilverConfig,
) -> tuple[list[Question], dict[str, int]]:
    questions: list[Question] = []
    reasons: dict[str, int] = {}
    for chunk in chunks:
        title = chunk.header.split(HEADER_SEPARATOR, 1)[0]
        message = f"Title: {title}\n\nPassage:\n{chunk.text}"
        try:
            reply = llm.complete(prompt.text, [Message("user", message)], config.max_tokens).text
        except LLMError as exc:
            log.warning("%s: %s", chunk.chunk_id, exc)
            reasons["llm_error"] = reasons.get("llm_error", 0) + 1
            continue
        question, reason = make_question(len(questions) + 1, chunk, reply, texts, config)
        if question is None:
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        questions.append(question)
        log.info("%s: %s", question.id, question.question)
    return questions, reasons


def write(
    path: Path,
    questions: Sequence[Question],
    reasons: dict[str, int],
    sampled: int,
    llm: LLM,
    prompt: Prompt,
    seed: int,
) -> None:
    write_jsonl(path, questions)
    write_json(
        path.with_name("meta.json"),
        SilverMeta(
            created_at=datetime.now(UTC),
            model=llm.model,
            prompt_sha256=prompt.sha256,
            seed=seed,
            sampled=sampled,
            kept=len(questions),
            rejected=reasons,
        ),
    )
