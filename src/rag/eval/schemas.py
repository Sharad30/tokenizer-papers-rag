"""The eval question format (`eval/questions.jsonl`), one Question per line.

Rules that concern a single question live in the model validator, so a bad
line fails as soon as it is read. Rules across questions (unique ids,
conversation turns, the type mix) are checked by `rag eval validate`.
"""

from enum import StrEnum

from pydantic import Field, model_validator

from rag.common.schemas import StrictModel


class QuestionType(StrEnum):
    LOOKUP = "lookup"  # one fact from one paper (often a table)
    MULTI_PAPER = "multi_paper"  # needs evidence from two or more papers
    FOLLOWUP = "followup"  # turn > 1 of a conversation; makes sense only with its history
    UNANSWERABLE = "unanswerable"  # in-domain, but the corpus doesn't support an answer


class UnanswerableKind(StrEnum):
    NOT_COVERED = "not_covered"  # on topic, but no paper answers it
    FALSE_PREMISE = "false_premise"  # assumes something the papers contradict
    OUTSIDE_CORPUS = "outside_corpus"  # about a real paper we didn't collect
    NONEXISTENT_DETAIL = "nonexistent_detail"  # asks for a number/result no paper reports


class Origin(StrEnum):
    HAND = "hand"
    # An LLM drafted it and Sharad verified and rewrote it; the write-up says so.
    LLM_ASSISTED = "llm_assisted"


class GoldSpan(StrictModel):
    """Evidence for an answer: verbatim text in one paper (D-022).

    Never a chunk id: chunk ids change whenever chunking does, the span doesn't.
    """

    record_id: str  # manifest id, e.g. "acl:P16-1162" or "arxiv:2105.13626"
    section: str | None = None  # for humans; checked against the chunk's section path
    span: str = Field(min_length=1)  # copied from chunk text (`rag eval find`)


class Question(StrictModel):
    id: str = Field(min_length=1)  # "q001"
    type: QuestionType
    question: str = Field(min_length=1)
    conversation_id: str | None = None
    turn: int | None = Field(default=None, ge=1)
    # The standalone question a perfect rewriter would produce; scores the
    # rewriter separately from retrieval (Phase 4 "oracle" follow-up scoring).
    oracle_rewrite: str | None = None
    answerable: bool
    unanswerable_kind: UnanswerableKind | None = None
    gold: list[GoldSpan] = []
    # True when the answer is restated in other papers or copies (arXiv vs ACL),
    # so a retriever can fill its top-5 with duplicates (Phase 5).
    duplicate_sensitive: bool = False
    restatement_notes: str | None = None
    origin: Origin = Origin.HAND
    notes: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> "Question":
        problems: list[str] = []
        unanswerable = self.type == QuestionType.UNANSWERABLE
        if unanswerable and self.answerable:
            problems.append("type unanswerable but answerable=true")
        if self.answerable and not self.gold:
            problems.append("answerable but no gold spans")
        if not self.answerable and self.gold:
            problems.append("not answerable but has gold spans")
        if (self.unanswerable_kind is None) == (not self.answerable):
            problems.append("unanswerable_kind is required exactly when answerable=false")
        if (self.conversation_id is None) != (self.turn is None):
            problems.append("conversation_id and turn go together")
        if self.type == QuestionType.FOLLOWUP:
            if self.turn is None or self.turn < 2:
                problems.append("a followup needs a conversation and turn >= 2")
            if not self.oracle_rewrite:
                problems.append("a followup needs an oracle_rewrite")
        elif self.oracle_rewrite and self.turn == 1:
            problems.append("turn 1 is already standalone: no oracle_rewrite")
        if self.type == QuestionType.MULTI_PAPER and len({g.record_id for g in self.gold}) < 2:
            problems.append("multi_paper gold must cover at least 2 papers")
        if problems:
            raise ValueError("; ".join(problems))
        return self
