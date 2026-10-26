import pytest
from pydantic import ValidationError

from rag.eval.schemas import Question

GOLD_A = {"record_id": "p:a", "span": "merges"}
GOLD_B = {"record_id": "p:b", "span": "bytes"}

LOOKUP = {
    "id": "q1",
    "type": "lookup",
    "question": "How many merges?",
    "answerable": True,
    "gold": [GOLD_A],
}


def make(**change) -> Question:
    return Question.model_validate({**LOOKUP, **change})


def test_valid_lookup() -> None:
    assert make().gold[0].record_id == "p:a"


def test_valid_unanswerable() -> None:
    q = make(type="unanswerable", answerable=False, gold=[], unanswerable_kind="false_premise")
    assert q.unanswerable_kind == "false_premise"


def test_valid_followup() -> None:
    q = make(type="followup", conversation_id="c1", turn=2, oracle_rewrite="How many BPE merges?")
    assert q.turn == 2


def test_valid_multi_paper() -> None:
    assert len(make(type="multi_paper", gold=[GOLD_A, GOLD_B]).gold) == 2


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"gold": []}, "no gold"),
        ({"answerable": False, "unanswerable_kind": "not_covered"}, "has gold"),
        ({"unanswerable_kind": "not_covered"}, "unanswerable_kind"),
        ({"type": "unanswerable", "answerable": True}, "type unanswerable"),
        ({"type": "followup", "conversation_id": "c1", "turn": 2}, "oracle_rewrite"),
        (
            {"type": "followup", "conversation_id": "c1", "turn": 1, "oracle_rewrite": "x"},
            "turn >= 2",
        ),
        ({"type": "followup", "oracle_rewrite": "x"}, "turn >= 2"),
        ({"conversation_id": "c1"}, "go together"),
        ({"conversation_id": "c1", "turn": 1, "oracle_rewrite": "x"}, "already standalone"),
        ({"type": "multi_paper", "gold": [GOLD_A, {**GOLD_A, "span": "other"}]}, "2 papers"),
        ({"surprise": 1}, "Extra inputs"),
    ],
)
def test_inconsistent_questions_are_rejected(change: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        make(**change)
