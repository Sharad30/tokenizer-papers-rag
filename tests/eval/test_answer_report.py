"""The answer report (D-046): simulated gate, sweep, labels, failure staging."""

from datetime import UTC, datetime

import pytest

from rag.eval import answer_report as rep
from rag.eval.answers import AnswerRecord, AnswersMeta, Claim
from rag.eval.labels import Label


def rec(qid: str, cosine: float, answerable: bool = True, **fields) -> AnswerRecord:
    base = dict(
        question_id=qid,
        type="lookup" if answerable else "unanswerable",
        answerable=answerable,
        user=f"question {qid}",
        rewrite=f"question {qid}",
        context=[],
        top_dense_cosine=cosine,
        model_abstained=False,
        completion="An answer [C1].",
        reply="An answer [1].",
        cited=["C1"],
        invalid_citations=[],
        uncited_sentences=0,
        claims=[Claim(index=0, text="An answer that cites [C1].", cites=["C1"])],
        gold_passages_in_context=1.0 if answerable else None,
        gold_papers_in_context=1.0 if answerable else None,
    )
    return AnswerRecord(**{**base, **fields})


RECORDS = [
    rec("A1", 0.85),
    rec("A2", 0.72),  # refused at 0.75 and above
    rec("A3", 0.90, model_abstained=True, claims=[]),  # the model refuses at any threshold
    rec("U1", 0.65, answerable=False, unanswerable_kind="not_covered"),
    rec("U2", 0.80, answerable=False, unanswerable_kind="false_premise"),
]


def labels(*items) -> rep.Labels:
    return rep.Labels([Label(**i) for i in items])


def test_the_simulated_gate_matches_the_engine() -> None:
    assert rep.abstains(RECORDS[0], 0.70) is None
    assert rep.abstains(RECORDS[1], 0.75) == "gate"
    assert rep.abstains(RECORDS[2], 0.0) == "model"
    assert rep.abstains(rec("X", None), 0.0) == "gate"  # no dense score at all


def test_false_premise_counts_as_handled_when_abstained_or_corrected() -> None:
    u2 = RECORDS[4]
    assert rep.handled_unanswerable(u2, 0.70, labels()) is None  # answered: needs a label
    corrected = labels({"question_id": "U2", "kind": "answer", "value": "correct"})
    assert rep.handled_unanswerable(u2, 0.70, corrected) is True
    assert rep.handled_unanswerable(u2, 0.85, labels()) is True  # gate refuses it
    assert rep.handled_unanswerable(RECORDS[3], 0.60, labels()) is False  # answered, not covered


def test_sweep_by_hand() -> None:
    judged = labels(
        {"question_id": "A1", "kind": "answer", "value": "correct"},
        {"question_id": "A2", "kind": "claim", "claim": 0, "value": "unsupported"},
        {"question_id": "U2", "kind": "answer", "value": "wrong"},
    )
    at_70, at_75 = rep.sweep(RECORDS, [0.70, 0.75], judged)
    # 0.70: answered A1, A2, U2 (A3 model, U1 gate) -> coverage 3/5
    assert at_70.coverage == pytest.approx(3 / 5)
    assert at_70.false_abstention == pytest.approx(1 / 3)  # A3
    assert at_70.unanswerable_handled == pytest.approx(1 / 2)  # U1 yes, U2 answered wrongly
    assert (at_70.risk, at_70.risk_n) == (pytest.approx(1 / 2), 2)  # A2 bad, A1 good
    # 0.75: A2 refused too -> answered A1, U2
    assert at_75.coverage == pytest.approx(2 / 5)
    assert at_75.false_abstention == pytest.approx(2 / 3)
    assert (at_75.risk, at_75.risk_n) == (0.0, 1)


def test_failures_name_a_suggested_stage_with_evidence() -> None:
    judged = labels(
        {"question_id": "A1", "kind": "answer", "value": "wrong"},
        {"question_id": "U2", "kind": "answer", "value": "wrong"},
    )
    no_gold = rec("A4", 0.80, gold_passages_in_context=0.0, gold_papers_in_context=0.0)
    found = {f.question_id: f for f in rep.failures([*RECORDS, no_gold], 0.75, judged)}
    assert found["A2"].stage == "gate" and "false abstention (gate)" in found["A2"].what
    assert found["A3"].what == "false abstention (model)"
    assert found["A1"].stage == "generation" and found["A1"].what == "wrong answer"
    assert found["U2"].what == "missed abstention"
    assert "A4" not in found  # nothing judged wrong yet
    judged.answer["A4"] = "partial"
    assert rep.failures([no_gold], 0.75, judged)[0].stage == "retrieval"
    assert found["A1"].inspect == 'uv run rag inspect query "question A1"'


def meta() -> AnswersMeta:
    return AnswersMeta(
        created_at=datetime.now(UTC),
        git_commit="abc123",
        config_hash="h",
        model="qwen3:8b",
        prompts={},
        questions_file="eval/gold/questions.jsonl",
        questions_sha256="s",
        n_questions=5,
        deployed_threshold=0.70,
    )


def test_report_without_labels_gives_code_only_numbers() -> None:
    report = rep.build(RECORDS, meta(), "a.jsonl", labels(), [0.7], (200, 0))
    assert not report.labelled and report.claims == {} and report.correct is None
    assert report.abstention["unanswerable awaiting a label"].count == 1  # U2
    assert report.gates == {"model": 1, "gate": 1}
    text = rep.markdown(report)
    assert "No labels yet" in text and "can't tune a threshold" in text


def test_report_with_labels_counts_claims_and_correctness() -> None:
    judged = labels(
        {"question_id": "A1", "kind": "claim", "claim": 0, "value": "supported"},
        {"question_id": "A2", "kind": "claim", "claim": 0, "value": "partial"},
        {"question_id": "A1", "kind": "answer", "value": "correct"},
        {"question_id": "A2", "kind": "answer", "value": "wrong"},
    )
    report = rep.build(RECORDS, meta(), "a.jsonl", judged, [0.7], (200, 0))
    assert report.claims["supported"].count == 1 and report.claims["partial"].count == 1
    assert report.correct.mean == pytest.approx(0.5) and report.correct.n == 2


def test_figure_renders_with_and_without_labels(tmp_path) -> None:
    for judged in (labels(), labels({"question_id": "A1", "kind": "answer", "value": "correct"})):
        report = rep.build(RECORDS, meta(), "a.jsonl", judged, [0.6, 0.7, 0.8], (200, 0))
        path = tmp_path / f"fig-{report.labelled}.png"
        rep.plot(report, path)
        assert path.stat().st_size > 0
