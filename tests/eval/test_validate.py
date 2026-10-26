import json
from pathlib import Path

import pytest

from rag.eval.config import EvalConfig
from rag.eval.spans import ChunkTexts
from rag.eval.validate import load_questions, render, validate, validate_file

FIXTURE = Path(__file__).parents[1] / "fixtures" / "eval" / "questions.jsonl"
INDEXED = {"p:a": True, "p:a-copy": True, "p:b": True, "p:noflag": False}


@pytest.fixture
def eval_config() -> EvalConfig:
    return EvalConfig(
        questions_file="questions.jsonl",
        mix_min={"lookup": 1, "multi_paper": 1, "followup": 1, "unanswerable": 1},
        duplicate_sensitive_min=1,
        conversations_min=1,
        max_span_words=12,
        find={"max_hits": 3},
    )


def lines() -> list[dict]:
    return [json.loads(line) for line in FIXTURE.read_text().splitlines()]


def write(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "questions.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def run(tmp_path: Path, rows: list[dict], texts: ChunkTexts, config: EvalConfig):
    return validate_file(write(tmp_path, rows), texts, INDEXED, config)


def test_fixture_passes(texts: ChunkTexts, eval_config: EvalConfig) -> None:
    report = validate_file(FIXTURE, texts, INDEXED, eval_config)
    assert report.ok, report.errors
    assert report.warnings == []
    assert render(report, FIXTURE).endswith("OK")


def test_missing_file_is_not_a_failure(
    tmp_path: Path, texts: ChunkTexts, eval_config: EvalConfig
) -> None:
    report = validate_file(tmp_path / "none.jsonl", texts, INDEXED, eval_config)
    assert report.ok and report.missing_file
    assert ("lookup", 0, 1) in report.mix
    assert "No questions yet" in render(report, tmp_path / "none.jsonl")


def test_bad_line_reports_its_line_number(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    path.write_text('{"id": "q1"}\n\nnot json\n')
    questions, errors = load_questions(path)
    assert questions == []
    assert [e.split(":")[0] for e in errors] == ["line 1", "line 3"]


@pytest.mark.parametrize(
    ("record_id", "span", "message"),
    [
        ("p:zzz", "joint", "not in the manifest"),
        ("p:noflag", "joint", "not in the index"),
        ("p:a", "Rare words are segmented into subword units", "span not found"),
    ],
)
def test_gold_problems_are_errors(
    tmp_path: Path,
    texts: ChunkTexts,
    eval_config: EvalConfig,
    record_id: str,
    span: str,
    message: str,
) -> None:
    rows = lines()
    rows[1]["gold"] = [{"record_id": record_id, "span": span}]
    report = run(tmp_path, rows, texts, eval_config)
    assert [e for e in report.errors if message in e]


def test_unmapped_span_comes_with_a_hint(
    tmp_path: Path, texts: ChunkTexts, eval_config: EvalConfig
) -> None:
    rows = lines()
    rows[1]["gold"] = [{"record_id": "p:a", "span": "Rare words are segmented into subword units"}]
    (error,) = run(tmp_path, rows, texts, eval_config).errors
    assert "closest: p:a#0001" in error


def test_duplicate_ids_and_turn_gaps_are_errors(
    tmp_path: Path, texts: ChunkTexts, eval_config: EvalConfig
) -> None:
    rows = lines()
    rows[2]["id"] = "q1"
    rows[1]["turn"] = 3
    errors = run(tmp_path, rows, texts, eval_config).errors
    assert any("q1: id used 2 times" in e for e in errors)
    assert any("turns [1, 3] are not 1..2" in e for e in errors)


def test_mix_shortfall_is_an_error(
    tmp_path: Path, texts: ChunkTexts, eval_config: EvalConfig
) -> None:
    report = run(tmp_path, lines()[:3], texts, eval_config)  # no unanswerable
    assert report.errors == ["mix: unanswerable has 0, needs at least 1"]
    shortfall = [ln for ln in render(report, Path("q")).splitlines() if ln.endswith("<")]
    assert [ln.split() for ln in shortfall] == [["unanswerable", "0", "1", "<"]]


def test_warnings_for_long_spans_sections_and_duplicates(
    texts: ChunkTexts, eval_config: EvalConfig
) -> None:
    questions, _ = load_questions(FIXTURE)
    q1 = questions[0]
    q1.duplicate_sensitive = False  # its span is also in p:a-copy
    q1.gold[0].section = "Results"  # the span is in Intro
    q2 = questions[1]
    q2.gold[0].span = "operations. The vocabulary is joint."  # crosses a sentence
    # Clearing the flag would also break the duplicate_sensitive minimum.
    config = eval_config.model_copy(update={"duplicate_sensitive_min": 0})
    report = validate(questions, texts, INDEXED, config)
    assert report.ok, report.errors
    joined = " | ".join(report.warnings)
    assert "q1 gold 1: span also in p:a-copy" in joined
    assert "q1 gold 1: section 'Results'" in joined
    assert "q2 gold 1: long or multi-sentence span" in joined
