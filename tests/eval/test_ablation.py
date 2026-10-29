import pytest

from rag.eval.ablation import read_queries


def test_queries_skip_comments_and_blank_lines(tmp_path) -> None:
    path = tmp_path / "q.txt"
    path.write_text("# header\nWhat is BPE?\n\n  How does WordPiece work?  \n# end\n")
    assert read_queries(path) == ["What is BPE?", "How does WordPiece work?"]


def test_shipped_ablation_conditions_are_valid(repo_config_path) -> None:
    from rag.common.config import load_config

    ablation = load_config(repo_config_path).eval.ablations["dedup"]
    names = [c.name for c in ablation.conditions]
    assert names[0] == "none" and not ablation.conditions[0].dedup.enabled
    assert len(set(names)) == len(names)


def _set_result(per_question: dict[str, float]):
    from rag.eval.ablation import SetResult
    from rag.eval.results import MetricSummary, ModeResult, QuestionResult

    rows = [
        QuestionResult(
            question_id=q,
            type="lookup",
            variant="raw",
            query=q,
            scored=True,
            metrics={"recall@5": r, "mrr@10": r},
        )
        for q, r in per_question.items()
    ]
    mean = sum(per_question.values()) / len(per_question)
    summary = MetricSummary(mean=mean, ci_low=mean, ci_high=mean, n=len(rows))
    return SetResult(
        file="f",
        sha256="x",
        n_questions=len(rows),
        result=ModeResult(
            aggregates={"overall": {"recall@5": summary, "mrr@10": summary}}, questions=rows
        ),
    )


def _condition(label: str, per_question: dict[str, float]):
    from rag.eval.ablation import ConditionResult

    return ConditionResult(
        label=label,
        steps=[],
        dedup={},
        dense_model=None,
        mode="hybrid",
        rerank=False,
        query_prompt=None,
        probe={},
        diff_vs_first={},
        queries=[],
        sets={"silver": _set_result(per_question)},
    )


def test_comparison_takes_the_two_best_and_counts_wins(repo_config_path) -> None:
    from rag.common.config import load_config
    from rag.eval.ablation import compare

    config = load_config(repo_config_path)
    results = {
        "weak": _condition("weak", {"q1": 0.0, "q2": 0.0, "q3": 0.0}),
        "best": _condition("best", {"q1": 1.0, "q2": 1.0, "q3": 0.0}),
        "second": _condition("second", {"q1": 1.0, "q2": 0.0, "q3": 1.0}),
    }
    cmp = compare(results, config)
    assert (cmp.set, cmp.a, cmp.b) == ("silver", "best", "second")
    assert (cmp.wins, cmp.losses, cmp.ties) == (1, 1, 1)
    assert cmp.diffs["recall@5"].mean == 0.0


def test_timed_retriever_adds_encoding_time_measured_elsewhere() -> None:
    from rag.eval.ablation import Timed, percentiles

    class Instant:
        def retrieve(self, query, k=None, mode=None):
            return []

    timed = Timed(Instant(), {"q": 25.0})
    timed.retrieve("q")
    timed.retrieve("other")
    assert timed.ms[0] >= 25.0 and timed.ms[1] < 25.0
    assert percentiles([10.0, 20.0, 30.0]) == {"p50": 20.0, "p95": 29.0, "n": 3.0}
    assert percentiles([]) == {}


def test_paired_compares_the_named_conditions(repo_config_path) -> None:
    from rag.common.config import load_config
    from rag.eval.ablation import paired

    config = load_config(repo_config_path)
    results = {
        "a": _condition("a", {"q1": 1.0, "q2": 1.0, "q3": 0.0}),
        "b": _condition("b", {"q1": 0.0, "q2": 1.0, "q3": 0.0}),
    }
    cmp = paired(results, config, "a", "b")
    assert (cmp.a, cmp.b, cmp.wins, cmp.losses, cmp.ties) == ("a", "b", 1, 0, 2)
    assert abs(cmp.diffs["recall@5"].mean - 1 / 3) < 1e-9


def test_pairs_must_name_conditions() -> None:
    from pydantic import ValidationError

    from rag.eval.config import AblationConfig

    condition = {"name": "x", "label": "X"}
    AblationConfig(queries_file="q.txt", conditions=[condition], pairs=[("x", "x")])
    with pytest.raises(ValidationError, match="unknown conditions"):
        AblationConfig(queries_file="q.txt", conditions=[condition], pairs=[("x", "y")])
