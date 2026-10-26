import pytest

from rag.eval.metrics import (
    bootstrap_ci,
    duplicate_rate_at_k,
    paired_bootstrap_diff,
    paper_recall_at_k,
    reciprocal_rank,
    span_recall_at_k,
)

RANKED = ["c1", "c2", "c3", "c4", "c5", "c6"]


def test_recall_counts_spans_not_chunks() -> None:
    # Span A is in c2 and c9 (overlap copy), span B in c6 only (rank 6).
    gold = [{"c2", "c9"}, {"c6"}]
    assert span_recall_at_k(gold, RANKED, k=5) == 0.5
    assert span_recall_at_k(gold, RANKED, k=6) == 1.0


def test_recall_with_no_relevant_result() -> None:
    assert span_recall_at_k([{"x"}], RANKED, k=5) == 0.0


def test_recall_needs_gold() -> None:
    with pytest.raises(ValueError):
        span_recall_at_k([], RANKED, k=5)


def test_span_found_by_two_chunks_counts_once() -> None:
    assert span_recall_at_k([{"c1", "c2"}], RANKED, k=5) == 1.0


def test_paper_recall() -> None:
    papers = ["p1", "p1", "p2", "p3", "p1"]
    assert paper_recall_at_k({"p2", "p4"}, papers, k=5) == 0.5


def test_reciprocal_rank_uses_first_relevant_chunk() -> None:
    # Two spans: the first relevant chunk overall is at rank 3.
    assert reciprocal_rank([{"c5"}, {"c3"}], RANKED, cutoff=10) == pytest.approx(1 / 3)


def test_reciprocal_rank_zero_beyond_cutoff() -> None:
    assert reciprocal_rank([{"c6"}], RANKED, cutoff=5) == 0.0


def test_duplicate_rate() -> None:
    # p1 at 1, 2 and 5: slots 2 and 5 are redundant -> 2/5.
    assert duplicate_rate_at_k(["p1", "p1", "p2", "p3", "p1", "p9"], k=5) == pytest.approx(0.4)


def test_duplicate_rate_short_and_empty_lists() -> None:
    assert duplicate_rate_at_k(["p1", "p1"], k=5) == 0.5
    assert duplicate_rate_at_k([], k=5) == 0.0


def test_bootstrap_is_deterministic_and_brackets_the_mean() -> None:
    values = [0, 1, 1, 0, 1, 1, 1, 0]
    mean, low, high = bootstrap_ci(values, samples=1000, seed=0)
    assert mean == 0.625
    assert low <= mean <= high
    assert bootstrap_ci(values, samples=1000, seed=0) == (mean, low, high)


def test_bootstrap_of_constant_values_has_zero_width() -> None:
    assert bootstrap_ci([0.5, 0.5, 0.5], samples=200, seed=1) == (0.5, 0.5, 0.5)


def test_paired_diff_cancels_shared_variation() -> None:
    # b is a + 0.1 on every question: every paired resample differs by exactly 0.1.
    a = [0.1, 0.9, 0.4, 0.7, 0.2]
    b = [x + 0.1 for x in a]
    mean, low, high = paired_bootstrap_diff(b, a, samples=500, seed=0)
    assert (mean, low, high) == pytest.approx((0.1, 0.1, 0.1))
