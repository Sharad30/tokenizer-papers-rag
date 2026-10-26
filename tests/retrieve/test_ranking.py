import numpy as np

from rag.retrieve.ranking import top_k


def test_best_first_ties_by_position() -> None:
    scores = np.array([0.5, 0.9, 0.5, 0.9, 0.1])
    assert top_k(scores, 4).tolist() == [1, 3, 0, 2]


def test_allowed_mask_and_positive_only() -> None:
    scores = np.array([0.0, 2.0, 1.0, 3.0])
    allowed = np.array([True, True, True, False])
    assert top_k(scores, 10, allowed, positive_only=True).tolist() == [1, 2]


def test_k_larger_than_corpus() -> None:
    assert top_k(np.array([1.0, 2.0]), 10).tolist() == [1, 0]
