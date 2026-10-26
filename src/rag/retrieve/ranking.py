"""Turning a score per chunk into a ranked list: shared by BM25 and dense."""

import numpy as np


def top_k(scores: np.ndarray, k: int, allowed: np.ndarray | None = None, positive_only=False):
    """Indices of the k best scores, best first; ties broken by corpus position.

    A stable sort on the negated scores keeps equal scores in index order, so
    the same query always gives the same ranking. `allowed` is a boolean mask
    (filters); `positive_only` drops chunks with score 0, which for BM25 means
    "no query term at all", not a weak match.
    """
    keep = np.ones(len(scores), dtype=bool) if allowed is None else allowed.copy()
    if positive_only:
        keep &= scores > 0
    candidates = np.flatnonzero(keep)
    order = np.argsort(-scores[candidates], kind="stable")[:k]
    return candidates[order]
