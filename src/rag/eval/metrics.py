"""Retrieval metrics (D-031). Pure functions over ranked chunk ids.

Relevance is binary and comes from gold spans: a chunk is relevant to a
span when it contains the span (`spans.map_span`), in the gold paper or a
linked copy of it. Every function here is checked against hand-computed
examples in tests/eval/test_metrics.py.
"""

from collections.abc import Sequence

import numpy as np


def span_recall_at_k(gold: Sequence[set[str]], ranked: Sequence[str], k: int) -> float:
    """Fraction of gold spans with at least one containing chunk in the top k.

    `gold` holds, per span, the ids of the chunks that contain it. A span
    found by two chunks (sentence overlap) still counts once.
    """
    if not gold:
        raise ValueError("recall is undefined without gold spans")
    top = set(ranked[:k])
    return sum(bool(chunks & top) for chunks in gold) / len(gold)


def paper_recall_at_k(gold_papers: set[str], ranked_papers: Sequence[str], k: int) -> float:
    """Fraction of gold papers (identity groups) with any chunk in the top k."""
    if not gold_papers:
        raise ValueError("recall is undefined without gold papers")
    return len(gold_papers & set(ranked_papers[:k])) / len(gold_papers)


def reciprocal_rank(gold: Sequence[set[str]], ranked: Sequence[str], cutoff: int) -> float:
    """1 / rank of the first chunk containing any gold span; 0 if none in the top `cutoff`."""
    relevant = set().union(*gold) if gold else set()
    for rank, chunk_id in enumerate(ranked[:cutoff], start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def duplicate_rate_at_k(ranked_papers: Sequence[str], k: int) -> float:
    """Fraction of the top-k slots whose paper (identity group) already appeared higher.

    Two chunks of the same paper count as a duplicate slot too: from the
    user's side they are a second result from one source. Divided by the
    slots actually returned; 0 when nothing was returned.
    """
    top = ranked_papers[:k]
    if not top:
        return 0.0
    return (len(top) - len(set(top))) / len(top)


def copy_duplicate_rate_at_k(
    ranked_records: Sequence[str], groups: dict[str, str], k: int
) -> float:
    """Fraction of the top-k slots whose paper already appeared higher as another record.

    Unlike dup@k, a second chunk of the same record doesn't count: this
    isolates what versions and venue copies add, the part Phase 5's merging
    removes. Divided by the slots returned; 0 when nothing was returned.
    """
    top = ranked_records[:k]
    if not top:
        return 0.0
    seen: dict[str, set[str]] = {}
    copies = 0
    for record in top:
        group = groups.get(record, record)
        if seen.get(group, set()) - {record}:
            copies += 1
        seen.setdefault(group, set()).add(record)
    return copies / len(top)


# TODO(sharad): Phase 4 "You do": add recall at 10 here (span-level, like
# span_recall_at_k), a test with a hand-computed example in
# tests/eval/test_metrics.py, and report it from the harness
# (harness.score_question) and in `rag report`.


def bootstrap_ci(values: Sequence[float], samples: int, seed: int) -> tuple[float, float, float]:
    """Mean and 95% percentile-bootstrap interval, resampling questions with replacement."""
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        raise ValueError("no values")
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, data.size, size=(samples, data.size))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(data.mean()), float(low), float(high)


def paired_bootstrap_diff(
    a: Sequence[float], b: Sequence[float], samples: int, seed: int
) -> tuple[float, float, float]:
    """Mean of (a - b) per question and its 95% interval.

    Paired: each resample keeps a question's two scores together, so the
    question-to-question variation that affects both systems cancels out.
    If the interval excludes 0, the difference is unlikely to be noise.
    """
    diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if diffs.size == 0:
        raise ValueError("no paired values")
    return bootstrap_ci(diffs, samples, seed)
