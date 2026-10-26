"""Reciprocal Rank Fusion (D-030; Cormack, Clarke and Büttcher, 2009).

Each retriever contributes 1 / (k + rank) for every item it ranks. Only
ranks are used, so BM25's unbounded scores and cosine similarities never
need to be put on one scale. k flattens the curve: with k = 1 a single
first place (1/2) outweighs being 3rd in both lists (2 x 1/4 = 1/2, a tie);
with k = 60 agreement wins (2/63 vs 1/61).
"""


def rrf(rankings: dict[str, list[int]], k: float) -> list[tuple[int, float]]:
    """Fuse ranked lists of item ids (best first) into (id, score), best first.

    Ties go to the item with the better best single rank, then the lower id
    (corpus position), so fusion is deterministic.
    """
    scores: dict[int, float] = {}
    best_rank: dict[int, int] = {}
    for ranked in rankings.values():
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            best_rank[item] = min(best_rank.get(item, rank), rank)
    order = sorted(scores, key=lambda item: (-scores[item], best_rank[item], item))
    return [(item, scores[item]) for item in order]
