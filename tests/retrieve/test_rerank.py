import numpy as np

from rag.retrieve.rerank import rerank
from rag.retrieve.retriever import Retriever
from tests.retrieve.conftest import FakeEncoder
from tests.retrieve.test_retriever import built, config  # noqa: F401  (fixtures)


class LengthScorer:
    """Scores a passage by its length: an easy order to predict."""

    def __init__(self) -> None:
        self.seen: list[list[str]] = []

    def score(self, query: str, passages) -> np.ndarray:
        self.seen.append(list(passages))
        return np.array([len(p) for p in passages], dtype=np.float32)


def test_only_the_top_depth_is_reordered() -> None:
    texts = ["aaaa", "a", "aaaaaaa", "aa"]
    ranked = [(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6)]
    out = rerank(ranked, "q", texts, LengthScorer(), depth=2)
    assert [i for i, _ in out] == [0, 1, 2, 3]  # 2 is longest but below the depth
    out = rerank(ranked, "q", texts, LengthScorer(), depth=3)
    assert [i for i, _ in out] == [2, 0, 1, 3]
    assert out[0][1] == 7.0  # the cross-encoder score replaces the fused one


def test_retriever_reranks_and_reports_the_score(built) -> None:  # noqa: F811
    scorer = LengthScorer()
    base = Retriever.load(built, {"hybrid"}, encoder=FakeEncoder())
    hits = base.with_rerank(True, scorer).retrieve("merge byte", k=3, mode="hybrid")
    assert len(scorer.seen) == 1 and all(" > " in p.split("\n")[0] for p in scorer.seen[0])
    lengths = [len(h.chunk.header) + 1 + len(h.chunk.text) for h in hits]
    assert lengths == sorted(lengths, reverse=True)
    assert all("rerank" in h.scores for h in hits)
    assert "rerank" not in base.retrieve("merge byte", k=3, mode="hybrid")[0].scores
