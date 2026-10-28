import numpy as np

from rag.collect.schemas import ManifestRecord, Source
from rag.dedup.config import PairSamplingConfig
from rag.dedup.pairs import candidates, sample_pairs, score
from rag.dedup.schemas import Edge, Work
from rag.eval.schemas import LabelledPair

SIG = {name: np.arange(8, dtype=np.uint64) + i for i, name in enumerate("abcdefg")}


def rec(rid: str, title: str, authors: list[str]) -> ManifestRecord:
    return ManifestRecord(record_id=rid, source=Source.ARXIV, title=title, authors=authors)


def work(members: list[str], edges: list[Edge] = ()) -> Work:
    return Work(
        work_id=f"work:{members[0]}",
        canonical=members[0],
        canonical_rule="r",
        members=members,
        aliases=members[1:],
        edges=list(edges),
        rejected=[],
    )


def test_each_band_draws_from_its_own_pool() -> None:
    records = [
        rec("a", "Paper A", ["X Ann"]),
        rec("b", "Paper A", ["X Ann"]),  # ID-linked to a
        rec("c", "Paper C", ["Y Bo"]),
        rec("d", "Paper C revised", ["Y Bo"]),  # fuzzy-merged with c
        rec("e", "Hard attention", ["Z Cy"]),
        rec("f", "Exact hard attention", ["Z Cy"]),  # same authors, close title, apart
        rec("g", "Something on fertility", ["W Di"]),
    ]
    works = [
        work(["a", "b"], [Edge(a="a", b="b", step="ids", reason="arxiv")]),
        work(["c", "d"], [Edge(a="c", b="d", step="title", reason="title")]),
        work(["e"]),
        work(["f"]),
        work(["g"]),
    ]
    vectors = {"a": np.array([1.0, 0.0]), "g": np.array([0.99, 0.14]), "e": np.array([0.0, 1.0])}
    config = PairSamplingConfig(
        seed=0,
        candidates_file="c.jsonl",
        per_band={"id_linked": 5, "fuzzy_merged": 5, "near_miss": 5, "topical": 1},
        near_miss_title_min=75,
        near_miss_jaccard_min=0.1,
        topical_pool=1,  # only the most similar pair
    )
    chosen = sample_pairs(records, works, SIG, vectors, 0.5, config)
    assert chosen == {
        ("a", "b"): "id_linked",
        ("c", "d"): "fuzzy_merged",
        ("e", "f"): "near_miss",
        ("a", "g"): "topical",  # most similar abstracts, different authors
    }


def test_candidates_show_evidence_but_not_the_band() -> None:
    by_id = {"a": rec("a", "Paper A", ["X Ann"]), "b": rec("b", "Paper A", ["X Ann"])}
    (row,) = candidates([("a", "b")], by_id, SIG, seed=0)
    assert row.title_ratio == 100.0 and row.author_overlap == 1.0
    assert "band" not in row.model_dump()


def test_score_counts_agreement_with_the_labels() -> None:
    record_work = {"a": "w1", "b": "w1", "c": "w2", "d": "w3"}
    labels = [
        LabelledPair(a="a", b="b", same=True, kind="version"),  # joined, same: TP
        LabelledPair(a="c", b="d", same=True, kind="venue_copy"),  # apart, same: FN
        LabelledPair(a="a", b="c", same=False, kind="follow_up"),  # apart, different: TN
    ]
    counts, by_band, errors = score(labels, record_work, {"c|d": "near_miss"})
    assert counts == {"tp": 1, "fp": 0, "fn": 1, "tn": 1}
    assert by_band["near_miss"]["fn"] == 1
    assert [(e.a, e.b, e.band) for e in errors] == [("c", "d", "near_miss")]
