import numpy as np
import pytest

from rag.dedup.schemas import Restatement, Work
from rag.retrieve.collapse import WorkMap, cap_per_work, demote_restatements, mmr
from rag.retrieve.config import DedupSettings
from rag.retrieve.retriever import Retriever
from tests.retrieve.conftest import FakeEncoder, make_chunk
from tests.retrieve.test_retriever import built, config  # noqa: F401  (fixtures)

# Two records of one paper (ACL copy canonical, arXiv copy an alias), a survey, another paper.
CHUNKS = [
    make_chunk("acl:bpe", 0, "bpe"),
    make_chunk("acl:bpe", 1, "bpe more"),
    make_chunk("arxiv:bpe", 0, "bpe preprint"),
    make_chunk("arxiv:survey", 0, "survey says bpe"),
    make_chunk("acl:other", 0, "unigram"),
]
WORKS = WorkMap.from_works(
    [
        Work(
            work_id="work:acl:bpe",
            canonical="acl:bpe",
            canonical_rule="venue",
            members=["acl:bpe", "arxiv:bpe"],
            aliases=["arxiv:bpe"],
            edges=[],
            rejected=[],
        ),
        Work(
            work_id="work:arxiv:survey",
            canonical="arxiv:survey",
            canonical_rule="only member",
            members=["arxiv:survey"],
            aliases=[],
            edges=[],
            rejected=[],
        ),
        Work(
            work_id="work:acl:other",
            canonical="acl:other",
            canonical_rule="only member",
            members=["acl:other"],
            aliases=[],
            edges=[],
            rejected=[],
        ),
    ],
    [
        Restatement(
            chunk_id="arxiv:survey#0000",
            work_id="work:acl:bpe",
            cosine=0.9,
            ref_id="b0",
            matched_by="title",
        )
    ],
)


def positions(ranked) -> list[int]:
    return [i for i, _ in ranked]


def test_canonical_mask_hides_aliases() -> None:
    assert WORKS.canonical_mask(CHUNKS).tolist() == [True, True, False, True, True]


def test_cap_keeps_the_best_chunks_per_work_and_records_the_hidden_ones() -> None:
    ranked = [(0, 0.9), (2, 0.8), (1, 0.7), (4, 0.6)]
    kept, hidden = cap_per_work(ranked, CHUNKS, WORKS, cap=1)
    assert positions(kept) == [0, 4]
    assert hidden == {0: [2, 1]}
    kept2, _ = cap_per_work(ranked, CHUNKS, WORKS, cap=2)
    assert positions(kept2) == [0, 2, 4]


def test_survey_chunk_moves_just_below_the_work_it_restates() -> None:
    ranked = [(3, 0.9), (4, 0.8), (0, 0.7), (1, 0.6)]
    assert positions(demote_restatements(ranked, CHUNKS, WORKS)) == [4, 0, 3, 1]


def test_survey_chunk_stays_when_the_restated_work_is_not_a_candidate() -> None:
    ranked = [(3, 0.9), (4, 0.8)]
    assert positions(demote_restatements(ranked, CHUNKS, WORKS)) == [3, 4]


def test_mmr_trades_relevance_for_novelty() -> None:
    # Candidates 0 and 1 are near-identical; 2 is different but a little less relevant.
    vectors = np.array([[1.0, 0.0], [0.99, 0.141], [0.0, 1.0]], dtype=np.float32)
    ranked = [(0, 1.0), (1, 0.95), (2, 0.9)]
    assert positions(mmr(ranked, vectors, lam=1.0, k=3)) == [0, 1, 2]  # relevance only
    assert positions(mmr(ranked, vectors, lam=0.5, k=3)) == [0, 2, 1]


def test_retriever_hides_alias_chunks_and_caps_per_work(built) -> None:  # noqa: F811
    base = Retriever.load(built, {"hybrid"}, encoder=FakeEncoder())
    works = WorkMap.from_works(
        [
            Work(
                work_id="work:p:a",
                canonical="p:a",
                canonical_rule="r",
                members=["p:a", "p:c"],
                aliases=["p:c"],
                edges=[],
                rejected=[],
            ),
            Work(
                work_id="work:p:b",
                canonical="p:b",
                canonical_rule="r",
                members=["p:b"],
                aliases=[],
                edges=[],
                rejected=[],
            ),
        ]
    )
    settings = DedupSettings(
        enabled=True, canonical_only=True, per_work_cap=1, survey_demotion=True
    )
    hits = base.with_dedup(settings, works).retrieve("merge byte", k=3, mode="hybrid")
    assert [h.chunk.record_id for h in hits] == ["p:a", "p:b"]  # p:c is an alias of p:a
    assert hits[0].work_id == "work:p:a" and hits[0].hidden == ["p:a#0001"]


def test_dedup_enabled_without_works_is_refused(built) -> None:  # noqa: F811
    base = Retriever.load(built, {"hybrid"}, encoder=FakeEncoder())
    settings = DedupSettings(
        enabled=True, canonical_only=True, per_work_cap=1, survey_demotion=False
    )
    with pytest.raises(RuntimeError, match="rag dedup run"):
        base.with_dedup(settings, None)
