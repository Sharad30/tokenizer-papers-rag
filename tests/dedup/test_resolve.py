"""Synthetic records for each identity rule (D-040)."""

import pytest

from rag.collect.schemas import ManifestRecord, Source
from rag.dedup.config import DedupConfig
from rag.dedup.minhash import MinHasher, shingle_hashes
from rag.dedup.resolve import canonical_member, normalise, resolve

CONFIG = DedupConfig.model_validate(
    {
        "seed": 0,
        "steps": ["ids", "title", "minhash"],
        "author_overlap_min": 0.5,
        "title": {
            "min_ratio": 97,
            "review_ratio": 85,
            "review_min_jaccard": 0.2,
            "year_gap_max": 2,
        },
        "minhash": {
            "shingle_words": 1,
            "num_perm": 256,
            "bands": 64,
            "rows": 4,
            "min_jaccard": 0.5,
        },
        "cluster_floor": 0.2,
        "canonical_order": ["acl", "arxiv", "s2"],
        "surveys": {"title_cues": ["survey"], "reference_title_min": 95, "restate_min": 0.8},
    }
)
HASHER = MinHasher(256, seed=0)
AUTHORS = ["Ada Lovelace", "Alan Turing"]


def words(start: int, n: int) -> set[str]:
    """Synthetic shingle sets: words(0, 100) and words(50, 100) have Jaccard 50/150."""
    return {f"w{i}" for i in range(start, start + n)}


def rec(record_id: str, title: str, authors=AUTHORS, year=2020, **ids) -> ManifestRecord:
    source = Source(record_id.split(":")[0])
    return ManifestRecord(
        record_id=record_id, source=source, title=title, authors=authors, year=year, **ids
    )


def run(records_and_text, steps=None, config=CONFIG):
    records = [r for r, _ in records_and_text]
    sigs = {r.record_id: HASHER.signature(shingle_hashes(text)) for r, text in records_and_text}
    resolution = resolve(records, sigs, {r.record_id: 1 for r in records}, config, steps)
    return resolution.record_work, resolution


def test_versions_with_the_same_arxiv_id_merge_even_with_little_shared_text() -> None:
    work, _ = run(
        [
            (
                rec(
                    "arxiv:1508.07909v1",
                    "Neural MT of rare words",
                    arxiv_id="1508.07909",
                    arxiv_version=1,
                ),
                words(0, 100),
            ),
            (
                rec(
                    "arxiv:1508.07909v3",
                    "Neural MT of rare words with subword units",
                    arxiv_id="1508.07909",
                    arxiv_version=3,
                ),
                words(95, 100),
            ),
        ]
    )
    assert work["arxiv:1508.07909v1"] == work["arxiv:1508.07909v3"] == "work:arxiv:1508.07909v3"


def test_preprint_and_venue_copy_with_a_changed_title_merge_on_text() -> None:
    work, resolution = run(
        [
            (rec("arxiv:2012.15524v1", "Linear-Time WordPiece Tokenization"), words(0, 100)),
            (
                rec("acl:2021.emnlp-main.160", "Fast WordPiece Tokenization", year=2021),
                words(10, 100),
            ),
        ]
    )
    assert work["arxiv:2012.15524v1"] == work["acl:2021.emnlp-main.160"]
    assert work["acl:2021.emnlp-main.160"] == "work:acl:2021.emnlp-main.160"  # venue copy wins
    assert resolution.counts["edges.minhash"] == 1


def test_same_author_follow_up_with_a_close_title_stays_separate() -> None:
    # Title ratio ~92 (borderline) but almost no shared text: a follow-up paper.
    work, _ = run(
        [
            (
                rec(
                    "acl:D18-1473", "Hard Non-Monotonic Attention for Character-Level Transduction"
                ),
                words(0, 100),
            ),
            (
                rec(
                    "acl:P19-1148",
                    "Exact Hard Monotonic Attention for Character-Level Transduction",
                    year=2019,
                ),
                words(98, 100),
            ),
        ]
    )
    assert work["acl:D18-1473"] != work["acl:P19-1148"]


def test_two_papers_on_the_same_topic_by_different_authors_stay_separate() -> None:
    # Identical text overlap would be suspicious, but without shared authors it never merges.
    work, _ = run(
        [
            (rec("arxiv:2001.00001v1", "Tokenizer fertility across languages"), words(0, 100)),
            (
                rec(
                    "arxiv:2002.00002v1",
                    "Tokenizer fertility across languages",
                    authors=["Grace Hopper"],
                ),
                words(0, 100),
            ),
        ]
    )
    assert work["arxiv:2001.00001v1"] != work["arxiv:2002.00002v1"]


@pytest.mark.parametrize(("shift", "merged"), [(30, True), (70, False)])
def test_text_merge_threshold(shift: int, merged: bool) -> None:
    # Jaccard of words(0,100) and words(shift,100): 70/130 = 0.54 vs 30/170 = 0.18.
    work, _ = run(
        [
            (rec("arxiv:2001.00001v1", "Paper one"), words(0, 100)),
            (rec("acl:2020.x-1", "Something else entirely"), words(shift, 100)),
        ]
    )
    assert (work["arxiv:2001.00001v1"] == work["acl:2020.x-1"]) is merged


def test_chain_guard_refuses_a_link_between_unrelated_ends() -> None:
    # A ~ B and B ~ C by text (Jaccard 1/3 each, above a lowered merge
    # threshold), A and C share nothing: B joins only one of them.
    loose = CONFIG.model_copy(
        update={
            # 128 bands of 2 rows: a pair at Jaccard 1/3 is a candidate almost surely.
            "minhash": CONFIG.minhash.model_copy(
                update={"min_jaccard": 0.25, "bands": 128, "rows": 2}
            )
        }
    )
    a = words(0, 100)
    b = words(0, 50) | words(1000, 50)
    c = words(1000, 50) | words(2000, 50)
    work, resolution = run(
        [
            (rec("arxiv:2001.00001v1", "A"), a),
            (rec("arxiv:2001.00002v1", "B"), b),
            (rec("arxiv:2001.00003v1", "C"), c),
        ],
        config=loose,
    )
    assert work["arxiv:2001.00001v1"] != work["arxiv:2001.00003v1"]
    assert resolution.counts["edges.rejected_by_guard"] == 1


def test_switching_steps_off_leaves_records_apart() -> None:
    pairs = [
        (rec("arxiv:2001.00001v1", "Same title"), words(0, 100)),
        (rec("acl:2020.x-1", "Same title"), words(0, 100)),
    ]
    assert run(pairs, steps=["ids"])[1].counts["works"] == 2
    assert run(pairs)[1].counts["works"] == 1


def test_canonical_prefers_venue_then_latest_version_then_most_chunks() -> None:
    records = {
        r.record_id: r
        for r in [
            rec("arxiv:1v1", "t", arxiv_version=1),
            rec("arxiv:1v3", "t", arxiv_version=3),
            rec("s2:x", "t"),
            rec("acl:P1", "t"),
        ]
    }
    order = ["acl", "arxiv", "s2"]
    assert canonical_member(list(records), records, {}, order) == (
        "acl:P1",
        "venue version (ACL Anthology)",
    )
    no_venue = ["arxiv:1v1", "arxiv:1v3", "s2:x"]
    assert canonical_member(no_venue, records, {}, order)[0] == "arxiv:1v3"
    assert canonical_member(["s2:x"], records, {}, order) == ("s2:x", "only member")


def test_normalise_strips_accents_and_punctuation() -> None:
    assert normalise("BabyLM’s First Words: Ge’ez") == normalise("BabyLM's first words: Ge'ez")
    assert normalise("Jindřich Libovický") == "jindrich libovicky"
