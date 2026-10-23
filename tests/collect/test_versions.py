from rag.collect.schemas import DecidedBy, Decision, ManifestRecord, Reason, Source
from rag.collect.versions import abstract_similarity, kept_v1, needs_v1_check

V5_ABSTRACT = "We introduce subword units and improve over the baseline by 1.1 and 1.3 BLEU."
V1_ABSTRACT = (
    "We propose segmenting rare words and improve over a back-off dictionary by 0.8 and 1.5 BLEU."
)


def arxiv_record(version: int, abstract: str | None, **extra) -> ManifestRecord:
    return ManifestRecord(
        record_id=f"arxiv:2401.00001v{version}",
        source=Source.ARXIV,
        title="T",
        abstract=abstract,
        arxiv_id="2401.00001",
        arxiv_version=version,
        **extra,
    )


LATEST = arxiv_record(
    5,
    V5_ABSTRACT,
    relevance=Decision.ACCEPT,
    decided_by=DecidedBy.KEYWORD,
    reason=Reason.TITLE_TERM,
    matched_terms=["subword"],
    found_via=["arxiv_query:subword"],
    s2_id="s2id",
    acl_id="P16-1162",
)


def test_abstract_similarity_bounds() -> None:
    assert abstract_similarity("same text", "same   text") == 1.0
    assert abstract_similarity("abc", "xyz") == 0.0


def test_needs_v1_check() -> None:
    assert needs_v1_check(LATEST)
    assert not needs_v1_check(arxiv_record(1, "x"))
    assert not needs_v1_check(ManifestRecord(record_id="acl:x", source=Source.ACL, title="T"))


def test_revised_v1_is_kept_and_inherits_decision() -> None:
    v1 = kept_v1(LATEST, arxiv_record(1, V1_ABSTRACT), threshold=0.95)
    assert v1 is not None
    assert v1.record_id == "arxiv:2401.00001v1"
    assert v1.abstract == V1_ABSTRACT  # v1 keeps its own text
    assert (v1.relevance, v1.decided_by, v1.reason) == (
        Decision.ACCEPT,
        DecidedBy.INHERITED,
        Reason.EARLIER_VERSION,
    )
    assert v1.derived_from == LATEST.record_id
    assert (v1.s2_id, v1.acl_id) == ("s2id", "P16-1162")


def test_unchanged_v1_is_dropped() -> None:
    typo_fix = V5_ABSTRACT.replace("baseline", "basline")
    assert kept_v1(LATEST, arxiv_record(1, typo_fix), threshold=0.95) is None


def test_missing_abstract_means_no_evidence_so_not_kept() -> None:
    assert kept_v1(LATEST, arxiv_record(1, None), threshold=0.95) is None
