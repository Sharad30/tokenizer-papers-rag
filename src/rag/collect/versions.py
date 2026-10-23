"""D-009: keep an arXiv v1 as its own record when the paper was substantively revised.

Evidence of revision is a changed abstract. Example from the live API: BPE
(1508.07909) v1 reports +0.8/+1.5 BLEU, v5 reports +1.1/+1.3, which is exactly
the kind of difference Phase 5 must not silently merge away.
"""

from difflib import SequenceMatcher

from rag.collect.schemas import DecidedBy, ManifestRecord, Reason


def abstract_similarity(a: str, b: str) -> float:
    """Ratio in [0, 1] of matching characters, after collapsing whitespace."""
    # autojunk=False: the default heuristic ignores frequent characters (like
    # spaces) in long strings, which distorts the ratio for abstracts.
    return SequenceMatcher(None, " ".join(a.split()), " ".join(b.split()), autojunk=False).ratio()


def needs_v1_check(record: ManifestRecord) -> bool:
    return record.arxiv_id is not None and (record.arxiv_version or 1) > 1


def kept_v1(latest: ManifestRecord, v1: ManifestRecord, threshold: float) -> ManifestRecord | None:
    """The v1 record (inheriting latest's decision) if its abstract differs enough, else None."""
    if not latest.abstract or not v1.abstract:
        return None
    if abstract_similarity(latest.abstract, v1.abstract) >= threshold:
        return None
    return v1.model_copy(
        update={
            "relevance": latest.relevance,
            "decided_by": DecidedBy.INHERITED,
            "reason": Reason.EARLIER_VERSION,
            "matched_terms": latest.matched_terms,
            "embedding_score": latest.embedding_score,
            "derived_from": latest.record_id,
            "found_via": [*latest.found_via],
            # Venue, S2 and ACL links belong to the work, so v1 shares them.
            "s2_id": latest.s2_id,
            "acl_id": latest.acl_id,
            "venue": latest.venue,
            "publication_types": latest.publication_types,
        }
    )
