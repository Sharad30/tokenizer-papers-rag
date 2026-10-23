"""Records that cross step boundaries inside the collect stage."""

from datetime import date
from enum import StrEnum

from rag.common.schemas import StrictModel


class Source(StrEnum):
    ARXIV = "arxiv"
    S2 = "s2"
    ACL = "acl"


class LicenceSource(StrEnum):
    ARXIV_OAI = "arxiv_oai"
    S2_OPEN_ACCESS = "s2_open_access"
    ACL_YEAR_RULE = "acl_year_rule"
    UNKNOWN = "unknown"


class Decision(StrEnum):
    ACCEPT = "accept"
    REJECT = "reject"
    AMBIGUOUS = "ambiguous"  # only ever an intermediate state; the judge resolves it


class DecidedBy(StrEnum):
    KEYWORD = "keyword"
    EMBEDDING = "embedding"
    # Copies of an already-decided paper (its venue version, its v1) take that
    # paper's decision, so two copies of one paper can never disagree.
    INHERITED = "inherited"


class Reason(StrEnum):
    NO_TOPIC_TERM = "no_topic_term"  # no topic term in title or abstract
    OFF_TOPIC_TERM = "off_topic_term"  # title names an excluded sense, e.g. "visual tokenizer"
    TITLE_TERM = "title_term"  # a strong topic term in the title
    ABSTRACT_TERM_ONLY = "abstract_term_only"  # sent to the embedding judge
    EMBEDDING_ABOVE = "embedding_above_threshold"
    EMBEDDING_BELOW = "embedding_below_threshold"
    LINKED_MANIFESTATION = "linked_manifestation"  # e.g. the ACL copy of an arXiv paper
    EARLIER_VERSION = "earlier_version"  # an arXiv v1 kept by the D-009 rule


class RelevanceDecision(StrictModel):
    decision: Decision
    decided_by: DecidedBy
    reason: Reason
    matched_terms: list[str]
    embedding_score: float | None = None


class CandidateRecord(StrictModel):
    """One manifestation of a paper as one source describes it.

    An arXiv v1, the arXiv latest version and the ACL venue copy of the same
    paper are three records. They are linked by shared IDs, never merged here;
    merging is Phase 5's job, so its before/after numbers stay honest.
    """

    record_id: str  # "arxiv:2305.15425v2", "acl:2022.acl-long.220", "s2:<paperId>"
    source: Source
    title: str
    abstract: str | None = None
    authors: list[str] = []
    year: int | None = None
    publication_date: date | None = None
    venue: str | None = None
    url: str | None = None
    pdf_url: str | None = None
    arxiv_id: str | None = None  # without version: "2305.15425"
    arxiv_version: int | None = None
    doi: str | None = None
    s2_id: str | None = None
    acl_id: str | None = None
    journal_ref: str | None = None
    categories: list[str] = []
    publication_types: list[str] = []
    licence: str | None = None
    licence_source: LicenceSource | None = None
    # How this record was found, e.g. "arxiv_query:bpe", "snowball_cit:arxiv:1508.07909".
    found_via: list[str] = []


class DownloadStatus(StrEnum):
    DOWNLOADED = "downloaded"
    ALREADY_PRESENT = "already_present"  # a valid PDF from an earlier run (resume)
    NO_URL = "no_url"
    NOT_PDF = "not_pdf"  # e.g. an open-access link that is really a landing page
    FAILED = "failed"
    NOT_ATTEMPTED = "not_attempted"  # rejected, source disabled, or past --limit


class ManifestRecord(CandidateRecord):
    """A candidate plus every decision made about it: the manifest's row schema."""

    relevance: Decision | None = None
    decided_by: DecidedBy | None = None
    reason: Reason | None = None
    matched_terms: list[str] = []
    embedding_score: float | None = None
    # Which record this one was derived from, for linked copies and kept v1s.
    derived_from: str | None = None
    download_status: DownloadStatus = DownloadStatus.NOT_ATTEMPTED
    pdf_sha256: str | None = None
    included_in_index: bool | None = None  # filled in by Phase 2

    def with_decision(self, decision: RelevanceDecision) -> "ManifestRecord":
        return self.model_copy(
            update={
                "relevance": decision.decision,
                "decided_by": decision.decided_by,
                "reason": decision.reason,
                "matched_terms": decision.matched_terms,
                "embedding_score": decision.embedding_score,
            }
        )
