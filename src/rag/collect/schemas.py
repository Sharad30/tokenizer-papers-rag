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
