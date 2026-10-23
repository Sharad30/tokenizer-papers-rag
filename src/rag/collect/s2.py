"""Semantic Scholar Graph API: resolve records to linked IDs, and snowball one hop.

Resolving gives every candidate its `externalIds` (arXiv, DOI, ACL), venue,
year and publication types, which link manifestations of the same paper
without merging them. Snowballing follows references and citations of the
anchor papers to find work that keyword search misses.
"""

import logging
from collections.abc import Iterator
from datetime import date
from typing import Any

from rag.collect.http import HttpError, PoliteClient
from rag.collect.schemas import CandidateRecord, LicenceSource, Source

logger = logging.getLogger(__name__)

API_URL = "https://api.semanticscholar.org/graph/v1"

# Requested for every paper, whether resolved directly or reached by snowball.
PAPER_FIELDS = (
    "paperId,externalIds,title,abstract,venue,year,publicationDate,"
    "publicationTypes,authors,openAccessPdf"
)
BAD_REQUEST = 400
DIRECTION_KEY = {"references": "citedPaper", "citations": "citingPaper"}


def resolve_batch(
    client: PoliteClient, ids: list[str], batch_size: int
) -> list[dict[str, Any] | None]:
    """Look up papers by ID ("ARXIV:1508.07909", "DOI:...", "ACL:..."), in input order.

    The batch endpoint returns null for IDs it doesn't know; those stay None.
    """
    results: list[dict[str, Any] | None] = []
    for start in range(0, len(ids), batch_size):
        chunk = ids[start : start + batch_size]
        response = client.post_json(
            f"{API_URL}/paper/batch", params={"fields": PAPER_FIELDS}, body={"ids": chunk}
        )
        if len(response) != len(chunk):
            raise ValueError(f"S2 batch returned {len(response)} items for {len(chunk)} ids")
        results.extend(response)
    return results


def snowball(
    client: PoliteClient,
    anchor_id: str,
    direction: str,
    page_size: int,
    max_items: int,
) -> Iterator[dict[str, Any]]:
    """Papers one hop from an anchor: what it cites ("references") or what cites it."""
    key = DIRECTION_KEY[direction]
    offset = 0
    while offset < max_items:
        params = {
            "fields": PAPER_FIELDS,
            "offset": offset,
            "limit": min(page_size, max_items - offset),
        }
        try:
            page = client.get_json(f"{API_URL}/paper/{anchor_id}/{direction}", params=params)
        except HttpError as exc:
            # The API caps how deep offset can go for very highly cited papers
            # (undocumented; it answers 400), so stop there and say so. Anything
            # else, e.g. retries exhausted on 429, must fail the step: carrying
            # on would silently drop papers. A rerun resumes from the cache.
            if exc.status != BAD_REQUEST or exc.retries_exhausted:
                raise
            logger.warning("Stopped %s of %s at offset %d: %s", direction, anchor_id, offset, exc)
            return
        for item in page.get("data") or []:
            paper = item.get(key)
            # Unresolved references come back as null or without a paperId.
            if paper and paper.get("paperId"):
                yield paper
        if "next" not in page:
            return
        offset = page["next"]


def to_candidate(paper: dict[str, Any], found_via: list[str]) -> CandidateRecord:
    external = paper.get("externalIds") or {}
    open_access = paper.get("openAccessPdf") or {}
    licence = open_access.get("license") or None
    return CandidateRecord(
        record_id=f"s2:{paper['paperId']}",
        source=Source.S2,
        title=" ".join((paper.get("title") or "").split()),
        abstract=paper.get("abstract") or None,
        authors=[a["name"] for a in paper.get("authors") or [] if a.get("name")],
        year=paper.get("year"),
        publication_date=_parse_date(paper.get("publicationDate")),
        venue=paper.get("venue") or None,
        url=f"https://www.semanticscholar.org/paper/{paper['paperId']}",
        pdf_url=open_access.get("url") or None,
        arxiv_id=external.get("ArXiv"),
        doi=external.get("DOI"),
        s2_id=paper["paperId"],
        acl_id=external.get("ACL"),
        publication_types=_normalise_types(paper.get("publicationTypes")),
        licence=licence,
        licence_source=LicenceSource.S2_OPEN_ACCESS if licence else None,
        found_via=found_via,
    )


def within_cutoff(record: CandidateRecord, max_date: date) -> bool:
    """Snowballed citations include papers newer than the collection cutoff; drop them.

    Uses the publication date when known, else the year. Records with neither
    are kept, so a missing date never silently removes a paper.
    """
    if record.publication_date is not None:
        return record.publication_date <= max_date
    if record.year is not None:
        return record.year <= max_date.year
    return True


def lookup_id(record: CandidateRecord) -> str | None:
    """The most specific S2 lookup key for a record from another source."""
    if record.s2_id:
        return record.s2_id
    if record.arxiv_id:
        return f"ARXIV:{record.arxiv_id}"
    if record.acl_id:
        return f"ACL:{record.acl_id}"
    if record.doi:
        return f"DOI:{record.doi}"
    return None


def _parse_date(value: str | None) -> date | None:
    try:
        return date.fromisoformat(value) if value else None
    except ValueError:
        return None


def _normalise_types(types: list[str] | None) -> list[str]:
    # The API docs show both "JournalArticle" and "Journal Article".
    return [t.replace(" ", "") for t in types or []]
