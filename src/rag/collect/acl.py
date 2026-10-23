"""ACL Anthology: local scan for tokenizer papers, and venue records by ACL id.

The official `acl-anthology` library clones the Anthology's metadata repo
(~120 MB) and reads it locally, so scanning ~100k papers costs no API calls.
"""

import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from rag.collect.schemas import CandidateRecord, LicenceSource, Source

logger = logging.getLogger(__name__)

# https://aclanthology.org/faq/copyright/ : material from 2016 on is CC BY 4.0;
# earlier material is CC BY-NC-SA 3.0. The library has no per-paper field.
LICENCE_CHANGE_YEAR = 2016
LICENCE_FROM_2016 = "CC BY 4.0"
LICENCE_BEFORE_2016 = "CC BY-NC-SA 3.0"


def load_anthology(data_dir: Path) -> Any:
    """Clone (first run) or update the Anthology data repo under data_dir."""
    # Imported here: loading the package is slow and only this step needs it.
    from acl_anthology import Anthology

    logger.info("Loading ACL Anthology data from %s (first run clones ~120 MB)", data_dir)
    return Anthology.from_repo(path=data_dir, verbose=False)


def licence_for_year(year: int | None) -> tuple[str | None, LicenceSource]:
    if year is None:
        return None, LicenceSource.UNKNOWN
    licence = LICENCE_FROM_2016 if year >= LICENCE_CHANGE_YEAR else LICENCE_BEFORE_2016
    return licence, LicenceSource.ACL_YEAR_RULE


def scan(
    papers: Iterator[Any], matches: Callable[[str, str], bool], max_year: int
) -> Iterator[CandidateRecord]:
    """Candidates whose title/abstract `matches` the topic terms, up to max_year.

    The Anthology has only years, not dates, so the cutoff is by year. This is
    a cheap local pre-screen (like an arXiv query); the relevance filter
    still makes the real decision.
    """
    for paper in papers:
        if paper.is_frontmatter or paper.is_deleted:
            continue
        record = to_candidate(paper, found_via=["acl_scan"])
        if record.year is not None and record.year > max_year:
            continue
        if matches(record.title, record.abstract or ""):
            yield record


def get_record(anthology: Any, acl_id: str, found_via: list[str]) -> CandidateRecord | None:
    paper = anthology.get_paper(acl_id)
    return None if paper is None else to_candidate(paper, found_via)


def to_candidate(paper: Any, found_via: list[str]) -> CandidateRecord:
    year = int(paper.year) if paper.year else None
    licence, licence_source = licence_for_year(year)
    abstract = paper.abstract.as_text() if paper.abstract is not None else None
    return CandidateRecord(
        record_id=f"acl:{paper.full_id}",
        source=Source.ACL,
        title=" ".join(paper.title.as_text().split()),
        abstract=" ".join(abstract.split()) if abstract else None,
        authors=[spec.name.as_first_last() for spec in paper.authors],
        year=year,
        venue=",".join(paper.venue_ids) or None,
        url=paper.web_url,
        pdf_url=paper.pdf.url if paper.pdf is not None else None,
        doi=paper.doi,
        acl_id=paper.full_id,
        licence=licence,
        licence_source=licence_source,
        found_via=found_via,
    )
