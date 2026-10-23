"""arXiv: keyword search over the query API, plus licence and versions from OAI-PMH.

The query API returns Atom XML; it has no licence field, so licences and the
full version history come from OAI-PMH (metadataPrefix=arXivRaw). Both are
parsed with the stdlib ElementTree and explicit namespaces.
"""

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from datetime import date, datetime
from email.utils import parsedate_to_datetime

from rag.collect.config import ArxivQuery
from rag.collect.http import PoliteClient
from rag.collect.schemas import CandidateRecord, LicenceSource, Source
from rag.common.schemas import StrictModel

API_URL = "https://export.arxiv.org/api/query"
OAI_URL = "https://oaipmh.arxiv.org/oai"
# export.arxiv.org is the host arXiv asks programmatic clients to use.
PDF_URL = "https://export.arxiv.org/pdf/{arxiv_id}v{version}"
ABS_URL = "https://arxiv.org/abs/{arxiv_id}v{version}"

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "raw": "http://arxiv.org/OAI/arXivRaw/",
}

# Matches new-style (2305.15425v2) and old-style (hep-ex/0307015v1) IDs.
_ID_RE = re.compile(r"abs/(?P<id>.+?)v(?P<version>\d+)$")
# arXiv's own earliest submission date; the API needs both ends of a date range.
_EARLIEST = "199101010000"


class ArxivError(Exception):
    """The API reported an error (bad query), as opposed to a transient failure."""


class Page(StrictModel):
    total_results: int
    records: list[CandidateRecord]


class OaiRecord(StrictModel):
    arxiv_id: str
    licence: str | None
    versions: dict[int, datetime]  # version number -> submission time
    abstract: str | None


def build_search_query(query: ArxivQuery, categories: list[str], max_date: date) -> str:
    """Wrap a seed query with the category filter and the collection cutoff date."""
    parts = [f"({query.query})"]
    if categories:
        parts.append("(" + " OR ".join(f"cat:{c}" for c in categories) + ")")
    parts.append(f"submittedDate:[{_EARLIEST} TO {max_date:%Y%m%d}2359]")
    return " AND ".join(parts)


def search(
    client: PoliteClient,
    query: ArxivQuery,
    categories: list[str],
    max_date: date,
    page_size: int,
    max_results: int,
) -> Iterator[CandidateRecord]:
    """Page through one seed query, yielding at most `max_results` records."""
    search_query = build_search_query(query, categories, max_date)
    start = 0
    while start < max_results:
        size = min(page_size, max_results - start)
        params = {"search_query": search_query, "start": start, "max_results": size}
        # The API sometimes returns an empty page mid-way under load. Rejecting
        # it makes the client retry instead of caching an empty page forever.
        body = client.request_text(
            "GET", API_URL, params=params, accept=lambda b, s=start: _page_is_complete(b, s)
        )
        page = parse_feed(body)
        for record in page.records:
            yield record.model_copy(update={"found_via": [f"arxiv_query:{query.name}"]})
        start += len(page.records)
        if not page.records or start >= page.total_results:
            return


def fetch_by_ids(client: PoliteClient, versioned_ids: list[str]) -> list[CandidateRecord]:
    """Metadata for specific versions, e.g. ["1508.07909v1"]: each version has its own abstract."""
    if not versioned_ids:
        return []
    params = {"id_list": ",".join(versioned_ids), "max_results": len(versioned_ids)}
    return parse_feed(client.request_text("GET", API_URL, params=params)).records


def fetch_oai_record(client: PoliteClient, arxiv_id: str) -> OaiRecord | None:
    params = {
        "verb": "GetRecord",
        "identifier": f"oai:arXiv.org:{arxiv_id}",
        "metadataPrefix": "arXivRaw",
    }
    return parse_oai_record(client.request_text("GET", OAI_URL, params=params))


def parse_feed(xml_text: str) -> Page:
    root = ET.fromstring(xml_text)
    entries = root.findall("atom:entry", NS)
    for entry in entries:
        # Errors arrive as a normal feed with one entry whose id is an error URL.
        if "/api/errors" in _text(entry, "atom:id"):
            raise ArxivError(_text(entry, "atom:summary") or "arXiv API error")
    total = int(_text(root, "opensearch:totalResults") or 0)
    return Page(total_results=total, records=[_parse_entry(e) for e in entries])


def parse_oai_record(xml_text: str) -> OaiRecord | None:
    root = ET.fromstring(xml_text)
    if root.find("oai:error", NS) is not None:  # e.g. idDoesNotExist
        return None
    raw = root.find(".//raw:arXivRaw", NS)
    if raw is None:
        return None
    versions = {
        int(v.attrib["version"].lstrip("v")): parsedate_to_datetime(_text(v, "raw:date"))
        for v in raw.findall("raw:version", NS)
    }
    return OaiRecord(
        arxiv_id=_text(raw, "raw:id"),
        licence=_text(raw, "raw:license") or None,
        versions=versions,
        abstract=_clean(_text(raw, "raw:abstract")) or None,
    )


def _parse_entry(entry: ET.Element) -> CandidateRecord:
    match = _ID_RE.search(_text(entry, "atom:id"))
    if match is None:
        raise ArxivError(f"Unrecognised arXiv id: {_text(entry, 'atom:id')}")
    arxiv_id, version = match["id"], int(match["version"])
    # `published` is when v1 was submitted, whichever version this entry is.
    published = datetime.fromisoformat(_text(entry, "atom:published").replace("Z", "+00:00"))
    return CandidateRecord(
        record_id=f"arxiv:{arxiv_id}v{version}",
        source=Source.ARXIV,
        title=_clean(_text(entry, "atom:title")),
        abstract=_clean(_text(entry, "atom:summary")) or None,
        authors=[_clean(_text(a, "atom:name")) for a in entry.findall("atom:author", NS)],
        year=published.year,
        publication_date=published.date(),
        url=ABS_URL.format(arxiv_id=arxiv_id, version=version),
        pdf_url=PDF_URL.format(arxiv_id=arxiv_id, version=version),
        arxiv_id=arxiv_id,
        arxiv_version=version,
        doi=_text(entry, "arxiv:doi") or None,
        journal_ref=_clean(_text(entry, "arxiv:journal_ref")) or None,
        categories=[c.attrib["term"] for c in entry.findall("atom:category", NS)],
    )


def _page_is_complete(xml_text: str, start: int) -> bool:
    """False for an empty page that should have had entries (a transient API glitch)."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return False
    total = int(_text(root, "opensearch:totalResults") or 0)
    return start >= total or bool(root.findall("atom:entry", NS))


def licence_fields(oai: OaiRecord | None) -> dict[str, object]:
    if oai is None or oai.licence is None:
        return {"licence": None, "licence_source": LicenceSource.UNKNOWN}
    return {"licence": oai.licence, "licence_source": LicenceSource.ARXIV_OAI}


def _text(element: ET.Element, path: str) -> str:
    found = element.find(path, NS)
    return (found.text or "").strip() if found is not None else ""


def _clean(text: str) -> str:
    # Atom titles and abstracts are hard-wrapped; collapse to single spaces.
    return " ".join(text.split())
