import json
from datetime import date
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
import respx

from rag.collect import s2
from rag.collect.http import HttpError
from rag.collect.schemas import CandidateRecord, LicenceSource, Source

# Synthetic paper in the shape the Graph API returns (checked against a live response).
PAPER = {
    "paperId": "abc123",
    "externalIds": {
        "ArXiv": "2401.00001",
        "DOI": "10.0000/x",
        "ACL": "2024.acl-long.1",
        "CorpusId": 42,
    },
    "title": "A Synthetic  Tokenizer\nPaper",
    "abstract": "We study tokenizers.",
    "venue": "Annual Meeting of the ACL",
    "year": 2024,
    "publicationDate": "2024-01-02",
    "publicationTypes": ["Journal Article", "Conference"],
    "authors": [{"authorId": "1", "name": "Ada Example"}, {"authorId": "2", "name": None}],
    "openAccessPdf": {"url": "https://example.org/a.pdf", "status": "GOLD", "license": "CCBY"},
}

BATCH_URL = f"{s2.API_URL}/paper/batch"


def test_to_candidate_maps_fields() -> None:
    record = s2.to_candidate(PAPER, found_via=["snowball_ref:ARXIV:1"])
    assert record.record_id == "s2:abc123"
    assert record.source is Source.S2
    assert record.title == "A Synthetic Tokenizer Paper"
    assert record.authors == ["Ada Example"]
    assert (record.arxiv_id, record.doi, record.acl_id) == (
        "2401.00001",
        "10.0000/x",
        "2024.acl-long.1",
    )
    assert record.publication_date == date(2024, 1, 2)
    assert record.publication_types == ["JournalArticle", "Conference"]
    assert record.pdf_url == "https://example.org/a.pdf"
    assert (record.licence, record.licence_source) == ("CCBY", LicenceSource.S2_OPEN_ACCESS)
    assert record.found_via == ["snowball_ref:ARXIV:1"]


def test_to_candidate_tolerates_missing_optional_fields() -> None:
    record = s2.to_candidate({"paperId": "p", "title": "T"}, found_via=[])
    assert record.abstract is None
    assert record.licence is None and record.licence_source is None
    assert record.publication_date is None


@respx.mock
def test_resolve_batch_chunks_and_keeps_order(make_client) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        ids = json.loads(request.content)["ids"]
        return httpx.Response(
            200, json=[None if i == "ARXIV:missing" else {"paperId": i} for i in ids]
        )

    route = respx.post(BATCH_URL).mock(side_effect=respond)
    with make_client() as client:
        results = s2.resolve_batch(client, ["A", "ARXIV:missing", "C"], batch_size=2)
    assert results == [{"paperId": "A"}, None, {"paperId": "C"}]
    assert [json.loads(c.request.content)["ids"] for c in route.calls] == [
        ["A", "ARXIV:missing"],
        ["C"],
    ]
    assert "fields=" in str(route.calls.last.request.url)


@respx.mock
def test_resolve_batch_rejects_length_mismatch(make_client) -> None:
    respx.post(BATCH_URL).mock(return_value=httpx.Response(200, json=[]))
    with make_client() as client, pytest.raises(ValueError, match="0 items for 1 ids"):
        s2.resolve_batch(client, ["A"], batch_size=10)


def _offset(request: httpx.Request) -> int:
    return int(parse_qs(urlsplit(str(request.url)).query)["offset"][0])


@respx.mock
def test_snowball_follows_next_and_skips_unresolved(make_client) -> None:
    pages = {
        0: {
            "offset": 0,
            "next": 2,
            "data": [{"citedPaper": {"paperId": "p1"}}, {"citedPaper": None}],
        },
        2: {
            "offset": 2,
            "data": [
                {"citedPaper": {"paperId": None, "title": "unresolved"}},
                {"citedPaper": {"paperId": "p2"}},
            ],
        },
    }
    route = respx.get(f"{s2.API_URL}/paper/ANCHOR/references").mock(
        side_effect=lambda request: httpx.Response(200, json=pages[_offset(request)])
    )
    with make_client() as client:
        papers = list(s2.snowball(client, "ANCHOR", "references", page_size=2, max_items=10))
    assert [p["paperId"] for p in papers] == ["p1", "p2"]
    assert route.call_count == 2


@respx.mock
def test_snowball_uses_citing_paper_key_for_citations(make_client) -> None:
    respx.get(f"{s2.API_URL}/paper/ANCHOR/citations").mock(
        return_value=httpx.Response(
            200, json={"offset": 0, "data": [{"citingPaper": {"paperId": "c1"}}]}
        )
    )
    with make_client() as client:
        papers = list(s2.snowball(client, "ANCHOR", "citations", page_size=5, max_items=5))
    assert [p["paperId"] for p in papers] == ["c1"]


@respx.mock
def test_snowball_respects_max_items(make_client) -> None:
    route = respx.get(f"{s2.API_URL}/paper/ANCHOR/references").mock(
        return_value=httpx.Response(
            200, json={"offset": 0, "next": 3, "data": [{"citedPaper": {"paperId": "p"}}] * 3}
        )
    )
    with make_client() as client:
        list(s2.snowball(client, "ANCHOR", "references", page_size=3, max_items=3))
    assert route.call_count == 1


@respx.mock
def test_snowball_stops_cleanly_on_http_error(make_client) -> None:
    respx.get(f"{s2.API_URL}/paper/ANCHOR/citations").mock(return_value=httpx.Response(400))
    with make_client() as client:
        assert list(s2.snowball(client, "ANCHOR", "citations", page_size=5, max_items=50)) == []


@pytest.mark.parametrize(
    ("publication_date", "year", "kept"),
    [
        (date(2026, 10, 15), 2026, True),  # on the cutoff day
        (date(2026, 10, 16), 2026, False),
        (None, 2026, True),  # same year, no date: can't tell, keep
        (None, 2027, False),
        (None, None, True),  # nothing known: never silently dropped
    ],
)
def test_within_cutoff(publication_date: date | None, year: int | None, kept: bool) -> None:
    record = CandidateRecord(
        record_id="s2:x", source=Source.S2, title="T", publication_date=publication_date, year=year
    )
    assert s2.within_cutoff(record, date(2026, 10, 15)) is kept


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({"s2_id": "abc", "arxiv_id": "1"}, "abc"),
        ({"arxiv_id": "2401.00001", "acl_id": "P16-1"}, "ARXIV:2401.00001"),
        ({"acl_id": "P16-1162", "doi": "10.1/x"}, "ACL:P16-1162"),
        ({"doi": "10.1/x"}, "DOI:10.1/x"),
        ({}, None),
    ],
)
def test_lookup_id_prefers_most_specific(fields: dict, expected: str | None) -> None:
    record = CandidateRecord(record_id="r", source=Source.ACL, title="T", **fields)
    assert s2.lookup_id(record) == expected


@respx.mock
def test_snowball_raises_when_rate_limit_retries_run_out(make_client) -> None:
    # Truncating here would silently drop papers; failing lets a rerun resume.
    respx.get(f"{s2.API_URL}/paper/ANCHOR/citations").mock(return_value=httpx.Response(429))
    with make_client() as client, pytest.raises(HttpError, match="after 3 attempts"):
        list(s2.snowball(client, "ANCHOR", "citations", page_size=5, max_items=50))
