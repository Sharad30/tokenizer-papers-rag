from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
import respx

from rag.collect import arxiv
from rag.collect.config import ArxivQuery
from rag.collect.schemas import LicenceSource, Source

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "arxiv"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_build_search_query_adds_categories_and_cutoff() -> None:
    query = ArxivQuery(name="bpe", query='abs:"byte pair encoding"')
    built = arxiv.build_search_query(query, ["cs.CL", "cs.LG"], date(2026, 10, 15))
    assert built == (
        '(abs:"byte pair encoding") AND (cat:cs.CL OR cat:cs.LG) '
        "AND submittedDate:[199101010000 TO 202610152359]"
    )


def test_build_search_query_without_categories() -> None:
    query = ArxivQuery(name="q", query="ti:bpe")
    built = arxiv.build_search_query(query, [], date(2020, 1, 1))
    assert built == "(ti:bpe) AND submittedDate:[199101010000 TO 202001012359]"


def test_parse_feed_extracts_fields() -> None:
    page = arxiv.parse_feed(fixture("feed_page.xml"))
    assert page.total_results == 3
    first, second = page.records
    assert first.record_id == "arxiv:2401.00001v3"
    assert first.source is Source.ARXIV
    assert (first.arxiv_id, first.arxiv_version) == ("2401.00001", 3)
    # Hard-wrapped Atom text is collapsed to single spaces.
    assert first.title == "A Synthetic Study of Subword Vocabularies"
    assert first.abstract == "We vary the vocabulary size of a byte pair encoding tokenizer."
    assert first.authors == ["Ada Example", "Ben Sample"]
    # `published` (v1 date), not `updated`, gives the year.
    assert first.publication_date == date(2024, 1, 2)
    assert first.year == 2024
    assert first.doi == "10.0000/synthetic.1"
    assert first.journal_ref == "Proceedings of Nowhere 2024"
    assert first.categories == ["cs.CL", "cs.LG"]
    assert first.pdf_url == "https://export.arxiv.org/pdf/2401.00001v3"
    assert second.record_id == "arxiv:cs/0501001v1"
    assert second.doi is None


def test_parse_feed_raises_on_error_feed() -> None:
    with pytest.raises(arxiv.ArxivError, match="incorrect id format"):
        arxiv.parse_feed(fixture("feed_error.xml"))


@pytest.mark.parametrize(
    ("name", "start", "expected"),
    [
        ("feed_page.xml", 0, True),
        ("feed_empty_page.xml", 2, False),  # empty, but totalResults says 3
        ("feed_empty_page.xml", 3, True),  # genuinely past the end
    ],
)
def test_page_is_complete(name: str, start: int, expected: bool) -> None:
    assert arxiv._page_is_complete(fixture(name), start) is expected


def test_page_is_complete_rejects_non_xml() -> None:
    assert arxiv._page_is_complete("<html>Service Unavailable", 0) is False


def _start(request: httpx.Request) -> int:
    return int(parse_qs(urlsplit(str(request.url)).query)["start"][0])


@respx.mock
def test_search_pages_until_total_and_tags_found_via(make_client) -> None:
    pages = {0: fixture("feed_page.xml"), 2: fixture("feed_page2.xml")}
    route = respx.get(arxiv.API_URL).mock(
        side_effect=lambda request: httpx.Response(200, text=pages[_start(request)])
    )
    query = ArxivQuery(name="bpe", query="abs:bpe")
    with make_client() as client:
        records = list(arxiv.search(client, query, ["cs.CL"], date(2026, 1, 1), 2, 10))
    assert [r.arxiv_id for r in records] == ["2401.00001", "cs/0501001", "2402.00002"]
    assert all(r.found_via == ["arxiv_query:bpe"] for r in records)
    assert route.call_count == 2


@respx.mock
def test_search_stops_at_max_results(make_client) -> None:
    route = respx.get(arxiv.API_URL).mock(
        return_value=httpx.Response(200, text=fixture("feed_page.xml"))
    )
    query = ArxivQuery(name="bpe", query="abs:bpe")
    with make_client() as client:
        records = list(arxiv.search(client, query, [], date(2026, 1, 1), 2, 2))
    assert len(records) == 2
    assert route.call_count == 1
    assert parse_qs(urlsplit(str(route.calls.last.request.url)).query)["max_results"] == ["2"]


@respx.mock
def test_search_retries_empty_page_instead_of_caching_it(make_client) -> None:
    responses = iter(
        [fixture("feed_page.xml"), fixture("feed_empty_page.xml"), fixture("feed_page2.xml")]
    )
    respx.get(arxiv.API_URL).mock(
        side_effect=lambda request: httpx.Response(200, text=next(responses))
    )
    query = ArxivQuery(name="bpe", query="abs:bpe")
    with make_client() as client:
        records = list(arxiv.search(client, query, [], date(2026, 1, 1), 2, 10))
    assert len(records) == 3


@respx.mock
def test_fetch_by_ids_requests_specific_versions(make_client) -> None:
    route = respx.get(arxiv.API_URL).mock(
        return_value=httpx.Response(200, text=fixture("feed_page2.xml"))
    )
    with make_client() as client:
        records = arxiv.fetch_by_ids(client, ["2402.00002v1"])
    assert records[0].record_id == "arxiv:2402.00002v1"
    params = parse_qs(urlsplit(str(route.calls.last.request.url)).query)
    assert params["id_list"] == ["2402.00002v1"]


def test_fetch_by_ids_with_no_ids_makes_no_request(make_client) -> None:
    with make_client() as client:
        assert arxiv.fetch_by_ids(client, []) == []
        assert client.counts["network"] == 0


def test_parse_oai_record() -> None:
    record = arxiv.parse_oai_record(fixture("oai_record.xml"))
    assert record is not None
    assert record.arxiv_id == "2401.00001"
    assert record.licence == "http://creativecommons.org/licenses/by/4.0/"
    assert sorted(record.versions) == [1, 2, 3]
    assert record.versions[1] == datetime(2024, 1, 2, 9, 30, tzinfo=UTC)
    assert record.abstract == "We vary the vocabulary size of a byte pair encoding tokenizer."


def test_parse_oai_error_returns_none() -> None:
    assert arxiv.parse_oai_record(fixture("oai_error.xml")) is None


def test_licence_fields() -> None:
    record = arxiv.parse_oai_record(fixture("oai_record.xml"))
    assert arxiv.licence_fields(record) == {
        "licence": "http://creativecommons.org/licenses/by/4.0/",
        "licence_source": LicenceSource.ARXIV_OAI,
    }
    assert arxiv.licence_fields(None)["licence_source"] is LicenceSource.UNKNOWN
