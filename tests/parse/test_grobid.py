from pathlib import Path

import httpx
import pytest
import respx

from rag.parse.config import GrobidConfig
from rag.parse.grobid import FULLTEXT_PATH, GrobidClient, GrobidOutcome

URL = "http://grobid.test"
TEI = b"<TEI/>"


@pytest.fixture
def grobid_config() -> GrobidConfig:
    return GrobidConfig(
        url=URL,
        image="grobid/grobid:test",
        workers=1,
        timeout_s=5,
        busy_retry_s=7,
        max_attempts=3,
        consolidate_header=0,
        consolidate_citations=0,
        tei_coordinates=["p", "s"],
    )


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.7 synthetic")
    return path


def make(config: GrobidConfig, sleeps: list[float]) -> GrobidClient:
    return GrobidClient(config, httpx.Client(base_url=URL), sleep=sleeps.append)


@respx.mock
def test_sends_configured_form_fields(grobid_config, pdf) -> None:
    route = respx.post(URL + FULLTEXT_PATH).mock(return_value=httpx.Response(200, content=TEI))
    result = make(grobid_config, []).process(pdf)
    assert (result.outcome, result.tei) == (GrobidOutcome.OK, TEI)
    body = route.calls.last.request.content
    assert b'name="consolidateHeader"\r\n\r\n0' in body
    assert b'name="segmentSentences"\r\n\r\n1' in body
    assert body.count(b'name="teiCoordinates"') == 2  # repeated field, one per element
    assert b"%PDF-1.7 synthetic" in body


@respx.mock
def test_busy_is_retried_after_waiting(grobid_config, pdf) -> None:
    respx.post(URL + FULLTEXT_PATH).mock(
        side_effect=[httpx.Response(503), httpx.Response(200, content=TEI)]
    )
    sleeps: list[float] = []
    result = make(grobid_config, sleeps).process(pdf)
    assert result.outcome is GrobidOutcome.OK
    assert sleeps == [7]


@respx.mock
def test_busy_until_attempts_run_out_is_an_error(grobid_config, pdf) -> None:
    respx.post(URL + FULLTEXT_PATH).mock(return_value=httpx.Response(503))
    sleeps: list[float] = []
    result = make(grobid_config, sleeps).process(pdf)
    assert result.outcome is GrobidOutcome.ERROR
    assert "retries exhausted" in (result.detail or "")
    assert len(sleeps) == 2  # no wait after the last attempt


@respx.mock
def test_no_content_is_not_retried(grobid_config, pdf) -> None:
    route = respx.post(URL + FULLTEXT_PATH).mock(return_value=httpx.Response(204))
    assert make(grobid_config, []).process(pdf).outcome is GrobidOutcome.NO_CONTENT
    assert route.call_count == 1


@respx.mock
def test_server_error_is_not_retried(grobid_config, pdf) -> None:
    route = respx.post(URL + FULLTEXT_PATH).mock(return_value=httpx.Response(500, text="boom"))
    result = make(grobid_config, []).process(pdf)
    assert result.outcome is GrobidOutcome.ERROR
    assert result.detail == "grobid 500: boom"
    assert route.call_count == 1


@respx.mock
def test_connection_errors_are_retried(grobid_config, pdf) -> None:
    respx.post(URL + FULLTEXT_PATH).mock(
        side_effect=[httpx.ConnectError("refused"), httpx.Response(200, content=TEI)]
    )
    assert make(grobid_config, []).process(pdf).outcome is GrobidOutcome.OK


@respx.mock
def test_is_alive(grobid_config) -> None:
    respx.get(URL + "/api/isalive").mock(return_value=httpx.Response(200, text="true"))
    assert make(grobid_config, []).is_alive()
    respx.get(URL + "/api/isalive").mock(side_effect=httpx.ConnectError("down"))
    assert not make(grobid_config, []).is_alive()
