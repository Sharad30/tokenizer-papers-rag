from pathlib import Path

import httpx
import respx

from rag.collect.download import download, pdf_path
from rag.collect.schemas import DownloadStatus, ManifestRecord, Source

URL = "https://export.arxiv.org/pdf/2401.00001v1"
PDF = b"%PDF-1.7 synthetic"


def record(pdf_url: str | None = URL, record_id: str = "arxiv:2401.00001v1") -> ManifestRecord:
    return ManifestRecord(record_id=record_id, source=Source.ARXIV, title="T", pdf_url=pdf_url)


def test_pdf_path_is_filesystem_safe(tmp_path: Path) -> None:
    assert pdf_path(tmp_path, "arxiv:cs/0501001v1") == tmp_path / "arxiv_cs_0501001v1.pdf"


@respx.mock
def test_downloads_and_records_sha(make_client, tmp_path: Path) -> None:
    respx.get(URL).mock(return_value=httpx.Response(200, content=PDF))
    with make_client() as client:
        result = download(client, record(), tmp_path)
    assert result.download_status is DownloadStatus.DOWNLOADED
    assert len(result.pdf_sha256 or "") == 64
    assert pdf_path(tmp_path, result.record_id).read_bytes() == PDF


@respx.mock
def test_existing_valid_pdf_is_not_fetched_again(make_client, tmp_path: Path) -> None:
    route = respx.get(URL).mock(return_value=httpx.Response(200, content=PDF))
    with make_client() as client:
        first = download(client, record(), tmp_path)
        second = download(client, record(), tmp_path)
    assert second.download_status is DownloadStatus.ALREADY_PRESENT
    assert second.pdf_sha256 == first.pdf_sha256
    assert route.call_count == 1


@respx.mock
def test_corrupt_existing_file_is_replaced(make_client, tmp_path: Path) -> None:
    pdf_path(tmp_path, "arxiv:2401.00001v1").write_bytes(b"<html>half")
    respx.get(URL).mock(return_value=httpx.Response(200, content=PDF))
    with make_client() as client:
        assert download(client, record(), tmp_path).download_status is DownloadStatus.DOWNLOADED


@respx.mock
def test_landing_page_is_not_pdf_and_not_saved(make_client, tmp_path: Path) -> None:
    respx.get(URL).mock(return_value=httpx.Response(200, content=b"<html>publisher page</html>"))
    with make_client() as client:
        result = download(client, record(), tmp_path)
    assert result.download_status is DownloadStatus.NOT_PDF
    assert result.pdf_sha256 is None
    assert list(tmp_path.iterdir()) == []


@respx.mock
def test_http_failure_is_recorded_not_raised(make_client, tmp_path: Path) -> None:
    respx.get(URL).mock(return_value=httpx.Response(404))
    with make_client() as client:
        assert download(client, record(), tmp_path).download_status is DownloadStatus.FAILED


def test_no_url(make_client, tmp_path: Path) -> None:
    with make_client() as client:
        assert (
            download(client, record(pdf_url=None), tmp_path).download_status
            is DownloadStatus.NO_URL
        )
