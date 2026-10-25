import json
from pathlib import Path

import httpx
import pymupdf
import pytest
import respx

from rag.collect.download import pdf_path
from rag.collect.manifest import read_manifest, write_manifest
from rag.collect.pipeline import PDFS
from rag.collect.schemas import Decision, DownloadStatus, ManifestRecord, Source
from rag.common.config import load_config
from rag.common.io import read_jsonl
from rag.parse import pipeline
from rag.parse.grobid import FULLTEXT_PATH
from rag.parse.schemas import Chunk, ElementType, IndexStatus, ParseOutcome, Parser
from rag.parse.show import render_paper
from tests.parse.conftest import WhitespaceCounter

TEI = (Path(__file__).parent.parent / "fixtures" / "tei" / "paper.tei.xml").read_bytes()
GROBID_URL = "http://grobid.test"

RECORDS = [
    ManifestRecord(
        record_id="arxiv:0000.00001v1",
        source=Source.ARXIV,
        title="A Synthetic Paper",
        arxiv_id="0000.00001",
        arxiv_version=1,
        relevance=Decision.ACCEPT,
        download_status=DownloadStatus.DOWNLOADED,
    ),
    ManifestRecord(
        record_id="acl:X00-0001",
        source=Source.ACL,
        title="A Scanned-Looking Paper",
        abstract="An abstract from the manifest.",
        relevance=Decision.ACCEPT,
        download_status=DownloadStatus.DOWNLOADED,
    ),
    ManifestRecord(
        record_id="s2:nopdf",
        source=Source.S2,
        title="No PDF",
        relevance=Decision.ACCEPT,
        download_status=DownloadStatus.NO_URL,
    ),
    ManifestRecord(
        record_id="s2:rejected", source=Source.S2, title="Rejected", relevance=Decision.REJECT
    ),
]


@pytest.fixture
def config(tmp_path: Path, minimal_config_text: str):
    path = tmp_path / "config.yaml"
    path.write_text(minimal_config_text.replace("busy_retry_s: 1", "busy_retry_s: 0"))
    config = load_config(path)
    write_manifest(RECORDS, config.paths.manifest)
    pdf_dir = config.paths.raw / PDFS
    pdf_dir.mkdir(parents=True)
    for record in RECORDS[:2]:
        document = pymupdf.open()
        document.new_page().insert_text((72, 72), "Fallback body text. It has two sentences.")
        document.save(pdf_path(pdf_dir, record.record_id))
        document.close()
    return config


@pytest.fixture
def grobid_service():
    def respond(request: httpx.Request) -> httpx.Response:
        # The arXiv paper parses; GROBID finds nothing in the ACL one.
        if b"arxiv_0000.00001v1.pdf" in request.content:
            return httpx.Response(200, content=TEI)
        return httpx.Response(204)

    with respx.mock(assert_all_called=False) as router:
        router.get(GROBID_URL + "/api/isalive").mock(return_value=httpx.Response(200, text="true"))
        router.post(GROBID_URL + FULLTEXT_PATH).mock(side_effect=respond)
        yield router


@pytest.fixture
def whitespace_counter(monkeypatch):
    monkeypatch.setattr(pipeline, "HFTokenCounter", lambda model_name: WhitespaceCounter())


def stats(config, step: str) -> dict:
    return json.loads((pipeline.step_dir(config, step) / "stats.json").read_text())["counts"]


def test_parse_all_end_to_end(config, grobid_service, whitespace_counter) -> None:
    pipeline.run_all(config)

    tei_counts = stats(config, "grobid")
    assert (
        tei_counts["outcome.grobid"],
        tei_counts["outcome.failed"],
        tei_counts["outcome.no_pdf"],
    ) == (
        1,
        1,
        1,
    )
    structure_counts = stats(config, "structure")
    assert structure_counts["outcome.fallback"] == 1
    assert structure_counts["references"] == 2

    chunks = read_jsonl(pipeline.step_dir(config, "chunk") / pipeline.CHUNKS, Chunk)
    by_record = {c.record_id for c in chunks}
    assert by_record == {"arxiv:0000.00001v1", "acl:X00-0001"}
    grobid_chunks = [c for c in chunks if c.record_id == "arxiv:0000.00001v1"]
    assert {c.element_type for c in grobid_chunks} == {
        ElementType.ABSTRACT,
        ElementType.PARAGRAPH,
        ElementType.TABLE,
        ElementType.FIGURE_CAPTION,
    }
    assert all(c.header.startswith("A Synthetic Paper") for c in grobid_chunks)  # manifest title
    assert all(c.arxiv_version == 1 for c in grobid_chunks)
    assert not any("Neural units for rare words" in c.text for c in chunks)  # references
    fallback = [c for c in chunks if c.record_id == "acl:X00-0001"]
    assert {c.parser for c in fallback} == {Parser.PYMUPDF}
    assert fallback[0].text == "An abstract from the manifest."

    marked = {r.record_id: r.included_in_index for r in read_manifest(config.paths.manifest)}
    assert marked == {"arxiv:0000.00001v1": True, "acl:X00-0001": True, "s2:nopdf": False}

    status = read_jsonl(pipeline.step_dir(config, "chunk") / pipeline.STATUS, IndexStatus)
    assert [s.outcome for s in status] == [
        ParseOutcome.FALLBACK,  # manifest order: acl:..., arxiv:..., s2:...
        ParseOutcome.GROBID,
        ParseOutcome.NO_PDF,
    ]


def test_rerun_sends_nothing_to_grobid_already_parsed(
    config, grobid_service, whitespace_counter
) -> None:
    pipeline.run_all(config)
    posts = grobid_service.calls.call_count
    pipeline.grobid(config)
    # Only the paper GROBID failed on is retried; the cached TEI is reused.
    assert grobid_service.calls.call_count - posts == 2  # isalive + one PDF
    assert stats(config, "grobid")["cached"] == 1


def test_limit_run_does_not_mark_the_manifest(config, grobid_service, whitespace_counter) -> None:
    pipeline.run_all(config, limit=1)
    assert {r.included_in_index for r in read_manifest(config.paths.manifest)} == {None}


def test_grobid_down_gives_a_clear_error(config) -> None:
    with respx.mock() as router:
        router.get(GROBID_URL + "/api/isalive").mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(pipeline.GrobidUnavailable, match="make grobid-up"):
            pipeline.grobid(config)


def test_inspect_paper_lists_chunks_under_sections(
    config, grobid_service, whitespace_counter
) -> None:
    pipeline.run_all(config)
    view = render_paper(config, "arxiv:0000.00001v1")
    assert "outcome=grobid" in view
    assert "2.1 Method > Merge rules" in view
    assert "#0000 abstract" in view
    assert "not in the last parse run" in render_paper(config, "arxiv:missing")
