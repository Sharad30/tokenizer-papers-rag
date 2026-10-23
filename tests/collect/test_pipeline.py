"""End-to-end run of `rag collect all` against mocked sources (all papers synthetic)."""

import csv
import json
import random
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx
import numpy as np
import pytest
import respx

from rag.collect import pipeline
from rag.collect.manifest import read_manifest
from rag.collect.schemas import DecidedBy, Decision, DownloadStatus, ManifestRecord, Reason, Source
from rag.common.config import load_config

ATOM = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/" xmlns="http://www.w3.org/2005/Atom">
<opensearch:totalResults>{total}</opensearch:totalResults>{entries}</feed>"""
ENTRY = """<entry><id>http://arxiv.org/abs/{id}</id><title>{title}</title>
<summary>{abstract}</summary><published>2021-01-01T00:00:00Z</published>
<author><name>Ada Example</name></author></entry>"""

# Synthetic arXiv papers by versioned id. The unversioned id returns the latest.
ARXIV = {
    "2401.00001v2": ("Rethinking BPE Merges", "We study how bpe merges are learned."),
    "2401.00001v1": ("Rethinking BPE Merges", "An early draft about something else entirely."),
    "2401.00002v1": ("Graph Networks", "Message passing on molecules."),
    "2401.00003v1": ("A Subword Study", "Subword units compared."),
    "0000.00000v1": ("Anchor Tokenizer Paper", "The anchor."),
}
LATEST = {"2401.00001": "2401.00001v2", "2401.00003": "2401.00003v1", "0000.00000": "0000.00000v1"}

S2_BY_LOOKUP = {
    "ARXIV:0000.00000": {
        "paperId": "anchor",
        "title": "Anchor Tokenizer Paper",
        "externalIds": {"ArXiv": "0000.00000"},
        "year": 2020,
    },
    "ARXIV:2401.00001": {
        "paperId": "p1",
        "title": "Rethinking BPE Merges",
        "externalIds": {"ArXiv": "2401.00001"},
        "venue": "SynthConf",
        "year": 2024,
    },
}
REFERENCES = [
    {
        "paperId": "p1",
        "title": "Rethinking BPE Merges",
        "externalIds": {"ArXiv": "2401.00001"},
        "year": 2024,
    },
    {
        "paperId": "p3",
        "title": "A Subword Study",
        "externalIds": {"ArXiv": "2401.00003"},
        "year": 2021,
    },
    {"paperId": "p4", "title": "Cooking With Data", "abstract": "Recipes.", "year": 2021},
    {
        "paperId": "p6",
        "title": "Summarising News",
        "abstract": "We train with a bpe tokenizer. The tokenizer matters.",
        "year": 2021,
    },
    {"paperId": "p5", "title": "A Future Tokenizer", "year": 2030},  # after the cutoff
]


def arxiv_response(request: httpx.Request) -> httpx.Response:
    params = parse_qs(urlsplit(str(request.url)).query)
    if "id_list" in params:
        ids = [LATEST.get(i, i) for i in params["id_list"][0].split(",")]
    else:  # a search: the seed query finds papers 1 and 2
        ids = ["2401.00001v2", "2401.00002v1"]
    entries = "".join(
        ENTRY.format(id=i, title=ARXIV[i][0], abstract=ARXIV[i][1]) for i in ids if i in ARXIV
    )
    return httpx.Response(200, text=ATOM.format(total=len(ids), entries=entries))


def oai_response(request: httpx.Request) -> httpx.Response:
    arxiv_id = parse_qs(urlsplit(str(request.url)).query)["identifier"][0].split(":")[-1]
    return httpx.Response(
        200,
        text=f"""<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><GetRecord><record><metadata>
        <arXivRaw xmlns="http://arxiv.org/OAI/arXivRaw/"><id>{arxiv_id}</id>
        <license>http://creativecommons.org/licenses/by/4.0/</license></arXivRaw>
        </metadata></record></GetRecord></OAI-PMH>""",
    )


def batch_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200, json=[S2_BY_LOOKUP.get(i) for i in json.loads(request.content)["ids"]]
    )


@pytest.fixture
def config(tmp_path: Path, minimal_config_text: str):
    # No waiting in tests: zero rate-limit gaps and backoff.
    text = minimal_config_text.replace("default_min_interval_s: 0.5", "default_min_interval_s: 0")
    text = text.replace("backoff_base_s: 1", "backoff_base_s: 0")
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return load_config(path)


@pytest.fixture
def mocked_sources():
    with respx.mock(assert_all_called=False) as router:
        router.get("https://export.arxiv.org/api/query").mock(side_effect=arxiv_response)
        router.get("https://oaipmh.arxiv.org/oai").mock(side_effect=oai_response)
        router.get(url__regex=r"https://export\.arxiv\.org/pdf/.*").mock(
            return_value=httpx.Response(200, content=b"%PDF-1.7 synthetic")
        )
        router.post("https://api.semanticscholar.org/graph/v1/paper/batch").mock(
            side_effect=batch_response
        )
        router.get(
            "https://api.semanticscholar.org/graph/v1/paper/ARXIV:0000.00000/references"
        ).mock(
            return_value=httpx.Response(
                200, json={"offset": 0, "data": [{"citedPaper": p} for p in REFERENCES]}
            )
        )
        yield router


@pytest.fixture
def fake_encoder(monkeypatch):
    """Every ambiguous paper scores 0.1, below the fixture threshold of 0.5."""

    def encode(texts: list[str]) -> np.ndarray:
        # The fixture's topic description maps to [1, 0]; every paper to a
        # unit vector whose dot product with it is 0.1.
        return np.array(
            [[1.0, 0.0] if t == "Tokenization research." else [0.1, 0.995] for t in texts]
        )

    def factory(model_name: str, batch_size: int):
        return encode

    monkeypatch.setattr(pipeline, "sentence_transformer_encoder", factory)


def by_id(records: list[ManifestRecord]) -> dict[str, ManifestRecord]:
    return {r.record_id: r for r in records}


def test_collect_all_end_to_end(config, mocked_sources, fake_encoder) -> None:
    pipeline.run_all(config)
    records = by_id(read_manifest(config.paths.manifest))

    assert set(records) == {
        "arxiv:2401.00001v2",  # seed query, also reached by snowball (folded in)
        "arxiv:2401.00001v1",  # revised v1 kept by D-009
        "arxiv:2401.00002v1",  # seed query, rejected
        "arxiv:2401.00003v1",  # S2-only find, materialised as its arXiv record
        "arxiv:0000.00000v1",  # the anchor itself, materialised
        "s2:p4",  # snowball find, rejected, stays an S2 record
        "s2:p6",  # ambiguous, rejected by the embedding judge
    }  # p5 is after corpus.max_date and never appears

    main = records["arxiv:2401.00001v2"]
    assert (main.relevance, main.reason) == (Decision.ACCEPT, Reason.TITLE_TERM)
    assert main.found_via == ["arxiv_query:bpe", "snowball_ref:ARXIV:0000.00000"]
    assert (main.s2_id, main.venue) == ("p1", "SynthConf")
    assert main.licence == "http://creativecommons.org/licenses/by/4.0/"
    assert main.download_status is DownloadStatus.DOWNLOADED

    v1 = records["arxiv:2401.00001v1"]
    assert (v1.decided_by, v1.reason, v1.derived_from) == (
        DecidedBy.INHERITED,
        Reason.EARLIER_VERSION,
        "arxiv:2401.00001v2",
    )

    materialised = records["arxiv:2401.00003v1"]
    assert materialised.source is Source.ARXIV
    assert materialised.found_via == ["snowball_ref:ARXIV:0000.00000"]
    assert materialised.s2_id == "p3"
    assert materialised.reason is Reason.TITLE_TERM  # its own decision, carried over

    assert records["s2:p4"].reason is Reason.NO_TOPIC_TERM
    judged = records["s2:p6"]
    assert (judged.decided_by, judged.reason) == (DecidedBy.EMBEDDING, Reason.EMBEDDING_BELOW)
    assert judged.embedding_score == pytest.approx(0.1)
    assert records["arxiv:2401.00002v1"].download_status is DownloadStatus.NOT_ATTEMPTED

    stats = json.loads((config.paths.manifest / "stats.json").read_text())
    assert stats["counts"]["accepted"] == 4
    for step in pipeline.STEP_DIRS:
        assert (pipeline.step_dir(config, step) / "stats.json").is_file()


def test_rerun_makes_no_network_calls(config, mocked_sources, fake_encoder) -> None:
    pipeline.run_all(config)
    calls_after_first = len(mocked_sources.calls)
    pipeline.run_all(config)
    assert len(mocked_sources.calls) == calls_after_first
    for step in ("discover", "resolve", "filter", "licence", "download"):
        stats = json.loads((pipeline.step_dir(config, step) / "stats.json").read_text())
        assert stats["counts"].get("network_calls", 0) == 0


def test_limit_caps_downloads(config, mocked_sources, fake_encoder) -> None:
    pipeline.run_all(config, limit=1)
    stats = json.loads((pipeline.step_dir(config, "download") / "stats.json").read_text())
    assert stats["counts"]["attempted"] <= 1


def test_audit_sample_is_stratified_seeded_and_skips_inherited(
    config, mocked_sources, fake_encoder
) -> None:
    pipeline.run_all(config)
    path = pipeline.audit_sample(config, per_decision=2, seed=0)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert [r["relevance"] for r in rows].count("accept") == 2
    assert [r["relevance"] for r in rows].count("reject") == 2
    assert "arxiv:2401.00001v1" not in {r["record_id"] for r in rows}  # inherited v1
    assert {r["sharad_in_scope"] for r in rows} == {""}
    # Same seed, same sample.
    again = pipeline.audit_sample(config, per_decision=2, seed=0)
    assert again.read_text() == path.read_text()


def test_stratified_round_robins_sources() -> None:
    pool = [
        ManifestRecord(record_id=f"arxiv:{i}", source=Source.ARXIV, title="t") for i in range(10)
    ]
    pool.append(ManifestRecord(record_id="acl:x", source=Source.ACL, title="t"))
    picked = pipeline._stratified(pool, 2, random.Random(0))
    assert {r.source for r in picked} == {Source.ARXIV, Source.ACL}


def test_linked_copies_fetches_missing_arxiv_copy_of_accepted_acl_paper(monkeypatch) -> None:
    # Regression: an ACL record's own arxiv_id used to count as "arXiv copy
    # present", so the arXiv record was never fetched.
    accepted_acl = ManifestRecord(
        record_id="acl:2022.acl-long.1",
        source=Source.ACL,
        title="Tokenizer Choice Matters",
        arxiv_id="2201.00001",
        acl_id="2022.acl-long.1",
        relevance=Decision.ACCEPT,
        decided_by=DecidedBy.KEYWORD,
        reason=Reason.TITLE_TERM,
        found_via=["acl_scan"],
    )
    arxiv_copy = ManifestRecord(
        record_id="arxiv:2201.00001v2",
        source=Source.ARXIV,
        title="Tokenizer Choice Matters",
        arxiv_id="2201.00001",
    )
    monkeypatch.setattr(
        pipeline, "_fetch_arxiv", lambda client, ids: [arxiv_copy] if ids == ["2201.00001"] else []
    )
    copies = pipeline._linked_copies(client=None, anthology=None, records=[accepted_acl])
    assert [c.record_id for c in copies] == ["arxiv:2201.00001v2"]
    assert (copies[0].decided_by, copies[0].reason, copies[0].derived_from) == (
        DecidedBy.INHERITED,
        Reason.LINKED_MANIFESTATION,
        "acl:2022.acl-long.1",
    )
    assert copies[0].found_via == ["acl_scan"]


def test_linked_copies_skips_copy_already_present(monkeypatch) -> None:
    acl_record = ManifestRecord(
        record_id="acl:x",
        source=Source.ACL,
        title="T",
        arxiv_id="2201.00001",
        relevance=Decision.ACCEPT,
    )
    arxiv_record = ManifestRecord(
        record_id="arxiv:2201.00001v1", source=Source.ARXIV, title="T", arxiv_id="2201.00001"
    )
    monkeypatch.setattr(
        pipeline,
        "_fetch_arxiv",
        lambda client, ids: pytest.fail(f"unexpected fetch {ids}") if ids else [],
    )
    assert (
        pipeline._linked_copies(client=None, anthology=None, records=[acl_record, arxiv_record])
        == []
    )
