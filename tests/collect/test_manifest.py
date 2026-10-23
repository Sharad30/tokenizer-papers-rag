import csv
from pathlib import Path

from rag.collect.manifest import (
    MANIFEST_CSV,
    REJECTED_CSV_GZ,
    manifest_counts,
    read_manifest,
    read_rejected,
    write_manifest,
)
from rag.collect.schemas import (
    DecidedBy,
    Decision,
    DownloadStatus,
    LicenceSource,
    ManifestRecord,
    Reason,
    Source,
)

RECORDS = [
    ManifestRecord(
        record_id="arxiv:2401.00002v1",
        source=Source.ARXIV,
        title="B, with a comma",
        authors=["Ada Example", "Ben Sample"],
        found_via=["arxiv_query:bpe", "snowball_cit:ARXIV:1508.07909"],
        relevance=Decision.ACCEPT,
        decided_by=DecidedBy.KEYWORD,
        reason=Reason.TITLE_TERM,
        licence="http://creativecommons.org/licenses/by/4.0/",
        licence_source=LicenceSource.ARXIV_OAI,
        download_status=DownloadStatus.DOWNLOADED,
    ),
    ManifestRecord(
        record_id="acl:2022.acl-long.1",
        source=Source.ACL,
        title="A",
        found_via=["acl_scan"],
        relevance=Decision.ACCEPT,
        decided_by=DecidedBy.KEYWORD,
        reason=Reason.TITLE_TERM,
        licence_source=LicenceSource.ACL_YEAR_RULE,
    ),
    ManifestRecord(
        record_id="s2:zzz",
        source=Source.S2,
        title="C",
        found_via=["snowball_cit:ARXIV:1508.07909"],
        relevance=Decision.REJECT,
        decided_by=DecidedBy.KEYWORD,
        reason=Reason.NO_TOPIC_TERM,
    ),
]


def test_json_round_trip_holds_accepted_sorted(tmp_path: Path) -> None:
    write_manifest(RECORDS, tmp_path)
    loaded = read_manifest(tmp_path)
    assert [r.record_id for r in loaded] == ["acl:2022.acl-long.1", "arxiv:2401.00002v1"]
    assert loaded == sorted(RECORDS[:2], key=lambda r: r.record_id)


def test_rejects_go_to_gzipped_csv_without_abstracts(tmp_path: Path) -> None:
    reject = RECORDS[2].model_copy(update={"abstract": "Long text."})
    write_manifest([*RECORDS[:2], reject], tmp_path)
    rows = read_rejected(tmp_path)
    assert [r["record_id"] for r in rows] == ["s2:zzz"]
    assert rows[0]["reason"] == "no_topic_term"
    assert "abstract" not in rows[0]


def test_csv_matches_json(tmp_path: Path) -> None:
    write_manifest(RECORDS, tmp_path)
    with (tmp_path / MANIFEST_CSV).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["record_id"] for r in rows] == [r.record_id for r in read_manifest(tmp_path)]
    arxiv_row = rows[1]
    assert arxiv_row["title"] == "B, with a comma"
    assert arxiv_row["authors"] == "Ada Example; Ben Sample"
    assert arxiv_row["relevance"] == "accept"
    assert set(rows[0]) == set(ManifestRecord.model_fields)


def test_rewriting_is_byte_identical(tmp_path: Path) -> None:
    write_manifest(RECORDS, tmp_path)
    first = [(tmp_path / name).read_bytes() for name in (MANIFEST_CSV, REJECTED_CSV_GZ)]
    write_manifest(list(reversed(RECORDS)), tmp_path)
    assert [(tmp_path / name).read_bytes() for name in (MANIFEST_CSV, REJECTED_CSV_GZ)] == first


def test_manifest_counts() -> None:
    counts = manifest_counts(RECORDS)
    assert counts["records"] == 3
    assert counts["accepted"] == 2
    assert counts["source.arxiv"] == 1
    assert counts["decision.reject.no_topic_term"] == 1
    assert counts["accepted_routes.arxiv_query+snowball_cit"] == 1
    assert counts["accepted_routes.acl_scan"] == 1
    assert counts["accepted_licence_source.acl_year_rule"] == 1
    assert counts["download.downloaded"] == 1
    assert counts["download.not_attempted"] == 2
