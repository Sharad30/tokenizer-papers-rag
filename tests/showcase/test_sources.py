import os
from pathlib import Path

from rag.showcase.sources import (
    fingerprint,
    read_snapshot,
    snapshot_file,
    stale_sources,
    write_snapshot,
)


def test_fingerprint_changes_when_a_file_changes(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("one", encoding="utf-8")
    before = fingerprint([path])
    path.write_text("three", encoding="utf-8")
    os.utime(path, ns=(before[0][1] + 1_000_000, before[0][1] + 1_000_000))
    assert fingerprint([path]) != before


def test_fingerprint_of_a_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"
    assert fingerprint([missing]) == ((str(missing), -1, -1),)


def test_snapshot_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "rejected.csv.gz"
    source.write_bytes(b"rows")
    path = snapshot_file(tmp_path / "snapshot", "collect")
    written = write_snapshot(
        path, "collect", {"rejected": source, "absent": tmp_path / "x"}, {"a": [1]}, "abc"
    )
    loaded = read_snapshot(path)
    assert loaded == written
    assert loaded.data == {"a": [1]}
    # A source missing at build time isn't recorded, so it can't look stale later.
    assert set(loaded.sources) == {"rejected"}


def test_read_snapshot_when_none_exists(tmp_path: Path) -> None:
    assert read_snapshot(tmp_path / "collect.json") is None


def test_stale_sources_flags_only_changed_files(tmp_path: Path) -> None:
    same, changed = tmp_path / "same", tmp_path / "changed"
    same.write_bytes(b"1")
    changed.write_bytes(b"2")
    snapshot = write_snapshot(
        tmp_path / "s.json", "collect", {"same": same, "changed": changed}, {}, "abc"
    )
    changed.write_bytes(b"22")
    gone = tmp_path / "gone"
    assert stale_sources(snapshot, {"same": same, "changed": changed, "gone": gone}) == ["changed"]
