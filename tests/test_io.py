import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag.common.io import read_jsonl, write_json, write_jsonl, write_stats
from rag.common.schemas import StageFailure, StageStats, StrictModel


class Toy(StrictModel):
    id: str
    tokens: int


def test_jsonl_round_trip(tmp_path: Path) -> None:
    records = [Toy(id="a", tokens=3), Toy(id="b", tokens=0)]
    path = tmp_path / "nested" / "toys.jsonl"
    assert write_jsonl(path, records) == 2
    assert read_jsonl(path, Toy) == records


def test_empty_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    assert write_jsonl(path, []) == 0
    assert read_jsonl(path, Toy) == []


def test_read_jsonl_rejects_schema_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "a", "tokens": 1, "extra": true}\n', encoding="utf-8")
    with pytest.raises(ValidationError):
        read_jsonl(path, Toy)


def test_write_leaves_no_temp_files(tmp_path: Path) -> None:
    write_json(tmp_path / "out.json", {"a": 1})
    write_json(tmp_path / "out.json", {"a": 2})
    assert [p.name for p in tmp_path.iterdir()] == ["out.json"]
    assert json.loads((tmp_path / "out.json").read_text()) == {"a": 2}


def test_write_stats_produces_valid_stage_stats(tmp_path: Path) -> None:
    now = datetime(2026, 10, 16, 12, 0, tzinfo=UTC)
    stats = StageStats(
        stage="parse",
        started_at=now,
        finished_at=now,
        limit=5,
        counts={"papers": 5, "chunks": 120},
        failures=[StageFailure(item_id="2301.00001v2", reason="grobid_timeout")],
        git_commit="abc123",
        config_hash="f" * 64,
    )
    path = write_stats(tmp_path, stats)
    assert path == tmp_path / "stats.json"
    assert StageStats.model_validate_json(path.read_text()) == stats
