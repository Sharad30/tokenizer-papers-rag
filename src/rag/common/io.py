"""Reading and writing stage outputs.

All writes are atomic (write a temp file, then rename over the target), so a
stage killed halfway never leaves a truncated file that the next stage would
read as if it were complete.
"""

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from rag.common.schemas import StageStats

STATS_FILENAME = "stats.json"

ModelT = TypeVar("ModelT", bound=BaseModel)


def write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # The temp file must be in the same directory: os.replace is only atomic
    # within one filesystem.
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def write_text_atomic(path: Path, text: str) -> None:
    write_bytes_atomic(path, text.encode("utf-8"))


def write_json(path: Path, data: BaseModel | dict) -> None:
    payload = data.model_dump(mode="json") if isinstance(data, BaseModel) else data
    write_text_atomic(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> int:
    """Write one JSON object per line; returns the number of records written."""
    lines = [record.model_dump_json() for record in records]
    write_text_atomic(path, "".join(line + "\n" for line in lines))
    return len(lines)


def read_jsonl(path: Path, model: type[ModelT]) -> list[ModelT]:
    """Read and validate every line, so a schema mismatch fails at the stage boundary."""
    with path.open(encoding="utf-8") as handle:
        return [model.model_validate_json(line) for line in handle if line.strip()]


def write_stats(out_dir: Path, stats: StageStats) -> Path:
    path = out_dir / STATS_FILENAME
    write_json(path, stats)
    return path
