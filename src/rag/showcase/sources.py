"""Where a showcase page's data comes from: stage outputs, or a committed snapshot.

Pages compute everything from the stage outputs when they exist (D-038).
Outputs a fresh clone doesn't have come from a snapshot, written by
`rag showcase snapshot` with the same functions. The snapshot records the
sha256 of each file it was built from, so a page can say when the data
has moved on since.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag.common.io import file_sha256, write_text_atomic
from rag.common.schemas import StrictModel

FileStamp = tuple[str, int, int]


def fingerprint(paths: list[Path]) -> tuple[FileStamp, ...]:
    """(path, mtime_ns, size) per file; -1s when missing.

    Passed as an argument to the UI's cached loaders: a changed file gives a
    different fingerprint, so the loader runs again instead of serving
    stale numbers.
    """
    stamps = []
    for path in paths:
        stat = path.stat() if path.exists() else None
        stamps.append((str(path), stat.st_mtime_ns if stat else -1, stat.st_size if stat else -1))
    return tuple(stamps)


class Snapshot(StrictModel):
    stage: str
    created_at: datetime
    git_commit: str
    sources: dict[str, str]  # source name -> sha256 of the file it was built from
    data: dict[str, Any]


def snapshot_file(snapshot_dir: Path, stage: str) -> Path:
    return snapshot_dir / f"{stage}.json"


def write_snapshot(
    path: Path, stage: str, sources: dict[str, Path], data: dict[str, Any], git_commit: str
) -> Snapshot:
    snapshot = Snapshot(
        stage=stage,
        created_at=datetime.now(UTC),
        git_commit=git_commit,
        sources={name: file_sha256(p) for name, p in sources.items() if p.exists()},
        data=data,
    )
    # Sorted keys and fixed indent, so a regenerated snapshot diffs cleanly in git.
    text = json.dumps(snapshot.model_dump(mode="json"), indent=1, sort_keys=True)
    write_text_atomic(path, text + "\n")
    return snapshot


def read_snapshot(path: Path) -> Snapshot | None:
    if not path.exists():
        return None
    return Snapshot.model_validate_json(path.read_text(encoding="utf-8"))


def stale_sources(snapshot: Snapshot, sources: dict[str, Path]) -> list[str]:
    """Sources present now whose content differs from what the snapshot was built from."""
    recorded = snapshot.sources
    return [
        name
        for name, path in sources.items()
        if path.exists() and name in recorded and file_sha256(path) != recorded[name]
    ]
