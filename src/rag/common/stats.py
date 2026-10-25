"""The stats.json every stage step writes next to its outputs."""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag.common.config import Config
from rag.common.io import write_stats
from rag.common.provenance import config_hash, git_commit
from rag.common.schemas import StageStats


@contextmanager
def stage_stats(config: Config, stage: str, out_dir: Path, limit: int | None) -> Iterator[dict]:
    """Collects counts and failures during a step, then writes its stats.json.

    The step fills `counts` (name -> int) and `failures` (StageFailure) in the
    yielded dict; elapsed seconds, git commit and config hash are added here.
    """
    started = datetime.now(UTC)
    tick = time.monotonic()
    collected: dict[str, Any] = {"counts": {}, "failures": []}
    yield collected
    collected["counts"]["seconds"] = round(time.monotonic() - tick)
    write_stats(
        out_dir,
        StageStats(
            stage=stage,
            started_at=started,
            finished_at=datetime.now(UTC),
            limit=limit,
            counts=collected["counts"],
            failures=collected["failures"],
            git_commit=git_commit(),
            config_hash=config_hash(config),
        ),
    )
