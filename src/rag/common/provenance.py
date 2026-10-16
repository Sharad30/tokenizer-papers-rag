"""Identify the code and config that produced an output, so any number is traceable."""

import hashlib
import json
import subprocess
from pathlib import Path

from rag.common.config import Config

UNKNOWN_COMMIT = "unknown"
DIRTY_SUFFIX = "-dirty"


def _git(args: list[str], cwd: Path | None) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


def git_commit(repo_dir: Path | None = None) -> str:
    """HEAD commit hash, with "-dirty" if there are uncommitted changes.

    A dirty marker matters: a result produced from uncommitted code cannot be
    reproduced from the commit hash alone.
    """
    try:
        commit = _git(["rev-parse", "HEAD"], repo_dir)
        dirty = _git(["status", "--porcelain"], repo_dir) != ""
    except (subprocess.CalledProcessError, FileNotFoundError):
        return UNKNOWN_COMMIT
    return commit + DIRTY_SUFFIX if dirty else commit


def config_hash(config: Config) -> str:
    """sha256 of the config's settings, independent of key order in the YAML.

    `paths` is excluded: resolved paths are absolute and differ between
    clones, but where files live does not change what a run computes.
    """
    settings = config.model_dump(mode="json", exclude={"paths"})
    canonical = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
