import subprocess
from pathlib import Path

from rag.common.config import load_config
from rag.common.provenance import UNKNOWN_COMMIT, config_hash, git_commit


def _git(repo: Path, *args: str) -> str:
    # Identity passed inline so the test doesn't depend on the machine's git config.
    cmd = ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args]
    return subprocess.run(cmd, cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


def test_git_commit_clean_then_dirty(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    (tmp_path / "a.txt").write_text("one")
    _git(tmp_path, "add", "a.txt")
    _git(tmp_path, "commit", "-q", "-m", "first")
    head = _git(tmp_path, "rev-parse", "HEAD")

    assert git_commit(tmp_path) == head
    (tmp_path / "a.txt").write_text("two")
    assert git_commit(tmp_path) == head + "-dirty"


def test_git_commit_outside_repo(tmp_path: Path, monkeypatch) -> None:
    # Stop git searching parent directories, so the test holds even if
    # tmp_path happens to sit inside some repository.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert git_commit(tmp_path) == UNKNOWN_COMMIT


def test_config_hash_ignores_key_order_and_paths(tmp_path: Path, minimal_config_text: str) -> None:
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    (a_dir / "config.yaml").write_text(minimal_config_text)
    # Same settings, sections in reverse order, in a different directory.
    sections = minimal_config_text.split("\n" + "logging:")
    reordered = "logging:" + sections[1] + "\n" + sections[0]
    (b_dir / "config.yaml").write_text(reordered)

    config_a = load_config(a_dir / "config.yaml")
    config_b = load_config(b_dir / "config.yaml")
    assert config_a.paths != config_b.paths
    assert config_hash(config_a) == config_hash(config_b)


def test_config_hash_changes_with_settings(write_config, minimal_config_text: str) -> None:
    before = config_hash(load_config(write_config(minimal_config_text)))
    changed = minimal_config_text.replace("level: DEBUG", "level: INFO")
    after = config_hash(load_config(write_config(changed)))
    assert before != after
