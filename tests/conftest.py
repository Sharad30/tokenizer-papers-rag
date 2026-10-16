from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def repo_config_path() -> Path:
    return REPO_ROOT / "config.yaml"


@pytest.fixture
def minimal_config_text() -> str:
    return (FIXTURES / "config_minimal.yaml").read_text(encoding="utf-8")


@pytest.fixture
def write_config(tmp_path: Path):
    """Write YAML text to a config file in a temp dir and return its path."""

    def _write(text: str) -> Path:
        path = tmp_path / "config.yaml"
        path.write_text(text, encoding="utf-8")
        return path

    return _write
