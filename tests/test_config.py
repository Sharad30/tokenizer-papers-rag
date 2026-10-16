from datetime import date
from pathlib import Path

import pytest

from rag.common.config import ConfigError, load_config


def test_repo_config_loads(repo_config_path: Path) -> None:
    config = load_config(repo_config_path)
    assert config.logging.level == "INFO"
    assert isinstance(config.corpus.max_date, date)


def test_paths_resolve_against_config_directory(write_config, minimal_config_text: str) -> None:
    path = write_config(minimal_config_text)
    config = load_config(path)
    assert config.paths.raw == (path.parent / "out" / "raw").resolve()
    assert all(p.is_absolute() for p in config.paths.model_dump().values())


def test_unknown_key_is_rejected(write_config, minimal_config_text: str) -> None:
    path = write_config(minimal_config_text + "chunk_szie: 300\n")
    with pytest.raises(ConfigError, match="chunk_szie"):
        load_config(path)


def test_unknown_nested_key_is_rejected(write_config, minimal_config_text: str) -> None:
    text = minimal_config_text.replace("level: DEBUG", "level: DEBUG\n  colour: true")
    with pytest.raises(ConfigError, match="colour"):
        load_config(write_config(text))


def test_wrong_type_is_rejected(write_config, minimal_config_text: str) -> None:
    text = minimal_config_text.replace("level: DEBUG", "level: LOUD")
    with pytest.raises(ConfigError, match="level"):
        load_config(write_config(text))


def test_missing_section_is_rejected(write_config, minimal_config_text: str) -> None:
    text = minimal_config_text.split("corpus:")[0]
    with pytest.raises(ConfigError, match="corpus"):
        load_config(write_config(text))


def test_missing_file_gives_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


@pytest.mark.parametrize("text", ["", "# only a comment\n", "- a\n- list\n"])
def test_empty_or_non_mapping_file_is_rejected(write_config, text: str) -> None:
    with pytest.raises(ConfigError, match="mapping"):
        load_config(write_config(text))


def test_invalid_yaml_is_rejected(write_config) -> None:
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(write_config("paths: [unclosed\n"))
