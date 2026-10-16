from pathlib import Path

import pytest
import yaml
from typer.core import TyperGroup
from typer.main import get_command
from typer.testing import CliRunner

from rag.cli import app
from rag.common.config import Config, load_config

runner = CliRunner()


def _command_paths(command: object, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    """Every command path in the CLI tree, e.g. ("inspect", "query").

    Typer 0.27 vendors Click privately (typer._click), so the walk uses
    typer's public TyperGroup rather than importing click.
    """
    paths = [prefix]
    if isinstance(command, TyperGroup):
        for name, sub in command.commands.items():
            paths.extend(_command_paths(sub, (*prefix, name)))
    return paths


# Walking the real command tree means a command added in a later phase is
# covered here automatically, without editing this list.
ALL_COMMANDS = _command_paths(get_command(app))


def test_command_tree_has_expected_commands() -> None:
    top_level = {path[0] for path in ALL_COMMANDS if len(path) == 1}
    assert {"collect", "parse", "dedup", "index", "eval", "report", "chat"} <= top_level
    assert ("inspect", "query") in ALL_COMMANDS
    assert ("config", "show") in ALL_COMMANDS


@pytest.mark.parametrize("path", ALL_COMMANDS, ids=lambda p: " ".join(p) or "rag")
def test_help_works_for_every_command(path: tuple[str, ...], tmp_path: Path, monkeypatch) -> None:
    # Run from an empty directory: help must not need a config file.
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, [*path, "--help"])
    assert result.exit_code == 0, result.output
    assert "Usage" in result.output


def test_config_show_round_trips(repo_config_path: Path) -> None:
    result = runner.invoke(app, ["--config", str(repo_config_path), "config", "show"])
    assert result.exit_code == 0, result.output
    shown = Config.model_validate(yaml.safe_load(result.stdout))
    assert shown == load_config(repo_config_path)


def test_stub_reports_limit(repo_config_path: Path) -> None:
    result = runner.invoke(app, ["--config", str(repo_config_path), "parse", "--limit", "3"])
    assert result.exit_code == 0, result.output
    assert "[stub] rag parse" in result.stdout
    assert "limit=3" in result.stdout


def test_limit_must_be_positive(repo_config_path: Path) -> None:
    result = runner.invoke(app, ["--config", str(repo_config_path), "parse", "--limit", "0"])
    assert result.exit_code == 2


def test_config_path_from_env_var(repo_config_path: Path) -> None:
    result = runner.invoke(app, ["parse"], env={"RAG_CONFIG": str(repo_config_path)})
    assert result.exit_code == 0, result.output


def test_missing_config_fails_with_clear_message(tmp_path: Path) -> None:
    result = runner.invoke(app, ["--config", str(tmp_path / "nope.yaml"), "parse"])
    assert result.exit_code == 1
    assert "Config file not found" in result.stderr
