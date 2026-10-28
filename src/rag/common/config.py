"""Typed configuration loaded from config.yaml.

Every tunable value in the project lives in config.yaml and is validated
here. Each stage keeps its own section model next to its code (e.g.
`rag.collect.config.CollectConfig`) and adds a field on `Config` for it.
"""

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import ValidationError

from rag.chat.config import ChatConfig
from rag.collect.config import CollectConfig
from rag.common.schemas import StrictModel
from rag.dedup.config import DedupConfig
from rag.eval.config import EvalConfig
from rag.index.config import IndexConfig
from rag.parse.config import ParseConfig
from rag.retrieve.config import RetrieveConfig
from rag.showcase.config import ShowcaseConfig

DEFAULT_CONFIG_PATH = Path("config.yaml")


class ConfigError(Exception):
    """config.yaml is missing, is not valid YAML, or fails validation."""


class PathsConfig(StrictModel):
    """Stage folders. Relative paths are resolved against the config file's directory."""

    raw: Path
    manifest: Path
    parsed: Path
    works: Path
    index: Path
    eval: Path
    results: Path
    figures: Path
    chat: Path
    prompts: Path
    snapshot: Path
    probes: Path

    def resolved_against(self, base: Path) -> "PathsConfig":
        # Resolving against the config file (not the current directory) means
        # `rag` behaves the same whichever directory it is run from.
        return PathsConfig(
            **{name: (base / value).resolve() for name, value in self.model_dump().items()}
        )


class LoggingConfig(StrictModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class CorpusConfig(StrictModel):
    # Pinned so that re-running collection later returns the same candidate set.
    max_date: date


class Config(StrictModel):
    paths: PathsConfig
    logging: LoggingConfig
    corpus: CorpusConfig
    collect: CollectConfig
    parse: ParseConfig
    dedup: DedupConfig
    eval: EvalConfig
    index: IndexConfig
    retrieve: RetrieveConfig
    chat: ChatConfig
    showcase: ShowcaseConfig
    # TODO(sharad): Phase 0 "You do": add one config field of your own here
    # (and to config.yaml), then use it in the `rag collect` stub in
    # src/rag/collect/cli.py.


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    """Read, validate and path-resolve a config file.

    Raises ConfigError with a readable message for any problem, so the CLI can
    report it in one line instead of a traceback.
    """
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Config file is not valid YAML: {path}\n{exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Config file must contain a YAML mapping: {path}")
    try:
        config = Config.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid config in {path}:\n{exc}") from exc
    return config.model_copy(update={"paths": config.paths.resolved_against(path.resolve().parent)})
