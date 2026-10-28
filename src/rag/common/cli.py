"""CLI plumbing shared by every command: config access and the --limit option."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import typer

from rag.common.config import Config, ConfigError, load_config
from rag.common.logging import setup_logging

LimitOption = Annotated[
    int | None,
    typer.Option(min=1, help="Process at most N items, for fast end-to-end runs. Default: all."),
]


@dataclass
class AppState:
    """Stored on the Typer context by the root callback.

    The config is loaded lazily on first use rather than in the callback,
    because Click runs the root callback even for `rag <cmd> --help`, and help
    should work without a valid config.yaml in the current directory.
    """

    config_path: Path
    log_level: str | None
    _config: Config | None = field(default=None, repr=False)

    def config(self) -> Config:
        if self._config is None:
            try:
                self._config = load_config(self.config_path)
            except ConfigError as exc:
                typer.echo(f"Error: {exc}", err=True)
                raise typer.Exit(code=1) from exc
            setup_logging(self.log_level or self._config.logging.level)
        return self._config


def get_config(ctx: typer.Context) -> Config:
    state = ctx.find_object(AppState)
    if state is None:
        raise RuntimeError("AppState missing: commands must run under the root `rag` app.")
    return state.config()
