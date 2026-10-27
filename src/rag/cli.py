"""The `rag` command: one subcommand per pipeline stage, plus inspect and config.

This module only wires commands together. Each stage's command lives in
`rag/<stage>/cli.py`, next to the code it runs.
"""

from pathlib import Path
from typing import Annotated, Literal

import typer
import yaml

from rag import inspection
from rag.chat.cli import chat
from rag.collect import cli as collect_cli
from rag.common.cli import AppState, get_config
from rag.common.config import DEFAULT_CONFIG_PATH
from rag.dedup.cli import dedup
from rag.eval import cli as eval_cli
from rag.index import cli as index_cli
from rag.parse import cli as parse_cli
from rag.report.cli import report
from rag.showcase import cli as showcase_cli

app = typer.Typer(
    help="RAG over research papers on LLM tokenizers.",
    no_args_is_help=True,
)
config_app = typer.Typer(help="Show or check the configuration.", no_args_is_help=True)


@app.callback()
def root(
    ctx: typer.Context,
    config: Annotated[
        Path,
        typer.Option(envvar="RAG_CONFIG", help="Path to the config file."),
    ] = DEFAULT_CONFIG_PATH,
    log_level: Annotated[
        Literal["DEBUG", "INFO", "WARNING", "ERROR"] | None,
        typer.Option(help="Override logging.level from the config."),
    ] = None,
) -> None:
    ctx.obj = AppState(config_path=config, log_level=log_level)


@config_app.command("show")
def config_show(ctx: typer.Context) -> None:
    """Print the validated config (paths resolved) as YAML."""
    config = get_config(ctx)
    typer.echo(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), nl=False)


# Registered in pipeline order, which is also the order `rag --help` lists them.
app.add_typer(collect_cli.app, name="collect")
app.add_typer(parse_cli.app, name="parse")
app.command()(dedup)
app.add_typer(index_cli.app, name="index")
app.add_typer(eval_cli.app, name="eval")
app.command()(report)
app.command()(chat)
app.add_typer(inspection.app, name="inspect")
app.add_typer(showcase_cli.app, name="showcase")
app.add_typer(config_app, name="config")


def main() -> None:
    app()
