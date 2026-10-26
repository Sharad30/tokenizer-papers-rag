from pathlib import Path
from typing import Annotated

import typer

from rag.common.cli import get_config


def report(
    ctx: typer.Context,
    run: Annotated[str, typer.Argument(help="Results file: a path, or <date>-<name> in results/")],
) -> None:
    """Write markdown tables and a figure for one eval run."""
    from rag.report.report import load, write_report

    config = get_config(ctx)
    path = Path(run) if run.endswith(".json") else config.paths.results / f"{run}.json"
    if not path.is_file():
        typer.echo(f"No results file {path}: run `rag eval run --name ...` first.", err=True)
        raise typer.Exit(1)
    for out in write_report(load(path), config.paths.results, config.paths.figures):
        typer.echo(f"Wrote {out}")
