from typing import Annotated

import typer

from rag.common.cli import get_config, print_stub


def report(
    ctx: typer.Context,
    run: Annotated[str, typer.Argument(help="Run name: results/<run>.json")],
) -> None:
    """Write markdown tables and figures for one eval run."""
    config = get_config(ctx)
    print_stub(
        "report",
        "4",
        reads=str(config.paths.results / f"{run}.json"),
        writes=str(config.paths.figures),
        limit=None,
    )
