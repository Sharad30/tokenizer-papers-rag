import typer

from rag.common.cli import LimitOption, get_config, print_stub


def index(ctx: typer.Context, limit: LimitOption = None) -> None:
    """Build BM25 and dense indexes over the chunks."""
    config = get_config(ctx)
    print_stub(
        "index",
        "4/6",
        reads=f"{config.paths.parsed} (+ {config.paths.works})",
        writes=str(config.paths.index),
        limit=limit,
    )
