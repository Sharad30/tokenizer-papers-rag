import typer

from rag.common.cli import LimitOption, get_config, print_stub


def dedup(ctx: typer.Context, limit: LimitOption = None) -> None:
    """Cluster paper records into works (versions, venue copies) and flag restatements."""
    config = get_config(ctx)
    print_stub(
        "dedup", "5", reads=str(config.paths.parsed), writes=str(config.paths.works), limit=limit
    )
