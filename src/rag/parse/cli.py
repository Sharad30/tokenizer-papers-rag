import typer

from rag.common.cli import LimitOption, get_config, print_stub


def parse(ctx: typer.Context, limit: LimitOption = None) -> None:
    """Parse PDFs with GROBID and write section-aware chunks and references."""
    config = get_config(ctx)
    print_stub(
        "parse", "2", reads=str(config.paths.raw), writes=str(config.paths.parsed), limit=limit
    )
