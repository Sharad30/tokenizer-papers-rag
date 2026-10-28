"""`rag dedup <step>`: resolve paper records into works."""

import typer

from rag.common.cli import LimitOption, get_config
from rag.dedup import pipeline

app = typer.Typer(
    help="Phase 5: cluster paper records into works (versions, venue copies) -> data/works/.",
    no_args_is_help=True,
)


@app.command()
def run(ctx: typer.Context, limit: LimitOption = None) -> None:
    """Manifest + chunks -> works (IDs, titles, MinHash text, chain guard) -> works/."""
    pipeline.run(get_config(ctx), limit)
