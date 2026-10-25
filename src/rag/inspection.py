"""`rag inspect ...`: look at intermediate outputs.

Lives outside any one stage because each subcommand reads a different
stage's output. (Not named `inspect.py`, which would read like the stdlib
module of that name.)
"""

from typing import Annotated

import typer

from rag.common.cli import get_config, print_stub

app = typer.Typer(
    help="Inspect intermediate outputs: papers, works, queries.", no_args_is_help=True
)


@app.command()
def paper(
    ctx: typer.Context,
    paper_id: Annotated[str, typer.Argument(help="Paper record id from the manifest")],
) -> None:
    """Show one paper's chunks under their section headings."""
    from rag.parse.show import render_paper

    typer.echo(render_paper(get_config(ctx), paper_id))


@app.command()
def work(
    ctx: typer.Context,
    work_id: Annotated[str, typer.Argument(help="work_id from the dedup clusters")],
) -> None:
    """Show one work's member records and why they were merged."""
    config = get_config(ctx)
    print_stub(f"inspect work {work_id}", "5", str(config.paths.works), "terminal", None)


@app.command()
def query(
    ctx: typer.Context,
    text: Annotated[str, typer.Argument(help="Query text")],
) -> None:
    """Rank chunks for a query with BM25, dense, fused and rerank scores side by side."""
    config = get_config(ctx)
    print_stub(f"inspect query {text!r}", "4", str(config.paths.index), "terminal", None)
