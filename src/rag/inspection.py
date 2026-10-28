"""`rag inspect ...`: look at intermediate outputs.

Lives outside any one stage because each subcommand reads a different
stage's output. (Not named `inspect.py`, which would read like the stdlib
module of that name.)
"""

from typing import Annotated

import typer

from rag.common.cli import get_config

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
    work_id: Annotated[str, typer.Argument(help="work_id, or any member's record id")],
) -> None:
    """Show one work's member records, its canonical record, and why they were merged."""
    from rag.dedup.show import render_work

    typer.echo(render_work(get_config(ctx), work_id))


@app.command()
def query(
    ctx: typer.Context,
    text: Annotated[str, typer.Argument(help="Query text")],
    k: Annotated[
        int | None, typer.Option("-k", min=1, help="Results (default: retrieve.default_k)")
    ] = None,
    mode: Annotated[
        str | None, typer.Option(help="bm25 | dense | hybrid (default: retrieve.default_mode)")
    ] = None,
) -> None:
    """Rank chunks for a query with BM25, dense and fused scores side by side."""
    from rag.retrieve.retriever import Retriever
    from rag.retrieve.show import render_hits

    config = get_config(ctx)
    mode = mode or config.retrieve.default_mode
    if mode not in ("bm25", "dense", "hybrid"):
        raise typer.BadParameter(f"unknown mode {mode!r}", param_hint="--mode")
    # Load both indexes even for one mode, so the other's rank shows alongside.
    retriever = Retriever.load(config, {"hybrid"})
    typer.echo(render_hits(text, mode, retriever.retrieve(text, k=k, mode=mode)))
