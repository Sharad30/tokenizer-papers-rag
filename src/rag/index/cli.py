"""`rag index <kind>`: build the retrieval indexes over the chunks."""

from typing import Annotated

import typer

from rag.common.cli import LimitOption, get_config
from rag.index import pipeline

app = typer.Typer(
    help="Phase 4: BM25 and dense indexes over the chunks -> data/index/.",
    no_args_is_help=True,
)


@app.command()
def bm25(ctx: typer.Context, limit: LimitOption = None) -> None:
    """Chunks -> BM25 index (bm25s, our tokeniser) -> index/bm25/."""
    pipeline.bm25(get_config(ctx), limit)


@app.command()
def dense(
    ctx: typer.Context,
    limit: LimitOption = None,
    model: Annotated[
        str | None, typer.Option(help="One of index.dense_models (default: index.dense_model)")
    ] = None,
) -> None:
    """Chunks -> embeddings (CPU: ~1 h bge-small, ~3 h a base model) -> index/dense/<model>/."""
    config = get_config(ctx)
    if model is not None and model not in config.index.dense_models:
        raise typer.BadParameter(
            f"known: {sorted(config.index.dense_models)}", param_hint="--model"
        )
    pipeline.dense(config, limit, model=model)


@app.command("all")
def all_(ctx: typer.Context, limit: LimitOption = None) -> None:
    """Build both indexes. --limit N indexes the first N papers' chunks."""
    pipeline.run_all(get_config(ctx), limit)
