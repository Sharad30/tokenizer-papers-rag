"""`rag index <kind>`: build the retrieval indexes over the chunks."""

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
def dense(ctx: typer.Context, limit: LimitOption = None) -> None:
    """Chunks -> embeddings (CPU; ~20 min for the full corpus) -> index/dense/<model>/."""
    pipeline.dense(get_config(ctx), limit)


@app.command("all")
def all_(ctx: typer.Context, limit: LimitOption = None) -> None:
    """Build both indexes. --limit N indexes the first N papers' chunks."""
    pipeline.run_all(get_config(ctx), limit)
