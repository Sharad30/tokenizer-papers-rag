"""`rag parse <step>`: one subcommand per pipeline step, plus `all`."""

import typer

from rag.common.cli import LimitOption, get_config
from rag.parse import pipeline

app = typer.Typer(
    help="Phase 2: PDFs to structured papers (GROBID TEI) to section-aware chunks.",
    no_args_is_help=True,
)


@app.command()
def grobid(ctx: typer.Context, limit: LimitOption = None) -> None:
    """Accepted PDFs -> TEI XML via the local GROBID service -> parsed/tei/."""
    pipeline.grobid(get_config(ctx), limit)


@app.command()
def structure(ctx: typer.Context, limit: LimitOption = None) -> None:
    """TEI (or the PyMuPDF fallback) -> structured papers + references -> parsed/papers/."""
    pipeline.structure(get_config(ctx), limit)


@app.command()
def chunk(ctx: typer.Context, limit: LimitOption = None) -> None:
    """Structured papers -> section-aware chunks -> parsed/chunks/."""
    pipeline.chunk(get_config(ctx), limit)


@app.command()
def mark(ctx: typer.Context, limit: LimitOption = None) -> None:
    """Write included_in_index into the manifest (skipped after a --limit chunk run)."""
    pipeline.mark(get_config(ctx), limit)


@app.command("all")
def all_(ctx: typer.Context, limit: LimitOption = None) -> None:
    """Run every step in order. GROBID output is cached, so reruns are cheap."""
    pipeline.run_all(get_config(ctx), limit)
