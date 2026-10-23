"""`rag collect <step>`: one subcommand per pipeline step, plus `all` and `audit-sample`."""

import typer

from rag.collect import pipeline
from rag.common.cli import LimitOption, get_config

app = typer.Typer(
    help="Phase 1: find, filter and download papers; write the corpus manifest.",
    no_args_is_help=True,
)


@app.callback()
def collect(ctx: typer.Context) -> None:
    # TODO(sharad): Phase 0 "You do": read the config field you added in
    # src/rag/common/config.py (get_config(ctx).<section>.<field>) and print
    # it here. This callback runs before every `rag collect` subcommand.
    pass


@app.command()
def discover(ctx: typer.Context, limit: LimitOption = None) -> None:
    """arXiv seed queries + ACL scan + Semantic Scholar snowball -> raw/discovered/."""
    pipeline.discover(get_config(ctx), limit)


@app.command()
def resolve(ctx: typer.Context, limit: LimitOption = None) -> None:
    """Link records to Semantic Scholar ids; fold duplicate S2 candidates -> raw/resolved/."""
    pipeline.resolve(get_config(ctx), limit)


@app.command("filter")
def filter_(ctx: typer.Context, limit: LimitOption = None) -> None:
    """Keyword rules + embedding judge; add linked copies and revised v1s -> raw/filtered/."""
    pipeline.filter_records(get_config(ctx), limit)


@app.command()
def licence(ctx: typer.Context, limit: LimitOption = None) -> None:
    """arXiv licences from OAI-PMH (one call per accepted arXiv paper) -> raw/licensed/."""
    pipeline.licence(get_config(ctx), limit)


@app.command()
def download(ctx: typer.Context, limit: LimitOption = None) -> None:
    """PDFs for accepted records (resumable) -> raw/pdfs/, raw/downloaded/."""
    pipeline.download_pdfs(get_config(ctx), limit)


@app.command()
def manifest(ctx: typer.Context, limit: LimitOption = None) -> None:
    """Write manifest.csv, manifest.json and stats.json to the manifest folder."""
    pipeline.manifest(get_config(ctx), limit)


@app.command("all")
def all_(ctx: typer.Context, limit: LimitOption = None) -> None:
    """Run every step in order. Cached and resumable, so safe to rerun after a crash."""
    pipeline.run_all(get_config(ctx), limit)


@app.command("audit-sample")
def audit_sample(ctx: typer.Context) -> None:
    """Write accepted + rejected records for hand labelling (sizes and seed in config)."""
    config = get_config(ctx)
    audit = config.collect.audit
    path = pipeline.audit_sample(config, audit.sample_per_decision, audit.seed)
    typer.echo(f"Wrote {path}")
