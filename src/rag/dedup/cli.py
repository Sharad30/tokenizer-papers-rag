"""`rag dedup <step>`: resolve paper records into works."""

import json

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


@app.command()
def pairs(ctx: typer.Context) -> None:
    """Works -> record pairs across similarity bands, unlabelled -> eval/pairs_candidates.jsonl."""
    path = pipeline.write_pair_candidates(get_config(ctx))
    typer.echo(f"Candidates: {path}. Label them into eval/pairs.jsonl (see eval/README.md).")


@app.command()
def score(ctx: typer.Context) -> None:
    """Resolver precision and recall on eval/pairs.jsonl -> results/<date>-dedup-resolver.json."""
    path = pipeline.score_resolver(get_config(ctx))
    if path is None:
        typer.echo("No labelled pairs yet (eval/pairs.jsonl). Run `rag dedup pairs` first.")
        return
    result = json.loads(path.read_text(encoding="utf-8"))
    for name in ("precision", "recall"):
        value = result[name]
        typer.echo(f"{name}: {'n/a' if value is None else f'{value:.3f}'}")
    typer.echo(f"{len(result['errors'])} disagreements; details in {path}")
