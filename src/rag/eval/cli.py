"""`rag eval ...`: build and check the eval set (Phase 3), score runs (Phase 4)."""

from pathlib import Path
from typing import Annotated

import typer

from rag.collect.manifest import read_manifest
from rag.common.cli import LimitOption, get_config, print_stub
from rag.common.config import Config
from rag.eval import find as find_mod
from rag.eval import validate as validate_mod
from rag.eval.spans import ChunkTexts
from rag.parse.pipeline import CHUNKS, step_dir

app = typer.Typer(
    help="Eval set tooling (find spans, validate questions) and eval runs.",
    no_args_is_help=True,
)


def _chunk_texts(config: Config) -> ChunkTexts:
    path = step_dir(config, "chunk") / CHUNKS
    if not path.is_file():
        typer.echo(f"No chunks at {path}: run `rag parse all` first.", err=True)
        raise typer.Exit(1)
    return ChunkTexts.from_file(path)


@app.command()
def find(
    ctx: typer.Context,
    text: Annotated[str, typer.Argument(help="Text to look for (case and quotes ignored)")],
    paper: Annotated[str | None, typer.Option(help="Only this record id")] = None,
    all_words: Annotated[
        bool, typer.Option("--all-words", help="Match chunks containing every word, any order")
    ] = False,
) -> None:
    """Grep the chunks for text to copy a gold span from. Unranked, in corpus order."""
    config = get_config(ctx)
    result = find_mod.find(
        _chunk_texts(config),
        text,
        max_hits=config.eval.find.max_hits,
        paper=paper,
        all_words=all_words,
    )
    typer.echo(find_mod.render(result, text))


@app.command()
def validate(
    ctx: typer.Context,
    file: Annotated[
        Path | None, typer.Option(help="Questions file (default: eval.questions_file)")
    ] = None,
) -> None:
    """Check the eval set: schema, papers indexed, spans map to chunks, conversations, mix."""
    config = get_config(ctx)
    path = file or config.paths.eval / config.eval.questions_file
    indexed = {r.record_id: bool(r.included_in_index) for r in read_manifest(config.paths.manifest)}
    report = validate_mod.validate_file(path, _chunk_texts(config), indexed, config.eval)
    typer.echo(validate_mod.render(report, path))
    if not report.ok:
        raise typer.Exit(1)


@app.command()
def run(ctx: typer.Context, limit: LimitOption = None) -> None:
    """Score retrieval (and later answers) on the eval set; write a results file."""
    config = get_config(ctx)
    print_stub(
        "eval run",
        "4",
        reads=f"{config.paths.index}, {config.paths.eval}",
        writes=str(config.paths.results),
        limit=limit,
    )
