"""`rag showcase snapshot`: save what the UI needs that a fresh clone won't have (D-038)."""

from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

from rag.common.cli import get_config
from rag.common.config import Config
from rag.common.provenance import git_commit
from rag.showcase import collect
from rag.showcase.sources import snapshot_file, write_snapshot

app = typer.Typer(
    help="Showcase UI data: `make ui` runs the pages themselves.", no_args_is_help=True
)


def _collect(config: Config) -> tuple[dict[str, Path], dict]:
    data = collect.load(config)
    if data.abstracts_from != "live":
        # Building a snapshot from a snapshot would only copy it.
        raise typer.BadParameter(
            f"needs the filter step's output ({collect.source_files(config)['filtered']}); "
            "run `rag collect filter` first"
        )
    pool = config.showcase.collect.pool_per_path
    return collect.source_files(config), collect.snapshot_data(data, pool, config.showcase.seed)


# One builder per stage page; later pages add theirs here.
BUILDERS: dict[str, Callable[[Config], tuple[dict[str, Path], dict]]] = {
    collect.STAGE: _collect,
}


@app.command()
def snapshot(
    ctx: typer.Context,
    stage: Annotated[
        list[str] | None,
        typer.Option(help=f"Stage(s) to snapshot: {', '.join(BUILDERS)}. Default: all."),
    ] = None,
) -> None:
    """Write data/snapshot/<stage>.json from the current stage outputs."""
    config = get_config(ctx)
    unknown = sorted(set(stage or []) - set(BUILDERS))
    if unknown:
        raise typer.BadParameter(f"unknown stage(s) {unknown}; known: {list(BUILDERS)}")
    commit = git_commit()
    for name in stage or BUILDERS:
        sources, data = BUILDERS[name](config)
        path = snapshot_file(config.paths.snapshot, name)
        written = write_snapshot(path, name, sources, data, commit)
        typer.echo(f"{name}: {path} (sources: {', '.join(written.sources)}; commit {commit[:12]})")
