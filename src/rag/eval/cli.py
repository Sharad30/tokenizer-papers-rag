import typer

from rag.common.cli import LimitOption, get_config, print_stub


# Named run_eval rather than eval to avoid shadowing the builtin; the CLI name is `eval`.
def run_eval(ctx: typer.Context, limit: LimitOption = None) -> None:
    """Score retrieval (and later answers) on the eval set; write a results file."""
    config = get_config(ctx)
    print_stub(
        "eval",
        "4",
        reads=f"{config.paths.index}, {config.paths.eval}",
        writes=str(config.paths.results),
        limit=limit,
    )
