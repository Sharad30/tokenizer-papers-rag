import typer

from rag.common.cli import get_config, print_stub


def chat(ctx: typer.Context) -> None:
    """Interactive grounded chat over the corpus, with citations."""
    config = get_config(ctx)
    print_stub("chat", "7", reads=str(config.paths.index), writes="terminal", limit=None)
