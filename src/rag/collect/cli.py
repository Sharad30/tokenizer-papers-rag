import typer

from rag.common.cli import LimitOption, get_config, print_stub


def collect(ctx: typer.Context, limit: LimitOption = None) -> None:
    """Find, filter and download papers; write the corpus manifest."""
    config = get_config(ctx)
    # TODO(sharad): Phase 0 "You do": read the config field you added in
    # src/rag/common/config.py (config.<section>.<field>) and print it here.
    print_stub(
        "collect",
        "1",
        reads="arXiv / Semantic Scholar / ACL Anthology",
        writes=f"{config.paths.raw}, {config.paths.manifest}",
        limit=limit,
    )
