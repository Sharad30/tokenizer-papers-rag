"""Logging setup shared by all CLI commands."""

import logging
import sys

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: str) -> None:
    # Logs go to stderr so stdout stays clean for command output (e.g.
    # `rag config show > snapshot.yaml`). force=True lets tests and repeated
    # CLI invocations in one process reconfigure the level.
    logging.basicConfig(level=level, format=LOG_FORMAT, stream=sys.stderr, force=True)
