"""`rag eval run`: load everything, score each mode, write results/<date>-<name>.json."""

import logging
from datetime import UTC, date, datetime
from pathlib import Path

from rag.collect.manifest import read_manifest
from rag.common.config import Config
from rag.common.io import file_sha256, write_json
from rag.common.provenance import config_hash, git_commit
from rag.eval.harness import Retrieves, run_mode
from rag.eval.identity import identity_groups
from rag.eval.results import RunResult
from rag.eval.spans import ChunkTexts
from rag.eval.validate import load_questions
from rag.parse.pipeline import CHUNKS, step_dir
from rag.retrieve.retriever import NEEDS, Retriever, bm25_dir, dense_dir, read_meta

log = logging.getLogger(__name__)


def results_path(config: Config, name: str, today: date | None = None) -> Path:
    return config.paths.results / f"{(today or date.today()).isoformat()}-{name}.json"


def run_eval(
    config: Config,
    name: str,
    modes: list[str],
    questions_file: Path | None = None,
    limit: int | None = None,
    retriever: Retrieves | None = None,
) -> Path | None:
    """Score `modes` on the questions; returns the results path, or None if there are none."""
    path = questions_file or config.paths.eval / config.eval.questions_file
    if not path.is_file():
        log.warning("No questions yet: %s doesn't exist. Nothing to score.", path)
        return None
    questions, errors = load_questions(path)
    if errors:
        raise ValueError(f"{path} has invalid lines; run `rag eval validate`: {errors[0]}")
    questions = questions[:limit] if limit else questions
    if not questions:
        log.warning("No questions in %s. Nothing to score.", path)
        return None

    metas = {}
    if retriever is None:  # a passed-in retriever (tests) has no index folders to record
        retriever = Retriever.load(config, set(modes))
        needed = set().union(*(NEEDS[m] for m in modes))
        metas = {
            kind: read_meta(folder(config))
            for kind, folder in (("bm25", bm25_dir), ("dense", dense_dir))
            if kind in needed
        }
    texts = ChunkTexts.from_file(step_dir(config, "chunk") / CHUNKS)
    groups = identity_groups(read_manifest(config.paths.manifest))
    result = RunResult(
        run=name,
        created_at=datetime.now(UTC),
        git_commit=git_commit(),
        config_hash=config_hash(config),
        config=config.model_dump(mode="json", exclude={"paths"}),
        questions_file=str(path),
        questions_sha256=file_sha256(path),
        n_questions=len(questions),
        index_meta=metas,
        modes={m: run_mode(retriever, m, questions, texts, groups, config.eval) for m in modes},
    )
    out = results_path(config, name)
    write_json(out, result)
    log.info("Results for %d questions, modes %s -> %s", len(questions), modes, out)
    return out
