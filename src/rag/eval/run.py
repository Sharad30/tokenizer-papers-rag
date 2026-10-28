"""`rag eval run`: load everything, score each mode, write results/<date>-<name>.json."""

import logging
from collections import defaultdict
from datetime import UTC, date, datetime
from pathlib import Path

from rag.chat.llm import LLM
from rag.chat.prompts import Prompt
from rag.chat.rewrite import Exchange, rewrite
from rag.collect.manifest import read_manifest
from rag.common.config import Config
from rag.common.io import file_sha256, write_json
from rag.common.provenance import config_hash, git_commit
from rag.eval.harness import Retrieves, run_mode
from rag.eval.identity import identity_groups, load_pairs
from rag.eval.results import RunResult
from rag.eval.schemas import Question, QuestionType
from rag.eval.spans import ChunkTexts
from rag.eval.validate import load_questions
from rag.parse.pipeline import CHUNKS, step_dir
from rag.retrieve.retriever import NEEDS, Retriever, bm25_dir, dense_dir, read_meta

log = logging.getLogger(__name__)


def results_path(config: Config, name: str, today: date | None = None) -> Path:
    return config.paths.results / f"{(today or date.today()).isoformat()}-{name}.json"


def system_rewrites(questions: list[Question], llm: LLM, prompt: Prompt, config: Config):
    """The chat's rewrite of every follow-up, given its conversation's earlier questions.

    The eval set has no gold answers, so the history holds the earlier
    questions with empty answers: the rewriter must resolve references from
    the questions alone, a slightly harder setting than live chat.
    """
    by_conversation: dict[str, list[Question]] = defaultdict(list)
    for q in questions:
        if q.conversation_id:
            by_conversation[q.conversation_id].append(q)
    out = {}
    for q in questions:
        if q.type != QuestionType.FOLLOWUP or not q.conversation_id:
            continue
        earlier = sorted(
            (p for p in by_conversation[q.conversation_id] if p.turn and p.turn < (q.turn or 0)),
            key=lambda p: p.turn or 0,
        )
        history = [Exchange(p.question, "") for p in earlier]
        out[q.id] = rewrite(llm, prompt, history, q.question, config.chat.rewrite).query
    return out


def run_eval(
    config: Config,
    name: str,
    modes: list[str],
    questions_file: Path | None = None,
    limit: int | None = None,
    retriever: Retrieves | None = None,
    rewriter: tuple[LLM, Prompt] | None = None,
) -> Path | None:
    """Score `modes`; with `rewriter`, also score follow-ups' system rewrites."""
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
    rewrites, rewriter_info = None, None
    if rewriter is not None:
        llm, prompt = rewriter
        rewrites = system_rewrites(questions, llm, prompt, config)
        rewriter_info = {"model": llm.model, "prompt_sha256": prompt.sha256}
    texts = ChunkTexts.from_file(step_dir(config, "chunk") / CHUNKS)
    pairs = load_pairs(config.paths.eval / config.eval.pairs_file)
    groups = identity_groups(read_manifest(config.paths.manifest), pairs)
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
        rewriter=rewriter_info,
        modes={
            m: run_mode(retriever, m, questions, texts, groups, config.eval, rewrites)
            for m in modes
        },
    )
    out = results_path(config, name)
    write_json(out, result)
    log.info("Results for %d questions, modes %s -> %s", len(questions), modes, out)
    return out
