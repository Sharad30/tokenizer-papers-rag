"""`rag eval ablate`: the same queries under several retrieval configurations.

Phase 5's table: how much each piece of duplicate handling changes dup@5
(and Recall@5 / MRR, once eval/questions.jsonl exists). Conditions live in
config.yaml (`eval.ablations.<name>.conditions`); each names the resolver
steps that build the works and the query-time settings. The indexes are
loaded once and reused.

Two dup@5 columns, on purpose:
- judge     duplicates by identifiers + Sharad's labelled pairs
            (`rag.eval.identity`, D-042): the metric.
- resolver  duplicates by the full resolver's own works. NOT ground truth:
            it shows what the resolver believes, and why it can't grade
            itself (it would score its own false merges as wins).
"""

import logging
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from rag.collect.manifest import read_manifest
from rag.common.config import Config
from rag.common.io import file_sha256, write_json, write_text_atomic
from rag.common.provenance import config_hash, git_commit
from rag.common.schemas import StrictModel
from rag.dedup import pipeline as dedup_pipeline
from rag.dedup.files import META as WORKS_META
from rag.dedup.resolve import resolve
from rag.dedup.schemas import WorksMeta
from rag.eval.config import Condition
from rag.eval.harness import run_mode
from rag.eval.identity import identity_groups, load_pairs
from rag.eval.metrics import (
    bootstrap_ci,
    copy_duplicate_rate_at_k,
    duplicate_rate_at_k,
    paired_bootstrap_diff,
)
from rag.eval.results import MetricSummary, ModeResult
from rag.eval.spans import ChunkTexts
from rag.eval.validate import load_questions
from rag.parse.pipeline import CHUNKS, step_dir
from rag.retrieve.collapse import WorkMap
from rag.retrieve.retriever import Retriever, bm25_dir, dense_dir, read_meta
from rag.retrieve.schemas import IndexMeta

log = logging.getLogger(__name__)


class QueryRow(StrictModel):
    query: str
    top_chunk_ids: list[str]
    metrics: dict[str, float]


class ConditionResult(StrictModel):
    label: str
    steps: list[str]
    dedup: dict[str, Any]
    probe: dict[str, MetricSummary]  # over the probe queries
    diff_vs_first: dict[str, MetricSummary]  # paired, this condition minus the first
    queries: list[QueryRow]
    eval: ModeResult | None = None  # on eval/questions.jsonl, when it exists


class AblationResult(StrictModel):
    run: str
    created_at: datetime
    git_commit: str
    config_hash: str
    config: dict[str, Any]
    mode: str
    queries_file: str
    queries_sha256: str
    n_queries: int
    labelled_pairs: int  # pairs.jsonl lines the judge used
    questions_sha256: str | None
    works_meta: WorksMeta
    index_meta: dict[str, IndexMeta]
    conditions: dict[str, ConditionResult]


def read_queries(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def results_path(config: Config, name: str) -> Path:
    return config.paths.results / f"{date.today().isoformat()}-{name}-ablation.json"


def run_ablation(config: Config, name: str, retriever: Retriever | None = None) -> Path:
    ablation = config.eval.ablations[name]
    mode = ablation.mode or config.retrieve.default_mode
    queries_path = config.paths.probes / ablation.queries_file
    queries = read_queries(queries_path)
    base = retriever or Retriever.load(config, {mode})
    work_maps = _work_maps(config, ablation.conditions)

    labelled = load_pairs(config.paths.eval / config.eval.pairs_file)
    records = read_manifest(config.paths.manifest)
    judge = identity_groups(records, labelled)
    full = work_maps[tuple(config.dedup.steps)].record_work
    questions = _questions(config)
    texts = ChunkTexts.from_file(step_dir(config, "chunk") / CHUNKS) if questions else None
    k = config.eval.dup_k
    depth = max(config.eval.recall_k, config.eval.mrr_cutoff, k)

    results: dict[str, ConditionResult] = {}
    first: list[QueryRow] | None = None
    for condition in ablation.conditions:
        works = work_maps[tuple(condition.steps)] if condition.dedup.enabled else None
        variant = base.with_dedup(condition.dedup, works)
        rows = []
        for query in queries:
            hits = variant.retrieve(query, k=depth, mode=mode)
            records_top = [h.chunk.record_id for h in hits]
            judged = [judge.get(r, r) for r in records_top]
            resolved = [full.get(r, r) for r in records_top]
            rows.append(
                QueryRow(
                    query=query,
                    top_chunk_ids=[h.chunk.chunk_id for h in hits],
                    metrics={
                        f"dup@{k}": duplicate_rate_at_k(judged, k),
                        f"any_dup@{k}": float(duplicate_rate_at_k(judged, k) > 0),
                        f"copy_dup@{k}": copy_duplicate_rate_at_k(records_top, judge, k),
                        f"copy_dup@{k}_resolver": copy_duplicate_rate_at_k(records_top, full, k),
                        f"dup@{k}_resolver": duplicate_rate_at_k(resolved, k),
                        f"papers@{k}_resolver": float(len(set(resolved[:k]))),
                    },
                )
            )
        first = first or rows
        eval_result = None
        if questions:
            eval_result = run_mode(variant, mode, questions, texts, judge, config.eval)
        results[condition.name] = ConditionResult(
            label=condition.label,
            steps=list(condition.steps),
            dedup=condition.dedup.model_dump(mode="json"),
            probe=_summaries(rows, config),
            diff_vs_first=_diffs(rows, first, config),
            queries=rows,
            eval=eval_result,
        )
        log.info("%s: %s", condition.name, _short(results[condition.name], k))

    questions_path = config.paths.eval / config.eval.questions_file
    result = AblationResult(
        run=name,
        created_at=datetime.now(UTC),
        git_commit=git_commit(),
        config_hash=config_hash(config),
        config=config.model_dump(mode="json", exclude={"paths"}),
        mode=mode,
        queries_file=str(queries_path),
        queries_sha256=file_sha256(queries_path),
        n_queries=len(queries),
        labelled_pairs=len(labelled),
        questions_sha256=file_sha256(questions_path) if questions else None,
        works_meta=WorksMeta.model_validate_json(
            (config.paths.works / WORKS_META).read_text(encoding="utf-8")
        ),
        index_meta={
            kind: read_meta(folder(config))
            for kind, folder, index in (
                ("bm25", bm25_dir, base.bm25),
                ("dense", dense_dir, base.dense),
            )
            if index is not None
        },
        conditions=results,
    )
    out = results_path(config, name)
    write_json(out, result)
    write_text_atomic(out.with_suffix(".md"), table(result, config))
    return out


def _work_maps(config: Config, conditions: Sequence[Condition]) -> dict[tuple, WorkMap]:
    """One WorkMap per distinct set of resolver steps. The configured steps come
    from data/works (with restatements); others are resolved here, in seconds."""
    chunks_path = step_dir(config, "chunk") / CHUNKS
    maps = {tuple(config.dedup.steps): WorkMap.load(config.paths.works, chunks_path)}
    other = {tuple(c.steps) for c in conditions if c.dedup.enabled} - set(maps)
    if other:
        chunks = dedup_pipeline.read_chunks(config)
        present = {c.record_id for c in chunks}
        records = [r for r in dedup_pipeline.indexed_records(config) if r.record_id in present]
        sigs = dedup_pipeline.signatures(chunks, config.dedup)
        n_chunks = Counter(c.record_id for c in chunks)
        for steps in sorted(other):
            resolution = resolve(records, sigs, n_chunks, config.dedup, list(steps))
            maps[steps] = WorkMap.from_works(resolution.works)
    for c in conditions:
        if (
            c.dedup.enabled
            and c.dedup.survey_demotion
            and tuple(c.steps) != tuple(config.dedup.steps)
        ):
            raise ValueError(f"{c.name}: survey demotion needs the configured resolver steps")
    return maps


def _questions(config: Config):
    path = config.paths.eval / config.eval.questions_file
    if not path.is_file():
        log.info("No %s yet: probe queries only (dup@k), no Recall/MRR", path.name)
        return []
    questions, errors = load_questions(path)
    if errors:
        raise ValueError(f"{path} has invalid lines; run `rag eval validate`: {errors[0]}")
    return questions


def _summaries(rows: Sequence[QueryRow], config: Config) -> dict[str, MetricSummary]:
    out = {}
    for metric in rows[0].metrics if rows else []:
        values = [r.metrics[metric] for r in rows]
        mean, low, high = bootstrap_ci(
            values, config.eval.bootstrap.samples, config.eval.bootstrap.seed
        )
        out[metric] = MetricSummary(mean=mean, ci_low=low, ci_high=high, n=len(values))
    return out


def _diffs(rows, first, config: Config) -> dict[str, MetricSummary]:
    out = {}
    for metric in rows[0].metrics if rows else []:
        mean, low, high = paired_bootstrap_diff(
            [r.metrics[metric] for r in rows],
            [r.metrics[metric] for r in first],
            config.eval.bootstrap.samples,
            config.eval.bootstrap.seed,
        )
        out[metric] = MetricSummary(mean=mean, ci_low=low, ci_high=high, n=len(rows))
    return out


def _short(result: ConditionResult, k: int) -> str:
    return ", ".join(f"{m} {s.mean:.3f}" for m, s in result.probe.items() if m.startswith("dup"))


def _cell(s: MetricSummary | None, digits: int = 3) -> str:
    if s is None:
        return "–"
    return f"{s.mean:.{digits}f} [{s.ci_low:.{digits}f}, {s.ci_high:.{digits}f}]"


def table(result: AblationResult, config: Config) -> str:
    k = config.eval.dup_k
    rk, mrr = f"recall@{config.eval.recall_k}", f"mrr@{config.eval.mrr_cutoff}"
    lines = [
        f"# Ablation `{result.run}` ({result.mode}, {result.n_queries} probe queries)",
        "",
        f"Commit {result.git_commit[:12]}, {result.created_at:%Y-%m-%d}. Probe queries:",
        f"`{Path(result.queries_file).name}` (not gold). Judge: identifiers + "
        f"{result.labelled_pairs} labelled pairs. Means with 95% bootstrap intervals.",
        "",
        f"| Condition | dup@{k} | Δ dup@{k} vs first | copy_dup@{k} | "
        f"copy_dup@{k} (resolver)* | papers in top {k} (resolver)* | {rk} | {mrr} |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in result.conditions.values():
        overall = c.eval.aggregates.get("overall", {}) if c.eval else {}
        lines.append(
            f"| {c.label} | {_cell(c.probe.get(f'dup@{k}'))} | "
            f"{_cell(c.diff_vs_first.get(f'dup@{k}'))} | "
            f"{_cell(c.probe.get(f'copy_dup@{k}'))} | "
            f"{_cell(c.probe.get(f'copy_dup@{k}_resolver'))} | "
            f"{_cell(c.probe.get(f'papers@{k}_resolver'), 2)} | "
            f"{_cell(overall.get(rk))} | {_cell(overall.get(mrr))} |"
        )
    lines += [
        "",
        f"- dup@{k} (D-031): share of the top {k} slots whose paper already appeared",
        "  higher. Two chunks of one paper count, so a per-work cap of 1 makes it 0 by",
        "  construction; that row's cost can only show in recall and MRR.",
        f"- copy_dup@{k}: the same, counting only another record of the paper (a",
        "  version or venue copy), not a second chunk of the same record. This is",
        "  what merging removes.",
        "- Judged by identifiers + labelled pairs (D-042): copies the resolver found by",
        "  title or text count as different papers until labelled, so these are lower",
        "  bounds and can even rise when a hidden copy's slot goes to another chunk.",
        "- \\* Judged by the full resolver's own works: what the resolver believes, not",
        "  ground truth. Shown to make the gap visible.",
        f"- {rk} and {mrr} need eval/questions.jsonl ('–' until it exists).",
        "",
        "Generated by `rag eval ablate`. Per-query results and any_dup in the JSON.",
    ]
    return "\n".join(lines) + "\n"
