"""`rag eval ablate`: the same queries under several retrieval configurations.

One runner for Phase 5's duplicate table and Phase 6's model comparison.
Conditions live in config.yaml (`eval.ablations.<name>.conditions`). Each
can set the resolver steps and query-time duplicate handling (Phase 5),
the dense model, the mode, the reranker and the query prompt (Phase 6);
anything unset is the deployed configuration. Indexes are loaded once per
dense model and reused.

What is measured per condition:
- on the probe queries: dup@k and copy_dup@k, judged by identifiers +
  Sharad's labelled pairs (D-042); also the resolver's own view, labelled
  as not ground truth (it would score its own false merges as wins);
- on each question set that exists (gold `eval/gold/questions.jsonl`, silver
  `eval/silver/questions.jsonl`, D-044): Recall@k and MRR through the
  Phase 4 harness, reported per set, never pooled;
- query latency p50/p95 over every query run, and the index's cost (embed
  time, dimensions, size on disk);
- for the two best conditions by Recall@k: paired bootstrap differences
  and per-question wins and losses.
"""

import json
import logging
import time
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from rag.collect.manifest import read_manifest
from rag.common.config import Config
from rag.common.io import file_sha256, write_json, write_text_atomic
from rag.common.provenance import config_hash, git_commit
from rag.common.schemas import StrictModel
from rag.dedup import pipeline as dedup_pipeline
from rag.dedup.files import META as WORKS_META
from rag.dedup.resolve import resolve
from rag.dedup.schemas import WorksMeta
from rag.eval.config import AblationConfig, Condition
from rag.eval.harness import run_mode, variants
from rag.eval.identity import identity_groups, load_pairs
from rag.eval.metrics import (
    bootstrap_ci,
    copy_duplicate_rate_at_k,
    duplicate_rate_at_k,
    paired_bootstrap_diff,
)
from rag.eval.results import MetricSummary, ModeResult
from rag.eval.schemas import Question
from rag.eval.spans import ChunkTexts
from rag.eval.validate import load_questions
from rag.parse.pipeline import CHUNKS, step_dir
from rag.retrieve.collapse import WorkMap
from rag.retrieve.dense import make_encoder
from rag.retrieve.rerank import CrossEncoderScorer, Scorer
from rag.retrieve.retriever import Retriever, bm25_dir, dense_dir
from rag.retrieve.schemas import Hit

log = logging.getLogger(__name__)


class QueryRow(StrictModel):
    query: str
    top_chunk_ids: list[str]
    metrics: dict[str, float]


class Cost(StrictModel):
    index: str  # "bm25" or the dense model's name
    seconds: int | None  # build time, from the index's stats.json
    dims: int | None
    size_mb: float
    truncated: int | None = None  # chunks longer than the model's input limit


class SetResult(StrictModel):
    file: str
    sha256: str
    n_questions: int
    result: ModeResult


class ConditionResult(StrictModel):
    label: str
    steps: list[str]
    dedup: dict[str, Any]
    dense_model: str | None
    mode: str
    rerank: bool
    query_prompt: str | None
    probe: dict[str, MetricSummary]  # over the probe queries
    diff_vs_first: dict[str, MetricSummary]  # paired, this condition minus the first
    queries: list[QueryRow]
    sets: dict[str, SetResult] = {}  # label ("gold", "silver") -> harness results
    latency_ms: dict[str, float] = {}  # p50, p95, n
    cost: list[Cost] = []


class Comparison(StrictModel):
    """Two conditions on one question set, question by question."""

    set: str
    a: str
    b: str
    diffs: dict[str, MetricSummary]  # a minus b, paired bootstrap
    wins: int  # questions where a's Recall@k is higher
    losses: int
    ties: int


class AblationResult(StrictModel):
    run: str
    created_at: datetime
    git_commit: str
    config_hash: str
    config: dict[str, Any]
    queries_file: str
    queries_sha256: str
    n_queries: int
    labelled_pairs: int  # pairs.jsonl lines the judge used
    works_meta: WorksMeta
    conditions: dict[str, ConditionResult]
    comparison: Comparison | None = None  # the top two by Recall@k
    pairs: list[Comparison] = []  # the ablation's configured pairs


def read_queries(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def results_path(config: Config, name: str) -> Path:
    return config.paths.results / f"{date.today().isoformat()}-{name}-ablation.json"


class Timed:
    """A retriever that records how long each query takes (ms).

    `encode_ms` adds query-encoding time measured elsewhere: SPECTER2 encodes
    queries in its own process, ahead of time (`Specter2Encoder`).
    """

    def __init__(self, retriever: Retriever, encode_ms: dict[str, float] | None = None) -> None:
        self.retriever = retriever
        self.encode_ms = encode_ms or {}
        self.ms: list[float] = []

    def retrieve(self, query: str, k: int | None = None, mode=None) -> list[Hit]:
        tick = time.perf_counter()
        hits = self.retriever.retrieve(query, k=k, mode=mode)
        self.ms.append((time.perf_counter() - tick) * 1000 + self.encode_ms.get(query, 0.0))
        return hits


def percentiles(ms: Sequence[float]) -> dict[str, float]:
    if not ms:
        return {}
    p50, p95 = np.percentile(ms, [50, 95])
    return {"p50": round(float(p50), 1), "p95": round(float(p95), 1), "n": float(len(ms))}


def run_ablation(config: Config, name: str, retriever: Retriever | None = None) -> Path:
    ablation = config.eval.ablations[name]
    default_mode = ablation.mode or config.retrieve.default_mode
    queries_path = config.paths.probes / ablation.queries_file
    queries = read_queries(queries_path)
    work_maps = _work_maps(config, ablation.conditions)
    labelled = load_pairs(config.paths.eval / config.eval.pairs_file)
    judge = identity_groups(read_manifest(config.paths.manifest), labelled)
    full = work_maps[tuple(config.dedup.steps)].record_work
    sets = _question_sets(config, ablation)
    texts = ChunkTexts.from_file(step_dir(config, "chunk") / CHUNKS) if sets else None
    all_texts = list(queries) + [
        text for qs in sets.values() for q in qs[1] for _, text in variants(q) if text
    ]
    k = config.eval.dup_k
    depth = max(config.eval.recall_k, config.eval.mrr_cutoff, k)

    bases: dict[str, Retriever] = {}
    reranker: Scorer | None = None
    results: dict[str, ConditionResult] = {}
    first: list[QueryRow] | None = None
    for condition in ablation.conditions:
        mode = condition.mode or default_mode
        model = condition.dense_model or config.index.dense_model
        if retriever is not None:
            base = retriever
        elif model not in bases:
            bases[model] = Retriever.load(config, {"hybrid"}, dense=model)
            base = bases[model]
        else:
            base = bases[model]
        if condition.rerank and reranker is None:
            reranker = CrossEncoderScorer(config.retrieve.rerank)
        variant = _variant(config, condition, base, work_maps, reranker)
        encode_ms = _pre_encode(variant, all_texts, mode)
        variant.retrieve(queries[0], k=depth, mode=mode)  # warm-up, not timed
        timed = Timed(variant, encode_ms)

        rows = [_probe_row(timed, q, depth, mode, judge, full, k) for q in queries]
        first = first or rows
        scored = {
            label: SetResult(
                file=_shown(path, config.paths.eval),
                sha256=file_sha256(path),
                n_questions=len(qs),
                result=run_mode(timed, mode, qs, texts, judge, config.eval),
            )
            for label, (path, qs) in sets.items()
        }
        results[condition.name] = ConditionResult(
            label=condition.label,
            steps=list(condition.steps or config.dedup.steps),
            dedup=(condition.dedup or config.retrieve.dedup).model_dump(mode="json"),
            dense_model=model if mode != "bm25" else None,
            mode=mode,
            rerank=condition.rerank,
            query_prompt=condition.query_prompt,
            probe=_summaries(rows, config),
            diff_vs_first=_diffs(rows, first, config),
            queries=rows,
            sets=scored,
            latency_ms=percentiles(timed.ms),
            cost=_costs(config, mode, model),
        )
        log.info("%s: %s", condition.name, _short(results[condition.name], config))

    result = AblationResult(
        run=name,
        created_at=datetime.now(UTC),
        git_commit=git_commit(),
        config_hash=config_hash(config),
        config=config.model_dump(mode="json", exclude={"paths"}),
        queries_file=_shown(queries_path, config.paths.probes),
        queries_sha256=file_sha256(queries_path),
        n_queries=len(queries),
        labelled_pairs=len(labelled),
        works_meta=WorksMeta.model_validate_json(
            (config.paths.works / WORKS_META).read_text(encoding="utf-8")
        ),
        conditions=results,
        comparison=compare(results, config),
        pairs=[c for a, b in ablation.pairs if (c := paired(results, config, a, b))],
    )
    out = results_path(config, name)
    write_json(out, result)
    write_text_atomic(out.with_suffix(".md"), table(result, config))
    return out


def _variant(
    config: Config,
    condition: Condition,
    base: Retriever,
    work_maps: dict[tuple, WorkMap],
    reranker: Scorer | None,
) -> Retriever:
    dedup = condition.dedup or config.retrieve.dedup
    steps = tuple(condition.steps or config.dedup.steps)
    variant = base.with_dedup(dedup, work_maps[steps] if dedup.enabled else None)
    variant = variant.with_rerank(condition.rerank, reranker)
    if condition.query_prompt is not None:
        model = condition.dense_model or config.index.dense_model
        settings = config.index.dense_models[model].model_copy(
            update={"query_prompt": condition.query_prompt}
        )
        variant = variant.with_encoder(make_encoder(settings))
    return variant


def _pre_encode(variant: Retriever, texts: Sequence[str], mode: str) -> dict[str, float]:
    """SPECTER2 encodes queries in another process: all at once, timed per query."""
    encoder = variant.encoder
    if mode == "bm25" or encoder is None or not hasattr(encoder, "encode_queries"):
        return {}
    encoder.encode_queries(texts)
    return dict(encoder.encode_ms)


def _probe_row(timed: Timed, query: str, depth: int, mode: str, judge, full, k: int) -> QueryRow:
    hits = timed.retrieve(query, k=depth, mode=mode)
    records = [h.chunk.record_id for h in hits]
    judged = [judge.get(r, r) for r in records]
    resolved = [full.get(r, r) for r in records]
    return QueryRow(
        query=query,
        top_chunk_ids=[h.chunk.chunk_id for h in hits],
        metrics={
            f"dup@{k}": duplicate_rate_at_k(judged, k),
            f"any_dup@{k}": float(duplicate_rate_at_k(judged, k) > 0),
            f"copy_dup@{k}": copy_duplicate_rate_at_k(records, judge, k),
            f"copy_dup@{k}_resolver": copy_duplicate_rate_at_k(records, full, k),
            f"dup@{k}_resolver": duplicate_rate_at_k(resolved, k),
            f"papers@{k}_resolver": float(len(set(resolved[:k]))),
        },
    )


def _work_maps(config: Config, conditions: Sequence[Condition]) -> dict[tuple, WorkMap]:
    """One WorkMap per distinct set of resolver steps. The configured steps come
    from data/works (with restatements); others are resolved here, in seconds."""
    chunks_path = step_dir(config, "chunk") / CHUNKS
    deployed = tuple(config.dedup.steps)
    maps = {deployed: WorkMap.load(config.paths.works, chunks_path)}
    wanted = {
        tuple(c.steps or deployed) for c in conditions if (c.dedup or config.retrieve.dedup).enabled
    }
    if wanted - set(maps):
        chunks = dedup_pipeline.read_chunks(config)
        present = {c.record_id for c in chunks}
        records = [r for r in dedup_pipeline.indexed_records(config) if r.record_id in present]
        sigs = dedup_pipeline.signatures(chunks, config.dedup)
        n_chunks = Counter(c.record_id for c in chunks)
        for steps in sorted(wanted - set(maps)):
            resolution = resolve(records, sigs, n_chunks, config.dedup, list(steps))
            maps[steps] = WorkMap.from_works(resolution.works)
    for c in conditions:
        dedup = c.dedup or config.retrieve.dedup
        if dedup.enabled and dedup.survey_demotion and tuple(c.steps or deployed) != deployed:
            raise ValueError(f"{c.name}: survey demotion needs the configured resolver steps")
    return maps


def _question_sets(config: Config, ablation: AblationConfig) -> dict[str, tuple[Path, list]]:
    """label -> (path, questions), for the sets whose file exists."""
    files = ablation.question_files or {"gold": config.eval.questions_file}
    out: dict[str, tuple[Path, list[Question]]] = {}
    for label, name in files.items():
        path = config.paths.eval / name
        if not path.is_file():
            log.info("No %s set yet (%s): skipped", label, path)
            continue
        questions, errors = load_questions(path)
        if errors:
            raise ValueError(f"{path} has invalid lines; run `rag eval validate`: {errors[0]}")
        out[label] = (path, questions)
    return out


def _shown(path: Path, root: Path) -> str:
    """A path as config names it (relative to its folder's parent), not this machine's."""
    return str(path.relative_to(root.parent)) if path.is_relative_to(root.parent) else str(path)


def _costs(config: Config, mode: str, model: str) -> list[Cost]:
    folders = []
    if mode in ("bm25", "hybrid"):
        folders.append(("bm25", bm25_dir(config)))
    if mode in ("dense", "hybrid"):
        folders.append((model, dense_dir(config, model)))
    out = []
    for name, folder in folders:
        stats_path = folder / "stats.json"
        counts = json.loads(stats_path.read_text())["counts"] if stats_path.is_file() else {}
        size = sum(f.stat().st_size for f in folder.iterdir() if f.is_file())
        out.append(
            Cost(
                index=name,
                seconds=counts.get("seconds"),
                dims=counts.get("dimensions"),
                size_mb=round(size / 1e6, 1),
                truncated=counts.get("truncated"),
            )
        )
    return out


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


def _per_question(result: ModeResult) -> dict[str, dict[str, float]]:
    """question id -> metrics, for the rows counted in "overall" (follow-ups as asked)."""
    return {q.question_id: q.metrics for q in result.questions if q.scored and q.variant == "raw"}


def _metric_names(config: Config) -> tuple[str, str, str]:
    """Recall@k (the gold passage), paper recall@k (the gold paper), MRR."""
    k = config.eval.recall_k
    return f"recall@{k}", f"paper_recall@{k}", f"mrr@{config.eval.mrr_cutoff}"


def _set_label(results: dict[str, ConditionResult]) -> str | None:
    """Gold if it exists, else the first set by name."""
    labels = {label for c in results.values() for label in c.sets}
    if not labels:
        return None
    return "gold" if "gold" in labels else sorted(labels)[0]


def compare(results: dict[str, ConditionResult], config: Config) -> Comparison | None:
    """The two best conditions by Recall@k, on gold if it exists, else the first set."""
    label = _set_label(results)
    if label is None:
        return None
    rk = f"recall@{config.eval.recall_k}"

    def recall(name: str) -> float:
        aggregates = results[name].sets[label].result.aggregates.get("overall", {})
        return aggregates[rk].mean if rk in aggregates else -1.0

    ranked = sorted((n for n in results if label in results[n].sets), key=recall, reverse=True)
    if len(ranked) < 2:
        return None
    return paired(results, config, ranked[0], ranked[1])


def paired(
    results: dict[str, ConditionResult], config: Config, a: str, b: str
) -> Comparison | None:
    """a minus b on the shared questions, with a paired bootstrap interval."""
    label = _set_label(results)
    if label is None or label not in results[a].sets or label not in results[b].sets:
        return None
    rk, prk, mrr = _metric_names(config)
    qa = _per_question(results[a].sets[label].result)
    qb = _per_question(results[b].sets[label].result)
    shared = sorted(set(qa) & set(qb))
    diffs = {}
    for metric in (rk, prk, mrr):
        if not all(metric in qa[q] and metric in qb[q] for q in shared):
            continue  # results written before the metric existed
        mean, low, high = paired_bootstrap_diff(
            [qa[q][metric] for q in shared],
            [qb[q][metric] for q in shared],
            config.eval.bootstrap.samples,
            config.eval.bootstrap.seed,
        )
        diffs[metric] = MetricSummary(mean=mean, ci_low=low, ci_high=high, n=len(shared))
    delta = [qa[q][rk] - qb[q][rk] for q in shared]
    return Comparison(
        set=label,
        a=a,
        b=b,
        diffs=diffs,
        wins=sum(d > 0 for d in delta),
        losses=sum(d < 0 for d in delta),
        ties=sum(d == 0 for d in delta),
    )


def _short(result: ConditionResult, config: Config) -> str:
    k, rk = config.eval.dup_k, f"recall@{config.eval.recall_k}"
    parts = [f"dup@{k} {result.probe[f'dup@{k}'].mean:.3f}"] if result.probe else []
    for label, s in result.sets.items():
        overall = s.result.aggregates.get("overall", {})
        if rk in overall:
            parts.append(f"{label} {rk} {overall[rk].mean:.3f}")
    if result.latency_ms:
        parts.append(f"p50 {result.latency_ms['p50']} ms")
    return ", ".join(parts)


def _cell(s: MetricSummary | None, digits: int = 3) -> str:
    if s is None:
        return "–"
    return f"{s.mean:.{digits}f} [{s.ci_low:.{digits}f}, {s.ci_high:.{digits}f}]"


def _cost_cell(costs: Sequence[Cost]) -> tuple[str, str, str]:
    """(build time, dims, size) over the condition's indexes."""
    minutes = " + ".join(
        f"{c.index} {c.seconds / 60:.0f} min" for c in costs if c.seconds is not None
    )
    dims = ", ".join(str(c.dims) for c in costs if c.dims) or "–"
    size = f"{sum(c.size_mb for c in costs):.0f} MB"
    return minutes or "–", dims, size


def table(result: AblationResult, config: Config) -> str:
    k = config.eval.dup_k
    rk, prk, mrr = _metric_names(config)
    labels = sorted({label for c in result.conditions.values() for label in c.sets})
    any_set = next(iter(result.conditions.values())).sets if result.conditions else {}
    set_desc = ", ".join(f"{label} ({any_set[label].n_questions} questions)" for label in labels)
    header = ["Condition"]
    for label in labels:
        header += [f"{rk} ({label})", f"{prk} ({label})", f"{mrr} ({label})"]
    header += [f"dup@{k}", f"copy_dup@{k}", "latency p50 / p95", "index build", "dims", "size"]
    lines = [
        f"# Ablation `{result.run}`",
        "",
        f"Commit {result.git_commit[:12]}, {result.created_at:%Y-%m-%d}. Question sets: "
        f"{set_desc or 'none yet'}. dup@{k} on {result.n_queries} probe queries "
        f"(`{Path(result.queries_file).name}`, not gold). Judge: identifiers + "
        f"{result.labelled_pairs} labelled pairs. Means with 95% bootstrap intervals.",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for c in result.conditions.values():
        row = [c.label]
        for label in labels:
            overall = c.sets[label].result.aggregates.get("overall", {}) if label in c.sets else {}
            row += [_cell(overall.get(rk)), _cell(overall.get(prk)), _cell(overall.get(mrr))]
        build, dims, size = _cost_cell(c.cost)
        latency = (
            f"{c.latency_ms['p50']:.0f} / {c.latency_ms['p95']:.0f} ms" if c.latency_ms else "–"
        )
        row += [
            _cell(c.probe.get(f"dup@{k}")),
            _cell(c.probe.get(f"copy_dup@{k}")),
            latency,
            build,
            dims,
            size,
        ]
        lines.append("| " + " | ".join(row) + " |")
    if result.comparison:
        cmp = result.comparison
        a, b = result.conditions[cmp.a].label, result.conditions[cmp.b].label
        lines += [
            "",
            f"**Top two by {rk} ({cmp.set}):** {a} vs {b}. "
            f"Paired difference {rk} {_cell(cmp.diffs[rk])}, {mrr} {_cell(cmp.diffs[mrr])}; "
            f"{rk} higher on {cmp.wins} questions, lower on {cmp.losses}, equal on {cmp.ties}. "
            "An interval that includes 0 means the data can't tell them apart.",
        ]
    if result.pairs:
        lines += [
            "",
            f"**Paired comparisons ({result.pairs[0].set}, a minus b):**",
            "",
            f"| a | b | {rk} | {prk} | {mrr} | a higher / lower / equal |",
            "|---|---|---|---|---|---|",
        ]
        for cmp in result.pairs:
            a, b = result.conditions[cmp.a].label, result.conditions[cmp.b].label
            lines.append(
                f"| {a} | {b} | {_cell(cmp.diffs[rk])} | {_cell(cmp.diffs.get(prk))} "
                f"| {_cell(cmp.diffs[mrr])} "
                f"| {cmp.wins} / {cmp.losses} / {cmp.ties} |"
            )
    lines += [
        "",
        f"- {rk}: share of a question's gold passages found in the top {config.eval.recall_k}"
        " chunks;",
        f"  {prk}: share of its gold papers (any copy) with any chunk there. A gap between",
        "  them means the right paper came back with a different passage of it (the",
        "  per-work cap shows one chunk per paper).",
        f"- dup@{k} (D-031): share of the top {k} slots whose paper already appeared higher;",
        f"  copy_dup@{k} counts only another record of the paper (a version or venue copy).",
        "  Judged by identifiers + labelled pairs (D-042): lower bounds until pairs are",
        "  labelled. A per-work cap of 1 makes dup@k 0 by construction.",
        "- Latency: wall time per query on this machine's CPU (encode + search + rerank +",
        "  duplicate handling), after one warm-up query; SPECTER2's query encoding runs in",
        "  its own process and is timed there. Index build: CPU minutes, from stats.json.",
    ]
    if "silver" in labels:
        lines += [
            "- Silver (D-044): questions an LLM wrote from one chunk each. One gold chunk per",
            "  question, so other passages that answer it count as misses; single-paper",
            "  lookups only; wording drawn from the passage favours BM25. Provisional until",
            "  the gold set exists; never pooled with it.",
        ]
    if "gold" not in labels:
        lines.append("- Gold: eval/gold/questions.jsonl doesn't exist yet; rerun once it does.")
    lines += ["", "Generated by `rag eval ablate`. Per-query results in the JSON."]
    return "\n".join(lines) + "\n"
