"""What the Versions and duplicates page shows, computed from data/works (D-038).

Live when `rag dedup run` has written data/works/. Otherwise from the
snapshot, which keeps the overview and sampled examples. The ablation table
comes from the latest committed results/<date>-dedup-ablation.json.
"""

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from rag.collect.manifest import MANIFEST_JSON, read_manifest
from rag.collect.schemas import ManifestRecord
from rag.common.config import Config
from rag.common.io import read_jsonl
from rag.common.schemas import StrictModel
from rag.dedup.files import RESTATEMENTS, SURVEYS, WORKS
from rag.dedup.schemas import Edge, Restatement, Survey, Work
from rag.eval.identity import identity_groups, load_pairs
from rag.parse.pipeline import CHUNKS, step_dir
from rag.parse.schemas import Chunk
from rag.showcase.config import DedupShowcaseConfig
from rag.showcase.parse import sample
from rag.showcase.sources import Snapshot, read_snapshot, snapshot_file, stale_sources

STAGE = "dedup"


class Member(StrictModel):
    record_id: str
    title: str
    year: int | None
    venue: str | None


class WorkExample(StrictModel):
    work_id: str
    canonical: str
    canonical_rule: str
    members: list[Member]
    edges: list[Edge]
    rejected: list[Edge]


class RestatementExample(StrictModel):
    chunk_id: str
    survey_title: str
    text: str
    work_id: str
    source_title: str
    cosine: float
    matched_by: str


class Overview(StrictModel):
    records: int
    id_groups: int  # what identifiers alone give (the dup@5 judge without labels)
    works: int
    multi_member: int
    sizes: dict[int, int]  # members per work -> works
    edges: dict[str, int]  # merges per step (ids counts links, not records)
    refused: int
    surveys: list[str]  # survey titles
    restatements: int
    labelled_pairs: int


class AblationRow(StrictModel):
    label: str
    dup: float
    dup_low: float
    dup_high: float
    copy_dup: float
    copy_dup_resolver: float


class Ablation(StrictModel):
    file: str
    git_commit: str
    n_queries: int
    labelled_pairs: int
    rows: list[AblationRow]


class DedupData(StrictModel):
    origin: Literal["live", "snapshot", "none"]
    snapshot: Snapshot | None = None
    stale: list[str] = []
    overview: Overview | None = None
    fuzzy_works: list[WorkExample] = []  # built or touched by title/text evidence
    id_works: list[WorkExample] = []  # built by identifiers only, several members
    restatements: list[RestatementExample] = []
    ablation: Ablation | None = None


def source_files(config: Config) -> dict[str, Path]:
    return {
        "works": config.paths.works / WORKS,
        "surveys": config.paths.works / SURVEYS,
        "restatements": config.paths.works / RESTATEMENTS,
        "manifest": config.paths.manifest / MANIFEST_JSON,
        "chunks": step_dir(config, "chunk") / CHUNKS,
    }


def load(config: Config) -> DedupData:
    files = source_files(config)
    ablation = latest_ablation(config.paths.results)
    if all(path.exists() for path in files.values()):
        return _live(config, files, ablation)
    snapshot = read_snapshot(snapshot_file(config.paths.snapshot, STAGE))
    if snapshot is None:
        return DedupData(origin="none", ablation=ablation)
    return DedupData.model_validate(
        {
            **snapshot.data,
            "origin": "snapshot",
            "snapshot": snapshot,
            "stale": stale_sources(snapshot, files),
            "ablation": ablation,
        }
    )


def _live(config: Config, files: dict[str, Path], ablation: Ablation | None) -> DedupData:
    works = read_jsonl(files["works"], Work)
    records = {r.record_id: r for r in read_manifest(config.paths.manifest)}
    members = {m for w in works for m in w.members}
    indexed = [r for r in records.values() if r.record_id in members]
    surveys = read_jsonl(files["surveys"], Survey)
    restatements = read_jsonl(files["restatements"], Restatement)
    labelled = load_pairs(config.paths.eval / config.eval.pairs_file)
    edges = Counter(e.step for w in works for e in w.edges)
    view = Overview(
        records=len(members),
        id_groups=len(set(identity_groups(indexed).values())),
        works=len(works),
        multi_member=sum(len(w.members) > 1 for w in works),
        sizes=dict(sorted(Counter(len(w.members) for w in works).items())),
        edges=dict(edges),
        refused=sum(len(w.rejected) for w in works),
        surveys=[records[s.record_id].title for s in surveys if s.record_id in records],
        restatements=len(restatements),
        labelled_pairs=len(labelled),
    )
    fuzzy = [w for w in works if w.rejected or any(e.step != "ids" for e in w.edges)]
    fuzzy_ids = {w.work_id for w in fuzzy}
    by_ids = [w for w in works if len(w.members) > 1 and w.work_id not in fuzzy_ids]
    return DedupData(
        origin="live",
        overview=view,
        fuzzy_works=[example(w, records) for w in fuzzy],
        id_works=[example(w, records) for w in by_ids],
        restatements=_restatement_examples(restatements, files["chunks"], records, works),
        ablation=ablation,
    )


def example(work: Work, records: dict[str, ManifestRecord]) -> WorkExample:
    return WorkExample(
        work_id=work.work_id,
        canonical=work.canonical,
        canonical_rule=work.canonical_rule,
        members=[
            Member(
                record_id=rid,
                title=records[rid].title if rid in records else "?",
                year=records[rid].year if rid in records else None,
                venue=records[rid].venue if rid in records else None,
            )
            for rid in work.members
        ],
        edges=work.edges,
        rejected=work.rejected,
    )


def _restatement_examples(
    restatements: Sequence[Restatement],
    chunks_path: Path,
    records: dict[str, ManifestRecord],
    works: Sequence[Work],
) -> list[RestatementExample]:
    wanted = {r.chunk_id for r in restatements}
    texts = {c.chunk_id: c.text for c in read_jsonl(chunks_path, Chunk) if c.chunk_id in wanted}
    canonical = {w.work_id: w.canonical for w in works}
    out = []
    for r in restatements:
        survey = r.chunk_id.split("#")[0]
        source = canonical.get(r.work_id, "")
        out.append(
            RestatementExample(
                chunk_id=r.chunk_id,
                survey_title=records[survey].title if survey in records else survey,
                text=texts.get(r.chunk_id, ""),
                work_id=r.work_id,
                source_title=records[source].title if source in records else source,
                cosine=r.cosine,
                matched_by=r.matched_by,
            )
        )
    return out


def latest_ablation(results_dir: Path) -> Ablation | None:
    """The newest committed dedup ablation, by file name (dates sort)."""
    files = sorted(results_dir.glob("*-dedup-ablation.json"))
    if not files:
        return None
    raw = json.loads(files[-1].read_text(encoding="utf-8"))
    rows = []
    for c in raw["conditions"].values():
        probe = c["probe"]
        dup = probe["dup@5"]
        rows.append(
            AblationRow(
                label=c["label"],
                dup=dup["mean"],
                dup_low=dup["ci_low"],
                dup_high=dup["ci_high"],
                copy_dup=probe["copy_dup@5"]["mean"],
                copy_dup_resolver=probe["copy_dup@5_resolver"]["mean"],
            )
        )
    return Ablation(
        file=files[-1].name,
        git_commit=raw["git_commit"],
        n_queries=raw["n_queries"],
        labelled_pairs=raw["labelled_pairs"],
        rows=rows,
    )


def work_key(work: WorkExample) -> str:
    return work.work_id


def restatement_key(r: RestatementExample) -> str:
    return f"{r.chunk_id}>{r.work_id}"


def snapshot_data(data: DedupData, show: DedupShowcaseConfig, seed: int) -> dict:
    reduced = DedupData(
        origin="snapshot",
        overview=data.overview,
        fuzzy_works=data.fuzzy_works,  # all of them: the merges to review (~120)
        id_works=sample(data.id_works, show.pool_size, seed, work_key),
        restatements=sample(data.restatements, show.pool_size, seed, restatement_key),
    )
    return reduced.model_dump(mode="json", exclude={"origin", "snapshot", "stale", "ablation"})
