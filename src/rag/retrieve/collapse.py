"""Query-time handling of duplicates (D-041), between ranking and the top-k cut.

Versions and venue copies were resolved into works at index time (`rag dedup
run`). Here, per query:

1. canonical only   chunks of non-canonical members (an old arXiv version, the
                    arXiv copy of an ACL paper) are masked out before ranking.
                    Same dense results as an index without them; BM25's IDF
                    still counts them.
2. per-work cap     at most N chunks per work. Chunks over the cap are hidden,
                    and listed on the hit that kept the work's place.
3. MMR (optional)   Maximal Marginal Relevance: pick each next chunk by
                    lambda * relevance - (1 - lambda) * (max cosine to the
                    chunks already picked), trading relevance for novelty.
4. survey demotion  within the top k, a survey chunk that restates work W
                    moves just below W's chunk: cite the source, not the
                    summary. It is moved, never dropped.
"""

import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rag.common.io import file_sha256, read_jsonl
from rag.dedup.files import META, RECORD_WORK, RESTATEMENTS, WORKS
from rag.dedup.schemas import Restatement, Work, WorksMeta
from rag.parse.schemas import Chunk
from rag.retrieve.config import DedupSettings

Ranked = list[tuple[int, float]]  # (chunk position, score), best first


class WorksUnavailable(RuntimeError):
    """data/works is missing or was built from other chunks. Run `rag dedup run`."""


@dataclass
class WorkMap:
    record_work: dict[str, str]
    canonical: set[str]  # record ids that represent their work
    restates: dict[str, set[str]] = field(default_factory=dict)  # chunk_id -> work_ids

    @classmethod
    def from_works(cls, works: Sequence[Work], restatements: Sequence[Restatement] = ()):
        restates: dict[str, set[str]] = defaultdict(set)
        for r in restatements:
            restates[r.chunk_id].add(r.work_id)
        return cls(
            record_work={m: w.work_id for w in works for m in w.members},
            canonical={w.canonical for w in works},
            restates=dict(restates),
        )

    @classmethod
    def load(cls, works_dir: Path, chunks_path: Path) -> "WorkMap":
        meta_path = works_dir / META
        if not meta_path.is_file():
            raise WorksUnavailable(f"no works at {works_dir}: run `rag dedup run`")
        meta = WorksMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
        if meta.chunks_sha256 != file_sha256(chunks_path):
            raise WorksUnavailable("chunks changed since the works were built: run `rag dedup run`")
        works = read_jsonl(works_dir / WORKS, Work)
        restatements = (
            read_jsonl(works_dir / RESTATEMENTS, Restatement)
            if (works_dir / RESTATEMENTS).is_file()
            else []
        )
        record_work = json.loads((works_dir / RECORD_WORK).read_text(encoding="utf-8"))
        loaded = cls.from_works(works, restatements)
        if loaded.record_work != record_work:
            raise WorksUnavailable("works.jsonl and record_work.json disagree: run `rag dedup run`")
        return loaded

    def work_of(self, chunk: Chunk) -> str:
        # A record outside the works (a --limit run) is its own work.
        return self.record_work.get(chunk.record_id, f"work:{chunk.record_id}")

    def canonical_mask(self, chunks: Sequence[Chunk]) -> np.ndarray:
        return np.array(
            [c.record_id in self.canonical or c.record_id not in self.record_work for c in chunks]
        )


def collapse(
    ranked: Ranked,
    chunks: Sequence[Chunk],
    works: WorkMap,
    settings: DedupSettings,
    k: int,
    vectors: np.ndarray | None = None,
) -> tuple[Ranked, dict[int, list[int]]]:
    """The top k after cap, MMR and demotion, and per kept position the positions it hides."""
    hidden: dict[int, list[int]] = {}
    if settings.per_work_cap:
        ranked, hidden = cap_per_work(ranked, chunks, works, settings.per_work_cap)
    if settings.mmr_lambda is not None:
        if vectors is None:
            raise ValueError("MMR needs the dense vectors")
        ranked = mmr(ranked, vectors, settings.mmr_lambda, k)
    ranked = ranked[:k]
    if settings.survey_demotion:
        # Last, and only within the results: a source deeper in the candidates
        # would otherwise push the survey out without coming in itself.
        ranked = demote_restatements(ranked, chunks, works)
    kept = {i for i, _ in ranked}
    return ranked, {i: hidden[i] for i in kept if i in hidden}


def demote_restatements(ranked: Ranked, chunks: Sequence[Chunk], works: WorkMap) -> Ranked:
    """Move each restating survey chunk to just after the best chunk of the work it
    restates, when that work is in `ranked` below it. Nothing is dropped."""
    best_of_work: dict[str, int] = {}
    for position, (i, _) in enumerate(ranked):
        best_of_work.setdefault(works.work_of(chunks[i]), position)
    keys = []
    for position, (i, _) in enumerate(ranked):
        targets = [
            best_of_work[w] for w in works.restates.get(chunks[i].chunk_id, ()) if w in best_of_work
        ]
        target = max(targets, default=-1)
        # A fractional key slots the survey chunk right after the target.
        keys.append(target + 0.5 if target > position else float(position))
    order = sorted(range(len(ranked)), key=lambda p: keys[p])
    return [ranked[p] for p in order]


def cap_per_work(
    ranked: Ranked, chunks: Sequence[Chunk], works: WorkMap, cap: int
) -> tuple[Ranked, dict[int, list[int]]]:
    kept: Ranked = []
    seen: dict[str, list[int]] = defaultdict(list)  # work -> kept positions
    hidden: dict[int, list[int]] = defaultdict(list)
    for i, score in ranked:
        work = works.work_of(chunks[i])
        if len(seen[work]) < cap:
            seen[work].append(i)
            kept.append((i, score))
        else:
            hidden[seen[work][0]].append(i)  # shown under the work's best chunk
    return kept, dict(hidden)


def mmr(ranked: Ranked, vectors: np.ndarray, lam: float, k: int) -> Ranked:
    """Greedy MMR over the candidates; relevance is the score scaled to [0, 1]."""
    if not ranked:
        return ranked
    scores = np.array([s for _, s in ranked])
    span = scores.max() - scores.min()
    relevance = (scores - scores.min()) / span if span > 0 else np.ones_like(scores)
    vecs = vectors[[i for i, _ in ranked]]
    picked: list[int] = []
    max_sim = np.full(len(ranked), -np.inf)
    remaining = set(range(len(ranked)))
    while remaining and len(picked) < k:
        candidates = sorted(remaining)
        novelty = np.where(np.isfinite(max_sim[candidates]), max_sim[candidates], 0.0)
        value = lam * relevance[candidates] - (1 - lam) * novelty
        best = candidates[int(np.argmax(value))]  # ties: the better-ranked candidate
        picked.append(best)
        remaining.remove(best)
        max_sim = np.maximum(max_sim, vecs @ vecs[best])
    rest = [p for p in range(len(ranked)) if p not in set(picked)]
    return [ranked[p] for p in picked + rest]
