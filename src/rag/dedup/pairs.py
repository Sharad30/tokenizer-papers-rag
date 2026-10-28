"""Labelled pairs for the resolver: sample candidates, then score against labels.

`rag dedup pairs` writes record pairs for Sharad to label, drawn from four
bands so the labels test both kinds of mistake:

    id_linked      joined by identifiers (should be same; checks the ID rules)
    fuzzy_merged   joined by title or text evidence (the merges most likely wrong)
    near_miss      shared authors and a close title or some shared text, but
                   left apart (the misses most likely wrong)
    topical        very similar abstracts, different authors, left apart
                   (semantic similarity is not identity)

The candidates file shows the evidence but not the verdict or the band; the
band of each pair goes to data/works/pair_bands.json for the score breakdown.

`rag dedup score` compares the current works with eval/pairs.jsonl:
precision (of the pairs the resolver joins, how many are the same paper)
and recall (of the pairs that are the same paper, how many it joins).
"""

import json
import logging
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz

from rag.collect.schemas import ManifestRecord
from rag.common.config import Config
from rag.common.io import write_json, write_jsonl
from rag.common.provenance import config_hash, git_commit
from rag.common.schemas import StrictModel
from rag.dedup.config import PairSamplingConfig
from rag.dedup.minhash import estimate_jaccard
from rag.dedup.resolve import author_overlap, normalise, surnames
from rag.dedup.schemas import Work
from rag.eval.schemas import LabelledPair, PairCandidate

log = logging.getLogger(__name__)

PAIR_BANDS = "pair_bands.json"
Pair = tuple[str, str]


def _key(a: str, b: str) -> Pair:
    return (a, b) if a < b else (b, a)


def sample_pairs(
    records: Sequence[ManifestRecord],
    works: Sequence[Work],
    signatures: Mapping[str, np.ndarray],
    abstract_vectors: Mapping[str, np.ndarray],
    author_overlap_min: float,
    config: PairSamplingConfig,
) -> dict[Pair, str]:
    """pair -> band, for up to `config.per_band[band]` pairs per band."""
    rng = random.Random(config.seed)
    work_of = {m: w.work_id for w in works for m in w.members}
    by_id = {r.record_id: r for r in records}
    authors = {r.record_id: surnames(r.authors) for r in records}

    pools: dict[str, list[Pair]] = {
        "id_linked": sorted({_key(e.a, e.b) for w in works for e in w.edges if e.step == "ids"}),
        "fuzzy_merged": sorted({_key(e.a, e.b) for w in works for e in w.edges if e.step != "ids"}),
        "near_miss": [],
        "topical": [],
    }
    # Records sharing a surname, in different works.
    by_surname: dict[str, list[str]] = defaultdict(list)
    for rid, names in authors.items():
        for name in names:
            by_surname[name].append(rid)
    shared = {_key(a, b) for ids in by_surname.values() for a, b in combinations(sorted(ids), 2)}
    for a, b in sorted(shared):
        if work_of[a] == work_of[b]:
            continue
        if author_overlap(authors[a], authors[b]) < author_overlap_min:
            continue
        ratio = fuzz.token_sort_ratio(normalise(by_id[a].title), normalise(by_id[b].title))
        jaccard = estimate_jaccard(signatures[a], signatures[b])
        if ratio >= config.near_miss_title_min or jaccard >= config.near_miss_jaccard_min:
            pools["near_miss"].append((a, b))

    ids = sorted(rid for rid in abstract_vectors if rid in work_of)
    if ids:
        vectors = np.stack([abstract_vectors[rid] for rid in ids])
        sims = np.triu(vectors @ vectors.T, k=1)
        order = np.dstack(np.unravel_index(np.argsort(-sims, axis=None), sims.shape))[0]
        for i, j in order:
            if len(pools["topical"]) >= config.topical_pool:
                break
            a, b = ids[i], ids[j]
            if work_of[a] != work_of[b] and (
                author_overlap(authors[a], authors[b]) < author_overlap_min
            ):
                pools["topical"].append((a, b))
    else:
        log.warning("No abstract vectors (dense index missing): no topical pairs")

    chosen: dict[Pair, str] = {}
    for band, pool in pools.items():
        # One record pair per pair of works: versions of one paper would
        # otherwise offer the same judgement several times.
        by_works: dict[Pair, Pair] = {}
        for a, b in pool:
            by_works.setdefault(_key(work_of[a], work_of[b]), (a, b))
        taken = {_key(work_of[a], work_of[b]) for a, b in chosen}
        free = [p for w, p in by_works.items() if w not in taken or band == "id_linked"]
        for pair in rng.sample(free, min(config.per_band[band], len(free))):
            chosen[pair] = band
    return chosen


def candidates(
    pairs: Sequence[Pair], records: Mapping[str, ManifestRecord], signatures, seed: int
) -> list[PairCandidate]:
    out = []
    for a, b in pairs:
        ra, rb = records[a], records[b]
        out.append(
            PairCandidate(
                a=a,
                b=b,
                title_a=ra.title,
                title_b=rb.title,
                authors_a=ra.authors,
                authors_b=rb.authors,
                year_a=ra.year,
                year_b=rb.year,
                venue_a=ra.venue,
                venue_b=rb.venue,
                url_a=ra.url,
                url_b=rb.url,
                title_ratio=round(
                    fuzz.token_sort_ratio(normalise(ra.title), normalise(rb.title)), 1
                ),
                author_overlap=round(author_overlap(surnames(ra.authors), surnames(rb.authors)), 2),
                text_jaccard=round(estimate_jaccard(signatures[a], signatures[b]), 3),
            )
        )
    random.Random(seed).shuffle(out)  # bands interleaved, so position gives nothing away
    return out


def write_candidates(config: Config, chosen: Mapping[Pair, str], rows: Sequence[PairCandidate]):
    path = config.paths.eval / config.dedup.pairs.candidates_file
    write_jsonl(path, rows)
    write_json(
        config.paths.works / PAIR_BANDS, {f"{a}|{b}": band for (a, b), band in chosen.items()}
    )
    return path


# --- scoring -------------------------------------------------------------------


class PairError(StrictModel):
    a: str
    b: str
    label_same: bool
    kind: str
    band: str | None


class ResolverScore(StrictModel):
    created_at: datetime
    git_commit: str
    config_hash: str
    dedup_config: dict
    n_pairs: int
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    precision: float | None  # None when the resolver joined none of the pairs
    recall: float | None  # None when no pair is labelled same
    by_band: dict[str, dict[str, int]]
    errors: list[PairError]


def score(
    pairs: Sequence[LabelledPair], record_work: Mapping[str, str], bands: Mapping[str, str]
) -> tuple[dict[str, int], dict[str, dict[str, int]], list[PairError]]:
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    by_band: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    errors = []
    for p in pairs:
        joined = record_work.get(p.a) is not None and record_work.get(p.a) == record_work.get(p.b)
        cell = ("t" if joined == p.same else "f") + ("p" if joined else "n")
        band = bands.get(f"{p.a}|{p.b}") or bands.get(f"{p.b}|{p.a}")
        counts[cell] += 1
        by_band[band or "unsampled"][cell] += 1
        if joined != p.same:
            errors.append(PairError(a=p.a, b=p.b, label_same=p.same, kind=p.kind, band=band))
    return counts, dict(by_band), errors


def resolver_score(config: Config, pairs: Sequence[LabelledPair], record_work, bands) -> Path:
    counts, by_band, errors = score(pairs, record_work, bands)
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    result = ResolverScore(
        created_at=datetime.now(UTC),
        git_commit=git_commit(),
        config_hash=config_hash(config),
        dedup_config=config.dedup.model_dump(mode="json"),
        n_pairs=len(pairs),
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        true_negative=counts["tn"],
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / (tp + fn) if tp + fn else None,
        by_band=by_band,
        errors=errors,
    )
    path = config.paths.results / f"{datetime.now(UTC).date().isoformat()}-dedup-resolver.json"
    write_json(path, result)
    return path


def read_bands(works_dir: Path) -> dict[str, str]:
    path = works_dir / PAIR_BANDS
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
