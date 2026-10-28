"""Resolve manifest records into works: which records are the same paper.

Three kinds of evidence, in order of confidence (D-040):

1. ids      a shared arXiv id, DOI, S2 id or ACL id, or a Phase 1 derivation
            (`rag.common.identity`). Trusted outright.
2. title    near-identical normalised titles, shared authors, years close.
3. minhash  most of the full text shared (MinHash estimate of shingle
            Jaccard), shared authors. Also confirms titles that are only
            close: "Towards Character-Level Transformer NMT ..." and
            "Towards Reasonably-Sized Character-Level ..." are one paper;
            "Hard Non-Monotonic Attention ..." and "Exact Hard Monotonic
            Attention ..." (same authors, title ratio 92) are two.

Semantic similarity is not identity: two papers measuring tokenizer
fertility read alike but are different work. That is why text evidence is
lexical (shared 5-word sequences, not embeddings) and always needs shared
authors.

Union-find makes merges transitive, so A≈B and B≈C would put A and C in one
work even if they share nothing. The chain guard only lets two parts join
when every pair of ID-linked units across them has direct evidence.
"""

import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from rapidfuzz import fuzz, process

from rag.collect.schemas import ManifestRecord
from rag.common.identity import UnionFind, id_links
from rag.dedup.config import DedupConfig
from rag.dedup.minhash import estimate_jaccard, lsh_candidates
from rag.dedup.schemas import Edge, Step, Work

NON_WORD = re.compile(r"[^\w\s]")
SPACES = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Lowercase words without accents or punctuation: "BabyLM’s Ge’ez" -> "babylm s ge ez"."""
    decomposed = unicodedata.normalize("NFKD", text)
    letters = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return SPACES.sub(" ", NON_WORD.sub(" ", letters.lower())).strip()


def surnames(authors: Sequence[str]) -> set[str]:
    return {words[-1] for words in (normalise(a).split() for a in authors) if words}


def author_overlap(a: set[str], b: set[str]) -> float:
    """Share of the shorter author list found in the other: a journal version
    that adds authors still overlaps fully."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


@dataclass
class Resolution:
    works: list[Work]
    record_work: dict[str, str]
    counts: dict[str, int]


class _Pairs:
    """Per-pair evidence, computed once."""

    def __init__(
        self, records: Sequence[ManifestRecord], signatures: Mapping[str, np.ndarray]
    ) -> None:
        self.records = {r.record_id: r for r in records}
        self.signatures = signatures
        self.authors = {r.record_id: surnames(r.authors) for r in records}

    def overlap(self, a: str, b: str) -> float:
        return author_overlap(self.authors[a], self.authors[b])

    def year_gap(self, a: str, b: str) -> int | None:
        ya, yb = self.records[a].year, self.records[b].year
        return abs(ya - yb) if ya is not None and yb is not None else None

    def jaccard(self, a: str, b: str) -> float:
        return estimate_jaccard(self.signatures[a], self.signatures[b])


def resolve(
    records: Sequence[ManifestRecord],
    signatures: Mapping[str, np.ndarray],
    n_chunks: Mapping[str, int],
    config: DedupConfig,
    steps: Sequence[Step] | None = None,
) -> Resolution:
    """Cluster records into works. `steps` overrides config.steps (for the ablation)."""
    steps = list(config.steps if steps is None else steps)
    records = sorted(records, key=lambda r: r.record_id)
    ids = [r.record_id for r in records]
    pairs = _Pairs(records, signatures)
    counts: dict[str, int] = defaultdict(int)

    # Units: records joined by trusted identifiers. Fuzzy evidence joins units.
    units = UnionFind()
    for rid in ids:
        units.find(rid)
    id_edges: list[Edge] = []
    if "ids" in steps:
        present = set(ids)
        for a, b, reason in id_links(records):
            if a in present and b in present:  # derived_from may point outside a --limit run
                id_edges.append(Edge(a=a, b=b, step="ids", reason=reason))
                units.union(a, b)
    unit_of = units.groups()
    counts["edges.ids"] = len(id_edges)

    fuzzy: dict[tuple[str, str], Edge] = {}
    review_pairs: list[tuple[str, str, float]] = []
    if "title" in steps:
        for edge, borderline in _title_edges(records, pairs, config):
            if unit_of[edge.a] == unit_of[edge.b]:
                continue  # already one unit: no new information
            if borderline:
                review_pairs.append((edge.a, edge.b, edge.title_ratio or 0.0))
            else:
                fuzzy[(edge.a, edge.b)] = edge
    if "minhash" in steps:
        for edge in _text_edges(ids, pairs, config, review_pairs, unit_of):
            fuzzy.setdefault((edge.a, edge.b), edge)
    for edge in fuzzy.values():
        counts[f"edges.{edge.step}"] += 1

    accepted, rejected = _guarded_merge(units, unit_of, list(fuzzy.values()), pairs, config)
    counts["edges.rejected_by_guard"] = len(rejected)

    groups: dict[str, list[str]] = defaultdict(list)
    for rid in ids:
        groups[units.find(rid)].append(rid)
    member_edges = id_edges + accepted
    works = [
        _work(members, member_edges, rejected, pairs, n_chunks, config)
        for members in groups.values()
    ]
    works.sort(key=lambda w: w.work_id)
    record_work = {rid: w.work_id for w in works for rid in w.members}
    counts["records"] = len(ids)
    counts["works"] = len(works)
    counts["works.multi_member"] = sum(len(w.members) > 1 for w in works)
    return Resolution(works=works, record_work=record_work, counts=dict(counts))


def _title_edges(
    records: Sequence[ManifestRecord], pairs: _Pairs, config: DedupConfig
) -> list[tuple[Edge, bool]]:
    """(edge, borderline) for title pairs at or above review_ratio with shared authors."""
    cfg = config.title
    titles = [normalise(r.title) for r in records]
    # All pairs at once in C; ~1.8k titles is a few million comparisons.
    scores = process.cdist(
        titles, titles, scorer=fuzz.token_sort_ratio, score_cutoff=cfg.review_ratio, workers=-1
    )
    out = []
    for i, j in zip(*np.nonzero(np.triu(scores, k=1)), strict=True):
        a, b = records[i].record_id, records[j].record_id
        overlap = pairs.overlap(a, b)
        gap = pairs.year_gap(a, b)
        if overlap < config.author_overlap_min or (gap is not None and gap > cfg.year_gap_max):
            continue
        ratio = round(float(scores[i, j]), 1)
        edge = Edge(
            a=a,
            b=b,
            step="title",
            reason="title",
            title_ratio=ratio,
            author_overlap=overlap,
            year_gap=gap,
            jaccard=round(pairs.jaccard(a, b), 3),
        )
        out.append((edge, ratio < cfg.min_ratio))
    return out


def _text_edges(
    ids: Sequence[str],
    pairs: _Pairs,
    config: DedupConfig,
    review_pairs: Sequence[tuple[str, str, float]],
    unit_of: Mapping[str, str],
) -> list[Edge]:
    """Pairs sharing most of their text (LSH candidates), and close titles confirmed by text."""
    cfg = config.minhash
    candidates = lsh_candidates({rid: pairs.signatures[rid] for rid in ids}, cfg.bands, cfg.rows)
    out = []
    for a, b in sorted(candidates):
        if unit_of[a] == unit_of[b]:
            continue
        jaccard = pairs.jaccard(a, b)
        overlap = pairs.overlap(a, b)
        if jaccard >= cfg.min_jaccard and overlap >= config.author_overlap_min:
            out.append(
                Edge(
                    a=a,
                    b=b,
                    step="minhash",
                    reason="text",
                    jaccard=round(jaccard, 3),
                    author_overlap=overlap,
                    year_gap=pairs.year_gap(a, b),
                )
            )
    for a, b, ratio in review_pairs:
        jaccard = pairs.jaccard(a, b)
        if jaccard >= config.title.review_min_jaccard:
            out.append(
                Edge(
                    a=a,
                    b=b,
                    step="minhash",
                    reason="title+text",
                    title_ratio=ratio,
                    jaccard=round(jaccard, 3),
                    author_overlap=pairs.overlap(a, b),
                    year_gap=pairs.year_gap(a, b),
                )
            )
    return out


def _guarded_merge(
    units: UnionFind,
    unit_of: Mapping[str, str],
    edges: list[Edge],
    pairs: _Pairs,
    config: DedupConfig,
) -> tuple[list[Edge], list[Edge]]:
    """Apply fuzzy edges strongest first; refuse one that would chain unrelated units."""
    members: dict[str, list[str]] = defaultdict(list)
    for rid, unit in unit_of.items():
        members[unit].append(rid)
    linked = {(e.a, e.b) for e in edges} | {(e.b, e.a) for e in edges}

    def evidence(unit_a: str, unit_b: str) -> bool:
        return any(
            (x, y) in linked or pairs.jaccard(x, y) >= config.cluster_floor
            for x in members[unit_a]
            for y in members[unit_b]
        )

    # Which units each current cluster holds.
    cluster_units: dict[str, set[str]] = {u: {u} for u in members}
    accepted, rejected = [], []
    for edge in sorted(edges, key=lambda e: (-(e.jaccard or 0), -(e.title_ratio or 0), e.a, e.b)):
        ca, cb = units.find(edge.a), units.find(edge.b)
        if ca == cb:
            accepted.append(edge)  # already joined through stronger edges
            continue
        if all(evidence(u, v) for u in cluster_units[ca] for v in cluster_units[cb]):
            joined = cluster_units.pop(ca) | cluster_units.pop(cb)
            units.union(ca, cb)
            cluster_units[units.find(ca)] = joined
            accepted.append(edge)
        else:
            rejected.append(edge)
    return accepted, rejected


def canonical_member(
    members: Sequence[str],
    records: Mapping[str, ManifestRecord],
    n_chunks: Mapping[str, int],
    order: Sequence[str],
) -> tuple[str, str]:
    """The record that represents the work, and the rule that picked it."""

    def key(rid: str) -> tuple:
        r = records[rid]
        rank = order.index(r.source.value) if r.source.value in order else len(order)
        return (rank, -(r.arxiv_version or 0), -n_chunks.get(rid, 0), rid)

    best = min(members, key=key)
    source = records[best].source.value
    if len(members) == 1:
        return best, "only member"
    if source == "arxiv":
        return best, "latest arXiv version (no venue copy)"
    if source == "acl":
        return best, "venue version (ACL Anthology)"
    return best, f"{source} record"


def _work(
    members: list[str],
    edges: Sequence[Edge],
    rejected: Sequence[Edge],
    pairs: _Pairs,
    n_chunks: Mapping[str, int],
    config: DedupConfig,
) -> Work:
    inside = set(members)
    canonical, rule = canonical_member(members, pairs.records, n_chunks, config.canonical_order)
    return Work(
        work_id=f"work:{canonical}",
        canonical=canonical,
        canonical_rule=rule,
        members=sorted(members),
        aliases=sorted(m for m in members if m != canonical),
        edges=[e for e in edges if e.a in inside and e.b in inside],
        rejected=[e for e in rejected if e.a in inside or e.b in inside],
    )
