"""Which manifest records are the same paper: the judge for dup@5 (D-031, D-042).

Two sources, both independent of Phase 5's resolver (which is what's being
tested, so it can never be the judge):

- trustworthy identifiers (`rag.common.identity`): shared arXiv id, DOI,
  S2 id or ACL id, or a Phase 1 derivation;
- Sharad's labelled pairs (`eval/pairs.jsonl`) marked `same`, which add the
  copies the identifiers miss.

A copy the identifiers miss and nobody labelled counts as a different paper,
so dup@5 is a lower bound.
"""

import json
from collections.abc import Iterable
from pathlib import Path

from rag.collect.schemas import ManifestRecord
from rag.common.identity import UnionFind, id_links
from rag.eval.schemas import LabelledPair


def identity_groups(
    records: Iterable[ManifestRecord], same_pairs: Iterable[LabelledPair] = ()
) -> dict[str, str]:
    """record_id -> group id (the smallest record_id in its group)."""
    uf = UnionFind()
    records = list(records)
    for r in records:
        uf.find(r.record_id)
    for a, b, _ in id_links(records):
        uf.union(a, b)
    for pair in same_pairs:
        if pair.same:
            uf.union(pair.a, pair.b)
    return uf.groups()


def load_pairs(path: Path) -> list[LabelledPair]:
    """The labelled pairs, or none if the file doesn't exist yet."""
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [LabelledPair.model_validate(json.loads(line)) for line in lines if line.strip()]
