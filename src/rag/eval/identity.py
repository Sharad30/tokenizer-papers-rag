"""Which manifest records are the same paper, by trustworthy identifiers only.

This is the judge for dup@5 and for counting a linked copy of a gold paper as
a hit (D-031). The rules live in `rag.common.identity`. Phase 5's fuzzy
resolver is what's being tested, so it can never be the judge; its labelled
pairs get added later.
"""

from collections.abc import Iterable

from rag.collect.schemas import ManifestRecord
from rag.common.identity import UnionFind, id_links


def identity_groups(records: Iterable[ManifestRecord]) -> dict[str, str]:
    """record_id -> group id (the smallest record_id in its group)."""
    uf = UnionFind()
    records = list(records)
    for r in records:
        uf.find(r.record_id)
    for a, b, _ in id_links(records):
        uf.union(a, b)
    return uf.groups()
