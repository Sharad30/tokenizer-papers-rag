"""Which manifest records are the same paper, by trustworthy identifiers only.

Records are linked when they share an arXiv id, DOI, Semantic Scholar id or
ACL id, or when one was derived from the other in Phase 1 (linked copies,
kept v1s). This is the judge for dup@5 and for counting a linked copy of a
gold paper as a hit (D-031). Phase 5's fuzzy resolver is what's being
tested, so it can never be the judge; its labelled pairs get added later.
"""

from collections.abc import Iterable

from rag.collect.schemas import ManifestRecord


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # The smaller id becomes the root, so group ids don't depend on input order.
            self.parent[max(ra, rb)] = min(ra, rb)


def identity_groups(records: Iterable[ManifestRecord]) -> dict[str, str]:
    """record_id -> group id (the smallest record_id in its group)."""
    uf = _UnionFind()
    first_with: dict[tuple[str, str], str] = {}
    for r in records:
        uf.find(r.record_id)
        keys = [
            ("arxiv", r.arxiv_id),
            ("doi", r.doi.lower() if r.doi else None),
            ("s2", r.s2_id),
            ("acl", r.acl_id),
        ]
        for key in keys:
            if key[1] is None:
                continue
            if key in first_with:
                uf.union(first_with[key], r.record_id)
            else:
                first_with[key] = r.record_id
        if r.derived_from:
            uf.union(r.derived_from, r.record_id)
    return {record_id: uf.find(record_id) for record_id in list(uf.parent)}
