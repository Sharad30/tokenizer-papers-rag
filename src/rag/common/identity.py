"""Union-find and the identifier rules that say two records are the same paper.

Shared by the eval judge (`rag.eval.identity`, D-031) and the first step of
Phase 5's resolver (`rag.dedup.resolve`): both trust exactly these links.
"""

from collections.abc import Iterable, Iterator

from rag.collect.schemas import ManifestRecord


class UnionFind:
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

    def groups(self) -> dict[str, str]:
        """item -> group id (the smallest item in its group)."""
        return {x: self.find(x) for x in list(self.parent)}


def id_links(records: Iterable[ManifestRecord]) -> Iterator[tuple[str, str, str]]:
    """(record_id, record_id, reason) for every trustworthy identifier link.

    Records are linked when they share an arXiv id, DOI, Semantic Scholar id or
    ACL id, or when one was derived from the other in Phase 1 (linked copies,
    kept v1s). Each record links to the first record seen with the same key.
    """
    first_with: dict[tuple[str, str], str] = {}
    for r in records:
        keys = [
            ("arxiv", r.arxiv_id),
            ("doi", r.doi.lower() if r.doi else None),
            ("s2", r.s2_id),
            ("acl", r.acl_id),
        ]
        for kind, value in keys:
            if value is None:
                continue
            if (kind, value) in first_with:
                yield first_with[(kind, value)], r.record_id, kind
            else:
                first_with[(kind, value)] = r.record_id
        if r.derived_from:
            yield r.derived_from, r.record_id, "derived_from"
