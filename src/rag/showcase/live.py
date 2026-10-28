"""One query's results, prepared for a page: shared by the duplicates and embeddings pages.

Pages pass in a loaded Retriever (the UI caches it); this module only runs
the query, times it, and marks each hit: is it the same paper as a hit
above it (by the resolver's works and by the dup@5 judge), is it another
record of that paper (a version or venue copy), does it contain an exact
term. No streamlit here, so it is tested like the rest of `rag`.
"""

import time
from collections.abc import Mapping

from rag.common.schemas import StrictModel
from rag.eval.metrics import copy_duplicate_rate_at_k, duplicate_rate_at_k
from rag.parse.chunker import HEADER_SEPARATOR
from rag.retrieve.retriever import Retriever
from rag.retrieve.text import chunk_text


class ShownHit(StrictModel):
    rank: int
    chunk_id: str
    record_id: str
    title: str
    section: str
    text: str
    score: float
    work_id: str  # by the resolver's works (the record itself when unknown)
    same_paper_as: int | None = None  # rank of the first hit above from the same work
    other_record: bool = False  # that earlier hit is a different record (version/copy)
    hidden: list[str] = []  # chunks of the same work hidden by the per-work cap
    has_term: bool | None = None  # contains the exact term asked about


class QueryView(StrictModel):
    hits: list[ShownHit]
    ms: float
    dup: float  # dup@k by the judge (identifiers + labelled pairs)
    copy_dup: float
    dup_resolver: float  # by the resolver's own works: not ground truth
    papers: int  # distinct works in the results


def run_query(
    retriever: Retriever,
    query: str,
    k: int,
    mode: str,
    judge: Mapping[str, str],
    work_of: Mapping[str, str],
    term: str | None = None,
) -> QueryView:
    tick = time.perf_counter()
    hits = retriever.retrieve(query, k=k, mode=mode)
    ms = (time.perf_counter() - tick) * 1000
    records = [h.chunk.record_id for h in hits]
    works = [work_of.get(r, r) for r in records]
    first_rank: dict[str, tuple[int, str]] = {}
    shown = []
    for h, work in zip(hits, works, strict=True):
        c = h.chunk
        earlier = first_rank.get(work)
        shown.append(
            ShownHit(
                rank=h.rank,
                chunk_id=c.chunk_id,
                record_id=c.record_id,
                title=c.header.split(HEADER_SEPARATOR, 1)[0],
                section=" > ".join(c.section_path) or "-",
                text=c.text,
                score=h.score,
                work_id=work,
                same_paper_as=earlier[0] if earlier else None,
                other_record=bool(earlier and earlier[1] != c.record_id),
                hidden=h.hidden,
                has_term=term.lower() in chunk_text(c, True).lower() if term else None,
            )
        )
        first_rank.setdefault(work, (h.rank, c.record_id))
    judged = [judge.get(r, r) for r in records]
    return QueryView(
        hits=shown,
        ms=round(ms, 1),
        dup=duplicate_rate_at_k(judged, k),
        copy_dup=copy_duplicate_rate_at_k(records, judge, k),
        dup_resolver=duplicate_rate_at_k(works, k),
        papers=len(set(works)),
    )


def badge(hit: ShownHit) -> str:
    """A short note for a hit that repeats a paper already shown above it."""
    if hit.same_paper_as is None:
        return ""
    if hit.other_record:
        return f"duplicate: another record (version or copy) of the paper at #{hit.same_paper_as}"
    return f"duplicate: another chunk of the paper at #{hit.same_paper_as}"


def preview(text: str, chars: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= chars else text[:chars].rsplit(" ", 1)[0] + " …"
