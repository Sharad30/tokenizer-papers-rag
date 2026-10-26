"""`rag eval find`: locate text in the chunks, to copy gold spans from (D-024).

This is grep, not a retriever. Hits come out in corpus order (paper, then
chunk position) and are never scored or ranked, so finding spans for
questions you've already written can't bias the gold towards whatever a
retriever happens to like (pooling bias).
"""

from dataclasses import dataclass

from rag.eval.spans import ChunkTexts, IndexedChunk, normalise
from rag.parse.chunker import HEADER_SEPARATOR
from rag.parse.schemas import Chunk

EXCERPT_CHARS = 100  # shown either side of the first match


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    excerpt: str  # normalised text, so it can be copied as a span as is


@dataclass(frozen=True)
class FindResult:
    hits: list[Hit]  # at most max_hits
    total: int  # all matching chunks


def find(
    texts: ChunkTexts,
    query: str,
    *,
    max_hits: int,
    paper: str | None = None,
    all_words: bool = False,
) -> FindResult:
    """Chunks containing `query` (or, with all_words, every word of it), in corpus order."""
    needles = normalise(query).split() if all_words else [normalise(query)]
    pool = texts.paper(paper) if paper else texts
    hits: list[Hit] = []
    total = 0
    for ic in pool:
        if all(n in ic.norm for n in needles):
            total += 1
            if len(hits) < max_hits:
                hits.append(Hit(ic.chunk, _excerpt(ic, needles[0])))
    return FindResult(hits, total)


def _excerpt(ic: IndexedChunk, needle: str) -> str:
    at = ic.norm.find(needle)
    lo, hi = max(0, at - EXCERPT_CHARS), at + len(needle) + EXCERPT_CHARS
    return ("…" if lo else "") + ic.norm[lo:hi] + ("…" if hi < len(ic.norm) else "")


def render(result: FindResult, query: str) -> str:
    if not result.hits:
        return f"No chunk contains {query!r}."
    lines: list[str] = []
    record = None
    for hit in result.hits:
        c = hit.chunk
        if c.record_id != record:
            record = c.record_id
            lines.append(f"\n{c.record_id}  {c.header.split(HEADER_SEPARATOR)[0]}")
        page = f"p{c.page}" if c.page else "p?"
        lines.append(f"  {c.chunk_id}  {page}  {' > '.join(c.section_path) or '-'}")
        lines.append(f"    {hit.excerpt}")
    shown = len(result.hits)
    more = f" (showing the first {shown}; narrow with --paper)" if result.total > shown else ""
    lines.append(f"\n{result.total} matching chunks{more}")
    return "\n".join(lines).lstrip("\n")
