"""Plain-text view of one query's results, for `rag inspect query`.

Each row shows the chunk's rank and score in every retriever side by side:
where BM25 and dense agree, where one finds what the other misses, and how
RRF combines them. "–" means the chunk wasn't in that retriever's candidates.
"""

from rag.parse.chunker import HEADER_SEPARATOR
from rag.retrieve.schemas import Hit

PREVIEW_CHARS = 80
HIDDEN_SHOWN = 3  # chunk ids listed per hit hidden by the per-work cap


def _rank_score(hit: Hit, name: str, fmt: str) -> str:
    if name not in hit.ranks:
        return f"{'–':>4} {'':>6}"
    return f"{hit.ranks[name]:>4} {format(hit.scores[name], fmt):>6}"


def render_hits(query: str, mode: str, hits: list[Hit]) -> str:
    lines = [
        f"query: {query!r}  mode={mode}  k={len(hits)}",
        "",
        f"{'#':>3} {'score':>7}  {'bm25 rk/score':>11}  {'dense rk/cos':>11}  chunk / section",
    ]
    for h in hits:
        c = h.chunk
        section = " > ".join(c.section_path) or "-"
        title = c.header.split(HEADER_SEPARATOR)[0]
        page = f"p{c.page}" if c.page else "p?"
        lines.append(
            f"{h.rank:>3} {h.score:>7.4f}  {_rank_score(h, 'bm25', '.2f')}  "
            f"{_rank_score(h, 'dense', '.3f')}  {c.chunk_id} {page} {c.element_type}"
        )
        lines.append(f"{'':>40}{title[:60]} | {section[:50]}")
        lines.append(f"{'':>40}{c.text.replace(chr(10), ' ')[:PREVIEW_CHARS]}")
        if h.hidden:
            # Over the per-work cap (D-041): same paper, not shown separately.
            shown = ", ".join(h.hidden[:HIDDEN_SHOWN]) + (
                ", …" if len(h.hidden) > HIDDEN_SHOWN else ""
            )
            lines.append(f"{'':>40}+{len(h.hidden)} more from {h.work_id}: {shown}")
    return "\n".join(lines)
