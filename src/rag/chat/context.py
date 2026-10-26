"""The retrieved chunks as the model sees them, and as the user sees sources (D-034).

Each chunk gets a label ([C1], [C2], ... in rank order) and a header built
from the chunk and the manifest. The model only ever cites labels; every
title, year and section the user sees comes from here, never from the model.
"""

from dataclasses import dataclass

from rag.collect.schemas import ManifestRecord
from rag.parse.chunker import HEADER_SEPARATOR
from rag.retrieve.schemas import Hit


@dataclass(frozen=True)
class ContextChunk:
    label: str  # "C1"
    hit: Hit
    title: str
    year: int | None

    @property
    def section(self) -> str:
        c = self.hit.chunk
        name = c.section_path[-1] if c.section_path else c.element_type
        return f"§{c.section_number} {name}" if c.section_number else name

    @property
    def page(self) -> str:
        return f"p.{self.hit.chunk.page}" if self.hit.chunk.page else ""


def build_context(hits: list[Hit], records: dict[str, ManifestRecord]) -> list[ContextChunk]:
    """Label hits C1..Cn in rank order (best first: D-034)."""
    out = []
    for i, hit in enumerate(hits, start=1):
        record = records.get(hit.chunk.record_id)
        # The manifest title is authoritative; the chunk header's title is a fallback.
        title = record.title if record else hit.chunk.header.split(HEADER_SEPARATOR)[0]
        out.append(ContextChunk(f"C{i}", hit, title, record.year if record else None))
    return out


def _year(year: int | None) -> str:
    return f" ({year})" if year else ""


def render_context(chunks: list[ContextChunk]) -> str:
    """The context block in the prompt: one labelled, headed passage per chunk."""
    blocks = []
    for cc in chunks:
        where = ", ".join(p for p in (cc.section, cc.page) if p)
        blocks.append(f"[{cc.label}] {cc.title}{_year(cc.year)} > {where}\n{cc.hit.chunk.text}")
    return "\n\n".join(blocks)


def source_line(cc: ContextChunk) -> str:
    """One entry of the Sources list, from metadata only (D-035)."""
    where = ", ".join(p for p in (cc.section, cc.page) if p)
    return f"{cc.title}{_year(cc.year)}, {where} · {cc.hit.chunk.chunk_id}"
