"""Plain-text view of one paper's chunks, for `rag inspect paper`.

Chunks are listed in order under their section headings, so a section
boundary in the output is a chunk boundary: a quick way to check "no chunk
crosses a section" by eye, next to the PDF.
"""

from rag.common.config import Config
from rag.common.io import read_jsonl
from rag.parse.pipeline import CHUNKS, STATUS, step_dir
from rag.parse.schemas import Chunk, IndexStatus

PREVIEW_CHARS = 80


def render_paper(config: Config, record_id: str) -> str:
    chunk_dir = step_dir(config, "chunk")
    if not (chunk_dir / CHUNKS).is_file():
        return f"No chunks yet in {chunk_dir}: run `rag parse all` first."
    status = next(
        (s for s in read_jsonl(chunk_dir / STATUS, IndexStatus) if s.record_id == record_id), None
    )
    if status is None:
        return f"{record_id} is not in the last parse run (wrong id, or outside --limit)."
    chunks = [c for c in read_jsonl(chunk_dir / CHUNKS, Chunk) if c.record_id == record_id]
    lines = [f"{record_id}  outcome={status.outcome}  chunks={status.n_chunks}"]
    if chunks:
        lines.append(f"title: {chunks[0].header.split(' > ')[0]}")
    previous: tuple[str | None, tuple[str, ...]] | None = None
    for c in chunks:
        heading = (c.section_number, tuple(c.section_path))
        if heading != previous:
            number = f"{c.section_number} " if c.section_number else ""
            indent = "  " * (len(c.section_path) - 1)
            lines.append(f"\n{indent}{number}{' > '.join(c.section_path)}")
            previous = heading
        preview = c.text.replace("\n", " ")[:PREVIEW_CHARS]
        page = f"p{c.page}" if c.page else "p?"
        lines.append(
            f"  #{c.chunk_id.rsplit('#', 1)[1]} {c.element_type:<14} "
            f"{c.token_count:>4} tok {page:>4}  {preview}"
        )
    return "\n".join(lines)
