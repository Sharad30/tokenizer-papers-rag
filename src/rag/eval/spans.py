"""Gold spans to chunks (D-022, D-023).

A gold span is verbatim text in a named paper. Which chunks contain it is
worked out here, every time it's needed, and never stored: a stored mapping
would go stale the moment chunking changes. The Phase 4 harness scores
Recall@5 with `map_span`.
"""

import difflib
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from rag.common.io import read_jsonl
from rag.eval.schemas import GoldSpan
from rag.parse.schemas import Chunk

# Characters PDFs and keyboards disagree on. Table pipes come from our own
# markdown rows (D-020), not from the paper, so they are dropped.
_UNIFY = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "−": "-",
        "|": " ",
    }
)
SUGGEST_CONTEXT_CHARS = 40  # how much text either side of a near-miss to show


def normalise(text: str) -> str:
    """The text both spans and chunks are compared in.

    NFKC folds ligatures ("ﬁ" -> "fi") and full-width forms; casefold makes
    "BPE" match "bpe". Hyphenation is not undone: "prob-lem" stays, which is
    why spans should be copied from chunk text (`rag eval find`).
    """
    folded = unicodedata.normalize("NFKC", text).casefold().translate(_UNIFY)
    return " ".join(folded.split())


@dataclass(frozen=True)
class IndexedChunk:
    chunk: Chunk
    norm: str  # normalise(chunk.text), computed once


class ChunkTexts:
    """Every chunk's normalised text, grouped by paper, in corpus order."""

    def __init__(self, chunks: Iterable[Chunk]) -> None:
        self.by_record: dict[str, list[IndexedChunk]] = {}
        for c in chunks:
            self.by_record.setdefault(c.record_id, []).append(IndexedChunk(c, normalise(c.text)))

    @classmethod
    def from_file(cls, path: Path) -> "ChunkTexts":
        return cls(read_jsonl(path, Chunk))

    def paper(self, record_id: str) -> list[IndexedChunk]:
        return self.by_record.get(record_id, [])

    def __iter__(self) -> Iterator[IndexedChunk]:
        for chunks in self.by_record.values():
            yield from chunks


def map_span(gold: GoldSpan, texts: ChunkTexts) -> list[str]:
    """Chunk ids in the gold paper that contain the whole span.

    More than one is normal: sentence overlap (D-017) copies a chunk's last
    sentences into the next chunk.
    """
    needle = normalise(gold.span)
    return [ic.chunk.chunk_id for ic in texts.paper(gold.record_id) if needle in ic.norm]


def elsewhere(span: str, texts: ChunkTexts, exclude: str) -> list[str]:
    """Other papers containing the same span: linked copies, restatements, quotes."""
    needle = normalise(span)
    return [
        record_id
        for record_id, chunks in texts.by_record.items()
        if record_id != exclude and any(needle in ic.norm for ic in chunks)
    ]


@dataclass(frozen=True)
class Suggestion:
    chunk_id: str
    similarity: float  # difflib ratio, 0..1
    excerpt: str  # normalised text around the closest match


def suggest(span: str, chunks: list[IndexedChunk]) -> Suggestion | None:
    """The passage in one paper closest to a span that didn't map.

    A repair hint only, never a match (D-023): fuzzy matching would need a
    threshold and can land on the wrong passage.
    """
    needle = normalise(span)
    best: Suggestion | None = None
    for ic in chunks:
        matcher = difflib.SequenceMatcher(None, needle, ic.norm, autojunk=False)
        block = matcher.find_longest_match(0, len(needle), 0, len(ic.norm))
        # Align the span on its longest exact run, then score that window.
        start = max(0, block.b - block.a)
        window = ic.norm[start : start + len(needle)]
        ratio = difflib.SequenceMatcher(None, needle, window, autojunk=False).ratio()
        if best is None or ratio > best.similarity:
            lo = max(0, start - SUGGEST_CONTEXT_CHARS)
            hi = start + len(needle) + SUGGEST_CONTEXT_CHARS
            best = Suggestion(ic.chunk.chunk_id, round(ratio, 3), ic.norm[lo:hi])
    return best
