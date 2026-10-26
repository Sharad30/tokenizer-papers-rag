"""What text each index sees, and how BM25 splits it into terms (D-027, D-029)."""

import re

from bm25s.stopwords import STOPWORDS_EN

from rag.parse.schemas import Chunk

# A term is a run of letters/digits (any script), optionally joined by
# word-internal "-", "." or "_": keeps "gpt-2", "byt5", "bpe-dropout", "3.5".
_TERM = re.compile(r"[^\W_]+(?:[-._][^\W_]+)*")
_JOINERS = re.compile(r"[-._]")
STOPWORDS = {"en": frozenset(STOPWORDS_EN), "none": frozenset()}


def chunk_text(chunk: Chunk, with_header: bool) -> str:
    """The chunk as an index sees it: optionally "Title > Section" first."""
    return f"{chunk.header}\n{chunk.text}" if with_header else chunk.text


def terms(text: str, stopwords: frozenset[str], split_compounds: bool) -> list[str]:
    """Lowercased BM25 terms. A compound keeps its whole form and, optionally, its parts.

    "Byte-level BPE" -> ["byte-level", "byte", "level", "bpe"], so a query
    "byte level" still matches, and "gpt-2" stays more specific than "gpt".
    """
    out: list[str] = []
    for match in _TERM.finditer(text.lower()):
        term = match.group()
        if term not in stopwords:
            out.append(term)
        if split_compounds and _JOINERS.search(term):
            out.extend(p for p in _JOINERS.split(term) if p and p not in stopwords)
    return out
