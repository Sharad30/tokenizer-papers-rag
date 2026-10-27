"""StructuredPaper -> Chunks, following the element rules in PLAN.md and D-017 to D-020.

    abstract        its own chunk(s)
    paragraphs      packed within one section up to target_tokens, never across
                    sections; a long paragraph is split on sentences, and a
                    sentence longer than max_tokens on tokens
    overlap         the previous chunk's last whole sentences, up to
                    overlap_tokens, within the same section only
    tables          caption + markdown rows, split by rows, header row repeated
    captions        figure captions as their own chunks
    formulas        display formulas kept if short, else a placeholder
    references      never chunked (they are not in StructuredPaper's sections)

Size invariant, checked per chunk: header tokens + body tokens + 2 special
tokens <= embedding_max_tokens. The header is capped at the room max_tokens
leaves, so the body can always use up to max_tokens.
"""

import logging
from dataclasses import dataclass, field

from rag.parse.config import ChunkConfig
from rag.parse.schemas import Chunk, ElementType, Paragraph, StructuredPaper, Table
from rag.parse.tokens import TokenCounter

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = 2  # [CLS] and [SEP], added by the model around every input
HEADER_SEPARATOR = " > "
PARAGRAPH_JOIN = "\n\n"
FORMULA_PLACEHOLDER = "[equation]"  # stands in for a long display formula (D-019)


@dataclass(frozen=True)
class Piece:
    """The unit chunks are built from: one sentence, or part of an over-long one."""

    text: str
    tokens: int
    paragraph: int  # index within its section, to rejoin paragraphs with a blank line
    page: int | None
    cited_refs: tuple[str, ...] = ()


@dataclass
class ChunkResult:
    chunks: list[Chunk]
    # Sentences longer than max_tokens, cut on tokens (worth watching in stats).
    split_sentences: int = 0
    formula_placeholders: int = 0


@dataclass
class _Draft:
    """A chunk being filled. Leading overlap pieces repeat the previous chunk."""

    pieces: list[Piece] = field(default_factory=list)
    n_overlap: int = 0

    @property
    def tokens(self) -> int:
        return sum(p.tokens for p in self.pieces)

    @property
    def has_new_content(self) -> bool:
        return len(self.pieces) > self.n_overlap


class Chunker:
    def __init__(self, config: ChunkConfig, counter: TokenCounter) -> None:
        self._config = config
        self._counter = counter
        self._header_max = config.embedding_max_tokens - SPECIAL_TOKENS - config.max_tokens

    def chunk(
        self,
        paper: StructuredPaper,
        arxiv_id: str | None = None,
        arxiv_version: int | None = None,
    ) -> ChunkResult:
        result = ChunkResult(chunks=[])
        emit = _Emitter(paper, arxiv_id, arxiv_version, result.chunks, self)

        if paper.abstract:
            for draft in self._pack(paper.abstract, result, overlap=False):
                emit(ElementType.ABSTRACT, ["Abstract"], None, draft)
        for section in paper.sections:
            for draft in self._pack(section.paragraphs, result, overlap=True):
                emit(ElementType.PARAGRAPH, section.path, section.number, draft)
        for table in paper.tables:
            for text in self._table_texts(table):
                emit.text(ElementType.TABLE, [table.label or "Table"], text, table.page)
        if self._config.captions == "own_chunk":
            for figure in paper.figures:
                caption = _labelled(figure.label, figure.caption)
                for text in self._fit(caption):
                    emit.text(
                        ElementType.FIGURE_CAPTION, [figure.label or "Figure"], text, figure.page
                    )
        return result

    def count(self, text: str) -> int:
        return self._counter.count(text)

    # --- headers ------------------------------------------------------------

    def header(self, title: str, path: list[str]) -> tuple[str, int]:
        header = HEADER_SEPARATOR.join([title, *path]) if title else HEADER_SEPARATOR.join(path)
        tokens = self._counter.count(header)
        if tokens > self._header_max:
            # A very long title or path: keep its start rather than squeeze the body.
            header = self._counter.split(header, self._header_max)[0]
            tokens = self._counter.count(header)
        return header, tokens

    # --- paragraphs ---------------------------------------------------------

    def _pieces(self, paragraphs: list[Paragraph], result: ChunkResult) -> list[list[Piece]]:
        out = []
        for index, paragraph in enumerate(paragraphs):
            pieces: list[Piece] = []
            for sentence in paragraph.sentences:
                text = sentence.text
                if paragraph.formula_id is not None and len(text) > self._config.formula_max_chars:
                    text = FORMULA_PLACEHOLDER
                    result.formula_placeholders += 1
                tokens = self._counter.count(text)
                if tokens <= self._config.max_tokens:
                    parts = [(text, tokens)]
                else:
                    result.split_sentences += 1
                    logger.debug("Sentence of %d tokens split on tokens", tokens)
                    parts = [
                        (t, self._counter.count(t))
                        for t in self._counter.split(text, self._config.max_tokens)
                    ]
                refs = tuple(sentence.cited_refs)
                pieces += [Piece(t, n, index, paragraph.page, refs) for t, n in parts]
            if pieces:
                out.append(pieces)
        return out

    def _pack(
        self, paragraphs: list[Paragraph], result: ChunkResult, overlap: bool
    ) -> list[_Draft]:
        """Greedy packing: whole paragraphs while they fit the target, else sentences."""
        c = self._config
        drafts: list[_Draft] = []
        draft = _Draft()

        def flush() -> None:
            nonlocal draft
            if not draft.has_new_content:
                return
            drafts.append(draft)
            draft = _Draft()
            if overlap:
                tail = _overlap_tail(drafts[-1].pieces, c.overlap_tokens)
                draft = _Draft(pieces=tail, n_overlap=len(tail))

        def add(piece: Piece) -> None:
            nonlocal draft
            if draft.tokens + piece.tokens > c.max_tokens:
                if draft.has_new_content:
                    flush()
                if draft.tokens + piece.tokens > c.max_tokens:
                    draft = _Draft()  # the overlap itself doesn't fit: drop it
            draft.pieces.append(piece)

        for pieces in self._pieces(paragraphs, result):
            paragraph_tokens = sum(p.tokens for p in pieces)
            if draft.has_new_content and draft.tokens + paragraph_tokens > c.target_tokens:
                flush()
            if draft.tokens + paragraph_tokens <= c.max_tokens:
                for piece in pieces:
                    add(piece)
                continue
            for piece in pieces:  # too long to keep whole: fill sentence by sentence
                if draft.has_new_content and draft.tokens + piece.tokens > c.target_tokens:
                    flush()
                add(piece)
        if draft.has_new_content:
            drafts.append(draft)
        return drafts

    # --- tables and captions ----------------------------------------------------

    def _table_texts(self, table: Table) -> list[str]:
        caption = _labelled(table.label, table.caption)
        if not table.rows:
            return self._fit(caption) if caption else []
        header_row, body = _markdown_row(table.rows[0]), [_markdown_row(r) for r in table.rows[1:]]
        lead = PARAGRAPH_JOIN.join(t for t in (caption, header_row) if t)
        texts: list[str] = []
        part: list[str] = []
        for row in body:
            candidate = "\n".join([lead, *part, row])
            too_many = len(part) >= self._config.table_max_rows_per_chunk
            if part and (too_many or self._counter.count(candidate) > self._config.max_tokens):
                texts.append("\n".join([lead, *part]))
                part = []
            part.append(row)
        texts.append("\n".join([lead, *part]))
        # A part can still be over the limit (a huge caption or row): cut it on tokens.
        return [piece for text in texts for piece in self._fit(text)]

    def _fit(self, text: str) -> list[str]:
        if not text:
            return []
        if self._counter.count(text) <= self._config.max_tokens:
            return [text]
        return self._counter.split(text, self._config.max_tokens)


class _Emitter:
    """Turns drafts and texts into numbered Chunk records."""

    def __init__(
        self,
        paper: StructuredPaper,
        arxiv_id: str | None,
        arxiv_version: int | None,
        out: list[Chunk],
        chunker: Chunker,
    ) -> None:
        self._paper, self._out, self._chunker = paper, out, chunker
        self._ids = (arxiv_id, arxiv_version)

    def __call__(
        self, kind: ElementType, path: list[str], number: str | None, draft: _Draft
    ) -> None:
        paragraphs: dict[int, list[str]] = {}
        for piece in draft.pieces:
            paragraphs.setdefault(piece.paragraph, []).append(piece.text)
        text = PARAGRAPH_JOIN.join(" ".join(texts) for texts in paragraphs.values())
        new = draft.pieces[draft.n_overlap :]
        refs = list(dict.fromkeys(r for p in draft.pieces for r in p.cited_refs))
        self._add(kind, path, number, text, new[0].page, refs)

    def text(self, kind: ElementType, path: list[str], text: str, page: int | None) -> None:
        self._add(kind, path, None, text, page, [])

    def _add(
        self,
        kind: ElementType,
        path: list[str],
        number: str | None,
        text: str,
        page: int | None,
        refs: list[str],
    ) -> None:
        header, header_tokens = self._chunker.header(self._paper.title, path)
        self._out.append(
            Chunk(
                chunk_id=f"{self._paper.record_id}#{len(self._out):04d}",
                record_id=self._paper.record_id,
                arxiv_id=self._ids[0],
                arxiv_version=self._ids[1],
                element_type=kind,
                section_path=path,
                section_number=number,
                page=page,
                header=header,
                text=text,
                token_count=self._chunker.count(text),
                header_token_count=header_tokens,
                cited_refs=refs,
                parser=self._paper.parser,
            )
        )


def _overlap_tail(pieces: list[Piece], budget: int) -> list[Piece]:
    """The last whole sentences of a chunk that fit in the overlap budget."""
    tail: list[Piece] = []
    for piece in reversed(pieces):
        if sum(p.tokens for p in tail) + piece.tokens > budget:
            break
        tail.insert(0, piece)
    return tail


def _labelled(label: str | None, caption: str) -> str:
    if label and caption:
        return f"{label}: {caption}"
    return label or caption


def _markdown_row(cells: list[str]) -> str:
    return "| " + " | ".join(cell.replace("|", "/") for cell in cells) + " |"
