"""Records that cross step boundaries inside the parse stage.

    PDF -> TEI (GROBID) -> StructuredPaper -> Chunk

A StructuredPaper keeps the paper's own shape (sections, paragraphs,
sentences, tables, figures, formulas); chunking is a separate step, so chunk
settings can change without re-parsing.
"""

from enum import StrEnum

from rag.common.schemas import StrictModel


class Parser(StrEnum):
    GROBID = "grobid"
    PYMUPDF = "pymupdf"  # fallback: flat text, no sections or references (D-015)


class ParseOutcome(StrEnum):
    GROBID = "grobid"  # TEI with content
    FALLBACK = "fallback"  # GROBID gave nothing usable; PyMuPDF text instead
    FAILED = "failed"  # neither parser produced text
    NO_PDF = "no_pdf"  # accepted record without a downloaded PDF


class ElementType(StrEnum):
    ABSTRACT = "abstract"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FIGURE_CAPTION = "figure_caption"


class ParseStatus(StrictModel):
    """One accepted record's result from the grobid step."""

    record_id: str
    outcome: ParseOutcome
    detail: str | None = None  # e.g. "grobid 204: no content", the error message
    seconds: float | None = None


class Sentence(StrictModel):
    text: str
    # Bibliography ids this sentence cites, e.g. ["b12"], from <ref type="bibr">.
    cited_refs: list[str] = []


class Paragraph(StrictModel):
    sentences: list[Sentence]
    page: int | None = None  # first page, from GROBID coords
    # Set when this "paragraph" is a display formula, kept in reading order;
    # the chunker keeps it or swaps in a placeholder (D-019).
    formula_id: str | None = None


class Section(StrictModel):
    """One heading and the paragraphs directly under it.

    GROBID's TEI is flat (one <div> per heading); nesting comes from the
    heading numbers: "3.1" is a child of "3".
    """

    id: str  # "s007", position in the paper
    number: str | None = None  # "3.1" as printed; None for unnumbered headings
    title: str
    path: list[str]  # titles from the top level down, e.g. ["Method", "Merge rules"]
    level: int  # 1 = top level
    paragraphs: list[Paragraph] = []


class Table(StrictModel):
    id: str
    label: str | None = None  # "Table 2"
    caption: str
    rows: list[list[str]] = []  # first row is the header when GROBID found one
    page: int | None = None
    section_id: str | None = None  # the section the table appears in


class Figure(StrictModel):
    id: str
    label: str | None = None  # "Figure 3"
    caption: str
    page: int | None = None
    section_id: str | None = None


class Formula(StrictModel):
    """A display formula. Its text is also placed in the paragraph flow (D-019)."""

    id: str
    text: str
    label: str | None = None  # "(3)"


class Reference(StrictModel):
    """One bibliography entry. Stored for dedup and citation work, never indexed."""

    record_id: str
    ref_id: str  # "b12", the target of <ref type="bibr" target="#b12">
    title: str | None = None
    authors: list[str] = []
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    raw: str | None = None  # the reference string as printed


class StructuredPaper(StrictModel):
    record_id: str
    parser: Parser
    title: str
    abstract: list[Paragraph] = []
    sections: list[Section] = []
    tables: list[Table] = []
    figures: list[Figure] = []
    formulas: list[Formula] = []


class Chunk(StrictModel):
    """One retrieval unit and citation target."""

    chunk_id: str  # "<record_id>#0007", stable across reruns with the same settings
    record_id: str
    arxiv_id: str | None = None
    arxiv_version: int | None = None
    element_type: ElementType
    section_path: list[str] = []
    section_number: str | None = None
    page: int | None = None
    # "Title > Section > Subsection". Stored apart and prepended only at
    # embedding time, so the header ablation needs no re-chunking.
    header: str
    text: str
    token_count: int  # body only
    header_token_count: int
    cited_refs: list[str] = []
    parser: Parser


class IndexStatus(StrictModel):
    """Per accepted record: does it reach the index? Written back to the manifest (D-021)."""

    record_id: str
    included_in_index: bool
    n_chunks: int
    outcome: ParseOutcome
