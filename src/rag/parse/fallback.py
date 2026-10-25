"""PyMuPDF fallback: PDF -> a flat StructuredPaper when GROBID gives nothing (D-015).

Lower quality by design, and marked parser=pymupdf: no heading structure
(one section per page), no tables, figures or references, and text blocks in
the PDF's own order, which two-column layouts can scramble. Text after a
"References" heading is dropped so the bibliography never reaches the index.
"""

import re
from pathlib import Path

import pymupdf

from rag.parse.schemas import Paragraph, Parser, Section, Sentence, StructuredPaper

TEXT_BLOCK = 0  # block_type in get_text("blocks"): 0 text, 1 image, 3 vector
REFERENCES_HEADING = re.compile(r"^\s*(\d+\.?\s*)?(references|bibliography)\s*$", re.IGNORECASE)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
WHITESPACE = re.compile(r"\s+")


def parse_pdf(pdf: Path, record_id: str, title: str, abstract: str | None) -> StructuredPaper:
    """The manifest's abstract stands in for one, since there are no zones to find it."""
    sections: list[Section] = []
    with pymupdf.open(pdf) as document:
        for page_number, page in enumerate(document, start=1):
            paragraphs = []
            for block in page.get_text("blocks", flags=pymupdf.TEXT_DEHYPHENATE):
                text, block_type = block[4], block[6]
                if block_type != TEXT_BLOCK:
                    continue
                if REFERENCES_HEADING.match(text):
                    return _paper(record_id, title, abstract, sections, paragraphs, page_number)
                paragraph = _paragraph(text, page_number)
                if paragraph.sentences:
                    paragraphs.append(paragraph)
            _add_page(sections, paragraphs, page_number)
    return _paper(record_id, title, abstract, sections, [], 0)


def _paper(
    record_id: str,
    title: str,
    abstract: str | None,
    sections: list[Section],
    pending: list[Paragraph],
    page_number: int,
) -> StructuredPaper:
    _add_page(sections, pending, page_number)
    return StructuredPaper(
        record_id=record_id,
        parser=Parser.PYMUPDF,
        title=title,
        abstract=[_paragraph(abstract, None)] if abstract else [],
        sections=sections,
    )


def _add_page(sections: list[Section], paragraphs: list[Paragraph], page_number: int) -> None:
    if paragraphs:
        title = f"Page {page_number}"
        sections.append(
            Section(
                id=f"s{len(sections):03d}",
                title=title,
                path=[title],
                level=1,
                paragraphs=paragraphs,
            )
        )


def _paragraph(text: str, page: int | None) -> Paragraph:
    clean = WHITESPACE.sub(" ", text).strip()
    return Paragraph(
        sentences=[Sentence(text=s) for s in SENTENCE_END.split(clean) if s],
        page=page,
    )
