from pathlib import Path

import pymupdf

from rag.parse.fallback import parse_pdf
from rag.parse.schemas import Parser


def make_pdf(path: Path, pages: list[list[str]]) -> Path:
    """A synthetic PDF: each inner list is one page's text blocks, top to bottom."""
    document = pymupdf.open()
    for blocks in pages:
        page = document.new_page()
        for i, text in enumerate(blocks):
            page.insert_text((72, 72 + 60 * i), text)
    document.save(path)
    document.close()
    return path


def test_one_section_per_page_with_sentences(tmp_path: Path) -> None:
    pdf = make_pdf(
        tmp_path / "p.pdf",
        [["Subword units help. They are learned."], ["Second page text."]],
    )
    paper = parse_pdf(pdf, "acl:X", "Title", "An abstract.")
    assert paper.parser is Parser.PYMUPDF
    assert [s.title for s in paper.sections] == ["Page 1", "Page 2"]
    assert [s.text for s in paper.sections[0].paragraphs[0].sentences] == [
        "Subword units help.",
        "They are learned.",
    ]
    assert paper.sections[1].paragraphs[0].page == 2
    assert paper.abstract[0].sentences[0].text == "An abstract."
    assert (paper.tables, paper.figures) == ([], [])


def test_text_after_references_heading_is_dropped(tmp_path: Path) -> None:
    pdf = make_pdf(
        tmp_path / "p.pdf",
        [["Body text."], ["More body.", "References", "Doe, J. 2016. A paper."]],
    )
    paper = parse_pdf(pdf, "acl:X", "Title", None)
    text = " ".join(s.text for sec in paper.sections for p in sec.paragraphs for s in p.sentences)
    assert "More body." in text
    assert "Doe" not in text
    assert paper.abstract == []
