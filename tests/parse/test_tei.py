from collections import Counter
from pathlib import Path

import pytest

from rag.parse.config import CleanConfig
from rag.parse.schemas import Parser
from rag.parse.tei import parse_references, parse_tei

FIXTURE = Path(__file__).parent.parent / "fixtures" / "tei" / "paper.tei.xml"
CLEAN = CleanConfig(min_entry_sentences=2, min_entry_share=0.3, max_fragment_words=15)


def body_tei(body: str) -> bytes:
    """A minimal TEI document around a <body> fragment."""
    return (
        '<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader/>'
        f"<text><body>{body}</body></text></TEI>"
    ).encode()


def figure(head: str, desc: str, table: bool = False) -> str:
    kind = ' type="table"' if table else ""
    return f"<figure{kind}><head>{head}</head><figDesc>{desc}</figDesc></figure>"


@pytest.fixture
def tei() -> bytes:
    return FIXTURE.read_bytes()


@pytest.fixture
def paper(tei):
    return parse_tei(tei, "arxiv:0000.00001v1", "Manifest Title", CLEAN)


def test_title_comes_from_the_manifest(paper) -> None:
    assert paper.title == "Manifest Title"
    assert paper.parser is Parser.GROBID


def test_abstract_sentences_and_page(paper) -> None:
    [para] = paper.abstract
    assert [s.text for s in para.sentences] == ["We study merge rules.", "They matter."]
    assert para.page == 1  # from the first sentence's coords


def test_sections_numbers_levels_and_paths(paper) -> None:
    got = [(s.number, s.title, s.level, s.path) for s in paper.sections]
    assert got == [
        ("1", "Introduction", 1, ["Introduction"]),
        ("2", "Method", 1, ["Method"]),  # "2." normalised
        ("2.1", "Merge rules", 2, ["Method", "Merge rules"]),
        (None, "Discussion", 1, ["Discussion"]),  # unnumbered: top level
        ("A", "Extra results", 1, ["Extra results"]),  # appendix kept
    ]
    assert "Acknowledgements" not in {s.title for s in paper.sections}


def test_headless_div_continues_previous_section(paper) -> None:
    method = paper.sections[1]
    texts = [p.sentences[0].text for p in method.paragraphs]
    assert texts == [
        "We define a merge.",
        "f(x) = x + 1",
        "Continued after a figure.",
        "Text after a mislabelled formula fragment.",  # fragment "heading" ignored
    ]


def test_formula_kept_in_reading_order_without_label(paper) -> None:
    [formula] = paper.formulas
    assert (formula.id, formula.text, formula.label) == ("formula_0", "f(x) = x + 1", "(1)")
    assert paper.sections[1].paragraphs[1].formula_id == "formula_0"
    assert paper.sections[1].paragraphs[1].page == 2


def test_citations_resolve_to_bibliography_ids(paper) -> None:
    first, second = paper.sections[0].paragraphs[0].sentences
    assert first.cited_refs == ["b0"]
    assert second.cited_refs == ["b1", "b0"]  # figure refs ignored, order kept
    assert first.text == "Subword units help (Doe et al., 2016)."


def test_paragraph_without_sentence_tags_and_list_items(paper) -> None:
    assert paper.sections[3].paragraphs[0].sentences[0].text == "No sentence tags here."
    assert paper.sections[2].paragraphs[1].sentences[0].text == "First item."


def test_tables_and_figures(paper) -> None:
    [table] = paper.tables
    assert (table.id, table.label, table.caption, table.page) == (
        "tab_0",
        "Table 1",
        "Vocabulary sizes.",
        4,
    )
    assert table.rows == [["Model", "Vocab"], ["A", "32k"]]  # empty row dropped
    [figure] = paper.figures
    assert (figure.label, figure.caption, figure.page) == ("Figure 1", "Merges over time.", 3)


def test_references(tei) -> None:
    b0, b1 = parse_references(tei, "arxiv:0000.00001v1")
    assert (b0.ref_id, b0.title, b0.year, b0.venue, b0.arxiv_id) == (
        "b0",
        "Neural units for rare words",
        2016,
        "Proceedings of SynthConf",
        "1508.00001",
    )
    assert b0.authors == ["Jane Doe", "Ann B Roe"]
    assert b0.raw.startswith("J. Doe and A. B. Roe.")
    # A book: own title and authors from <monogr>, no venue.
    assert (b1.title, b1.authors, b1.venue, b1.year, b1.doi) == (
        "A Book on Tokens",
        ["Poe"],
        None,
        None,
        "10.0000/synthetic.1",
    )


def test_references_never_become_sections(paper) -> None:
    all_text = " ".join(
        s.text for sec in paper.sections for p in sec.paragraphs for s in p.sentences
    )
    assert "Neural units for rare words" not in all_text


def test_empty_body() -> None:
    tei = b'<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader/><text><body/></text></TEI>'
    paper = parse_tei(tei, "x", "T", CLEAN)
    assert (paper.sections, paper.abstract, paper.tables) == ([], [], [])


def test_label_grobid_copied_into_the_caption_is_stripped() -> None:
    counts: Counter[str] = Counter()
    body = figure("Figure 2", "Figure2: BLEU by vocabulary size.") + figure(
        "Fig. 3", "Fig. 3. Merge counts."
    )
    paper = parse_tei(body_tei(body), "x", "T", CLEAN, counts)
    got = [(f.label, f.caption) for f in paper.figures]
    assert got == [("Figure 2", "BLEU by vocabulary size."), ("Figure 3", "Merge counts.")]
    assert counts == {"captions_label_stripped": 2, "caption_label_replaced": 1}


def test_the_captions_own_label_replaces_a_junk_head() -> None:
    paper = parse_tei(body_tei(figure("'", "Figure1: Shorter sequences.")), "x", "T", CLEAN)
    assert (paper.figures[0].label, paper.figures[0].caption) == ("Figure 1", "Shorter sequences.")


def test_caption_prose_about_a_table_is_not_a_label() -> None:
    body = figure("Table 2", "Table 2 compares F1 scores.", table=True) + figure(
        "", "Fig. 10 and App. F.1 show examples."
    )
    paper = parse_tei(body_tei(body), "x", "T", CLEAN)
    assert (paper.tables[0].label, paper.tables[0].caption) == (
        "Table 2",
        "Table 2 compares F1 scores.",
    )
    assert (paper.figures[0].label, paper.figures[0].caption) == (
        None,
        "Fig. 10 and App. F.1 show examples.",
    )


def test_a_reference_list_in_the_body_is_skipped() -> None:
    counts: Counter[str] = Counter()
    body = (
        '<div><head n="5">Conclusion</head><p>We are done.</p></div>'
        "<div><head>References</head><p>Doe, J. 2016. Rare words.</p></div>"
    )
    paper = parse_tei(body_tei(body), "x", "T", CLEAN, counts)
    assert [s.title for s in paper.sections] == ["Conclusion"]
    assert counts["reference_divs_skipped"] == 1
