"""GROBID TEI XML -> StructuredPaper.

The parts of GROBID's TEI used here (namespace http://www.tei-c.org/ns/1.0):

    teiHeader//profileDesc/abstract//p      the abstract
    text/body/div                           one per heading: <head n="3.1">, then <p> and <formula>
    text/body/figure[@type="table"]         tables: <head>, <figDesc>, <table><row><cell>
    text/body/figure                        figures: <head>, <figDesc>
    text/back/div[@type="annex"]            appendices, treated like body sections
    text/back//listBibl/biblStruct          the bibliography, xml:id="b0", "b1", ...
    <s>                                     one sentence (segmentSentences=1)
    <ref type="bibr" target="#b0">          an in-text citation of bibliography entry b0
    coords="page,x,y,w,h;..."               position, when requested with teiCoordinates

The title comes from the manifest, not from GROBID: the manifest's title is
authoritative and GROBID's header extraction can be wrong.
"""

import re
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterator

from rag.parse.cleanup import cut_bibliography, is_reference_heading, non_heading_kind
from rag.parse.config import CleanConfig
from rag.parse.schemas import (
    Figure,
    Formula,
    Paragraph,
    Parser,
    Reference,
    Section,
    Sentence,
    StructuredPaper,
    Table,
)

TEI = "http://www.tei-c.org/ns/1.0"
NS = {"tei": TEI}
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
# Back-matter divs that hold no paper content worth indexing.
SKIPPED_BACK_DIVS = {"references", "acknowledgement", "availability", "funding"}
ARXIV_PREFIX = re.compile(r"^arxiv:\s*", re.IGNORECASE)
LABEL_TAIL = re.compile(r"\s*[:.]\s*$")  # "Table 1 :" -> "Table 1"
# GROBID copies the printed label into most figure captions ("Figure2: BLEU by
# vocabulary size."; journals print "Figure 2 General architecture"). A ":" or
# "." or a capital letter must follow the number, so a caption that is prose
# about a table ("Table 2 compares ...") keeps its text.
CAPTION_LABEL = re.compile(
    r"^(Figure|Fig\.?|Table)\s*([A-Z]?\d+[a-z]?)(?:\s*[:.]\s*|\s+(?=(?-i:[A-Z])))", re.IGNORECASE
)
WHITESPACE = re.compile(r"\s+")
LETTER = re.compile(r"[^\W\d_]")
# CRF GROBID sometimes labels a formula fragment as a heading ("←h m )." in
# the BPE paper). Fewer letters than this: not a heading.
MIN_HEADING_LETTERS = 3


def parse_tei(
    tei: bytes,
    record_id: str,
    title: str,
    clean: CleanConfig,
    counts: Counter[str] | None = None,
) -> StructuredPaper:
    """`counts`, if given, gets what the clean-up rules changed (for stats.json)."""
    counts = counts if counts is not None else Counter()
    root = ET.fromstring(tei)
    text = root.find("tei:text", NS)
    body = text.find("tei:body", NS) if text is not None else None
    back = text.find("tei:back", NS) if text is not None else None
    abstract = root.find(".//tei:profileDesc/tei:abstract", NS)

    formulas: list[Formula] = []
    divs = list(body.findall("tei:div", NS)) if body is not None else []
    if back is not None:
        for wrapper in back.findall("tei:div", NS):
            if wrapper.get("type") not in SKIPPED_BACK_DIVS:
                # Back matter wraps its sections: <div type="annex"><div><head>...
                divs += wrapper.findall("tei:div", NS) or [wrapper]
    sections = _sections(divs, formulas, clean, counts)
    figures_parent = body if body is not None else ET.Element("none")
    return StructuredPaper(
        record_id=record_id,
        parser=Parser.GROBID,
        title=title,
        abstract=[_paragraph(p) for p in abstract.iter(f"{{{TEI}}}p")]
        if abstract is not None
        else [],
        sections=sections,
        tables=[_table(f, counts) for f in figures_parent.iter(f"{{{TEI}}}figure") if _is_table(f)],
        figures=[
            _figure(f, counts) for f in figures_parent.iter(f"{{{TEI}}}figure") if not _is_table(f)
        ],
        formulas=formulas,
    )


def parse_references(tei: bytes, record_id: str) -> list[Reference]:
    root = ET.fromstring(tei)
    return [_reference(b, record_id) for b in root.iterfind(".//tei:listBibl/tei:biblStruct", NS)]


# --- sections ----------------------------------------------------------------


def _sections(
    divs: list[ET.Element], formulas: list[Formula], clean: CleanConfig, counts: Counter[str]
) -> list[Section]:
    sections: list[Section] = []
    # Open headings by level, for building paths: {1: ("3", "Method"), 2: ("3.1", ...)}.
    open_heads: dict[int, tuple[str | None, str]] = {}
    for div in divs:
        head = div.find("tei:head", NS)
        if head is not None and is_reference_heading(_text(head)):
            # A reference list GROBID kept in the body: never indexed (D-021).
            counts["reference_divs_skipped"] += 1
            continue
        paragraphs = cut_bibliography(list(_div_paragraphs(div, formulas)), clean, counts)
        not_heading = non_heading_kind(_clean(_text(head))) if head is not None else None
        if not_heading:
            counts[f"non_heading_folded.{not_heading}"] += 1
        if head is None or len(LETTER.findall(_text(head))) < MIN_HEADING_LETTERS or not_heading:
            # GROBID starts a new headless div after a figure or page break,
            # and mislabels some fragments as headings (formula pieces, figure
            # labels, running heads; D-039); all continue the previous section.
            if sections:
                sections[-1].paragraphs.extend(paragraphs)
                continue
            title, number = "", None
        else:
            title, number = _clean(_text(head)), _number(head.get("n"))
        level = number.count(".") + 1 if number else 1
        open_heads = {k: v for k, v in open_heads.items() if k < level}
        open_heads[level] = (number, title)
        sections.append(
            Section(
                id=f"s{len(sections):03d}",
                number=number,
                title=title,
                path=[t for _, (_, t) in sorted(open_heads.items()) if t],
                level=level,
                paragraphs=paragraphs,
            )
        )
    return sections


def _number(n: str | None) -> str | None:
    """head@n as printed: "3.1." -> "3.1"; "" -> None."""
    if not n:
        return None
    return n.strip().rstrip(".") or None


def _div_paragraphs(div: ET.Element, formulas: list[Formula]) -> Iterator[Paragraph]:
    for child in div:
        tag = _local(child.tag)
        if tag == "p":
            yield _paragraph(child)
        elif tag == "formula":
            formula = _formula(child, len(formulas))
            formulas.append(formula)
            yield Paragraph(
                sentences=[Sentence(text=formula.text)],
                page=_page(child),
                formula_id=formula.id,
            )
        elif tag == "list":
            yield from (_paragraph(item) for item in child.iterfind("tei:item", NS))


def _paragraph(p: ET.Element) -> Paragraph:
    sentences = p.findall("tei:s", NS)
    # Without segmentSentences, a paragraph is one "sentence".
    units = sentences or [p]
    return Paragraph(
        sentences=[s for s in (_sentence(u) for u in units) if s.text],
        page=_page(p) or (_page(sentences[0]) if sentences else None),
    )


def _sentence(element: ET.Element) -> Sentence:
    refs = [
        ref.get("target", "").lstrip("#")
        for ref in element.iter(f"{{{TEI}}}ref")
        if ref.get("type") == "bibr" and ref.get("target")
    ]
    return Sentence(text=_clean(_text(element)), cited_refs=list(dict.fromkeys(refs)))


def _formula(element: ET.Element, index: int) -> Formula:
    label = element.find("tei:label", NS)
    label_text = _clean(_text(label)) if label is not None else None
    # The formula's own text, without its "(3)" label.
    parts = [element.text or ""] + [
        (_text(child) if _local(child.tag) != "label" else "") + (child.tail or "")
        for child in element
    ]
    return Formula(
        id=element.get(XML_ID) or f"formula_{index}",
        text=_clean("".join(parts)),
        label=label_text or None,
    )


# --- tables and figures -----------------------------------------------------


def _is_table(figure: ET.Element) -> bool:
    return figure.get("type") == "table"


def _caption_parts(figure: ET.Element, counts: Counter[str]) -> tuple[str | None, str]:
    head = figure.find("tei:head", NS)
    desc = figure.find("tei:figDesc", NS)
    label = LABEL_TAIL.sub("", _clean(_text(head))) if head is not None else ""
    caption = _clean(_text(desc)) if desc is not None else ""
    printed = CAPTION_LABEL.match(caption)
    if printed:
        # The chunker adds the label back once. The caption's own label wins
        # over <head>, which is sometimes junk ("'", "Figure 3 :Figure 4").
        caption = caption[printed.end() :]
        kind = "Table" if printed.group(1).lower() == "table" else "Figure"
        printed_label = f"{kind} {printed.group(2)}"
        counts["captions_label_stripped"] += 1
        if label and label != printed_label:
            counts["caption_label_replaced"] += 1
        label = printed_label
    return (label or None), caption


def _table(figure: ET.Element, counts: Counter[str]) -> Table:
    label, caption = _caption_parts(figure, counts)
    rows = [
        [_clean(_text(cell)) for cell in row.iterfind("tei:cell", NS)]
        for row in figure.iterfind("tei:table/tei:row", NS)
    ]
    return Table(
        id=figure.get(XML_ID) or "",
        label=label,
        caption=caption,
        rows=[r for r in rows if any(r)],
        page=_page(figure),
    )


def _figure(figure: ET.Element, counts: Counter[str]) -> Figure:
    label, caption = _caption_parts(figure, counts)
    return Figure(id=figure.get(XML_ID) or "", label=label, caption=caption, page=_page(figure))


# --- references -------------------------------------------------------------


def _reference(bibl: ET.Element, record_id: str) -> Reference:
    analytic = bibl.find("tei:analytic", NS)
    monogr = bibl.find("tei:monogr", NS)
    # An article has <analytic> (its own title, authors) inside <monogr> (the venue);
    # a book or report has only <monogr>.
    own = analytic if analytic is not None else monogr
    title = own.find("tei:title", NS) if own is not None else None
    venue = monogr.find("tei:title", NS) if (monogr is not None and analytic is not None) else None
    date = bibl.find(".//tei:imprint/tei:date[@type='published']", NS)
    when = date.get("when", "") if date is not None else ""
    doi = bibl.find(".//tei:idno[@type='DOI']", NS)
    arxiv = bibl.find(".//tei:idno[@type='arXiv']", NS)
    raw = bibl.find("tei:note[@type='raw_reference']", NS)
    return Reference(
        record_id=record_id,
        ref_id=bibl.get(XML_ID) or "",
        title=_clean(_text(title)) or None if title is not None else None,
        authors=[_person(p) for p in own.iterfind("tei:author/tei:persName", NS)]
        if own is not None
        else [],
        year=int(when[:4]) if when[:4].isdigit() else None,
        venue=_clean(_text(venue)) or None if venue is not None else None,
        doi=_clean(_text(doi)) or None if doi is not None else None,
        arxiv_id=ARXIV_PREFIX.sub("", _clean(_text(arxiv))) or None if arxiv is not None else None,
        raw=_clean(_text(raw)) or None if raw is not None else None,
    )


def _person(pers: ET.Element) -> str:
    names = [_text(f) for f in pers.iterfind("tei:forename", NS)]
    names += [_text(s) for s in pers.iterfind("tei:surname", NS)]
    return _clean(" ".join(names))


# --- small helpers ------------------------------------------------------------


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _text(element: ET.Element) -> str:
    return "".join(element.itertext())


def _clean(text: str) -> str:
    return WHITESPACE.sub(" ", text).strip()


def _page(element: ET.Element) -> int | None:
    """First page from coords="page,x,y,w,h;..."."""
    coords = element.get("coords")
    if not coords:
        return None
    first = coords.split(";", 1)[0].split(",", 1)[0]
    return int(first) if first.isdigit() else None
