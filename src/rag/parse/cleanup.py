"""Clean-up rules for text GROBID puts in the wrong place (D-039).

Bibliography text in the body: CRF GROBID leaves some reference-list lines
in the last body section (often Limitations or Ethics, which sit next to the
references on the page), split into sentences like

    "Name, Name, and Name." | "2023." | "Title." | "Preprint, arXiv:2306.16842."

A paragraph's tail is cut from the first such entry piece when enough of the
tail is entry pieces. Prose citing a paper ("Phan et al. (2024).") is never an
entry piece, and one stray piece never cuts.

Non-headings: GROBID also takes some figure labels, equations, running heads
and reference titles for section headings. Their div continues the previous
section. The rules are narrow on purpose: short real headings ("BPE", "NER",
"GPT-4") stay headings.
"""

import re
from collections import Counter

from rag.parse.config import CleanConfig
from rag.parse.schemas import Paragraph

# Pieces of a printed bibliography entry. Anchored where the piece starts or
# ends a sentence, so a sentence mentioning "In Proceedings of" mid-way is prose.
VENUE = re.compile(
    r"arXiv preprint arXiv:\d|Preprint, arXiv:|CoRR, abs/|pages \d+[–-]\d+[,.]"
    r"|^In (?:Proceedings of|Findings of|Advances in Neural)"
    r"|Association for Computational Linguistics\.$|^URL https?://"
)
YEAR_ONLY = re.compile(r"^\(?(?:19|20)\d{2}[a-z]?\)?\.$")  # "2023." on its own
# ACL style: "Name Name, Name Name and Name Name. 2019. Title" (or ending at the year)
ENTRY_START = re.compile(
    r"^[A-Z][\w'’.\- ]+(?:, [A-Z][\w'’.\- ]+)*,? (?:and [A-Z][\w'’.\- ]+|et al)?\. "
    r"(?:19|20)\d{2}[a-z]?\.(?: [A-Z]|$)"
)
# Only names: "Vilém Zouhar, Clara Meister, and Ryan Cotterell." GROBID splits
# an entry's author list from its year. Not a piece on its own (prose can look
# like this); it only joins a cut that starts right after it.
_NAME = r"[A-Z][\w'’.\-]*(?: [A-Z][\w'’.\-]*){0,3}"
AUTHOR_LIST = re.compile(rf"^{_NAME}(?:(?:,| and|, and) {_NAME})*(?:,? et al)?\.$")
# Body sections that are the reference list itself.
REFERENCE_HEADINGS = {"references", "bibliography", "reference"}

RELATION = re.compile(r"[=∝∈≤≥⩽⩾]")
WORD = re.compile(r"[^\W\d_]{4,}")
EQUATION_MAX_WORDS = 2  # fewer words of 4+ letters than this around a relation: an equation
SECTION_NUMBER = re.compile(r"^[A-Z]?\d*(?:\.\d+)+\.?\s")  # "F.2 ", "3.1 ", "A.4. "

# Heading text that isn't a heading, by kind (for stats.json).
NON_HEADINGS = {
    "figure_label": re.compile(r"^(?:Fig\.?|Figure|Table|Tab\.)\s*[A-Z]?\d+\b", re.IGNORECASE),
    "letter_spaced": re.compile(r"^(?:\S ){4,}\S$"),  # "A R T I C L E I N P R E S S"
    "reference_title": re.compile(
        r"\bIn (?:Proceedings|Findings) of|Conference on|Association for Computational"
        r"|arXiv:\d"
    ),
}


def is_entry_piece(sentence: str) -> bool:
    return bool(VENUE.search(sentence) or YEAR_ONLY.match(sentence) or ENTRY_START.match(sentence))


def is_reference_heading(heading: str) -> bool:
    return heading.strip(" .:").lower() in REFERENCE_HEADINGS


def non_heading_kind(heading: str) -> str | None:
    """Why this heading text isn't a real heading, or None if it is one."""
    for kind, pattern in NON_HEADINGS.items():
        if pattern.search(heading):
            return kind
    if _is_equation(heading):
        return "equation"
    return None


def _is_equation(heading: str) -> bool:
    """A relation with almost no words around it ("AE(w) = W E(w) SE(w)", "OSR =").

    Relations, not arrows ("FR+ES→EN transformer" is a heading). Headings that
    name a bound or setting ("Lower Bound (L = 1)", "F.2 |V| = 60000") have words
    or a section number, so they stay.
    """
    return (
        bool(RELATION.search(heading))
        and len(WORD.findall(heading)) < EQUATION_MAX_WORDS
        and not SECTION_NUMBER.match(heading)
    )


def cut_bibliography(
    paragraphs: list[Paragraph], config: CleanConfig, counts: Counter[str]
) -> list[Paragraph]:
    """The paragraphs with any bibliography tail cut off; emptied paragraphs dropped."""
    kept = []
    for paragraph in paragraphs:
        if paragraph.formula_id is not None:
            kept.append(paragraph)
            continue
        cut = _cut_at([s.text for s in paragraph.sentences], config)
        if cut is None:
            kept.append(paragraph)
            continue
        counts["bibliography_sentences_dropped"] += len(paragraph.sentences) - cut
        if cut == 0:
            counts["bibliography_paragraphs_dropped"] += 1
            continue
        kept.append(paragraph.model_copy(update={"sentences": paragraph.sentences[:cut]}))
    return kept


def _cut_at(sentences: list[str], config: CleanConfig) -> int | None:
    """Index of the first sentence to drop, or None to keep the paragraph whole."""
    flags = [is_entry_piece(s) for s in sentences]
    for start, flagged in enumerate(flags):
        if not flagged:
            continue
        tail = flags[start:]
        if sum(tail) >= config.min_entry_sentences and sum(tail) / len(tail) >= (
            config.min_entry_share
        ):
            return _with_leading_fragment(sentences, start, config)
    return None


def _with_leading_fragment(sentences: list[str], start: int, config: CleanConfig) -> int:
    """Move the cut back over the entry's author list (and title), and over a short first
    sentence: the end of an entry cut by a column or page break ("with subword
    units." or a title, before "In Proceedings of ...")."""
    # The author list sits right before the cut, or before the entry's title.
    for back in (1, 2):
        if start - back >= 0 and AUTHOR_LIST.match(sentences[start - back]):
            start -= back
            break
    starts_an_entry = bool(
        AUTHOR_LIST.match(sentences[start]) or ENTRY_START.match(sentences[start])
    )
    # Mid-entry (a venue or year line first): the sentence before it is the rest
    # of that entry, not prose.
    if (
        start == 1
        and not starts_an_entry
        and len(sentences[0].split()) <= config.max_fragment_words
    ):
        return 0
    return start
