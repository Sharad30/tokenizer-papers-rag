"""What the Parse page shows, computed from the parse outputs every time (D-038).

Live when data/parsed/ exists. Otherwise from the snapshot, which stores
this module's own results: the overview, sampled chunk pools per element
type and per quality check, sampled references, and a few whole papers.
The quality checks are heuristics that look for known parse defects, so
the page shows their counts and lets you read the flagged chunks.
"""

import random
import re
import statistics
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, TypeVar

from rag.common.config import Config
from rag.common.io import read_jsonl
from rag.common.schemas import StrictModel
from rag.parse.chunker import FORMULA_PLACEHOLDER, HEADER_SEPARATOR, SPECIAL_TOKENS
from rag.parse.pipeline import CHUNKS, RECORDS, REFERENCES, STATUS, safe_name, step_dir
from rag.parse.schemas import (
    Chunk,
    ElementType,
    IndexStatus,
    ParseOutcome,
    ParseStatus,
    Reference,
    StructuredPaper,
)
from rag.showcase.config import ParseShowcaseConfig
from rag.showcase.sources import Snapshot, read_snapshot, snapshot_file, stale_sources

STAGE = "parse"

# --- Quality checks ---------------------------------------------------------

# "Figure 2: Figure 2." — the caption already starts with the label the chunker adds.
_REPEATED_LABEL = re.compile(r"^(Figure|Fig\.|Table)\s*(\S+?)\s*[:.]\s*\1\s*\2\b", re.IGNORECASE)
# Pieces of a printed bibliography entry. Body text cites as "(Author, 2020)",
# so several of these in one paragraph chunk mean reference-list text leaked in.
_BIBLIOGRAPHY = re.compile(
    r"arXiv preprint arXiv:\d|In Proceedings of|Preprint, arXiv"
    r"|pages \d+[–-]\d+|\b(?:19|20)\d{2}\. [A-Z]"
)
_WORD = re.compile(r"[^\W\d_]{4,}")
# Short real headings are acronyms or names: "BPE", "B.1 NER", "GPT-4", "ByT5".
_ACRONYM_HEADING = re.compile(r"^[A-Z0-9][\w.+\-]*(?: [A-Z0-9][\w.+\-]*)*$")
_FIGURE_LABEL = re.compile(r"^(Fig\.?|Figure|Table)\s*\d+", re.IGNORECASE)


class QualityCheck(StrictModel):
    key: str
    label: str
    why: str  # what the defect does to retrieval or citations


CHECKS = [
    QualityCheck(
        key="repeated_label",
        label="Caption repeats its label",
        why='"Table 1: Table 1, …". GROBID copies the label into the caption; the structure '
        "step strips it (D-039), so what's left is mostly prose GROBID took for a caption.",
    ),
    QualityCheck(
        key="bibliography",
        label="Reference-list text in a paragraph",
        why="Bibliography entries leaked into a body chunk (usually the last section). They "
        "match queries on paper titles and venues, and references should never be indexed. "
        "The structure step cuts them (D-039); what's left is mostly author lists split "
        "across paragraphs.",
    ),
    QualityCheck(
        key="odd_heading",
        label="Suspicious section heading",
        why='A heading with no word of 4+ letters that isn\'t an acronym ("( a )", "OSR =") '
        'or that is a figure label ("Fig. 1"): figure text or a formula taken as a heading, '
        "so the chunk's section path and citation are wrong. Acronym headings such as "
        '"BPE" or "GPT-4" are not flagged.',
    ),
]


def check_flags(chunk: Chunk, bibliography_min_matches: int) -> list[str]:
    """The quality checks this chunk fails."""
    flags = []
    if chunk.element_type in (
        ElementType.FIGURE_CAPTION,
        ElementType.TABLE,
    ) and _REPEATED_LABEL.match(chunk.text):
        flags.append("repeated_label")
    if (
        chunk.element_type is ElementType.PARAGRAPH
        and len(_BIBLIOGRAPHY.findall(chunk.text)) >= bibliography_min_matches
    ):
        flags.append("bibliography")
    heading = chunk.section_path[-1] if chunk.section_path else ""
    if (
        chunk.element_type is ElementType.PARAGRAPH
        and heading
        and (
            (not _WORD.search(heading) and not _ACRONYM_HEADING.match(heading))
            or _FIGURE_LABEL.match(heading)
        )
    ):
        flags.append("odd_heading")
    return flags


# --- Overview ---------------------------------------------------------------


class TokenBin(StrictModel):
    start: int  # tokens the model sees: header + body + special tokens
    element: str
    chunks: int


class ParseOverview(StrictModel):
    accepted: int  # records sent to parsing
    outcomes: dict[str, int]  # grobid | fallback | failed | no_pdf
    indexed_papers: int
    chunks: int
    per_element: dict[str, int]
    median_tokens: dict[str, int]  # per element, body tokens
    chunks_per_paper_median: int
    chunks_per_paper_max: int
    token_bins: list[TokenBin]
    model_limit: int
    over_limit: int  # chunks the model would truncate: must be 0
    formula_placeholders: int
    references: int
    papers_with_references: int
    paragraphs_citing: int  # paragraph chunks with at least one resolved citation
    paragraphs: int
    checks: dict[str, int]  # quality check key -> flagged chunks


def model_tokens(chunk: Chunk) -> int:
    return chunk.header_token_count + chunk.token_count + SPECIAL_TOKENS


def overview(
    grobid_status: Sequence[ParseStatus],
    index_status: Sequence[IndexStatus],
    chunks: Sequence[Chunk],
    references: Sequence[Reference],
    model_limit: int,
    bin_width: int,
    bibliography_min_matches: int,
) -> ParseOverview:
    per_paper = Counter(c.record_id for c in chunks)
    by_element: dict[str, list[int]] = {}
    for c in chunks:
        by_element.setdefault(c.element_type.value, []).append(c.token_count)
    bins = Counter((model_tokens(c) // bin_width * bin_width, c.element_type.value) for c in chunks)
    paragraphs = [c for c in chunks if c.element_type is ElementType.PARAGRAPH]
    flags = Counter(f for c in chunks for f in check_flags(c, bibliography_min_matches))
    return ParseOverview(
        accepted=len(grobid_status),
        outcomes={o.value: sum(s.outcome is o for s in grobid_status) for o in ParseOutcome},
        indexed_papers=sum(s.included_in_index for s in index_status),
        chunks=len(chunks),
        per_element={e.value: len(by_element.get(e.value, [])) for e in ElementType},
        median_tokens={e: round(statistics.median(v)) for e, v in by_element.items()},
        chunks_per_paper_median=round(statistics.median(per_paper.values())) if per_paper else 0,
        chunks_per_paper_max=max(per_paper.values(), default=0),
        token_bins=[TokenBin(start=s, element=e, chunks=n) for (s, e), n in sorted(bins.items())],
        model_limit=model_limit,
        over_limit=sum(model_tokens(c) > model_limit for c in chunks),
        formula_placeholders=sum(c.text.count(FORMULA_PLACEHOLDER) for c in chunks),
        references=len(references),
        papers_with_references=len({r.record_id for r in references}),
        paragraphs_citing=sum(bool(c.cited_refs) for c in paragraphs),
        paragraphs=len(paragraphs),
        checks={check.key: flags[check.key] for check in CHECKS},
    )


# --- One paper ----------------------------------------------------------------


class SectionOutline(StrictModel):
    number: str | None
    title: str
    level: int
    paragraphs: int


class PaperView(StrictModel):
    record_id: str
    title: str
    parser: str
    sections: list[SectionOutline]
    tables: int
    figures: int
    formulas: int
    references: int
    chunks: list[Chunk]


def paper_view(paper: StructuredPaper, chunks: Sequence[Chunk], references: int) -> PaperView:
    return PaperView(
        record_id=paper.record_id,
        title=paper.title,
        parser=paper.parser.value,
        sections=[
            SectionOutline(
                number=s.number, title=s.title, level=s.level, paragraphs=len(s.paragraphs)
            )
            for s in paper.sections
        ],
        tables=len(paper.tables),
        figures=len(paper.figures),
        formulas=len(paper.formulas),
        references=references,
        chunks=list(chunks),
    )


def overlap(previous: Chunk, current: Chunk) -> str:
    """The text `current` repeats from the end of `previous` (D-017's overlap).

    Only within one section: across sections there is never any overlap.
    """
    if (
        previous.section_path != current.section_path
        or previous.element_type != current.element_type
    ):
        return ""
    text = previous.text
    for start in range(len(text)):
        # The overlap starts at a sentence start, so only try after whitespace.
        if (start == 0 or text[start - 1].isspace()) and current.text.startswith(text[start:]):
            return text[start:].strip()
    return ""


# --- Loading --------------------------------------------------------------------

Origin = Literal["live", "snapshot", "none"]


class ParseData(StrictModel):
    origin: Origin
    snapshot: Snapshot | None = None
    stale: list[str] = []
    overview: ParseOverview | None = None
    element_pools: dict[str, list[Chunk]] = {}  # every chunk, by element type (live)
    formula_pool: list[Chunk] = []  # chunks where a long formula became a placeholder
    check_pools: dict[str, list[Chunk]] = {}
    references: list[Reference] = []
    papers: dict[str, PaperView] = {}  # snapshot only; live papers are opened on demand
    paper_ids: list[str] = []  # papers that can be opened
    titles: dict[str, str] = {}  # record_id -> title, for picking a paper


def source_files(config: Config) -> dict[str, Path]:
    return {
        "grobid_status": step_dir(config, "grobid") / RECORDS,
        "chunks": step_dir(config, "chunk") / CHUNKS,
        "index_status": step_dir(config, "chunk") / STATUS,
        "references": step_dir(config, "structure") / REFERENCES,
    }


def load(config: Config) -> ParseData:
    files = source_files(config)
    if all(path.exists() for path in files.values()):
        return _live(config, files)
    snapshot = read_snapshot(snapshot_file(config.paths.snapshot, STAGE))
    if snapshot is None:
        return ParseData(origin="none")
    return ParseData.model_validate(
        {
            **snapshot.data,
            "origin": "snapshot",
            "snapshot": snapshot,
            "stale": stale_sources(snapshot, files),
        }
    )


def _live(config: Config, files: dict[str, Path]) -> ParseData:
    chunks = read_jsonl(files["chunks"], Chunk)
    references = read_jsonl(files["references"], Reference)
    show = config.showcase.parse
    view = overview(
        read_jsonl(files["grobid_status"], ParseStatus),
        read_jsonl(files["index_status"], IndexStatus),
        chunks,
        references,
        config.parse.chunk.embedding_max_tokens,
        show.token_bin_width,
        show.bibliography_min_matches,
    )
    return ParseData(
        origin="live",
        overview=view,
        element_pools={e.value: [c for c in chunks if c.element_type is e] for e in ElementType},
        formula_pool=[c for c in chunks if FORMULA_PLACEHOLDER in c.text],
        check_pools={
            check.key: [
                c for c in chunks if check.key in check_flags(c, show.bibliography_min_matches)
            ]
            for check in CHECKS
        },
        references=references,
        paper_ids=sorted({c.record_id for c in chunks}),
        titles={c.record_id: _title(c) for c in chunks},
    )


def _title(chunk: Chunk) -> str:
    # The header is "Title > Section > ...": its first part is the paper title.
    return chunk.header.split(HEADER_SEPARATOR, 1)[0]


def open_paper(config: Config, data: ParseData, record_id: str) -> PaperView | None:
    """One paper's structure and chunks: from the snapshot, or read live."""
    if record_id in data.papers:
        return data.papers[record_id]
    path = step_dir(config, "structure") / f"{safe_name(record_id)}.json"
    if data.origin != "live" or not path.exists():
        return None
    paper = StructuredPaper.model_validate_json(path.read_text(encoding="utf-8"))
    chunks = sorted(
        (c for pool in data.element_pools.values() for c in pool if c.record_id == record_id),
        key=lambda c: c.chunk_id,
    )
    references = sum(r.record_id == record_id for r in data.references)
    return paper_view(paper, chunks, references)


def snapshot_data(config: Config, data: ParseData, show: ParseShowcaseConfig, seed: int) -> dict:
    """The live view, down-sampled to what a fresh clone needs (a few hundred KB)."""
    papers = {}
    for record_id in sample(data.paper_ids, show.snapshot_papers, seed, str):
        paper = open_paper(config, data, record_id)
        if paper is not None:
            papers[record_id] = paper
    reduced = ParseData(
        origin="snapshot",
        overview=data.overview,
        element_pools={
            k: sample(v, show.pool_size, seed, chunk_key) for k, v in data.element_pools.items()
        },
        check_pools={
            k: sample(v, show.pool_size, seed, chunk_key) for k, v in data.check_pools.items()
        },
        formula_pool=sample(data.formula_pool, show.pool_size, seed, chunk_key),
        references=sample(data.references, show.pool_size, seed, reference_key),
        papers=papers,
        paper_ids=sorted(papers),
        titles={record_id: data.titles[record_id] for record_id in papers},
    )
    return reduced.model_dump(mode="json", exclude={"origin", "snapshot", "stale"})


T = TypeVar("T")


def chunk_key(chunk: Chunk) -> str:
    return chunk.chunk_id


def reference_key(reference: Reference) -> str:
    return f"{reference.record_id}#{reference.ref_id}"


def sample(items: Sequence[T], k: int, seed: int, key: Callable[[T], str]) -> list[T]:
    """A reproducible sample: sorted by `key` first, so input order doesn't matter."""
    ordered = sorted(items, key=key)
    return random.Random(seed).sample(ordered, min(k, len(ordered)))
