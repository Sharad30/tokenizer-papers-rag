"""Drafted questions to a verified eval set (D-045).

`eval/questions.draft.jsonl` holds questions an LLM drafted: each gold entry
names its paper (arXiv id, other id, title) and gives search terms and a
gist, but no span. Sharad turns each into gold by hand:

    rag eval check-papers   which gold papers are in the corpus
    rag eval resolve        pick a verbatim span per gold entry (interactive)
    rag eval absent         check that unanswerable questions really aren't answered
    rag eval finalize       write eval/questions.jsonl from entries marked verified

The tools show evidence and write only what Sharad picks. They never choose
a span, and never change a question, gist or label. Candidates come in
corpus order, never ranked (D-024): ranking would pull gold towards
whatever the ranking likes.
"""

import json
import re
import shutil
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

from rapidfuzz import fuzz, process

from rag.collect.schemas import ManifestRecord
from rag.common.io import write_text_atomic
from rag.common.schemas import StrictModel
from rag.dedup.resolve import normalise as normalise_title
from rag.eval.identity import identity_groups
from rag.eval.schemas import AbsentCheck, ExpectedBehavior, QuestionType, UnanswerableKind
from rag.eval.spans import ChunkTexts

ARXIV_VERSION = re.compile(r"v\d+$")


class DraftGold(StrictModel):
    title: str
    arxiv_id: str | None = None
    alt_id: str | None = None  # another record id: "s2:…", "acl:…"
    section_hint: str | None = None
    location_hint: str | None = None
    span_search: list[str] = []
    answer_gist: str | None = None
    span: str | None = None  # null until Sharad picks one
    section: str | None = None
    record_id: str | None = None  # written by `resolve` with the span


class DraftQuestion(StrictModel):
    id: str
    type: QuestionType
    conversation_id: str | None = None
    turn: int | None = None
    question: str
    oracle_rewrite: str | None = None
    answerable: bool
    expected_behavior: ExpectedBehavior | None = None
    gold: list[DraftGold] = []
    verify_absent: list[str] = []
    duplicate_sensitive: bool = False
    notes: str | None = None
    # Sharad sets "verified" by hand once a question is ready; `finalize`
    # takes only those. "rejected" keeps a dropped question in the file.
    status: Literal["draft", "verified", "rejected"] = "draft"
    provenance: str | None = None
    restatement_notes: str | None = None
    unanswerable_kind: UnanswerableKind | None = None  # picked in `resolve`
    absent_check: AbsentCheck | None = None  # written by `absent`


# Fields the tools add. Left out of the file while unset, so the draft keeps
# the shape Sharad wrote it in.
_ADDED = {"unanswerable_kind", "absent_check", "restatement_notes"}
_ADDED_GOLD = {"record_id"}


def read_draft(path: Path) -> list[DraftQuestion]:
    with path.open(encoding="utf-8") as handle:
        return [DraftQuestion.model_validate_json(line) for line in handle if line.strip()]


def write_draft(path: Path, questions: Sequence[DraftQuestion]) -> None:
    """Rewrite the draft atomically, keeping the previous version as <file>.bak."""
    if path.is_file():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    lines = []
    for q in questions:
        row = q.model_dump(mode="json")
        for key in _ADDED:
            if row[key] is None:
                del row[key]
        for gold in row["gold"]:
            for key in _ADDED_GOLD:
                if gold[key] is None:
                    del gold[key]
        lines.append(json.dumps(row, ensure_ascii=False))
    write_text_atomic(path, "".join(line + "\n" for line in lines))


# --- which records are the gold paper -------------------------------------------------


class PaperMatch(StrictModel):
    status: Literal["matched", "title_only", "not_indexed", "missing"]
    how: str  # "arxiv_id", "alt_id", "title 94", or "" when missing
    records: list[str] = []  # the paper's indexed records, canonical first
    unindexed: list[str] = []  # its records that have no chunks


class Corpus:
    """The manifest and chunks a draft is checked against, indexed once."""

    def __init__(
        self,
        records: Iterable[ManifestRecord],
        texts: ChunkTexts,
        canonical: set[str] | None = None,
    ) -> None:
        self.records = {r.record_id: r for r in records}
        self.texts = texts
        self.canonical = canonical or set()
        self.by_arxiv: dict[str, list[str]] = defaultdict(list)
        for r in self.records.values():
            if r.arxiv_id:
                self.by_arxiv[ARXIV_VERSION.sub("", r.arxiv_id)].append(r.record_id)
        groups = identity_groups(self.records.values())
        self.copies: dict[str, list[str]] = defaultdict(list)
        for record_id, group in groups.items():
            self.copies[group].append(record_id)
        self.group_of = groups
        self._titles = {rid: normalise_title(r.title or "") for rid, r in self.records.items()}

    def paper(self, record_ids: Iterable[str]) -> list[str]:
        """These records and every record identifiers link to them."""
        out: dict[str, None] = {}
        for rid in record_ids:
            for copy in self.copies.get(self.group_of.get(rid, rid), [rid]):
                out[copy] = None
        return list(out)

    def match(self, gold: DraftGold, title_min_ratio: float) -> PaperMatch:
        how, found = "", []
        if gold.arxiv_id:
            found = self.by_arxiv.get(ARXIV_VERSION.sub("", gold.arxiv_id), [])
            how = "arxiv_id"
        if not found and gold.alt_id and gold.alt_id in self.records:
            found, how = [gold.alt_id], "alt_id"
        title_only = False
        if not found:
            best = process.extractOne(
                normalise_title(gold.title), self._titles, scorer=fuzz.token_sort_ratio
            )
            if best and best[1] >= title_min_ratio:
                found, how, title_only = [best[2]], f"title {best[1]:.0f}", True
        if not found:
            return PaperMatch(status="missing", how="")
        paper = self.paper(found)
        indexed = [r for r in paper if self.texts.paper(r)]
        # Canonical first (the record retrieval shows), then the order found.
        indexed.sort(key=lambda r: r not in self.canonical)
        unindexed = [r for r in paper if r not in indexed]
        if not indexed:
            status = "not_indexed"
        else:
            status = "title_only" if title_only else "matched"
        return PaperMatch(status=status, how=how, records=indexed, unindexed=unindexed)


def check_papers(
    questions: Sequence[DraftQuestion], corpus: Corpus, title_min_ratio: float
) -> list[tuple[DraftQuestion, int, PaperMatch]]:
    """(question, gold index, match) for every gold entry."""
    return [
        (q, i, corpus.match(g, title_min_ratio)) for q in questions for i, g in enumerate(q.gold)
    ]


def render_check(rows: list[tuple[DraftQuestion, int, PaperMatch]], corpus: Corpus) -> str:
    counts: dict[str, int] = defaultdict(int)
    problems: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for q, i, m in rows:
        counts[m.status] += 1
        if m.status != "matched":
            g = q.gold[i]
            key = f"{g.title}  [{g.arxiv_id or g.alt_id or 'no id'}]"
            detail = m.how + (f" -> {', '.join(m.records or m.unindexed)}" if m.how else "")
            problems[m.status][f"{key}  {detail}".rstrip()].append(q.id)
    lines = [
        f"{len(rows)} gold entries: "
        + ", ".join(f"{counts[s]} {s}" for s in ("matched", "title_only", "not_indexed", "missing"))
    ]
    notes = {
        "missing": "Not in the corpus under any id or a similar title: add the paper "
        "(re-collect, re-parse, re-index) or relabel the questions.",
        "not_indexed": "In the manifest but with no chunks (rejected, or not parsed).",
        "title_only": "Matched by title alone: check it is the same paper.",
    }
    for status in ("missing", "not_indexed", "title_only"):
        if problems[status]:
            lines += ["", f"{status.upper()} ({notes[status]})"]
            for paper, ids in problems[status].items():
                lines.append(f"  {paper}")
                lines.append(f"    questions: {', '.join(sorted(set(ids)))}")
    return "\n".join(lines)
