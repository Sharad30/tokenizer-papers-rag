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
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Literal

from rapidfuzz import fuzz, process

from rag.collect.schemas import ManifestRecord
from rag.common.io import write_text_atomic
from rag.common.schemas import StrictModel
from rag.dedup.resolve import normalise as normalise_title
from rag.eval.config import DraftConfig
from rag.eval.identity import identity_groups
from rag.eval.schemas import (
    AbsentCheck,
    ExpectedBehavior,
    GoldSpan,
    Origin,
    Question,
    QuestionType,
    UnanswerableKind,
)
from rag.eval.spans import ChunkTexts, Suggestion, map_span, normalise, suggest
from rag.parse.schemas import Chunk

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


# --- resolve: candidate passages and Sharad's pick -----------------------------------

# A sentence ends at . ! ? followed by space and an upper-case letter, digit or
# bracket. Good enough to offer sentences to pick from; Sharad can always type
# the span himself.
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[\"'])")


class Candidate(StrictModel):
    chunk_id: str
    record_id: str
    element: str
    section: str
    terms: list[str]  # the span_search terms it contains
    excerpt: str
    copies: list[str] = []  # other chunk ids with the same text (linked copies)


def section_of(chunk: Chunk) -> str:
    return " > ".join(chunk.section_path)


def candidates(
    terms: Sequence[str], records: Sequence[str], texts: ChunkTexts, excerpt_chars: int
) -> list[Candidate]:
    """Chunks of these records containing any term: record by record, in corpus order.

    Never ranked by how many terms match (D-024). Identical text in a linked
    copy is listed once, with the copy's chunk id noted.
    """
    needles = [(t, normalise(t)) for t in terms if normalise(t)]
    out: list[Candidate] = []
    seen: dict[str, Candidate] = {}
    for record_id in records:
        for ic in texts.paper(record_id):
            found = [t for t, n in needles if n in ic.norm]
            if not found:
                continue
            if ic.norm in seen:
                seen[ic.norm].copies.append(ic.chunk.chunk_id)
                continue
            first = normalise(found[0])
            at = ic.norm.find(first)
            lo, hi = max(0, at - excerpt_chars), at + len(first) + excerpt_chars
            c = Candidate(
                chunk_id=ic.chunk.chunk_id,
                record_id=record_id,
                element=ic.chunk.element_type,
                section=section_of(ic.chunk) or "-",
                terms=found,
                excerpt=("…" if lo else "") + ic.norm[lo:hi] + ("…" if hi < len(ic.norm) else ""),
            )
            seen[ic.norm] = c
            out.append(c)
    return out


def sentences(chunk: Chunk) -> list[str]:
    """The chunk's text as pickable pieces, verbatim: table rows, else sentences."""
    if chunk.element_type == "table":
        return [line for line in chunk.text.splitlines() if line.strip()]
    return [s for s in SENTENCE_END.split(" ".join(chunk.text.split())) if s]


class SpanCheck(StrictModel):
    record_id: str | None = None  # the first record whose chunks contain the span
    chunk_ids: list[str] = []
    section: str | None = None
    hint: Suggestion | None = None  # the nearest passage, when it doesn't map


def check_span(span: str, records: Sequence[str], texts: ChunkTexts) -> SpanCheck:
    """Does the span map to a chunk of the gold paper? (`map_span`, as the harness does.)"""
    for record_id in records:
        ids = map_span(GoldSpan(record_id=record_id, span=span), texts)
        if ids:
            chunk = next(ic.chunk for ic in texts.paper(record_id) if ic.chunk.chunk_id == ids[0])
            return SpanCheck(record_id=record_id, chunk_ids=ids, section=section_of(chunk) or None)
    pool = [ic for r in records for ic in texts.paper(r)]
    return SpanCheck(hint=suggest(span, pool) if pool else None)


def with_span(q: DraftQuestion, index: int, check: SpanCheck, span: str) -> DraftQuestion:
    """The question with gold entry `index` resolved; nothing else changes."""
    gold = list(q.gold)
    gold[index] = gold[index].model_copy(
        update={"span": span, "record_id": check.record_id, "section": check.section}
    )
    return q.model_copy(update={"gold": gold})


Ask = Callable[[str], str]
Say = Callable[[str], None]

HELP = (
    "  <n> open candidate n · t <words> search this paper · e type a span · "
    "k skip · q save and quit"
)


def resolve_gold(
    q: DraftQuestion,
    index: int,
    match: PaperMatch,
    texts: ChunkTexts,
    config: DraftConfig,
    ask: Ask,
    say: Say,
) -> DraftQuestion | Literal["skip", "quit"]:
    """One gold entry: show candidates, let Sharad pick or type a span, check it maps."""
    g = q.gold[index]
    say(f"\n=== {q.id} gold {index + 1}/{len(q.gold)} · {q.type} · {q.question}")
    say(f"    paper: {g.title}  ({match.how}: {', '.join(match.records)})")
    say(f"    hint: {g.section_hint or '-'} · {g.location_hint or '-'}")
    say(f"    gist (to read, not to copy): {g.answer_gist or '-'}")
    shown = candidates(g.span_search, match.records, texts, config.excerpt_chars)
    by_id = {ic.chunk.chunk_id: ic.chunk for r in match.records for ic in texts.paper(r)}
    while True:
        _list(shown, config.max_candidates, f"terms {g.span_search}", say)
        say(HELP)
        answer = ask("> ").strip()
        if answer in ("k", ""):
            return "skip"
        if answer == "q":
            return "quit"
        if answer.startswith("t "):
            shown = candidates([answer[2:]], match.records, texts, config.excerpt_chars)
            continue
        span: str | None = None
        if answer == "e":
            span = ask("span> ").strip()
        elif answer.isdigit() and 1 <= int(answer) <= min(len(shown), config.max_candidates):
            span = _pick_sentences(by_id[shown[int(answer) - 1].chunk_id], ask, say)
        else:
            say("  ?")
            continue
        if not span:
            continue
        check = check_span(span, match.records, texts)
        if check.record_id:
            say(f"  ok: {check.record_id} · {check.section or '-'} · {', '.join(check.chunk_ids)}")
            return with_span(q, index, check, span)
        say("  that span is in no chunk of this paper (spelling and hyphenation must match).")
        if check.hint:
            hint = check.hint
            say(f"  nearest ({hint.similarity:.2f}, {hint.chunk_id}): {hint.excerpt}")


def _list(shown: list[Candidate], limit: int, what: str, say: Say) -> None:
    if not shown:
        say(f"  no chunk contains {what}")
        return
    for n, c in enumerate(shown[:limit], 1):
        copies = f" (same text in {', '.join(c.copies)})" if c.copies else ""
        say(f"  [{n}] {c.chunk_id} · {c.element} · {c.section} · {', '.join(c.terms)}{copies}")
        say(f"      {c.excerpt}")
    if len(shown) > limit:
        say(f"  … {len(shown) - limit} more: narrow with t <words>")


def _pick_sentences(chunk: Chunk, ask: Ask, say: Say) -> str | None:
    pieces = sentences(chunk)
    for n, piece in enumerate(pieces, 1):
        say(f"    ({n}) {piece}")
    answer = ask("  sentences k or k-m, e to type, b back> ").strip()
    if answer == "e":
        return ask("span> ").strip() or None
    found = re.fullmatch(r"(\d+)(?:-(\d+))?", answer)
    if not found:
        return None
    lo, hi = int(found.group(1)), int(found.group(2) or found.group(1))
    if not 1 <= lo <= hi <= len(pieces):
        say("  out of range")
        return None
    return " ".join(pieces[lo - 1 : hi])


KIND_CHOICES = list(UnanswerableKind)


def ask_kind(q: DraftQuestion, ask: Ask, say: Say) -> UnanswerableKind | None:
    """Sharad picks the unanswerable kind; nothing is pre-selected."""
    say(f"\n=== {q.id} · not answerable ({q.expected_behavior or '-'}) · {q.question}")
    for n, kind in enumerate(KIND_CHOICES, 1):
        say(f"  {n} {kind.value}")
    answer = ask("  kind (number, or enter to skip)> ").strip()
    if answer.isdigit() and 1 <= int(answer) <= len(KIND_CHOICES):
        return KIND_CHOICES[int(answer) - 1]
    return None


def resolve(
    questions: list[DraftQuestion],
    corpus: Corpus,
    config: DraftConfig,
    ask: Ask,
    say: Say,
    save: Callable[[list[DraftQuestion]], None],
    only: str | None = None,
) -> int:
    """Walk the unresolved gold entries (or every entry of question `only`).

    Saves after every change, so quitting or a crash loses nothing. Returns
    the number of changes.
    """
    changed = 0
    for qi, q in enumerate(questions):
        if only and q.id != only:
            continue
        if not q.answerable and q.unanswerable_kind is None:
            kind = ask_kind(q, ask, say)
            if kind is not None:
                questions[qi] = q = q.model_copy(update={"unanswerable_kind": kind})
                save(questions)
                changed += 1
        if not q.answerable and q.verify_absent and (q.absent_check is None or only):
            checked = check_absent(q, corpus.texts, config, ask, say)
            if checked is not None:
                questions[qi] = q = checked
                save(questions)
                changed += 1
        for gi, g in enumerate(q.gold):
            if g.span is not None and not only:
                continue
            match = corpus.match(g, config.title_min_ratio)
            if not match.records:
                say(f"\n=== {q.id} gold {gi + 1}: {g.title} is {match.status}: skipped")
                continue
            result = resolve_gold(q, gi, match, corpus.texts, config, ask, say)
            if result == "quit":
                return changed
            if result != "skip":
                questions[qi] = q = result
                save(questions)
                changed += 1
    return changed


# --- absent: is an unanswerable question really unanswered? --------------------------


class TermHits(StrictModel):
    term: str
    chunks: int  # matching chunks
    papers: dict[str, int]  # record_id -> matching chunks, in corpus order
    examples: list[tuple[str, str]]  # (chunk_id, excerpt), the first few


def term_pattern(term: str, short_term_chars: int) -> re.Pattern[str]:
    """Short alphanumeric terms ("VQ") match whole words only, so "VQA" doesn't count."""
    needle = normalise(term)
    if len(needle) <= short_term_chars and needle.isalnum():
        return re.compile(rf"(?<!\w){re.escape(needle)}(?!\w)")
    return re.compile(re.escape(needle))


def absent_hits(
    terms: Sequence[str],
    texts: ChunkTexts,
    config: DraftConfig,
    paper: str | None = None,
) -> list[TermHits]:
    pool = texts.paper(paper) if paper else texts
    out = []
    for term in terms:
        pattern = term_pattern(term, config.short_term_chars)
        papers: dict[str, int] = defaultdict(int)
        examples: list[tuple[str, str]] = []
        for ic in pool:
            found = pattern.search(ic.norm)
            if not found:
                continue
            papers[ic.chunk.record_id] += 1
            if len(examples) < config.absent_max_excerpts:
                lo = max(0, found.start() - config.excerpt_chars)
                examples.append(
                    (ic.chunk.chunk_id, ic.norm[lo : found.end() + config.excerpt_chars])
                )
        out.append(
            TermHits(term=term, chunks=sum(papers.values()), papers=dict(papers), examples=examples)
        )
    return out


def render_absent(q: DraftQuestion, hits: list[TermHits], paper: str | None) -> str:
    scope = f"in {paper}" if paper else "in the whole corpus"
    lines = [f"\n=== {q.id} · {q.unanswerable_kind or 'kind not set'} · {q.question}"]
    if not hits:
        lines.append("  no verify_absent terms: add some to the draft to check this one.")
        return "\n".join(lines)
    for h in hits:
        lines.append(f"  {h.term!r}: {h.chunks} chunks in {len(h.papers)} papers {scope}")
        for chunk_id, excerpt in h.examples:
            lines.append(f"    {chunk_id}: …{excerpt}…")
    lines.append(
        "  Zero hits supports 'not covered' but doesn't prove it (a paraphrase could "
        "answer it); many hits for a generic term say little: narrow with --paper."
    )
    return "\n".join(lines)


def check_absent(
    q: DraftQuestion,
    texts: ChunkTexts,
    config: DraftConfig,
    ask: Ask,
    say: Say,
    paper: str | None = None,
) -> DraftQuestion | None:
    """Show the hits; Sharad confirms or flags the label. None: nothing to record."""
    hits = absent_hits(q.verify_absent, texts, config, paper)
    say(render_absent(q, hits, paper))
    if not hits:
        return None
    answer = ask("  c confirm unanswerable · r flag for relabel · enter skip> ").strip()
    if answer not in ("c", "r"):
        return None
    if answer == "r":
        say("  flagged: edit the question or its labels in the draft by hand.")
    check = AbsentCheck(
        terms=list(q.verify_absent),
        hits={h.term: h.chunks for h in hits},
        paper=paper,
        confirmed=answer == "c",
    )
    return q.model_copy(update={"absent_check": check})


# --- finalize: verified drafts to eval/questions.jsonl ------------------------------


class Finalized(StrictModel):
    questions: list[Question]
    unverified: list[str]  # status draft (or rejected): left out
    refused: dict[str, str]  # verified but not ready: id -> why
    held: dict[str, list[str]]  # conversation -> its verified turns held back


def to_question(d: DraftQuestion) -> Question:
    """The eval-set form: draft-only hints dropped; provenance, gists and checks kept."""
    missing = [i + 1 for i, g in enumerate(d.gold) if not (g.span and g.record_id)]
    if missing:
        raise ValueError(f"gold {missing} has no span yet (rag eval resolve)")
    return Question(
        id=d.id,
        type=d.type,
        question=d.question,
        conversation_id=d.conversation_id,
        turn=d.turn,
        oracle_rewrite=d.oracle_rewrite,
        answerable=d.answerable,
        unanswerable_kind=d.unanswerable_kind,
        gold=[
            GoldSpan(
                record_id=g.record_id, section=g.section, span=g.span, answer_gist=g.answer_gist
            )
            for g in d.gold
            if g.span and g.record_id
        ],
        duplicate_sensitive=d.duplicate_sensitive,
        restatement_notes=d.restatement_notes,
        # An LLM drafted it and Sharad verified each span by hand (Origin's definition).
        origin=Origin.LLM_ASSISTED if d.provenance else Origin.HAND,
        provenance=d.provenance,
        expected_behavior=d.expected_behavior,
        absent_check=d.absent_check,
        notes=d.notes,
    )


def finalize(drafts: Sequence[DraftQuestion]) -> Finalized:
    ready: dict[str, Question] = {}
    unverified: list[str] = []
    refused: dict[str, str] = {}
    for d in drafts:
        if d.status != "verified":
            unverified.append(d.id)
            continue
        try:
            ready[d.id] = to_question(d)
        except ValueError as exc:  # pydantic's ValidationError is a ValueError
            refused[d.id] = " ".join(str(exc).split())
    # A conversation goes in whole or not at all: a missing turn would break
    # the turn order the validator checks, and follow-ups need their history.
    held: dict[str, list[str]] = {}
    turns: dict[str, list[DraftQuestion]] = defaultdict(list)
    for d in drafts:
        if d.conversation_id:
            turns[d.conversation_id].append(d)
    for conversation, members in turns.items():
        if any(m.id not in ready for m in members):
            held[conversation] = [m.id for m in members if m.id in ready]
            for m in members:
                ready.pop(m.id, None)
    order = [d.id for d in drafts]
    return Finalized(
        questions=sorted(ready.values(), key=lambda q: order.index(q.id)),
        unverified=unverified,
        refused=refused,
        held={c: ids for c, ids in held.items() if ids},
    )


def render_finalized(result: Finalized, path: Path) -> str:
    lines = [f"{len(result.questions)} questions -> {path}"]
    if result.unverified:
        lines.append(f"Not verified (left out): {', '.join(result.unverified)}")
    for qid, why in result.refused.items():
        lines.append(f"Refused {qid}: {why}")
    for conversation, ids in result.held.items():
        lines.append(
            f"Held back conversation {conversation} (a turn isn't ready): {', '.join(ids)}"
        )
    return "\n".join(lines)
