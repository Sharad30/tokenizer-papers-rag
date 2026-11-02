"""Apply eval/gold/claude_resolution.yaml to eval/gold/questions.draft.jsonl (one-off, 2026-11-02).

Sharad asked Claude to run the draft pipeline instead of picking spans by
hand (eval/gold/GOLD_BUILD.md). This does what `rag eval resolve` would with
Claude at the keyboard, through the same functions: each phrase is expanded
to the original text from its first to its last phrase in the gold paper's
chunks (canonical record first), and written only if `check_span` maps it. Questions, gists
and labels are not edited; provenance records that Claude resolved them.

Run:  uv run python eval/gold/resolve_by_claude.py            (prints what it would write)
      uv run python eval/gold/resolve_by_claude.py --write    (writes the draft, keeps a .bak)
"""

import sys
from pathlib import Path

import yaml

from rag.common.config import load_config
from rag.eval import draft
from rag.eval.cli import _corpus
from rag.eval.schemas import AbsentCheck, UnanswerableKind
from rag.eval.spans import normalise

CHOICES = Path(__file__).with_name("claude_resolution.yaml")
PROVENANCE = "llm_drafted; spans and labels picked by Claude at Sharad's request (2026-11-02)"


def _normalised_with_offsets(text: str) -> tuple[str, list[int]]:
    """normalise(text), plus the original index each normalised character came from."""
    out: list[str] = []
    where: list[int] = []
    for i, ch in enumerate(text):
        piece = normalise(ch) if not ch.isspace() else " "
        for p in piece or " ":
            if p == " " and (not out or out[-1] == " "):
                continue  # collapse runs of whitespace, as normalise does
            out.append(p)
            where.append(i)
    return "".join(out), where


def expand(phrases: list[str], records: list[str], corpus: draft.Corpus) -> str | None:
    """Original chunk text from the start of the first phrase to the end of the last."""
    first, last = normalise(phrases[0]), normalise(phrases[-1])
    for record_id in records:
        for ic in corpus.texts.paper(record_id):
            if first not in ic.norm or last not in ic.norm:
                continue
            norm, where = _normalised_with_offsets(ic.chunk.text)
            lo = norm.find(first)
            hi = norm.find(last, lo) + len(last) if lo >= 0 else -1
            if lo < 0 or hi < len(last):
                continue
            # One line: chunk text keeps PDF line breaks; matching ignores whitespace.
            return " ".join(ic.chunk.text[where[lo] : where[hi - 1] + 1].split())
    return None


def main(write: bool) -> int:
    config = load_config()
    path = config.paths.eval / config.eval.draft.file
    choices = yaml.safe_load(CHOICES.read_text(encoding="utf-8"))
    questions = draft.read_draft(path)
    corpus = _corpus(config)
    failures = 0
    for qi, q in enumerate(questions):
        update: dict = {}
        picks = choices["spans"].get(q.id)
        if picks is not None:
            per_gold = [picks] if isinstance(picks[0], str) else picks
            if len(per_gold) != len(q.gold):
                print(f"{q.id}: {len(per_gold)} picks for {len(q.gold)} gold entries")
                failures += 1
                continue
            for gi, phrases in enumerate(per_gold):
                match = corpus.match(q.gold[gi], config.eval.draft.title_min_ratio)
                span = expand(phrases, match.records, corpus)
                check = draft.check_span(span, match.records, corpus.texts) if span else None
                if not check or not check.record_id:
                    print(f"{q.id} g{gi}: no sentence contains {phrases}")
                    failures += 1
                    continue
                q = draft.with_span(q, gi, check, span)
                where = f"{check.record_id} {check.chunk_ids} §{check.section}"
                print(f"{q.id} g{gi}: {where}\n    {span}")
        if q.id in choices["kinds"]:
            update["unanswerable_kind"] = UnanswerableKind(choices["kinds"][q.id])
        if q.id in choices["absent"]:
            spec = choices["absent"][q.id]
            paper = spec.get("paper")
            hits = draft.absent_hits(q.verify_absent, corpus.texts, config.eval.draft, paper)
            update["absent_check"] = AbsentCheck(
                terms=list(q.verify_absent),
                hits={h.term: h.chunks for h in hits},
                paper=paper,
                confirmed=spec["confirmed"],
            )
        ready = q.id not in choices["skip"] and all(g.span for g in q.gold)
        if ready:
            update["status"] = "verified"
            update["provenance"] = PROVENANCE
        questions[qi] = q.model_copy(update=update)
    verified = [q.id for q in questions if q.status == "verified"]
    left = len(questions) - len(verified)
    print(f"\n{len(verified)} verified, {left} left as draft, {failures} failures")
    if write and not failures:
        draft.write_draft(path, questions)
        print(f"written: {path} (previous version in {path.name}.bak)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main("--write" in sys.argv))
