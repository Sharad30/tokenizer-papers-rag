"""`rag eval label`: Sharad judges the chat's answers by hand (D-046).

No model judges anything here. For each answer from `rag eval answers` the
labeller shows the question, the gold gist (what a correct answer says) and
the answer, then walks its claims, each with the full text of the passages it
cites. Sharad labels:

    claim     supported | partial | unsupported   (against its cited passages)
    answer    correct | partial | wrong            (against the gist), or
              appropriate | inappropriate          (for an abstention)
    rewrite   equivalent | different               (follow-ups: system vs oracle rewrite)

Labels go to <answers>.labels.jsonl, saved after every one; a rerun skips
what is already labelled. Code never writes a label on its own.
"""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from rag.common.io import read_jsonl, write_jsonl
from rag.common.schemas import StrictModel
from rag.eval.answers import AnswerRecord, Claim
from rag.eval.spans import ChunkTexts

Kind = Literal["claim", "answer", "rewrite"]
VALUES: dict[str, dict[str, str]] = {
    # key typed -> value stored, per kind
    "claim": {"s": "supported", "p": "partial", "u": "unsupported"},
    "answer": {"c": "correct", "p": "partial", "w": "wrong"},
    "abstention": {"a": "appropriate", "i": "inappropriate"},
    "rewrite": {"e": "equivalent", "d": "different"},
}
PASSAGE_CHARS = 1200  # of each cited passage shown (chunks are ~300 tokens)


class Label(StrictModel):
    question_id: str
    kind: Kind
    claim: int | None = None  # claim index, for kind "claim"
    value: str


def labels_path(answers_path: Path) -> Path:
    return answers_path.with_name(answers_path.stem + ".labels.jsonl")


def read_labels(path: Path) -> list[Label]:
    return read_jsonl(path, Label) if path.is_file() else []


def key(label: Label) -> tuple[str, str, int | None]:
    return (label.question_id, label.kind, label.claim)


Ask = Callable[[str], str]
Say = Callable[[str], None]


def label_all(
    records: Sequence[AnswerRecord],
    gists: dict[str, list[str]],
    texts: ChunkTexts,
    labels: list[Label],
    ask: Ask,
    say: Say,
    save: Callable[[list[Label]], None],
    only: str | None = None,
) -> int:
    """Ask for every missing label. Returns how many were added; stops on "q"."""
    done = {key(lb) for lb in labels}
    by_chunk = {ic.chunk.chunk_id: ic.chunk for ic in texts}
    added = 0

    def put(label: Label) -> None:
        nonlocal added
        labels.append(label)
        done.add(key(label))
        save(labels)
        added += 1

    for r in records:
        if only and r.question_id != only:
            continue
        todo_claims = [c for c in r.claims if (r.question_id, "claim", c.index) not in done]
        todo_answer = (r.question_id, "answer", None) not in done
        todo_rewrite = r.oracle_rewrite and (r.question_id, "rewrite", None) not in done
        if not (todo_claims or todo_answer or todo_rewrite):
            continue
        _header(r, gists.get(r.question_id, []), say)
        for claim in todo_claims:
            _show_claim(claim, r, by_chunk, say)
            value = _choose("claim", ask, say)
            if value == "quit":
                return added
            if value:
                put(Label(question_id=r.question_id, kind="claim", claim=claim.index, value=value))
        if todo_answer:
            kind = "abstention" if r.model_abstained else "answer"
            say(f"  Whole answer ({'abstained' if r.model_abstained else 'answered'}):")
            value = _choose(kind, ask, say)
            if value == "quit":
                return added
            if value:
                put(Label(question_id=r.question_id, kind="answer", value=value))
        if todo_rewrite:
            say(f"  Rewrite  (system): {r.rewrite}\n           (oracle): {r.oracle_rewrite}")
            value = _choose("rewrite", ask, say)
            if value == "quit":
                return added
            if value:
                put(Label(question_id=r.question_id, kind="rewrite", value=value))
    return added


def _header(r: AnswerRecord, gists: list[str], say: Say) -> None:
    turn = f" · {r.conversation_id} turn {r.turn}" if r.conversation_id else ""
    expected = r.expected_behavior or ("answer" if r.answerable else "abstain")
    say(f"\n=== {r.question_id} · {r.type}{turn} · expected: {expected}")
    say(f"Q: {r.user}")
    if r.rewrite != r.user:
        say(f"   (retrieved with: {r.rewrite})")
    for g in gists:
        say(f"Gist (reference): {g}")
    say(f"Answer:\n{r.completion or r.reply}")


def _show_claim(claim: Claim, r: AnswerRecord, by_chunk: dict, say: Say) -> None:
    say(f"\n  Claim {claim.index + 1}/{len(r.claims)}: {claim.text}")
    labels = claim.cites or [c.label for c in r.context]
    if not claim.cites:
        say("  (uncited: judge it against the whole context)")
    entries = {c.label: c for c in r.context}
    for label in labels:
        entry = entries.get(label)
        if entry is None:
            say(f"    [{label}] not in the context (an invalid citation)")
            continue
        text = " ".join(by_chunk[entry.chunk_id].text.split())[:PASSAGE_CHARS]
        say(f"    [{label}] {entry.title} ({entry.chunk_id})\n      {text}")


def _choose(kind: str, ask: Ask, say: Say) -> str | None:
    """A value for this kind, None to skip, "quit" to stop."""
    options = VALUES[kind]
    hint = " · ".join(f"{k} {v}" for k, v in options.items())
    while True:
        answer = ask(f"  {hint} · enter skip · q quit> ").strip().lower()
        if answer == "q":
            return "quit"
        if answer == "":
            return None
        if answer in options:
            return options[answer]
        say("  ?")


def save_to(path: Path) -> Callable[[list[Label]], None]:
    return lambda labels: write_jsonl(path, labels)
