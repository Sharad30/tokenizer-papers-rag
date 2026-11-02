"""The hand-labelling loop (D-046): writes only what Sharad enters."""

from pathlib import Path

from rag.eval import labels
from rag.eval.answers import AnswerRecord, Claim, ContextEntry
from rag.eval.spans import ChunkTexts
from tests.eval.conftest import chunk

TEXTS = ChunkTexts([chunk("p:a", 0, "We learn 32k merge operations."), chunk("p:b", 0, "Bytes.")])


def record(qid: str = "L1", **fields) -> AnswerRecord:
    base = dict(
        question_id=qid,
        type="lookup",
        answerable=True,
        user="How many merges?",
        rewrite="How many merges?",
        context=[
            ContextEntry(label="C1", chunk_id="p:a#0000", record_id="p:a", title="BPE"),
            ContextEntry(label="C2", chunk_id="p:b#0000", record_id="p:b", title="Bytes"),
        ],
        top_dense_cosine=0.8,
        model_abstained=False,
        completion="They learn 32k merges [C1]. Bytes are fine too.",
        reply="They learn 32k merges [1]. Bytes are fine too.",
        cited=["C1"],
        invalid_citations=[],
        uncited_sentences=1,
        claims=[
            Claim(index=0, text="They learn 32k merges [C1].", cites=["C1"]),
            Claim(index=1, text="Bytes are fine too.", cites=[]),
        ],
    )
    return AnswerRecord(**{**base, **fields})


class Script:
    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.said: list[str] = []
        self.saves = 0

    def ask(self, prompt: str) -> str:
        return self.answers.pop(0) if self.answers else "q"

    def say(self, text: str) -> None:
        self.said.append(text)

    def save(self, _labels) -> None:
        self.saves += 1


def run(records, script, existing=None, only=None):
    store = list(existing or [])
    gists = {"L1": ["32k merges"]}
    added = labels.label_all(
        records, gists, TEXTS, store, script.ask, script.say, script.save, only
    )
    return store, added


def test_labels_claims_then_the_answer_and_shows_the_cited_passages() -> None:
    script = Script("s", "u", "c")
    store, added = run([record()], script)
    assert [(lb.kind, lb.claim, lb.value) for lb in store] == [
        ("claim", 0, "supported"),
        ("claim", 1, "unsupported"),
        ("answer", None, "correct"),
    ]
    assert added == 3 and script.saves == 3
    text = "\n".join(script.said)
    assert "We learn 32k merge operations." in text  # the cited passage, in full
    assert "uncited: judge it against the whole context" in text
    assert "Gist (reference): 32k merges" in text


def test_nothing_is_written_on_skip_quit_or_no_input() -> None:
    for answers in [("", "", ""), ("q",), ()]:
        store, added = run([record()], Script(*answers))
        assert store == [] and added == 0


def test_a_rerun_asks_only_for_missing_labels() -> None:
    existing = [labels.Label(question_id="L1", kind="claim", claim=0, value="supported")]
    script = Script("p", "w")
    store, _ = run([record()], script, existing)
    assert [(lb.claim, lb.value) for lb in store[1:]] == [(1, "partial"), (None, "wrong")]
    assert not any("Claim 1/2" in line for line in script.said)


def test_abstentions_and_follow_up_rewrites_get_their_own_choices() -> None:
    abstained = record(
        "U1",
        answerable=False,
        model_abstained=True,
        completion="NOT_SUPPORTED",
        claims=[],
        cited=[],
        uncited_sentences=0,
    )
    followup = record(
        "C1-T2",
        type="followup",
        claims=[],
        rewrite="How many merges does BPE learn?",
        oracle_rewrite="How many merge operations does the BPE paper learn?",
    )
    store, _ = run([abstained, followup], Script("a", "c", "e"))
    assert [(lb.question_id, lb.kind, lb.value) for lb in store] == [
        ("U1", "answer", "appropriate"),
        ("C1-T2", "answer", "correct"),
        ("C1-T2", "rewrite", "equivalent"),
    ]


def test_labels_round_trip_next_to_the_answers_file(tmp_path: Path) -> None:
    path = labels.labels_path(tmp_path / "2026-11-02-answers.jsonl")
    assert path.name == "2026-11-02-answers.labels.jsonl"
    saved = [labels.Label(question_id="L1", kind="answer", value="correct")]
    labels.save_to(path)(saved)
    assert labels.read_labels(path) == saved
