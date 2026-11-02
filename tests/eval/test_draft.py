"""Drafted questions to gold (D-045), on a tiny synthetic corpus."""

import json
from pathlib import Path

import pytest

from rag.collect.schemas import ManifestRecord, Source
from rag.eval import draft
from rag.eval.spans import ChunkTexts
from tests.eval.conftest import chunk


def rec(record_id: str, title: str, **ids) -> ManifestRecord:
    return ManifestRecord(record_id=record_id, source=Source.ARXIV, title=title, **ids)


BPE = "Neural Machine Translation of Rare Words with Subword Units"
SP = "SentencePiece: A simple and language independent subword tokenizer"
RECORDS = [
    # One paper, three records: two arXiv versions (v1 not indexed) and the ACL copy.
    rec("arxiv:1508.07909v1", BPE, arxiv_id="1508.07909", s2_id="S1"),
    rec("arxiv:1508.07909v5", BPE, arxiv_id="1508.07909", s2_id="S1"),
    rec("acl:P16-1162", BPE, acl_id="P16-1162", s2_id="S1"),
    rec("acl:D18-2012", SP, acl_id="D18-2012"),
    rec("s2:abc", "Tokenization efficiency for low-resource languages"),
    rec("arxiv:2000.00001v1", "Byte models are not indexed", arxiv_id="2000.00001"),
]
CHUNKS = [
    chunk("arxiv:1508.07909v5", 0, "We adapt BPE from Gage (1994).", ["Intro"]),
    chunk(
        "arxiv:1508.07909v5",
        1,
        "We iteratively replace the most frequent pair. Merge operations set the size.",
        ["Byte Pair Encoding"],
    ),
    chunk("acl:P16-1162", 0, "We iteratively replace the most frequent pair.", ["BPE"]),
    chunk("acl:D18-2012", 0, "Whitespace is escaped as U+2581 so tokenization is lossless."),
    chunk("s2:abc", 0, "Fertility is the number of tokens per word."),
]


@pytest.fixture
def corpus() -> draft.Corpus:
    return draft.Corpus(RECORDS, ChunkTexts(CHUNKS), canonical={"acl:P16-1162"})


def gold(title: str, **fields) -> draft.DraftGold:
    return draft.DraftGold(title=title, **fields)


# --- the draft file -------------------------------------------------------------------

DRAFT_LINE = {
    "id": "L01",
    "type": "lookup",
    "conversation_id": None,
    "turn": None,
    "question": "What does BPE repeat?",
    "oracle_rewrite": None,
    "answerable": True,
    "expected_behavior": "answer",
    "gold": [
        {
            "title": BPE,
            "arxiv_id": "1508.07909",
            "alt_id": None,
            "section_hint": "Byte Pair Encoding",
            "location_hint": "paragraph",
            "span_search": ["most frequent pair"],
            "answer_gist": "It merges the most frequent pair.",
            "span": None,
            "section": None,
        }
    ],
    "verify_absent": [],
    "duplicate_sensitive": True,
    "notes": "restated widely",
    "status": "draft",
    "provenance": "llm_drafted",
}


def write_lines(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_draft_round_trips_without_adding_unset_fields(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "d.jsonl", [DRAFT_LINE])
    draft.write_draft(path, draft.read_draft(path))
    assert json.loads(path.read_text(encoding="utf-8")) == DRAFT_LINE
    assert (tmp_path / "d.jsonl.bak").is_file()


def test_a_mistyped_field_fails_loudly(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "d.jsonl", [{**DRAFT_LINE, "staus": "verified"}])
    with pytest.raises(ValueError, match="staus"):
        draft.read_draft(path)


# --- matching gold papers -------------------------------------------------------------


def test_arxiv_id_matches_every_version_and_linked_copy(corpus: draft.Corpus) -> None:
    m = corpus.match(gold(BPE, arxiv_id="1508.07909v3"), 90)
    assert (m.status, m.how) == ("matched", "arxiv_id")
    assert m.records == ["acl:P16-1162", "arxiv:1508.07909v5"]  # canonical first
    assert m.unindexed == ["arxiv:1508.07909v1"]


def test_alt_id_matches_a_record_id(corpus: draft.Corpus) -> None:
    m = corpus.match(gold("whatever", alt_id="s2:abc"), 90)
    assert (m.status, m.how, m.records) == ("matched", "alt_id", ["s2:abc"])


def test_title_is_the_fallback_and_is_reported_as_such(corpus: draft.Corpus) -> None:
    title = "SentencePiece - a simple and language-independent subword tokenizer"
    m = corpus.match(gold(title), 90)
    assert m.status == "title_only" and m.records == ["acl:D18-2012"]
    assert m.how.startswith("title ")


def test_unknown_paper_is_missing_and_unindexed_paper_is_reported(
    corpus: draft.Corpus,
) -> None:
    assert corpus.match(gold("MEGABYTE", arxiv_id="2305.07185"), 90).status == "missing"
    m = corpus.match(gold("Byte models", arxiv_id="2000.00001"), 90)
    assert (m.status, m.unindexed) == ("not_indexed", ["arxiv:2000.00001v1"])


def test_check_report_lists_problem_papers_with_their_questions(corpus: draft.Corpus) -> None:
    q1 = draft.DraftQuestion.model_validate(DRAFT_LINE)
    q2 = draft.DraftQuestion.model_validate(
        {**DRAFT_LINE, "id": "L08", "gold": [{"title": "MEGABYTE", "arxiv_id": "2305.07185"}]}
    )
    rows = draft.check_papers([q1, q2], corpus, 90)
    text = draft.render_check(rows, corpus)
    assert text.startswith("2 gold entries: 1 matched, 0 title_only, 0 not_indexed, 1 missing")
    assert "MEGABYTE  [2305.07185]" in text and "questions: L08" in text


# --- resolve --------------------------------------------------------------------------


def typed(c, element: str):
    return c.model_copy(update={"element_type": element})


def test_candidates_are_in_corpus_order_with_any_term_and_copies_listed_once() -> None:
    texts = ChunkTexts(
        [
            chunk("acl:x", 0, "Merges happen here. Nothing about pairs."),
            chunk("acl:x", 1, "The most frequent pair is merged, many merges."),
            typed(chunk("acl:x", 2, "| Merges | 32k |"), "table"),
            typed(chunk("acl:x", 3, "Figure 2: merges per step."), "figure_caption"),
            chunk("arxiv:x", 0, "Merges happen here. Nothing about pairs."),  # linked copy
        ]
    )
    found = draft.candidates(["most frequent pair", "MERGES"], ["acl:x", "arxiv:x"], texts, 10)
    # Corpus order, not by how many terms match (#0001 matches both, still second).
    assert [c.chunk_id for c in found] == ["acl:x#0000", "acl:x#0001", "acl:x#0002", "acl:x#0003"]
    assert found[1].terms == ["most frequent pair", "MERGES"]
    assert [c.element for c in found[2:]] == ["table", "figure_caption"]
    assert found[0].copies == ["arxiv:x#0000"]


def test_sentences_are_verbatim_and_tables_split_by_row() -> None:
    para = chunk("p", 0, "We use BPE. It has 32k merges (Sennrich et al., 2016). Done!")
    assert draft.sentences(para) == [
        "We use BPE.",
        "It has 32k merges (Sennrich et al., 2016).",
        "Done!",
    ]
    table = typed(chunk("p", 1, "| Model | BLEU |\n| BPE | 24.1 |"), "table")
    assert draft.sentences(table) == ["| Model | BLEU |", "| BPE | 24.1 |"]


def test_a_span_maps_to_the_first_record_containing_it(corpus: draft.Corpus) -> None:
    records = ["acl:P16-1162", "arxiv:1508.07909v5"]
    check = draft.check_span("iteratively REPLACE the most frequent pair", records, corpus.texts)
    assert (check.record_id, check.chunk_ids, check.section) == (
        "acl:P16-1162",
        ["acl:P16-1162#0000"],
        "BPE",
    )
    only_v5 = draft.check_span("Merge operations set the size.", records, corpus.texts)
    assert only_v5.record_id == "arxiv:1508.07909v5"


def test_a_span_that_does_not_map_is_refused_with_the_nearest_passage(
    corpus: draft.Corpus,
) -> None:
    check = draft.check_span(
        "we iteratively replace the most frequnt pair", ["acl:P16-1162"], corpus.texts
    )
    assert check.record_id is None
    assert check.hint is not None and check.hint.chunk_id == "acl:P16-1162#0000"


class Script:
    """Scripted answers for the interactive loop, and everything it printed."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.said: list[str] = []
        self.saves = 0

    def ask(self, prompt: str) -> str:
        return self.answers.pop(0) if self.answers else "q"

    def say(self, text: str) -> None:
        self.said.append(text)

    def save(self, questions) -> None:
        self.saves += 1


def config():
    from rag.eval.config import DraftConfig

    return DraftConfig(
        file="d.jsonl",
        title_min_ratio=90,
        max_candidates=5,
        excerpt_chars=30,
        absent_max_excerpts=2,
        short_term_chars=3,
    )


def lookup() -> draft.DraftQuestion:
    return draft.DraftQuestion.model_validate(DRAFT_LINE)


def test_resolve_writes_only_the_span_sharad_picks(corpus: draft.Corpus) -> None:
    # Candidates: acl #0000 first (canonical), then v5 #0001. Open [2], take sentence 1.
    script = Script("2", "1")
    questions = [lookup()]
    assert draft.resolve(questions, corpus, config(), script.ask, script.say, script.save) == 1
    g = questions[0].gold[0]
    assert (g.span, g.record_id, g.section) == (
        "We iteratively replace the most frequent pair.",
        "acl:P16-1162",  # the same sentence is in the canonical copy, checked first
        "BPE",
    )
    assert (
        questions[0].question == DRAFT_LINE["question"]
        and g.answer_gist == "It merges the most frequent pair."
    )
    assert script.saves == 1


def test_resolve_writes_nothing_on_skip_quit_or_no_input(corpus: draft.Corpus) -> None:
    for answers in [("k",), ("q",), ()]:
        script = Script(*answers)
        questions = [lookup()]
        assert draft.resolve(questions, corpus, config(), script.ask, script.say, script.save) == 0
        assert questions[0].gold[0].span is None and script.saves == 0


def test_resolve_retries_a_typed_span_until_it_maps(corpus: draft.Corpus) -> None:
    script = Script("e", "the most frequnt pair", "e", "the most frequent pair")
    questions = [lookup()]
    draft.resolve(questions, corpus, config(), script.ask, script.say, script.save)
    assert questions[0].gold[0].span == "the most frequent pair"
    assert any("nearest" in line for line in script.said)


def test_resolve_asks_for_the_unanswerable_kind_and_never_guesses(corpus: draft.Corpus) -> None:
    row = {**DRAFT_LINE, "type": "unanswerable", "answerable": False, "gold": []}
    row["expected_behavior"] = "abstain"
    script = Script("")  # enter: skip
    questions = [draft.DraftQuestion.model_validate(row)]
    draft.resolve(questions, corpus, config(), script.ask, script.say, script.save)
    assert questions[0].unanswerable_kind is None
    script = Script("1")
    draft.resolve(questions, corpus, config(), script.ask, script.say, script.save)
    assert questions[0].unanswerable_kind == "not_covered"
