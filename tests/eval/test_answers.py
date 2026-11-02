"""The answers run (D-046) on the synthetic chat corpus, with a scripted LLM."""

from rag.chat.citations import claim_sentences
from rag.eval import answers
from rag.eval.schemas import Question
from rag.eval.spans import ChunkTexts
from tests.chat.test_engine import CHUNKS_DATA, config, make_engine  # noqa: F401 (fixture)


def q(id: str, text: str, **fields) -> Question:  # noqa: A002
    base = {"id": id, "type": "lookup", "question": text, "answerable": True}
    base["gold"] = [{"record_id": "p:a", "span": "merge merge operations"}]
    return Question.model_validate({**base, **fields})


def test_claims_are_sentences_with_their_citations() -> None:
    text = "BPE merges pairs of symbols [C1]. It starts from characters [C1, C3].\nShort one.\n"
    text += "No citation in this claim at all."
    assert claim_sentences(text) == [
        ("BPE merges pairs of symbols [C1].", ["C1"]),
        ("It starts from characters [C1, C3].", ["C1", "C3"]),
        ("No citation in this claim at all.", []),
    ]  # "Short one." is under MIN_CLAIM_WORDS


def test_conversations_run_in_turn_order_and_standalones_alone() -> None:
    qs = [
        q("C1-T2", "and then?", type="followup", conversation_id="C1", turn=2, oracle_rewrite="x"),
        q("L1", "merge operations"),
        q("C1-T1", "merge operations", conversation_id="C1", turn=1),
    ]
    groups = answers.conversations(qs)
    assert [[x.id for x in g] for g in groups] == [["C1-T1", "C1-T2"], ["L1"]]


def test_run_records_context_gold_and_claims(config) -> None:  # noqa: F811
    engine, llm = make_engine(config, lambda s, m: "BPE merges symbol pairs [C1, C9]. Okay.")
    texts = ChunkTexts(CHUNKS_DATA)
    seen = []
    records = answers.run([q("L1", "merge operations")], engine, texts, {}, on_answer=seen.append)
    r = records[0]
    assert seen == records
    assert r.context[0].chunk_id == "p:a#0000" and r.context[0].label == "C1"
    assert r.gold_passages_in_context == 1.0 and r.gold_papers_in_context == 1.0
    assert r.cited == ["C1"] and r.invalid_citations == ["C9"]
    assert [c.cites for c in r.claims] == [["C1", "C9"]]  # claims keep what the model wrote
    assert r.top_dense_cosine is not None and not r.model_abstained


def test_each_conversation_starts_from_a_fresh_history(config) -> None:  # noqa: F811
    engine, llm = make_engine(config, lambda s, m: "byte level models [C1].")
    qs = [
        q("A-T1", "merge operations", conversation_id="A", turn=1),
        q("L1", "byte level models"),
    ]
    answers.run(qs, engine, ChunkTexts(CHUNKS_DATA), {})
    assert engine.history and len(engine.history) == 1  # reset before L1


def test_model_abstention_has_no_claims(config) -> None:  # noqa: F811
    engine, _ = make_engine(config, lambda s, m: "NOT_SUPPORTED")
    unanswerable = q(
        "U1",
        "merge operations",
        type="unanswerable",
        answerable=False,
        gold=[],
        unanswerable_kind="not_covered",
    )
    r = answers.run([unanswerable], engine, ChunkTexts(CHUNKS_DATA), {})[0]
    assert r.model_abstained and r.claims == [] and r.gold_passages_in_context is None
