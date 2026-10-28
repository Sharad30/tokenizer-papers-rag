import json

from rag.eval.config import SilverConfig
from rag.eval.silver import make_question, parse_reply, sample_chunks, word_overlap
from rag.eval.spans import ChunkTexts
from tests.retrieve.conftest import make_chunk

CFG = SilverConfig(
    file="silver/questions.jsonl",
    prompt="p.md",
    n=10,
    seed=0,
    elements=["paragraph"],
    min_tokens=3,
    max_tokens=100,
    min_question_words=3,
    max_evidence_words=30,
)
CHUNK = make_chunk("acl:x", 0, "We train BPE with 32k merges. The vocabulary covers 99% of tokens.")


def test_sampling_takes_one_eligible_chunk_per_canonical_non_survey_paper() -> None:
    chunks = [
        make_chunk("acl:a", 0, "one two three four"),
        make_chunk("acl:a", 1, "five six seven eight"),
        make_chunk("arxiv:a-alias", 0, "one two three four"),  # not canonical
        make_chunk("acl:survey", 0, "a survey of many things"),  # a survey
        make_chunk("acl:short", 0, "too short"),  # under min_tokens
    ]
    picked = sample_chunks(chunks, {"acl:a", "acl:survey", "acl:short"}, {"acl:survey"}, CFG)
    assert [c.record_id for c in picked] == ["acl:a"]
    assert sample_chunks(chunks, {"acl:a"}, set(), CFG) == picked  # seeded


def test_reply_parsing_tolerates_text_around_the_json() -> None:
    reply = 'Sure!\n```json\n{"question": "How many merges?", "evidence": "x"}\n```'
    assert parse_reply(reply) == ("How many merges?", "x")
    assert parse_reply("no json here") is None
    assert parse_reply('{"question": 3}') is None


def test_a_question_is_kept_only_when_its_evidence_is_in_the_chunk() -> None:
    texts = ChunkTexts([CHUNK])

    def reply(question: str, evidence: str) -> str:
        return json.dumps({"question": question, "evidence": evidence})

    kept, reason = make_question(
        1,
        CHUNK,
        reply("How many BPE merge operations were used?", "We train BPE with 32k merges."),
        texts,
        CFG,
    )
    assert reason == "kept" and kept.id == "s001" and kept.origin == "silver_llm"
    assert kept.gold[0].record_id == "acl:x"
    assert json.loads(kept.notes)["chunk_id"] == CHUNK.chunk_id
    assert (
        make_question(
            1, CHUNK, reply("How many merges were used?", "We use 64k merges."), texts, CFG
        )[1]
        == "evidence_not_in_chunk"
    )
    assert (
        make_question(1, CHUNK, reply("Merges", "We train BPE with 32k merges."), texts, CFG)[1]
        == "not_a_question"
    )


def test_word_overlap_counts_content_words_found_in_the_passage() -> None:
    passage = "We train BPE with 32k merges."
    # content words: many, bpe, merges; "bpe" and "merges" are in the passage
    assert word_overlap("How many BPE merges?", passage) == 2 / 3
    assert word_overlap("What vocabulary size for Unigram?", passage) == 0.0
