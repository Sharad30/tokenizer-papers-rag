import json
from pathlib import Path

import pytest

from rag.chat.engine import ChatEngine, format_sources
from rag.chat.llm import ScriptedLLM
from rag.chat.prompts import load_prompt
from rag.chat.transcript import Transcript
from rag.collect.schemas import ManifestRecord, Source
from rag.common.config import load_config
from rag.common.io import write_jsonl
from rag.index import pipeline
from rag.parse.pipeline import CHUNKS, step_dir
from rag.retrieve.retriever import Retriever
from tests.retrieve.conftest import FakeEncoder, make_chunk

CHUNKS_DATA = [
    make_chunk("p:a", 0, "merge merge operations", header="BPE > Method"),
    make_chunk("p:a", 1, "byte level models", header="BPE > Bytes"),
    make_chunk("p:b", 0, "unigram language model", header="LM > Method"),
]
RECORDS = {
    "p:a": ManifestRecord(record_id="p:a", source=Source.ACL, title="BPE Paper", year=2016),
    "p:b": ManifestRecord(record_id="p:b", source=Source.ARXIV, title="Unigram Paper", year=2018),
}


@pytest.fixture
def config(tmp_path: Path, minimal_config_text: str):
    path = tmp_path / "config.yaml"
    path.write_text(minimal_config_text)
    config = load_config(path)
    write_jsonl(step_dir(config, "chunk") / CHUNKS, CHUNKS_DATA)
    pipeline.bm25(config)
    pipeline.dense(config, encoder=FakeEncoder())
    config.paths.prompts.mkdir(parents=True)
    (config.paths.prompts / "answer_system.md").write_text("Answer from context.")
    (config.paths.prompts / "rewrite_system.md").write_text("Rewrite.")
    return config


def make_engine(config, reply, min_cosine: float = -1.0) -> tuple[ChatEngine, ScriptedLLM]:
    chat = config.chat.model_copy(
        update={"abstain": config.chat.abstain.model_copy(update={"min_dense_cosine": min_cosine})}
    )
    llm = ScriptedLLM(reply)
    engine = ChatEngine(
        Retriever.load(config, {"hybrid"}, encoder=FakeEncoder()),
        llm,
        RECORDS,
        chat,
        load_prompt(config.paths.prompts, "answer_system.md"),
        load_prompt(config.paths.prompts, "rewrite_system.md"),
    )
    return engine, llm


def test_answer_with_valid_and_invalid_citations(config) -> None:
    engine, llm = make_engine(
        config, lambda s, m: "BPE merges symbol pairs [C1, C9]. Bytes too [C2]."
    )
    result = engine.turn("merge operations")
    assert result.abstained is None
    assert result.reply == "BPE merges symbol pairs [1]. Bytes too [2]."
    assert result.cited.invalid == ["C9"]
    sources = format_sources(result)
    assert sources[0].startswith("[1] BPE Paper (2016), paragraph")  # no section_path here
    assert "p:a#0000" in sources[0]
    # The generator saw the context and the question, not the chat history.
    system, messages, _ = llm.calls[0]
    assert system == "Answer from context."
    assert messages[0].content.startswith("Context passages:\n\n[C1] BPE Paper (2016)")
    assert messages[0].content.endswith("Question: merge operations")


def test_gate_one_abstains_without_calling_the_llm(config) -> None:
    engine, llm = make_engine(config, lambda s, m: "unused", min_cosine=0.999)
    result = engine.turn("merge operations")
    assert result.abstained == "retrieval_gate"
    assert llm.calls == []
    assert "below the 1.00 needed" in result.reply
    assert "Nearest papers (not supporting an answer): BPE Paper" in result.reply


@pytest.mark.parametrize(
    "raw", ["NOT_SUPPORTED", "NOT_SUPPORTED.", "The passages don't cover baking. NOT_SUPPORTED."]
)
def test_model_not_supported_is_an_abstention(config, raw: str) -> None:
    engine, _ = make_engine(config, lambda s, m: raw)
    result = engine.turn("merge operations")
    assert result.abstained == "model"
    assert result.reply.startswith("The retrieved passages don't support an answer")
    assert result.sources == []


def test_follow_up_is_rewritten_with_history_and_reset_clears_it(config) -> None:
    def reply(system, messages):
        if system == "Rewrite.":
            return "unigram language model segmentation"
        return "Unigram is a language model [C1]."

    engine, llm = make_engine(config, reply)
    first = engine.turn("byte level")
    assert not first.rewrite.rewritten and first.turn == 1
    second = engine.turn("and the unigram one?")
    assert second.rewrite.rewritten and second.turn == 2
    assert second.rewrite.query == "unigram language model segmentation"
    rewrite_prompt = next(m for s, m, _ in llm.calls if s == "Rewrite.")[0].content
    assert "User: byte level" in rewrite_prompt
    engine.reset()
    assert engine.turn("unigram").turn == 1


def test_transcript_records_session_and_turns(config, tmp_path: Path) -> None:
    engine, _ = make_engine(config, lambda s, m: "Merges [C1].")
    transcript = Transcript(tmp_path / "chat")
    transcript.write_header("abc", "hash", "scripted", [engine.answer_prompt], {"k": 3})
    transcript.write_turn(engine.turn("merge operations"))
    transcript.write_reset()
    lines = [json.loads(line) for line in transcript.path.read_text().splitlines()]
    assert [line["type"] for line in lines] == ["session", "turn", "reset"]
    assert lines[0]["prompts"]["answer_system.md"] == engine.answer_prompt.sha256
    turn = lines[1]
    assert turn["raw_answer"] == "Merges [C1]." and turn["reply"] == "Merges [1]."
    assert turn["hits"][0]["label"] == "C1" and "dense" in turn["top_scores"]
    assert turn["sources"] == [{"n": 1, "chunk_id": turn["hits"][0]["chunk_id"]}]
