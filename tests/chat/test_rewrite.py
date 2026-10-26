from pathlib import Path

import pytest

from rag.chat.config import RewriteConfig
from rag.chat.llm import LLMError, ScriptedLLM
from rag.chat.prompts import load_prompt
from rag.chat.rewrite import Exchange, clean, rewrite

CONFIG = RewriteConfig(enabled=True, history_turns=2, max_answer_chars=10, max_tokens=20)
HISTORY = [
    Exchange("What is BPE?", "BPE merges frequent pairs of symbols."),
    Exchange("Who proposed BPE-dropout?", "Provilkov et al. proposed it."),
    Exchange("What does it drop?", "It drops merges at random."),
]


@pytest.fixture
def prompt(tmp_path: Path):
    (tmp_path / "rewrite.md").write_text("Rewrite it.\n")
    return load_prompt(tmp_path, "rewrite.md")


def test_first_turn_is_not_rewritten(prompt) -> None:
    llm = ScriptedLLM(lambda s, m: "never")
    result = rewrite(llm, prompt, [], "What is BPE?", CONFIG)
    assert (result.query, result.rewritten, result.fallback) == ("What is BPE?", False, False)
    assert llm.calls == []


def test_uses_only_recent_exchanges_and_truncates_answers(prompt) -> None:
    llm = ScriptedLLM(lambda s, m: "How does BPE-dropout compare to Unigram?")
    result = rewrite(llm, prompt, HISTORY, "How does it compare to Unigram?", CONFIG)
    assert result.rewritten and result.query == "How does BPE-dropout compare to Unigram?"
    system, messages, max_tokens = llm.calls[0]
    sent = messages[0].content
    assert system == "Rewrite it." and max_tokens == 20
    assert "What is BPE?" not in sent  # only the last 2 exchanges
    assert "Provilkov " in sent and "Provilkov et al." not in sent  # cut to 10 chars
    assert sent.endswith("Latest message: How does it compare to Unigram?\n\nStandalone query:")


def test_empty_reply_falls_back_to_the_raw_message(prompt) -> None:
    result = rewrite(ScriptedLLM(lambda s, m: "  \n"), prompt, HISTORY, "And then?", CONFIG)
    assert (result.query, result.rewritten, result.fallback) == ("And then?", False, True)


def test_llm_error_falls_back(prompt) -> None:
    def fail(s, m):
        raise LLMError("down")

    result = rewrite(ScriptedLLM(fail), prompt, HISTORY, "And then?", CONFIG)
    assert result.fallback and result.error == "down"


def test_disabled_rewrite_passes_through(prompt) -> None:
    off = CONFIG.model_copy(update={"enabled": False})
    assert rewrite(ScriptedLLM(lambda s, m: "x"), prompt, HISTORY, "And?", off).query == "And?"


@pytest.mark.parametrize(
    ("raw", "query"),
    [
        ('"BPE vs Unigram"', "BPE vs Unigram"),
        ("Standalone query: BPE vs Unigram\nexplanation", "BPE vs Unigram"),
        ("  query: “BPE”  ", "BPE"),
    ],
)
def test_clean_strips_echoes_and_quotes(raw: str, query: str) -> None:
    assert clean(raw) == query


def test_prompt_hash_and_missing_file(tmp_path: Path, prompt) -> None:
    assert len(prompt.sha256) == 64 and prompt.text == "Rewrite it."
    with pytest.raises(FileNotFoundError, match="chat.prompts"):
        load_prompt(tmp_path, "missing.md")
