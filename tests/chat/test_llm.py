import json
from types import SimpleNamespace

import httpx
import pytest
import respx

from rag.chat.config import AnthropicConfig, OllamaConfig
from rag.chat.llm import AnthropicClient, LLMError, Message, OllamaClient, ScriptedLLM

OLLAMA = OllamaConfig(
    url="http://ollama.test",
    model="qwen3:8b",
    think=False,
    temperature=0,
    num_ctx=4096,
    timeout_s=5,
)
CLAUDE = AnthropicConfig(model="claude-haiku-5-5", api_key_env="TEST_ANTHROPIC_KEY", timeout_s=5)
MESSAGES = [Message("user", "What is BPE?")]


@respx.mock
def test_ollama_request_body_and_reply() -> None:
    route = respx.post("http://ollama.test/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "qwen3:8b",
                "message": {"role": "assistant", "content": "  BPE merges pairs [C1].  "},
                "prompt_eval_count": 120,
                "eval_count": 9,
            },
        )
    )
    completion = OllamaClient(OLLAMA).complete("Be grounded.", MESSAGES, max_tokens=50)
    body = json.loads(route.calls[0].request.content)
    assert body["messages"][0] == {"role": "system", "content": "Be grounded."}
    assert body["messages"][1] == {"role": "user", "content": "What is BPE?"}
    assert body["stream"] is False and body["think"] is False
    assert body["options"] == {"temperature": 0, "num_ctx": 4096, "num_predict": 50}
    assert completion.text == "BPE merges pairs [C1]."
    assert (completion.input_tokens, completion.output_tokens) == (120, 9)


@respx.mock
def test_ollama_error_names_the_fix() -> None:
    respx.post("http://ollama.test/api/chat").mock(return_value=httpx.Response(404))
    with pytest.raises(LLMError, match="ollama pull qwen3:8b"):
        OllamaClient(OLLAMA).complete("s", MESSAGES, 10)


class FakeMessages:
    def __init__(self) -> None:
        self.kwargs: dict = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            model="claude-haiku-5-5",
            content=[
                SimpleNamespace(type="thinking", thinking="hmm"),
                SimpleNamespace(type="text", text="BPE merges pairs [C1]."),
            ],
            usage=SimpleNamespace(input_tokens=100, output_tokens=8),
        )


def test_anthropic_request_and_text_blocks_only() -> None:
    fake = SimpleNamespace(messages=FakeMessages())
    completion = AnthropicClient(CLAUDE, client=fake).complete("Be grounded.", MESSAGES, 50)
    assert fake.messages.kwargs == {
        "model": "claude-haiku-5-5",
        "max_tokens": 50,
        "system": "Be grounded.",
        "messages": [{"role": "user", "content": "What is BPE?"}],
    }
    assert completion.text == "BPE merges pairs [C1]."
    assert completion.output_tokens == 8


def test_anthropic_without_key_says_where_to_put_it(monkeypatch) -> None:
    monkeypatch.delenv("TEST_ANTHROPIC_KEY", raising=False)
    with pytest.raises(LLMError, match="TEST_ANTHROPIC_KEY is not set"):
        AnthropicClient(CLAUDE)


def test_scripted_llm_records_calls() -> None:
    llm = ScriptedLLM(lambda system, messages: messages[-1].content.upper())
    assert llm.complete("s", MESSAGES, 5).text == "WHAT IS BPE?"
    assert llm.calls[0][2] == 5
