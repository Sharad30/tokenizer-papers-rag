"""LLM clients behind one small interface (D-032).

`complete(system, messages, max_tokens)` is all the chat needs: one system
prompt, a list of user/assistant turns, one text reply. Ollama is called
over its HTTP API with httpx; Anthropic through its SDK. Tests use
`ScriptedLLM`, so nothing here is called over the network in tests.
"""

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

import httpx

from rag.chat.config import AnthropicConfig, ChatConfig, OllamaConfig, Provider

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    seconds: float


class LLMError(RuntimeError):
    """The LLM couldn't be reached or answered with an error."""


class LLM(Protocol):
    model: str

    def complete(self, system: str, messages: list[Message], max_tokens: int) -> Completion: ...


class OllamaClient:
    """POST /api/chat on a local Ollama server, non-streaming."""

    def __init__(self, config: OllamaConfig, http: httpx.Client | None = None) -> None:
        self.config = config
        self.model = config.model
        self._http = http or httpx.Client(timeout=config.timeout_s)

    def complete(self, system: str, messages: list[Message], max_tokens: int) -> Completion:
        body = {
            "model": self.config.model,
            "messages": [{"role": "system", "content": system}]
            + [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "think": self.config.think,
            "options": {
                "temperature": self.config.temperature,
                "num_ctx": self.config.num_ctx,
                "num_predict": max_tokens,
            },
        }
        tick = time.monotonic()
        try:
            response = self._http.post(f"{self.config.url}/api/chat", json=body)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(
                f"Ollama at {self.config.url} failed ({exc}). Is `ollama serve` running and "
                f"`{self.config.model}` pulled (`ollama pull {self.config.model}`)?"
            ) from exc
        data = response.json()
        return Completion(
            text=data["message"]["content"].strip(),
            model=data.get("model", self.config.model),
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
            seconds=time.monotonic() - tick,
        )


class AnthropicClient:
    """Claude through the anthropic SDK. The key is read from the environment only."""

    def __init__(self, config: AnthropicConfig, client=None) -> None:
        self.config = config
        self.model = config.model
        if client is None:
            key = os.environ.get(config.api_key_env)
            if not key:
                raise LLMError(
                    f"{config.api_key_env} is not set: add it to .env (see .env.example), "
                    "or use `--provider ollama`."
                )
            import anthropic  # imported here: only needed when Claude is chosen

            client = anthropic.Anthropic(api_key=key, timeout=config.timeout_s)
        self._client = client

    def complete(self, system: str, messages: list[Message], max_tokens: int) -> Completion:
        tick = time.monotonic()
        try:
            response = self._client.messages.create(
                model=self.config.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": m.role, "content": m.content} for m in messages],
            )
        except Exception as exc:  # the SDK raises its own hierarchy; surface it as one error
            raise LLMError(f"Anthropic API call failed: {exc}") from exc
        # Adaptive thinking can add thinking blocks; the answer is the text blocks.
        text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
        return Completion(
            text=text.strip(),
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            seconds=time.monotonic() - tick,
        )


class ScriptedLLM:
    """For tests: replies come from a function of (system, messages); calls are recorded."""

    def __init__(self, reply: Callable[[str, list[Message]], str], model: str = "scripted") -> None:
        self.reply = reply
        self.model = model
        self.calls: list[tuple[str, list[Message], int]] = []

    def complete(self, system: str, messages: list[Message], max_tokens: int) -> Completion:
        self.calls.append((system, messages, max_tokens))
        text = self.reply(system, messages)
        return Completion(text, self.model, None, None, 0.0)


def make_llm(config: ChatConfig, provider: Provider | None = None) -> LLM:
    chosen = provider or config.provider
    if chosen == "anthropic":
        return AnthropicClient(config.anthropic)
    return OllamaClient(config.ollama)
