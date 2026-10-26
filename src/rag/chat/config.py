"""Config models for the chat (the `chat:` section of config.yaml)."""

from typing import Literal

from pydantic import Field

from rag.common.schemas import StrictModel
from rag.retrieve.config import Mode

Provider = Literal["ollama", "anthropic"]


class OllamaConfig(StrictModel):
    url: str
    model: str
    think: bool  # qwen3 is a thinking model; off for short grounded answers
    temperature: float = Field(ge=0)
    num_ctx: int = Field(ge=1024)  # Ollama's context window; must hold prompt + context + answer
    timeout_s: float = Field(gt=0)


class AnthropicConfig(StrictModel):
    model: str
    api_key_env: str  # the key itself never goes in config
    timeout_s: float = Field(gt=0)


class PromptsConfig(StrictModel):
    # File names under paths.prompts. Prompts are files, not config strings,
    # so they can be long, diffed and edited by hand.
    answer: str
    rewrite: str


class RewriteConfig(StrictModel):
    enabled: bool
    history_turns: int = Field(ge=1)  # exchanges the rewriter sees (D-033)
    max_tokens: int = Field(ge=8)


class ChatRetrievalConfig(StrictModel):
    mode: Mode
    k: int = Field(ge=1)  # chunks given to the model as context (D-034)


class AbstainConfig(StrictModel):
    # Gate 1 (D-036): below this top dense cosine, don't call the LLM at all.
    min_dense_cosine: float = Field(ge=-1, le=1)
    not_supported_token: str  # gate 2: the model's exact "can't answer" reply


class ChatConfig(StrictModel):
    provider: Provider
    ollama: OllamaConfig
    anthropic: AnthropicConfig
    max_answer_tokens: int = Field(ge=16)
    prompts: PromptsConfig
    rewrite: RewriteConfig
    retrieval: ChatRetrievalConfig
    abstain: AbstainConfig
