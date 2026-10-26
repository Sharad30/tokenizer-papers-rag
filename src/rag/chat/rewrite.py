"""Follow-up rewriting: the latest message plus recent history -> one standalone query (D-033).

"How does it compare to Unigram?" means nothing to a retriever. The rewrite
turns it into "How does BPE-dropout compare to Unigram LM subword
regularization?" using the last few exchanges. It's skipped on the first
turn, and any failure falls back to the raw message (and is flagged), so a
broken rewrite can't silence the chat.
"""

from dataclasses import dataclass

from rag.chat.config import RewriteConfig
from rag.chat.llm import LLM, Completion, LLMError, Message
from rag.chat.prompts import Prompt, rewrite_message

# Small models sometimes echo a label or wrap the query in quotes.
_ECHOED_PREFIXES = ("standalone query:", "query:", "rewritten query:")
_QUOTES = "\"'“”‘’`"


@dataclass(frozen=True)
class Exchange:
    user: str
    assistant: str


@dataclass(frozen=True)
class RewriteResult:
    query: str
    rewritten: bool  # an LLM rewrite was used
    fallback: bool  # a rewrite was attempted and failed: the raw message was used
    completion: Completion | None = None
    error: str | None = None


def rewrite(
    llm: LLM,
    prompt: Prompt,
    history: list[Exchange],
    message: str,
    config: RewriteConfig,
) -> RewriteResult:
    if not config.enabled or not history:
        return RewriteResult(message, rewritten=False, fallback=False)
    recent = [
        (e.user, e.assistant[: config.max_answer_chars]) for e in history[-config.history_turns :]
    ]
    try:
        completion = llm.complete(
            prompt.text, [Message("user", rewrite_message(recent, message))], config.max_tokens
        )
    except LLMError as exc:
        return RewriteResult(message, rewritten=False, fallback=True, error=str(exc))
    query = clean(completion.text)
    if not query:
        return RewriteResult(message, rewritten=False, fallback=True, completion=completion)
    return RewriteResult(query, rewritten=True, fallback=False, completion=completion)


def clean(text: str) -> str:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    line = lines[0]
    for prefix in _ECHOED_PREFIXES:
        if line.lower().startswith(prefix):
            line = line[len(prefix) :].strip()
    return line.strip(_QUOTES).strip()
