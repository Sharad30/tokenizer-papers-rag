"""One chat turn, end to end.

    user text
      -> rewrite (turn 2 onwards)                     D-033
      -> retrieve (hybrid, top k + best cosine)
      -> gate 1: best dense cosine too low? abstain   D-036
      -> context [C1]..[Cn], best first               D-034
      -> generate from context only
      -> gate 2: model said NOT_SUPPORTED? abstain    D-036
      -> citations checked and renumbered by code     D-035

Everything that happened is returned in a TurnResult, which the CLI prints
and the transcript stores.
"""

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from rag.chat.citations import CitedAnswer, resolve
from rag.chat.config import ChatConfig
from rag.chat.context import ContextChunk, build_context, render_context, source_line
from rag.chat.llm import LLM, Completion, Message
from rag.chat.prompts import Prompt, answer_message
from rag.chat.rewrite import Exchange, RewriteResult, rewrite
from rag.collect.schemas import ManifestRecord
from rag.parse.schemas import Chunk
from rag.retrieve.retriever import Retriever
from rag.retrieve.schemas import Retrieval

Abstention = Literal["retrieval_gate", "model"]
NEAREST_SHOWN = 3  # titles listed when abstaining, as "nearest, not supporting"


@dataclass
class TurnResult:
    turn: int
    user: str
    rewrite: RewriteResult
    retrieval: Retrieval
    context: list[ContextChunk]
    abstained: Abstention | None = None
    completion: Completion | None = None
    cited: CitedAnswer | None = None
    reply: str = ""  # what the user sees, without the Sources list
    sources: list[tuple[int, ContextChunk]] = field(default_factory=list)
    seconds: dict[str, float] = field(default_factory=dict)

    @property
    def top_dense_cosine(self) -> float | None:
        return self.retrieval.top_scores.get("dense")


class ChatEngine:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        records: dict[str, ManifestRecord],
        config: ChatConfig,
        answer_prompt: Prompt,
        rewrite_prompt: Prompt,
        allow: Callable[[Chunk], bool] | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.records = records
        self.config = config
        self.answer_prompt = answer_prompt
        self.rewrite_prompt = rewrite_prompt
        self.allow = allow
        self.history: list[Exchange] = []

    def reset(self) -> None:
        """Start a new conversation: follow-ups no longer see earlier turns."""
        self.history = []

    def turn(self, user_text: str) -> TurnResult:
        timings: dict[str, float] = {}
        tick = time.monotonic()
        rewritten = rewrite(
            self.llm, self.rewrite_prompt, self.history, user_text, self.config.rewrite
        )
        timings["rewrite"] = time.monotonic() - tick

        tick = time.monotonic()
        cfg = self.config.retrieval
        retrieval = self.retriever.search(rewritten.query, cfg.k, cfg.mode, self.allow)
        timings["retrieve"] = time.monotonic() - tick
        context = build_context(retrieval.hits, self.records)
        result = TurnResult(len(self.history) + 1, user_text, rewritten, retrieval, context)
        result.seconds = timings

        top = result.top_dense_cosine
        if top is None or top < self.config.abstain.min_dense_cosine:
            result.abstained = "retrieval_gate"
            result.reply = self._gate_message(top, context)
        else:
            self._answer(result, rewritten.query)
        self.history.append(Exchange(user_text, result.reply))
        return result

    def _answer(self, result: TurnResult, question: str) -> None:
        tick = time.monotonic()
        message = answer_message(question, render_context(result.context))
        completion = self.llm.complete(
            self.answer_prompt.text, [Message("user", message)], self.config.max_answer_tokens
        )
        result.seconds["generate"] = time.monotonic() - tick
        result.completion = completion
        if self._is_not_supported(completion.text):
            result.abstained = "model"
            result.reply = (
                "The retrieved passages don't support an answer to this."
                + self._nearest(result.context)
            )
            return
        cited = resolve(completion.text, {cc.label for cc in result.context})
        by_label = {cc.label: cc for cc in result.context}
        result.cited = cited
        result.reply = cited.text
        result.sources = [(n, by_label[label]) for n, label in enumerate(cited.used, start=1)]

    def _is_not_supported(self, text: str) -> bool:
        # The prompt asks for the token alone, but a small model often explains
        # first ("The passages don't mention X. NOT_SUPPORTED."). Anywhere counts.
        return (
            re.search(rf"\b{re.escape(self.config.abstain.not_supported_token)}\b", text)
            is not None
        )

    def _gate_message(self, top: float | None, context: list[ContextChunk]) -> str:
        threshold = self.config.abstain.min_dense_cosine
        score = f"{top:.2f}" if top is not None else "n/a"
        return (
            "The corpus doesn't seem to cover this: the closest passage scores "
            f"{score}, below the {threshold:.2f} needed to attempt an answer."
            + self._nearest(context)
        )

    @staticmethod
    def _nearest(context: list[ContextChunk]) -> str:
        titles = list(dict.fromkeys(f"{cc.title}" for cc in context))[:NEAREST_SHOWN]
        if not titles:
            return ""
        return "\nNearest papers (not supporting an answer): " + "; ".join(titles)


def format_sources(result: TurnResult) -> list[str]:
    return [f"[{n}] {source_line(cc)}" for n, cc in result.sources]
