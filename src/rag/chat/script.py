"""A scripted conversation, written up as a readable transcript (Phase 8).

`rag eval transcript --turns <file>` runs the turns through the deployed
chat (both gates on, as a user would meet it) and writes Markdown: each turn's
message, the standalone query it was rewritten to, the best dense cosine,
the reply with its sources, and why it abstained when it did. The turns file
is plain text, one message per line; lines starting with "#" are comments.
"""

from collections.abc import Sequence
from pathlib import Path

from rag.chat.engine import ChatEngine, TurnResult, format_sources

ABSTAINED = {
    "retrieval_gate": "abstained at the retrieval gate (best passage below the threshold)",
    "model": "abstained: the model found the passages don't support an answer",
}


def read_turns(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def run_script(engine: ChatEngine, turns: Sequence[str]) -> list[TurnResult]:
    engine.reset()
    return [engine.turn(text) for text in turns]


def markdown(results: Sequence[TurnResult], header: Sequence[str]) -> str:
    lines = ["# Phase 8 transcript", "", *header, ""]
    for r in results:
        lines += [f"## Turn {r.turn}", "", f"**You:** {r.user}", ""]
        if r.rewrite.rewritten and r.rewrite.query != r.user:
            lines += [f"*Rewritten for retrieval:* {r.rewrite.query}", ""]
        cosine = r.top_dense_cosine
        lines.append(f"*Best dense cosine:* {'n/a' if cosine is None else f'{cosine:.3f}'}")
        if r.abstained:
            lines.append(f" · *{ABSTAINED[r.abstained]}*")
        lines += ["", "**Chat:**", "", r.reply, ""]
        sources = format_sources(r)
        if sources:
            lines += ["Sources:", "", *[f"- {s}" for s in sources], ""]
    return "\n".join(lines) + "\n"
