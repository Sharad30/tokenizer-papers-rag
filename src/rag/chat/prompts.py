"""Prompt files and the user messages built around them.

The system prompts live in `paths.prompts` as plain files so they can be
read, diffed and edited by hand (Sharad's Phase 7 task). Each transcript
records their sha256, so an answer can be traced to the exact prompt.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Prompt:
    name: str
    text: str
    sha256: str


def load_prompt(prompts_dir: Path, filename: str) -> Prompt:
    path = prompts_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file missing: {path} (see chat.prompts in config.yaml)")
    text = path.read_text(encoding="utf-8").strip()
    return Prompt(filename, text, hashlib.sha256(text.encode("utf-8")).hexdigest())


def answer_message(question: str, context: str) -> str:
    """The single user message the generator sees: the context, then the question."""
    return f"Context passages:\n\n{context}\n\nQuestion: {question}"


def rewrite_message(history: list[tuple[str, str]], message: str) -> str:
    turns = "\n".join(f"User: {u}\nAssistant: {a}" for u, a in history)
    return f"Conversation so far:\n{turns}\n\nLatest message: {message}\n\nStandalone query:"
