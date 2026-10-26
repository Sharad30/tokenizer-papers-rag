"""Chat transcripts: one JSONL file per session in paths.chat.

The first line identifies the session (commit, config hash, model, prompt
hashes); each later line is one turn with everything needed to judge it in
Phase 8: the rewrite, every retrieved chunk with its scores, the gate
values, the raw model output, citations, and timings.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from rag.chat.engine import TurnResult
from rag.chat.prompts import Prompt


class Transcript:
    def __init__(self, out_dir: Path, started: datetime | None = None) -> None:
        self.started = started or datetime.now(UTC)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.path = out_dir / f"{self.started:%Y%m%dT%H%M%SZ}.jsonl"

    def write_header(
        self, git_commit: str, config_hash: str, model: str, prompts: list[Prompt], settings: dict
    ) -> None:
        self._append(
            {
                "type": "session",
                "started_at": self.started.isoformat(),
                "git_commit": git_commit,
                "config_hash": config_hash,
                "model": model,
                "prompts": {p.name: p.sha256 for p in prompts},
                "settings": settings,
            }
        )

    def write_reset(self) -> None:
        self._append({"type": "reset", "at": datetime.now(UTC).isoformat()})

    def write_turn(self, result: TurnResult) -> None:
        completion = result.completion
        self._append(
            {
                "type": "turn",
                "turn": result.turn,
                "user": result.user,
                "query": result.rewrite.query,
                "rewritten": result.rewrite.rewritten,
                "rewrite_fallback": result.rewrite.fallback,
                "top_scores": result.retrieval.top_scores,
                "hits": [
                    {
                        "label": cc.label,
                        "chunk_id": cc.hit.chunk.chunk_id,
                        "rank": cc.hit.rank,
                        "score": cc.hit.score,
                        "ranks": cc.hit.ranks,
                        "scores": cc.hit.scores,
                    }
                    for cc in result.context
                ],
                "abstained": result.abstained,
                "raw_answer": completion.text if completion else None,
                "model": completion.model if completion else None,
                "tokens": [completion.input_tokens, completion.output_tokens]
                if completion
                else None,
                "reply": result.reply,
                "sources": [
                    {"n": n, "chunk_id": cc.hit.chunk.chunk_id} for n, cc in result.sources
                ],
                "invalid_citations": result.cited.invalid if result.cited else [],
                "uncited_sentences": result.cited.uncited_sentences if result.cited else 0,
                "seconds": {k: round(v, 3) for k, v in result.seconds.items()},
            }
        )

    def _append(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
