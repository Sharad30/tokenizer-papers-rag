"""Cross-encoder reranking of the fused candidates (D-043).

A bi-encoder (the dense index) embeds query and chunk separately, so the
chunk vectors can be computed once, offline. A cross-encoder reads query
and chunk together through every layer, so it can see which query words
the chunk actually answers, but it must run once per (query, chunk) pair
at query time. Hence the two-stage pattern: cheap retrieval of the top
`depth`, then the expensive model reorders only those.
"""

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from rag.retrieve.config import RerankConfig


class Scorer(Protocol):
    """(query, passage) pairs -> relevance scores, higher is better. Tests inject a fake."""

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray: ...


class CrossEncoderScorer:
    def __init__(self, config: RerankConfig) -> None:
        from sentence_transformers import CrossEncoder  # torch loads slowly; only when used

        self._config = config
        try:  # the cached copy first, as for the dense encoder
            self._model = CrossEncoder(
                config.model, device=config.device, max_length=config.max_length,
                local_files_only=True,
            )  # fmt: skip
        except OSError:
            self._model = CrossEncoder(
                config.model, device=config.device, max_length=config.max_length
            )

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        # Raw logits (model card: unbounded); only their order matters here.
        return np.asarray(
            self._model.predict(
                [(query, p) for p in passages],
                batch_size=self._config.batch_size,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )


def rerank(
    ranked: list[tuple[int, float]], query: str, texts: Sequence[str], scorer: Scorer, depth: int
) -> list[tuple[int, float]]:
    """Reorder the top `depth` by cross-encoder score; the rest keep their order below."""
    head, tail = ranked[:depth], ranked[depth:]
    if not head:
        return ranked
    scores = scorer.score(query, [texts[i] for i, _ in head])
    order = sorted(range(len(head)), key=lambda j: (-float(scores[j]), j))  # ties: earlier rank
    return [(head[j][0], float(scores[j])) for j in order] + tail
