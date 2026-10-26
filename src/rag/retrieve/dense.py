"""The dense index: one normalised embedding per chunk, exact search (D-028).

With unit-length vectors, cosine similarity is a dot product, so scoring
every chunk against a query is one matrix-vector product: 93k x 384 floats,
a few milliseconds. No approximate index is needed at this scale.
"""

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from rag.index.config import DenseConfig
from rag.parse.schemas import Chunk
from rag.retrieve.text import chunk_text

EMBEDDINGS = "embeddings.npy"
CHUNK_IDS = "chunk_ids.json"


class Encoder(Protocol):
    """Text -> unit-length float32 vectors. Tests inject a fake one."""

    def encode_documents(self, texts: list[str]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEncoder:
    def __init__(self, config: DenseConfig, show_progress: bool = False) -> None:
        # Imported here: loading torch takes seconds, and BM25-only commands
        # and tests shouldn't pay for it.
        from sentence_transformers import SentenceTransformer

        self._config = config
        self._show_progress = show_progress
        self._model = SentenceTransformer(config.model, device=config.device)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        # No prompt: bge's instruction is for queries only (model card).
        return self._encode(texts, prompt=None)

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode([text], prompt=self._config.query_prompt)[0]

    def _encode(self, texts: list[str], prompt: str | None) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            prompt=prompt,
            batch_size=self._config.batch_size,
            normalize_embeddings=True,  # the model normalises too; this guards a swapped model
            convert_to_numpy=True,
            show_progress_bar=self._show_progress,
        )
        return vectors.astype(np.float32, copy=False)


def model_slug(model: str) -> str:
    """Index folder name per model, so Phase 6 models sit side by side."""
    return model.rsplit("/", 1)[-1]


class DenseIndex:
    def __init__(self, embeddings: np.ndarray, chunk_ids: list[str]) -> None:
        self.embeddings = embeddings
        self.chunk_ids = chunk_ids

    @classmethod
    def build(cls, chunks: list[Chunk], config: DenseConfig, encoder: Encoder) -> "DenseIndex":
        texts = [chunk_text(c, config.prepend_header) for c in chunks]
        return cls(encoder.encode_documents(texts), [c.chunk_id for c in chunks])

    def scores(self, query_vector: np.ndarray) -> np.ndarray:
        """Cosine similarity of every chunk to the query, in index order."""
        return self.embeddings @ query_vector

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / EMBEDDINGS, self.embeddings)
        (out_dir / CHUNK_IDS).write_text(json.dumps(self.chunk_ids), encoding="utf-8")

    @classmethod
    def load(cls, index_dir: Path) -> "DenseIndex":
        embeddings = np.load(index_dir / EMBEDDINGS)
        chunk_ids = json.loads((index_dir / CHUNK_IDS).read_text(encoding="utf-8"))
        return cls(embeddings, chunk_ids)
