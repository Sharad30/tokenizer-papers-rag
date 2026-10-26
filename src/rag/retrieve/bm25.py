"""The BM25 index: bm25s does the sparse scoring, our code does the rest (D-026)."""

import json
from pathlib import Path

import bm25s
import numpy as np

from rag.index.config import Bm25Config
from rag.parse.schemas import Chunk
from rag.retrieve.text import STOPWORDS, chunk_text, terms

CHUNK_IDS = "chunk_ids.json"


class Bm25Index:
    def __init__(self, model: bm25s.BM25, chunk_ids: list[str], config: Bm25Config) -> None:
        self.model = model
        self.chunk_ids = chunk_ids
        self.config = config

    @classmethod
    def build(cls, chunks: list[Chunk], config: Bm25Config) -> "Bm25Index":
        model = bm25s.BM25(method=config.method, k1=config.k1, b=config.b)
        model.index(
            [cls._terms(chunk_text(c, config.include_header), config) for c in chunks],
            show_progress=False,
        )
        return cls(model, [c.chunk_id for c in chunks], config)

    def scores(self, query: str) -> np.ndarray:
        """A BM25 score for every chunk, in index order. Unknown terms add nothing."""
        query_terms = [t for t in self._terms(query, self.config) if t in self.model.vocab_dict]
        if not query_terms:
            return np.zeros(len(self.chunk_ids), dtype=np.float32)
        return self.model.get_scores(query_terms)

    def save(self, out_dir: Path) -> None:
        self.model.save(str(out_dir))
        (out_dir / CHUNK_IDS).write_text(json.dumps(self.chunk_ids), encoding="utf-8")

    @classmethod
    def load(cls, index_dir: Path, config: Bm25Config) -> "Bm25Index":
        model = bm25s.BM25.load(str(index_dir))
        chunk_ids = json.loads((index_dir / CHUNK_IDS).read_text(encoding="utf-8"))
        return cls(model, chunk_ids, config)

    @staticmethod
    def _terms(text: str, config: Bm25Config) -> list[str]:
        return terms(text, STOPWORDS[config.stopwords], config.split_compounds)
