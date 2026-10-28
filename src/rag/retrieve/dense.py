"""The dense index: one normalised embedding per chunk, exact search (D-028).

With unit-length vectors, cosine similarity is a dot product, so scoring
every chunk against a query is one matrix-vector product: 93k x 384 floats,
a few milliseconds. No approximate index is needed at this scale.
"""

import json
import logging
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from rag.index.config import DenseConfig
from rag.parse.schemas import Chunk
from rag.retrieve.text import chunk_text

log = logging.getLogger(__name__)

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
        try:
            # The cached copy first: otherwise every load asks the HF Hub for
            # metadata, an external call per `rag inspect query` or chat start.
            self._model = SentenceTransformer(
                config.model, device=config.device, local_files_only=True
            )
        except OSError:  # not downloaded yet: fetch it once
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


class Specter2Encoder:
    """SPECTER2 with its ad-hoc query adapter for queries and the proximity
    adapter for documents, run as a separate process (`specter2_worker.py`)
    in an isolated environment (D-043).

    Starting that process takes ~30 s, so queries are encoded in batches
    with `encode_queries` and cached; `encode_query` on an unseen query
    starts the process for that one query.
    """

    def __init__(
        self, config: DenseConfig, run: Callable[[list[str], list[str]], None] | None = None
    ) -> None:
        if not (config.query_adapter and config.document_adapter):
            raise ValueError("specter2 needs query_adapter and document_adapter")
        self._config = config
        self._run = run or _run_isolated
        self._cache: dict[str, np.ndarray] = {}
        self.encode_ms: dict[str, float] = {}  # per query, measured inside the worker
        self.truncated = 0  # documents longer than max_length, in the last encode_documents

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        vectors, report = self._call(texts, self._config.document_adapter, one_at_a_time=False)
        self.truncated = report["truncated"]
        if self.truncated:
            log.warning("specter2: %d of %d documents truncated", self.truncated, len(texts))
        return vectors

    def encode_queries(self, texts: Sequence[str]) -> None:
        """Encode queries in one process; each is timed on its own (latency)."""
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if not todo:
            return
        vectors, report = self._call(todo, self._config.query_adapter, one_at_a_time=True)
        for text, vector, ms in zip(todo, vectors, report["encode_ms"], strict=True):
            self._cache[text] = vector
            self.encode_ms[text] = ms

    def encode_query(self, text: str) -> np.ndarray:
        if text not in self._cache:
            self.encode_queries([text])
        return self._cache[text]

    def _call(self, texts: list[str], adapter: str | None, one_at_a_time: bool):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "in.jsonl").write_text(
                "".join(json.dumps({"text": t}) + "\n" for t in texts), encoding="utf-8"
            )
            args = [
                "--model", self._config.model,
                "--adapter", str(adapter),
                "--input", str(folder / "in.jsonl"),
                "--output", str(folder / "out.npy"),
                "--report", str(folder / "report.json"),
                "--max-length", str(self._config.max_length or 512),
                "--batch-size", str(self._config.batch_size),
            ]  # fmt: skip
            if one_at_a_time:
                args.append("--one-at-a-time")
            self._run(self._config.isolated_with, args)
            vectors = np.load(folder / "out.npy")
            report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        # Unit length, so scoring stays a dot product (D-028). The model card
        # ranks by L2 distance; the Phase 6 probe measures the difference.
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors.astype(np.float32), report


WORKER = Path(__file__).with_name("specter2_worker.py")


def _run_isolated(packages: list[str], args: list[str]) -> None:
    command = ["uv", "run", "--isolated", "--no-project"]
    for package in packages:
        command += ["--with", package]
    subprocess.run([*command, "python", str(WORKER), *args], check=True)


def make_encoder(config: DenseConfig, show_progress: bool = False) -> Encoder:
    if config.kind == "specter2":
        return Specter2Encoder(config)
    return SentenceTransformerEncoder(config, show_progress=show_progress)


def model_slug(model: str) -> str:
    """Index folder name per model, so Phase 6 models sit side by side."""
    return model.rsplit("/", 1)[-1]


class DenseIndex:
    def __init__(self, embeddings: np.ndarray, chunk_ids: list[str]) -> None:
        self.embeddings = embeddings
        self.chunk_ids = chunk_ids

    @classmethod
    def build(cls, chunks: list[Chunk], config: DenseConfig, encoder: Encoder) -> "DenseIndex":
        texts = [chunk_text(c, config.prepend_header, config.header_separator) for c in chunks]
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
