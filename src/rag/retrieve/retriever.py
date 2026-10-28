"""One interface over BM25, dense and hybrid retrieval.

`retrieve(query, k, mode, allow)` returns ranked chunks with every
retriever's rank and score, so the eval harness, `rag inspect query` and
the Phase 7 chat all see the same results. With `retrieve.dedup.enabled`,
duplicates are handled between ranking and the top-k cut (`collapse`, D-041).
"""

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np

from rag.common.config import Config
from rag.common.io import file_sha256, read_jsonl
from rag.parse.pipeline import CHUNKS, step_dir
from rag.parse.schemas import Chunk
from rag.retrieve.bm25 import Bm25Index
from rag.retrieve.collapse import WorkMap, WorksUnavailable, collapse
from rag.retrieve.config import DedupSettings, Mode, RetrieveConfig
from rag.retrieve.dense import DenseIndex, Encoder, SentenceTransformerEncoder, model_slug
from rag.retrieve.fusion import rrf
from rag.retrieve.ranking import top_k
from rag.retrieve.schemas import Hit, IndexMeta, Retrieval

log = logging.getLogger(__name__)

META = "meta.json"
NEEDS: dict[str, set[str]] = {"bm25": {"bm25"}, "dense": {"dense"}, "hybrid": {"bm25", "dense"}}


class IndexUnavailable(RuntimeError):
    """An index is missing, stale, or built with other settings. Rebuild with `rag index`."""


def bm25_dir(config: Config) -> Path:
    return config.paths.index / "bm25"


def dense_dir(config: Config) -> Path:
    return config.paths.index / "dense" / model_slug(config.index.dense.model)


class Retriever:
    def __init__(
        self,
        chunks: list[Chunk],
        config: RetrieveConfig,
        bm25: Bm25Index | None = None,
        dense: DenseIndex | None = None,
        encoder: Encoder | None = None,
        works: WorkMap | None = None,
    ) -> None:
        self.chunks = chunks
        self.config = config
        self.bm25 = bm25
        self.dense = dense
        self.encoder = encoder
        for index in (bm25, dense):
            if index is not None and index.chunk_ids != [c.chunk_id for c in chunks]:
                raise IndexUnavailable("index rows don't line up with the chunks: run `rag index`")
        if config.dedup.enabled and works is None:
            raise WorksUnavailable("retrieve.dedup.enabled needs the works: run `rag dedup run`")
        self.works = works if config.dedup.enabled else None
        # Fixed per corpus, so computed once rather than per query.
        self._canonical = works.canonical_mask(chunks) if self.works is not None else None

    def with_dedup(self, settings: DedupSettings, works: WorkMap | None) -> "Retriever":
        """The same indexes with other duplicate handling (for the Phase 5 ablation)."""
        config = self.config.model_copy(update={"dedup": settings})
        return Retriever(self.chunks, config, self.bm25, self.dense, self.encoder, works)

    @classmethod
    def load(cls, config: Config, modes: set[str], encoder: Encoder | None = None) -> "Retriever":
        """Load the indexes `modes` need, after checking each against chunks.jsonl and config."""
        chunks_path = step_dir(config, "chunk") / CHUNKS
        sha = file_sha256(chunks_path)
        needed = set().union(*(NEEDS[m] for m in modes))
        bm25 = dense = None
        if "bm25" in needed:
            _check(bm25_dir(config), sha, config.index.bm25.model_dump(mode="json"))
            bm25 = Bm25Index.load(bm25_dir(config), config.index.bm25)
        if "dense" in needed:
            _check(dense_dir(config), sha, config.index.dense.model_dump(mode="json"))
            dense = DenseIndex.load(dense_dir(config))
            encoder = encoder or SentenceTransformerEncoder(config.index.dense)
        # A --limit index covers a subset: keep exactly its chunks, in its order.
        by_id = {c.chunk_id: c for c in read_jsonl(chunks_path, Chunk)}
        ids = (bm25 or dense).chunk_ids
        works = (
            WorkMap.load(config.paths.works, chunks_path) if config.retrieve.dedup.enabled else None
        )
        return cls([by_id[i] for i in ids], config.retrieve, bm25, dense, encoder, works)

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        mode: Mode | None = None,
        allow: Callable[[Chunk], bool] | None = None,
    ) -> list[Hit]:
        return self.search(query, k, mode, allow).hits

    def search(
        self,
        query: str,
        k: int | None = None,
        mode: Mode | None = None,
        allow: Callable[[Chunk], bool] | None = None,
    ) -> Retrieval:
        """Ranked hits plus each retriever's best score over all allowed chunks."""
        k = k or self.config.default_k
        mode = mode or self.config.default_mode
        if not NEEDS[mode] <= self._available():
            raise IndexUnavailable(f"mode {mode!r} needs the {sorted(NEEDS[mode])} index(es)")
        mask = np.array([allow(c) for c in self.chunks]) if allow else None
        dedup = self.config.dedup
        if self._canonical is not None and dedup.canonical_only:
            mask = self._canonical if mask is None else mask & self._canonical
        depth = max(k, self.config.candidates)
        lists: dict[str, list[int]] = {}
        scores: dict[str, np.ndarray] = {}
        if self.bm25 is not None:
            scores["bm25"] = self.bm25.scores(query)
            # Score 0 means no query term at all: not a candidate.
            lists["bm25"] = top_k(scores["bm25"], depth, mask, positive_only=True).tolist()
        if self.dense is not None and self.encoder is not None:
            scores["dense"] = self.dense.scores(self.encoder.encode_query(query))
            lists["dense"] = top_k(scores["dense"], depth, mask).tolist()

        if mode == "hybrid":
            ranked = rrf(lists, self.config.rrf_k)
        else:
            ranked = [(i, float(scores[mode][i])) for i in lists[mode]]
        hidden: dict[int, list[int]] = {}
        if self.works is not None:
            vectors = self.dense.embeddings if self.dense is not None else None
            ranked, hidden = collapse(ranked, self.chunks, self.works, dedup, k, vectors)
        hits = [
            self._hit(i, rank, score, lists, scores, hidden.get(i, []))
            for rank, (i, score) in enumerate(ranked[:k], 1)
        ]
        # Each list is already sorted, so its first entry is that retriever's best.
        top = {name: float(scores[name][ids[0]]) for name, ids in lists.items() if ids}
        return Retrieval(hits=hits, top_scores=top)

    def _available(self) -> set[str]:
        have = set()
        if self.bm25 is not None:
            have.add("bm25")
        if self.dense is not None and self.encoder is not None:
            have.add("dense")
        return have

    def _hit(self, i: int, rank: int, score: float, lists, scores, hidden: list[int]) -> Hit:
        positions = {name: ranked.index(i) + 1 for name, ranked in lists.items() if i in ranked}
        return Hit(
            chunk=self.chunks[i],
            rank=rank,
            score=score,
            ranks=positions,
            scores={name: float(s[i]) for name, s in scores.items()},
            work_id=self.works.work_of(self.chunks[i]) if self.works is not None else None,
            hidden=[self.chunks[j].chunk_id for j in hidden],
        )


def read_meta(index_dir: Path) -> IndexMeta:
    path = index_dir / META
    if not path.is_file():
        raise IndexUnavailable(f"no index at {index_dir}: run `rag index`")
    return IndexMeta.model_validate_json(path.read_text(encoding="utf-8"))


def _check(index_dir: Path, chunks_sha: str, settings: dict) -> IndexMeta:
    meta = read_meta(index_dir)
    if meta.chunks_sha256 != chunks_sha:
        raise IndexUnavailable(f"chunks changed since {index_dir} was built: run `rag index`")
    if meta.settings != settings:
        raise IndexUnavailable(f"{index_dir} was built with other settings: run `rag index`")
    if meta.limit is not None:
        log.warning("%s was built with --limit %d: results cover a subset", index_dir, meta.limit)
    return meta
