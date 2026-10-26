import numpy as np
import pytest

from rag.index.config import Bm25Config, DenseConfig
from rag.parse.schemas import Chunk


def make_chunk(record_id: str, n: int, text: str, header: str = "") -> Chunk:
    return Chunk(
        chunk_id=f"{record_id}#{n:04d}",
        record_id=record_id,
        element_type="paragraph",
        header=header or f"Paper {record_id}",
        text=text,
        token_count=len(text.split()),
        header_token_count=2,
        parser="grobid",
    )


@pytest.fixture
def bm25_config() -> Bm25Config:
    return Bm25Config(
        method="lucene",
        k1=1.5,
        b=0.75,
        include_header=False,
        stopwords="en",
        split_compounds=True,
    )


@pytest.fixture
def dense_config() -> DenseConfig:
    return DenseConfig(
        model="test/fake",
        query_prompt="query: ",
        prepend_header=True,
        batch_size=2,
        device="cpu",
    )


class FakeEncoder:
    """Bag-of-words vectors over a fixed vocabulary, normalised: enough to rank.

    Records what it was asked to encode, so tests can check prompts and headers.
    """

    VOCAB = ["merge", "byte", "level", "unigram", "query:", "paper"]

    def __init__(self, prompt: str = "query: ") -> None:
        self.prompt = prompt
        self.seen_documents: list[str] = []
        self.seen_queries: list[str] = []

    def _vector(self, text: str) -> np.ndarray:
        words = text.lower().split()
        v = np.array([words.count(w) for w in self.VOCAB], dtype=np.float32) + 1e-3
        return v / np.linalg.norm(v)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        self.seen_documents.extend(texts)
        return np.stack([self._vector(t) for t in texts])

    def encode_query(self, text: str) -> np.ndarray:
        self.seen_queries.append(self.prompt + text)
        return self._vector(self.prompt + text)


@pytest.fixture
def fake_encoder() -> FakeEncoder:
    return FakeEncoder()
