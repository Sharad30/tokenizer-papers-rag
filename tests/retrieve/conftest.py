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
