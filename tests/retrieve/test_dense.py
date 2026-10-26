from pathlib import Path

import numpy as np

from rag.index.config import DenseConfig
from rag.retrieve.dense import DenseIndex, model_slug
from rag.retrieve.ranking import top_k
from tests.retrieve.conftest import FakeEncoder, make_chunk

CHUNKS = [
    make_chunk("p:a", 0, "merge merge", header="BPE > Method"),
    make_chunk("p:a", 1, "byte level", header="BPE > Bytes"),
    make_chunk("p:b", 0, "unigram", header="LM > Method"),
]


def test_ranks_by_cosine(dense_config: DenseConfig, fake_encoder: FakeEncoder) -> None:
    index = DenseIndex.build(CHUNKS, dense_config, fake_encoder)
    scores = index.scores(fake_encoder.encode_query("byte level"))
    assert top_k(scores, 3)[0] == 1
    assert np.allclose(np.linalg.norm(index.embeddings, axis=1), 1)


def test_header_prepended_to_documents_only(
    dense_config: DenseConfig, fake_encoder: FakeEncoder
) -> None:
    DenseIndex.build(CHUNKS[:1], dense_config, fake_encoder)
    assert fake_encoder.seen_documents == ["BPE > Method\nmerge merge"]
    no_header = dense_config.model_copy(update={"prepend_header": False})
    DenseIndex.build(CHUNKS[:1], no_header, fake_encoder)
    assert fake_encoder.seen_documents[-1] == "merge merge"


def test_prompt_only_on_queries(dense_config: DenseConfig, fake_encoder: FakeEncoder) -> None:
    DenseIndex.build(CHUNKS, dense_config, fake_encoder)
    fake_encoder.encode_query("merge")
    assert not any(d.startswith("query: ") for d in fake_encoder.seen_documents)
    assert fake_encoder.seen_queries == ["query: merge"]


def test_save_and_load(
    tmp_path: Path, dense_config: DenseConfig, fake_encoder: FakeEncoder
) -> None:
    index = DenseIndex.build(CHUNKS, dense_config, fake_encoder)
    index.save(tmp_path)
    loaded = DenseIndex.load(tmp_path)
    assert loaded.chunk_ids == index.chunk_ids
    assert np.array_equal(loaded.embeddings, index.embeddings)


def test_model_slug() -> None:
    assert model_slug("BAAI/bge-small-en-v1.5") == "bge-small-en-v1.5"
