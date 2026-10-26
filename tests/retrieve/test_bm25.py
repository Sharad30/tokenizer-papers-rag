import math
from pathlib import Path

import pytest

from rag.index.config import Bm25Config
from rag.retrieve.bm25 import Bm25Index
from tests.retrieve.conftest import make_chunk

CHUNKS = [
    make_chunk("p:a", 0, "merge merge operations"),
    make_chunk("p:a", 1, "byte level models"),
    make_chunk("p:b", 0, "merge"),
]


def lucene(tf: int, df: int, n_docs: int, dl: int, avgdl: float, k1=1.5, b=0.75) -> float:
    idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
    return idf * tf / (tf + k1 * (1 - b + b * dl / avgdl))


def test_scores_match_the_lucene_formula(bm25_config: Bm25Config) -> None:
    index = Bm25Index.build(CHUNKS, bm25_config)
    scores = index.scores("merge")
    avgdl = (3 + 3 + 1) / 3
    assert scores[0] == pytest.approx(lucene(tf=2, df=2, n_docs=3, dl=3, avgdl=avgdl), rel=1e-5)
    assert scores[1] == 0
    assert scores[2] == pytest.approx(lucene(tf=1, df=2, n_docs=3, dl=1, avgdl=avgdl), rel=1e-5)


def test_compound_query_matches_parts(bm25_config: Bm25Config) -> None:
    index = Bm25Index.build(CHUNKS, bm25_config)
    assert index.scores("byte-level")[1] > 0


def test_unknown_terms_score_zero(bm25_config: Bm25Config) -> None:
    index = Bm25Index.build(CHUNKS, bm25_config)
    assert index.scores("klingon").tolist() == [0, 0, 0]
    assert index.scores("klingon merge")[2] > 0


def test_header_is_indexed_when_enabled(bm25_config: Bm25Config) -> None:
    chunks = [make_chunk("p:a", 0, "body text", header="Unigram > Method")]
    with_header = Bm25Index.build(chunks, bm25_config.model_copy(update={"include_header": True}))
    without = Bm25Index.build(chunks, bm25_config)
    assert with_header.scores("unigram")[0] > 0
    assert without.scores("unigram")[0] == 0


def test_save_and_load_round_trip(tmp_path: Path, bm25_config: Bm25Config) -> None:
    index = Bm25Index.build(CHUNKS, bm25_config)
    index.save(tmp_path)
    loaded = Bm25Index.load(tmp_path, bm25_config)
    assert loaded.chunk_ids == ["p:a#0000", "p:a#0001", "p:b#0000"]
    assert loaded.scores("merge").tolist() == index.scores("merge").tolist()
