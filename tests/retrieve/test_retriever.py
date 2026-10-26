import json
from pathlib import Path

import pytest

from rag.common.config import load_config
from rag.common.io import write_jsonl
from rag.index import pipeline
from rag.parse.pipeline import CHUNKS, step_dir
from rag.retrieve.retriever import IndexUnavailable, Retriever, bm25_dir, dense_dir
from tests.retrieve.conftest import FakeEncoder, make_chunk

CHUNKS_DATA = [
    make_chunk("p:a", 0, "merge merge operations", header="BPE > Method"),
    make_chunk("p:a", 1, "byte level models", header="BPE > Bytes"),
    make_chunk("p:b", 0, "unigram language model", header="LM > Method"),
    make_chunk("p:c", 0, "merge byte pairs", header="Other > Intro"),
]


@pytest.fixture
def config(tmp_path: Path, minimal_config_text: str):
    path = tmp_path / "config.yaml"
    path.write_text(minimal_config_text)
    config = load_config(path)
    write_jsonl(step_dir(config, "chunk") / CHUNKS, CHUNKS_DATA)
    return config


@pytest.fixture
def built(config):
    pipeline.bm25(config)
    pipeline.dense(config, encoder=FakeEncoder())
    return config


def ids(hits) -> list[str]:
    return [h.chunk.chunk_id for h in hits]


def test_index_folders_have_meta_and_stats(built) -> None:
    for folder in (bm25_dir(built), dense_dir(built)):
        meta = json.loads((folder / "meta.json").read_text())
        assert meta["n_chunks"] == 4 and meta["limit"] is None
        assert (folder / "stats.json").is_file()
    assert dense_dir(built).name == "fake"


def test_modes_rank_and_report_every_retriever(built) -> None:
    r = Retriever.load(built, {"hybrid"}, encoder=FakeEncoder())
    bm25_hits = r.retrieve("merge", k=3, mode="bm25")
    assert ids(bm25_hits) == ["p:a#0000", "p:c#0000"]  # chunks without "merge" aren't candidates
    hybrid = r.retrieve("merge", k=2, mode="hybrid")
    assert hybrid[0].chunk.chunk_id == "p:a#0000"
    assert set(hybrid[0].ranks) == {"bm25", "dense"}
    assert hybrid[0].score == pytest.approx(2 / 61)


def test_allow_filters_chunks(built) -> None:
    r = Retriever.load(built, {"hybrid"}, encoder=FakeEncoder())
    hits = r.retrieve("merge", k=3, mode="hybrid", allow=lambda c: c.record_id != "p:a")
    assert all(h.chunk.record_id != "p:a" for h in hits)


def test_stale_index_is_refused(built) -> None:
    write_jsonl(step_dir(built, "chunk") / CHUNKS, CHUNKS_DATA[:3])
    with pytest.raises(IndexUnavailable, match="chunks changed"):
        Retriever.load(built, {"bm25"})


def test_changed_settings_are_refused(built) -> None:
    changed = built.model_copy(
        update={
            "index": built.index.model_copy(
                update={"bm25": built.index.bm25.model_copy(update={"k1": 0.9})}
            )
        }
    )
    with pytest.raises(IndexUnavailable, match="other settings"):
        Retriever.load(changed, {"bm25"})


def test_missing_index_is_refused(config) -> None:
    with pytest.raises(IndexUnavailable, match="no index"):
        Retriever.load(config, {"bm25"})


def test_mode_without_its_index_is_refused(built) -> None:
    r = Retriever.load(built, {"bm25"})
    with pytest.raises(IndexUnavailable, match="needs"):
        r.retrieve("merge", mode="dense")


def test_limited_index_covers_a_subset_and_warns(config, caplog) -> None:
    pipeline.bm25(config, limit=1)
    with caplog.at_level("WARNING"):
        r = Retriever.load(config, {"bm25"})
    assert {c.record_id for c in r.chunks} == {"p:a"}
    assert "--limit 1" in caplog.text


def test_search_reports_best_scores_over_all_chunks(built) -> None:
    r = Retriever.load(built, {"hybrid"}, encoder=FakeEncoder())
    result = r.search("unigram", k=1, mode="bm25")
    assert [h.chunk.chunk_id for h in result.hits] == ["p:b#0000"]
    # Dense's best is reported even in bm25 mode, and is the max over every chunk.
    dense_all = r.dense.scores(FakeEncoder().encode_query("unigram"))
    assert result.top_scores["dense"] == pytest.approx(float(dense_all.max()))
    assert result.top_scores["bm25"] == pytest.approx(result.hits[0].scores["bm25"])


def test_search_top_scores_respect_the_filter(built) -> None:
    r = Retriever.load(built, {"hybrid"}, encoder=FakeEncoder())
    full = r.search("unigram", mode="dense").top_scores["dense"]
    filtered = r.search("unigram", mode="dense", allow=lambda c: c.record_id != "p:b")
    assert filtered.top_scores["dense"] < full
