import json
from pathlib import Path

import pytest

from rag.collect.manifest import write_manifest
from rag.collect.schemas import Decision, ManifestRecord, Source
from rag.common.config import load_config
from rag.common.io import write_jsonl
from rag.eval.results import RunResult
from rag.eval.run import run_eval
from rag.index import pipeline as index_pipeline
from rag.parse.pipeline import CHUNKS as CHUNKS_FILE
from rag.parse.pipeline import step_dir
from rag.retrieve.schemas import Hit
from tests.eval.conftest import CHUNKS
from tests.retrieve.conftest import FakeEncoder

FIXTURE = Path(__file__).parents[1] / "fixtures" / "eval" / "questions.jsonl"
BY_ID = {c.chunk_id: c for c in CHUNKS}

# Fixture config: recall_k 2, mrr_cutoff 3, dup_k 2.
RANKINGS = {
    "How many merge operations are learned?": ["p:a-copy#0000", "p:a#0000", "p:b#0000"],
    "Is that vocabulary shared?": ["p:b#0000", "p:b#0001", "p:a#0001"],
    "Is the BPE vocabulary joint across languages?": ["p:a#0001", "p:b#0000", "p:b#0001"],
    "How do subword and byte-level models handle rare words?": [
        "p:a#0001",
        "p:b#0000",
        "p:a#0000",
    ],
    "What BLEU does the byte-level model reach on Klingon?": ["p:b#0001", "p:a#0002"],
}


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, str]] = []

    def retrieve(self, query: str, k=None, mode=None) -> list[Hit]:
        self.calls.append((query, k, mode))
        return [
            Hit(chunk=BY_ID[cid], rank=i, score=1 / i, scores={"dense": 0.9 - i / 10})
            for i, cid in enumerate(RANKINGS[query][:k], start=1)
        ]


MANIFEST = [
    ManifestRecord(
        record_id="p:a",
        source=Source.ARXIV,
        title="A",
        relevance=Decision.ACCEPT,
        included_in_index=True,
    ),
    ManifestRecord(
        record_id="p:a-copy",
        source=Source.ACL,
        title="A",
        derived_from="p:a",
        relevance=Decision.ACCEPT,
        included_in_index=True,
    ),
    ManifestRecord(
        record_id="p:b",
        source=Source.ARXIV,
        title="B",
        relevance=Decision.ACCEPT,
        included_in_index=True,
    ),
]


@pytest.fixture
def config(tmp_path: Path, minimal_config_text: str):
    path = tmp_path / "config.yaml"
    path.write_text(minimal_config_text)
    config = load_config(path)
    write_jsonl(step_dir(config, "chunk") / CHUNKS_FILE, CHUNKS)
    write_manifest(MANIFEST, config.paths.manifest)
    return config


def run_fake(config) -> RunResult:
    out = run_eval(config, "t", ["hybrid"], FIXTURE, retriever=FakeRetriever())
    return RunResult.model_validate_json(out.read_text())


def rows(result: RunResult) -> dict[tuple[str, str], dict]:
    return {(r.question_id, r.variant): r for r in result.modes["hybrid"].questions}


def test_per_question_metrics_by_hand(config) -> None:
    r = rows(run_fake(config))
    # q1: the linked copy's chunk holds the span at rank 1 (D-031); slots 1-2 are one paper.
    assert r["q1", "raw"].metrics == {
        "recall@2": 1.0,
        "paper_recall@2": 1.0,
        "mrr@3": 1.0,
        "dup@2": 0.5,
        "any_dup@2": 1.0,
    }
    # q2 as asked: gold chunk only at rank 3.
    assert r["q2", "raw"].metrics["recall@2"] == 0.0
    assert r["q2", "raw"].metrics["mrr@3"] == pytest.approx(1 / 3)
    assert r["q2", "oracle"].metrics["recall@2"] == 1.0
    assert not r["q2", "system"].scored and r["q2", "system"].query is None
    # q3: both papers' spans in the top 2.
    assert r["q3", "raw"].metrics["recall@2"] == 1.0
    assert r["q3", "raw"].metrics["dup@2"] == 0.0
    # q4: unanswerable, not scored, top cosine kept for the abstention gate.
    assert not r["q4", "raw"].scored
    assert r["q4", "raw"].top_dense_cosine == pytest.approx(0.8)


def test_aggregate_slices(config) -> None:
    agg = run_fake(config).modes["hybrid"].aggregates
    assert agg["overall"]["recall@2"].n == 3  # q1, q2 raw, q3
    assert agg["overall"]["recall@2"].mean == pytest.approx(2 / 3)
    assert agg["overall"]["mrr@3"].mean == pytest.approx((1 + 1 / 3 + 1) / 3)
    assert agg["followup:oracle"]["recall@2"].mean == 1.0
    assert agg["followup:raw"]["recall@2"].mean == 0.0
    assert "followup:oracle" in agg and "type:unanswerable" not in agg
    s = agg["overall"]["recall@2"]
    assert s.ci_low <= s.mean <= s.ci_high


def test_results_file_records_provenance(config) -> None:
    result = run_fake(config)
    assert result.n_questions == 4
    assert result.questions_sha256 and result.git_commit and result.config_hash
    assert result.config["eval"]["recall_k"] == 2
    assert "paths" not in result.config


def test_retriever_asked_for_the_deepest_cutoff(config) -> None:
    fake = FakeRetriever()
    run_eval(config, "t", ["bm25"], FIXTURE, retriever=fake)
    assert {(k, mode) for _, k, mode in fake.calls} == {(3, "bm25")}


def test_no_questions_writes_nothing(config, tmp_path: Path) -> None:
    assert (
        run_eval(config, "t", ["bm25"], tmp_path / "none.jsonl", retriever=FakeRetriever()) is None
    )
    assert not config.paths.results.exists()


def test_limit_scores_the_first_questions(config) -> None:
    out = run_eval(config, "t", ["hybrid"], FIXTURE, limit=1, retriever=FakeRetriever())
    assert json.loads(out.read_text())["n_questions"] == 1


def test_end_to_end_on_real_indexes(config, monkeypatch) -> None:
    index_pipeline.bm25(config)
    index_pipeline.dense(config, encoder=FakeEncoder())
    monkeypatch.setattr(
        "rag.retrieve.retriever.SentenceTransformerEncoder", lambda _cfg: FakeEncoder()
    )
    out = run_eval(config, "e2e", ["bm25", "dense", "hybrid"], FIXTURE)
    result = RunResult.model_validate_json(out.read_text())
    assert set(result.modes) == {"bm25", "dense", "hybrid"}
    assert set(result.index_meta) == {"bm25", "dense"}
    assert out.name.endswith("-e2e.json")
