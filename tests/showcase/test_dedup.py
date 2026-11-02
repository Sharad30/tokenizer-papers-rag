"""The Versions and duplicates page: numbers from data/works, snapshot fallback."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from rag.common.config import Config
from rag.common.io import write_jsonl
from rag.dedup.files import RESTATEMENTS, SURVEYS, WORKS
from rag.dedup.schemas import Edge, Work
from rag.parse.pipeline import CHUNKS, step_dir
from rag.showcase import dedup
from rag.showcase.sources import snapshot_file, write_snapshot
from tests.retrieve.conftest import make_chunk

UI = Path(__file__).resolve().parents[2] / "ui"


def write_works(config: Config) -> None:
    work = Work(
        work_id="work:acl:2020.acl-main.1",
        canonical="acl:2020.acl-main.1",
        canonical_rule="venue version (ACL Anthology)",
        members=["acl:2020.acl-main.1", "arxiv:2001.00001v1"],
        aliases=["arxiv:2001.00001v1"],
        edges=[Edge(a="arxiv:2001.00001v1", b="acl:2020.acl-main.1", step="ids", reason="doi")],
        rejected=[],
    )
    write_jsonl(config.paths.works / WORKS, [work])
    write_jsonl(config.paths.works / SURVEYS, [])
    write_jsonl(config.paths.works / RESTATEMENTS, [])
    write_jsonl(
        step_dir(config, "chunk") / CHUNKS,
        [make_chunk("acl:2020.acl-main.1", 0, "bpe"), make_chunk("arxiv:2001.00001v1", 0, "bpe")],
    )


def test_live_overview_counts_records_groups_and_works(config: Config) -> None:
    write_works(config)
    data = dedup.load(config)
    assert data.origin == "live"
    view = data.overview
    assert (view.records, view.id_groups, view.works, view.multi_member) == (2, 1, 1, 1)
    assert view.edges == {"ids": 1} and view.sizes == {2: 1}
    assert [w.work_id for w in data.id_works] == ["work:acl:2020.acl-main.1"]
    assert data.fuzzy_works == []


@pytest.fixture
def dedup_page(config: Config, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.setenv("RAG_CONFIG", str(config.paths.manifest.parent.parent / "config.yaml"))
    return AppTest.from_file(str(UI / "pages" / "dedup.py"), default_timeout=30)


def test_page_without_works_says_what_to_run(dedup_page: AppTest) -> None:
    at = dedup_page.run()
    assert "rag dedup run" in at.error[0].value


def test_page_live_and_from_the_snapshot(config: Config, dedup_page: AppTest) -> None:
    write_works(config)
    at = dedup_page.run()
    assert not at.exception
    assert {m.label: m.value for m in at.metric}["Works"] == "1"
    at.button(key="dedup_resample").click().run()
    assert not at.exception

    data = dedup.snapshot_data(dedup.load(config), config.showcase.dedup, seed=0)
    write_snapshot(
        snapshot_file(config.paths.snapshot, dedup.STAGE),
        dedup.STAGE,
        dedup.source_files(config),
        data,
        "abc12345",
    )
    (config.paths.works / WORKS).unlink()
    at = dedup_page.run()
    assert not at.exception
    assert "snapshot from commit abc12345" in at.caption[0].value


def test_embeddings_page_without_indexes_says_so(config: Config, monkeypatch) -> None:
    from rag.showcase import embeddings

    assert {m.status for m in embeddings.models(config)} == {"missing"}
    monkeypatch.setenv("RAG_CONFIG", str(config.paths.manifest.parent.parent / "config.yaml"))
    at = AppTest.from_file(str(UI / "pages" / "embeddings.py"), default_timeout=30).run()
    assert not at.exception
    assert any("No comparison yet" in i.value for i in at.info)
    assert any("Live queries need the indexes" in i.value for i in at.info)


def test_latest_ablation_reads_recall_per_question_set(tmp_path) -> None:
    import json

    from rag.showcase.dedup import latest_ablation

    probe = {
        "dup@5": {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0},
        "copy_dup@5": {"mean": 0.0},
        "copy_dup@5_resolver": {"mean": 0.0},
    }
    overall = {"recall@5": {"mean": 0.2}, "paper_recall@5": {"mean": 0.5}, "mrr@10": {"mean": 0.1}}
    old = {"label": "old", "probe": probe}  # written before any questions existed
    new = {
        **old,
        "label": "new",
        "sets": {"gold": {"result": {"aggregates": {"overall": overall}}}},
    }
    raw = {"git_commit": "abc", "n_queries": 1, "labelled_pairs": 0}
    raw["conditions"] = {"old": old, "new": new}
    (tmp_path / "2026-11-02-dedup-ablation.json").write_text(json.dumps(raw), encoding="utf-8")
    rows = latest_ablation(tmp_path).rows
    assert rows[0].recall == {}
    assert rows[1].recall == {"gold": {"recall@5": 0.2, "paper_recall@5": 0.5}}
