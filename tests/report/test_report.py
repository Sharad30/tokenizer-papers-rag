from pathlib import Path

import pytest

from rag.collect.manifest import write_manifest
from rag.common.config import load_config
from rag.common.io import write_jsonl
from rag.eval.run import run_eval
from rag.parse.pipeline import CHUNKS as CHUNKS_FILE
from rag.parse.pipeline import step_dir
from rag.report.report import headline_table, load, paired_table, write_report
from tests.eval.conftest import CHUNKS
from tests.eval.test_harness import FIXTURE, MANIFEST, FakeRetriever


@pytest.fixture
def result_path(tmp_path: Path, minimal_config_text: str) -> tuple[Path, object]:
    path = tmp_path / "config.yaml"
    path.write_text(minimal_config_text)
    config = load_config(path)
    write_jsonl(step_dir(config, "chunk") / CHUNKS_FILE, CHUNKS)
    write_manifest(MANIFEST, config.paths.manifest)
    return run_eval(config, "toy", ["bm25", "hybrid"], FIXTURE, retriever=FakeRetriever()), config


def test_report_writes_markdown_figure_and_table_view(result_path) -> None:
    path, config = result_path
    outputs = write_report(load(path), config.paths.results, config.paths.figures)
    names = sorted(p.name for p in outputs)
    assert names[0].endswith("-toy-metrics.md") and names[1].endswith("-toy-metrics.png")
    assert all(p.stat().st_size > 0 for p in outputs)
    report = next(p for p in outputs if p.parent == config.paths.results).read_text()
    for heading in ("## Headline", "## Paired differences", "## By slice", "## Per question"):
        assert heading in report


def test_headline_table_has_a_row_per_mode(result_path) -> None:
    path, _ = result_path
    table = headline_table(load(path))
    assert "n = 3" in table
    assert "| bm25 | 0.667 [" in table and "| hybrid | 0.667 [" in table


def test_identical_modes_differ_by_zero(result_path) -> None:
    # The fake retriever ignores the mode, so every paired difference is exactly 0.
    path, _ = result_path
    table = paired_table(load(path))
    assert "| hybrid − bm25 | +0.000 [+0.000, +0.000]" in table
    assert "*" not in table.splitlines()[2]
