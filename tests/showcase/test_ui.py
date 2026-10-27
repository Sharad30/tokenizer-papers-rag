"""The Streamlit pages run headlessly (AppTest) on the synthetic manifest."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from rag.common.config import Config
from rag.showcase import collect
from rag.showcase.sources import snapshot_file, write_snapshot
from tests.showcase.conftest import write_filtered

UI = Path(__file__).resolve().parents[2] / "ui"
TIMEOUT_S = 30


@pytest.fixture
def collect_page(config: Config, monkeypatch: pytest.MonkeyPatch):
    config_path = config.paths.manifest.parent.parent / "config.yaml"
    monkeypatch.setenv("RAG_CONFIG", str(config_path))
    return AppTest.from_file(str(UI / "pages" / "collect.py"), default_timeout=TIMEOUT_S)


def titles(at: AppTest) -> list[str]:
    return [m.value for m in at.markdown if m.value.startswith("#### ")]


def test_app_opens_on_the_home_page() -> None:
    at = AppTest.from_file(str(UI / "app.py"), default_timeout=TIMEOUT_S).run()
    assert not at.exception
    assert at.title[0].value.startswith("RAG over research papers")


def test_home_page_shows_the_architecture_diagram() -> None:
    at = AppTest.from_file(str(UI / "pages" / "home.py"), default_timeout=TIMEOUT_S).run()
    assert not at.exception
    assert "RETRIEVER" in at.code[0].value


def test_collect_page_numbers_come_from_the_data(config: Config, collect_page: AppTest) -> None:
    write_filtered(config)
    at = collect_page.run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Candidates found"] == "10"
    assert metrics["Accepted records"] == "3"
    assert metrics["Distinct papers"] == "2"
    assert "Rejected abstracts: live" in at.caption[0].value


def test_switching_path_and_resampling(config: Config, collect_page: AppTest) -> None:
    write_filtered(config)
    at = collect_page.run()
    at.selectbox[0].set_value("passing_mention").run()
    first = titles(at)
    assert len(first) == 2  # examples_shown in the fixture config
    assert all("Summariser" in t for t in first)
    seen = {tuple(first)}
    for _ in range(5):
        at.button[0].click().run()
        seen.add(tuple(titles(at)))
    assert len(seen) > 1
    assert not at.exception


def test_year_filter_applies_only_when_narrowed(config: Config, collect_page: AppTest) -> None:
    write_filtered(config)
    at = collect_page.run()
    at.selectbox[0].set_value("passing_mention").run()
    assert len(titles(at)) == 2  # these rejects have no year, and the slider is untouched
    at.slider[0].set_range(2021, 2021).run()
    assert titles(at) == []


def test_rejects_fall_back_to_the_snapshot(config: Config, collect_page: AppTest) -> None:
    write_snapshot(
        snapshot_file(config.paths.snapshot, "collect"),
        "collect",
        collect.source_files(config),
        {collect.REJECT_ABSTRACTS: {"s2:below": "Fertility matters. Fertility again."}},
        "abc12345",
    )
    at = collect_page.run()
    assert not at.exception
    assert "snapshot from commit abc12345" in at.caption[0].value
    at.selectbox[0].set_value("embedding_below").run()
    assert titles(at) == ["#### Quality Estimation"]


def test_search_and_try_the_rules(config: Config, collect_page: AppTest) -> None:
    at = collect_page.run()
    at.text_input(key="try_title").input("A Visual Tokenizer for Images").run()
    assert any("off_topic_term" in m.value for m in at.markdown)
    at.text_input[0].input("merges").run()
    assert not at.exception
    assert at.selectbox(key="meta_record").options == [
        "BPE Merges Revisited (acl:2020.acl-main.1)",
        "BPE Merges Revisited (arxiv:2001.00001v1)",
    ]
