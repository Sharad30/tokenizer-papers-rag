"""The Streamlit pages run headlessly (AppTest) on the synthetic manifest."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from rag.common.config import Config, load_config
from rag.showcase import collect, parse
from rag.showcase.sources import snapshot_file, write_snapshot
from tests.showcase.conftest import write_filtered
from tests.showcase.test_parse import write_parse_outputs

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


@pytest.fixture
def parse_page(write_config, minimal_config_text: str, monkeypatch: pytest.MonkeyPatch):
    config_path = write_config(minimal_config_text)
    monkeypatch.setenv("RAG_CONFIG", str(config_path))
    page = AppTest.from_file(str(UI / "pages" / "parse.py"), default_timeout=TIMEOUT_S)
    return load_config(config_path), page


def test_parse_page_without_data_says_what_to_run(parse_page) -> None:
    _, page = parse_page
    at = page.run()
    assert "rag parse all" in at.error[0].value


def test_parse_page_live(parse_page) -> None:
    config, page = parse_page
    write_parse_outputs(config)
    at = page.run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Chunks"] == "7"
    # The fixture config's model limit is 40 tokens: the 40- and 510-token chunks exceed it.
    assert metrics["Chunks over the limit"] == "2"
    assert [e.label for e in at.expander] == [
        "Caption repeats its label: 1 chunks",
        "Reference-list text in a paragraph: 1 chunks",
        "Suspicious section heading: 1 chunks",
    ]
    at.selectbox(key="parse_section").set_value("Intro").run()
    assert any("It starts from characters." in c.value for c in at.caption if "Overlap" in c.value)
    at.button(key="parse_resample").click().run()
    assert not at.exception


def test_parse_page_from_the_snapshot(parse_page) -> None:
    config, page = parse_page
    write_parse_outputs(config)
    data = parse.snapshot_data(config, parse.load(config), config.showcase.parse, seed=1)
    write_snapshot(
        snapshot_file(config.paths.snapshot, parse.STAGE),
        parse.STAGE,
        parse.source_files(config),
        data,
        "abc12345",
    )
    for file in parse.source_files(config).values():
        file.unlink()
    at = page.run()
    assert not at.exception
    assert "snapshot from commit abc12345" in at.caption[0].value
    assert {m.label: m.value for m in at.metric}["Chunks"] == "7"
