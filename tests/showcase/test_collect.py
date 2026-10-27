from typer.testing import CliRunner

from rag.cli import app
from rag.collect.manifest import write_manifest
from rag.collect.relevance import KeywordRules
from rag.collect.schemas import Decision
from rag.common.config import Config
from rag.showcase import collect
from rag.showcase.sources import snapshot_file, write_snapshot
from tests.showcase.conftest import ACCEPTED, REJECTED, write_filtered


def test_load_takes_reject_abstracts_live_when_present(config: Config) -> None:
    write_filtered(config)
    data = collect.load(config)
    assert data.abstracts_from == "live"
    assert {r.record_id: r.abstract for r in data.rejected}["s2:below"] == (
        "Fertility matters. Fertility again."
    )
    assert data.accepted == sorted(ACCEPTED, key=lambda r: r.record_id)


def test_load_falls_back_to_the_snapshot(config: Config) -> None:
    write_snapshot(
        snapshot_file(config.paths.snapshot, "collect"),
        "collect",
        collect.source_files(config),
        {collect.REJECT_ABSTRACTS: {"s2:below": "From the snapshot."}},
        "abc",
    )
    data = collect.load(config)
    assert data.abstracts_from == "snapshot"
    assert data.stale == []
    abstracts = {r.record_id: r.abstract for r in data.rejected}
    assert abstracts["s2:below"] == "From the snapshot."
    assert abstracts["s2:mention0"] is None


def test_snapshot_goes_stale_when_the_manifest_changes(config: Config) -> None:
    write_snapshot(
        snapshot_file(config.paths.snapshot, "collect"),
        "collect",
        collect.source_files(config),
        {collect.REJECT_ABSTRACTS: {}},
        "abc",
    )
    write_manifest(ACCEPTED[:1], config.paths.manifest)
    assert collect.load(config).stale == ["manifest", "rejected"]


def test_load_without_raw_or_snapshot(config: Config) -> None:
    data = collect.load(config)
    assert data.abstracts_from == "none"
    assert all(r.abstract is None for r in data.rejected)


def test_snapshot_data_samples_rejects_per_path(config: Config) -> None:
    write_filtered(config)
    data = collect.load(config)
    sample = collect.snapshot_data(data, pool_per_path=3, seed=1)[collect.REJECT_ABSTRACTS]
    mentions = [k for k in sample if k.startswith("s2:mention")]
    assert len(mentions) == 3  # capped at the pool size
    assert "s2:below" in sample
    assert "acl:visual" not in sample  # no abstract to show
    assert not any(k in sample for k in ("s2:judged", "arxiv:2001.00001v1"))  # accepts are live
    assert sample == collect.snapshot_data(data, pool_per_path=3, seed=1)[collect.REJECT_ABSTRACTS]


def test_overview_counts() -> None:
    view = collect.overview(ACCEPTED, REJECTED)
    assert (view.candidates, view.accepted, view.rejected) == (10, 3, 7)
    assert (view.downloaded, view.indexed) == (1, 1)
    assert view.distinct_accepted == 2  # the ACL copy shares the arXiv paper's DOI
    by_source = {s.source: (s.candidates, s.accepted) for s in view.by_source}
    assert by_source == {"arxiv": (1, 1), "acl": (2, 1), "s2": (7, 1)}
    assert view.by_path["passing_mention"] == 5
    assert view.by_path["linked_copy"] == 1
    assert view.by_path["no_topic_term"] == 0
    assert view.routes == {"arxiv_query + snowball_cit": 1, "acl_scan": 1, "snowball_cit": 1}
    assert [(y.year, y.source, y.papers) for y in view.per_year] == [
        (2020, "acl", 1),
        (2020, "arxiv", 1),
        (2021, "s2", 1),
    ]
    assert view.missing == {"venue": 2, "year": 0, "authors": 2, "abstract": 0}
    assert sorted((j.score, j.accepted) for j in view.judged) == [(0.2, False), (0.7, True)]


def test_sources_summary_reads_config_and_counts(config: Config) -> None:
    data = collect.load(config)
    view = collect.overview(data.accepted, data.rejected)
    arxiv, s2, acl = collect.sources_summary(config, data, view)
    assert arxiv.details == ['bpe: abs:"byte pair encoding"']
    assert (arxiv.candidates, arxiv.accepted) == (1, 1)
    # API hosts from the client modules; unlisted hosts get the default interval.
    assert arxiv.hosts == {"export.arxiv.org": 0.5, "oaipmh.arxiv.org": 0.5}
    assert s2.details == ["ARXIV:0000.00000"]
    assert acl.how == "Disabled in config."


def test_example_pool_filters_and_prefers_abstracts() -> None:
    records = [*ACCEPTED, *REJECTED]
    assert {r.record_id for r in collect.example_pool(records, "passing_mention")} == {
        f"s2:mention{i}" for i in range(5)
    }
    assert collect.example_pool(records, "title_term", sources=["acl"]) == []
    assert collect.example_pool(records, "title_term", years=(2021, 2026)) == []
    # No abstract anywhere on the path: the records are still shown.
    assert [r.record_id for r in collect.example_pool(records, "off_topic")] == ["acl:visual"]


def test_pick_is_reproducible_and_seed_dependent() -> None:
    pool = collect.example_pool(REJECTED, "passing_mention")
    first = collect.pick(pool, 2, seed=1)
    assert first == collect.pick(list(reversed(pool)), 2, seed=1)
    assert len(first) == 2
    assert any(collect.pick(pool, 2, seed=s) != first for s in range(2, 10))
    assert len(collect.pick(pool, 99, seed=1)) == 5


def test_explain_agrees_with_the_recorded_decision(config: Config) -> None:
    rules = KeywordRules(config.collect.relevance)
    relevance = config.collect.relevance
    mention = collect.explain(REJECTED[0], rules, relevance)
    assert mention.agrees is True
    assert [terms for _, terms in mention.trace.sentences] == [["bpe", "tokenizer"], []]
    judged = collect.explain(ACCEPTED[2], rules, relevance)  # ambiguous, resolved by its score
    assert (judged.today, judged.agrees) == (Decision.ACCEPT, True)
    copy = collect.explain(ACCEPTED[1], rules, relevance)
    assert (copy.inherited_from, copy.agrees) == ("arxiv:2001.00001v1", None)


def test_explain_flags_a_decision_todays_config_would_change(config: Config) -> None:
    relevance = config.collect.relevance.model_copy(update={"embedding_threshold": 0.9})
    judged = collect.explain(ACCEPTED[2], KeywordRules(relevance), relevance)
    assert (judged.today, judged.agrees) == (Decision.REJECT, False)
    dropped = relevance.model_copy(update={"title_accept_terms": ["tokenizer"]})
    bpe = collect.explain(ACCEPTED[0], KeywordRules(dropped), dropped)
    assert bpe.agrees is False


def test_linked_copies_group_by_shared_ids() -> None:
    copies = collect.linked_copies(ACCEPTED)
    assert {r.record_id for r in copies["acl:2020.acl-main.1"]} == {
        "arxiv:2001.00001v1",
        "acl:2020.acl-main.1",
    }
    assert [r.record_id for r in copies["s2:judged"]] == ["s2:judged"]


def test_search_by_title_or_id() -> None:
    assert [r.record_id for r in collect.search(ACCEPTED, "merges", 5)] == [
        "arxiv:2001.00001v1",
        "acl:2020.acl-main.1",
    ]
    assert [r.record_id for r in collect.search(ACCEPTED, "S2:JUD", 5)] == ["s2:judged"]
    assert collect.search(ACCEPTED, "  ", 5) == []
    assert len(collect.search(ACCEPTED, "merges", 1)) == 1


def test_snapshot_command_needs_live_data_then_writes(config: Config) -> None:
    config_path = config.paths.manifest.parent.parent / "config.yaml"
    run = ["--config", str(config_path), "showcase", "snapshot"]
    refused = CliRunner().invoke(app, run)
    assert refused.exit_code != 0
    assert "rag collect filter" in refused.output
    write_filtered(config)
    result = CliRunner().invoke(app, run)
    assert result.exit_code == 0, result.output
    assert collect.load(config).abstracts_from == "live"
    snapshot = (config.paths.snapshot / "collect.json").read_text(encoding="utf-8")
    assert "s2:below" in snapshot
