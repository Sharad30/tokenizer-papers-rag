import pytest

from rag.collect.manifest import write_manifest
from rag.collect.pipeline import RECORDS, step_dir
from rag.collect.relevance import KeywordRules
from rag.collect.schemas import (
    DecidedBy,
    Decision,
    DownloadStatus,
    LicenceSource,
    ManifestRecord,
    Reason,
    Source,
)
from rag.common.config import Config, load_config
from rag.common.io import write_jsonl
from rag.showcase import collect
from rag.showcase.sources import snapshot_file, write_snapshot


def rec(record_id: str, decision: Decision, reason: Reason, **fields) -> ManifestRecord:
    defaults = {
        "source": Source.ARXIV,
        "title": record_id,
        "decided_by": DecidedBy.KEYWORD,
    }
    return ManifestRecord(
        record_id=record_id, relevance=decision, reason=reason, **{**defaults, **fields}
    )


ACCEPTED = [
    rec(
        "arxiv:2001.00001v1",
        Decision.ACCEPT,
        Reason.TITLE_TERM,
        title="BPE Merges Revisited",
        abstract="We study BPE.",
        year=2020,
        arxiv_id="2001.00001",
        doi="10.1/x",
        authors=["Ada"],
        venue="ACL",
        pdf_url="https://export.arxiv.org/pdf/2001.00001v1",
        licence_source=LicenceSource.ARXIV_OAI,
        download_status=DownloadStatus.DOWNLOADED,
        included_in_index=True,
        found_via=["arxiv_query:bpe", "snowball_cit:ARXIV:1"],
    ),
    rec(
        "acl:2020.acl-main.1",
        Decision.ACCEPT,
        Reason.LINKED_MANIFESTATION,
        source=Source.ACL,
        title="BPE Merges Revisited",
        abstract="We study BPE.",
        year=2020,
        doi="10.1/X",  # case differs: still the same paper
        derived_from="arxiv:2001.00001v1",
        decided_by=DecidedBy.INHERITED,
        download_status=DownloadStatus.FAILED,
        found_via=["acl_scan"],
    ),
    rec(
        "s2:judged",
        Decision.ACCEPT,
        Reason.EMBEDDING_ABOVE,
        source=Source.S2,
        title="Translation Across Languages",
        abstract="Fertility predicts quality. We measure fertility widely.",
        embedding_score=0.7,
        decided_by=DecidedBy.EMBEDDING,
        year=2021,
        found_via=["snowball_cit:ARXIV:1"],
    ),
]
REJECTED = [
    rec(
        f"s2:mention{i}",
        Decision.REJECT,
        Reason.PASSING_MENTION,
        source=Source.S2,
        title=f"A Summariser {i}",
        abstract="We train with a BPE tokenizer. Results improve.",
        found_via=["snowball_cit:ARXIV:1"],
    )
    for i in range(5)
] + [
    rec(
        "s2:below",
        Decision.REJECT,
        Reason.EMBEDDING_BELOW,
        source=Source.S2,
        title="Quality Estimation",
        abstract="Fertility matters. Fertility again.",
        embedding_score=0.2,
        decided_by=DecidedBy.EMBEDDING,
    ),
    rec(
        "acl:visual",
        Decision.REJECT,
        Reason.OFF_TOPIC_TERM,
        source=Source.ACL,
        title="A Visual Tokenizer",
    ),
]


@pytest.fixture
def config(write_config, minimal_config_text: str) -> Config:
    config = load_config(write_config(minimal_config_text))
    write_manifest([*ACCEPTED, *REJECTED], config.paths.manifest)
    return config


def write_filtered(config: Config) -> None:
    write_jsonl(step_dir(config, "filter") / RECORDS, [*ACCEPTED, *REJECTED])


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
