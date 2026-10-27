import pytest
from typer.testing import CliRunner

from rag.cli import app
from rag.common.config import Config, load_config
from rag.common.io import write_jsonl, write_text_atomic
from rag.parse.pipeline import CHUNKS, RECORDS, REFERENCES, STATUS, safe_name, step_dir
from rag.parse.schemas import (
    Chunk,
    ElementType,
    IndexStatus,
    Paragraph,
    ParseOutcome,
    ParseStatus,
    Reference,
    Section,
    Sentence,
    StructuredPaper,
)
from rag.showcase import parse
from rag.showcase.sources import snapshot_file, write_snapshot

PAPER = "arxiv:2001.00001v1"
BIB = "In Proceedings of ACL, pages 1-10. arXiv preprint arXiv:2001.1 In Proceedings of EMNLP"


def chunk(n: int, element: ElementType, text: str, path=("Intro",), tokens=10, refs=()) -> Chunk:
    return Chunk(
        chunk_id=f"{PAPER}#{n:04d}",
        record_id=PAPER,
        element_type=element,
        section_path=list(path),
        header="BPE Revisited > " + " > ".join(path),
        text=text,
        token_count=tokens,
        header_token_count=5,
        cited_refs=list(refs),
        parser="grobid",
    )


CHUNKS_ = [
    chunk(0, ElementType.ABSTRACT, "We study BPE.", path=("Abstract",), tokens=20),
    chunk(1, ElementType.PARAGRAPH, "BPE merges pairs. It starts from characters.", refs=["b0"]),
    chunk(2, ElementType.PARAGRAPH, "It starts from characters. Then it stops.", tokens=40),
    chunk(3, ElementType.PARAGRAPH, BIB, path=("Conclusion",)),
    chunk(4, ElementType.PARAGRAPH, "Some text [equation] here.", path=("Fig. 1",)),
    chunk(5, ElementType.FIGURE_CAPTION, "Figure 2: Figure 2. Merge counts.", path=("Figure 2",)),
    chunk(6, ElementType.TABLE, "Table 1: Results\n| a | b |", path=("Table 1",), tokens=510),
]
REFS = [
    Reference(record_id=PAPER, ref_id="b0", title="Neural MT of Rare Words", year=2016),
    Reference(record_id=PAPER, ref_id="b1", title="SentencePiece", year=2018),
]


def test_check_flags_find_each_known_defect() -> None:
    flags = {c.chunk_id[-1]: parse.check_flags(c, bibliography_min_matches=3) for c in CHUNKS_}
    assert flags["5"] == ["repeated_label"]
    assert flags["3"] == ["bibliography"]
    assert flags["4"] == ["odd_heading"]
    assert flags["6"] == []  # "Table 1: Results" doesn't repeat the label
    assert flags["1"] == [] and flags["0"] == []


def test_short_real_heading_is_flagged_but_a_word_is_not() -> None:
    bpe = chunk(9, ElementType.PARAGRAPH, "x", path=("BPE",))
    method = chunk(9, ElementType.PARAGRAPH, "x", path=("Method",))
    assert parse.check_flags(bpe, 3) == ["odd_heading"]
    assert parse.check_flags(method, 3) == []


def test_overview_counts() -> None:
    status = [
        ParseStatus(record_id=PAPER, outcome=ParseOutcome.GROBID),
        ParseStatus(record_id="acl:x", outcome=ParseOutcome.NO_PDF),
    ]
    index = [
        IndexStatus(
            record_id=PAPER, included_in_index=True, n_chunks=7, outcome=ParseOutcome.GROBID
        ),
        IndexStatus(
            record_id="acl:x", included_in_index=False, n_chunks=0, outcome=ParseOutcome.NO_PDF
        ),
    ]
    view = parse.overview(
        status, index, CHUNKS_, REFS, model_limit=512, bin_width=10, bibliography_min_matches=3
    )
    assert view.accepted == 2
    assert view.outcomes == {"grobid": 1, "fallback": 0, "failed": 0, "no_pdf": 1}
    assert (view.indexed_papers, view.chunks) == (1, 7)
    assert view.per_element == {"abstract": 1, "paragraph": 4, "table": 1, "figure_caption": 1}
    assert view.median_tokens["paragraph"] == 10
    assert (view.chunks_per_paper_median, view.chunks_per_paper_max) == (7, 7)
    assert view.over_limit == 1  # the 510-token table: 5 + 510 + 2 > 512
    assert view.formula_placeholders == 1
    assert (view.references, view.papers_with_references) == (2, 1)
    assert (view.paragraphs_citing, view.paragraphs) == (1, 4)
    assert view.checks == {"repeated_label": 1, "bibliography": 1, "odd_heading": 1}
    # Bins are of model tokens (header 5 + body + 2 special): 17 -> 10, 27 -> 20.
    assert ("abstract", 20) in {(b.element, b.start) for b in view.token_bins}
    assert sum(b.chunks for b in view.token_bins) == 7


def test_overlap_is_the_repeated_sentences_within_a_section() -> None:
    assert parse.overlap(CHUNKS_[1], CHUNKS_[2]) == "It starts from characters."
    assert parse.overlap(CHUNKS_[0], CHUNKS_[1]) == ""  # different section
    assert parse.overlap(CHUNKS_[2], CHUNKS_[3]) == ""


@pytest.fixture
def config(write_config, minimal_config_text: str) -> Config:
    return load_config(write_config(minimal_config_text))


def write_parse_outputs(config: Config) -> None:
    write_jsonl(
        step_dir(config, "grobid") / RECORDS,
        [ParseStatus(record_id=PAPER, outcome=ParseOutcome.GROBID)],
    )
    write_jsonl(step_dir(config, "chunk") / CHUNKS, CHUNKS_)
    write_jsonl(
        step_dir(config, "chunk") / STATUS,
        [
            IndexStatus(
                record_id=PAPER, included_in_index=True, n_chunks=7, outcome=ParseOutcome.GROBID
            )
        ],
    )
    write_jsonl(step_dir(config, "structure") / REFERENCES, REFS)
    paper = StructuredPaper(
        record_id=PAPER,
        parser="grobid",
        title="BPE Revisited",
        sections=[
            Section(
                id="s0",
                number="1",
                title="Intro",
                path=["Intro"],
                level=1,
                paragraphs=[Paragraph(sentences=[Sentence(text="BPE merges pairs.")])],
            )
        ],
    )
    write_text_atomic(
        step_dir(config, "structure") / f"{safe_name(PAPER)}.json", paper.model_dump_json()
    )


def test_load_live_and_open_a_paper(config: Config) -> None:
    write_parse_outputs(config)
    data = parse.load(config)
    assert data.origin == "live"
    assert data.overview.chunks == 7
    assert [c.chunk_id[-1] for c in data.check_pools["bibliography"]] == ["3"]
    assert data.paper_ids == [PAPER]
    assert data.titles == {PAPER: "BPE Revisited"}
    assert [c.chunk_id[-1] for c in data.formula_pool] == ["4"]
    paper = parse.open_paper(config, data, PAPER)
    assert paper.title == "BPE Revisited"
    assert paper.sections[0].paragraphs == 1
    assert [c.chunk_id for c in paper.chunks] == sorted(c.chunk_id for c in CHUNKS_)
    assert paper.references == 2
    assert parse.open_paper(config, data, "acl:missing") is None


def test_load_without_outputs_or_snapshot(config: Config) -> None:
    assert parse.load(config).origin == "none"


def test_snapshot_round_trip_is_down_sampled(config: Config) -> None:
    write_parse_outputs(config)
    live = parse.load(config)
    data = parse.snapshot_data(config, live, config.showcase.parse, seed=1)
    path = snapshot_file(config.paths.snapshot, parse.STAGE)
    write_snapshot(path, parse.STAGE, parse.source_files(config), data, "abc")
    for file in parse.source_files(config).values():
        file.unlink()  # a fresh clone: no parse outputs
    snap = parse.load(config)
    assert snap.origin == "snapshot"
    assert snap.overview == live.overview  # numbers computed live, stored as is
    assert len(snap.element_pools["paragraph"]) == 2  # pool_size in the fixture config
    assert len(snap.references) == 2
    assert list(snap.papers) == [PAPER]
    assert snap.titles == {PAPER: "BPE Revisited"}
    assert len(snap.formula_pool) == 1
    assert parse.open_paper(config, snap, PAPER).title == "BPE Revisited"


def test_snapshot_goes_stale_when_chunks_change(config: Config) -> None:
    write_parse_outputs(config)
    data = parse.snapshot_data(config, parse.load(config), config.showcase.parse, seed=1)
    path = snapshot_file(config.paths.snapshot, parse.STAGE)
    write_snapshot(path, parse.STAGE, parse.source_files(config), data, "abc")
    files = parse.source_files(config)
    write_jsonl(files["chunks"], CHUNKS_[:2])
    files["references"].unlink()  # incomplete outputs: falls back to the snapshot
    assert parse.load(config).stale == ["chunks"]


def test_sample_is_reproducible_and_order_independent() -> None:
    first = parse.sample(CHUNKS_, 3, 1, parse.chunk_key)
    assert first == parse.sample(list(reversed(CHUNKS_)), 3, 1, parse.chunk_key)
    assert len(parse.sample(CHUNKS_, 99, 1, parse.chunk_key)) == 7


def test_snapshot_command_for_parse(config: Config) -> None:
    config_path = config.paths.raw.parent.parent / "config.yaml"
    run = ["--config", str(config_path), "showcase", "snapshot", "--stage", "parse"]
    refused = CliRunner().invoke(app, run)
    assert refused.exit_code != 0
    write_parse_outputs(config)
    result = CliRunner().invoke(app, run)
    assert result.exit_code == 0, result.output
    assert (config.paths.snapshot / "parse.json").exists()
