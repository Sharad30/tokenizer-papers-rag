import numpy as np

from rag.collect.schemas import ManifestRecord, Source
from rag.dedup.config import SurveyConfig
from rag.dedup.surveys import ReferenceResolver, find_restatements, find_surveys
from rag.parse.schemas import Chunk, ElementType, Parser, Reference

CFG = SurveyConfig(
    title_cues=["survey", "overview", "brief history"], reference_title_min=95, restate_min=0.8
)


def rec(record_id: str, title: str, **ids) -> ManifestRecord:
    return ManifestRecord(record_id=record_id, source=Source.ARXIV, title=title, **ids)


def chunk(chunk_id: str, cited: list[str] | None = None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        record_id=chunk_id.split("#")[0],
        element_type=ElementType.PARAGRAPH,
        header="h",
        text="t",
        token_count=1,
        header_token_count=1,
        cited_refs=cited or [],
        parser=Parser.GROBID,
    )


def test_survey_cues_are_whole_words_in_the_title() -> None:
    records = [
        rec("a", "Between words and characters: A Brief History of Open-Vocabulary Modeling"),
        rec("b", "Discrete Tokenization for Multimodal LLMs: A Comprehensive Survey"),
        rec("c", "A Systematic Study of Leveraging Subword Information"),  # an experiment
        rec("d", "Surveying tokenizers"),  # not the word "survey"
    ]
    assert {s.record_id: s.cues for s in find_surveys(records, CFG)} == {
        "a": ["brief history"],
        "b": ["survey"],
    }


def test_references_resolve_by_arxiv_id_then_doi_then_title() -> None:
    records = [
        rec(
            "arxiv:1910.13267v2",
            "BPE-Dropout: Simple and Effective Subword Regularization",
            arxiv_id="1910.13267",
        ),
        rec(
            "acl:P16-1162",
            "Neural Machine Translation of Rare Words with Subword Units",
            doi="10.18653/v1/P16-1162",
        ),
    ]
    resolver = ReferenceResolver(records, title_min=95)

    def ref(**fields) -> Reference:
        return Reference(record_id="s", ref_id="b0", **fields)

    assert resolver.resolve(ref(arxiv_id="1910.13267v1")) == ("arxiv:1910.13267v2", "arxiv")
    assert resolver.resolve(ref(doi="10.18653/V1/P16-1162")) == ("acl:P16-1162", "doi")
    assert resolver.resolve(
        ref(title="Neural machine translation of rare words with subword units.")
    ) == ("acl:P16-1162", "title")
    assert resolver.resolve(ref(title="Some other paper")) is None


def test_a_survey_chunk_restates_a_cited_work_only_when_it_says_the_same() -> None:
    records = [
        rec("survey", "A Survey of Tokenizers"),
        rec("bpe", "Neural Machine Translation of Rare Words with Subword Units"),
        rec("other", "Unigram Language Model Tokenization"),
    ]
    chunks = [
        chunk("survey#0", cited=["b0", "b1"]),  # cites both, close to bpe only
        chunk("bpe#0"),
        chunk("other#0"),
    ]
    references = [
        Reference(record_id="survey", ref_id="b0", title=records[1].title),
        Reference(record_id="survey", ref_id="b1", title=records[2].title),
    ]
    vectors = np.array([[1.0, 0.0], [0.9, 0.436], [0.0, 1.0]], dtype=np.float32)
    record_work = {"survey": "work:survey", "bpe": "work:bpe", "other": "work:other"}
    found = find_restatements(
        find_surveys(records, CFG),
        chunks,
        references,
        ReferenceResolver(records, 95),
        record_work,
        vectors,
        [c.chunk_id for c in chunks],
        restate_min=0.8,
    )
    assert [(r.chunk_id, r.work_id, r.matched_by) for r in found] == [
        ("survey#0", "work:bpe", "title")
    ]
    assert found[0].cosine == 0.9
