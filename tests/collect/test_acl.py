from types import SimpleNamespace

import pytest

from rag.collect import acl
from rag.collect.schemas import LicenceSource, Source


class FakeText:
    """Stands in for the library's MarkupText (only as_text is used)."""

    def __init__(self, text: str) -> None:
        self._text = text

    def as_text(self) -> str:
        return self._text


def fake_paper(**overrides):
    fields = dict(
        full_id="2022.acl-long.1",
        title=FakeText("Tokenizer  Choice\nMatters"),
        abstract=FakeText("We compare BPE and Unigram."),
        authors=[SimpleNamespace(name=SimpleNamespace(as_first_last=lambda: "Ada Example"))],
        year="2022",
        venue_ids=("acl",),
        web_url="https://aclanthology.org/2022.acl-long.1/",
        pdf=SimpleNamespace(url="https://aclanthology.org/2022.acl-long.1.pdf"),
        doi="10.18653/v1/2022.acl-long.1",
        is_frontmatter=False,
        is_deleted=False,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize(
    ("year", "expected"),
    [
        (2015, ("CC BY-NC-SA 3.0", LicenceSource.ACL_YEAR_RULE)),
        (2016, ("CC BY 4.0", LicenceSource.ACL_YEAR_RULE)),
        (2024, ("CC BY 4.0", LicenceSource.ACL_YEAR_RULE)),
        (None, (None, LicenceSource.UNKNOWN)),
    ],
)
def test_licence_for_year(year: int | None, expected: tuple) -> None:
    assert acl.licence_for_year(year) == expected


def test_to_candidate_maps_fields() -> None:
    record = acl.to_candidate(fake_paper(), found_via=["acl_scan"])
    assert record.record_id == "acl:2022.acl-long.1"
    assert record.source is Source.ACL
    assert record.title == "Tokenizer Choice Matters"
    assert record.abstract == "We compare BPE and Unigram."
    assert record.authors == ["Ada Example"]
    assert record.year == 2022
    assert record.venue == "acl"
    assert record.pdf_url == "https://aclanthology.org/2022.acl-long.1.pdf"
    assert record.acl_id == "2022.acl-long.1"
    assert record.licence == "CC BY 4.0"


def test_to_candidate_without_abstract_or_pdf() -> None:
    record = acl.to_candidate(fake_paper(abstract=None, pdf=None), found_via=[])
    assert record.abstract is None
    assert record.pdf_url is None


def test_scan_keeps_matches_and_skips_frontmatter_deleted_and_late() -> None:
    papers = [
        fake_paper(full_id="a"),
        fake_paper(full_id="b", title=FakeText("Parsing Trees"), abstract=FakeText("Syntax.")),
        fake_paper(full_id="c", is_frontmatter=True),
        fake_paper(full_id="d", is_deleted=True),
        fake_paper(full_id="e", year="2027"),
    ]

    def matches(title: str, abstract: str) -> bool:
        return "tokenizer" in (title + " " + abstract).lower()

    records = list(acl.scan(iter(papers), matches, max_year=2026))
    assert [r.acl_id for r in records] == ["a"]
    assert records[0].found_via == ["acl_scan"]


def test_get_record_returns_none_for_unknown_id() -> None:
    anthology = SimpleNamespace(get_paper=lambda acl_id: None)
    assert acl.get_record(anthology, "X99-1", found_via=[]) is None
