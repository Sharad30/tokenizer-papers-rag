"""Synthetic manifest records for the showcase tests, one or more per decision path."""

import pytest

from rag.collect.manifest import write_manifest
from rag.collect.pipeline import RECORDS, step_dir
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
