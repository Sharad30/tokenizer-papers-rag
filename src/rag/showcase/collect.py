"""What the Collect page shows, computed from the manifest every time (D-038).

Nothing here is a stored number or a hand-picked paper. Counts come from
manifest.json + rejected.csv.gz, examples are sampled from the records on
each decision path, and an example's explanation re-runs today's keyword
rules on it, so the page shows when the config has moved on since
collection.
"""

import json
import random
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from rag.collect import arxiv, s2
from rag.collect.config import RelevanceConfig
from rag.collect.manifest import (
    MANIFEST_JSON,
    REJECTED_CSV_GZ,
    discovery_routes,
    read_manifest,
    read_rejected_records,
)
from rag.collect.pipeline import RECORDS, step_dir
from rag.collect.relevance import KeywordRules, RuleTrace
from rag.collect.schemas import Decision, DownloadStatus, ManifestRecord, Reason, Source
from rag.common.config import Config
from rag.common.schemas import StrictModel
from rag.eval.identity import identity_groups
from rag.showcase.sources import Snapshot, read_snapshot, snapshot_file, stale_sources

STAGE = "collect"
REJECT_ABSTRACTS = "reject_abstracts"  # the snapshot's one field: record_id -> abstract
DOWNLOADED = {DownloadStatus.DOWNLOADED, DownloadStatus.ALREADY_PRESENT}


# --- Decision paths ---------------------------------------------------------


class DecisionPath(StrictModel):
    key: str
    label: str
    decision: Decision
    reason: Reason
    how: str  # may name {min_sentences} and {threshold}, filled from config


PATHS: list[DecisionPath] = [
    DecisionPath(
        key="title_term",
        label="Accepted: topic term in the title",
        decision=Decision.ACCEPT,
        reason=Reason.TITLE_TERM,
        how="A tokenizer term (tokenizer, BPE, subword, byte-level, …) is in the title, "
        "and the title names no excluded sense of 'token'.",
    ),
    DecisionPath(
        key="embedding_above",
        label="Accepted by the embedding judge",
        decision=Decision.ACCEPT,
        reason=Reason.EMBEDDING_ABOVE,
        how="Topic terms only in the abstract, in at least {min_sentences} sentences, so "
        "ambiguous. Its cosine to the written topic description reached {threshold}.",
    ),
    DecisionPath(
        key="linked_copy",
        label="Accepted: linked copy",
        decision=Decision.ACCEPT,
        reason=Reason.LINKED_MANIFESTATION,
        how="Another manifestation (e.g. the ACL copy of an arXiv paper) of an accepted "
        "paper. It inherits that decision, so copies never disagree.",
    ),
    DecisionPath(
        key="earlier_version",
        label="Accepted: earlier arXiv version",
        decision=Decision.ACCEPT,
        reason=Reason.EARLIER_VERSION,
        how="An arXiv v1 kept because its abstract was revised later (D-009). It inherits "
        "the latest version's decision.",
    ),
    DecisionPath(
        key="off_topic",
        label="Rejected: another sense of 'token'",
        decision=Decision.REJECT,
        reason=Reason.OFF_TOPIC_TERM,
        how="The title names a modality or domain outside the text-only boundary (vision, "
        "speech, proteins, graphs, …), even if it also has a tokenizer term.",
    ),
    DecisionPath(
        key="passing_mention",
        label="Rejected: passing mention",
        decision=Decision.REJECT,
        reason=Reason.PASSING_MENTION,
        how="Topic terms only in the abstract, in fewer than {min_sentences} sentences: "
        "the paper uses a tokenizer rather than studies one.",
    ),
    DecisionPath(
        key="embedding_below",
        label="Rejected by the embedding judge",
        decision=Decision.REJECT,
        reason=Reason.EMBEDDING_BELOW,
        how="Ambiguous like the accepted ones, but its cosine to the topic description "
        "stayed below {threshold}.",
    ),
    DecisionPath(
        key="no_topic_term",
        label="Rejected: no topic term",
        decision=Decision.REJECT,
        reason=Reason.NO_TOPIC_TERM,
        how="No topic term anywhere in the title or abstract. Mostly papers the citation "
        "snowball brought in that only cite a tokenizer paper.",
    ),
]
PATH_BY_KEY = {p.key: p for p in PATHS}


def describe(path: DecisionPath, relevance: RelevanceConfig) -> str:
    return path.how.format(
        min_sentences=relevance.min_abstract_sentences_with_term,
        threshold=relevance.embedding_threshold,
    )


def path_of(record: ManifestRecord) -> str | None:
    for path in PATHS:
        if record.relevance is path.decision and record.reason is path.reason:
            return path.key
    return None


# --- Loading ----------------------------------------------------------------

AbstractOrigin = Literal["live", "snapshot", "none"]


class CollectData(StrictModel):
    accepted: list[ManifestRecord]
    rejected: list[ManifestRecord]  # abstracts filled in where known
    abstracts_from: AbstractOrigin  # where the rejected abstracts came from
    snapshot: Snapshot | None  # set when abstracts_from == "snapshot"
    stale: list[str]  # snapshot sources that changed since it was built


def source_files(config: Config) -> dict[str, Path]:
    """Every file the Collect page reads, by name (also the snapshot's source names)."""
    return {
        "manifest": config.paths.manifest / MANIFEST_JSON,
        "rejected": config.paths.manifest / REJECTED_CSV_GZ,
        "filtered": step_dir(config, "filter") / RECORDS,
    }


def load(config: Config) -> CollectData:
    files = source_files(config)
    accepted = read_manifest(config.paths.manifest)
    rejected = read_rejected_records(config.paths.manifest)
    snapshot = None
    stale: list[str] = []
    if files["filtered"].exists():
        abstracts = read_abstracts(files["filtered"], {r.record_id for r in rejected})
        origin: AbstractOrigin = "live"
    else:
        snapshot = read_snapshot(snapshot_file(config.paths.snapshot, STAGE))
        abstracts = snapshot.data[REJECT_ABSTRACTS] if snapshot else {}
        origin = "snapshot" if snapshot else "none"
        stale = stale_sources(snapshot, files) if snapshot else []
    rejected = [r.model_copy(update={"abstract": abstracts.get(r.record_id)}) for r in rejected]
    return CollectData(
        accepted=accepted, rejected=rejected, abstracts_from=origin, snapshot=snapshot, stale=stale
    )


def read_abstracts(filtered_records: Path, wanted: set[str]) -> dict[str, str]:
    """record_id -> abstract from the filter step's output (it has every candidate's)."""
    abstracts = {}
    with filtered_records.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)  # plain json: only two fields are needed
            if row["record_id"] in wanted and row.get("abstract"):
                abstracts[row["record_id"]] = row["abstract"]
    return abstracts


def snapshot_data(data: CollectData, pool_per_path: int, seed: int) -> dict:
    """A seeded sample of rejected abstracts per path: all a fresh clone lacks."""
    abstracts = {}
    for path in PATHS:
        if path.decision is not Decision.REJECT:
            continue  # accepted abstracts are in the committed manifest
        pool = [r for r in data.rejected if path_of(r) == path.key and r.abstract]
        for r in pick(pool, pool_per_path, seed):
            abstracts[r.record_id] = r.abstract
    return {REJECT_ABSTRACTS: dict(sorted(abstracts.items()))}


# --- Overview numbers -------------------------------------------------------


class SourceCount(StrictModel):
    source: str
    candidates: int
    accepted: int


class YearCount(StrictModel):
    year: int
    source: str
    papers: int


class JudgedScore(StrictModel):
    score: float
    accepted: bool


class CollectOverview(StrictModel):
    candidates: int
    accepted: int
    rejected: int
    downloaded: int  # accepted with a PDF
    indexed: int  # accepted and chunked into the index (Phase 2)
    by_source: list[SourceCount]
    by_path: dict[str, int]  # path key -> records
    routes: dict[str, int]  # discovery-route combination -> accepted records
    per_year: list[YearCount]  # accepted records with a year
    licences: dict[str, int]  # licence source -> accepted records
    downloads: dict[str, int]  # download status -> accepted records
    missing: dict[str, int]  # metadata field -> accepted records without it
    judged: list[JudgedScore]  # every paper the embedding judge decided


def overview(
    accepted: Sequence[ManifestRecord], rejected: Sequence[ManifestRecord]
) -> CollectOverview:
    everything = [*accepted, *rejected]
    candidates = Counter(r.source for r in everything)
    accepted_by_source = Counter(r.source for r in accepted)
    paths = Counter(path_of(r) for r in everything)
    years = Counter((r.year, r.source) for r in accepted if r.year is not None)
    return CollectOverview(
        candidates=len(everything),
        accepted=len(accepted),
        rejected=len(rejected),
        downloaded=sum(r.download_status in DOWNLOADED for r in accepted),
        indexed=sum(bool(r.included_in_index) for r in accepted),
        by_source=[
            SourceCount(source=s, candidates=candidates[s], accepted=accepted_by_source[s])
            for s in Source
        ],
        by_path={p.key: paths[p.key] for p in PATHS},
        routes=_most_common(" + ".join(sorted(discovery_routes(r))) for r in accepted),
        per_year=[YearCount(year=y, source=s, papers=n) for (y, s), n in sorted(years.items())],
        licences=_most_common(str(r.licence_source or "none") for r in accepted),
        downloads=_most_common(str(r.download_status) for r in accepted),
        missing={
            "venue": sum(not r.venue for r in accepted),
            "year": sum(r.year is None for r in accepted),
            "authors": sum(not r.authors for r in accepted),
            "abstract": sum(not r.abstract for r in accepted),
        },
        judged=[
            JudgedScore(score=r.embedding_score, accepted=r.relevance is Decision.ACCEPT)
            for r in everything
            if r.reason in (Reason.EMBEDDING_ABOVE, Reason.EMBEDDING_BELOW)
            and r.embedding_score is not None
        ],
    )


def _most_common(values) -> dict[str, int]:
    return dict(Counter(values).most_common())


# --- Sources ----------------------------------------------------------------


class SourceInfo(StrictModel):
    source: str
    name: str
    how: str
    details: list[str]  # seed queries, anchors, …
    hosts: dict[str, float]  # host -> minimum seconds between requests
    candidates: int
    accepted: int


def sources_summary(config: Config, data: CollectData, view: CollectOverview) -> list[SourceInfo]:
    """How each source was searched, read from config; its counts, from the data."""
    c = config.collect
    counts = {s.source: s for s in view.by_source}

    def hosts(api_hosts: list[str], source: Source) -> dict[str, float]:
        # The PDF host comes from the records themselves, not from a list here.
        pdf_hosts = Counter(
            urlparse(r.pdf_url).hostname for r in data.accepted if r.source is source and r.pdf_url
        )
        names = [*api_hosts, *(h for h, _ in pdf_hosts.most_common(1) if h not in api_hosts)]
        return {h: c.http.min_interval_s.get(h, c.http.default_min_interval_s) for h in names}

    return [
        SourceInfo(
            source=Source.ARXIV,
            name="arXiv API",
            how=f"{len(c.arxiv.queries)} seed keyword queries over titles and abstracts in "
            f"{', '.join(c.arxiv.categories)}, up to {config.corpus.max_date}.",
            details=[f"{q.name}: {q.query}" for q in c.arxiv.queries],
            hosts=hosts(
                [urlparse(arxiv.API_URL).hostname, urlparse(arxiv.OAI_URL).hostname], Source.ARXIV
            ),
            candidates=counts[Source.ARXIV].candidates,
            accepted=counts[Source.ARXIV].accepted,
        ),
        SourceInfo(
            source=Source.S2,
            name="Semantic Scholar",
            how="Resolves every record's ids, then a one-hop snowball "
            f"({' and '.join(c.s2.directions)}) from {len(c.s2.anchors)} anchor papers, "
            f"at most {c.s2.max_items_per_anchor:,} per anchor.",
            details=list(c.s2.anchors),
            hosts=hosts([urlparse(s2.API_URL).hostname], Source.S2),
            candidates=counts[Source.S2].candidates,
            accepted=counts[Source.S2].accepted,
        ),
        SourceInfo(
            source=Source.ACL,
            name="ACL Anthology",
            how="Scans the whole Anthology's metadata through the official library (a local "
            "clone, no API calls), keeping papers with a topic term."
            if c.acl.enabled
            else "Disabled in config.",
            details=[],
            hosts=hosts([], Source.ACL),
            candidates=counts[Source.ACL].candidates,
            accepted=counts[Source.ACL].accepted,
        ),
    ]


# --- Examples ---------------------------------------------------------------


def example_pool(
    records: Sequence[ManifestRecord],
    path_key: str,
    sources: Sequence[str] = (),
    years: tuple[int, int] | None = None,
) -> list[ManifestRecord]:
    """Records on one decision path, optionally narrowed by source and year.

    Records with an abstract are preferred: the explanation needs one. Only
    when none has an abstract are the rest returned.
    """
    pool = [
        r
        for r in records
        if path_of(r) == path_key
        and (not sources or r.source in sources)
        and (years is None or (r.year is not None and years[0] <= r.year <= years[1]))
    ]
    with_abstract = [r for r in pool if r.abstract]
    return with_abstract or pool


def pick(pool: Sequence[ManifestRecord], k: int, seed: int) -> list[ManifestRecord]:
    """A reproducible sample: the same seed shows the same papers."""
    ordered = sorted(pool, key=lambda r: r.record_id)  # independent of input order
    return random.Random(seed).sample(ordered, min(k, len(ordered)))


class Explanation(StrictModel):
    trace: RuleTrace  # today's keyword rules on this paper
    has_abstract: bool
    embedding_score: float | None  # as recorded at collection
    threshold: float  # from today's config
    today: Decision  # what today's rules decide (ambiguous if it needs a score we don't have)
    recorded: Decision | None
    inherited_from: str | None  # copies take their source paper's decision
    agrees: bool | None  # None when the decision was inherited or can't be redone


def explain(record: ManifestRecord, rules: KeywordRules, relevance: RelevanceConfig) -> Explanation:
    trace = rules.trace(record.title, record.abstract or "")
    today = trace.decision.decision
    if today is Decision.AMBIGUOUS and record.embedding_score is not None:
        accepted = record.embedding_score >= relevance.embedding_threshold
        today = Decision.ACCEPT if accepted else Decision.REJECT
    inherited = record.reason in (Reason.LINKED_MANIFESTATION, Reason.EARLIER_VERSION)
    comparable = not inherited and today is not Decision.AMBIGUOUS
    return Explanation(
        trace=trace,
        has_abstract=bool(record.abstract),
        embedding_score=record.embedding_score,
        threshold=relevance.embedding_threshold,
        today=today,
        recorded=record.relevance,
        inherited_from=record.derived_from if inherited else None,
        agrees=(today is record.relevance) if comparable else None,
    )


# --- Metadata ---------------------------------------------------------------


def linked_copies(records: Sequence[ManifestRecord]) -> dict[str, list[ManifestRecord]]:
    """record_id -> every record of the same paper, by shared ids (eval/identity.py)."""
    groups = identity_groups(records)
    members: dict[str, list[ManifestRecord]] = {}
    for r in records:
        members.setdefault(groups[r.record_id], []).append(r)
    return {r.record_id: members[groups[r.record_id]] for r in records}


def search(records: Sequence[ManifestRecord], text: str, limit: int) -> list[ManifestRecord]:
    """Records whose title or id contains the text, case-insensitively."""
    needle = text.strip().casefold()
    if not needle:
        return []
    hits = [r for r in records if needle in r.title.casefold() or needle in r.record_id.casefold()]
    return hits[:limit]
