"""The collect stage as six steps, each reading the previous step's folder.

    discover -> resolve -> filter -> licence -> download -> manifest

Every step is idempotent: HTTP responses are cached and PDFs are resumed,
so rerunning a step (or `all`) after a crash costs only what is missing.
Intermediate folders live under paths.raw (gitignored); the manifest goes to
paths.manifest (committed).
"""

import logging
import os
import random
from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path
from typing import Any

from rag.collect import acl, arxiv, s2
from rag.collect.download import download
from rag.collect.http import PoliteClient
from rag.collect.manifest import manifest_counts, write_manifest
from rag.collect.relevance import EmbeddingJudge, KeywordRules, sentence_transformer_encoder
from rag.collect.schemas import (
    CandidateRecord,
    DecidedBy,
    Decision,
    DownloadStatus,
    LicenceSource,
    ManifestRecord,
    Reason,
    Source,
)
from rag.collect.versions import kept_v1, needs_v1_check
from rag.common.config import Config
from rag.common.io import read_jsonl, write_jsonl, write_text_atomic
from rag.common.schemas import StageFailure
from rag.common.stats import stage_stats

logger = logging.getLogger(__name__)

# Folder layout under paths.raw. Fixed names, not settings: they are the
# stage contract between steps.
HTTP_CACHE = "http_cache"
ACL_DATA = "acl-anthology"
PDFS = "pdfs"
STEP_DIRS = {
    "discover": "discovered",
    "resolve": "resolved",
    "filter": "filtered",
    "licence": "licensed",
    "download": "downloaded",
}
RECORDS = "records.jsonl"
AUDIT_SAMPLE = "audit_sample.csv"
# arXiv id_list lookups per request; small enough for a modest response.
ARXIV_IDS_PER_CALL = 100


def step_dir(config: Config, step: str) -> Path:
    return config.paths.raw / STEP_DIRS[step]


def make_client(config: Config) -> PoliteClient:
    key = os.environ.get(config.collect.s2.api_key_env, "").strip()
    host_headers = {"api.semanticscholar.org": {"x-api-key": key}} if key else {}
    return PoliteClient(
        config.collect.http, config.paths.raw / HTTP_CACHE, host_headers=host_headers
    )


def _stats(config: Config, step: str, out_dir: Path, limit: int | None):
    return stage_stats(config, f"collect.{step}", out_dir, limit)


def _cap(configured: int, limit: int | None) -> int:
    return configured if limit is None else min(configured, limit)


def _merge(records: Iterable[CandidateRecord]) -> dict[str, CandidateRecord]:
    """Same record found several ways: keep one, union the found_via routes."""
    merged: dict[str, CandidateRecord] = {}
    for record in records:
        if record.record_id in merged:
            existing = merged[record.record_id]
            routes = sorted({*existing.found_via, *record.found_via})
            merged[record.record_id] = existing.model_copy(update={"found_via": routes})
        else:
            merged[record.record_id] = record
    return merged


# --- 1. discover -----------------------------------------------------------


def discover(config: Config, limit: int | None = None) -> None:
    c = config.collect
    out = step_dir(config, "discover")
    with _stats(config, "discover", out, limit) as stats, make_client(config) as client:
        arxiv_records = _merge(
            record
            for query in c.arxiv.queries
            for record in arxiv.search(
                client,
                query,
                c.arxiv.categories,
                config.corpus.max_date,
                c.arxiv.page_size,
                _cap(c.arxiv.max_results_per_query, limit),
            )
        )
        write_jsonl(out / "arxiv.jsonl", arxiv_records.values())

        acl_records: dict[str, CandidateRecord] = {}
        if c.acl.enabled:
            anthology = acl.load_anthology(config.paths.raw / ACL_DATA)
            rules = KeywordRules(c.relevance)
            scanned = acl.scan(
                anthology.papers(), rules.has_topic_term, config.corpus.max_date.year
            )
            acl_records = _merge(islice(scanned, limit))
        write_jsonl(out / "acl.jsonl", acl_records.values())

        s2_records = _merge(_snowball_records(config, client, limit))
        write_jsonl(out / "s2.jsonl", s2_records.values())

        stats["counts"].update(
            arxiv=len(arxiv_records),
            acl=len(acl_records),
            s2=len(s2_records),
            network_calls=client.counts["network"],
            cache_hits=client.counts["cache_hit"],
        )


def _snowball_records(
    config: Config, client: PoliteClient, limit: int | None
) -> Iterator[CandidateRecord]:
    c = config.collect.s2
    # The anchors themselves are candidates too.
    for anchor_id, paper in zip(
        c.anchors, s2.resolve_batch(client, c.anchors, c.batch_size), strict=True
    ):
        if paper is not None:
            yield s2.to_candidate(paper, found_via=[f"anchor:{anchor_id}"])
    for anchor_id in c.anchors:
        for direction in c.directions:
            tag = "snowball_ref" if direction == "references" else "snowball_cit"
            papers = s2.snowball(
                client, anchor_id, direction, c.page_size, _cap(c.max_items_per_anchor, limit)
            )
            for paper in papers:
                record = s2.to_candidate(paper, found_via=[f"{tag}:{anchor_id}"])
                if s2.within_cutoff(record, config.corpus.max_date):
                    yield record


# --- 2. resolve ------------------------------------------------------------


def resolve(config: Config, limit: int | None = None) -> None:
    """Link arXiv/ACL records to S2 ids; fold S2 candidates into the records they duplicate."""
    src, out = step_dir(config, "discover"), step_dir(config, "resolve")
    with _stats(config, "resolve", out, limit) as stats, make_client(config) as client:
        primary = [
            ManifestRecord(**r.model_dump())
            for name in ("arxiv.jsonl", "acl.jsonl")
            for r in read_jsonl(src / name, CandidateRecord)
        ][:limit]
        papers = s2.resolve_batch(
            client, [s2.lookup_id(r) or "" for r in primary], config.collect.s2.batch_size
        )
        primary = [_link(r, p) for r, p in zip(primary, papers, strict=True)]

        index = _id_index(primary)
        leftovers: list[ManifestRecord] = []
        folded = 0
        for candidate in read_jsonl(src / "s2.jsonl", CandidateRecord)[:limit]:
            target = _find(index, candidate)
            if target is None:
                leftovers.append(ManifestRecord(**candidate.model_dump()))
                continue
            folded += 1
            for record in target:
                record.found_via = sorted({*record.found_via, *candidate.found_via})

        records = primary + leftovers
        write_jsonl(out / RECORDS, records)
        stats["counts"].update(
            records=len(records),
            linked_to_s2=sum(r.s2_id is not None for r in primary),
            s2_folded_into_existing=folded,
            s2_only=len(leftovers),
            network_calls=client.counts["network"],
        )


def _link(record: ManifestRecord, paper: dict[str, Any] | None) -> ManifestRecord:
    if paper is None:
        return record
    linked = s2.to_candidate(paper, found_via=[])
    # Fill gaps only: a record's own source is the authority for its own fields.
    return record.model_copy(
        update={
            "s2_id": linked.s2_id,
            "arxiv_id": record.arxiv_id or linked.arxiv_id,
            "acl_id": record.acl_id or linked.acl_id,
            "doi": record.doi or linked.doi,
            "venue": record.venue or linked.venue,
            "abstract": record.abstract or linked.abstract,
            "publication_types": linked.publication_types,
        }
    )


def _id_index(records: list[ManifestRecord]) -> dict[str, list[ManifestRecord]]:
    """Every record reachable by any of its ids (s2, arXiv, ACL)."""
    index: dict[str, list[ManifestRecord]] = {}
    for record in records:
        for key in _id_keys(record):
            index.setdefault(key, []).append(record)
    return index


def _id_keys(record: CandidateRecord) -> list[str]:
    keys = [f"s2:{record.s2_id}"] if record.s2_id else []
    keys += [f"arxiv:{record.arxiv_id}"] if record.arxiv_id else []
    keys += [f"acl:{record.acl_id}"] if record.acl_id else []
    return keys


def _find(
    index: dict[str, list[ManifestRecord]], record: CandidateRecord
) -> list[ManifestRecord] | None:
    for key in _id_keys(record):
        if key in index:
            return index[key]
    return None


# --- 3. filter -------------------------------------------------------------


def filter_records(config: Config, limit: int | None = None) -> None:
    """Decide relevance, then add linked copies and revised v1s of accepted papers."""
    rc = config.collect.relevance
    src, out = step_dir(config, "resolve"), step_dir(config, "filter")
    with _stats(config, "filter", out, limit) as stats, make_client(config) as client:
        records = read_jsonl(src / RECORDS, ManifestRecord)[:limit]
        rules = KeywordRules(rc)
        decided = [r.with_decision(rules.decide(r.title, r.abstract or "")) for r in records]

        ambiguous = [i for i, r in enumerate(decided) if r.relevance is Decision.AMBIGUOUS]
        if ambiguous:
            judge = EmbeddingJudge(
                sentence_transformer_encoder(rc.embedding_model, rc.embedding_batch_size),
                rc.topic_description,
                rc.embedding_threshold,
            )
            scores = judge.scores([decided[i] for i in ambiguous])
            for i, score in zip(ambiguous, scores, strict=True):
                keyword = rules.decide(decided[i].title, decided[i].abstract or "")
                decided[i] = decided[i].with_decision(judge.resolve(keyword, score))

        anthology = (
            acl.load_anthology(config.paths.raw / ACL_DATA) if config.collect.acl.enabled else None
        )
        materialised = _materialise_s2(client, anthology, decided)
        linked = _linked_copies(client, anthology, materialised)
        v1s = _revised_v1s(client, config, materialised + linked)
        final = materialised + linked + v1s
        write_jsonl(out / RECORDS, final)
        stats["counts"].update(
            decided=len(decided),
            sent_to_embedding_judge=len(ambiguous),
            accepted=sum(r.relevance is Decision.ACCEPT for r in final),
            linked_copies_added=len(linked),
            v1_kept=len(v1s),
            network_calls=client.counts["network"],
        )


def _materialise_s2(
    client: PoliteClient, anthology: Any, records: list[ManifestRecord]
) -> list[ManifestRecord]:
    """An accepted S2-only record that is really on arXiv or ACL becomes that record.

    S2 merges a paper's versions into one entry, so it isn't a manifestation;
    the arXiv (or ACL) record is. The decision carries over unchanged: it was
    made on the same paper's title and abstract.
    """
    targets = [r for r in records if r.source is Source.S2 and r.relevance is Decision.ACCEPT]
    on_arxiv = {r.arxiv_id: r for r in targets if r.arxiv_id}
    fetched = {a.arxiv_id: a for a in _fetch_arxiv(client, list(on_arxiv))}
    replaced: dict[str, ManifestRecord] = {}
    for record in targets:
        new: CandidateRecord | None = None
        if record.arxiv_id and record.arxiv_id in fetched:
            new = fetched[record.arxiv_id]
        elif record.acl_id and anthology is not None:
            new = acl.get_record(anthology, record.acl_id, found_via=[])
        if new is not None:
            replaced[record.record_id] = _carry_over(record, new)
    return [replaced.get(r.record_id, r) for r in records]


def _carry_over(s2_record: ManifestRecord, new: CandidateRecord) -> ManifestRecord:
    return ManifestRecord(
        **new.model_dump(exclude={"found_via"}),
        found_via=s2_record.found_via,
    ).model_copy(
        update={
            "s2_id": s2_record.s2_id,
            "arxiv_id": new.arxiv_id or s2_record.arxiv_id,
            "acl_id": new.acl_id or s2_record.acl_id,
            "doi": new.doi or s2_record.doi,
            "venue": new.venue or s2_record.venue,
            "publication_types": s2_record.publication_types,
            "relevance": s2_record.relevance,
            "decided_by": s2_record.decided_by,
            "reason": s2_record.reason,
            "matched_terms": s2_record.matched_terms,
            "embedding_score": s2_record.embedding_score,
        }
    )


def _linked_copies(
    client: PoliteClient, anthology: Any, records: list[ManifestRecord]
) -> list[ManifestRecord]:
    """The ACL copy of an accepted arXiv paper and vice versa, as separate records."""
    # Only a record *of that source* counts as the copy being present. An
    # ACL record that knows its own arXiv id is not the arXiv record.
    present = {f"arxiv:{r.arxiv_id}" for r in records if r.source is Source.ARXIV}
    present |= {f"acl:{r.acl_id}" for r in records if r.source is Source.ACL}
    accepted = [r for r in records if r.relevance is Decision.ACCEPT]
    copies: list[tuple[ManifestRecord, CandidateRecord]] = []
    if anthology is not None:
        for r in accepted:
            if r.source is Source.ARXIV and r.acl_id and f"acl:{r.acl_id}" not in present:
                found = acl.get_record(anthology, r.acl_id, found_via=[])
                if found is not None:
                    copies.append((r, found))
                    present.add(f"acl:{r.acl_id}")
    need_arxiv = {
        r.arxiv_id: r
        for r in accepted
        if r.source is Source.ACL and r.arxiv_id and f"arxiv:{r.arxiv_id}" not in present
    }
    for found in _fetch_arxiv(client, list(need_arxiv)):
        copies.append((need_arxiv[found.arxiv_id], found))
    return [_inherit(origin, copy, Reason.LINKED_MANIFESTATION) for origin, copy in copies]


def _revised_v1s(
    client: PoliteClient, config: Config, records: list[ManifestRecord]
) -> list[ManifestRecord]:
    latest = [
        r
        for r in records
        if r.relevance is Decision.ACCEPT and r.source is Source.ARXIV and needs_v1_check(r)
    ]
    v1s = {v.arxiv_id: v for v in _fetch_arxiv(client, [f"{r.arxiv_id}v1" for r in latest])}
    threshold = config.collect.versions.keep_v1_if_abstract_similarity_below
    kept = []
    for record in latest:
        v1 = v1s.get(record.arxiv_id)
        if v1 is not None:
            candidate = kept_v1(record, ManifestRecord(**v1.model_dump()), threshold)
            if candidate is not None:
                kept.append(candidate)
    return kept


def _inherit(origin: ManifestRecord, copy: CandidateRecord, reason: Reason) -> ManifestRecord:
    return ManifestRecord(
        **copy.model_dump(exclude={"found_via"}), found_via=origin.found_via
    ).model_copy(
        update={
            "s2_id": origin.s2_id,
            "arxiv_id": copy.arxiv_id or origin.arxiv_id,
            "acl_id": copy.acl_id or origin.acl_id,
            "relevance": origin.relevance,
            "decided_by": DecidedBy.INHERITED,
            "reason": reason,
            "matched_terms": origin.matched_terms,
            "embedding_score": origin.embedding_score,
            "derived_from": origin.record_id,
        }
    )


def _fetch_arxiv(client: PoliteClient, ids: list[str]) -> list[CandidateRecord]:
    records: list[CandidateRecord] = []
    for start in range(0, len(ids), ARXIV_IDS_PER_CALL):
        records += arxiv.fetch_by_ids(client, ids[start : start + ARXIV_IDS_PER_CALL])
    return records


# --- 4. licence ------------------------------------------------------------


def licence(config: Config, limit: int | None = None) -> None:
    """arXiv licences via OAI (one call per accepted arXiv paper); others are already set."""
    src, out = step_dir(config, "filter"), step_dir(config, "licence")
    with _stats(config, "licence", out, limit) as stats, make_client(config) as client:
        records = read_jsonl(src / RECORDS, ManifestRecord)
        looked_up = 0
        result = []
        for record in records:
            if record.source is Source.ARXIV and record.relevance is Decision.ACCEPT:
                if limit is None or looked_up < limit:
                    oai = arxiv.fetch_oai_record(client, record.arxiv_id or "")
                    record = record.model_copy(update=arxiv.licence_fields(oai))
                    looked_up += 1
            if record.licence_source is None:
                record = record.model_copy(update={"licence_source": LicenceSource.UNKNOWN})
            result.append(record)
        write_jsonl(out / RECORDS, result)
        stats["counts"].update(oai_lookups=looked_up, network_calls=client.counts["network"])


# --- 5. download -----------------------------------------------------------


def download_pdfs(config: Config, limit: int | None = None) -> None:
    src, out = step_dir(config, "licence"), step_dir(config, "download")
    enabled = set(config.collect.download.enabled_sources)
    with _stats(config, "download", out, limit) as stats, make_client(config) as client:
        records = read_jsonl(src / RECORDS, ManifestRecord)
        attempted = 0
        result = []
        for record in records:
            wanted = record.relevance is Decision.ACCEPT and record.source.value in enabled
            if wanted and (limit is None or attempted < limit):
                record = download(client, record, config.paths.raw / PDFS)
                attempted += 1
                if record.download_status in (DownloadStatus.FAILED, DownloadStatus.NOT_PDF):
                    stats["failures"].append(
                        StageFailure(item_id=record.record_id, reason=record.download_status.value)
                    )
            result.append(record)
        write_jsonl(out / RECORDS, result)
        stats["counts"].update(attempted=attempted, network_calls=client.counts["network"])


# --- 6. manifest -----------------------------------------------------------


def manifest(config: Config, limit: int | None = None) -> None:
    src, out = step_dir(config, "download"), config.paths.manifest
    with _stats(config, "manifest", out, limit) as stats:
        records = read_jsonl(src / RECORDS, ManifestRecord)[:limit]
        write_manifest(records, out)
        stats["counts"].update(manifest_counts(records))


STEPS = {
    "discover": discover,
    "resolve": resolve,
    "filter": filter_records,
    "licence": licence,
    "download": download_pdfs,
    "manifest": manifest,
}


def run_all(config: Config, limit: int | None = None) -> None:
    for name, step in STEPS.items():
        logger.info("collect: starting %s", name)
        step(config, limit)


# --- audit sample ----------------------------------------------------------


def audit_sample(config: Config, per_decision: int, seed: int) -> Path:
    """Accepted and rejected records for Sharad to label, stratified by source.

    Linked copies and kept v1s are excluded: they inherit a decision rather
    than having one made, so they would only re-test the same paper.
    Reads the download step's full records, not the committed manifest,
    because labelling a reject needs its abstract (D-010 drops those).
    """
    records = [
        r
        for r in read_jsonl(step_dir(config, "download") / RECORDS, ManifestRecord)
        if r.decided_by is not DecidedBy.INHERITED
    ]
    rng = random.Random(seed)
    rows: list[ManifestRecord] = []
    for decision in (Decision.ACCEPT, Decision.REJECT):
        pool = [r for r in records if r.relevance is decision]
        rows += _stratified(pool, per_decision, rng)
    path = config.paths.manifest / AUDIT_SAMPLE
    write_text_atomic(path, _audit_csv(rows))
    return path


def _stratified(pool: list[ManifestRecord], n: int, rng: random.Random) -> list[ManifestRecord]:
    """Round-robin over sources so a big source can't crowd out a small one."""
    by_source: dict[Source, list[ManifestRecord]] = {}
    for record in sorted(pool, key=lambda r: r.record_id):
        by_source.setdefault(record.source, []).append(record)
    for group in by_source.values():
        rng.shuffle(group)
    picked: list[ManifestRecord] = []
    while len(picked) < n and any(by_source.values()):
        for source in sorted(by_source):
            if by_source[source] and len(picked) < n:
                picked.append(by_source[source].pop())
    return picked


def _audit_csv(rows: list[ManifestRecord]) -> str:
    import csv
    import io

    buffer = io.StringIO()
    columns = ["record_id", "source", "title", "abstract", "relevance", "reason", "embedding_score"]
    writer = csv.writer(buffer, lineterminator="\n")
    # The last two columns are for Sharad: in_scope = yes / no, per D-005.
    writer.writerow([*columns, "sharad_in_scope", "sharad_notes"])
    for row in rows:
        writer.writerow([getattr(row, c) for c in columns] + ["", ""])
    return buffer.getvalue()
