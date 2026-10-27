"""The parse stage as four steps, each reading the previous step's folder.

    grobid -> structure -> chunk -> mark

    grobid      accepted PDFs -> parsed/tei/*.tei.xml (the TEI files are the cache)
    structure   TEI (or the PyMuPDF fallback) -> parsed/papers/*.json + references.jsonl
    chunk       papers -> parsed/chunks/chunks.jsonl + status.jsonl
    mark        status -> included_in_index in the committed manifest (D-021)

`mark` is the one documented exception to "a stage writes only its own
folder": the manifest is the single record of the corpus. Rerun it after
`rag collect manifest`, which rewrites the manifest without the flag.
"""

import logging
import statistics
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rag.collect.download import pdf_path
from rag.collect.manifest import read_manifest, write_accepted
from rag.collect.pipeline import PDFS
from rag.collect.schemas import DownloadStatus, ManifestRecord
from rag.common.config import Config
from rag.common.io import STATS_FILENAME, read_jsonl, write_bytes_atomic, write_json, write_jsonl
from rag.common.schemas import StageFailure, StageStats
from rag.common.stats import stage_stats
from rag.parse.chunker import SPECIAL_TOKENS, Chunker
from rag.parse.fallback import parse_pdf
from rag.parse.grobid import GrobidClient, GrobidOutcome, GrobidResult
from rag.parse.schemas import (
    Chunk,
    IndexStatus,
    ParseOutcome,
    ParseStatus,
    Reference,
    StructuredPaper,
)
from rag.parse.tei import parse_references, parse_tei
from rag.parse.tokens import HFTokenCounter, TokenCounter

logger = logging.getLogger(__name__)

STEP_DIRS = {"grobid": "tei", "structure": "papers", "chunk": "chunks"}
RECORDS = "records.jsonl"
REFERENCES = "references.jsonl"
CHUNKS = "chunks.jsonl"
STATUS = "status.jsonl"
HAS_PDF = {DownloadStatus.DOWNLOADED, DownloadStatus.ALREADY_PRESENT}
PROGRESS_EVERY = 50
# Upper edges of the token-length histogram in stats.json.
TOKEN_BUCKETS = (50, 100, 200, 300, 400, 512)


class GrobidUnavailable(RuntimeError):
    """GROBID isn't answering; the message says how to start it."""


def step_dir(config: Config, step: str) -> Path:
    return config.paths.parsed / STEP_DIRS[step]


def safe_name(record_id: str) -> str:
    # Same rule as the PDF file names: ":" and "/" are not path-safe.
    return pdf_path(Path(), record_id).stem


def _accepted(config: Config, limit: int | None) -> list[ManifestRecord]:
    return read_manifest(config.paths.manifest)[:limit]


# --- grobid ---------------------------------------------------------------------


def grobid(
    config: Config,
    limit: int | None = None,
    client_factory: Callable[[], GrobidClient] | None = None,
) -> None:
    out = step_dir(config, "grobid")
    pdf_dir = config.paths.raw / PDFS
    records = _accepted(config, limit)
    with stage_stats(config, "parse.grobid", out, limit) as stats:
        statuses: dict[str, ParseStatus] = {}
        todo: list[ManifestRecord] = []
        for r in records:
            if r.download_status not in HAS_PDF:
                statuses[r.record_id] = ParseStatus(
                    record_id=r.record_id, outcome=ParseOutcome.NO_PDF
                )
            elif (out / f"{safe_name(r.record_id)}.tei.xml").is_file():
                statuses[r.record_id] = ParseStatus(
                    record_id=r.record_id, outcome=ParseOutcome.GROBID, detail="cached"
                )
            else:
                todo.append(r)
        if todo:
            make_client = client_factory or (lambda: GrobidClient(config.parse.grobid))
            with make_client() as client:
                if not client.is_alive():
                    raise GrobidUnavailable(
                        f"GROBID is not answering at {config.parse.grobid.url}. "
                        "Start Docker, then `make grobid-up`, and wait for /api/isalive."
                    )

                def run(record: ManifestRecord) -> tuple[ManifestRecord, GrobidResult, float]:
                    tick = time.monotonic()
                    result = client.process(pdf_path(pdf_dir, record.record_id))
                    return record, result, time.monotonic() - tick

                with ThreadPoolExecutor(config.parse.grobid.workers) as pool:
                    for done, (record, result, seconds) in enumerate(pool.map(run, todo), 1):
                        statuses[record.record_id] = _grobid_status(out, record, result, seconds)
                        if done % PROGRESS_EVERY == 0:
                            logger.info("GROBID: %d/%d PDFs", done, len(todo))
        ordered = [statuses[r.record_id] for r in records]
        write_jsonl(out / RECORDS, ordered)
        counts = Counter(f"outcome.{s.outcome}" for s in ordered)
        stats["counts"].update(
            records=len(ordered),
            sent_to_grobid=len(todo),
            cached=sum(s.detail == "cached" for s in ordered),
            **dict(sorted(counts.items())),
        )
        seconds = [s.seconds for s in ordered if s.seconds is not None]
        if seconds:
            stats["counts"]["median_ms_per_pdf"] = round(1000 * statistics.median(seconds))
        stats["failures"] += [
            StageFailure(item_id=s.record_id, reason=s.detail or "")
            for s in ordered
            if s.outcome is ParseOutcome.FAILED
        ]


def _grobid_status(
    out: Path, record: ManifestRecord, result: GrobidResult, seconds: float
) -> ParseStatus:
    if result.outcome is GrobidOutcome.OK and result.tei:
        write_bytes_atomic(out / f"{safe_name(record.record_id)}.tei.xml", result.tei)
        return ParseStatus(record_id=record.record_id, outcome=ParseOutcome.GROBID, seconds=seconds)
    # Not cached: a rerun tries GROBID again. The structure step may still
    # rescue the paper with the fallback.
    logger.warning("GROBID failed for %s: %s", record.record_id, result.detail)
    return ParseStatus(
        record_id=record.record_id,
        outcome=ParseOutcome.FAILED,
        detail=result.detail,
        seconds=seconds,
    )


# --- structure ------------------------------------------------------------------


def structure(config: Config, limit: int | None = None) -> None:
    src, out = step_dir(config, "grobid"), step_dir(config, "structure")
    manifest = {r.record_id: r for r in read_manifest(config.paths.manifest)}
    with stage_stats(config, "parse.structure", out, limit) as stats:
        counts: Counter[str] = Counter()
        statuses: list[ParseStatus] = []
        references: list[Reference] = []
        for status in read_jsonl(src / RECORDS, ParseStatus)[:limit]:
            record = manifest[status.record_id]
            paper, refs, status = _structure_one(config, record, status, counts)
            statuses.append(status)
            counts[f"outcome.{status.outcome}"] += 1
            if paper is None:
                if status.outcome is ParseOutcome.FAILED:
                    stats["failures"].append(
                        StageFailure(item_id=status.record_id, reason=status.detail or "")
                    )
                continue
            write_json(out / f"{safe_name(paper.record_id)}.json", paper)
            references += refs
            counts["sections"] += len(paper.sections)
            counts["paragraphs"] += sum(len(s.paragraphs) for s in paper.sections)
            counts["tables"] += len(paper.tables)
            counts["tables_without_rows"] += sum(not t.rows for t in paper.tables)
            counts["figures"] += len(paper.figures)
            counts["formulas"] += len(paper.formulas)
            counts["papers_without_abstract"] += not paper.abstract
            counts["papers_without_numbered_sections"] += not any(s.number for s in paper.sections)
        write_jsonl(out / RECORDS, statuses)
        counts["references"] = write_jsonl(out / REFERENCES, references)
        stats["counts"].update(dict(sorted(counts.items())))


def _structure_one(
    config: Config, record: ManifestRecord, status: ParseStatus, counts: Counter[str]
) -> tuple[StructuredPaper | None, list[Reference], ParseStatus]:
    if status.outcome is ParseOutcome.NO_PDF:
        return None, [], status
    reason = status.detail
    if status.outcome is ParseOutcome.GROBID:
        tei = (step_dir(config, "grobid") / f"{safe_name(record.record_id)}.tei.xml").read_bytes()
        # Prefixed so the clean-up counts sort together in stats.json.
        cleanup: Counter[str] = Counter()
        paper = parse_tei(tei, record.record_id, record.title, config.parse.clean, cleanup)
        counts.update({f"cleanup.{k}": v for k, v in cleanup.items()})
        if any(s.paragraphs for s in paper.sections):
            return paper, parse_references(tei, record.record_id), status
        reason = "grobid TEI has no body text"
    if not config.parse.fallback.enabled:
        return (
            None,
            [],
            status.model_copy(update={"outcome": ParseOutcome.FAILED, "detail": reason}),
        )
    try:
        pdf = pdf_path(config.paths.raw / PDFS, record.record_id)
        paper = parse_pdf(pdf, record.record_id, record.title, record.abstract)
    except Exception as exc:  # a PDF PyMuPDF can't open: record it, don't stop the run
        detail = f"{reason}; pymupdf: {type(exc).__name__}: {exc}"
        return (
            None,
            [],
            status.model_copy(update={"outcome": ParseOutcome.FAILED, "detail": detail}),
        )
    if not any(s.paragraphs for s in paper.sections):
        detail = f"{reason}; pymupdf: no text (scanned PDF?)"
        return (
            None,
            [],
            status.model_copy(update={"outcome": ParseOutcome.FAILED, "detail": detail}),
        )
    return paper, [], status.model_copy(update={"outcome": ParseOutcome.FALLBACK, "detail": reason})


# --- chunk ------------------------------------------------------------------------


def chunk(
    config: Config,
    limit: int | None = None,
    counter: TokenCounter | None = None,
) -> None:
    src, out = step_dir(config, "structure"), step_dir(config, "chunk")
    manifest = {r.record_id: r for r in read_manifest(config.paths.manifest)}
    chunker = Chunker(
        config.parse.chunk, counter or HFTokenCounter(config.parse.chunk.tokenizer_model)
    )
    with stage_stats(config, "parse.chunk", out, limit) as stats:
        counts: Counter[str] = Counter()
        chunks: list[Chunk] = []
        index_status: list[IndexStatus] = []
        per_paper: list[int] = []
        for status in read_jsonl(src / RECORDS, ParseStatus)[:limit]:
            n = 0
            if status.outcome in (ParseOutcome.GROBID, ParseOutcome.FALLBACK):
                paper = StructuredPaper.model_validate_json(
                    (src / f"{safe_name(status.record_id)}.json").read_text(encoding="utf-8")
                )
                record = manifest[status.record_id]
                result = chunker.chunk(paper, record.arxiv_id, record.arxiv_version)
                chunks += result.chunks
                n = len(result.chunks)
                per_paper.append(n)
                counts["split_sentences"] += result.split_sentences
                counts["formula_placeholders"] += result.formula_placeholders
            index_status.append(
                IndexStatus(
                    record_id=status.record_id,
                    included_in_index=n > 0,
                    n_chunks=n,
                    outcome=status.outcome,
                )
            )
        counts["chunks"] = write_jsonl(out / CHUNKS, chunks)
        write_jsonl(out / STATUS, index_status)
        counts.update(_chunk_counts(chunks, per_paper, index_status))
        # The chunker guarantees both limits; a failure here means a bug, not bad data.
        limit_tokens = config.parse.chunk.embedding_max_tokens
        max_body = config.parse.chunk.max_tokens
        stats["failures"] += [
            StageFailure(item_id=c.chunk_id, reason=f"{_model_tokens(c)} tokens > {limit_tokens}")
            for c in chunks
            if _model_tokens(c) > limit_tokens
        ]
        stats["failures"] += [
            StageFailure(item_id=c.chunk_id, reason=f"body {c.token_count} > max {max_body}")
            for c in chunks
            if c.token_count > max_body
        ]
        counts["over_max_tokens"] = sum(c.token_count > max_body for c in chunks)
        stats["counts"].update(dict(sorted(counts.items())))


def _model_tokens(chunk: Chunk) -> int:
    return chunk.header_token_count + chunk.token_count + SPECIAL_TOKENS


def _chunk_counts(
    chunks: list[Chunk], per_paper: list[int], status: list[IndexStatus]
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for c in chunks:
        counts[f"element.{c.element_type}"] += 1
        counts[f"parser.{c.parser}"] += 1
        bucket = next((edge for edge in TOKEN_BUCKETS if c.token_count <= edge), None)
        counts[f"tokens_le_{bucket}" if bucket else "tokens_over_512"] += 1
    counts["papers_included"] = sum(s.included_in_index for s in status)
    counts["papers_excluded"] = sum(not s.included_in_index for s in status)
    if per_paper:
        counts["chunks_per_paper_median"] = round(statistics.median(per_paper))
        counts["chunks_per_paper_max"] = max(per_paper)
    if chunks:
        counts["max_model_tokens"] = max(_model_tokens(c) for c in chunks)
    return counts


# --- mark ---------------------------------------------------------------------------


def mark(config: Config, limit: int | None = None) -> None:
    """Write included_in_index into the accepted manifest (D-021).

    Only after a full chunk run: flags from a --limit run would describe a
    handful of papers and dirty the committed manifest. `limit` is accepted
    for a uniform CLI and ignored.
    """
    chunk_dir = step_dir(config, "chunk")
    chunk_stats = StageStats.model_validate_json((chunk_dir / STATS_FILENAME).read_text())
    if chunk_stats.limit is not None:
        logger.warning(
            "Not marking the manifest: the chunk step ran with --limit %d", chunk_stats.limit
        )
        return
    status = {s.record_id: s for s in read_jsonl(chunk_dir / STATUS, IndexStatus)}
    accepted = read_manifest(config.paths.manifest)
    updated = [
        r.model_copy(update={"included_in_index": status[r.record_id].included_in_index})
        if r.record_id in status
        else r
        for r in accepted
    ]
    write_accepted(updated, config.paths.manifest)
    logger.info(
        "Marked %d of %d accepted records (%d included)",
        sum(r.record_id in status for r in accepted),
        len(accepted),
        sum(bool(r.included_in_index) for r in updated),
    )


STEPS: dict[str, Callable[..., None]] = {
    "grobid": grobid,
    "structure": structure,
    "chunk": chunk,
    "mark": mark,
}


def run_all(config: Config, limit: int | None = None) -> None:
    for name, step in STEPS.items():
        logger.info("parse %s", name)
        step(config, limit)
