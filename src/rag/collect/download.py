"""PDF downloads with resume: a valid file from an earlier run is never fetched again.

PDFs are gitignored and never redistributed (arXiv terms); the download
code and the manifest's sha256 are what make the corpus reproducible.
"""

import hashlib
import logging
from pathlib import Path

from rag.collect.http import HttpError, PoliteClient
from rag.collect.schemas import DownloadStatus, ManifestRecord
from rag.common.io import write_bytes_atomic

logger = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF"


def pdf_path(pdf_dir: Path, record_id: str) -> Path:
    # record ids contain ":" and, for old arXiv ids, "/" (cs/0501001v1).
    safe = record_id.replace(":", "_").replace("/", "_")
    return pdf_dir / f"{safe}.pdf"


def is_pdf(data: bytes) -> bool:
    return data.startswith(PDF_MAGIC)


def download(client: PoliteClient, record: ManifestRecord, pdf_dir: Path) -> ManifestRecord:
    """Fetch one record's PDF and return the record with status and sha256 filled in."""
    path = pdf_path(pdf_dir, record.record_id)
    if path.is_file():
        existing = path.read_bytes()
        if is_pdf(existing):
            return _with(record, DownloadStatus.ALREADY_PRESENT, existing)
    if not record.pdf_url:
        return _with(record, DownloadStatus.NO_URL, None)
    try:
        data = client.fetch_bytes(record.pdf_url)
    except HttpError as exc:
        logger.warning("Download failed for %s: %s", record.record_id, exc)
        return _with(record, DownloadStatus.FAILED, None)
    if not is_pdf(data):
        return _with(record, DownloadStatus.NOT_PDF, None)
    write_bytes_atomic(path, data)
    return _with(record, DownloadStatus.DOWNLOADED, data)


def _with(record: ManifestRecord, status: DownloadStatus, data: bytes | None) -> ManifestRecord:
    sha = hashlib.sha256(data).hexdigest() if data is not None else None
    return record.model_copy(update={"download_status": status, "pdf_sha256": sha})
