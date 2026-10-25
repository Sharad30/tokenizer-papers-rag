"""The corpus manifest: one row per manifestation, every decision recorded (D-010).

Accepted records are written as JSON (exact types, for code) and CSV (for
reading in a spreadsheet). Rejected records go to a gzipped CSV without
abstracts: they are ~90% of rows and most of the bytes, and committing them
in full would put ~70 MB in git. Every reject keeps its IDs, title and reason
code, so it can still be audited and its abstract looked up by ID.
List fields are joined with "; " in the CSVs.
"""

import csv
import gzip
import io
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from rag.collect.schemas import Decision, ManifestRecord
from rag.common.io import write_bytes_atomic, write_text_atomic

MANIFEST_JSON = "manifest.json"
MANIFEST_CSV = "manifest.csv"
REJECTED_CSV_GZ = "rejected.csv.gz"
REJECTED_DROPPED_FIELDS = ("abstract",)
LIST_SEPARATOR = "; "


def write_manifest(records: Sequence[ManifestRecord], out_dir: Path) -> None:
    accepted = [r for r in records if r.relevance is Decision.ACCEPT]
    write_accepted(accepted, out_dir)
    # Sorted so reruns produce identical files and git diffs show real changes.
    rejected = [
        r.model_dump(mode="json")
        for r in sorted(records, key=lambda r: r.record_id)
        if r.relevance is not Decision.ACCEPT
    ]
    rejected_csv = _to_csv(rejected, _fields(drop=REJECTED_DROPPED_FIELDS))
    # mtime=0 keeps the gzip header fixed, so an unchanged rerun is byte-identical.
    write_bytes_atomic(out_dir / REJECTED_CSV_GZ, gzip.compress(rejected_csv.encode(), mtime=0))


def write_accepted(accepted: Sequence[ManifestRecord], out_dir: Path) -> None:
    """Rewrite only the accepted part (used by `rag parse mark` to fill included_in_index)."""
    rows = [r.model_dump(mode="json") for r in sorted(accepted, key=lambda r: r.record_id)]
    write_text_atomic(
        out_dir / MANIFEST_JSON, json.dumps(rows, indent=1, ensure_ascii=False) + "\n"
    )
    write_text_atomic(out_dir / MANIFEST_CSV, _to_csv(rows, _fields()))


def read_manifest(out_dir: Path) -> list[ManifestRecord]:
    """The accepted records, with their full types."""
    rows = json.loads((out_dir / MANIFEST_JSON).read_text(encoding="utf-8"))
    return [ManifestRecord.model_validate(row) for row in rows]


def read_rejected(out_dir: Path) -> list[dict[str, str]]:
    """The rejected records as CSV rows (strings; no abstracts)."""
    text = gzip.decompress((out_dir / REJECTED_CSV_GZ).read_bytes()).decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def _fields(drop: Sequence[str] = ()) -> list[str]:
    return [f for f in ManifestRecord.model_fields if f not in drop]


def _to_csv(rows: list[dict], fields: list[str]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {k: LIST_SEPARATOR.join(v) if isinstance(v, list) else v for k, v in row.items()}
        )
    return buffer.getvalue()


def discovery_routes(record: ManifestRecord) -> set[str]:
    """Coarse route names from found_via: "arxiv_query:bpe" -> "arxiv_query"."""
    return {via.split(":", 1)[0] for via in record.found_via}


def manifest_counts(records: Sequence[ManifestRecord]) -> dict[str, int]:
    """Flat counts for stats.json: sources, decisions by reason, routes, downloads, licences."""
    counts: Counter[str] = Counter()
    accepted = [r for r in records if r.relevance is Decision.ACCEPT]
    counts["records"] = len(records)
    counts["accepted"] = len(accepted)
    for r in records:
        counts[f"source.{r.source}"] += 1
        counts[f"decision.{r.relevance}.{r.reason}"] += 1
        counts[f"download.{r.download_status}"] += 1
    for r in accepted:
        counts[f"accepted_source.{r.source}"] += 1
        counts[f"accepted_licence_source.{r.licence_source}"] += 1
        # Overlap: which combination of discovery routes found each accepted record.
        counts["accepted_routes." + "+".join(sorted(discovery_routes(r)))] += 1
    return dict(sorted(counts.items()))
