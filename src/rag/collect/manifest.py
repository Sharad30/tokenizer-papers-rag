"""The corpus manifest: one row per manifestation, every decision recorded (D-010).

Written as JSON (exact types, for code) and CSV (for reading and auditing in
a spreadsheet). List fields are joined with "; " in the CSV.
"""

import csv
import io
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from rag.collect.schemas import Decision, ManifestRecord
from rag.common.io import write_text_atomic

MANIFEST_JSON = "manifest.json"
MANIFEST_CSV = "manifest.csv"
LIST_SEPARATOR = "; "


def write_manifest(records: Sequence[ManifestRecord], out_dir: Path) -> None:
    # Sorted so reruns produce identical files and git diffs show real changes.
    ordered = sorted(records, key=lambda r: r.record_id)
    payload = [r.model_dump(mode="json") for r in ordered]
    write_text_atomic(
        out_dir / MANIFEST_JSON, json.dumps(payload, indent=1, ensure_ascii=False) + "\n"
    )
    write_text_atomic(out_dir / MANIFEST_CSV, _to_csv(payload))


def read_manifest(out_dir: Path) -> list[ManifestRecord]:
    rows = json.loads((out_dir / MANIFEST_JSON).read_text(encoding="utf-8"))
    return [ManifestRecord.model_validate(row) for row in rows]


def _to_csv(rows: list[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=list(ManifestRecord.model_fields), lineterminator="\n"
    )
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
