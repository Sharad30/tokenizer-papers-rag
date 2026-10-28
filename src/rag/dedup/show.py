"""Plain-text view of one work, for `rag inspect work`: who is in it and why."""

from collections.abc import Sequence

from rag.collect.manifest import read_manifest
from rag.common.config import Config
from rag.common.io import read_jsonl
from rag.dedup.files import RESTATEMENTS, WORKS
from rag.dedup.schemas import Edge, Restatement, Work


def find_work(works: Sequence[Work], key: str) -> Work | None:
    """By work_id, or by any member's record id."""
    for w in works:
        if key in (w.work_id, *w.members):
            return w
    return None


def _edge(e: Edge) -> str:
    evidence = [
        f"{name} {value}"
        for name, value in (
            ("title", e.title_ratio),
            ("authors", e.author_overlap),
            ("years", e.year_gap),
            ("text", e.jaccard),
        )
        if value is not None
    ]
    detail = f" ({', '.join(evidence)})" if evidence else ""
    return f"{e.step}/{e.reason}: {e.a} ~ {e.b}{detail}"


def render_work(config: Config, key: str) -> str:
    path = config.paths.works / WORKS
    if not path.is_file():
        return f"No works at {config.paths.works}: run `rag dedup run`."
    work = find_work(read_jsonl(path, Work), key)
    if work is None:
        return f"No work with id or member {key!r}."
    records = {r.record_id: r for r in read_manifest(config.paths.manifest)}
    lines = [f"{work.work_id}", f"canonical: {work.canonical} ({work.canonical_rule})", ""]
    lines.append(f"members ({len(work.members)}):")
    for rid in work.members:
        r = records.get(rid)
        mark = "*" if rid == work.canonical else " "
        about = f"{r.year or '?'}  {r.venue or '-'}  {r.title}" if r else "(not in manifest)"
        lines.append(f" {mark} {rid:<45} {about[:90]}")
    lines += ["", "merged by:"]
    lines += [f"  {_edge(e)}" for e in work.edges] or ["  (single record)"]
    if work.rejected:
        lines += ["", "refused by the chain guard:"]
        lines += [f"  {_edge(e)}" for e in work.rejected]
    restatements_path = config.paths.works / RESTATEMENTS
    if restatements_path.is_file():
        restating = [
            r for r in read_jsonl(restatements_path, Restatement) if r.work_id == work.work_id
        ]
        if restating:
            lines += ["", f"restated by {len(restating)} survey chunk(s):"]
            lines += [
                f"  {r.chunk_id} (cosine {r.cosine}, cites {r.ref_id} by {r.matched_by})"
                for r in restating
            ]
    return "\n".join(lines)
