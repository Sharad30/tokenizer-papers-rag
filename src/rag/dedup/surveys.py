"""Surveys and the work they restate: flagged, never merged (D-041).

A survey summarising BPE-dropout is not the BPE-dropout paper, so it keeps
its own work. But when both turn up for a query, the primary paper should
rank first. A survey chunk *restates* work W when

- it cites W: one of its in-text citations points to a bibliography entry
  that resolves to an indexed record of W (arXiv id, then DOI, then title), and
- it says much the same thing: its dense cosine to W's closest chunk is at
  least `restate_min`.

The query-time step (`rag.retrieve.collapse`) uses these to demote the
survey chunk below W's chunks.
"""

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np
from rapidfuzz import fuzz, process

from rag.collect.schemas import ManifestRecord
from rag.dedup.config import SurveyConfig
from rag.dedup.resolve import normalise
from rag.dedup.schemas import Restatement, Survey
from rag.parse.schemas import Chunk, Reference

VERSION = re.compile(r"v\d+$")  # "2004.03720v2" -> "2004.03720"


def find_surveys(records: Sequence[ManifestRecord], config: SurveyConfig) -> list[Survey]:
    """Title cues only: S2's "Review" type also tags ordinary research papers."""
    cue = re.compile(r"\b(?:" + "|".join(map(re.escape, config.title_cues)) + r")\b", re.I)
    out = []
    for r in records:
        found = sorted({m.group(0).lower() for m in cue.finditer(r.title)})
        if found:
            out.append(Survey(record_id=r.record_id, cues=found))
    return out


class ReferenceResolver:
    """Bibliography entry -> indexed record_id, by arXiv id, DOI, then title."""

    def __init__(self, records: Sequence[ManifestRecord], title_min: float) -> None:
        self.by_arxiv = {r.arxiv_id: r.record_id for r in records if r.arxiv_id}
        self.by_doi = {r.doi.lower(): r.record_id for r in records if r.doi}
        self.titles = [normalise(r.title) for r in records]
        self.ids = [r.record_id for r in records]
        self.title_min = title_min

    def resolve(self, ref: Reference) -> tuple[str, str] | None:
        """(record_id, matched_by) or None."""
        arxiv_id = VERSION.sub("", ref.arxiv_id) if ref.arxiv_id else None
        if arxiv_id in self.by_arxiv:
            return self.by_arxiv[arxiv_id], "arxiv"
        if ref.doi and ref.doi.lower() in self.by_doi:
            return self.by_doi[ref.doi.lower()], "doi"
        if ref.title:
            hit = process.extractOne(
                normalise(ref.title), self.titles, scorer=fuzz.ratio, score_cutoff=self.title_min
            )
            if hit is not None:
                return self.ids[hit[2]], "title"
        return None


def find_restatements(
    surveys: Sequence[Survey],
    chunks: Sequence[Chunk],
    references: Sequence[Reference],
    resolver: ReferenceResolver,
    record_work: Mapping[str, str],
    embeddings: np.ndarray,
    chunk_ids: Sequence[str],
    restate_min: float,
) -> list[Restatement]:
    survey_ids = {s.record_id for s in surveys}
    refs = {(r.record_id, r.ref_id): r for r in references if r.record_id in survey_ids}
    row = {cid: i for i, cid in enumerate(chunk_ids)}
    work_rows: dict[str, list[int]] = defaultdict(list)
    for chunk in chunks:
        if chunk.chunk_id in row and chunk.record_id in record_work:
            work_rows[record_work[chunk.record_id]].append(row[chunk.chunk_id])

    out = []
    for chunk in chunks:
        if chunk.record_id not in survey_ids or chunk.chunk_id not in row:
            continue
        own_work = record_work.get(chunk.record_id)
        best: dict[str, Restatement] = {}
        for ref_id in chunk.cited_refs:
            ref = refs.get((chunk.record_id, ref_id))
            target = resolver.resolve(ref) if ref is not None else None
            if target is None:
                continue
            work = record_work.get(target[0])
            if work is None or work == own_work or not work_rows[work]:
                continue
            cosine = float((embeddings[work_rows[work]] @ embeddings[row[chunk.chunk_id]]).max())
            if cosine >= restate_min and (work not in best or cosine > best[work].cosine):
                best[work] = Restatement(
                    chunk_id=chunk.chunk_id,
                    work_id=work,
                    cosine=round(cosine, 4),
                    ref_id=ref_id,
                    matched_by=target[1],
                )
        out.extend(best.values())
    return out
