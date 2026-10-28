"""Versions and duplicates: records resolved into works, and what that does to the top 5.

Rendering only. Numbers and examples come from `rag.showcase.dedup`, computed
from data/works/ on load (or from the snapshot on a fresh clone); the
ablation table from the latest committed results file.
"""

import os
from pathlib import Path

import altair as alt
import streamlit as st

from rag.common.config import DEFAULT_CONFIG_PATH, load_config
from rag.dedup.schemas import Edge
from rag.showcase import dedup
from rag.showcase.parse import sample
from rag.showcase.sources import fingerprint

CONFIG_PATH = Path(os.environ.get("RAG_CONFIG", DEFAULT_CONFIG_PATH))
FIGURES = Path(__file__).resolve().parents[2] / "docs" / "figures"
STEP_LABELS = {
    "ids": "shared identifier",
    "title": "near-exact title + shared authors",
    "minhash": "shared full text + shared authors",
}


@st.cache_resource(show_spinner="Reading the works…")
def load(config_stamp, data_stamp):
    config = load_config(CONFIG_PATH)
    return config, dedup.load(config)


config = load_config(CONFIG_PATH)
files = dedup.source_files(config)
results = sorted(config.paths.results.glob("*-dedup-ablation.json"))
config, data = load(fingerprint([CONFIG_PATH]), fingerprint([*files.values(), *results]))
cfg = config.dedup

st.title("3 · Versions and duplicates")
st.write(
    "The corpus holds several records of one paper on purpose: arXiv versions, the arXiv "
    "preprint and the ACL Anthology copy, a paper at two venues. Resolving them into "
    "**works** stops one paper filling the top results, without merging papers that are "
    "merely about the same thing."
)

if data.origin == "none":
    st.error("No works and no snapshot: run `rag dedup run`, then reload.")
    st.stop()
if data.origin == "live":
    st.caption(f"Data: live from `{config.paths.works}`.")
else:
    st.caption(
        f"Data: snapshot from commit {data.snapshot.git_commit[:8]} "
        f"({data.snapshot.created_at:%Y-%m-%d}). Numbers cover the whole corpus; "
        "examples are a sample."
    )
if data.stale:
    st.warning(
        f"The snapshot predates the current {', '.join(data.stale)}: run "
        "`uv run rag showcase snapshot --stage dedup`."
    )
view = data.overview
seed_key = "dedup_seed"
st.session_state.setdefault(seed_key, config.showcase.seed)

# --- Headline numbers ---------------------------------------------------------------

cols = st.columns(5)
cols[0].metric("Indexed records", f"{view.records:,}")
cols[1].metric(
    "Groups by identifiers",
    f"{view.id_groups:,}",
    help="Shared arXiv id, DOI, S2 id, ACL id, or a Phase 1 link. This is also the "
    "dup@5 judge before labelled pairs (D-031, D-042).",
)
cols[2].metric("Works", f"{view.works:,}", help="After title and full-text evidence (D-040).")
cols[3].metric("Works with several records", f"{view.multi_member:,}")
cols[4].metric("Surveys flagged", len(view.surveys), help="\n".join(view.surveys))

st.markdown(
    "Each step adds evidence that two records are one paper; every fuzzy merge needs "
    "**shared authors** plus a title or text match:\n"
    f"1. **Identifiers**: trusted outright.\n"
    f"2. **Title**: normalised titles at ratio ≥ {cfg.title.min_ratio:g} "
    f"(`token_sort_ratio`), years ≤ {cfg.title.year_gap_max} apart.\n"
    f"3. **Full text**: MinHash estimate of 5-word-shingle Jaccard ≥ "
    f"{cfg.minhash.min_jaccard:g}; titles at {cfg.title.review_ratio:g}–{cfg.title.min_ratio:g} "
    f"also need Jaccard ≥ {cfg.title.review_min_jaccard:g}.\n"
    f"4. **Chain guard**: union-find is transitive, so a merge is refused when any two "
    f"ID-linked parts it would join share no evidence (Jaccard < {cfg.cluster_floor:g}).\n\n"
    "The **canonical** record (ACL copy, else latest arXiv version) represents the work; "
    "the others stay in the index but are hidden at query time (`src/rag/dedup/resolve.py`)."
)

left, right = st.columns([2, 1])
with left:
    sizes = [{"records per work": str(k), "works": v} for k, v in view.sizes.items()]
    chart = (
        alt.Chart(alt.Data(values=sizes))
        .mark_bar(color="#2a78d6", cornerRadiusEnd=4)
        .encode(
            x=alt.X("records per work:N", title="records per work", sort=None),
            y=alt.Y("works:Q", title="works"),
            tooltip=["records per work:N", "works:Q"],
        )
    )
    st.altair_chart(chart, width="stretch")
with right:
    st.dataframe(
        [
            {"merged by": STEP_LABELS.get(step, step), "edges": n}
            for step, n in sorted(view.edges.items())
        ]
        + [{"merged by": "refused by the chain guard", "edges": view.refused}],
        hide_index=True,
    )
    st.caption(
        "Identifier edges count links (one pair can share several ids), not records. "
        f"{view.restatements} survey chunks restate a work they cite."
    )

# --- Examples -------------------------------------------------------------------------


def edge_line(e: Edge) -> str:
    evidence = [
        f"{name} {value}"
        for name, value in (
            ("title", e.title_ratio),
            ("authors", e.author_overlap),
            ("years apart", e.year_gap),
            ("text Jaccard", e.jaccard),
        )
        if value is not None
    ]
    return f"`{e.a}` ~ `{e.b}`: {STEP_LABELS.get(e.step, e.step)} ({e.reason})" + (
        f"; {', '.join(evidence)}" if evidence else ""
    )


def render_work(work: dedup.WorkExample) -> None:
    with st.container(border=True):
        st.caption(f"`{work.work_id}` · canonical: {work.canonical_rule}")
        st.dataframe(
            [
                {
                    "": "★" if m.record_id == work.canonical else "",
                    "record": m.record_id,
                    "year": m.year,
                    "venue": m.venue,
                    "title": m.title,
                }
                for m in work.members
            ],
            hide_index=True,
        )
        for e in work.edges:
            st.markdown(f"- {edge_line(e)}")
        for e in work.rejected:
            st.markdown(f"- refused: {edge_line(e)}")


st.header("Merges, drawn from the data")
if st.button("Show others", key="dedup_resample"):
    st.session_state[seed_key] += 1
seed = st.session_state[seed_key]
tabs = st.tabs(["Title or text evidence", "Identifiers only", "Survey restatements"])
with tabs[0]:
    st.caption(
        f"{len(data.fuzzy_works):,} works were built or touched by fuzzy evidence: the "
        "merges most likely to be wrong. All are listed in `data/works/review.md` for review."
    )
    for work in sample(data.fuzzy_works, 2, seed, dedup.work_key):
        render_work(work)
with tabs[1]:
    st.caption("Versions and venue copies the identifiers already link.")
    for work in sample(data.id_works, 2, seed, dedup.work_key):
        render_work(work)
with tabs[2]:
    st.caption(
        "A survey is never merged with what it summarises. A survey chunk *restates* work W "
        f"when it cites W and its cosine to W's closest chunk is ≥ {cfg.surveys.restate_min:g}; "
        "in the results it then ranks just below W's chunk (D-041)."
    )
    for r in sample(data.restatements, 2, seed, dedup.restatement_key):
        with st.container(border=True):
            st.caption(
                f"`{r.chunk_id}` · {r.survey_title} → restates {r.source_title} "
                f"(cosine {r.cosine:.3f}, citation matched by {r.matched_by})"
            )
            st.code(r.text, language=None, wrap_lines=True)

# --- Effect on the top 5 ---------------------------------------------------------------

st.header("What it does to the top 5")
ablation = data.ablation
if ablation is None:
    st.info("No ablation results yet: run `uv run rag eval ablate --name dedup`.")
else:
    st.dataframe(
        [
            {
                "condition": row.label,
                "dup@5": f"{row.dup:.3f} [{row.dup_low:.3f}, {row.dup_high:.3f}]",
                "copy_dup@5": f"{row.copy_dup:.3f}",
                "copy_dup@5 (resolver's view)": f"{row.copy_dup_resolver:.3f}",
            }
            for row in ablation.rows
        ],
        hide_index=True,
    )
    st.caption(
        f"`results/{ablation.file}` (commit {ablation.git_commit[:8]}): {ablation.n_queries} "
        "probe queries written for the probe, not eval gold. **dup@5** is the share of top-5 "
        "slots whose paper already appeared higher, a second chunk of one paper included, so "
        "a per-work cap of 1 makes it 0 by construction; its cost can only show in recall, "
        "once the eval questions exist. **copy_dup@5** counts only another record of the "
        f"paper. Both are judged by identifiers + {ablation.labelled_pairs} labelled pairs "
        "(D-042): the resolver's own view is shown beside them, but it can't grade itself."
    )

# --- Probe --------------------------------------------------------------------------------

scurve = FIGURES / "phase5-scurve.png"
if scurve.exists():
    st.header("Where the text threshold comes from")
    st.image(str(scurve))
    st.caption(
        "Top: the chance two papers become MinHash candidates, 1 − (1 − s^r)^b, for the "
        f"configured {cfg.minhash.bands} bands × {cfg.minhash.rows} rows. Bottom: exact "
        "Jaccard of real pairs: records the identifiers link (blue) and same-author pairs "
        "with no shared id (orange). From `probes/phase-5-minhash.py`; table in "
        "`docs/figures/phase5-minhash.md`."
    )
