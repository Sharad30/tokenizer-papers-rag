"""Collect: which papers are in the corpus, how relevance was judged, and examples.

Rendering only. Numbers, pools and explanations come from
`rag.showcase.collect`, computed from the manifest on load.
"""

import os
from datetime import datetime
from pathlib import Path

import altair as alt
import streamlit as st

from rag.collect.relevance import KeywordRules, RuleTrace
from rag.collect.schemas import Decision, ManifestRecord
from rag.common.config import DEFAULT_CONFIG_PATH, load_config
from rag.showcase import collect
from rag.showcase.markup import escape, highlight
from rag.showcase.sources import fingerprint

CONFIG_PATH = Path(os.environ.get("RAG_CONFIG", DEFAULT_CONFIG_PATH))
SOURCE_NAMES = {"arxiv": "arXiv", "acl": "ACL Anthology", "s2": "Semantic Scholar"}
DECISION_COLOURS = {"accept": "#2a78d6", "reject": "#eb6834"}


# cache_resource, not cache_data: 19k records would be copied on every rerun.
# The stamps are the cache key, so an edited config or data file reloads.
@st.cache_resource(show_spinner="Reading the manifest…")
def load(config_stamp, data_stamp):
    config = load_config(CONFIG_PATH)
    data = collect.load(config)
    view = collect.overview(data.accepted, data.rejected)
    return config, data, view, collect.linked_copies([*data.accepted, *data.rejected])


config = load_config(CONFIG_PATH)
files = collect.source_files(config)
config, data, view, copies = load(fingerprint([CONFIG_PATH]), fingerprint(list(files.values())))
relevance = config.collect.relevance
rules = KeywordRules(relevance)
everything = [*data.accepted, *data.rejected]

st.title("1 · Collect the corpus")
st.write(
    "Find candidate papers in three open sources, decide which are about LLM "
    "tokenization with an explicit rule, and keep every decision with its reason."
)


# --- Where the data comes from ------------------------------------------------


def _as_of() -> str:
    parts = []
    for path in files.values():
        if path.exists():
            modified = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            parts.append(f"`{path.name}` ({modified})")
    origin = {
        "live": "live",
        "snapshot": f"snapshot from commit {data.snapshot.git_commit[:8]}" if data.snapshot else "",
        "none": "not available",
    }[data.abstracts_from]
    return f"Data as of: {', '.join(parts)}. Rejected abstracts: {origin}."


st.caption(_as_of())
if data.stale:
    st.warning(
        f"The snapshot was built from older {', '.join(data.stale)}: run "
        "`uv run rag showcase snapshot` to refresh it."
    )

# --- Headline numbers ---------------------------------------------------------

cols = st.columns(5)
cols[0].metric("Candidates found", f"{view.candidates:,}")
cols[1].metric("Rejected", f"{view.rejected:,}")
cols[2].metric(
    "Accepted records",
    f"{view.accepted:,}",
    help="Includes linked copies (arXiv + ACL) and kept earlier versions.",
)
cols[3].metric(
    "Distinct papers", f"{view.distinct_accepted:,}", help="Accepted records grouped by shared ids."
)
cols[4].metric(
    "Indexed",
    f"{view.indexed:,}",
    help=f"Parsed and chunked. {view.downloaded:,} accepted records have a PDF.",
)

# --- Sources --------------------------------------------------------------------

st.header("Open sources")
for column, info in zip(st.columns(3), collect.sources_summary(config, data, view), strict=True):
    with column.container(border=True):
        st.subheader(info.name)
        st.write(info.how)
        left, right = st.columns(2)
        left.metric("Candidates", f"{info.candidates:,}")
        right.metric("Accepted", f"{info.accepted:,}")
        if info.hosts:
            limits = ", ".join(f"`{host}` {seconds:g} s" for host, seconds in info.hosts.items())
            st.caption(f"Min. gap between requests: {limits}. Every response cached.")
        if info.details:
            with st.expander(f"{len(info.details)} queries / anchors"):
                st.code("\n".join(info.details), language=None)

# --- Relevance --------------------------------------------------------------------

st.header("How relevance is judged")
counts = view.by_path
ambiguous = counts["embedding_above"] + counts["embedding_below"]
n = {key: f"{value:,}" for key, value in counts.items()}
min_sentences = relevance.min_abstract_sentences_with_term
st.markdown(
    "\n".join(
        [
            "**1. Keyword rules** on title + abstract (`collect/relevance.py`, D-005):",
            f"- no topic term anywhere → reject ({n['no_topic_term']})",
            '- title names another sense of "token" (vision, speech, protein, …) → '
            f"reject ({n['off_topic']})",
            f"- topic term in the title → accept ({n['title_term']})",
            f"- topic term only in the abstract, in fewer than {min_sentences} sentences → "
            f"reject as a passing mention ({n['passing_mention']})",
            f"- otherwise ambiguous → step 2 ({ambiguous:,})",
            "",
            "**2. Embedding judge** for the ambiguous ones (D-008): cosine between the "
            "paper's title + abstract and the written topic description, with "
            f"`{relevance.embedding_model}`. At or above **{relevance.embedding_threshold}** → "
            f"accept ({n['embedding_above']}), else reject ({n['embedding_below']}).",
            "",
            "**3. Copies inherit** their paper's decision, so two copies never disagree: "
            f"linked copies ({n['linked_copy']}) and revised earlier versions "
            f"({n['earlier_version']}).",
        ]
    )
)
with st.expander("The topic description the judge compares against"):
    st.write(relevance.topic_description)

left, right = st.columns(2)
with left:
    st.subheader("Records per decision path")
    path_rows = [
        {"path": p.label, "records": counts[p.key], "decision": p.decision.value}
        for p in collect.PATHS
    ]
    st.altair_chart(
        alt.Chart(alt.Data(values=path_rows))
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("records:Q", title="records", scale=alt.Scale(type="symlog")),
            y=alt.Y("path:N", sort=None, title=None),
            color=alt.Color(
                "decision:N",
                scale=alt.Scale(
                    domain=list(DECISION_COLOURS), range=list(DECISION_COLOURS.values())
                ),
            ),
            tooltip=["path:N", "records:Q"],
        ),
        width="stretch",
    )
    st.caption("Log scale: the paths differ by three orders of magnitude.")
with right:
    st.subheader("Embedding judge scores")
    judged = [
        {"score": j.score, "decision": "accept" if j.accepted else "reject"} for j in view.judged
    ]
    bars = (
        alt.Chart(alt.Data(values=judged))
        .mark_bar()
        .encode(
            x=alt.X("score:Q", bin=alt.Bin(step=0.01), title="cosine to the topic description"),
            y=alt.Y("count():Q", title="papers"),
            color=alt.Color(
                "decision:N",
                scale=alt.Scale(
                    domain=list(DECISION_COLOURS), range=list(DECISION_COLOURS.values())
                ),
            ),
            tooltip=["decision:N", "count():Q"],
        )
    )
    line = (
        alt.Chart(alt.Data(values=[{"threshold": relevance.embedding_threshold}]))
        .mark_rule(strokeDash=[4, 3])
        .encode(x="threshold:Q")
    )
    st.altair_chart(bars + line, width="stretch")
    st.caption(
        f"{len(view.judged):,} ambiguous papers; dashed line = threshold "
        f"{relevance.embedding_threshold} from config.yaml."
    )

# --- Examples -----------------------------------------------------------------------

st.header("Examples, drawn from the data")


def render_trace(trace: RuleTrace, title: str, abstract: str) -> None:
    title_marks = trace.title_terms + trace.exclude_terms
    st.markdown(f"**Title:** {highlight(title, title_marks)}")
    if trace.exclude_terms:
        st.markdown(f"Excluded sense in the title: {', '.join(trace.exclude_terms)}")
    if abstract:
        st.markdown(highlight(abstract, trace.topic_terms))
        with_terms = sum(1 for _, terms in trace.sentences if terms)
        st.caption(
            f"Topic terms in {with_terms} of {len(trace.sentences)} abstract sentences "
            f"(the passing-mention rule needs {trace.min_sentences}). "
            f"Matched: {', '.join(trace.topic_terms) or 'none'}."
        )
    else:
        st.caption("No abstract available: the rules saw the title only.")


def render_record(record: ManifestRecord) -> None:
    explanation = collect.explain(record, rules, relevance)
    with st.container(border=True):
        authors = ", ".join(record.authors[:3]) + (" et al." if len(record.authors) > 3 else "")
        meta = " · ".join(
            str(x) for x in (authors, record.year, record.venue, SOURCE_NAMES[record.source]) if x
        )
        st.markdown(f"#### {escape(record.title)}")
        st.caption(
            f"{escape(meta)} · `{record.record_id}` · found via {', '.join(record.found_via)}"
        )
        render_trace(explanation.trace, record.title, record.abstract or "")
        recorded = f"**Recorded:** {record.relevance} · `{record.reason}` · by {record.decided_by}"
        if explanation.embedding_score is not None:
            recorded += (
                f" · score {explanation.embedding_score:.3f} vs threshold {explanation.threshold}"
            )
        st.markdown(recorded)
        if explanation.inherited_from:
            st.info(f"Inherited from `{explanation.inherited_from}`.")
        elif explanation.agrees is True:
            st.success(f"Today's rules and config agree: {explanation.today}.")
        elif explanation.agrees is False:
            st.warning(
                f"Today's rules would {explanation.today.value} this paper. The config "
                "changed since collection, or the decision was made on different text "
                "(e.g. another version's title or abstract)."
            )
        else:
            st.info("Ambiguous by the keyword rules; no recorded score to re-check.")


labels = {p.key: f"{p.label} ({counts[p.key]:,})" for p in collect.PATHS}
path_key = st.selectbox("Decision path", list(labels), format_func=labels.get)
st.caption(collect.describe(collect.PATH_BY_KEY[path_key], relevance))
f1, f2, f3 = st.columns([2, 3, 1])
sources = f1.multiselect("Source", list(SOURCE_NAMES), format_func=SOURCE_NAMES.get)
known_years = sorted({r.year for r in everything if r.year is not None})
years = f2.slider("Year", known_years[0], known_years[-1], (known_years[0], known_years[-1]))
seed_key = "collect_seed"
st.session_state.setdefault(seed_key, config.showcase.seed)
if f3.button("Show others", width="stretch"):
    st.session_state[seed_key] += 1
full_range = years == (known_years[0], known_years[-1])
# Untouched slider = no year filter, so records without a year still show.
pool = collect.example_pool(
    everything, path_key, sources=sources, years=None if full_range else years
)
shown = collect.pick(pool, config.showcase.examples_shown, st.session_state[seed_key])
on_path = counts[path_key]
st.caption(
    f"Showing {len(shown)} of {len(pool):,} matching records "
    f"({on_path:,} on this path; those with an abstract are preferred)."
)
if data.abstracts_from != "live" and collect.PATH_BY_KEY[path_key].decision is Decision.REJECT:
    st.caption("Rejected abstracts come from the snapshot here, so the pool is a sample.")
for record in shown:
    render_record(record)

# --- Metadata ----------------------------------------------------------------------

st.header("What is kept for every paper")
query = st.text_input("Search a title or record id", placeholder="e.g. How Good is Your Tokenizer")
if query:
    matches = collect.search(data.accepted, query, limit=20)
else:
    # A paper with several manifestations shows best how records link up.
    linked = [r for r in data.accepted if len(copies[r.record_id]) >= 3]
    matches = collect.pick(linked, 1, st.session_state[seed_key])
if not matches:
    st.info("No accepted record matches.")
else:
    chosen = st.selectbox(
        "Record", matches, format_func=lambda r: f"{r.title} ({r.record_id})", key="meta_record"
    )
    left, right = st.columns([3, 2])
    with left:
        st.json(chosen.model_dump(mode="json", exclude={"abstract"}), expanded=True)
    with right:
        group = copies[chosen.record_id]
        st.markdown(f"**The same paper in the corpus: {len(group)} record(s)**")
        st.dataframe(
            [
                {
                    "record": r.record_id,
                    "source": r.source.value,
                    "year": r.year,
                    "decision": f"{r.relevance} ({r.reason})",
                    "indexed": r.included_in_index,
                }
                for r in group
            ],
            hide_index=True,
        )
        st.caption(
            "Linked by a shared arXiv id, DOI, Semantic Scholar id or ACL id. "
            "They stay separate records until Phase 5 merges them."
        )

# --- Coverage -----------------------------------------------------------------------

st.header("Coverage")
left, right = st.columns([3, 2])
with left:
    st.subheader("Accepted records per year")
    st.bar_chart(
        [
            {"year": str(y.year), "source": SOURCE_NAMES[y.source], "records": y.papers}
            for y in view.per_year
        ],
        x="year",
        y="records",
        color="source",
    )
with right:
    st.subheader("Licences and downloads")
    st.dataframe(
        [{"licence source": k, "accepted records": v} for k, v in view.licences.items()],
        hide_index=True,
    )
    st.dataframe(
        [{"download status": k, "accepted records": v} for k, v in view.downloads.items()],
        hide_index=True,
    )
    missing = ", ".join(f"{field} {n:,}" for field, n in view.missing.items())
    st.caption(f"Accepted records missing a field: {missing}.")
with st.expander("Discovery routes: how each accepted record was found"):
    st.dataframe(
        [{"routes": k, "accepted records": v} for k, v in view.routes.items()],
        hide_index=True,
    )

# --- Try it ----------------------------------------------------------------------------

with st.expander("Try the keyword rules on any title and abstract"):
    title = st.text_input("Title", key="try_title")
    abstract = st.text_area("Abstract", key="try_abstract")
    if title or abstract:
        trace = rules.trace(title, abstract)
        render_trace(trace, title, abstract)
        decision = trace.decision
        verdict = {
            Decision.ACCEPT: "accept",
            Decision.REJECT: "reject",
            Decision.AMBIGUOUS: "ambiguous: the embedding judge would decide",
        }[decision.decision]
        st.markdown(f"**Keyword rules:** {verdict} · `{decision.reason}`")
