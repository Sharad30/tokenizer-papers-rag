"""Parse and chunk: PDFs to section-aware chunks, and what happens to each element.

Rendering only. Numbers, pools and checks come from `rag.showcase.parse`,
computed from data/parsed/ on load (or from the snapshot on a fresh clone).
Chunk text is shown exactly as stored: that is what gets embedded.
"""

import os
from pathlib import Path

import altair as alt
import streamlit as st

from rag.common.config import DEFAULT_CONFIG_PATH, load_config
from rag.parse.schemas import Chunk
from rag.showcase import parse
from rag.showcase.sources import fingerprint

CONFIG_PATH = Path(os.environ.get("RAG_CONFIG", DEFAULT_CONFIG_PATH))
FIGURES = Path(__file__).resolve().parents[2] / "docs" / "figures"
ELEMENT_COLOURS = {
    "paragraph": "#2a78d6",
    "abstract": "#1baf7a",
    "table": "#eb6834",
    "figure_caption": "#8a63d2",
}


# cache_resource: 93k chunks shouldn't be copied on every rerun. The stamps
# are the cache key, so a re-chunk or an edited config reloads.
@st.cache_resource(show_spinner="Reading the chunks…")
def load(config_stamp, data_stamp):
    config = load_config(CONFIG_PATH)
    return config, parse.load(config)


config = load_config(CONFIG_PATH)
files = parse.source_files(config)
config, data = load(fingerprint([CONFIG_PATH]), fingerprint(list(files.values())))
chunking = config.parse.chunk

st.title("2 · Parse and chunk")
st.write(
    "Turn each PDF into a structured paper (sections, paragraphs, tables, figures, "
    "formulas, references), then into chunks that never cross a section boundary and "
    "always fit the embedding model."
)

if data.origin == "none":
    st.error("No parse outputs and no snapshot: run `rag parse all`, then reload.")
    st.stop()
if data.origin == "live":
    st.caption(f"Data: live from `{files['chunks'].parent.parent}`.")
else:
    st.caption(
        f"Data: snapshot from commit {data.snapshot.git_commit[:8]} "
        f"({data.snapshot.created_at:%Y-%m-%d}). Numbers cover the whole corpus; "
        "examples are a sample."
    )
if data.stale:
    st.warning(
        f"The snapshot predates the current {', '.join(data.stale)}: run "
        "`uv run rag showcase snapshot --stage parse`."
    )
view = data.overview
seed_key = "parse_seed"
st.session_state.setdefault(seed_key, config.showcase.seed)

# --- Headline numbers ---------------------------------------------------------------

cols = st.columns(5)
cols[0].metric("Accepted records", f"{view.accepted:,}")
cols[1].metric("Parsed by GROBID", f"{view.outcomes['grobid']:,}")
cols[2].metric(
    "Not parsed",
    f"{view.outcomes['no_pdf'] + view.outcomes['failed'] + view.outcomes['fallback']:,}",
    help=f"No PDF: {view.outcomes['no_pdf']}, GROBID failed: {view.outcomes['failed']}, "
    f"PyMuPDF fallback used: {view.outcomes['fallback']}.",
)
cols[3].metric("Papers indexed", f"{view.indexed_papers:,}")
cols[4].metric(
    "Chunks",
    f"{view.chunks:,}",
    help=f"Median {view.chunks_per_paper_median} per paper, max {view.chunks_per_paper_max}.",
)

grobid = config.parse.grobid
st.markdown(
    f"**PDF → TEI XML** with GROBID (`{grobid.image}`, CRF models, CPU; D-013), "
    "external lookups off so output depends only on the PDF (D-014) → **structured "
    "paper**: section tree rebuilt from heading numbers, paragraphs, sentences with "
    "their citation markers, tables, figures, formulas, references (`parse/tei.py`) → "
    "**chunks** (`parse/chunker.py`). PyMuPDF is the fallback when GROBID returns "
    f"nothing (D-015); used {view.outcomes['fallback']} times in this run."
)

# --- Element decisions --------------------------------------------------------------

st.header("What happens to each element, and why")
per = view.per_element
rows = [
    {
        "element": "Abstract",
        "decision": "Its own chunk(s)",
        "why": "Usually the best single summary of a paper; never mixed with body text.",
        "chunks": f"{per['abstract']:,}",
        "decision ref": "PLAN",
    },
    {
        "element": "Paragraphs",
        "decision": f"Packed within one section to ~{chunking.target_tokens} tokens "
        f"(max {chunking.max_tokens}); {chunking.overlap_tokens}-token overlap of whole "
        "sentences, same section only",
        "why": "A chunk is the citation target: crossing a section would cite the wrong one. "
        "Overlap keeps a sentence's context when a section is split.",
        "chunks": f"{per['paragraph']:,}",
        "decision ref": "D-016, D-017",
    },
    {
        "element": "Tables",
        "decision": "Own chunk: caption + markdown rows, split every "
        f"{chunking.table_max_rows_per_chunk} rows with the header row repeated",
        "why": 'Tables answer lookup questions ("what vocabulary size did X use?"); each part '
        "keeps its header so the numbers keep their meaning.",
        "chunks": f"{per['table']:,}",
        "decision ref": "D-020",
    },
    {
        "element": "Figure captions",
        "decision": "Own chunk" if chunking.captions == "own_chunk" else "Dropped",
        "why": "Captions state results in one sentence; attached to nearby text they would "
        "dilute it, alone they are precise targets.",
        "chunks": f"{per['figure_caption']:,}",
        "decision ref": "D-018",
    },
    {
        "element": "Formulas",
        "decision": f"Display formulas up to {chunking.formula_max_chars} characters kept in "
        "place; longer ones become `[equation]`",
        "why": "Short formulas carry searchable symbols; long ones are GROBID noise that "
        "wastes tokens and confuses the embedder.",
        "chunks": f"{view.formula_placeholders:,} placeholders",
        "decision ref": "D-019",
    },
    {
        "element": "References",
        "decision": "Never chunked; stored in references.jsonl",
        "why": "A bibliography entry matches queries on titles of *other* papers. Kept on "
        "disk for duplicate detection and citation analysis.",
        "chunks": f"0 ({view.references:,} stored)",
        "decision ref": "PLAN",
    },
]
st.dataframe(rows, hide_index=True, width="stretch")
st.caption(
    f"{view.paragraphs_citing:,} of {view.paragraphs:,} paragraph chunks carry at least one "
    "resolved citation marker (which bibliography entry a sentence cites); references are "
    f"stored for {view.papers_with_references:,} papers."
)

# --- Examples per element -------------------------------------------------------------


def render_chunk(chunk: Chunk, note: str = "") -> None:
    with st.container(border=True):
        number = f"§{chunk.section_number} " if chunk.section_number else ""
        page = f"p.{chunk.page}" if chunk.page else "page ?"
        st.caption(
            f"`{chunk.chunk_id}` · {chunk.element_type} · {chunk.token_count} body tokens "
            f"+ {chunk.header_token_count} header · {page} · {number}"
            f"{' > '.join(chunk.section_path)}"
            + (f" · cites {', '.join(chunk.cited_refs)}" if chunk.cited_refs else "")
        )
        st.code(f"{chunk.header}\n{chunk.text}", language=None, wrap_lines=True)
        if note:
            st.caption(note)


st.header("Examples, drawn from the data")
st.caption(
    "Each box is exactly what the embedder sees: the header line (`Title > Section`, "
    "stored separately, D-029) and then the chunk text."
)
if st.button("Show others", key="parse_resample"):
    st.session_state[seed_key] += 1
seed = st.session_state[seed_key]
tabs = st.tabs(["Abstract", "Paragraph", "Table", "Figure caption", "Formula", "Reference"])
pools = [
    data.element_pools["abstract"],
    data.element_pools["paragraph"],
    data.element_pools["table"],
    data.element_pools["figure_caption"],
    data.formula_pool,
]
for tab, pool in zip(tabs[:5], pools, strict=True):
    with tab:
        picked = parse.sample(pool, 1, seed, parse.chunk_key)
        st.caption(f"1 of {len(pool):,} in the pool.")
        for chunk in picked:
            render_chunk(chunk)
with tabs[5]:
    for ref in parse.sample(data.references, 3, seed, parse.reference_key):
        with st.container(border=True):
            st.caption(f"`{ref.record_id}` · `{ref.ref_id}` · never indexed")
            st.write(ref.raw or ref.title or "(no text)")
            st.caption(
                f"Parsed: title={ref.title!r}, year={ref.year}, "
                f"authors={len(ref.authors)}, doi={ref.doi}, arXiv={ref.arxiv_id}"
            )

# --- Sizes -------------------------------------------------------------------------------

st.header("Chunk sizes: nothing is silently truncated")
left, right = st.columns([3, 1])
with left:
    bins = [{"tokens": b.start, "element": b.element, "chunks": b.chunks} for b in view.token_bins]
    elements = list(ELEMENT_COLOURS)
    bars = (
        alt.Chart(alt.Data(values=bins))
        .mark_bar()
        .encode(
            x=alt.X(
                "tokens:Q",
                bin=alt.Bin(step=config.showcase.parse.token_bin_width),
                title="tokens the model sees (header + body + 2 special)",
            ),
            y=alt.Y("sum(chunks):Q", title="chunks"),
            color=alt.Color(
                "element:N",
                scale=alt.Scale(domain=elements, range=[ELEMENT_COLOURS[e] for e in elements]),
            ),
            tooltip=["element:N", "sum(chunks):Q"],
        )
    )
    limit = (
        alt.Chart(alt.Data(values=[{"limit": view.model_limit}]))
        .mark_rule(strokeDash=[4, 3])
        .encode(x="limit:Q")
    )
    st.altair_chart(bars + limit, width="stretch")
    st.caption(
        f"Dashed line: the model's limit, {view.model_limit} tokens. Target body size "
        f"{chunking.target_tokens}, hard max {chunking.max_tokens}; the header must fit in "
        "the rest."
    )
with right:
    st.metric("Chunks over the limit", view.over_limit, help="Must be 0: checked per chunk.")
    st.dataframe(
        [{"element": e, "median body tokens": n} for e, n in view.median_tokens.items()],
        hide_index=True,
    )
    st.caption(
        f"Counted with the embedding model's own tokenizer (`{chunking.tokenizer_model}`, "
        "D-016): a different tokenizer would miscount and the model would cut the end off."
    )

# --- One paper ------------------------------------------------------------------------------

st.header("One paper, structure to chunks")
ids = data.paper_ids
default = parse.sample(ids, 1, seed, str)
record_id = st.selectbox(
    "Paper (type to search)",
    ids,
    index=ids.index(default[0]) if default else 0,
    format_func=lambda r: f"{data.titles.get(r, r)} ({r})",
    key="parse_paper",
)
paper = parse.open_paper(config, data, record_id) if record_id else None
if paper is None:
    st.info("This paper's structure isn't available.")
else:
    st.caption(
        f"Parser: {paper.parser} · {len(paper.sections)} sections · {paper.tables} tables · "
        f"{paper.figures} figures · {paper.formulas} formulas · {paper.references} references "
        f"(stored, not indexed) · {len(paper.chunks)} chunks"
    )
    left, right = st.columns([1, 2])
    with left:
        st.markdown("**Section tree (rebuilt from heading numbers)**")
        st.code(
            "\n".join(
                f"{'  ' * (s.level - 1)}{s.number + ' ' if s.number else ''}{s.title}"
                f"  ({s.paragraphs} ¶)"
                for s in paper.sections
            )
            or "(no sections)",
            language=None,
        )
    with right:
        groups = sorted(
            {" > ".join(c.section_path) for c in paper.chunks},
            key=lambda g: min(c.chunk_id for c in paper.chunks if " > ".join(c.section_path) == g),
        )
        group = st.selectbox("Section", groups, key="parse_section")
        in_group = [c for c in paper.chunks if " > ".join(c.section_path) == group]
        previous = None
        for chunk in in_group:
            repeated = parse.overlap(previous, chunk) if previous else ""
            note = (
                f"Overlap: repeats “{repeated[:160]}” from the previous chunk." if repeated else ""
            )
            render_chunk(chunk, note)
            previous = chunk

# --- Quality checks -------------------------------------------------------------------------

st.header("Quality checks over every chunk")
st.write(
    "Heuristics for problems seen while inspecting chunks, counted over the whole corpus. "
    "The clean-up rules in D-039 fixed most of them (before and after counts are recorded "
    "there); what's left is what the rules don't catch."
)
for check in parse.CHECKS:
    pool = data.check_pools.get(check.key, [])
    with st.expander(f"{check.label}: {view.checks[check.key]:,} chunks"):
        st.write(check.why)
        for chunk in parse.sample(pool, 2, seed, parse.chunk_key):
            render_chunk(chunk)

# --- Probe ---------------------------------------------------------------------------------

dilution = FIGURES / "phase2-dilution.png"
if dilution.exists():
    st.header("Why ~300 tokens: the dilution probe")
    st.image(str(dilution))
    st.caption(
        "One answering paragraph embedded alone, then padded with more of the paper, "
        "compared with the question. Similarity falls as unrelated text is added when the "
        "answer comes last, and past 512 tokens the model silently ignores the rest. "
        "From `probes/phase-2-dilution.py`; table in `docs/figures/phase2-dilution.md`."
    )
