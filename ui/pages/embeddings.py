"""Pick an embedding model with evidence: the candidates, the comparison, and a live query.

Rendering only. Model states come from `rag.showcase.embeddings` (each index
folder's meta and stats), the table from the latest committed
results/<date>-embeddings-ablation.json, the live results from
`rag.showcase.live`.
"""

import os
from pathlib import Path

import streamlit as st

from rag.collect.manifest import read_manifest
from rag.common.config import DEFAULT_CONFIG_PATH, load_config
from rag.eval.identity import identity_groups, load_pairs
from rag.parse.pipeline import CHUNKS, step_dir
from rag.retrieve.collapse import WorkMap, WorksUnavailable
from rag.retrieve.dense import Specter2Encoder
from rag.retrieve.retriever import IndexUnavailable, Retriever
from rag.showcase import embeddings, live
from rag.showcase.sources import fingerprint

CONFIG_PATH = Path(os.environ.get("RAG_CONFIG", DEFAULT_CONFIG_PATH))
FIGURES = Path(__file__).resolve().parents[2] / "docs" / "figures"
STATUS = {"built": "✅ built", "stale": "⚠ stale (rebuild)", "missing": "⏳ not built yet"}


@st.cache_resource(show_spinner="Reading the indexes…")
def load(config_stamp, data_stamp):
    config = load_config(CONFIG_PATH)
    return config, embeddings.load(config)


config = load_config(CONFIG_PATH)
files = embeddings.source_files(config)
results_files = sorted(config.paths.results.glob(embeddings.RESULTS_GLOB))
config, data = load(fingerprint([CONFIG_PATH]), fingerprint([*files.values(), *results_files]))

st.title("4 · Pick an embedding model")
st.write(
    "Which model should embed the chunks, and should keyword search (BM25) sit alongside it? "
    "Tokenizer papers are full of exact names (SentencePiece, BPE-dropout, ByT5, fertility) "
    "that a vector model may blur into their general topic. The comparison runs every "
    "candidate with the deployed duplicate handling on, on the same questions, and reports "
    "Recall@5, MRR, latency and cost."
)
if data.origin == "snapshot":
    st.caption(
        f"Index states: snapshot from commit {data.snapshot.git_commit[:8]} "
        f"({data.snapshot.created_at:%Y-%m-%d}); the indexes aren't on this machine."
    )

# --- Candidates --------------------------------------------------------------------------

st.header("The candidates")
st.dataframe(
    [
        {
            "name": m.name + (" (deployed)" if m.deployed and m.kind != "keyword" else ""),
            "model": m.model,
            "kind": m.kind,
            "index": STATUS[m.status],
            "dims": m.dims,
            "build (CPU min)": m.minutes,
            "size (MB)": m.size_mb,
            "truncated chunks": m.truncated,
            "query prompt": m.query_prompt or "–",
        }
        for m in data.models
    ],
    hide_index=True,
)
st.markdown(
    "- **bge-small / bge-base** (BAAI, general retrieval models): the query gets an "
    "instruction prefix, passages don't (model card).\n"
    "- **SPECTER2** (AllenAI, scientific): trained on citation links, so it learns "
    "*paper-to-paper* similarity. Its ad-hoc query adapter encodes questions and the "
    "proximity adapter encodes passages. It needs an older transformers, so it runs in "
    "its own isolated environment (D-043): a real deployment cost.\n"
    "- **BM25**: exact term matching, no model. **Hybrid** fuses BM25 and one dense model "
    "by rank (RRF, D-030). The **reranker** (bge-reranker-base) is a cross-encoder that reads "
    "query and chunk together and reorders the fused top 50."
)

# --- The comparison ----------------------------------------------------------------------

st.header("The comparison")
result = data.results
if result is None:
    st.info(
        "No comparison yet: it runs once every index is built "
        "(`uv run rag eval ablate --name embeddings`)."
    )
else:
    k = config.eval.recall_k
    rk, mrr = f"recall@{k}", f"mrr@{config.eval.mrr_cutoff}"
    labels = sorted({label for c in result.conditions.values() for label in c.sets})
    rows = []
    for c in result.conditions.values():
        row = {"condition": c.label}
        for label in labels:
            overall = c.sets[label].result.aggregates.get("overall", {}) if label in c.sets else {}
            for metric in (rk, mrr):
                s = overall.get(metric)
                row[f"{metric} ({label})"] = (
                    f"{s.mean:.3f} [{s.ci_low:.2f}, {s.ci_high:.2f}]" if s else "–"
                )
        dup = c.probe.get(f"dup@{config.eval.dup_k}")
        row[f"dup@{config.eval.dup_k}"] = f"{dup.mean:.2f}" if dup else "–"
        row["latency p50 / p95 (ms)"] = (
            f"{c.latency_ms['p50']:.0f} / {c.latency_ms['p95']:.0f}" if c.latency_ms else "–"
        )
        rows.append(row)
    st.dataframe(rows, hide_index=True)
    if result.comparison:
        cmp = result.comparison
        diff = cmp.diffs[rk]
        st.markdown(
            f"**Top two by {rk} ({cmp.set}):** {result.conditions[cmp.a].label} vs "
            f"{result.conditions[cmp.b].label}: difference {diff.mean:+.3f} "
            f"[{diff.ci_low:+.3f}, {diff.ci_high:+.3f}], higher on {cmp.wins} questions, "
            f"lower on {cmp.losses}, equal on {cmp.ties}. An interval containing 0 means "
            "the questions can't separate them."
        )
    st.caption(
        f"`results/{data.results_file}` (commit {result.git_commit[:8]}). Silver questions "
        "were written by an LLM from one passage each (D-044): one gold passage per "
        "question, single-paper lookups, wording that can favour BM25. Gold numbers replace "
        "them once eval/questions.jsonl exists."
    )

terms = FIGURES / "phase6-terms.png"
if terms.exists():
    st.subheader("Do the models know tokenizer terms?")
    st.image(str(terms))
    st.caption("From `probes/phase-6-terms.py`; table in `docs/figures/phase6-terms.md`.")

# --- Live -------------------------------------------------------------------------------

st.header("Try it: one query, several retrievers")
built = [m.name for m in data.models if m.status == "built" and m.kind != "keyword"]
if data.origin == "snapshot" or not built:
    st.info("Live queries need the indexes on this machine.")
    st.stop()

query = st.text_input("Query", value="How does SentencePiece handle whitespace?", key="emb_query")
term = st.text_input(
    "Exact term to look for (optional)",
    value="SentencePiece",
    key="emb_term",
    help="Marks results whose text contains it: does the retriever keep the exact name?",
)
choices = ["BM25"] + [f"{m} dense" for m in built] + [f"{m} hybrid" for m in built]
picked = st.multiselect(
    "Retrievers", choices, default=["BM25", f"{built[0]} dense", f"{built[0]} hybrid"]
)
k = st.slider("Results", 3, 10, 5, key="emb_k")
if any(p.startswith("specter2") for p in picked):
    st.caption("SPECTER2 encodes the query in its own process: ~30 s on first use per query.")


@st.cache_resource(show_spinner="Loading the indexes and the embedding model…")
def retriever_for(model: str, config_stamp, index_stamp):
    config = load_config(CONFIG_PATH)
    return Retriever.load(config, {"hybrid"}, dense=model)


@st.cache_resource(show_spinner="Reading the identity groups…")
def judge_and_works(config_stamp, works_stamp):
    config = load_config(CONFIG_PATH)
    judge = identity_groups(
        read_manifest(config.paths.manifest), load_pairs(config.paths.eval / config.eval.pairs_file)
    )
    works = WorkMap.load(config.paths.works, step_dir(config, "chunk") / CHUNKS)
    return judge, works.record_work


try:
    judge, work_of = judge_and_works(
        fingerprint([CONFIG_PATH]), fingerprint([config.paths.works / "meta.json"])
    )
except (IndexUnavailable, WorksUnavailable, FileNotFoundError) as exc:
    st.info(f"Live queries need the indexes and works on this machine ({exc}).")
    st.stop()

if query.strip() and picked:
    columns = st.columns(len(picked))
    for column, choice in zip(columns, picked, strict=True):
        model = built[0] if choice == "BM25" else choice.split()[0]
        mode = "bm25" if choice == "BM25" else choice.split()[1]
        retriever = retriever_for(
            model, fingerprint([CONFIG_PATH]), fingerprint([files[model], files["bm25"]])
        )
        if isinstance(retriever.encoder, Specter2Encoder) and mode != "bm25":
            with st.spinner("SPECTER2 is encoding the query…"):
                retriever.encoder.encode_queries([query])
        view = live.run_query(retriever, query, k, mode, judge, work_of, term or None)
        with column:
            st.subheader(choice)
            found = sum(bool(h.has_term) for h in view.hits)
            st.caption(
                f"{view.ms:.0f} ms"
                + (f" · '{term}' in {found} of {len(view.hits)}" if term else "")
            )
            for hit in view.hits:
                with st.container(border=True):
                    mark = "✅ " if hit.has_term else ("· " if term else "")
                    st.caption(f"{mark}**#{hit.rank}** · {hit.title[:60]}")
                    st.write(live.preview(hit.text, 160))
