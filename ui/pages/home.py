"""The system at a glance: the block diagram from docs/architecture.md."""

from pathlib import Path

import streamlit as st

ARCHITECTURE = Path(__file__).resolve().parents[2] / "docs" / "architecture.md"

st.title("RAG over research papers on LLM tokenizers")
st.write(
    "Each page in the sidebar walks through one stage: its key numbers, how it "
    "decides, and examples drawn from the data. Everything is computed from the "
    "stage outputs when the page loads, so it always matches the current data."
)
text = ARCHITECTURE.read_text(encoding="utf-8")
# The first fenced block is the diagram; read it from the doc so they can't diverge.
diagram = text.split("```", 2)[1].strip("\n")
st.code(diagram, language=None)
st.caption(f"From `{ARCHITECTURE.relative_to(ARCHITECTURE.parents[1])}`.")
