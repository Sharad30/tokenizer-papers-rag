"""Showcase UI entrypoint: `make ui` (or `uv run streamlit run ui/app.py`).

One page per pipeline stage, added as each stage is reviewed. Pages only
render: every number and example comes from `rag.showcase`, computed from
the stage outputs (D-037, D-038). Set RAG_CONFIG to use another config.
"""

from pathlib import Path

import streamlit as st

PAGES = Path(__file__).parent / "pages"

st.set_page_config(page_title="Tokenizer papers RAG", layout="wide")
page = st.navigation(
    {
        "Overview": [st.Page(PAGES / "home.py", title="The system", default=True)],
        "Pipeline": [
            st.Page(PAGES / "collect.py", title="1 · Collect"),
            st.Page(PAGES / "parse.py", title="2 · Parse and chunk"),
        ],
    }
)
page.run()
