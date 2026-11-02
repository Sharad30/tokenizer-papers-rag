"""Showcase UI entrypoint: `make ui` (or `uv run streamlit run ui/app.py`).

One page per pipeline stage, added as each stage is reviewed. Pages only
render: every number and example comes from `rag.showcase`, computed from
the stage outputs (D-037, D-038). Set RAG_CONFIG to use another config.
"""

import logging
from pathlib import Path

import streamlit as st

PAGES = Path(__file__).parent / "pages"

# After each run Streamlit's file watcher touches every loaded module to find
# its files. On transformers' lazy image models (YOLOS, ZoeDepth...) that
# triggers an import of torchvision, which this project doesn't install, and
# the watcher logs ~100 tracebacks. Harmless: those modules are never used,
# and site-packages isn't watched anyway. Silence only that logger.
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

st.set_page_config(page_title="Tokenizer papers RAG", layout="wide")
page = st.navigation(
    {
        "Overview": [st.Page(PAGES / "home.py", title="The system", default=True)],
        "Pipeline": [
            st.Page(PAGES / "collect.py", title="1 · Collect"),
            st.Page(PAGES / "parse.py", title="2 · Parse and chunk"),
            st.Page(PAGES / "dedup.py", title="3 · Versions and duplicates"),
            st.Page(PAGES / "embeddings.py", title="4 · Pick an embedding model"),
        ],
    }
)
page.run()
