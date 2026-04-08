import streamlit as st

# ─── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CNN Embedding Analysis Suite",
    page_icon=":brain:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Page Definitions ──────────────────────────────────────────────────────────
extractor_page = st.Page(
    "feature_extraction.py",
    title="1. Extract Embeddings",
    default=True
)

visualizer_page = st.Page(
    "cluster_analysis.py",
    title="2. Cluster & t-SNE",
)

robinson_foulds = st.Page(
    "hierarchical_analysis.py",
    title="3. Robinson-Foulds Analysis",
)

tree_vis = st.Page(
    "tree_visualization.py",
    title="4. Tree Visualization",
)

# ─── Navigation ────────────────────────────────────────────────────────────────
pg = st.navigation([extractor_page, visualizer_page, robinson_foulds, tree_vis])

# ─── Sidebar Footer ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.caption("v1.0.0 · CNN Embedding Analysis Suite")

# ─── Run ───────────────────────────────────────────────────────────────────────
pg.run()