import streamlit as st

# 1. Main Page Configuration
st.set_page_config(
    page_title="CNN Analysis Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Define the pages
extractor_page = st.Page(
    "feature_extraction.py", 
    title="1. Extract Embeddings", 
    icon="🔬",
    default=True  # Lands here first if they need to generate embeddings
)

visualizer_page = st.Page(
    "cluster_analysis.py", 
    title="2. Cluster Metrics", 
    icon="📊"
)

robinson_foulds = st.Page(
    "hierarchical_analysis.py", 
    title="3. Hierarchical Analysis", 
    icon="📊"
)

tree_vis = st.Page(
    "tree_visualization.py", 
    title="4. Tree Visualization", 
    icon="📊"
)

# 3. Setup Navigation
pg = st.navigation([extractor_page, visualizer_page,robinson_foulds,tree_vis])

# 4. Run the app
pg.run()