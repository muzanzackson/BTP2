import streamlit as st
import streamlit.components.v1 as components
import subprocess
from pathlib import Path

# ── Project root (same dir as main.py) ─────────────────────────────────────────
ROOT_DIR = Path(__file__).parent

st.set_page_config(page_title="Tree Visualization", layout="wide")

st.title("🌳 3D Phylogenetic Tree Visualization")

# -------------------------------
# SESSION STATE
# -------------------------------
if "generated" not in st.session_state:
    st.session_state.generated = False

# -------------------------------
# 1. Class Selection
# -------------------------------
st.header("1. Select Class Configuration")

class_option = st.radio(
    "Choose class source:",
    ["Default (CIFAR)", "Upload Custom JSON"]
)

label_path = None
default_tree = None

if class_option == "Default (CIFAR)":
    dataset = st.selectbox("Select dataset:", ["CIFAR10", "CIFAR100"])

    if dataset == "CIFAR10":
        label_path = str(ROOT_DIR / "data" / "CIFAR10_classes.json")
        default_tree = str(ROOT_DIR / "data" / "CIFAR10_WordNet.nwk")
    else:
        label_path = str(ROOT_DIR / "data" / "CIFAR100_classes.json")
        default_tree = str(ROOT_DIR / "data" / "CIFAR100_WordNet.nwk")

else:
    uploaded_json = st.file_uploader("Upload JSON file", type=["json"])
    if uploaded_json:
        save_path = Path("temp_labels.json")
        with open(save_path, "wb") as f:
            f.write(uploaded_json.read())
        label_path = str(save_path)

# -------------------------------
# 2. Tree Selection
# -------------------------------
st.header("2. Select Tree")

tree_option = st.radio(
    "Choose tree source:",
    ["Use Default Tree", "Select from trees/ folder", "Upload New Tree"]
)

tree_path = None

if tree_option == "Use Default Tree":
    if default_tree:
        tree_path = default_tree
        st.info(f"Using default tree: {tree_path}")
    else:
        st.warning("No default tree available")

elif tree_option == "Select from trees/ folder":
    tree_dir = Path("trees")
    tree_files = list(tree_dir.glob("*.nwk"))

    if tree_files:
        tree_path = st.selectbox(
            "Choose tree file:",
            [str(f) for f in tree_files]
        )
    else:
        st.warning("No .nwk files found")

else:
    uploaded_tree = st.file_uploader("Upload Newick file", type=["nwk"])
    if uploaded_tree:
        save_path = Path("temp_tree.nwk")
        with open(save_path, "wb") as f:
            f.write(uploaded_tree.read())
        tree_path = str(save_path)

# -------------------------------
# 3. Run Visualization
# -------------------------------
st.header("3. Generate Visualization")

output_html = "tree_visualization.html"

if st.button("Generate Tree Visualization"):
    if not tree_path or not label_path:
        st.error("Please select both tree and label files.")
    else:
        cmd = [
            "python",
            str(ROOT_DIR / "tree_rendering.py"),
            "--nwk", tree_path,
            "--labels", label_path,
            "--output", output_html,
            "--no-open"   # ✅ ADD THIS
         ]

        st.code(" ".join(cmd))

        try:
            with st.spinner("Generating visualization..."):
                subprocess.run(cmd, check=True)

            # ✅ mark as generated
            st.session_state.generated = True

            # ✅ force refresh immediately
            st.rerun()

        except subprocess.CalledProcessError as e:
            st.error(f"Error: {e}")

# -------------------------------
# 4. Display HTML (AUTO REFRESH)
# -------------------------------
st.header("4. View Visualization")

html_file = Path(output_html)

if st.session_state.generated and html_file.exists():
    with open(html_file, "r", encoding="utf-8") as f:
        html_data = f.read()

    components.html(html_data, height=900, scrolling=True)

else:
    st.info("Generate a visualization to view it here.")