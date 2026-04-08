"""
Hierarchical Relationships & Embedding Quality Analyzer
========================================================
Streamlit application for hierarchical embedding analysis using UPGMA trees
and Robinson-Foulds distance.
"""

import io
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.cluster.hierarchy import linkage, to_tree
import numpy as np
import streamlit as st
from ete3 import Tree
import json, pathlib
import subprocess, re
# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="Hierarchical Embedding Analyzer",
#     page_icon="🌿",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Dark editorial aesthetic with sharp accents
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg-primary: #0a0c10;
    --bg-secondary: #111318;
    --bg-card: #161a22;
    --bg-card-hover: #1c2130;
    --border: #252c3d;
    --border-accent: #2e3a55;
    --accent-green: #00e5a0;
    --accent-blue: #4d9fff;
    --accent-amber: #ffb347;
    --accent-purple: #c084fc;
    --accent-red: #ff6b6b;
    --text-primary: #e8eaf0;
    --text-secondary: #8892a4;
    --text-muted: #4a5568;
    --font-display: 'DM Serif Display', serif;
    --font-body: 'Syne', sans-serif;
    --font-mono: 'DM Mono', monospace;
}

/* Global reset */
html, body, [class*="css"] {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
}

/* Main container */
.main .block-container {
    padding: 2rem 2.5rem !important;
    max-width: 1400px !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 1.5rem 1rem !important;
}

/* Hero header */
.hero-header {
    padding: 3rem 0 2rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2.5rem;
}
.hero-title {
    font-family: var(--font-display);
    font-size: 3.2rem;
    line-height: 1.1;
    color: var(--text-primary);
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.02em;
}
.hero-title span {
    color: var(--accent-green);
    font-style: italic;
}
.hero-subtitle {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--text-muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 0.8rem 0 0 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,229,160,0.08);
    border: 1px solid rgba(0,229,160,0.25);
    color: var(--accent-green);
    font-family: var(--font-mono);
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.3rem 0.75rem;
    border-radius: 2px;
    margin-top: 1rem;
}

/* Upload zone */
.upload-zone {
    background: var(--bg-card);
    border: 1.5px dashed var(--border-accent);
    border-radius: 8px;
    padding: 3rem 2rem;
    text-align: center;
    transition: all 0.3s ease;
    margin: 1rem 0;
    position: relative;
    overflow: hidden;
}
.upload-zone::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-green), transparent);
    opacity: 0.5;
}
.upload-icon {
    font-size: 2.5rem;
    margin-bottom: 0.75rem;
    display: block;
}
.upload-text {
    font-family: var(--font-mono);
    font-size: 0.8rem;
    color: var(--text-secondary);
    letter-spacing: 0.05em;
}
.upload-hint {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--text-muted);
    margin-top: 0.4rem;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1.5rem 0;
}
.metric-card {
    flex: 1;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
}
.metric-card.green::after { background: var(--accent-green); }
.metric-card.blue::after  { background: var(--accent-blue); }
.metric-card.amber::after { background: var(--accent-amber); }
.metric-card.purple::after{ background: var(--accent-purple); }
.metric-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-muted);
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: var(--font-display);
    font-size: 2rem;
    line-height: 1;
    color: var(--text-primary);
}
.metric-sub {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--text-secondary);
    margin-top: 0.3rem;
}

/* Section headers */
.section-header {
    display: flex;
    align-items: baseline;
    gap: 1rem;
    margin: 2.5rem 0 1.5rem 0;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border);
}
.section-title {
    font-family: var(--font-display);
    font-size: 1.6rem;
    color: var(--text-primary);
    margin: 0;
    letter-spacing: -0.01em;
}
.section-num {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--accent-green);
    letter-spacing: 0.1em;
}

/* RF Table */
.rf-table-container {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    margin: 1rem 0;
}
.rf-table-header {
    background: var(--bg-secondary);
    padding: 1rem 1.5rem;
    border-bottom: 1px solid var(--border);
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--text-secondary);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
table.rf-table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--font-mono);
    font-size: 0.8rem;
}
table.rf-table th {
    background: var(--bg-secondary);
    padding: 0.7rem 1rem;
    text-align: center;
    color: var(--text-secondary);
    font-weight: 500;
    border-bottom: 1px solid var(--border);
    font-size: 0.72rem;
    letter-spacing: 0.05em;
}
table.rf-table td {
    padding: 0.65rem 1rem;
    text-align: center;
    border-bottom: 1px solid var(--border);
    color: var(--text-primary);
}
table.rf-table tr:last-child td { border-bottom: none; }
table.rf-table tr:hover td { background: var(--bg-card-hover); }
.rf-low  { color: var(--accent-green) !important; font-weight: 600; }
.rf-mid  { color: var(--accent-amber) !important; font-weight: 600; }
.rf-high { color: var(--accent-red) !important;   font-weight: 600; }

/* Newick display */
.newick-block {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent-green);
    border-radius: 4px;
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--accent-green);
    word-break: break-all;
    line-height: 1.6;
    overflow-x: auto;
}
.newick-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    margin-bottom: 0.3rem;
}
.newick-rf {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--text-secondary);
    margin-top: 0.3rem;
}

/* Progress / log */
.log-container {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1rem 1.25rem;
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--text-secondary);
    max-height: 200px;
    overflow-y: auto;
    line-height: 1.8;
}
.log-ok   { color: var(--accent-green); }
.log-info { color: var(--accent-blue); }
.log-warn { color: var(--accent-amber); }

/* Buttons */
.stButton > button {
    background: var(--accent-green) !important;
    color: #0a0c10 !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: var(--font-body) !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    padding: 0.65rem 2rem !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    background: #00ffb3 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(0,229,160,0.3) !important;
}

/* Selectbox / multiselect */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 4px !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1.5px dashed var(--border-accent) !important;
    border-radius: 8px !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"] label {
    color: var(--text-secondary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
}

/* Streamlit override for headers */
h1, h2, h3, h4 {
    font-family: var(--font-display) !important;
    color: var(--text-primary) !important;
}

/* Info/warning/success boxes */
[data-testid="stAlert"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}
[data-testid="stExpander"] summary {
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
}

/* Divider */
hr {
    border-color: var(--border) !important;
    margin: 2rem 0 !important;
}

/* Tabs */
[data-testid="stTabs"] button {
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    color: var(--text-secondary) !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent-green) !important;
    border-bottom-color: var(--accent-green) !important;
}

/* Sidebar labels */
[data-testid="stSidebar"] label {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: var(--text-secondary) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border-accent); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* Image containers */
.img-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.5rem;
    margin: 1rem 0;
}
.img-caption {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--text-muted);
    text-align: center;
    margin-top: 0.75rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Sidebar brand */
.sidebar-brand {
    font-family: var(--font-display);
    font-size: 1.3rem;
    color: var(--accent-green);
    margin-bottom: 0.25rem;
}
.sidebar-version {
    font-family: var(--font-mono);
    font-size: 0.62rem;
    color: var(--text-muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
}
</style>
""", unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

METRICS       = ["euclidean", "cityblock", "canberra"]
METRIC_LABELS = ["Euclidean", "Manhattan", "Canberra"]

METRIC_COLORS = {
    "euclidean": "#4d9fff",
    "cityblock":  "#ffb347",
    "canberra":   "#c084fc",
}
REF_COLOR  = "#ff6b6b"

MPL_BG     = "#0a0c10"
MPL_CARD   = "#161a22"
MPL_BORDER = "#252c3d"
MPL_TEXT   = "#e8eaf0"
MPL_TEXT2  = "#8892a4"

# ─────────────────────────────────────────────────────────────────────────────
# TAXONOMY LOADER  — default files OR custom uploads
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_DIR  = pathlib.Path(__file__).parent          # project root (same folder as main.py)
_DATA_DIR     = _DEFAULT_DIR / "data"                  # taxonomy reference files

def load_taxonomy(nwk_source, json_source) -> tuple[str, dict]:
    """
    Accepts either:
      • pathlib.Path  → read from disk
      • bytes         → uploaded file bytes
    Returns (newick_string, {str(idx): class_name})
    """
    # ── Newick ──────────────────────────────────────────────────────────────
    if isinstance(nwk_source, (str, pathlib.Path)):
        newick = pathlib.Path(nwk_source).read_text().strip()
    else:
        newick = nwk_source.decode("utf-8").strip()

    # ── Class names (JSON or plain text "idx name" per line) ─────────────────
    if isinstance(json_source, (str, pathlib.Path)):
        raw = pathlib.Path(json_source).read_text()
    else:
        raw = json_source.decode("utf-8")

    try:
        data = json.loads(raw)
        # Support both {idx: name} and [{id: idx, name: name}, ...]
        if isinstance(data, list):
            names = {str(item["id"]): item["name"] for item in data}
        else:
            names = {str(k): str(v) for k, v in data.items()}
    except json.JSONDecodeError:
        # Fallback: plain text "0 airplane\n1 automobile\n…"
        names = {}
        for line in raw.strip().splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                names[parts[0]] = parts[1]

    return newick, names


def get_default_taxonomy(dataset_type: str) -> tuple[str, dict]:
    """Load from the default .nwk / .json files shipped with the app."""
    prefix = "CIFAR10" if dataset_type == "cifar10" else "CIFAR100"
    nwk  = _DATA_DIR / f"{prefix}_WordNet.nwk"
    jsn  = _DATA_DIR / f"{prefix}_classes.json"
    return load_taxonomy(nwk, jsn)
# ─────────────────────────────────────────────────────────────────────────────
# CORE ALGORITHMS
# ─────────────────────────────────────────────────────────────────────────────

def load_data(file_bytes, names_map: dict | None = None):
    data = np.load(io.BytesIO(file_bytes))
    embed_key = next(k for k in data.files if "embed" in k.lower() or "feat" in k.lower())
    label_key = next(k for k in data.files if "label" in k.lower() or "target" in k.lower())
    embeddings = data[embed_key].astype(np.float64)
    raw_labels = data[label_key]

    # Try integer labels first
    try:
        labels = raw_labels.astype(int)
    except (ValueError, TypeError):
        # Fall back to string name → index lookup using provided names_map
        if names_map is not None:
            name_to_idx = {v.lower(): int(k) for k, v in names_map.items()}
        else:
            # Last resort: build a sorted unique mapping on the fly
            unique_names = sorted(set(str(l).lower() for l in raw_labels))
            name_to_idx  = {name: i for i, name in enumerate(unique_names)}

        try:
            labels = np.array([name_to_idx[str(l).lower()] for l in raw_labels], dtype=int)
        except KeyError as e:
            raise ValueError(
                f"Label '{e.args[0]}' not found in class names map. "
                "Check that your taxonomy .json matches the dataset."
            )

    n_classes    = len(np.unique(labels))
    dataset_type = "cifar10" if n_classes <= 10 else "cifar100"
    return embeddings, labels, list(data.files), dataset_type

def list_embedding_folder(folder: str = "embeddings") -> list[str]:
    """Return sorted list of .npz files inside the embeddings/ folder."""
    p = pathlib.Path(folder)
    if not p.exists():
        return []
    return sorted([f.name for f in p.glob("*.npz")])

def boundary_group(embeddings, labels, class_id, metric):
    target_mask = (labels == class_id)
    class_pts   = embeddings[target_mask]
    other_pts   = embeddings[~target_mask]

    G_original = class_pts.mean(axis=0)

    d_class = cdist(class_pts, G_original[None], metric=metric).ravel()
    d_other = cdist(other_pts, G_original[None], metric=metric).ravel()

    r = float(np.min(d_class))

    step = max((np.max(d_class) - np.min(d_class)) / 50.0, 1e-6)

    while True:
        dy = np.sum((d_class > r) & (d_class <= r + step))
        dz = np.sum((d_other > r) & (d_other <= r + step))

        if dy >= dz:
            r += step
        else:
            break

    d_all = cdist(embeddings, G_original[None], metric=metric).ravel()
    group_pts = embeddings[d_all <= r]

    return group_pts, G_original


def compute_centres(embeddings, labels, unique_labels, metrics, progress_cb=None):
    centres = {"G": {}, "MPP": {}}
    total = len(metrics) * len(unique_labels)
    done  = 0

    for metric in metrics:
        g_list, mpp_list = [], []

        for lbl in unique_labels:
            group, G_orig = boundary_group(embeddings, labels, lbl, metric)

            if len(group) == 0:
                group = embeddings[labels == lbl]

            G_boundary = group.mean(axis=0)
            mu = (group - G_orig).mean(axis=0)
            P = G_boundary + mu

            g_list.append(G_boundary)
            mpp_list.append(P)

            done += 1
            if progress_cb:
                progress_cb(done / total, f"Boundary estimation: {metric} | class {lbl}")

        centres["G"][metric]   = np.vstack(g_list)
        centres["MPP"][metric] = np.vstack(mpp_list)

    return centres



def upgma(pts, leaf_names, metric):
    Z = linkage(pts, method="average", metric=metric)
    tree, _ = to_tree(Z, rd=True)

    def build(node):
        if node.is_leaf():
            return str(leaf_names[node.id])
        left  = build(node.left)
        right = build(node.right)
        return f"({left},{right})"

    return tree, build(tree) + ";"


def rf_distance(newick_gen, newick_ref):
    t_gen = Tree(newick_gen, format=1)
    t_ref = Tree(newick_ref, format=1)

    assert set(t_gen.get_leaf_names()) == set(t_ref.get_leaf_names())

    rf, max_rf, *_ = t_gen.robinson_foulds(t_ref, unrooted_trees=False)

    return int(rf), int(max_rf)

def save_trees_to_disk(trees_6, metrics, metric_labels, base_dir: pathlib.Path):
    """
    Always writes exactly 6 .nwk files into <base_dir>/trees/
    Named: UPGMA_<MetricLabel>_<CentreType>.nwk  (e.g. UPGMA_Euclidean_G.nwk)
    Overwrites existing files on every run.
    """
    trees_dir = base_dir / "trees"
    trees_dir.mkdir(exist_ok=True)

    for gm, gm_label in zip(metrics, metric_labels):
        for ctype in ["G", "MPP"]:
            _, nwk = trees_6[(gm, ctype)]
            fname  = trees_dir / f"UPGMA_{gm_label}_{ctype}.nwk"
            fname.write_text(nwk, encoding="utf-8")

    return trees_dir

def build_all_trees(centres, leaf_names, ref_newick, metrics, metric_labels, progress_cb=None):
    trees_6   = {}
    rf_rows   = []
    all_trees = {}
    cache     = {}

    total = len(metrics) ** 2 * 2
    done  = 0

    # Primary trees
    for gm, gm_label in zip(metrics, metric_labels):
        for ctype in ["G", "MPP"]:
            pts = centres[ctype][gm]

            key = (ctype, gm, gm)
            if key not in cache:
                cache[key] = upgma(pts, leaf_names, gm)

            root, nwk = cache[key]
            trees_6[(gm, ctype)] = (root, nwk)

    # RF computation
    for gm, gm_label in zip(metrics, metric_labels):
        row = {"Group Distance": gm_label}

        for tm, tm_label in zip(metrics, metric_labels):
            for ctype in ["G", "MPP"]:
                pts = centres[ctype][gm]

                key = (ctype, gm, tm)
                if key not in cache:
                    cache[key] = upgma(pts, leaf_names, tm)

                _, nwk = cache[key]

                rf, _ = rf_distance(nwk, ref_newick)

                k = (gm, tm, ctype)
                row[k] = rf
                all_trees[k] = nwk

                done += 1
                if progress_cb:
                    progress_cb(done / total,
                        f"UPGMA + RF: group={gm_label} tree={tm_label} centre={ctype} → RF={rf}")

        rf_rows.append(row)

    return trees_6, rf_rows, all_trees

# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING  (dark theme matplotlib)
# ─────────────────────────────────────────────────────────────────────────────


def get_leaf_names_in_order(node, leaf_names):
    if node.is_leaf():
        return [leaf_names[node.id]]
    return (
        get_leaf_names_in_order(node.left, leaf_names) +
        get_leaf_names_in_order(node.right, leaf_names)
    )

def _apply_dark_fig(fig):
    fig.patch.set_facecolor(MPL_BG)
    for ax in fig.axes:
        ax.set_facecolor(MPL_CARD)
        ax.tick_params(colors=MPL_TEXT2, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(MPL_BORDER)
        ax.xaxis.label.set_color(MPL_TEXT2)
        ax.yaxis.label.set_color(MPL_TEXT2)
        ax.title.set_color(MPL_TEXT)


def draw_cladogram(tree_newick, leaf_names_map, ax, title, color, n_classes=10):
    t      = Tree(tree_newick)
    leaves = t.get_leaves()
    n      = len(leaves)
    leaf_y = {lf.name: i / max(n - 1, 1) for i, lf in enumerate(leaves)}

    def depth(node):
        d, nd = 0, node
        while nd.up:
            d += 1; nd = nd.up
        return d

    max_depth = max(depth(lf) for lf in leaves)

    def mean_y(node):
        return np.mean([leaf_y[lf.name] for lf in node.get_leaves()])

    # Scale font size down for CIFAR-100 (100 leaves)
    leaf_fontsize = 7.5 if n_classes <= 10 else 5.0

    ax.set_facecolor(MPL_CARD)
    ax.set_xlim(-0.05, 1.55)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")
    ax.set_title(title, fontsize=8.5, fontweight="bold", pad=5, color=color, wrap=True)

    for node in t.traverse("levelorder"):
        x = depth(node) / max_depth
        y = mean_y(node)
        if node.is_leaf():
            label = leaf_names_map.get(node.name, node.name)
            ax.text(x + 0.03, y, label, va="center", ha="left",
                    fontsize=leaf_fontsize, color=MPL_TEXT)
            ax.plot(x, y, "o", color=color, ms=3 if n_classes > 10 else 5, zorder=3)
        else:
            child_ys = [mean_y(c) for c in node.children]
            ax.plot([x, x], [min(child_ys), max(child_ys)],
                    color=color, lw=1.2 if n_classes > 10 else 1.8, zorder=2)
            ax.plot(x, y, "s", color=color, ms=2 if n_classes > 10 else 4, alpha=0.7, zorder=3)
        if node.up:
            px = depth(node.up) / max_depth
            ax.plot([px, x], [y, y], color=color,
                    lw=1.2 if n_classes > 10 else 1.8, zorder=2)


def draw_upgma_dendrogram(upgma_root, leaf_names_map, ax, title, color, n_classes=10):
    leaf_names = list(leaf_names_map.keys())

    ordered = get_leaf_names_in_order(upgma_root, leaf_names)
    n = len(ordered)

    leaf_y = {name: i for i, name in enumerate(ordered)}

    def get_y(node):
        if node.is_leaf():
            return leaf_y[leaf_names[node.id]]
        return (get_y(node.left) + get_y(node.right)) / 2.0

    all_heights = []

    def collect_h(node):
        if not node.is_leaf():
            all_heights.append(node.dist)
            collect_h(node.left)
            collect_h(node.right)

    collect_h(upgma_root)
    max_h = max(all_heights) if all_heights else 1.0

    leaf_fontsize = 8 if n_classes <= 10 else 4.5

    ax.set_facecolor(MPL_CARD)
    ax.set_title(title, fontsize=8.5, fontweight="bold", pad=5, color=color, wrap=True)
    ax.set_xlabel("Merge height", fontsize=8, color=MPL_TEXT2)

    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [leaf_names_map.get(name, name) for name in ordered],
        fontsize=leaf_fontsize, color=MPL_TEXT
    )

    ax.invert_yaxis()
    ax.tick_params(axis='x', colors=MPL_TEXT2, labelsize=7)

    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_edgecolor(MPL_BORDER)

    def draw_node(node):
        if node.is_leaf():
            return

        h = node.dist / max_h
        yl = get_y(node.left)
        yr = get_y(node.right)

        hl = (node.left.dist / max_h) if not node.left.is_leaf() else 0.0
        hr = (node.right.dist / max_h) if not node.right.is_leaf() else 0.0

        ax.plot([h, h], [yl, yr], color=color, lw=1.8)
        ax.plot([hl, h], [yl, yl], color=color, lw=1.8)
        ax.plot([hr, h], [yr, yr], color=color, lw=1.8)

        draw_node(node.left)
        draw_node(node.right)

    draw_node(upgma_root)

    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.5, n - 0.5)
    ax.grid(axis='x', color=MPL_BORDER, linewidth=0.5, alpha=0.5)

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()


def make_cladogram_figure(ref_newick, trees_6, rf_rows, leaf_names_map,
                          metrics, metric_labels, n_classes=10):
    panels = []
    for gm, gm_label in zip(metrics, metric_labels):
        for ctype in ["G", "MPP"]:
            rf = rf_rows[metrics.index(gm)].get((gm, gm, ctype), "?")
            _, nwk = trees_6[(gm, ctype)]
            panels.append({"metric": gm, "ctype": ctype, "label": gm_label,
                            "nwk": nwk, "rf": rf, "color": METRIC_COLORS[gm]})

    # Taller figure for CIFAR-100
    fig_h = 15 if n_classes <= 10 else 28
    fig, axes = plt.subplots(3, 3, figsize=(22, fig_h))
    fig.patch.set_facecolor(MPL_BG)

    draw_cladogram(ref_newick, leaf_names_map, axes[0, 0],
                   "WordNet Reference Tree\n(hardcoded topology)", REF_COLOR, n_classes)

    positions = [(0,1),(0,2),(1,0),(1,1),(1,2),(2,0)]
    for idx, (r, c) in enumerate(positions):
        p = panels[idx]
        title = f"UPGMA — {p['label']} / {p['ctype']}\nRF (same metric) = {p['rf']}"
        draw_cladogram(p["nwk"], leaf_names_map, axes[r, c], title, p["color"], n_classes)

    for r, c in [(2,1),(2,2)]:
        axes[r, c].axis("off")
        axes[r, c].set_facecolor(MPL_BG)

    legend_handles = [
        mpatches.Patch(color=REF_COLOR,                   label="WordNet Reference"),
        mpatches.Patch(color=METRIC_COLORS["euclidean"],  label="Euclidean"),
        mpatches.Patch(color=METRIC_COLORS["cityblock"],  label="Manhattan"),
        mpatches.Patch(color=METRIC_COLORS["canberra"],   label="Canberra"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=10, framealpha=0.15, facecolor=MPL_CARD,
               edgecolor=MPL_BORDER, labelcolor=MPL_TEXT,
               bbox_to_anchor=(0.5, 0.01))
    dataset_label = "CIFAR-10" if n_classes <= 10 else "CIFAR-100"
    fig.suptitle(
        f"All 7 Trees: WordNet Reference + 6 UPGMA Trees (G & MPP \u00d7 3 metrics) \u2014 {dataset_label}",
        fontsize=13, fontweight="bold", y=1.01, color=MPL_TEXT)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    return fig_to_bytes(fig)


def make_dendrogram_figure(ref_newick, trees_6, leaf_names_map,
                           metrics, metric_labels, n_classes=10):
    panels = []
    for gm, gm_label in zip(metrics, metric_labels):
        for ctype in ["G", "MPP"]:
            root, nwk = trees_6[(gm, ctype)]
            panels.append({"metric": gm, "ctype": ctype, "label": gm_label,
                            "root": root, "color": METRIC_COLORS[gm]})

    fig_h = 15 if n_classes <= 10 else 32
    fig, axes = plt.subplots(3, 3, figsize=(22, fig_h))
    fig.patch.set_facecolor(MPL_BG)

    draw_cladogram(ref_newick, leaf_names_map, axes[0, 0],
                   "WordNet Reference Tree\n(hardcoded topology)", REF_COLOR, n_classes)

    positions = [(0,1),(0,2),(1,0),(1,1),(1,2),(2,0)]
    for idx, (r, c) in enumerate(positions):
        p = panels[idx]
        title = f"UPGMA Dendrogram — {p['label']} / {p['ctype']}"
        draw_upgma_dendrogram(p["root"], leaf_names_map, axes[r, c], title, p["color"], n_classes)

    for r, c in [(2,1),(2,2)]:
        axes[r, c].axis("off")
        axes[r, c].set_facecolor(MPL_BG)

    legend_handles = [
        mpatches.Patch(color=REF_COLOR,                   label="WordNet Reference"),
        mpatches.Patch(color=METRIC_COLORS["euclidean"],  label="Euclidean"),
        mpatches.Patch(color=METRIC_COLORS["cityblock"],  label="Manhattan"),
        mpatches.Patch(color=METRIC_COLORS["canberra"],   label="Canberra"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=10, framealpha=0.15, facecolor=MPL_CARD,
               edgecolor=MPL_BORDER, labelcolor=MPL_TEXT,
               bbox_to_anchor=(0.5, 0.01))
    dataset_label = "CIFAR-10" if n_classes <= 10 else "CIFAR-100"
    fig.suptitle(
        f"UPGMA Dendrograms with Branch Heights (G & MPP \u00d7 3 metrics) \u2014 {dataset_label}",
        fontsize=13, fontweight="bold", y=1.01, color=MPL_TEXT)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    return fig_to_bytes(fig)


def make_heatmap_figure(rf_rows, metrics, metric_labels, n_classes=10):
    label_to_key = dict(zip(metric_labels, metrics))
    col_labels   = [f"{ml}\n{ct}" for ml in metric_labels for ct in ["G", "MPP"]]
    row_labels   = [r["Group Distance"] for r in rf_rows]

    data = np.zeros((len(rf_rows), len(col_labels)))
    for ri, row in enumerate(rf_rows):
        gm_key = label_to_key[row["Group Distance"]]
        ci = 0
        for tm in metrics:
            for ct in ["G", "MPP"]:
                data[ri, ci] = row.get((gm_key, tm, ct), np.nan)
                ci += 1

    fig, ax = plt.subplots(figsize=(13, 4.5))
    fig.patch.set_facecolor(MPL_BG)
    ax.set_facecolor(MPL_CARD)

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "rf", ["#00e5a0", "#ffb347", "#ff6b6b"], N=256)
    im = ax.imshow(data, cmap=cmap, aspect="auto",
                   vmin=np.nanmin(data), vmax=np.nanmax(data))

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9, color=MPL_TEXT2)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=11, color=MPL_TEXT)
    ax.set_xlabel("Tree Distance Metric × Centre Type", fontsize=10, color=MPL_TEXT2)
    ax.set_ylabel("Group Distance Metric", fontsize=10, color=MPL_TEXT2)

    max_rf = 2 * (n_classes - 1)   # rooted RF max for n leaves
    ax.set_title(
        f"Robinson-Foulds Distance vs WordNet Reference\n"
        f"(lower = more similar to WordNet  |  max rooted RF = {max_rf} for {n_classes} leaves)",
        fontsize=11, fontweight="bold", color=MPL_TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor(MPL_BORDER)
    ax.tick_params(axis='both', colors=MPL_TEXT2)

    thresh = np.nanmax(data) * 0.65
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{int(v)}", ha="center", va="center",
                        fontsize=12, fontweight="bold",
                        color="white" if v > thresh else "#0a0c10")

    cbar = plt.colorbar(im, ax=ax, label="RF distance (rooted)")
    cbar.ax.yaxis.label.set_color(MPL_TEXT2)
    cbar.ax.tick_params(colors=MPL_TEXT2)
    cbar.outline.set_edgecolor(MPL_BORDER)

    plt.tight_layout()
    return fig_to_bytes(fig)



def run_defin_for_all_trees(trees_dir: pathlib.Path, ref_nwk: pathlib.Path | str,
                             metrics, metric_labels, base_dir: pathlib.Path) -> list[dict]:
    """
    Runs DefIn for all 6 UPGMA trees against the reference Newick.
    Handles:
      - ref_nwk as a pathlib.Path (default file on disk)
      - ref_nwk as a str (raw Newick string → written to a temp file)
      - DefIn not installed / not found
      - DefIn returning non-zero exit code
      - Unexpected output format

    Returns a list of dicts:
      [{ "label": "Euclidean / G",
         "file":  "trees/UPGMA_Euclidean_G.nwk",
         "clades": 9,
         "deformity": 5.78,
         "raw": "<full stdout>",
         "error": None  }, ...]
    """
    results = []

    # ── Resolve reference file ────────────────────────────────────────────
    _tmp_ref = None
    if isinstance(ref_nwk, pathlib.Path) and ref_nwk.exists():
        ref_path = str(ref_nwk)
    elif isinstance(ref_nwk, str) and pathlib.Path(ref_nwk).exists():
        ref_path = ref_nwk
    else:
        # ref_nwk is a raw Newick string — write to a temp file
        import tempfile
        _tmp_ref = tempfile.NamedTemporaryFile(
            mode="w", suffix=".nwk", delete=False, encoding="utf-8")
        _tmp_ref.write(str(ref_nwk).strip())
        _tmp_ref.flush()
        ref_path = _tmp_ref.name

    try:
        for gm, gm_label in zip(metrics, metric_labels):
            for ctype in ["G", "MPP"]:
                fname  = trees_dir / f"UPGMA_{gm_label}_{ctype}.nwk"
                result = {
                    "label":      f"{gm_label} / {ctype}",
                    "file":       str(fname.relative_to(base_dir)),
                    "clades":     None,
                    "deformity":  None,
                    "raw":        "",
                    "error":      None,
                }

                # ── File existence check ──────────────────────────────────
                if not fname.exists():
                    result["error"] = f"Tree file not found: {fname}"
                    results.append(result)
                    continue

                # ── Run DefIn ─────────────────────────────────────────────
                cmd = ["python", "lib/DefIn/DefIn.py",
                       "-i", str(fname), "-r", ref_path, "-t"]
                try:
                    proc = subprocess.run(
                        cmd,
                        capture_output=True, text=True, timeout=120)

                    stdout = proc.stdout.strip()
                    stderr = proc.stderr.strip()
                    result["raw"] = stdout + ("\n" + stderr if stderr else "")

                    if proc.returncode != 0:
                        result["error"] = (
                            f"DefIn exited with code {proc.returncode}."
                            + (f"\nstderr: {stderr}" if stderr else ""))
                        results.append(result)
                        continue

                    # ── Parse output ──────────────────────────────────────
                    # Tolerant regex — matches "Number of clades:  9"
                    #                  and    "DEFORMITY INDEX:  5.781481481481482"
                    m_clades = re.search(
                        r"number\s+of\s+clades[:\s]+([0-9]+)",
                        stdout, re.IGNORECASE)
                    m_deform = re.search(
                        r"deformity\s+index[:\s]+([0-9]*\.?[0-9]+(?:e[+-]?[0-9]+)?)",
                        stdout, re.IGNORECASE)

                    if m_clades:
                        result["clades"] = int(m_clades.group(1))
                    if m_deform:
                        result["deformity"] = float(m_deform.group(1))

                    if result["deformity"] is None:
                        result["error"] = (
                            "Could not parse Deformity Index from output.\n"
                            f"Raw output:\n{stdout}")

                except FileNotFoundError:
                    result["error"] = (
                        "DefIn script not found at lib/DefIn/DefIn.py\n"
                        "Check that the lib/DefIn directory is present and "
                        "DefIn.py is executable.")
                except subprocess.TimeoutExpired:
                    result["error"] = "DefIn timed out after 120 s."
                except Exception as exc:
                    result["error"] = f"Unexpected error: {exc}"

                results.append(result)

    finally:
        # Clean up temp reference file if we created one
        if _tmp_ref is not None:
            try:
                pathlib.Path(_tmp_ref.name).unlink()
            except Exception:
                pass

    return results

# ─────────────────────────────────────────────────────────────────────────────
# RF TABLE HTML RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def render_rf_table_html(rf_rows, metrics, metric_labels, n_classes=10):
    max_rf = 2 * (n_classes - 1)

    def rf_class(v, max_rf):
        low_thresh = max_rf * 0.25
        mid_thresh = max_rf * 0.60
        if v <= low_thresh: return "rf-low"
        if v <= mid_thresh: return "rf-mid"
        return "rf-high"

    header_cols = ""
    for ml in metric_labels:
        header_cols += f'<th>{ml}<br><small>G</small></th><th>{ml}<br><small>MPP</small></th>'

    rows_html = ""
    for i, row in enumerate(rf_rows):
        gm = metrics[i]
        rows_html += (f"<tr><td style='text-align:left;font-weight:600;"
                      f"color:var(--text-primary)'>{row['Group Distance']}</td>")
        for tm in metrics:
            for ct in ["G", "MPP"]:
                v   = row.get((gm, tm, ct), "—")
                cls = rf_class(v, max_rf) if isinstance(v, int) else ""
                rows_html += f"<td class='{cls}'>{v}</td>"
        rows_html += "</tr>"

    dataset_label = "CIFAR-10 — 10 leaves" if n_classes <= 10 else "CIFAR-100 — 100 leaves"
    return f"""
    <div class="rf-table-container">
        <div class="rf-table-header">
            Robinson-Foulds Distance Table &nbsp;·&nbsp; Lower = More Similar to WordNet
            &nbsp;·&nbsp; Max Rooted RF ({dataset_label}) = {max_rf}
        </div>
        <table class="rf-table">
            <thead>
                <tr>
                    <th style="text-align:left">Group Distance</th>
                    {header_cols}
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    """


# ─────────────────────────────────────────────────────────────────────────────
# NEWICK STRINGS RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def render_newicks(ref_newick, trees_6, rf_rows, metrics, metric_labels, leaf_names_map):
    blocks = []

    leaf_names = list(leaf_names_map.keys())

    blocks.append({
        "idx": 0,
        "label": "WordNet Reference (hardcoded)",
        "newick": ref_newick,
        "color": REF_COLOR,
        "leaf_order": None,
        "rf": None,
    })

    for i, (gm, gm_label) in enumerate(zip(metrics, metric_labels)):
        for ctype in ["G", "MPP"]:
            root, nwk = trees_6[(gm, ctype)]

            rf = rf_rows[i].get((gm, gm, ctype), "?")

            # ✅ FIX: replace leaves_in_order()
            leaves = get_leaf_names_in_order(root, leaf_names)

            named = " → ".join([leaf_names_map.get(l, l) for l in leaves])

            blocks.append({
                "idx": len(blocks),
                "label": f"UPGMA — {gm_label} / {ctype}",
                "newick": nwk,
                "color": METRIC_COLORS[gm],
                "leaf_order": named,
                "rf": rf,
            })

    return blocks
# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="sidebar-brand">HEA</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-version">Hierarchical Embedding Analyzer · v1.0</div>',
        unsafe_allow_html=True)

    # Dataset badge — shown after file upload
    if "dataset_type" in st.session_state:
        dtype  = st.session_state["dataset_type"]
        color  = "#4d9fff" if dtype == "cifar10" else "#c084fc"
        blabel = "CIFAR-10  ·  10 classes" if dtype == "cifar10" else "CIFAR-100  ·  100 classes"
        st.markdown(
            f'<div style="background:rgba(77,159,255,0.08);border:1px solid {color}33;'
            f'color:{color};font-family:\'DM Mono\',monospace;font-size:0.68rem;'
            f'letter-spacing:0.08em;text-transform:uppercase;padding:0.3rem 0.75rem;'
            f'border-radius:2px;margin-bottom:1rem;text-align:center">{blabel}</div>',
            unsafe_allow_html=True)

    st.markdown("**Methodology Reference**")
    st.markdown(
        '<span style="font-family:\'DM Mono\',monospace;font-size:0.72rem;color:#8892a4">'
        'Hierarchical Relationship Analysis<br>of CNN Embeddings via<br>'
        'Phylogenetic Tree Distance Metrics</span>',
        unsafe_allow_html=True)

    st.divider()

    # WordNet topology info — dynamic based on detected dataset
    st.markdown("**WordNet Taxonomy Source**")

    taxonomy_source = st.radio(
        "Taxonomy source",
        ["Default (CIFAR-10 / CIFAR-100)", "Custom Upload"],
        label_visibility="collapsed",
    )
    st.session_state["taxonomy_source"] = taxonomy_source

    if taxonomy_source == "Custom Upload":
        st.markdown(
            '<span style="font-family:\'DM Mono\',monospace;font-size:0.68rem;'
            'color:var(--text-muted)">Upload your own .nwk and .json/.txt</span>',
            unsafe_allow_html=True)
        custom_nwk  = st.file_uploader("Newick file (.nwk)",  type=["nwk","txt"], key="custom_nwk")
        custom_json = st.file_uploader("Classes file (.json/.txt)", type=["json","txt"], key="custom_json")
        if custom_nwk and custom_json:
            try:
                _nwk, _names = load_taxonomy(custom_nwk.read(), custom_json.read())
                st.session_state["custom_taxonomy"] = (_nwk, _names)
                st.success("✓ Custom taxonomy loaded")
            except Exception as e:
                st.error(f"Parse error: {e}")
                st.session_state.pop("custom_taxonomy", None)
        else:
            st.session_state.pop("custom_taxonomy", None)
            st.markdown(
                '<span style="font-family:\'DM Mono\',monospace;font-size:0.65rem;'
                'color:var(--accent-amber)">⚠ Both files required</span>',
                unsafe_allow_html=True)
    else:
        st.session_state.pop("custom_taxonomy", None)
        # Show topology hint only for CIFAR-10 (too large for CIFAR-100)
        if st.session_state.get("dataset_type") == "cifar10":
            st.markdown(
                '<span style="font-family:\'DM Mono\',monospace;font-size:0.68rem;'
                'color:var(--text-muted)">Vehicles: {airplane, ship}<br>'
                '{automobile, truck}<br><br>Animals: {bird, frog}<br>'
                '{cat, dog}<br>{deer, horse}</span>',
                unsafe_allow_html=True)
        elif st.session_state.get("dataset_type") == "cifar100":
            st.markdown(
                '<span style="font-family:\'DM Mono\',monospace;font-size:0.68rem;'
                'color:var(--text-muted)">CIFAR-100 WordNet hierarchy<br>'
                '(100 classes, deep taxonomy)<br>See Newick string for full topology</span>',
                unsafe_allow_html=True)

    st.divider()
    st.markdown("**Color Key**")
    for metric, color in METRIC_COLORS.items():
        lbl = {"euclidean": "Euclidean", "cityblock": "Manhattan", "canberra": "Canberra"}[metric]
        st.markdown(
            f'<span style="font-family:\'DM Mono\',monospace;font-size:0.72rem">'
            f'<span style="color:{color}">●</span> &nbsp;{lbl}</span>',
            unsafe_allow_html=True)
    st.markdown(
        f'<span style="font-family:\'DM Mono\',monospace;font-size:0.72rem">'
        f'<span style="color:{REF_COLOR}">●</span> &nbsp;WordNet Reference</span>',
        unsafe_allow_html=True)

    st.divider()
    # Class list — dynamic
    if st.session_state.get("dataset_type") == "cifar100":
        st.markdown(
            '<span style="font-family:\'DM Mono\',monospace;font-size:0.62rem;color:#4a5568">'
            'CIFAR-100: 100 fine-grained classes<br>(animals, vehicles, objects,<br>'
            'household items, plants…)</span>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<span style="font-family:\'DM Mono\',monospace;font-size:0.65rem;color:#4a5568">'
            'CIFAR-10 classes:<br>0:airplane 1:automobile<br>2:bird 3:cat 4:deer<br>'
            '5:dog 6:frog 7:horse<br>8:ship 9:truck</span>',
            unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-header">
    <div class="hero-title">Hierarchical <span>Embedding</span><br>Quality Analyzer</div>
    <div class="hero-subtitle">Latent Space · UPGMA · Robinson-Foulds · CIFAR-10 / CIFAR-100</div>
    <div class="hero-badge">Embedding Analysis Suite · Research Tool</div>
</div>
""", unsafe_allow_html=True)

# ─── Upload Section ───────────────────────────────────────────────────────────
# ─── Upload / Select Section ──────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <span class="section-num">01 —</span>
    <span class="section-title">Load Embedding File</span>
</div>
""", unsafe_allow_html=True)

col_src, col_info = st.columns([3, 2])

with col_src:
    _folder_files = list_embedding_folder("embeddings")

    load_mode = st.radio(
        "Source",
        ["📁 From embeddings/ folder", "⬆ Upload from local machine"],
        horizontal=True,
        label_visibility="collapsed",
    )

    uploaded_file   = None   # will hold a file-like or None
    _selected_bytes = None   # raw bytes resolved below

    if load_mode == "📁 From embeddings/ folder":
        if _folder_files:
            chosen = st.selectbox(
                "Select an embedding file",
                options=_folder_files,
                index=0,
                help="Files found in the embeddings/ folder next to app.py",
            )
            chosen_path = pathlib.Path("embeddings") / chosen
            st.markdown(
                f'<div style="font-family:\'DM Mono\',monospace;font-size:0.72rem;'
                f'color:var(--accent-green);margin-top:0.4rem">📦 {chosen_path}</div>',
                unsafe_allow_html=True,
            )
            # Wrap as a file-like object so downstream code works identically
            _selected_bytes = chosen_path.read_bytes()

            class _Fakefile:
                name = chosen
                def read(self): return _selected_bytes

            uploaded_file = _Fakefile()
        else:
            st.markdown(
                '<div style="font-family:\'DM Mono\',monospace;font-size:0.78rem;'
                'color:var(--accent-amber);padding:1rem;background:var(--bg-card);'
                'border:1px solid rgba(255,179,71,0.3);border-radius:6px">'
                '⚠ No .npz files found in <code>embeddings/</code> folder.<br>'
                'Create the folder and place your .npz files there, '
                'or switch to the upload option.</div>',
                unsafe_allow_html=True,
            )
    else:
        uploaded_file = st.file_uploader(
            "Drop your .npz embedding file here",
            type=["npz"],
            help="NumPy compressed archive — CIFAR-10 (10 classes) or CIFAR-100 (100 classes)",
            label_visibility="visible",
        )

with col_info:
    st.markdown("""
    <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:6px;
         padding:1.25rem;margin-top:0.5rem">
        <div style="font-family:'DM Mono',monospace;font-size:0.7rem;text-transform:uppercase;
             letter-spacing:0.1em;color:var(--text-muted);margin-bottom:0.75rem">Expected Format</div>
        <div style="font-family:'DM Mono',monospace;font-size:0.78rem;color:var(--text-secondary);
             line-height:2">
            📦 <span style="color:var(--accent-green)">.npz</span> file<br>
            🔑 Key containing <span style="color:var(--accent-blue)">embed</span>
               or <span style="color:var(--accent-blue)">feat</span><br>
            🔑 Key containing <span style="color:var(--accent-amber)">label</span>
               or <span style="color:var(--accent-amber)">target</span><br>
            📐 Auto-detects <span style="color:var(--accent-green)">CIFAR-10</span>
               or <span style="color:var(--accent-purple)">CIFAR-100</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── File loaded ──────────────────────────────────────────────────────────────
if uploaded_file is not None:
    file_bytes = uploaded_file.read()

    try:
        # ── Step 1: quick load to detect dataset type (integer labels path) ──────
        embeddings, labels, all_keys, dataset_type = load_data(file_bytes)

        unique_labels = np.sort(np.unique(labels))
        leaf_names    = [str(int(l)) for l in unique_labels]
        n_classes     = len(unique_labels)

        dataset_label = "CIFAR-10" if dataset_type == "cifar10" else "CIFAR-100"
        max_rf        = 2 * (n_classes - 1)

        # ── Step 2: resolve taxonomy ──────────────────────────────────────────────
        _custom = st.session_state.get("custom_taxonomy")
        if _custom:
            WORDNET_NEWICK, leaf_names_map = _custom
            _taxonomy_badge = "custom"
        else:
            try:
                WORDNET_NEWICK, leaf_names_map = get_default_taxonomy(dataset_type)
                _taxonomy_badge = "default"
            except FileNotFoundError as _fe:
                st.error(
                    f"⚠ Default taxonomy file not found: {_fe}\n\n"
                    "Place **CIFAR10_WordNet.nwk**, **CIFAR10_classes.json**, "
                    "**CIFAR100_WordNet.nwk**, **CIFAR100_classes.json** "
                    "in the `data/` directory, or use the Custom Upload option in the sidebar."
                )
                st.stop()

        # ── Step 3: if labels were strings, reload with the names_map ────────────
        if not np.issubdtype(labels.dtype, np.integer):
            embeddings, labels, all_keys, dataset_type = load_data(file_bytes, leaf_names_map)

        # Persist dataset_type so sidebar can read it
        st.session_state["dataset_type"] = dataset_type

        # ── Metadata cards ───────────────────────────────────────────────────
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f"""
            <div class="metric-card green">
                <div class="metric-label">Dataset</div>
                <div class="metric-value" style="font-size:1.4rem">{dataset_label}</div>
                <div class="metric-sub">auto-detected</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card blue">
                <div class="metric-label">Total Samples</div>
                <div class="metric-value">{embeddings.shape[0]:,}</div>
                <div class="metric-sub">embedding vectors</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card amber">
                <div class="metric-label">Embedding Dim</div>
                <div class="metric-value">{embeddings.shape[1]:,}</div>
                <div class="metric-sub">feature dimensions</div>
            </div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card purple">
                <div class="metric-label">Classes Found</div>
                <div class="metric-value">{n_classes}</div>
                <div class="metric-sub">unique labels</div>
            </div>""", unsafe_allow_html=True)
        with col5:
            sz_kb  = len(file_bytes) / 1024
            sz_str = f"{sz_kb/1024:.1f} MB" if sz_kb > 1024 else f"{sz_kb:.0f} KB"
            st.markdown(f"""
            <div class="metric-card green">
                <div class="metric-label">File Size</div>
                <div class="metric-value" style="font-size:1.3rem">{sz_str}</div>
                <div class="metric-sub">{uploaded_file.name[:18]}…</div>
            </div>""", unsafe_allow_html=True)

        # ── Class distribution expander ──────────────────────────────────────
        with st.expander("📊 Class Distribution", expanded=False):
            # For CIFAR-100 show in a scrollable grid (10 per row)
            rows_of = 10
            chunks  = [list(unique_labels)[i:i+rows_of]
                       for i in range(0, len(unique_labels), rows_of)]
            for chunk in chunks:
                cols = st.columns(len(chunk))
                for ci, lbl in enumerate(chunk):
                    count = int(np.sum(labels == lbl))
                    name  = leaf_names_map.get(str(int(lbl)), str(lbl))
                    with cols[ci]:
                        st.markdown(f"""
                        <div style="text-align:center;padding:0.4rem 0.2rem;
                             background:var(--bg-secondary);border-radius:4px;
                             border:1px solid var(--border);margin-bottom:0.4rem">
                            <div style="font-family:'DM Mono',monospace;font-size:0.58rem;
                                 color:var(--text-muted);text-transform:uppercase;
                                 white-space:nowrap;overflow:hidden;text-overflow:ellipsis">
                                {name}</div>
                            <div style="font-family:'DM Serif Display',serif;font-size:1.1rem;
                                 color:var(--accent-green)">{count:,}</div>
                        </div>""", unsafe_allow_html=True)

        st.divider()

        # ─── Run Analysis ────────────────────────────────────────────────────
        st.markdown("""
        <div class="section-header">
            <span class="section-num">02 —</span>
            <span class="section-title">Run Analysis</span>
        </div>
        """, unsafe_allow_html=True)

        # Warn CIFAR-100 users about runtime
        if dataset_type == "cifar100":
            st.markdown("""
            <div style="background:rgba(255,179,71,0.08);border:1px solid rgba(255,179,71,0.3);
                 border-radius:6px;padding:0.9rem 1.2rem;margin-bottom:1rem;
                 font-family:'DM Mono',monospace;font-size:0.78rem;color:#ffb347">
                ⚠️ CIFAR-100 has 100 classes — boundary estimation and UPGMA will take
                significantly longer than CIFAR-10. Please be patient.
            </div>
            """, unsafe_allow_html=True)

        run_col, _ = st.columns([1, 3])
        with run_col:
            run_btn = st.button("⚡ Run Full Analysis", use_container_width=True)

        if run_btn or "results" in st.session_state:
            if run_btn:
                # Clear stale results from a previous file
                st.session_state.pop("results", None)

                progress_bar = st.progress(0)
                status_text  = st.empty()
                log_lines    = []
                log_box      = st.empty()

                def update_progress(frac, msg):
                    progress_bar.progress(min(frac, 1.0))
                    status_text.markdown(
                        f'<span style="font-family:\'DM Mono\',monospace;font-size:0.78rem;'
                        f'color:var(--accent-green)">⟳ {msg}</span>',
                        unsafe_allow_html=True)
                    log_lines.append(f"<span class='log-ok'>✓</span> {msg}")
                    if len(log_lines) > 8:
                        log_lines.pop(0)
                    log_box.markdown(
                        f'<div class="log-container">{"<br>".join(log_lines)}</div>',
                        unsafe_allow_html=True)

                # Phase 1
                status_text.markdown(
                    '<span style="font-family:\'DM Mono\',monospace;font-size:0.78rem;'
                    'color:var(--accent-amber)">⟳ Running boundary estimation algorithm…</span>',
                    unsafe_allow_html=True)
                centres = compute_centres(
                    embeddings, labels, unique_labels, METRICS,
                    progress_cb=lambda f, m: update_progress(f * 0.4, m))

                # Phase 2
                status_text.markdown(
                    '<span style="font-family:\'DM Mono\',monospace;font-size:0.78rem;'
                    'color:var(--accent-blue)">⟳ Building UPGMA trees and computing RF distances…</span>',
                    unsafe_allow_html=True)
                trees_6, rf_rows, all_trees = build_all_trees(
                    centres, leaf_names, WORDNET_NEWICK, METRICS, METRIC_LABELS,
                    progress_cb=lambda f, m: update_progress(0.4 + f * 0.5, m))
                
                saved_dir = save_trees_to_disk(trees_6, METRICS, METRIC_LABELS, _DEFAULT_DIR)
                update_progress(0.91, f"Saved 6 Newick trees → {saved_dir}")        
                # ── Run DefIn for all 6 trees ────────────────────────────────────────────
                update_progress(0.93, "Running DefIn deformity analysis…")
                _ref_source = (
                    (_DATA_DIR / f"{'CIFAR10' if n_classes <= 10 else 'CIFAR100'}_WordNet.nwk")
                    if not st.session_state.get("custom_taxonomy")
                    else WORDNET_NEWICK          # raw string from custom upload
                )
                defin_results = run_defin_for_all_trees(
                    saved_dir, _ref_source, METRICS, METRIC_LABELS, _DEFAULT_DIR)
                
                # Phase 3
                status_text.markdown(
                    '<span style="font-family:\'DM Mono\',monospace;font-size:0.78rem;'
                    'color:var(--accent-purple)">⟳ Generating visualizations…</span>',
                    unsafe_allow_html=True)
                update_progress(0.92, "Rendering cladogram figure…")
                clad_bytes = make_cladogram_figure(
                    WORDNET_NEWICK, trees_6, rf_rows, leaf_names_map,
                    METRICS, METRIC_LABELS, n_classes)
                update_progress(0.95, "Rendering dendrogram figure…")
                dend_bytes = make_dendrogram_figure(
                    WORDNET_NEWICK, trees_6, leaf_names_map,
                    METRICS, METRIC_LABELS, n_classes)
                update_progress(0.98, "Rendering RF heatmap…")
                heat_bytes = make_heatmap_figure(rf_rows, METRICS, METRIC_LABELS, n_classes)
                update_progress(1.0, "All done ✓")

                st.session_state["results"] = {
                    "trees_6": trees_6, "rf_rows": rf_rows, "all_trees": all_trees,
                    "clad_bytes": clad_bytes, "dend_bytes": dend_bytes, "heat_bytes": heat_bytes,
                    "leaf_names_map": leaf_names_map,
                    "WORDNET_NEWICK": WORDNET_NEWICK,
                    "n_classes": n_classes,
                    "dataset_label": dataset_label,
                    "defin_results": defin_results,
                }

                progress_bar.empty()
                status_text.empty()
                log_box.empty()
                st.success("✓ Analysis complete — scroll down to explore results")

            # ─── RESULTS ─────────────────────────────────────────────────────
            if "results" in st.session_state:
                res            = st.session_state["results"]
                trees_6        = res["trees_6"]
                rf_rows        = res["rf_rows"]
                all_trees      = res["all_trees"]
                clad_bytes     = res["clad_bytes"]
                dend_bytes     = res["dend_bytes"]
                heat_bytes     = res["heat_bytes"]
                leaf_names_map = res["leaf_names_map"]
                WORDNET_NEWICK = res["WORDNET_NEWICK"]
                n_classes      = res["n_classes"]
                dataset_label  = res["dataset_label"]

                st.divider()

                # ── RF TABLE ──────────────────────────────────────────────────
                st.markdown("""
                <div class="section-header">
                    <span class="section-num">03 —</span>
                    <span class="section-title">Robinson-Foulds Distance Table</span>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(
                    render_rf_table_html(rf_rows, METRICS, METRIC_LABELS, n_classes),
                    unsafe_allow_html=True)

                all_rf_vals = [
                    v for row in rf_rows
                    for k, v in row.items()
                    if isinstance(k, tuple) and isinstance(v, int)
                ]
                if all_rf_vals:
                    best_rf  = min(all_rf_vals)
                    worst_rf = max(all_rf_vals)
                    avg_rf   = np.mean(all_rf_vals)
                    max_rf   = 2 * (n_classes - 1)

                    sc1, sc2, sc3, sc4 = st.columns(4)
                    with sc1:
                        st.markdown(f"""
                        <div class="metric-card green" style="margin-top:1rem">
                            <div class="metric-label">Best RF</div>
                            <div class="metric-value">{best_rf}</div>
                            <div class="metric-sub">most WordNet-aligned</div>
                        </div>""", unsafe_allow_html=True)
                    with sc2:
                        st.markdown(f"""
                        <div class="metric-card amber" style="margin-top:1rem">
                            <div class="metric-label">Mean RF</div>
                            <div class="metric-value">{avg_rf:.1f}</div>
                            <div class="metric-sub">across all configs</div>
                        </div>""", unsafe_allow_html=True)
                    with sc3:
                        st.markdown(f"""
                        <div class="metric-card purple" style="margin-top:1rem">
                            <div class="metric-label">Worst RF</div>
                            <div class="metric-value">{worst_rf}</div>
                            <div class="metric-sub">least aligned</div>
                        </div>""", unsafe_allow_html=True)
                    with sc4:
                        st.markdown(f"""
                        <div class="metric-card blue" style="margin-top:1rem">
                            <div class="metric-label">Max Possible RF</div>
                            <div class="metric-value">{max_rf}</div>
                            <div class="metric-sub">{n_classes} leaves (rooted)</div>
                        </div>""", unsafe_allow_html=True)

                st.divider()

                # ── FIGURES ───────────────────────────────────────────────────
                st.markdown("""
                <div class="section-header">
                    <span class="section-num">04 —</span>
                    <span class="section-title">Tree Visualizations</span>
                </div>
                """, unsafe_allow_html=True)

                tab1, tab2, tab3 = st.tabs([
                    "🌿  Cladograms",
                    "📊  Dendrograms",
                    "🔥  RF Heatmap",
                ])

                with tab1:
                    st.markdown('<div class="img-card">', unsafe_allow_html=True)
                    st.image(clad_bytes)
                    st.markdown(
                        f'<div class="img-caption">Figure 1 — WordNet Reference + 6 UPGMA '
                        f'Cladograms (topology only) — {dataset_label}</div></div>',
                        unsafe_allow_html=True)
                    st.download_button(
                        "⬇ Download Cladogram PNG", clad_bytes,
                        f"cladograms_{dataset_label.lower().replace('-','')}.png",
                        "image/png", use_container_width=False)

                with tab2:
                    st.markdown('<div class="img-card">', unsafe_allow_html=True)
                    st.image(dend_bytes)
                    st.markdown(
                        f'<div class="img-caption">Figure 2 — UPGMA Dendrograms with Branch '
                        f'Heights (D/2) — {dataset_label}</div></div>',
                        unsafe_allow_html=True)
                    st.download_button(
                        "⬇ Download Dendrogram PNG", dend_bytes,
                        f"dendrograms_{dataset_label.lower().replace('-','')}.png",
                        "image/png", use_container_width=False)

                with tab3:
                    st.markdown('<div class="img-card">', unsafe_allow_html=True)
                    st.image(heat_bytes)
                    st.markdown(
                        '<div class="img-caption">Figure 3 — RF Distance Heatmap '
                        '(green=low, red=high)</div></div>',
                        unsafe_allow_html=True)
                    st.download_button(
                        "⬇ Download Heatmap PNG", heat_bytes,
                        f"rf_heatmap_{dataset_label.lower().replace('-','')}.png",
                        "image/png", use_container_width=False)

                st.divider()

                # ── NEWICK STRINGS ────────────────────────────────────────────
                st.markdown("""
                <div class="section-header">
                    <span class="section-num">05 —</span>
                    <span class="section-title">Newick Strings</span>
                </div>
                """, unsafe_allow_html=True)

                newick_blocks = render_newicks(
                    WORDNET_NEWICK, trees_6, rf_rows,
                    METRICS, METRIC_LABELS, leaf_names_map)

                for block in newick_blocks:
                    idx_str = f"[ {block['idx']} ]"
                    color   = block['color']
                    label   = block['label']
                    newick  = block['newick']

                    st.markdown(
                        f'<div class="newick-label" style="margin-top:1.2rem">'
                        f'{idx_str} &nbsp; {label}</div>',
                        unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="newick-block" style="border-left-color:{color}">'
                        f'{newick}</div>',
                        unsafe_allow_html=True)
                    if block['rf'] is not None:
                        rf_line = (f"Leaf order: {block['leaf_order']}"
                                   f"  ·  RF = {block['rf']}")
                        st.markdown(
                            f'<div class="newick-rf">{rf_line}</div>',
                            unsafe_allow_html=True)

                st.divider()
                st.divider()

                # ── DEFORMITY INDEX ───────────────────────────────────────────────────────
                st.markdown("""
                <div class="section-header">
                    <span class="section-num">06 —</span>
                    <span class="section-title">DefIn — Deformity Index</span>
                </div>
                """, unsafe_allow_html=True)

                if not defin_results:
                    st.markdown(
                        '<div style="font-family:\'DM Mono\',monospace;font-size:0.8rem;'
                        'color:var(--text-muted)">DefIn results not available — re-run the analysis.</div>',
                        unsafe_allow_html=True)
                else:
                    # ── Summary cards ─────────────────────────────────────────────────
                    valid = [r for r in defin_results if r["deformity"] is not None]
                    if valid:
                        best  = min(valid, key=lambda r: r["deformity"])
                        worst = max(valid, key=lambda r: r["deformity"])
                        avg   = sum(r["deformity"] for r in valid) / len(valid)

                        dc1, dc2, dc3, dc4 = st.columns(4)
                        with dc1:
                            st.markdown(f"""
                            <div class="metric-card green" style="margin-top:0">
                                <div class="metric-label">Best Deformity</div>
                                <div class="metric-value">{best['deformity']:.4f}</div>
                                <div class="metric-sub">{best['label']}</div>
                            </div>""", unsafe_allow_html=True)
                        with dc2:
                            st.markdown(f"""
                            <div class="metric-card amber" style="margin-top:0">
                                <div class="metric-label">Mean Deformity</div>
                                <div class="metric-value">{avg:.4f}</div>
                                <div class="metric-sub">across {len(valid)} trees</div>
                            </div>""", unsafe_allow_html=True)
                        with dc3:
                            st.markdown(f"""
                            <div class="metric-card purple" style="margin-top:0">
                                <div class="metric-label">Worst Deformity</div>
                                <div class="metric-value">{worst['deformity']:.4f}</div>
                                <div class="metric-sub">{worst['label']}</div>
                            </div>""", unsafe_allow_html=True)
                        with dc4:
                            st.markdown(f"""
                            <div class="metric-card blue" style="margin-top:0">
                                <div class="metric-label">Trees Evaluated</div>
                                <div class="metric-value">{len(valid)}/6</div>
                                <div class="metric-sub">successful runs</div>
                            </div>""", unsafe_allow_html=True)

                    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

                    # ── Per-tree result table ──────────────────────────────────────────
                    table_rows = ""
                    for r in defin_results:
                        gm_key = next(
                            (METRICS[i] for i, lbl in enumerate(METRIC_LABELS)
                            if r["label"].startswith(lbl)), None)
                        row_color = METRIC_COLORS.get(gm_key, "#e8eaf0")

                        if r["error"]:
                            di_cell   = f"<td class='rf-high'>ERROR</td>"
                            cld_cell  = "<td style='color:var(--text-muted)'>—</td>"
                        else:
                            # Color deformity: low=green, mid=amber, high=red  (thresholds are heuristic)
                            di_val = r['deformity']
                            if   di_val < 3.0:  di_cls = "rf-low"
                            elif di_val < 7.0:  di_cls = "rf-mid"
                            else:                di_cls = "rf-high"
                            di_cell  = f"<td class='{di_cls}'>{di_val:.6f}</td>"
                            cld_cell = f"<td style='color:var(--text-secondary)'>{r['clades'] if r['clades'] is not None else '—'}</td>"

                        table_rows += f"""
                        <tr>
                            <td style='color:{row_color};font-weight:600;text-align:left'>{r['label']}</td>
                            <td style='color:var(--text-muted);font-size:0.68rem;text-align:left'>{r['file']}</td>
                            {cld_cell}
                            {di_cell}
                        </tr>"""

                    st.markdown(f"""
                    <div class="rf-table-container">
                        <div class="rf-table-header">
                            DefIn · Deformity Index vs WordNet Reference
                            &nbsp;·&nbsp; Lower = More Faithful Hierarchy
                        </div>
                        <table class="rf-table">
                            <thead>
                                <tr>
                                    <th style='text-align:left'>Tree</th>
                                    <th style='text-align:left'>File</th>
                                    <th>Clades</th>
                                    <th>Deformity Index</th>
                                </tr>
                            </thead>
                            <tbody>{table_rows}</tbody>
                        </table>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Raw output expander ────────────────────────────────────────────
                    with st.expander("🔬 Raw DefIn Output", expanded=False):
                        for r in defin_results:
                            st.markdown(
                                f'<div class="newick-label" style="margin-top:1rem">{r["label"]} '
                                f'— {r["file"]}</div>',
                                unsafe_allow_html=True)
                            content = r["error"] if r["error"] else r["raw"]
                            color   = "var(--accent-red)" if r["error"] else "var(--accent-green)"
                            st.markdown(
                                f'<div class="newick-block" style="border-left-color:{color};'
                                f'color:{color}">{content}</div>',
                                unsafe_allow_html=True)
                # ── FULL RF MATRIX ────────────────────────────────────────────
                with st.expander(
                        "🔬 Full RF Matrix (all group × tree metric combinations)",
                        expanded=False):
                    st.markdown(
                        '<div style="font-family:\'DM Mono\',monospace;font-size:0.72rem;'
                        'color:var(--text-muted);margin-bottom:1rem">'
                        'All 18 combinations: 3 group metrics × 3 tree metrics × 2 centre types (G, MPP)'
                        '</div>',
                        unsafe_allow_html=True)

                    rows_all = []
                    for (gm, tm, ct), nwk in all_trees.items():
                        gml   = METRIC_LABELS[METRICS.index(gm)]
                        tml   = METRIC_LABELS[METRICS.index(tm)]
                        rf, _ = rf_distance(nwk, WORDNET_NEWICK)
                        rows_all.append({
                            "Group Metric": gml, "Tree Metric": tml,
                            "Centre": ct, "RF Distance": rf, "Newick": nwk,
                        })

                    max_rf_full = 2 * (n_classes - 1)
                    table_html  = """
                    <div class="rf-table-container">
                    <table class="rf-table">
                    <thead><tr>
                        <th>Group Metric</th><th>Tree Metric</th>
                        <th>Centre</th><th>RF</th><th>Newick</th>
                    </tr></thead><tbody>"""
                    for r in rows_all:
                        rf_val = r["RF Distance"]
                        low_t  = max_rf_full * 0.25
                        mid_t  = max_rf_full * 0.60
                        cls    = ("rf-low" if rf_val <= low_t
                                  else "rf-mid" if rf_val <= mid_t else "rf-high")
                        gm_key = METRICS[METRIC_LABELS.index(r["Group Metric"])]
                        color  = METRIC_COLORS.get(gm_key, "#fff")
                        nwk_preview = r["Newick"][:60] + "…" if len(r["Newick"]) > 60 else r["Newick"]
                        table_html += (
                            f"<tr>"
                            f"<td style='color:{color};font-weight:600'>{r['Group Metric']}</td>"
                            f"<td>{r['Tree Metric']}</td>"
                            f"<td style='color:var(--accent-green)'>{r['Centre']}</td>"
                            f"<td class='{cls}'>{rf_val}</td>"
                            f"<td style='font-size:0.65rem;color:var(--text-muted)'>{nwk_preview}</td>"
                            f"</tr>")
                    table_html += "</tbody></table></div>"
                    st.markdown(table_html, unsafe_allow_html=True)

    except StopIteration:
        st.error("⚠️ Could not find required keys in the .npz file. "
                 "Expected keys containing 'embed'/'feat' and 'label'/'target'.")
    except Exception as e:
        st.error(f"⚠️ Error loading file: {e}")

else:
    # Clear stale results when file is removed
    st.session_state.pop("results", None)
    st.session_state.pop("dataset_type", None)

    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;background:var(--bg-card);
         border:1px dashed var(--border-accent);border-radius:8px;margin:2rem 0">
        <div style="font-size:3rem;margin-bottom:1rem">🌿</div>
        <div style="font-family:'DM Serif Display',serif;font-size:1.6rem;
             color:var(--text-primary);margin-bottom:0.5rem">Ready to Analyze</div>
        <div style="font-family:'DM Mono',monospace;font-size:0.8rem;
             color:var(--text-secondary);max-width:440px;margin:0 auto;line-height:1.8">
            Upload a <span style="color:var(--accent-green)">CIFAR-10</span> or
            <span style="color:var(--accent-purple)">CIFAR-100</span>
            <span style="color:var(--accent-green)">.npz</span> embedding file above.<br>
            Dataset is auto-detected from the number of classes.
        </div>
        <div style="margin-top:2rem;display:flex;justify-content:center;gap:2rem;flex-wrap:wrap">
            <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:var(--text-muted)">
                <span style="color:var(--accent-green)">01</span> Boundary Estimation
            </div>
            <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:var(--text-muted)">
                <span style="color:var(--accent-blue)">02</span> UPGMA Trees (G & MPP)
            </div>
            <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:var(--text-muted)">
                <span style="color:var(--accent-amber)">03</span> Robinson-Foulds RF
            </div>
            <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:var(--text-muted)">
                <span style="color:var(--accent-purple)">04</span> Visualizations
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:4rem;padding-top:1.5rem;border-top:1px solid var(--border);
     font-family:'DM Mono',monospace;font-size:0.65rem;color:var(--text-muted);
     display:flex;justify-content:space-between;align-items:center">
    <span>Hierarchical Embedding Analyzer · CNN Embedding Analysis Suite</span>
    <span>CIFAR-10 / CIFAR-100 · UPGMA · Robinson-Foulds</span>
</div>
""", unsafe_allow_html=True)