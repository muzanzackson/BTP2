import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from matplotlib.patches import Circle
import os
import warnings
import logging
import zipfile
import io
import tempfile
from pathlib import Path
from scipy.spatial.distance import pdist, squareform, cdist
import pandas as pd
import streamlit as st

warnings.simplefilter("ignore", category=UserWarning)
logging.basicConfig(filename="app.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# ─── Page Config ────────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="CNN Embedding Analysis",
#     page_icon="🧠",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

h1, h2, h3, .stMarkdown h1 {
    font-family: 'Space Mono', monospace !important;
}

.stApp {
    background: #0d0d0d;
    color: #e0e0e0;
}

section[data-testid="stSidebar"] {
    background: #111111;
    border-right: 1px solid #2a2a2a;
}

.log-box {
    background: #111;
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    padding: 12px 16px;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: #7effa0;
    max-height: 280px;
    overflow-y: auto;
    white-space: pre-wrap;
    line-height: 1.6;
}

.log-entry-error   { color: #ff5252; }
.log-entry-success { color: #69ff97; }
.log-entry-info    { color: #40c4ff; }
.log-entry-warning { color: #ffd740; }
.log-entry-debug   { color: #b0bec5; }
.log-entry-deformity { color: #ce93d8; }

.metric-card {
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 16px;
    text-align: center;
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    color: #69ff97;
}

.metric-label {
    font-size: 12px;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

.quality-excellent { color: #69ff97; }
.quality-good      { color: #40c4ff; }
.quality-moderate  { color: #ffd740; }
.quality-poor      { color: #ff7043; }
.quality-verypoor  { color: #ff5252; }

.stButton > button {
    background: #1a1a1a;
    color: #e0e0e0;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    transition: all 0.2s;
}
.stButton > button:hover {
    border-color: #69ff97;
    color: #69ff97;
    background: #111;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stRadio"] label {
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    color: #888 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid #222;
    padding-bottom: 6px;
    margin-bottom: 12px;
}

.stDataFrame { background: #111 !important; }

.stTabs [data-baseweb="tab-list"] {
    background: #111;
    border-bottom: 1px solid #2a2a2a;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: #666;
}
.stTabs [aria-selected="true"] {
    color: #69ff97 !important;
    border-bottom: 2px solid #69ff97 !important;
}

.stProgress > div > div { background: #69ff97; }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ─────────────────────────────────────────────────────────
for key, default in {
    "data": None,
    "labels": None,
    "file_path": None,
    "log_messages": [],
    "distances_per_class": {},
    "max_samples": 1000,
    "results_ready": False,
    "tsne_fig": None,
    "heatmap_fig": None,
    "radius_df": None,
    "deformity_df": None,
    "deformity_avg": None,
    "clustering_metrics": None,
    "centroids_txt": None,
    "distances_csv": None,
    "radius_csv": None,
    "deformity_csv": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

MAX_SAMPLES = 1000

# ─── Logging Helper ─────────────────────────────────────────────────────────────
def log(tag, message):
    st.session_state.log_messages.append((tag, message))
    logging.debug(f"{tag}: {message}")

def clear_log():
    st.session_state.log_messages = []

def render_log():
    lines = []
    for tag, msg in st.session_state.log_messages[-120:]:
        css = {
            "Error": "log-entry-error",
            "Success": "log-entry-success",
            "Info": "log-entry-info",
            "Warning": "log-entry-warning",
            "Debug": "log-entry-debug",
            "DEFORMITY": "log-entry-deformity",
        }.get(tag, "log-entry-info")
        lines.append(f'<span class="{css}">[{tag}] {msg}</span>')
    html = '<div class="log-box">' + "<br>".join(lines) + '</div>'
    st.markdown(html, unsafe_allow_html=True)

# ─── Core Analysis Functions (identical logic to tkinter version) ────────────────

def is_feature_array(arr):
    return arr.ndim == 2 and arr.shape[1] >= 2 and np.issubdtype(arr.dtype, np.number)

def compute_mode_distance(distances):
    if distances.size == 0:
        return 0.0
    hist, bin_edges = np.histogram(distances, bins=20, density=True)
    mode_idx = np.argmax(hist)
    return (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2

def preprocess_1d_array(data):
    n_elements = len(data)
    log("Info", f"Processing 1D array with {n_elements} elements...")
    possible_feature_sizes = [512, 1024, 2048, 4096]
    for n_features in possible_feature_sizes:
        if n_elements % n_features == 0:
            n_samples = n_elements // n_features
            if n_samples > 1 and n_samples <= MAX_SAMPLES * 10:
                features = data.reshape(n_samples, n_features)
                log("Info", f"Reshaped to {n_samples} samples x {n_features} features.")
                n_clusters = min(10, n_samples)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(StandardScaler().fit_transform(features))
                log("Info", f"Generated {n_clusters} pseudo-labels using K-means.")
                return {'features': features, 'labels': labels}
    window_size = 1024
    n_samples = n_elements // window_size
    if n_samples > 1:
        features = data[:n_samples * window_size].reshape(n_samples, window_size)
        log("Info", f"Segmented into {n_samples} samples x {window_size} features.")
        n_clusters = min(10, n_samples)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(StandardScaler().fit_transform(features))
        log("Info", f"Generated {n_clusters} pseudo-labels using K-means.")
        return {'features': features, 'labels': labels}
    log("Error", "Failed to reshape or segment 1D array into features.")
    return None

def compute_max_radius(features_tsne, labels, centroids, unique_labels, distance_metric):
    max_radii = {}
    point_counts = {}
    shifted_centroids = {}
    for label, centroid in centroids:
        own_mask = labels == label
        other_mask = labels != label
        own_points = features_tsne[own_mask]
        other_points = features_tsne[other_mask]
        if own_points.size == 0:
            log("Warning", f"Class {label}: No own points. Setting radius to 0.")
            max_radii[label] = 0.0
            point_counts[label] = (0, 0)
            shifted_centroids[label] = centroid
            continue
        own_distances = cdist(own_points, [centroid], metric=distance_metric).flatten()
        other_distances = cdist(other_points, [centroid], metric=distance_metric).flatten() if other_points.size > 0 else np.array([])
        log("Debug", f"Class {label}: {own_points.shape[0]} own pts, {other_points.shape[0]} other pts")
        own_indices = np.argsort(own_distances)
        sorted_own_distances = own_distances[own_indices]
        last_valid_radius = 0.0
        last_own_count = 0
        last_other_count = 0
        last_valid_centroid = centroid
        for k in range(1, len(sorted_own_distances) + 1):
            radius = sorted_own_distances[k - 1]
            own_count = k
            other_count = np.sum(other_distances <= radius)
            if own_count > other_count:
                last_valid_radius = radius
                last_own_count = own_count
                last_other_count = other_count
                points_within_radius = own_points[own_indices[:k]]
                if points_within_radius.size > 0:
                    last_valid_centroid = np.mean(points_within_radius, axis=0)
                else:
                    last_valid_centroid = centroid
        max_radii[label] = last_valid_radius
        point_counts[label] = (last_own_count, last_other_count)
        shifted_centroids[label] = last_valid_centroid
        log("Info", f"Class {label}: Final radius={last_valid_radius:.4f}, own={last_own_count}, other={last_other_count}")
    return max_radii, point_counts, shifted_centroids

def compute_deformity_index(features_tsne, labels, centroids_dict, max_radii, point_counts, distance_metric):
    deformity_scores = {}
    for label in np.unique(labels):
        mask = labels == label
        points = features_tsne[mask]
        if len(points) < 3:
            deformity_scores[label] = 1.0
            log("DEFORMITY", f"Class {label}: 1.000 (too few points)")
            continue
        centroid = centroids_dict[label]
        distances = cdist(points, [centroid], metric=distance_metric).flatten()
        compactness = np.std(distances) / (np.mean(distances) + 1e-8)
        own_count, other_count = point_counts.get(label, (len(points), 0))
        overlap_penalty = 1.0 - (own_count / (own_count + other_count + 1e-8))
        if len(points) > 1:
            nn_dists = np.sort(cdist(points, points, metric=distance_metric), axis=1)[:, 1]
            sparsity = np.mean(distances) / (np.mean(nn_dists) + 1e-8)
            sparsity_penalty = min(sparsity / 5.0, 1.0)
        else:
            sparsity_penalty = 0.0
        deformity = 0.4 * compactness + 0.4 * overlap_penalty + 0.2 * sparsity_penalty
        deformity = np.clip(deformity, 0.0, 1.0)
        deformity_scores[label] = deformity
        quality = get_quality_label(deformity)
        log("DEFORMITY", f"Class {label}: {deformity:.3f} [{quality}] | C={compactness:.3f}, O={overlap_penalty:.3f}, S={sparsity_penalty:.3f}")
    avg = np.mean(list(deformity_scores.values()))
    overall = get_quality_label(avg)
    log("DEFORMITY", f"OVERALL DEFORMITY INDEX: {avg:.3f} → {overall} cluster quality")
    return deformity_scores, avg

def get_quality_label(score):
    if score < 0.25: return "Excellent"
    if score < 0.45: return "Good"
    if score < 0.65: return "Moderate"
    if score < 0.85: return "Poor"
    return "Very Poor"

def compute_clustering_metrics(features, true_labels, file_path):
    n_samples = features.shape[0]
    n_clusters = min(10, n_samples)
    log("Info", f"Computing clustering metrics for {n_samples} samples...")
    features_scaled = StandardScaler().fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    pred_labels = kmeans.fit_predict(features_scaled)
    if true_labels is None:
        log("Warning", "No true labels. Using K-means pseudo-labels.")
        true_labels = pred_labels
    true_labels = np.array(true_labels, dtype=str)
    pred_labels = np.array(pred_labels, dtype=int)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    unique_true_labels = np.unique(true_labels)
    unique_pred_labels = np.unique(pred_labels)
    contingency_matrix = np.zeros((len(unique_true_labels), len(unique_pred_labels)))
    for i, tl in enumerate(unique_true_labels):
        for j, pl in enumerate(unique_pred_labels):
            contingency_matrix[i, j] = np.sum((true_labels == tl) & (pred_labels == pl))
    purity = np.sum(np.max(contingency_matrix, axis=0)) / n_samples
    rand_index = adjusted_rand_score(true_labels, pred_labels)
    log("Info", f"NMI: {nmi:.4f}, Purity: {purity:.4f}, Rand: {rand_index:.4f}")
    return {'Dataset': os.path.basename(file_path), 'NMI': nmi, 'Purity Index': purity, 'Rand Index': rand_index}

# ─── Main Visualize Function ─────────────────────────────────────────────────────
def visualize_features(features, labels, feature_name, dim, method, distance_metric):
    """Same logic as original; returns figures and data instead of rendering Tkinter widgets."""
    results = {}

    if not is_feature_array(features):
        if features.ndim == 4 and features.shape[0] == 1:
            features = features.reshape(1, -1)
        else:
            log("Error", f"Cannot visualize {feature_name} with shape {features.shape}.")
            return None

    n_samples, n_features = features.shape
    if n_samples > MAX_SAMPLES:
        log("Info", f"Downsampling {feature_name} from {n_samples} to {MAX_SAMPLES} samples.")
        indices = np.random.choice(n_samples, MAX_SAMPLES, replace=False)
        features = features[indices]
        labels = labels[indices] if labels is not None else None
        n_samples = MAX_SAMPLES
    elif n_samples < 2:
        log("Error", f"Single sample; {dim} t-SNE not applicable.")
        return None

    log("Info", f"Normalizing features for {feature_name}...")
    features = StandardScaler().fit_transform(features)

    if n_features > 50:
        n_components = min(n_samples, n_features, 50)
        log("Info", f"Reducing {n_features} features to {n_components} with {method}...")
        try:
            reducer = PCA(n_components=n_components, random_state=42) if method == "PCA" else TruncatedSVD(n_components=n_components, random_state=42)
            features = reducer.fit_transform(features)
        except ValueError as e:
            log("Error", f"{method} failed: {e}")
            return None

    n_components_tsne = 2 if dim == "2D" else 3
    perplexity = min(5, n_samples - 1) if n_samples < 50 else min(30, n_samples - 1)
    log("Info", f"Generating {dim} t-SNE plot (perplexity={perplexity})...")

    try:
        tsne = TSNE(n_components=n_components_tsne, perplexity=perplexity, n_iter=1000, random_state=42, n_jobs=-1)
        features_tsne = tsne.fit_transform(features)
    except (ValueError, MemoryError) as e:
        log("Error", f"{dim} t-SNE failed: {e}")
        return None

    # ── Build scatter plot ──────────────────────────────────────────────────────
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 8), facecolor='#0d0d0d')
    if dim == "2D":
        ax = fig.add_subplot(111, facecolor='#111111')
        x, y = features_tsne[:, 0], features_tsne[:, 1]
    else:
        ax = fig.add_subplot(111, projection='3d', facecolor='#111111')
        x, y, z = features_tsne[:, 0], features_tsne[:, 1], features_tsne[:, 2]

    centroids = []
    shifted_centroids_list = []
    distances_per_class = {}
    table_data = []

    if labels is not None:
        labels = np.array(labels, dtype=str)
        unique_labels = np.unique(labels)
        log("Info", f"Found {len(unique_labels)} unique labels: {unique_labels}")

        if len(unique_labels) <= 40:
            cmap_tab20 = plt.colormaps['tab20']
            cmap_tab20b = plt.colormaps['tab20b']
            colors = [cmap_tab20(i / 20) for i in range(20)] + [cmap_tab20b(i / 20) for i in range(20)]
            colors = colors[:len(unique_labels)]
        else:
            cmap = plt.colormaps['viridis']
            colors = [cmap(i / len(unique_labels)) for i in range(len(unique_labels))]

        for idx, label in enumerate(unique_labels):
            mask = labels == label
            if np.sum(mask) > 0:
                if dim == "2D":
                    ax.scatter(x[mask], y[mask], c=[colors[idx]], marker=".", label=f"Class {label}", alpha=0.5, s=80)
                    centroid = np.mean(features_tsne[mask], axis=0)
                    ax.scatter([centroid[0]], [centroid[1]], c=[colors[idx]], marker='X', s=200, edgecolors='white', zorder=5)
                    for point in features_tsne[mask]:
                        ax.plot([point[0], centroid[0]], [point[1], centroid[1]], c=colors[idx], alpha=0.15, linewidth=0.4)
                else:
                    ax.scatter(x[mask], y[mask], z[mask], c=[colors[idx]], marker=".", label=f"Class {label}", alpha=0.5, s=80)
                    centroid = np.mean(features_tsne[mask], axis=0)
                    ax.scatter([centroid[0]], [centroid[1]], [centroid[2]], c=[colors[idx]], marker='X', s=200, edgecolors='white', zorder=5)
                    for point in features_tsne[mask]:
                        ax.plot([point[0], centroid[0]], [point[1], centroid[1]], [point[2], centroid[2]], c=colors[idx], alpha=0.15, linewidth=0.4)
                centroids.append((label, centroid))
                distances = cdist(features_tsne[mask], [centroid], metric=distance_metric).flatten()
                distances_per_class[label] = distances
                mode_distance = compute_mode_distance(distances)
                table_data.append([f"Class {label}", f"{np.mean(distances):.4f}", f"{mode_distance:.4f}", f"{np.max(distances):.4f}"])

        # Radii + deformity
        max_radii, point_counts, shifted_centroids = compute_max_radius(features_tsne, labels, centroids, unique_labels, distance_metric)
        centroids_dict = dict(centroids)
        deformity_scores, avg_deformity = compute_deformity_index(features_tsne, labels, centroids_dict, max_radii, point_counts, distance_metric)

        # Draw circles / spheres
        for label, centroid in centroids:
            idx = unique_labels.tolist().index(label)
            radius = max_radii.get(label, 0.0)
            shifted_centroid = shifted_centroids.get(label, centroid)
            if radius > 0:
                if dim == "2D":
                    circle = Circle(centroid, radius, color=colors[idx], fill=False, linestyle='--', alpha=0.7, linewidth=1.2)
                    ax.add_patch(circle)
                    ax.scatter([shifted_centroid[0]], [shifted_centroid[1]], c=[colors[idx]], marker='*', s=250, edgecolors='white', zorder=6)
                else:
                    u = np.linspace(0, 2 * np.pi, 20)
                    v = np.linspace(0, np.pi, 20)
                    xs = radius * np.outer(np.cos(u), np.sin(v)) + centroid[0]
                    ys = radius * np.outer(np.sin(u), np.sin(v)) + centroid[1]
                    zs = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + centroid[2]
                    ax.plot_wireframe(xs, ys, zs, color=colors[idx], alpha=0.3, linestyle='--')
                    ax.scatter([shifted_centroid[0]], [shifted_centroid[1]], [shifted_centroid[2]], c=[colors[idx]], marker='*', s=250, edgecolors='white', zorder=6)
            shifted_centroids_list.append((label, shifted_centroid))

        max_legend_classes = 20
        if len(unique_labels) > max_legend_classes:
            handles, lleg = ax.get_legend_handles_labels()
            ax.legend(handles[:max_legend_classes], lleg[:max_legend_classes],
                      title=f"Classes ({len(unique_labels)} total)", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=7, ncol=2,
                      facecolor='#1a1a1a', edgecolor='#333', labelcolor='#ccc')
        else:
            ax.legend(title=f"Classes: {len(unique_labels)}", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=7, ncol=2,
                      facecolor='#1a1a1a', edgecolor='#333', labelcolor='#ccc')
    else:
        if dim == "2D":
            ax.scatter(x, y, c='#69ff97', alpha=0.5, s=80)
        else:
            ax.scatter(x, y, z, c='#69ff97', alpha=0.5, s=80)
        deformity_scores, avg_deformity, max_radii, point_counts, shifted_centroids = {}, None, {}, {}, {}

    ax.set_title(f"{dim} t-SNE: Semantic Relationships ({feature_name})", color='#e0e0e0', fontsize=13, pad=12)
    ax.set_xlabel("t-SNE 1", color='#888', fontsize=10)
    ax.set_ylabel("t-SNE 2", color='#888', fontsize=10)
    if dim == "3D":
        ax.set_zlabel("t-SNE 3", color='#888', fontsize=10)
    ax.tick_params(colors='#666')
    ax.grid(True, alpha=0.15, color='#444')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
    plt.tight_layout()
    fig.subplots_adjust(right=0.75 if dim == "2D" else 0.65)

    results['tsne_fig'] = fig
    results['deformity_scores'] = deformity_scores
    results['avg_deformity'] = avg_deformity
    results['max_radii'] = max_radii
    results['point_counts'] = point_counts
    results['shifted_centroids'] = shifted_centroids
    results['centroids'] = centroids
    results['shifted_centroids_list'] = shifted_centroids_list
    results['distances_per_class'] = distances_per_class
    results['table_data'] = table_data
    results['unique_labels'] = unique_labels if labels is not None else []

    # ── Heatmap ─────────────────────────────────────────────────────────────────
    if len(centroids) > 1:
        centroid_array = np.array([c[1] for c in centroids])
        centroid_labels = [c[0] for c in centroids]
        dist_matrix = squareform(pdist(centroid_array, metric=distance_metric))
        fig_w = min(max(6, len(centroid_labels)), 32)
        fig_h = min(max(5, len(centroid_labels)), 28)
        hfig, hax = plt.subplots(figsize=(fig_w, fig_h), facecolor='#0d0d0d')
        hax.set_facecolor('#111')
        n_classes = len(centroid_labels)
        show_annot = n_classes <= 20          # annotations get unreadable beyond ~20 classes
        tick_labels = centroid_labels if n_classes <= 40 else False   # axis labels unreadable beyond ~40

        sns.heatmap(dist_matrix,
                    annot=show_annot,
                    fmt='.2f' if show_annot else '',
                    xticklabels=tick_labels,
                    yticklabels=tick_labels,
                    cmap='viridis', ax=hax,
                    cbar_kws={'label': f'{distance_metric.capitalize()} Distance'},
                    annot_kws={'size': 8, 'color': 'white'} if show_annot else {})
        hax.set_title(f"Inter-Centroid Distances ({method} {dim})", color='#e0e0e0', fontsize=12)
        hax.tick_params(colors='#aaa', labelsize=8)
        plt.tight_layout()
        results['heatmap_fig'] = hfig
        log("Info", "Inter-centroid distance heatmap generated.")

    log("Success", f"{dim} t-SNE completed for {feature_name}.")
    return results

# ─── Load Functions ──────────────────────────────────────────────────────────────
def process_npz(file_bytes, file_name):
    clear_log()
    log("Info", f"Loading NPZ: {file_name}")
    try:
        buf = io.BytesIO(file_bytes)
        with zipfile.ZipFile(buf, 'r') as zf:
            if zf.testzip() is not None:
                raise zipfile.BadZipFile("NPZ file is corrupted")
        buf.seek(0)
        data = np.load(buf)
        loaded = {}
        labels = data.get('labels', None)
        for key in data:
            if key == 'labels':
                continue
            arr = data[key]
            if is_feature_array(arr):
                loaded[key] = arr
            else:
                log("Warning", f"Skipping {key}: invalid shape {arr.shape} dtype {arr.dtype}")
        if not loaded:
            log("Error", "No valid feature arrays found in .npz file!")
            return
        if 'features' in loaded:
            n_s, n_f = loaded['features'].shape
            if n_f < 2:
                log("Error", f"Too few features ({n_f}) for visualization.")
                return
            if n_s < 2:
                log("Error", f"Too few samples ({n_s}) for visualization.")
                return
            if labels is not None and len(labels) != n_s:
                log("Error", f"Labels shape mismatch: {labels.shape} vs {loaded['features'].shape}")
                labels = None
        st.session_state.data = loaded
        st.session_state.labels = labels
        st.session_state.file_path = file_name
        log("Success", f"Loaded {file_name}: {list(loaded.keys())}")
        for k, arr in loaded.items():
            log("Info", f"  {k}: shape={arr.shape}, dtype={arr.dtype}")
    except Exception as e:
        log("Error", f"Failed to load NPZ: {e}")

def process_npy(file_bytes, file_name):
    clear_log()
    log("Info", f"Loading NPY: {file_name}")
    try:
        buf = io.BytesIO(file_bytes)
        data = np.load(buf, allow_pickle=True)
        if data.ndim == 1:
            processed = preprocess_1d_array(data)
            if processed is None:
                log("Error", "Cannot process 1D array.")
                return
            st.session_state.data = {'features': processed['features']}
            if 'labels' in processed:
                st.session_state.labels = processed['labels']
        elif data.ndim == 4 and data.shape[0] == 1:
            st.session_state.data = {'features': data.reshape(1, -1)}
            label = os.path.splitext(file_name)[0]
            st.session_state.labels = np.array([label])
        else:
            st.session_state.data = {'data': data}
            st.session_state.labels = None
        st.session_state.file_path = file_name
        log("Success", f"Loaded {file_name}")
    except Exception as e:
        log("Error", f"Failed to load NPY: {e}")

def process_npy_folder(uploaded_files):
    clear_log()
    log("Info", f"Loading folder of {len(uploaded_files)} .npy files...")
    features_list = []
    labels_list = []
    expected_shape = None

    for uploaded in uploaded_files:
        fname = uploaded.name
        try:
            buf = io.BytesIO(uploaded.read())
            data = np.load(buf, allow_pickle=True)

            if data.ndim >= 3:
                if data.shape[0] == 1 or data.ndim == 3:
                    flattened = data.reshape(1, -1)
                    if expected_shape is None:
                        expected_shape = flattened.shape[1:]
                    elif flattened.shape[1:] != expected_shape:
                        log("Warning", f"Shape mismatch: {fname} → {flattened.shape}")
                        continue
                    features_list.append(flattened)
                    label = os.path.splitext(fname)[0].split('_')[0]
                    labels_list.append(label)
                    log("Success", f"Loaded {fname} → Class: {label} | Shape: {flattened.shape}")
                else:
                    log("Warning", f"Skipping {fname}: unexpected batch size {data.shape[0]}")
            elif data.ndim == 1:
                processed = preprocess_1d_array(data)
                if processed is None:
                    log("Warning", f"Skipping {fname}: 1D array couldn't be reshaped")
                    continue
                features_list.append(processed['features'])
                if 'labels' in processed:
                    labels_list.extend(processed['labels'])
                else:
                    label = os.path.splitext(fname)[0].split('_')[0]
                    labels_list.extend([label] * processed['features'].shape[0])
                log("Success", f"Processed 1D {fname} → {processed['features'].shape}")
            elif data.ndim == 2:
                features_list.append(data)
                label = os.path.splitext(fname)[0].split('_')[0]
                labels_list.extend([label] * data.shape[0])
                log("Success", f"Loaded pre-flattened {fname} → {data.shape}")
            else:
                log("Warning", f"Skipping {fname}: unsupported shape {data.shape}")
        except Exception as e:
            log("Error", f"Failed to load {fname}: {str(e)}")

    if not features_list:
        log("Error", "No valid data was loaded from any file!")
        return

    try:
        combined = np.vstack(features_list)
        labels_arr = np.array(labels_list, dtype=str)
        n_samples, n_features = combined.shape
        log("Success", f"SUCCESS! Loaded {n_samples} samples × {n_features} features")
        log("Info", f"Unique classes: {len(np.unique(labels_arr))} → {np.unique(labels_arr)}")
        if n_samples < 2 or n_features < 2:
            log("Error", f"Not enough data: {n_samples} samples, {n_features} features")
            return
        st.session_state.data = {'features': combined}
        st.session_state.labels = labels_arr
        st.session_state.file_path = "folder_upload"
    except Exception as e:
        log("Error", f"Failed to combine features: {e}")

# ─── UI ──────────────────────────────────────────────────────────────────────────
# Header
st.markdown("""
<div style="padding:24px 0 8px 0">
  <span style="font-family:'Space Mono',monospace; font-size:22px; color:#e0e0e0; font-weight:700; letter-spacing:2px;">
    🧠 CNN EMBEDDING ANALYSIS
  </span><br>
  <span style="font-family:'IBM Plex Sans',sans-serif; font-size:13px; color:#555; letter-spacing:1px;">
    Semantic Relationship Visualization · t-SNE · Deformity Index · Cluster Metrics
  </span>
</div>
<hr style="border:none; border-top:1px solid #222; margin:8px 0 20px 0">
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-header">⚙ Configuration</div>', unsafe_allow_html=True)

    dim = st.selectbox("Dimension", ["2D", "3D"], index=0)
    method = st.selectbox("Reduction Method", ["PCA", "TruncatedSVD"], index=0)
    distance_metric = st.selectbox("Distance Metric", ["euclidean", "cosine", "cityblock", "canberra"], index=0)

    # Data info
    if st.session_state.data:
        st.markdown('<div class="section-header" style="margin-top:20px">📊 Loaded Data</div>', unsafe_allow_html=True)
        for k, arr in st.session_state.data.items():
            st.markdown(f'<span style="font-family:Space Mono;font-size:11px;color:#69ff97">{k}</span><br>'
                        f'<span style="font-family:Space Mono;font-size:10px;color:#666">{arr.shape} · {arr.dtype}</span>',
                        unsafe_allow_html=True)
        if st.session_state.labels is not None:
            st.markdown(f'<span style="font-family:Space Mono;font-size:11px;color:#ffd740">labels</span><br>'
                        f'<span style="font-family:Space Mono;font-size:10px;color:#666">{st.session_state.labels.shape} · {len(np.unique(st.session_state.labels))} unique</span>',
                        unsafe_allow_html=True)
            
# ─── Main Load Panel ─────────────────────────────────────────────────────────────
EMBEDDINGS_DIR = Path("embeddings")

def get_available_embeddings():
    if not EMBEDDINGS_DIR.exists():
        return []
    return sorted([f.name for f in EMBEDDINGS_DIR.glob("*.np[yz]")])

st.markdown('<div class="section-header">📁 Load Data</div>', unsafe_allow_html=True)

col_src, col_load = st.columns([3, 1])

with col_src:
    load_mode = st.radio(
        "Source",
        ["📂 Pre-loaded Embeddings", "⬆ Upload NPY/NPZ File", "📁 Upload Folder of NPY Files"],
        horizontal=True,
        label_visibility="collapsed",
    )

run_btn = False  # default

if load_mode == "📂 Pre-loaded Embeddings":
    available = get_available_embeddings()
    col_sel, col_btn = st.columns([4, 1])
    with col_sel:
        if available:
            selected_embedding = st.selectbox(
                "Choose embedding file",
                options=["— Select an embedding,from previous downloads —"] + available,
                label_visibility="collapsed",
            )
        else:
            st.warning("No embeddings found in `embeddings/` folder.")
            selected_embedding = None
    with col_btn:
        load_file_btn = st.button("▶  Load", use_container_width=True,
                                  disabled=(not available or selected_embedding == "— Select an embedding —"))
    if load_file_btn and selected_embedding and selected_embedding != "— Select an embedding —":
        fpath = EMBEDDINGS_DIR / selected_embedding
        file_bytes = fpath.read_bytes()
        if selected_embedding.endswith(".npz"):
            process_npz(file_bytes, selected_embedding)
        else:
            process_npy(file_bytes, selected_embedding)
        st.session_state.results_ready = False

elif load_mode == "⬆ Upload NPY/NPZ File":
    col_up, col_btn = st.columns([4, 1])
    with col_up:
        uploaded_file = st.file_uploader("Upload NPY or NPZ", type=["npy", "npz"], label_visibility="collapsed")
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)  # vertical align
        load_file_btn = st.button("▶  Load", use_container_width=True, disabled=(uploaded_file is None))
    if load_file_btn and uploaded_file:
        file_bytes = uploaded_file.read()
        if uploaded_file.name.endswith(".npz"):
            process_npz(file_bytes, uploaded_file.name)
        else:
            process_npy(file_bytes, uploaded_file.name)
        st.session_state.results_ready = False

else:  # Folder upload
    col_up, col_btn = st.columns([4, 1])
    with col_up:
        uploaded_files = st.file_uploader(
            "Upload NPY files", type=["npy"], accept_multiple_files=True, label_visibility="collapsed"
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        load_file_btn = st.button("▶  Load Folder", use_container_width=True, disabled=(not uploaded_files))
    if load_file_btn and uploaded_files:
        process_npy_folder(uploaded_files)
        st.session_state.results_ready = False

st.markdown("<hr style='border:none;border-top:1px solid #222;margin:16px 0'>", unsafe_allow_html=True)

# Run button — full width, prominent
run_btn = st.button(
    "🚀  Run Visualization",
    use_container_width=True,
    disabled=(st.session_state.data is None),
    type="primary",
)

st.markdown("<hr style='border:none;border-top:1px solid #222;margin:16px 0'>", unsafe_allow_html=True)
# ─── Run Analysis ────────────────────────────────────────────────────────────────
if run_btn and st.session_state.data is not None:
    with st.spinner("Running t-SNE analysis… this may take a moment."):
        # Reset
        st.session_state.tsne_fig = None
        st.session_state.heatmap_fig = None
        st.session_state.radius_df = None
        st.session_state.deformity_df = None
        st.session_state.deformity_avg = None
        st.session_state.clustering_metrics = None
        st.session_state.distances_per_class = {}

        for key, arr in st.session_state.data.items():
            if is_feature_array(arr):
                results = visualize_features(
                    arr,
                    labels=st.session_state.labels,
                    feature_name=key,
                    dim=dim,
                    method=method,
                    distance_metric=distance_metric,
                )
                if results:
                    st.session_state.tsne_fig = results.get('tsne_fig')
                    st.session_state.heatmap_fig = results.get('heatmap_fig')
                    st.session_state.distances_per_class = results.get('distances_per_class', {})

                    # Build radius DataFrame
                    radii = results.get('max_radii', {})
                    pc = results.get('point_counts', {})
                    sc = results.get('shifted_centroids', {})
                    tdlist = results.get('table_data', [])
                    radius_rows = []
                    for i, lbl in enumerate(sorted(radii.keys())):
                        own_c, other_c = pc.get(lbl, (0, 0))
                        mean_d, mode_d, max_d = (tdlist[i][1], tdlist[i][2], tdlist[i][3]) if i < len(tdlist) else ('—', '—', '—')
                        sc_pt = sc.get(lbl, [0, 0])
                        centroid_str = f"({sc_pt[0]:.4f}, {sc_pt[1]:.4f}" + (f", {sc_pt[2]:.4f})" if len(sc_pt) == 3 else ")")
                        radius_rows.append({'Class': f'Class {lbl}', 'Max Radius': f"{radii[lbl]:.4f}",
                                            'Own Points': own_c, 'Other Points': other_c,
                                            'Mean Dist': mean_d, 'Mode Dist': mode_d, 'Max Dist': max_d,
                                            'Shifted Centroid': centroid_str})
                    if radius_rows:
                        st.session_state.radius_df = pd.DataFrame(radius_rows)

                    # Build deformity DataFrame
                    d_scores = results.get('deformity_scores', {})
                    st.session_state.deformity_avg = results.get('avg_deformity')
                    def_rows = [{'Class': f'Class {lbl}', 'Deformity Index': f"{sc:.4f}", 'Quality': get_quality_label(sc)}
                                for lbl, sc in sorted(d_scores.items(), key=lambda x: x[1], reverse=True)]
                    if def_rows:
                        st.session_state.deformity_df = pd.DataFrame(def_rows)

                    # Clustering metrics
                    cm = compute_clustering_metrics(arr, st.session_state.labels, st.session_state.file_path or "upload")
                    st.session_state.clustering_metrics = cm

                    # Prepare export CSVs in memory
                    # Distances CSV
                    dist_buf = io.StringIO()
                    dist_buf.write("Class,Mean Distance,Mode Distance,Max Distance,Individual Distances\n")
                    for lbl, dists in sorted(st.session_state.distances_per_class.items()):
                        mode_d = compute_mode_distance(dists)
                        dist_buf.write(f'"Class {lbl}",{np.mean(dists):.4f},{mode_d:.4f},{np.max(dists):.4f},'
                                       f'"{";".join([f"{d:.4f}" for d in dists])}"\n')
                    st.session_state.distances_csv = dist_buf.getvalue()

                    # Radius CSV
                    if st.session_state.radius_df is not None:
                        st.session_state.radius_csv = st.session_state.radius_df.to_csv(index=False)

                    # Deformity CSV
                    if st.session_state.deformity_df is not None:
                        d_buf = io.StringIO()
                        d_buf.write("Class,Deformity_Index,Quality\n")
                        for lbl, sc in sorted(d_scores.items(), key=lambda x: x[1], reverse=True):
                            d_buf.write(f"{lbl},{sc:.4f},{get_quality_label(sc)}\n")
                        if st.session_state.deformity_avg is not None:
                            d_buf.write(f"\nAverage,{st.session_state.deformity_avg:.4f},Overall\n")
                        st.session_state.deformity_csv = d_buf.getvalue()

                    # Centroids TXT
                    c_buf = io.StringIO()
                    centroids = results.get('centroids', [])
                    sc_list = results.get('shifted_centroids_list', [])
                    c_buf.write(f"Class Centroids in {dim} t-SNE Space:\n")
                    for lbl, c in centroids:
                        c_buf.write(f"Class {lbl}: ({c[0]:.4f}, {c[1]:.4f}" + (f", {c[2]:.4f})\n" if len(c) == 3 else ")\n"))
                    c_buf.write(f"\nShifted Centroids:\n")
                    for lbl, c in sc_list:
                        c_buf.write(f"Class {lbl}: ({c[0]:.4f}, {c[1]:.4f}" + (f", {c[2]:.4f})\n" if len(c) == 3 else ")\n"))
                    st.session_state.centroids_txt = c_buf.getvalue()
            else:
                log("Info", f"Skipping {key}: not a valid feature array")

    st.session_state.results_ready = True

# ─── Results Tabs ─────────────────────────────────────────────────────────────────
if st.session_state.results_ready and st.session_state.tsne_fig is not None:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈  t-SNE Plot",
        "🌡  Centroid Heatmap",
        "📐  Radius Analysis",
        "🔬  Deformity Index",
        "📊  Cluster Metrics",
    ])

    with tab1:
        st.pyplot(st.session_state.tsne_fig, use_container_width=True)
        # Save to buffer for download
        buf = io.BytesIO()
        st.session_state.tsne_fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
        buf.seek(0)
        st.download_button("⬇  Download t-SNE Plot (PNG)", buf, file_name="tsne_plot.png", mime="image/png")

    with tab2:
        if st.session_state.heatmap_fig:
            st.pyplot(st.session_state.heatmap_fig, use_container_width=True)
            buf = io.BytesIO()
            st.session_state.heatmap_fig.savefig(buf, format='png', dpi=72, bbox_inches='tight', facecolor='#0d0d0d')
            buf.seek(0)
            st.download_button("⬇  Download Heatmap (PNG)", buf, file_name="centroid_heatmap.png", mime="image/png")
        else:
            st.info("Heatmap requires at least 2 classes.")

    with tab3:
        if st.session_state.radius_df is not None:
            st.markdown('<div class="section-header">Centroid Radius Analysis</div>', unsafe_allow_html=True)
            st.dataframe(st.session_state.radius_df, use_container_width=True)
            if st.session_state.radius_csv:
                st.download_button("⬇  Download Radius CSV", st.session_state.radius_csv, file_name="radius_analysis.csv", mime="text/csv")
            if st.session_state.centroids_txt:
                st.download_button("⬇  Download Centroids TXT", st.session_state.centroids_txt, file_name="centroids.txt", mime="text/plain")
            if st.session_state.distances_csv:
                st.download_button("⬇  Download Distances CSV", st.session_state.distances_csv, file_name="distances.csv", mime="text/csv")
        else:
            st.info("No radius data available.")

    with tab4:
        if st.session_state.deformity_df is not None and st.session_state.deformity_avg is not None:
            avg = st.session_state.deformity_avg
            quality = get_quality_label(avg)
            q_color = {"Excellent": "#69ff97", "Good": "#40c4ff", "Moderate": "#ffd740", "Poor": "#ff7043", "Very Poor": "#ff5252"}.get(quality, "#888")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{avg:.4f}</div><div class="metric-label">Overall Deformity Index</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{q_color}">{quality}</div><div class="metric-label">Cluster Quality</div></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{len(st.session_state.deformity_df)}</div><div class="metric-label">Classes Analyzed</div></div>', unsafe_allow_html=True)
            st.markdown("")
            st.markdown('<div class="section-header">Per-Class Deformity Report</div>', unsafe_allow_html=True)
            st.dataframe(st.session_state.deformity_df, use_container_width=True)
            if st.session_state.deformity_csv:
                st.download_button("⬇  Download Deformity CSV", st.session_state.deformity_csv, file_name="deformity_index.csv", mime="text/csv")
        else:
            st.info("No deformity data available.")

    with tab5:
        if st.session_state.clustering_metrics:
            m = st.session_state.clustering_metrics
            c1, c2, c3 = st.columns(3)
            for col, key, label in [(c1, 'NMI', 'Normalized Mutual Info'), (c2, 'Purity Index', 'Purity Index'), (c3, 'Rand Index', 'Adjusted Rand Index')]:
                with col:
                    st.markdown(f'<div class="metric-card"><div class="metric-value">{m[key]:.4f}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)
            st.markdown("")
            metrics_df = pd.DataFrame([{'Dataset': m['Dataset'], 'NMI': f"{m['NMI']:.4f}",
                                         'Purity Index': f"{m['Purity Index']:.4f}", 'Rand Index': f"{m['Rand Index']:.4f}"}])
            st.dataframe(metrics_df, use_container_width=True)
            metrics_csv = metrics_df.to_csv(index=False)
            st.download_button("⬇  Download Metrics CSV", metrics_csv, file_name="clustering_metrics.csv", mime="text/csv")
        else:
            st.info("No clustering metrics available.")

elif st.session_state.data is None:
    st.markdown("""
    <div style="text-align:center; padding:60px 0; color:#333">
      <div style="font-family:'Space Mono',monospace; font-size:40px; margin-bottom:16px">⬆</div>
      <div style="font-family:'Space Mono',monospace; font-size:14px; color:#444; letter-spacing:2px">SELECT AN EMBEDDING OR UPLOAD A FILE ABOVE</div>
      <div style="font-size:12px; color:#333; margin-top:8px">Supports .npy and .npz files · Folder batch loading</div>
    </div>
    """, unsafe_allow_html=True)

# ─── Log Panel ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header" style="margin-top:24px">📋 Activity Log</div>', unsafe_allow_html=True)
if st.session_state.log_messages:
    render_log()
else:
    st.markdown('<div class="log-box" style="color:#333">[Waiting for activity...]</div>', unsafe_allow_html=True)
