"""
3D Interactive Phylogenetic Tree Visualizer
============================================
Usage:
    python visualize_tree.py --nwk tree.nwk --labels labels.json
    python visualize_tree.py --nwk tree.nwk --labels labels.json --layout radial
    python visualize_tree.py --nwk tree.nwk --labels labels.json --layout cone --output my_tree.html

Arguments:
    --nwk       Path to your .nwk (Newick) file
    --labels    Path to JSON file mapping class names to node IDs
                Format: {"classname": ["id"], ...}
    --layout    Layout style: radial (default), spiral, cone
    --output    Output HTML filename (default: tree_visualization.html)
    --no-open   Don't auto-open the browser after rendering
"""

import argparse
import json
import math
import sys
import webbrowser
from io import StringIO
from pathlib import Path

try:
    from Bio import Phylo
except ImportError:
    print("ERROR: biopython not installed. Run: pip install biopython")
    sys.exit(1)

try:
    import plotly.graph_objects as go
except ImportError:
    print("ERROR: plotly not installed. Run: pip install plotly")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Tree parsing & coordinate assignment
# ---------------------------------------------------------------------------

def load_tree(nwk_path: str):
    """Load a Newick tree from a .nwk file."""
    with open(nwk_path, "r") as f:
        content = f.read().strip()
    return Phylo.read(StringIO(content), "newick")


def load_labels(json_path: str) -> dict:
    """
    Load label mapping from JSON.
    Supports two formats:
      { "classname": ["id"] }   →  {"0": "airplane", ...}
      { "classname": "id" }     →  {"0": "airplane", ...}
    Returns a dict mapping node_id_string → class_name.
    """
    with open(json_path, "r") as f:
        raw = json.load(f)

    id_to_name = {}
    for class_name, ids in raw.items():
        if isinstance(ids, list):
            for i in ids:
                id_to_name[str(i)] = class_name
        else:
            id_to_name[str(ids)] = class_name
    return id_to_name


def get_all_clades(clade):
    """Recursively collect all clades in the tree."""
    yield clade
    for child in clade.clades:
        yield from get_all_clades(child)


def assign_3d_coords(clade, depth=0, angle_h=0, angle_v=0,
                     spread_h=2 * math.pi, spread_v=math.pi,
                     layout="radial"):
    """
    Recursively assign (x, y, z) coordinates to each clade.
    Supports three layout modes: radial, spiral, cone.
    """
    r = depth
    if layout == "spiral":
        x = r * math.cos(angle_h) * math.cos(angle_v)
        y = r * math.sin(angle_h) * math.cos(angle_v) + depth * 0.3
        z = r * math.sin(angle_v) + depth * 0.2
    elif layout == "cone":
        cone_r = depth * 0.4
        x = cone_r * math.cos(angle_h)
        y = cone_r * math.sin(angle_h)
        z = depth
    else:  # radial (default)
        x = r * math.cos(angle_h) * math.cos(angle_v)
        y = r * math.sin(angle_h) * math.cos(angle_v)
        z = r * math.sin(angle_v)

    clade._x = x
    clade._y = y
    clade._z = z
    clade._depth = depth

    children = clade.clades
    if not children:
        return

    n = len(children)
    d_h = spread_h / max(n, 1)
    d_v = spread_v / max(n, 1)
    for i, child in enumerate(children):
        child_angle_h = angle_h - spread_h / 2 + d_h / 2 + i * d_h
        child_angle_v = angle_v - spread_v / 2 + d_v / 2 + i * d_v
        branch_len = child.branch_length if child.branch_length else 1.0
        assign_3d_coords(
            child,
            depth=depth + branch_len,
            angle_h=child_angle_h,
            angle_v=child_angle_v,
            spread_h=d_h * 0.85,
            spread_v=d_v * 0.85,
            layout=layout
        )


def depth_color(depth, max_depth):
    """Map a depth value to an RGB color string (blue → teal → amber → coral)."""
    stops = [
        (55, 138, 221),   # blue
        (29, 158, 117),   # teal
        (186, 117, 23),   # amber
        (211, 90, 48),    # coral
        (164, 83, 126),   # pink
    ]
    t = depth / max_depth if max_depth > 0 else 0
    t = max(0.0, min(1.0, t))
    idx = min(int(t * (len(stops) - 1)), len(stops) - 2)
    f = t * (len(stops) - 1) - idx
    c1, c2 = stops[idx], stops[idx + 1]
    r = int(c1[0] + (c2[0] - c1[0]) * f)
    g = int(c1[1] + (c2[1] - c1[1]) * f)
    b = int(c1[2] + (c2[2] - c1[2]) * f)
    return f"rgb({r},{g},{b})"


# ---------------------------------------------------------------------------
# Build Plotly figure
# ---------------------------------------------------------------------------

def build_figure(tree, id_to_name: dict, layout: str = "radial") -> go.Figure:
    """Build the 3D Plotly figure from a parsed + coordinate-assigned tree."""

    all_clades = list(get_all_clades(tree.root))
    max_depth = max(c._depth for c in all_clades) or 1.0

    # ---- edges ----
    ex, ey, ez = [], [], []
    for clade in all_clades:
        for child in clade.clades:
            ex += [clade._x, child._x, None]
            ey += [clade._y, child._y, None]
            ez += [clade._z, child._z, None]

    edge_trace = go.Scatter3d(
        x=ex, y=ey, z=ez,
        mode="lines",
        line=dict(color="#9999aa", width=2),
        hoverinfo="none",
        name="branches"
    )

    # ---- nodes ----
    nx, ny, nz = [], [], []
    node_colors, node_sizes = [], []
    hover_texts, display_labels = [], []

    for clade in all_clades:
        nx.append(clade._x)
        ny.append(clade._y)
        nz.append(clade._z)

        raw_name = clade.name or ""
        is_leaf = len(clade.clades) == 0

        # Resolve display name: use label map if available
        display_name = id_to_name.get(raw_name, raw_name)
        display_labels.append(display_name if is_leaf else "")

        bl = f"{clade.branch_length:.4f}" if clade.branch_length else "N/A"
        hover = (
            f"<b>{display_name or 'internal'}</b><br>"
            f"ID: {raw_name or '—'}<br>"
            f"Depth: {clade._depth:.3f}<br>"
            f"Branch length: {bl}<br>"
            + ("Leaf node" if is_leaf else f"Children: {len(clade.clades)}")
        )
        hover_texts.append(hover)
        node_colors.append(depth_color(clade._depth, max_depth))
        node_sizes.append(9 if is_leaf else 6)

    node_trace = go.Scatter3d(
        x=nx, y=ny, z=nz,
        mode="markers+text",
        text=display_labels,
        textposition="top center",
        textfont=dict(size=11, color="#555555"),
        marker=dict(
            size=node_sizes,
            color=node_colors,
            opacity=0.92,
            line=dict(color="white", width=1)
        ),
        hovertemplate=hover_texts,
        name="nodes"
    )

    # ---- layout ----
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=dict(
            text=f"3D Phylogenetic Tree — {layout.capitalize()} layout",
            font=dict(size=15),
            x=0.5
        ),
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=50, b=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="white",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.2))
        ),
        showlegend=False,
        annotations=[
            dict(
                text=(
                    "🖱 Drag to rotate  |  Scroll to zoom  |  "
                    "Shift+drag to pan  |  Hover nodes for details"
                ),
                xref="paper", yref="paper",
                x=0.5, y=0.01,
                showarrow=False,
                font=dict(size=11, color="#888888"),
                xanchor="center"
            )
        ]
    )
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="3D Interactive Phylogenetic Tree Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--nwk", required=True, help="Path to .nwk Newick file")
    parser.add_argument("--labels", default=None,
                        help="Path to JSON label mapping file (optional)")
    parser.add_argument("--layout", default="radial",
                        choices=["radial", "spiral", "cone"],
                        help="3D layout style (default: radial)")
    parser.add_argument("--output", default="tree_visualization.html",
                        help="Output HTML file (default: tree_visualization.html)")
    parser.add_argument("--no-open", action="store_true",
                        help="Don't auto-open the browser")
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Load tree ---
    print(f"Loading tree from: {args.nwk}")
    if not Path(args.nwk).exists():
        print(f"ERROR: File not found: {args.nwk}")
        sys.exit(1)
    tree = load_tree(args.nwk)
    print(f"  Loaded. Terminals: {tree.count_terminals()}, "
          f"Total clades: {sum(1 for _ in tree.find_clades())}")

    # --- Load labels ---
    id_to_name = {}
    if args.labels:
        print(f"Loading labels from: {args.labels}")
        if not Path(args.labels).exists():
            print(f"ERROR: Label file not found: {args.labels}")
            sys.exit(1)
        id_to_name = load_labels(args.labels)
        print(f"  Loaded {len(id_to_name)} label mappings: {id_to_name}")
    else:
        print("  No label file provided — using raw node names.")

    # --- Assign 3D coordinates ---
    print(f"Assigning 3D coordinates (layout: {args.layout}) ...")
    assign_3d_coords(tree.root, layout=args.layout)

    # --- Build figure ---
    print("Building Plotly figure ...")
    fig = build_figure(tree, id_to_name, layout=args.layout)

    # --- Save & open ---
    out_path = args.output
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"\nSaved interactive HTML to: {out_path}")

    if not args.no_open:
        print("Opening in browser ...")
        webbrowser.open(Path(out_path).resolve().as_uri())

    print("Done!")


if __name__ == "__main__":
    main()