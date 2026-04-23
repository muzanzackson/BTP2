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
    with open(nwk_path, "r") as f:
        content = f.read().strip()
    return Phylo.read(StringIO(content), "newick")


def load_labels(json_path: str) -> dict:
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
    yield clade
    for child in clade.clades:
        yield from get_all_clades(child)


def assign_3d_coords(clade, depth=0, angle_h=0, angle_v=0,
                     spread_h=2 * math.pi, spread_v=math.pi,
                     layout="radial"):
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
    else:
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
    stops = [
        (55, 138, 221),
        (29, 158, 117),
        (186, 117, 23),
        (211, 90, 48),
        (164, 83, 126),
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
# Serialize tree to JSON for in-browser use
# ---------------------------------------------------------------------------

def assign_unique_ids(clade, counter=None):
    if counter is None:
        counter = [0]
    # Assign a guaranteed unique ID to every node for the frontend
    clade._uid = f"node_{counter[0]}"
    counter[0] += 1
    for child in clade.clades:
        assign_unique_ids(child, counter)

def clade_to_dict(clade, id_to_name: dict) -> dict:
    """Recursively convert a clade to a plain dict for JSON export."""
    raw_name = clade.name or ""
    display_name = id_to_name.get(raw_name, raw_name)
    return {
        "id": clade._uid,
        "name": display_name,
        "branch_length": clade.branch_length if clade.branch_length else 1.0,
        "x": round(clade._x, 6),
        "y": round(clade._y, 6),
        "z": round(clade._z, 6),
        "depth": round(clade._depth, 6),
        "is_leaf": len(clade.clades) == 0,
        "children": [clade_to_dict(c, id_to_name) for c in clade.clades],
    }


# ---------------------------------------------------------------------------
# Build HTML page with embedded controls
# ---------------------------------------------------------------------------

def build_html(tree, id_to_name: dict, layout: str = "radial") -> str:
    assign_unique_ids(tree.root)
    assign_3d_coords(tree.root, layout=layout)
    tree_dict = clade_to_dict(tree.root, id_to_name)
    tree_json = json.dumps(tree_dict, separators=(",", ":"))

    # collect all leaf names + ids for the autocomplete
    leaves = []
    for c in get_all_clades(tree.root):
        if len(c.clades) == 0:
            raw = c.name or ""
            display = id_to_name.get(raw, raw)
            if display or raw:
                leaves.append({"id": c._uid, "name": display if display else raw})
    leaves_json = json.dumps(leaves, separators=(",", ":"))

    # collect ALL node names (internal + leaf) for root picker
    all_nodes = []
    for c in get_all_clades(tree.root):
        raw = c.name or ""
        display = id_to_name.get(raw, raw)
        label = display if display else f"[internal:{raw}]" if raw else "[root]"
        all_nodes.append({"id": c._uid, "name": label, "is_leaf": len(c.clades) == 0})
    nodes_json = json.dumps(all_nodes, separators=(",", ":"))

    layout_title = layout.capitalize()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>3D Phylogenetic Tree Viewer</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  :root {{
    --bg: #0d0f14;
    --panel: #131720;
    --panel2: #1a1f2e;
    --border: #252d3d;
    --accent: #4f9cf9;
    --accent2: #34d399;
    --accent3: #f97316;
    --text: #e2e8f0;
    --muted: #64748b;
    --danger: #f87171;
    --radius: 10px;
    --font-mono: 'Space Mono', monospace;
    --font-body: 'DM Sans', sans-serif;
  }}

  html, body {{ height: 100%; background: var(--bg); color: var(--text); font-family: var(--font-body); }}

  /* ---- Layout ---- */
  #app {{ display: flex; height: 100vh; overflow: hidden; }}

  #sidebar {{
    width: 320px; min-width: 280px; max-width: 380px;
    background: var(--panel);
    border-right: 1px solid var(--border);
    display: flex; flex-direction: column;
    overflow: hidden;
  }}

  #main {{ flex: 1; display: flex; flex-direction: column; overflow: hidden; }}

  /* ---- Header ---- */
  .header {{
    padding: 18px 20px 14px;
    border-bottom: 1px solid var(--border);
    background: linear-gradient(135deg, #0f1623 0%, #131c2e 100%);
  }}
  .header h1 {{
    font-family: var(--font-mono);
    font-size: 13px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 3px;
  }}
  .header p {{
    font-size: 11px;
    color: var(--muted);
    font-weight: 300;
  }}

  /* ---- Panels ---- */
  .panel-body {{ flex: 1; overflow-y: auto; padding: 14px; display: flex; flex-direction: column; gap: 14px; }}
  .panel-body::-webkit-scrollbar {{ width: 4px; }}
  .panel-body::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 4px; }}

  .card {{
    background: var(--panel2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px;
  }}
  .card-title {{
    font-family: var(--font-mono);
    font-size: 10px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 10px;
    display: flex; align-items: center; gap: 6px;
  }}
  .card-title .dot {{
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--accent);
    display: inline-block;
  }}

  /* ---- Inputs ---- */
  .input-row {{ display: flex; gap: 8px; align-items: flex-start; }}
  .input-wrap {{ position: relative; flex: 1; }}

  input[type=text] {{
    width: 100%;
    background: #0d0f14;
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    font-family: var(--font-body);
    font-size: 13px;
    padding: 8px 10px;
    outline: none;
    transition: border-color 0.2s;
  }}
  input[type=text]:focus {{ border-color: var(--accent); }}
  input[type=text]::placeholder {{ color: var(--muted); }}

  .dropdown-list {{
    position: absolute; top: calc(100% + 4px); left: 0; right: 0;
    background: #1e2535;
    border: 1px solid var(--border);
    border-radius: 6px;
    max-height: 180px; overflow-y: auto;
    z-index: 999;
    display: none;
  }}
  .dropdown-list.open {{ display: block; }}
  .dropdown-list::-webkit-scrollbar {{ width: 3px; }}
  .dropdown-list::-webkit-scrollbar-thumb {{ background: var(--border); }}
  .dropdown-item {{
    padding: 7px 10px;
    font-size: 12px;
    cursor: pointer;
    display: flex; align-items: center; gap: 6px;
    transition: background 0.15s;
  }}
  .dropdown-item:hover {{ background: #252d3d; }}
  .dropdown-item .badge {{
    font-size: 9px; font-family: var(--font-mono);
    padding: 1px 5px; border-radius: 3px;
    background: #252d3d; color: var(--muted);
    flex-shrink: 0;
  }}
  .dropdown-item.leaf .badge {{ background: #1a3a2a; color: var(--accent2); }}

  /* ---- Buttons ---- */
  .btn {{
    font-family: var(--font-body);
    font-size: 12px; font-weight: 600;
    padding: 8px 14px;
    border-radius: 6px;
    border: none; cursor: pointer;
    transition: opacity 0.15s, transform 0.1s;
    white-space: nowrap;
    display: inline-flex; align-items: center; gap: 5px;
  }}
  .btn:hover {{ opacity: 0.85; }}
  .btn:active {{ transform: scale(0.97); }}
  .btn-primary {{ background: var(--accent); color: #000; }}
  .btn-success {{ background: var(--accent2); color: #000; }}
  .btn-ghost {{
    background: transparent; color: var(--muted);
    border: 1px solid var(--border);
  }}
  .btn-ghost:hover {{ color: var(--text); border-color: var(--muted); }}
  .btn-danger-ghost {{
    background: transparent; color: var(--danger);
    border: 1px solid #3d1f1f; font-size: 11px;
    padding: 5px 10px;
  }}
  .btn-danger-ghost:hover {{ background: #1f1010; }}
  .btn-full {{ width: 100%; justify-content: center; }}
  .btn-sm {{ padding: 5px 10px; font-size: 11px; }}

  /* ---- Tag list ---- */
  .tag-list {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }}
  .tag {{
    background: #1a2235;
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3px 10px 3px 10px;
    font-size: 11px;
    display: inline-flex; align-items: center; gap: 5px;
    color: var(--text);
  }}
  .tag .tag-x {{
    color: var(--muted); cursor: pointer; font-size: 13px; line-height: 1;
    transition: color 0.15s;
  }}
  .tag .tag-x:hover {{ color: var(--danger); }}

  /* ---- Status bar ---- */
  #statusbar {{
    padding: 8px 16px;
    background: var(--panel);
    border-top: 1px solid var(--border);
    font-size: 11px; font-family: var(--font-mono);
    color: var(--muted);
    display: flex; align-items: center; gap: 10px;
  }}
  #statusbar .sep {{ opacity: 0.3; }}
  #statusbar .highlight {{ color: var(--accent2); }}

  /* ---- Plot area ---- */
  #plot {{ flex: 1; }}

  /* ---- Top bar ---- */
  #topbar {{
    padding: 8px 16px;
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 10px;
    font-size: 12px;
  }}
  #topbar .crumb {{
    color: var(--muted); font-family: var(--font-mono); font-size: 11px;
  }}
  #topbar .crumb span {{ color: var(--accent); }}

  .divider {{ height: 1px; background: var(--border); margin: 2px 0; }}

  /* ---- Reset notice ---- */
  .notice {{
    font-size: 11px; color: var(--muted);
    padding: 6px 8px;
    background: #0f1420;
    border: 1px solid var(--border);
    border-radius: 6px;
    margin-top: 6px;
  }}
  .notice b {{ color: var(--text); }}

  /* ---- Click hint ---- */
  .click-hint {{
    font-size: 11px;
    color: var(--muted);
    padding: 8px 10px;
    background: #0f1420;
    border: 1px solid var(--border);
    border-radius: 6px;
    display: flex;
    align-items: flex-start;
    gap: 6px;
    line-height: 1.5;
  }}
  .click-hint .icon {{
    color: var(--accent);
    font-size: 14px;
    flex-shrink: 0;
    margin-top: 1px;
  }}

  /* ---- Animations ---- */
  @keyframes fadeIn {{ from {{ opacity:0; transform: translateY(4px); }} to {{ opacity:1; transform:none; }} }}
  .fade-in {{ animation: fadeIn 0.25s ease forwards; }}

  .empty-hint {{
    color: var(--muted); font-size: 11px; font-style: italic;
    text-align: center; padding: 10px 0;
  }}
</style>
</head>
<body>
<div id="app">

  <div id="sidebar">
    <div class="header">
      <h1>&#x1F9EC; PhyloViz 3D</h1>
      <p>Interactive phylogenetic tree explorer &mdash; {layout_title} layout</p>
    </div>

    <div class="panel-body">

      <div class="click-hint">
        <span class="icon">&#x1F5B1;</span>
        <span><b style="color:var(--text)">Click any node</b> in the tree to view its subtree. All other nodes will be hidden. Use <b style="color:var(--text)">&#x21BA; Full Tree</b> to reset.</span>
      </div>

      <div class="card">
        <div class="card-title" style="--accent:#34d399;">
          <span class="dot" style="background:var(--accent2);"></span>Find Least Common Ancestor
        </div>
        <p style="font-size:11px;color:var(--muted);margin-bottom:10px;">
          Add 2 or more leaf nodes. The subtree from their LCA will be shown.
        </p>
        <div class="input-row">
          <div class="input-wrap">
            <input type="text" id="lcaInput" placeholder="Add a leaf node&hellip;" autocomplete="off"/>
            <div class="dropdown-list" id="lcaDropdown"></div>
          </div>
          <button class="btn btn-ghost btn-sm" onclick="addLcaNode()">+ Add</button>
        </div>
        <div class="tag-list" id="lcaTags">
          <span class="empty-hint" id="lcaEmptyHint">No nodes added yet.</span>
        </div>
        <div style="display:flex;gap:8px;margin-top:10px;">
          <button class="btn btn-success btn-full" onclick="applyLCA()">&#x2605; Show LCA Subtree</button>
          <button class="btn btn-danger-ghost btn-sm" onclick="clearLcaNodes()">Clear</button>
        </div>
        <div class="notice" id="lcaNotice" style="display:none"></div>
      </div>

    </div></div><div id="main">
    <div id="topbar">
      <span class="crumb" id="breadcrumb">Full tree &mdash; <span id="nodeCount">0</span> nodes</span>
      <div style="flex:1"></div>
      <button class="btn btn-ghost btn-sm" onclick="resetToFullTree()">&#x21BA; Full Tree</button>
    </div>
    <div id="plot"></div>
    <div id="statusbar">
      <span id="sb-nodes" class="highlight">&#x25CB; &mdash;</span>
      <span class="sep">|</span>
      <span id="sb-leaves">Leaves: &mdash;</span>
      <span class="sep">|</span>
      <span id="sb-depth">Max depth: &mdash;</span>
      <span class="sep">|</span>
      <span id="sb-layout">Layout: {layout_title}</span>
    </div>
  </div>

</div><script>
// ============================================================
// DATA
// ============================================================
const FULL_TREE  = {tree_json};
const ALL_NODES  = {nodes_json};
const ALL_LEAVES = {leaves_json};
const LAYOUT_MODE = "{layout}";

// ============================================================
// Tree utilities
// ============================================================

function getAllClades(node) {{
  const result = [node];
  for (const c of (node.children || [])) result.push(...getAllClades(c));
  return result;
}}

function getLeaves(node) {{
  if (!node.children || node.children.length === 0) return [node];
  return node.children.flatMap(getLeaves);
}}

/** Find a clade by id (BFS). */
function findById(root, id) {{
  const q = [root];
  while (q.length) {{
    const n = q.shift();
    if (n.id === id) return n;
    for (const c of (n.children || [])) q.push(c);
  }}
  return null;
}}

/** Find a clade by display name (BFS, case-insensitive). */
function findByName(root, name) {{
  const q = [root];
  const nl = name.toLowerCase();
  while (q.length) {{
    const n = q.shift();
    if (n.name && n.name.toLowerCase() === nl) return n;
    for (const c of (n.children || [])) q.push(c);
  }}
  return null;
}}

/** Find a clade by id OR name. */
function findNode(root, query) {{
  return findById(root, query) || findByName(root, query);
}}

/**
 * Find the path (list of nodes) from root to a target node.
 * Returns null if not found.
 */
function pathToNode(root, targetId) {{
  if (root.id === targetId) return [root];
  for (const c of (root.children || [])) {{
    const sub = pathToNode(c, targetId);
    if (sub) return [root, ...sub];
  }}
  return null;
}}

/**
 * Compute the Least Common Ancestor of a list of node IDs.
 */
function computeLCA(root, ids) {{
  const paths = ids.map(id => {{
    const path = pathToNode(root, id);
    return path ? path.map(n => n.id) : null;
  }});
  if (paths.some(p => p === null)) return null;

  const minLen = Math.min(...paths.map(p => p.length));
  let lcaId = paths[0][0];
  for (let depth = 0; depth < minLen; depth++) {{
    const candidate = paths[0][depth];
    if (paths.every(p => p[depth] === candidate)) {{
      lcaId = candidate;
    }} else {{
      break;
    }}
  }}
  return findById(root, lcaId);
}}

// ============================================================
// Coordinate re-assignment (client-side)
// ============================================================

function assignCoords(node, depth, angleH, angleV, spreadH, spreadV) {{
  const r = depth;
  if (LAYOUT_MODE === "spiral") {{
    node.x = r * Math.cos(angleH) * Math.cos(angleV);
    node.y = r * Math.sin(angleH) * Math.cos(angleV) + depth * 0.3;
    node.z = r * Math.sin(angleV) + depth * 0.2;
  }} else if (LAYOUT_MODE === "cone") {{
    const cr = depth * 0.4;
    node.x = cr * Math.cos(angleH);
    node.y = cr * Math.sin(angleH);
    node.z = depth;
  }} else {{
    node.x = r * Math.cos(angleH) * Math.cos(angleV);
    node.y = r * Math.sin(angleH) * Math.cos(angleV);
    node.z = r * Math.sin(angleV);
  }}
  node.depth = depth;

  const children = node.children || [];
  if (!children.length) return;
  const n = children.length;
  const dH = spreadH / Math.max(n, 1);
  const dV = spreadV / Math.max(n, 1);
  children.forEach((child, i) => {{
    const cH = angleH - spreadH / 2 + dH / 2 + i * dH;
    const cV = angleV - spreadV / 2 + dV / 2 + i * dV;
    const bl = child.branch_length || 1.0;
    assignCoords(child, depth + bl, cH, cV, dH * 0.85, dV * 0.85);
  }});
}}

function recomputeSubtree(subtreeRoot) {{
  // deep clone to avoid mutating original
  const clone = JSON.parse(JSON.stringify(subtreeRoot));
  assignCoords(clone, 0, 0, 0, 2 * Math.PI, Math.PI);
  return clone;
}}

// ============================================================
// Plotly rendering
// ============================================================

function depthColor(depth, maxDepth) {{
  const stops = [
    [55, 138, 221],
    [29, 158, 117],
    [186, 117, 23],
    [211, 90, 48],
    [164, 83, 126],
  ];
  let t = maxDepth > 0 ? depth / maxDepth : 0;
  t = Math.max(0, Math.min(1, t));
  const idx = Math.min(Math.floor(t * (stops.length - 1)), stops.length - 2);
  const f = t * (stops.length - 1) - idx;
  const c1 = stops[idx], c2 = stops[idx + 1];
  const r = Math.round(c1[0] + (c2[0] - c1[0]) * f);
  const g = Math.round(c1[1] + (c2[1] - c1[1]) * f);
  const b = Math.round(c1[2] + (c2[2] - c1[2]) * f);
  return `rgb(${{r}},${{g}},${{b}})`;
}}

function renderTree(subtreeRoot, label) {{
  const allNodes = getAllClades(subtreeRoot);
  const maxDepth = Math.max(...allNodes.map(n => n.depth), 1);

  // edges
  const ex = [], ey = [], ez = [];
  allNodes.forEach(n => {{
    (n.children || []).forEach(c => {{
      ex.push(n.x, c.x, null);
      ey.push(n.y, c.y, null);
      ez.push(n.z, c.z, null);
    }});
  }});

  const edgeTrace = {{
    type: "scatter3d", mode: "lines",
    x: ex, y: ey, z: ez,
    line: {{ color: "#334155", width: 2 }},
    hoverinfo: "none", name: "branches"
  }};

  // nodes 
  const nx = [], ny = [], nz = [];
  const colors = [], sizes = [];
  const texts = [], hovers = [];
  const customIds = [];

  allNodes.forEach(n => {{
    customIds.push(n.id);
    nx.push(n.x); ny.push(n.y); nz.push(n.z);
    const isLeaf = !n.children || n.children.length === 0;
    texts.push(isLeaf ? (n.name || "") : "");
    const bl = n.branch_length != null ? n.branch_length.toFixed(4) : "N/A";
    hovers.push(
      `<b>${{n.name || "internal"}}</b><br>` +
      `Depth: ${{n.depth.toFixed(3)}}<br>` +
      `Branch length: ${{bl}}<br>` +
      (isLeaf ? "🍃 Leaf node — click to zoom" : `🔀 Children: ${{n.children.length}} — click to view subtree`)
    );
    colors.push(depthColor(n.depth, maxDepth));
    sizes.push(isLeaf ? 9 : 6);
  }});

  const nodeTrace = {{
    type: "scatter3d", mode: "markers+text",
    x: nx, y: ny, z: nz,
    customdata: customIds,
    text: texts,
    textposition: "top center",
    textfont: {{ size: 10, color: "#94a3b8" }},
    marker: {{ size: sizes, color: colors, opacity: 0.93, line: {{ color: "#0f172a", width: 1 }} }},
    hovertemplate: hovers,
    name: "nodes"
  }};

  const leafCount = allNodes.filter(n => !n.children || n.children.length === 0).length;

  Plotly.react("plot", [edgeTrace, nodeTrace], {{
    paper_bgcolor: "#0d0f14",
    plot_bgcolor: "#0d0f14",
    margin: {{ l: 0, r: 0, t: 0, b: 0 }},
    scene: {{
      xaxis: {{ visible: false }},
      yaxis: {{ visible: false }},
      zaxis: {{ visible: false }},
      bgcolor: "#0d0f14",
      camera: {{ eye: {{ x: 1.6, y: 1.6, z: 1.2 }} }}
    }},
    showlegend: false,
    annotations: [{{
      text: "Click node to zoom in &nbsp;|&nbsp; Drag to rotate &nbsp;|&nbsp; Scroll to zoom &nbsp;|&nbsp; Shift+drag to pan",
      xref: "paper", yref: "paper",
      x: 0.5, y: 0.01,
      showarrow: false,
      font: {{ size: 11, color: "#334155" }},
      xanchor: "center"
    }}]
  }}, {{ responsive: true }});

  // Update status bar
  document.getElementById("nodeCount").textContent = allNodes.length;
  document.getElementById("breadcrumb").innerHTML = label;
  document.getElementById("sb-nodes").textContent = "⬤ " + allNodes.length + " nodes";
  document.getElementById("sb-leaves").textContent = "Leaves: " + leafCount;
  document.getElementById("sb-depth").textContent = "Max depth: " + maxDepth.toFixed(3);

  // FIX: Clear existing listeners and re-attach safely
  const plotEl = document.getElementById("plot");
  if (plotEl.removeAllListeners) {{
    plotEl.removeAllListeners('plotly_click');
  }}
  attachClickHandler();
}}

// ============================================================
// Click-to-subtree handler
// ============================================================

function attachClickHandler() {{
  const plotEl = document.getElementById("plot");
  plotEl.on("plotly_click", function(data) {{
    if (!data || !data.points || data.points.length === 0) return;
    
    const pt = data.points[0];

    // Ensure we clicked a valid node by checking if our custom ID exists
    if (!pt.customdata) return;
    const clickedId = pt.customdata;

    // Find the node in the FULL tree
    const found = findById(FULL_TREE, clickedId);
    if (!found) return;

    // FIX: Defer the WebGL re-render using setTimeout 
    // to prevent the Plotly event loop from crashing/freezing.
    setTimeout(() => {{
      const sub = recomputeSubtree(found);
      const allN = getAllClades(sub);
      const nodeLabel = found.name || "internal node";
      
      renderTree(sub, `Subtree: <span>${{nodeLabel}}</span> &mdash; ${{allN.length}} node(s)`);
      document.getElementById("lcaNotice").style.display = "none";
    }}, 10);
  }});
}}

// ============================================================
// Actions
// ============================================================

function resetToFullTree() {{
  const root = recomputeSubtree(FULL_TREE);
  renderTree(root, `Full tree &mdash; <span>${{getAllClades(root).length}}</span> nodes`);
  document.getElementById("lcaNotice").style.display = "none";
}}

// LCA state
let lcaSelectedIds = [];

function addLcaNode() {{
  const query = document.getElementById("lcaInput").value.trim();
  if (!query) return;
  const match = ALL_LEAVES.find(l =>
    l.name.toLowerCase() === query.toLowerCase() || l.id === query
  );
  if (!match) {{ showNotice("lcaNotice", `⚠ Leaf not found: "${{query}}"`, "warn"); return; }}
  if (lcaSelectedIds.includes(match.id)) {{ showNotice("lcaNotice", "Already added.", "warn"); return; }}
  lcaSelectedIds.push(match.id);
  document.getElementById("lcaInput").value = "";
  document.getElementById("lcaDropdown").classList.remove("open");
  renderLcaTags();
  document.getElementById("lcaNotice").style.display = "none";
}}

function removeLcaNode(id) {{
  lcaSelectedIds = lcaSelectedIds.filter(x => x !== id);
  renderLcaTags();
}}

function clearLcaNodes() {{
  lcaSelectedIds = [];
  renderLcaTags();
  document.getElementById("lcaNotice").style.display = "none";
}}

function renderLcaTags() {{
  const container = document.getElementById("lcaTags");
  container.innerHTML = "";
  if (lcaSelectedIds.length === 0) {{
    container.appendChild(Object.assign(document.createElement("span"), {{
      className: "empty-hint", textContent: "No nodes added yet."
    }}));
    return;
  }}
  lcaSelectedIds.forEach(id => {{
    const leaf = ALL_LEAVES.find(l => l.id === id);
    const label = leaf ? leaf.name : id;
    const tag = document.createElement("span");
    tag.className = "tag fade-in";
    tag.innerHTML = `${{label}} <span class="tag-x" onclick="removeLcaNode('${{id}}')">&#x00D7;</span>`;
    container.appendChild(tag);
  }});
}}

function applyLCA() {{
  if (lcaSelectedIds.length < 2) {{
    showNotice("lcaNotice", "⚠ Add at least 2 leaf nodes.", "warn"); return;
  }}
  const lcaNode = computeLCA(FULL_TREE, lcaSelectedIds);
  if (!lcaNode) {{
    showNotice("lcaNotice", "⚠ Could not find LCA. Some IDs may be invalid.", "warn"); return;
  }}
  const sub = recomputeSubtree(lcaNode);
  const allN = getAllClades(sub);
  const lcaLabel = lcaNode.name || "internal";
  renderTree(sub,
    `LCA subtree: <span>${{lcaLabel}}</span> &mdash; ${{allN.length}} nodes`);
  showNotice("lcaNotice",
    `✓ LCA found. Showing ${{allN.length}} nodes covering ${{lcaSelectedIds.length}} selected leaves.`,
    "ok");
}}

function showNotice(id, html, type) {{
  const el = document.getElementById(id);
  el.innerHTML = html;
  el.style.display = "block";
  el.style.color = type === "ok" ? "var(--accent2)" : "var(--danger)";
  el.style.borderColor = type === "ok" ? "#1a3a2a" : "#3d1f1f";
}}

// ============================================================
// Autocomplete
// ============================================================

function setupAutocomplete(inputId, dropdownId, dataList, onSelect) {{
  const input = document.getElementById(inputId);
  const dropdown = document.getElementById(dropdownId);

  input.addEventListener("input", () => {{
    const q = input.value.toLowerCase().trim();
    dropdown.innerHTML = "";
    if (!q) {{ dropdown.classList.remove("open"); return; }}
    const matches = dataList.filter(item =>
      (item.name && item.name.toLowerCase().includes(q)) ||
      (item.id && item.id.toLowerCase().includes(q))
    ).slice(0, 20);
    if (!matches.length) {{ dropdown.classList.remove("open"); return; }}
    matches.forEach(item => {{
      const div = document.createElement("div");
      div.className = "dropdown-item" + (item.is_leaf !== false ? " leaf" : "");
      div.innerHTML = `<span style="flex:1">${{item.name || item.id}}</span>` +
        `<span class="badge">${{item.is_leaf !== false ? "leaf" : "internal"}}</span>`;
      div.addEventListener("mousedown", (e) => {{
        e.preventDefault();
        input.value = item.name || item.id;
        dropdown.classList.remove("open");
        onSelect(item);
      }});
      dropdown.appendChild(div);
    }});
    dropdown.classList.add("open");
  }});

  input.addEventListener("focus", () => {{
    if (input.value) input.dispatchEvent(new Event("input"));
  }});

  document.addEventListener("click", (e) => {{
    if (!input.contains(e.target) && !dropdown.contains(e.target))
      dropdown.classList.remove("open");
  }});

  input.addEventListener("keydown", (e) => {{
    if (e.key === "Enter") onSelect({{ name: input.value, id: input.value }});
  }});
}}

// ============================================================
// Init
// ============================================================
window.addEventListener("DOMContentLoaded", () => {{
  resetToFullTree();

  // LCA input — leaves only
  setupAutocomplete("lcaInput", "lcaDropdown", ALL_LEAVES.map(l => ({{...l, is_leaf: true}})), (item) => {{
    document.getElementById("lcaInput").value = item.name || item.id;
  }});
}});
</script>
</body>
</html>"""

    return html


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

    print(f"Loading tree from: {args.nwk}")
    if not Path(args.nwk).exists():
        print(f"ERROR: File not found: {args.nwk}")
        sys.exit(1)
    tree = load_tree(args.nwk)
    print(f"  Loaded. Terminals: {tree.count_terminals()}, "
          f"Total clades: {sum(1 for _ in tree.find_clades())}")

    id_to_name = {}
    if args.labels:
        print(f"Loading labels from: {args.labels}")
        if not Path(args.labels).exists():
            print(f"ERROR: Label file not found: {args.labels}")
            sys.exit(1)
        id_to_name = load_labels(args.labels)
        print(f"  Loaded {len(id_to_name)} label mappings.")
    else:
        print("  No label file provided — using raw node names.")

    print(f"Building HTML (layout: {args.layout}) ...")
    html = build_html(tree, id_to_name, layout=args.layout)

    out_path = args.output
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nSaved interactive HTML to: {out_path}")

    if not args.no_open:
        print("Opening in browser ...")
        webbrowser.open(Path(out_path).resolve().as_uri())

    print("Done!")


if __name__ == "__main__":
    main()