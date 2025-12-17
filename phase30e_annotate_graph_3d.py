#!/usr/bin/env python3
# phase30e_annotate_graph_3d.py
#
# Step 3A readability pass:
# - Load 3D node coords from phase30d_nodes_3d.csv
# - Load edges from phase30c_edges.csv
# - OPTIONAL: load visit counts (supports BOTH)
#     A) long: columns like node,count (or name,visits, etc)
#     B) wide: first col w_novelty, remaining cols are node names with counts per w
# - Render:
#   * anchors vs discovered nodes
#   * node size scaled by total visits
#   * edges optionally scaled by 'count' if present

import os, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def infer_edge_cols(df):
    if {"src","dst"}.issubset(df.columns):
        return "src","dst"
    if {"u","v"}.issubset(df.columns):
        return "u","v"
    if {"from","to"}.issubset(df.columns):
        return "from","to"
    raise ValueError(f"edges csv missing src/dst-like columns. got={list(df.columns)}")

def load_visits(vis_path, known_nodes):
    """
    Returns dict node->total_visits.
    Supports:
      - long schema: node/name + count/visits/n/total
      - wide schema: w_novelty + node columns (counts per w); totals are column sums
    """
    vcount = {n: 0.0 for n in known_nodes}
    if not vis_path:
        return vcount

    vdf = pd.read_csv(vis_path)
    cols = list(vdf.columns)

    # --- WIDE schema ---
    # Example: ['w_novelty', 'nodeA', 'nodeB', ...]
    if "w_novelty" in cols and any(c in known_nodes for c in cols):
        for n in known_nodes:
            if n in vdf.columns:
                # sum over all w rows
                try:
                    vcount[n] = float(pd.to_numeric(vdf[n], errors="coerce").fillna(0).sum())
                except Exception:
                    vcount[n] = 0.0
        return vcount

    # --- LONG schema ---
    node_col = None
    for c in ["node", "name"]:
        if c in cols:
            node_col = c
            break
    if node_col is None:
        raise ValueError(f"visits csv needs node/name (long) or w_novelty (wide). got={cols}")

    count_col = None
    for c in ["count", "visits", "n", "N", "total"]:
        if c in cols:
            count_col = c
            break
    if count_col is None:
        # fallback: second column
        if len(cols) >= 2:
            count_col = cols[1]
        else:
            raise ValueError("visits csv (long): cannot infer count column.")

    for _, r in vdf.iterrows():
        n = str(r[node_col])
        if n in vcount:
            try:
                vcount[n] = float(r[count_col])
            except Exception:
                pass
    return vcount


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="transport cache (for anchor list)")
    ap.add_argument("--nodes3d", required=True, help="phase30d_nodes_3d.csv")
    ap.add_argument("--edges", required=True, help="phase30c_edges.csv")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--visits", default="", help="optional visits csv (wide or long)")
    ap.add_argument("--title", default="Phase30E: Annotated 3D Graph")
    ap.add_argument("--label_anchors", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    cache = load_json(args.cache)
    anchors = set(cache.get("anchors", {}).keys())

    nodes = pd.read_csv(args.nodes3d)
    edges = pd.read_csv(args.edges)
    s_col, t_col = infer_edge_cols(edges)

    # coord map
    nodes["node"] = nodes["node"].astype(str)
    coord = {r["node"]:(float(r["x"]), float(r["y"]), float(r["z"])) for _, r in nodes.iterrows()}
    known_nodes = list(coord.keys())

    # visit counts (optional; wide or long)
    vcount = load_visits(args.visits, known_nodes)

    is_anchor = np.array([n in anchors for n in known_nodes], dtype=bool)
    counts = np.array([vcount[n] for n in known_nodes], dtype=float)

    # sizes: base + scaled
    if counts.max() > 0:
        sizes = 25 + 220 * (counts / counts.max())
    else:
        sizes = np.full(len(known_nodes), 45.0)

    X = nodes["x"].to_numpy()
    Y = nodes["y"].to_numpy()
    Z = nodes["z"].to_numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # edges
    edge_count_col = "count" if "count" in edges.columns else None
    ecounts = edges[edge_count_col].to_numpy() if edge_count_col else None
    emax = float(ecounts.max()) if ecounts is not None and len(ecounts) else 0.0

    drawn = 0
    for i, r in edges.iterrows():
        s = str(r[s_col]); t = str(r[t_col])
        if s not in coord or t not in coord:
            continue

        xs = [coord[s][0], coord[t][0]]
        ys = [coord[s][1], coord[t][1]]
        zs = [coord[s][2], coord[t][2]]

        if ecounts is not None and emax > 0:
            w = float(r[edge_count_col]) / emax
            lw = 0.6 + 2.6 * w
            al = 0.05 + 0.55 * w
        else:
            lw = 1.0
            al = 0.20

        ax.plot(xs, ys, zs, linewidth=lw, alpha=al)
        drawn += 1

    # nodes: discovered then anchors
    ax.scatter(X[~is_anchor], Y[~is_anchor], Z[~is_anchor], s=sizes[~is_anchor], alpha=0.85, label="discovered")
    ax.scatter(X[is_anchor],  Y[is_anchor],  Z[is_anchor],  s=sizes[is_anchor]*1.15, alpha=0.95, label="anchors")

    if args.label_anchors:
        for n in known_nodes:
            if n in anchors:
                x,y,z = coord[n]
                ax.text(x, y, z, n, fontsize=8)

    ax.set_title(f"{args.title}\n(nodes={len(known_nodes)} edges_drawn={drawn})")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.legend(loc="best")

    out_png = os.path.join(args.outdir, "phase30e_graph_3d_annotated.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"[ok] wrote: {out_png}")

if __name__ == "__main__":
    main()
