#!/usr/bin/env python3
# phase30g_flowfield_streamlines_3d.py
#
# Step 3C:
# Build a smooth 3D "flow field" from a directed edge graph, then integrate
# streamlines to visualize curvature/attractors in the relation manifold.
#
# Inputs:
#   --nodes3d : CSV with columns [node,x,y,z]  (from phase30d)
#   --edges   : CSV with columns [src,dst] (or [u,v]) plus optional weight columns
#   --cache   : transport_cache.json (only used to label anchors if requested)
#
# Outputs:
#   - phase30g_flowfield.png
#   - phase30g_streamlines.csv

import os, json, argparse, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def norm(v, eps=1e-9):
    n = float(np.linalg.norm(v))
    return v / (n + eps), n

def infer_edge_cols(edf):
    if {"src","dst"}.issubset(edf.columns):
        return "src", "dst"
    if {"u","v"}.issubset(edf.columns):
        return "u", "v"
    raise ValueError(f"edges csv missing src/dst-like columns. got={list(edf.columns)}")

def infer_weight_col(edf):
    # best-effort: use a sensible weight if present; else 1.0
    for c in ["count", "w", "weight", "n", "freq"]:
        if c in edf.columns:
            return c
    return None

def build_node_flow(nodes, edges, weight_col=None):
    """
    nodes: dict node -> pos (3,)
    edges: dataframe with src,dst
    returns: dict node -> flow vec (3,) in 3D, normalized (but magnitude preserved in separate map)
    """
    flow_sum = {n: np.zeros(3, dtype=np.float32) for n in nodes.keys()}
    mag_sum  = {n: 0.0 for n in nodes.keys()}

    for _, r in edges.iterrows():
        s = str(r["src"]); t = str(r["dst"])
        if s not in nodes or t not in nodes or s == t:
            continue
        w = float(r[weight_col]) if (weight_col and weight_col in r and pd.notna(r[weight_col])) else 1.0
        d = nodes[t] - nodes[s]
        # accumulate raw direction; weight by w
        flow_sum[s] += w * d
        mag_sum[s] += abs(w)

    flow = {}
    flow_mag = {}
    for n in nodes.keys():
        v = flow_sum[n]
        v_unit, vlen = norm(v)
        flow[n] = v_unit.astype(np.float32)
        flow_mag[n] = float(vlen)

    return flow, flow_mag

def knn_field(points, flows, k=6, eps=1e-9):
    """
    points: (N,3) array of node positions
    flows:  (N,3) array of per-node unit flow vectors (or 0)
    returns: function f(x)->(3,) continuous field via inverse-distance weights
    """
    N = points.shape[0]

    def f(x):
        # distances to nodes
        d = np.linalg.norm(points - x[None, :], axis=1)
        idx = np.argsort(d)[:min(k, N)]
        di = d[idx]
        wi = 1.0 / (di + eps)
        wsum = wi.sum() + eps
        v = (flows[idx] * wi[:, None]).sum(axis=0) / wsum
        v_unit, _ = norm(v)
        return v_unit.astype(np.float32)

    return f

def integrate_streamline(f, x0, step=0.06, n_steps=60, stop_radius=0.01):
    pts = [x0.astype(np.float32)]
    x = x0.astype(np.float32)

    for _ in range(n_steps):
        v = f(x)
        if not np.all(np.isfinite(v)):
            break
        if float(np.linalg.norm(v)) < stop_radius:
            break
        x = x + step * v
        pts.append(x.astype(np.float32))

    return np.stack(pts, axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--nodes3d", required=True)
    ap.add_argument("--edges", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--title", default="30g: flow field + streamlines (3D)")
    ap.add_argument("--label_anchors", action="store_true")
    ap.add_argument("--k", type=int, default=6, help="kNN for field interpolation")
    ap.add_argument("--n_stream", type=int, default=20, help="# streamlines from random seeds (in addition to anchors)")
    ap.add_argument("--steps", type=int, default=70, help="integration steps")
    ap.add_argument("--step_size", type=float, default=0.06, help="integration step size in PCA space units")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    cache = load_json(args.cache)
    anchors = set(cache.get("anchors", {}).keys())

    ndf = pd.read_csv(args.nodes3d)
    edf = pd.read_csv(args.edges)

    # normalize expected columns
    if "node" not in ndf.columns:
        raise ValueError(f"nodes3d csv needs 'node' column. got={list(ndf.columns)}")
    for c in ["x","y","z"]:
        if c not in ndf.columns:
            raise ValueError(f"nodes3d csv needs columns x,y,z. got={list(ndf.columns)}")

    s_col, t_col = infer_edge_cols(edf)
    edf = edf.rename(columns={s_col: "src", t_col: "dst"})

    wcol = infer_weight_col(edf)

    nodes = {str(r["node"]): np.array([float(r["x"]), float(r["y"]), float(r["z"])], dtype=np.float32)
             for _, r in ndf.iterrows()}
    node_list = list(nodes.keys())
    P = np.stack([nodes[n] for n in node_list], axis=0)

    # compute per-node flow (unit direction)
    flow_dir, flow_mag = build_node_flow(nodes, edf, weight_col=wcol)
    F = np.stack([flow_dir[n] for n in node_list], axis=0)

    # build continuous field
    field = knn_field(P, F, k=args.k)

    # choose seeds: all anchors present + random nodes
    anchors_present = [n for n in node_list if n in anchors]
    random_nodes = [n for n in node_list if n not in anchors_present]
    random.shuffle(random_nodes)
    seeds = anchors_present + random_nodes[:max(0, args.n_stream)]

    # integrate
    all_lines = []
    for n in seeds:
        line = integrate_streamline(field, nodes[n], step=args.step_size, n_steps=args.steps)
        all_lines.append((n, line))

    # --- plot ---
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # edges (light)
    for _, r in edf.iterrows():
        s = str(r["src"]); t = str(r["dst"])
        if s not in nodes or t not in nodes:
            continue
        xs = [nodes[s][0], nodes[t][0]]
        ys = [nodes[s][1], nodes[t][1]]
        zs = [nodes[s][2], nodes[t][2]]
        ax.plot(xs, ys, zs, linewidth=1.0, alpha=0.10)

    # nodes
    is_anchor = np.array([n in anchors_present for n in node_list], dtype=bool)
    X = P[:,0]; Y = P[:,1]; Z = P[:,2]
    ax.scatter(X[~is_anchor], Y[~is_anchor], Z[~is_anchor], s=35, alpha=0.85, label="nodes")
    ax.scatter(X[is_anchor],  Y[is_anchor],  Z[is_anchor],  s=70, alpha=0.95, label="anchors")

    # local flow arrows (quiver)
    # scale arrows by a percentile of flow_mag so they’re visible but not huge
    mags = np.array([flow_mag[n] for n in node_list], dtype=np.float32)
    scale = float(np.percentile(mags, 75) + 1e-6)
    U = F[:,0] * 0.35
    V = F[:,1] * 0.35
    W = F[:,2] * 0.35
    ax.quiver(X, Y, Z, U, V, W, length=1.0, normalize=False, linewidth=1.0, alpha=0.35)

    # streamlines
    for name, line in all_lines:
        ax.plot(line[:,0], line[:,1], line[:,2], linewidth=2.2, alpha=0.85)
        # label start
        if args.label_anchors and name in anchors_present:
            ax.text(line[0,0], line[0,1], line[0,2], name, fontsize=8)

    ax.set_title(f"{args.title}\n(field k={args.k}, streamlines={len(all_lines)})")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.legend(loc="best")

    out_png = os.path.join(args.outdir, "phase30g_flowfield.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    print(f"[ok] wrote: {out_png}")

    # write streamlines CSV
    rows = []
    for name, line in all_lines:
        for i, p in enumerate(line):
            rows.append({"seed": name, "t": i, "x": float(p[0]), "y": float(p[1]), "z": float(p[2])})
    out_csv = os.path.join(args.outdir, "phase30g_streamlines.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[ok] wrote: {out_csv}")

if __name__ == "__main__":
    main()
