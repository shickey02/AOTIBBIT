#!/usr/bin/env python3
# phase30h_basin_map_3d.py
#
# Step 3D:
# Build a smooth flow field on the 3D embedded graph and compute "basins":
# for each node (and optional random seeds), integrate forward and label by
# nearest anchor at the terminal point.
#
# Inputs:
#   --cache   : transport_cache.json (for anchor names)
#   --nodes3d : CSV with columns [node,x,y,z] (from phase30d)
#   --edges   : CSV with columns [src,dst] or [u,v] (from phase30c)
#
# Outputs:
#   - phase30h_basin_nodes.csv  (node -> basin_anchor, end_xyz, dist)
#   - phase30h_basin_summary.csv (counts)
#   - phase30h_basin_plot.png   (3D plot colored by basin)

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
    for c in ["count", "w", "weight", "n", "freq"]:
        if c in edf.columns:
            return c
    return None

def build_node_flow(nodes, edges, weight_col=None):
    flow_sum = {n: np.zeros(3, dtype=np.float32) for n in nodes.keys()}
    for _, r in edges.iterrows():
        s = str(r["src"]); t = str(r["dst"])
        if s not in nodes or t not in nodes or s == t:
            continue
        w = float(r[weight_col]) if (weight_col and weight_col in r and pd.notna(r[weight_col])) else 1.0
        d = nodes[t] - nodes[s]
        flow_sum[s] += w * d

    flow = {}
    for n in nodes.keys():
        v = flow_sum[n]
        v_unit, vlen = norm(v)
        # if a node has no outgoing signal, leave it zero
        if vlen < 1e-6:
            flow[n] = np.zeros(3, dtype=np.float32)
        else:
            flow[n] = v_unit.astype(np.float32)
    return flow

def knn_field(points, flows, k=6, eps=1e-9):
    N = points.shape[0]
    def f(x):
        d = np.linalg.norm(points - x[None, :], axis=1)
        idx = np.argsort(d)[:min(k, N)]
        di = d[idx]
        wi = 1.0 / (di + eps)
        v = (flows[idx] * wi[:, None]).sum(axis=0) / (wi.sum() + eps)
        v_unit, vlen = norm(v)
        if vlen < 1e-6:
            return np.zeros(3, dtype=np.float32)
        return v_unit.astype(np.float32)
    return f

def integrate(f, x0, step=0.06, n_steps=90, stop_v=1e-4):
    x = x0.astype(np.float32)
    for _ in range(n_steps):
        v = f(x)
        if not np.all(np.isfinite(v)):
            break
        if float(np.linalg.norm(v)) < stop_v:
            break
        x = x + step * v
    return x

def nearest_anchor(x, anchor_pos):
    # anchor_pos: dict name->(3,)
    best = None
    best_d = None
    for a, p in anchor_pos.items():
        d = float(np.linalg.norm(x - p))
        if best is None or d < best_d:
            best = a; best_d = d
    return best, best_d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--nodes3d", required=True)
    ap.add_argument("--edges", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--title", default="30h: basin map (flow to nearest anchor)")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--step_size", type=float, default=0.06)
    ap.add_argument("--random_seeds", type=int, default=0, help="extra random points inside bbox")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed)

    cache = load_json(args.cache)
    anchors = list(cache.get("anchors", {}).keys())

    ndf = pd.read_csv(args.nodes3d)
    edf = pd.read_csv(args.edges)
    if "node" not in ndf.columns:
        raise ValueError(f"nodes3d csv needs 'node'. got={list(ndf.columns)}")
    for c in ["x","y","z"]:
        if c not in ndf.columns:
            raise ValueError(f"nodes3d csv needs x,y,z. got={list(ndf.columns)}")

    s_col, t_col = infer_edge_cols(edf)
    edf = edf.rename(columns={s_col:"src", t_col:"dst"})
    wcol = infer_weight_col(edf)

    nodes = {str(r["node"]): np.array([float(r["x"]), float(r["y"]), float(r["z"])], dtype=np.float32)
             for _, r in ndf.iterrows()}
    node_list = list(nodes.keys())
    P = np.stack([nodes[n] for n in node_list], axis=0)

    # anchor positions (only those present in nodes3d)
    anchor_pos = {a: nodes[a] for a in anchors if a in nodes}
    if len(anchor_pos) < 2:
        raise ValueError(f"Need >=2 anchors present in nodes3d. have={list(anchor_pos.keys())}")

    # node flow + continuous field
    flow = build_node_flow(nodes, edf, weight_col=wcol)
    F = np.stack([flow[n] for n in node_list], axis=0)
    field = knn_field(P, F, k=args.k)

    # bbox for random seeds
    mins = P.min(axis=0); maxs = P.max(axis=0)

    rows = []

    # classify each node
    for n in node_list:
        x_end = integrate(field, nodes[n], step=args.step_size, n_steps=args.steps)
        basin, dist = nearest_anchor(x_end, anchor_pos)
        rows.append({
            "seed": n,
            "seed_type": "node",
            "basin": basin,
            "dist_to_basin": dist,
            "end_x": float(x_end[0]),
            "end_y": float(x_end[1]),
            "end_z": float(x_end[2]),
        })

    # optional extra random seeds (gives a “filled” basin picture even with few nodes)
    for i in range(int(args.random_seeds)):
        x0 = mins + (maxs - mins) * np.random.rand(3).astype(np.float32)
        x_end = integrate(field, x0, step=args.step_size, n_steps=args.steps)
        basin, dist = nearest_anchor(x_end, anchor_pos)
        rows.append({
            "seed": f"rand_{i:04d}",
            "seed_type": "random",
            "basin": basin,
            "dist_to_basin": dist,
            "end_x": float(x_end[0]),
            "end_y": float(x_end[1]),
            "end_z": float(x_end[2]),
        })

    out_nodes = os.path.join(args.outdir, "phase30h_basin_nodes.csv")
    df = pd.DataFrame(rows)
    df.to_csv(out_nodes, index=False)
    print(f"[ok] wrote: {out_nodes}")

    summary = df[df["seed_type"]=="node"].groupby("basin").size().reset_index(name="count_nodes")
    out_sum = os.path.join(args.outdir, "phase30h_basin_summary.csv")
    summary.to_csv(out_sum, index=False)
    print(f"[ok] wrote: {out_sum}")

    # --- plot (nodes colored by basin) ---
    # Map basins to integer ids for consistent coloring
    basin_names = sorted(summary["basin"].tolist())
    basin_id = {b:i for i,b in enumerate(basin_names)}

    node_df = df[df["seed_type"]=="node"].copy()
    node_df["basin_id"] = node_df["basin"].map(basin_id).fillna(-1).astype(int)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # edges (light)
    for _, r in edf.iterrows():
        s = str(r["src"]); t = str(r["dst"])
        if s not in nodes or t not in nodes:
            continue
        ax.plot([nodes[s][0], nodes[t][0]],
                [nodes[s][1], nodes[t][1]],
                [nodes[s][2], nodes[t][2]],
                linewidth=1.0, alpha=0.10)

    # scatter by basin
    for b in basin_names:
        mask = (node_df["basin"] == b)
        sub = node_df[mask]
        # default matplotlib assigns different colors automatically by call order
        ax.scatter(sub["end_x"], sub["end_y"], sub["end_z"], s=55, alpha=0.9, label=f"basin→{b}")

    # plot anchors as bigger points at their actual node positions
    for a, p in anchor_pos.items():
        ax.scatter([p[0]], [p[1]], [p[2]], s=160, alpha=1.0)
        ax.text(p[0], p[1], p[2], a, fontsize=9)

    ax.set_title(f"{args.title}\n(k={args.k}, steps={args.steps}, step={args.step_size})")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.legend(loc="best")

    out_png = os.path.join(args.outdir, "phase30h_basin_plot.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    print(f"[ok] wrote: {out_png}")

if __name__ == "__main__":
    main()
