#!/usr/bin/env python3
# phase30f_overlay_plane_axes.py
#
# Step 3B:
# Overlay three global "plane axes" on top of a 3D node graph.
# We derive each axis as a delta between two anchor nodes in the 3D PCA space.
#
# This avoids relying on transport_map schema (which may store 2x2 coeff-space maps).
# Instead: treat anchor pairs as semantic endpoints:
#   - bo     : between_deep_clear -> overlap_deep_only   (between vs overlap)
#   - lt     : overlap_boundary_only -> overlap_deep_only (boundary vs deep within overlap)
#   - bo_lr  : overlap_boundary_only -> left_clean       (overlap vs left)
#
# If any endpoint is missing, we fallback to the farthest pair among anchors.

import os, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def norm(v, eps=1e-9):
    n = float(np.linalg.norm(v))
    return v / (n + eps), n

def pick_pair(anchors_present, coord3d, preferred_a, preferred_b):
    if preferred_a in anchors_present and preferred_b in anchors_present:
        return preferred_a, preferred_b, "preferred"

    # fallback: farthest anchor pair
    A = list(anchors_present)
    if len(A) < 2:
        return None, None, "insufficient-anchors"

    best = (None, None, -1.0)
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            u, v = A[i], A[j]
            du = np.array(coord3d[u])
            dv = np.array(coord3d[v])
            d = float(np.linalg.norm(dv - du))
            if d > best[2]:
                best = (u, v, d)
    return best[0], best[1], "farthest-fallback"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--nodes3d", required=True)
    ap.add_argument("--edges", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--title", default="30f: graph + global relation axes")
    ap.add_argument("--axis_scale", type=float, default=1.15, help="how long to draw axes (relative to graph radius)")
    ap.add_argument("--label_anchors", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    cache = load_json(args.cache)
    anchors = set(cache.get("anchors", {}).keys())

    ndf = pd.read_csv(args.nodes3d)
    edf = pd.read_csv(args.edges)

    # infer edge columns
    if {"src","dst"}.issubset(edf.columns):
        s_col, t_col = "src", "dst"
    elif {"u","v"}.issubset(edf.columns):
        s_col, t_col = "u", "v"
    else:
        raise ValueError(f"edges csv missing src/dst-like columns. got={list(edf.columns)}")

    ndf["node"] = ndf["node"].astype(str)
    coord = {r["node"]:(float(r["x"]), float(r["y"]), float(r["z"])) for _, r in ndf.iterrows()}
    nodes = list(coord.keys())

    anchors_present = set([n for n in nodes if n in anchors])

    # graph centroid + radius
    P = np.array([coord[n] for n in nodes], dtype=np.float32)
    center = P.mean(axis=0)
    radius = float(np.max(np.linalg.norm(P - center[None, :], axis=1)))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # draw edges
    drawn = 0
    for _, r in edf.iterrows():
        s = str(r[s_col]); t = str(r[t_col])
        if s not in coord or t not in coord:
            continue
        xs = [coord[s][0], coord[t][0]]
        ys = [coord[s][1], coord[t][1]]
        zs = [coord[s][2], coord[t][2]]
        ax.plot(xs, ys, zs, linewidth=1.0, alpha=0.18)
        drawn += 1

    # draw nodes
    is_anchor = np.array([n in anchors_present for n in nodes], dtype=bool)
    X = np.array([coord[n][0] for n in nodes])
    Y = np.array([coord[n][1] for n in nodes])
    Z = np.array([coord[n][2] for n in nodes])

    ax.scatter(X[~is_anchor], Y[~is_anchor], Z[~is_anchor], s=40, alpha=0.85, label="discovered")
    ax.scatter(X[is_anchor],  Y[is_anchor],  Z[is_anchor],  s=70, alpha=0.95, label="anchors")

    if args.label_anchors:
        for n in anchors_present:
            x,y,z = coord[n]
            ax.text(x, y, z, n, fontsize=8)

    # axis definitions (preferred endpoint pairs)
    axis_specs = [
        ("bo",    "between_deep_clear",    "overlap_deep_only"),
        ("lt",    "overlap_boundary_only", "overlap_deep_only"),
        ("bo_lr", "overlap_boundary_only", "left_clean"),
    ]

    # draw axes from center
    for name, a_pref, b_pref in axis_specs:
        a, b, mode = pick_pair(anchors_present, coord, a_pref, b_pref)
        if a is None:
            print(f"[warn] axis {name}: cannot pick pair (need >=2 anchors)")
            continue

        va = np.array(coord[a], dtype=np.float32)
        vb = np.array(coord[b], dtype=np.float32)
        d = vb - va
        d_unit, d_len = norm(d)

        # scale axis to graph radius
        L = args.axis_scale * radius
        tip = center + d_unit * L

        ax.plot([center[0], tip[0]], [center[1], tip[1]], [center[2], tip[2]],
                linewidth=3.0, alpha=0.9)
        ax.text(tip[0], tip[1], tip[2], f"{name} ({mode}: {a}→{b})", fontsize=9)

    ax.set_title(f"{args.title}\n(nodes={len(nodes)} edges_drawn={drawn})")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.legend(loc="best")

    out_png = os.path.join(args.outdir, "phase30f_graph_3d_with_axes.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    print(f"[ok] wrote: {out_png}")

if __name__ == "__main__":
    main()
