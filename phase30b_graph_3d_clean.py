#!/usr/bin/env python3
# phase30b_graph_3d_clean.py
#
# Build a CLEAN graph from node vectors:
# - kNN graph (optionally mutual)
# - optional MST to ensure connectivity
# - PCA -> 3D coords
# - plots 2D + 3D
# NEW: semantic coloring by node family (between/overlap/left/right/above/below/boundary)
#
# Expected input nodes CSV:
# - must have a "node" column
# - must have embedding columns (e.g. dim000..dim255 OR any numeric columns except helper cols)
#
# Outputs:
# - phase30b_nodes_3d.csv (node,x,y,z,family)
# - phase30b_edges_knn.csv (src,dst,weight,kind)
# - phase30b_graph_2d_clean.png
# - phase30b_graph_3d_clean.png

import os
import argparse
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# -----------------------------
# Coloring / families
# -----------------------------
FAMILY_COLORS = {
    "between": "#1f77b4",   # blue
    "overlap": "#ff7f0e",   # orange
    "left":   "#2ca02c",    # green
    "right":  "#d62728",    # red
    "above":  "#9467bd",    # purple
    "below":  "#8c564b",    # brown
    "boundary":"#7f7f7f",   # gray
    "other":  "#000000",    # black
}

def node_family(name: str) -> str:
    n = name.lower()
    # boundary is a property that can appear inside other names
    if "boundary" in n:
        return "boundary"
    if n.startswith("between"):
        return "between"
    if n.startswith("overlap"):
        return "overlap"
    if n.startswith("left"):
        return "left"
    if n.startswith("right"):
        return "right"
    if n.startswith("above"):
        return "above"
    if n.startswith("below"):
        return "below"
    return "other"

def color_for_node(name: str) -> str:
    return FAMILY_COLORS.get(node_family(name), FAMILY_COLORS["other"])


# -----------------------------
# Graph helpers
# -----------------------------
def pairwise_dist(X: np.ndarray) -> np.ndarray:
    # squared euclidean -> euclidean
    # (N,D) -> (N,N)
    G = X @ X.T
    s = np.sum(X*X, axis=1, keepdims=True)
    D2 = s - 2*G + s.T
    D2 = np.maximum(D2, 0.0)
    return np.sqrt(D2)

def build_knn_edges(X: np.ndarray, k: int, mutual: bool):
    D = pairwise_dist(X)
    N = D.shape[0]
    edges = set()

    for i in range(N):
        nn = np.argsort(D[i])[1:k+1]  # skip self
        for j in nn:
            if mutual:
                # check if i is in j's kNN
                nnj = np.argsort(D[j])[1:k+1]
                if i not in set(nnj.tolist()):
                    continue
            a, b = (i, j) if i < j else (j, i)
            edges.add((a, b, float(D[i, j])))

    return list(edges)

def mst_edges(X: np.ndarray):
    # Prim's algorithm on euclidean distances
    D = pairwise_dist(X)
    N = D.shape[0]
    in_tree = np.zeros(N, dtype=bool)
    in_tree[0] = True
    best = D[0].copy()
    parent = np.zeros(N, dtype=int)
    parent[:] = 0

    edges = []
    for _ in range(N-1):
        j = np.argmin(np.where(in_tree, np.inf, best))
        if not np.isfinite(best[j]):
            break
        in_tree[j] = True
        i = parent[j]
        a, b = (i, j) if i < j else (j, i)
        edges.append((a, b, float(D[i, j])))
        # update frontier
        for t in range(N):
            if not in_tree[t] and D[j, t] < best[t]:
                best[t] = D[j, t]
                parent[t] = j
    return edges

def write_edges_csv(out_path, nodes, edges, kind):
    rows = []
    for a, b, w in edges:
        rows.append({"src": nodes[a], "dst": nodes[b], "weight": w, "kind": kind})
    pd.DataFrame(rows).to_csv(out_path, index=False)

def plot_2d(nodes3d_df, edges_df, out_png, title):
    x = nodes3d_df["x"].values
    y = nodes3d_df["y"].values
    names = nodes3d_df["node"].tolist()

    plt.figure(figsize=(12, 8))
    # edges
    for _, r in edges_df.iterrows():
        a = r["src"]; b = r["dst"]
        ia = names.index(a); ib = names.index(b)
        # edge color: lightly tinted by src family
        c = color_for_node(a)
        plt.plot([x[ia], x[ib]], [y[ia], y[ib]], color=c, alpha=0.25, linewidth=1)

    # nodes (colored)
    for i, n in enumerate(names):
        plt.scatter([x[i]], [y[i]], s=55, color=color_for_node(n), edgecolor="none", alpha=0.95)

    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_3d(nodes3d_df, edges_df, out_png, title):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    x = nodes3d_df["x"].values
    y = nodes3d_df["y"].values
    z = nodes3d_df["z"].values
    names = nodes3d_df["node"].tolist()

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # edges
    for _, r in edges_df.iterrows():
        a = r["src"]; b = r["dst"]
        ia = names.index(a); ib = names.index(b)
        c = color_for_node(a)
        ax.plot([x[ia], x[ib]], [y[ia], y[ib]], [z[ia], z[ib]], color=c, alpha=0.25, linewidth=1)

    # nodes
    for i, n in enumerate(names):
        ax.scatter([x[i]], [y[i]], [z[i]], s=60, color=color_for_node(n), alpha=0.95)

    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True, help="CSV with node + embedding dims")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--mutual", action="store_true")
    ap.add_argument("--add_mst", action="store_true")
    ap.add_argument("--title", default="30b: graph CLEAN kNN+MST (colored)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.nodes)
    if "node" not in df.columns:
        raise ValueError(f"nodes csv must have column 'node'. cols={list(df.columns)[:20]}")

    nodes = df["node"].astype(str).tolist()

    # pick numeric embedding columns robustly:
    # exclude obvious non-emb cols
    exclude = {"node", "name", "family", "seed", "anchor", "plane"}
    num_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)

    if len(num_cols) < 3:
        raise ValueError(f"Could not find enough numeric embedding columns. Found {len(num_cols)}: {num_cols[:10]}")

    X = df[num_cols].values.astype(np.float32)
    # normalize for distance stability
    X = X - X.mean(axis=0, keepdims=True)
    X = X / (X.std(axis=0, keepdims=True) + 1e-9)

    # edges
    knn = build_knn_edges(X, k=args.k, mutual=args.mutual)
    all_edges = {(a,b): w for a,b,w in knn}

    if args.add_mst:
        mst = mst_edges(X)
        for a,b,w in mst:
            all_edges[(a,b)] = min(all_edges.get((a,b), w), w)

    # build edges df
    edges_rows = []
    for (a,b), w in all_edges.items():
        edges_rows.append({"src": nodes[a], "dst": nodes[b], "weight": float(w), "kind": "knn_mst"})
    edges_df = pd.DataFrame(edges_rows)

    # PCA -> 3D
    pca = PCA(n_components=3)
    P = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_.tolist()

    nodes3d = pd.DataFrame({
        "node": nodes,
        "x": P[:,0],
        "y": P[:,1],
        "z": P[:,2],
        "family": [node_family(n) for n in nodes]
    })

    out_nodes3d = os.path.join(args.outdir, "phase30b_nodes_3d.csv")
    out_edges   = os.path.join(args.outdir, "phase30b_edges_knn.csv")
    out_2d      = os.path.join(args.outdir, "phase30b_graph_2d_clean.png")
    out_3d      = os.path.join(args.outdir, "phase30b_graph_3d_clean.png")

    nodes3d.to_csv(out_nodes3d, index=False)
    edges_df.to_csv(out_edges, index=False)

    title = f"{args.title}\nEVR: {evr[0]:.3f}, {evr[1]:.3f}, {evr[2]:.3f}"
    plot_2d(nodes3d, edges_df, out_2d, title.replace("(colored)", "(colored) (2D)"))
    plot_3d(nodes3d, edges_df, out_3d, title)

    print(f"[ok] nodes={len(nodes3d)} edges={len(edges_df)} (k={args.k} mutual={args.mutual} add_mst={args.add_mst})")
    print("[ok] wrote:", out_nodes3d)
    print("[ok] wrote:", out_edges)
    print("[ok] wrote:", out_2d)
    print("[ok] wrote:", out_3d)


if __name__ == "__main__":
    main()
