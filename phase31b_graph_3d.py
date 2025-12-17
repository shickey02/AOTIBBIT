#!/usr/bin/env python3
# phase31b_graph_3d.py
#
# Take an expanded node-vector table (e.g. phase31a_nodes.csv) and an edge list,
# compute PCA->3D coordinates, and render a 3D graph plot.
#
# Inputs:
#   --nodes_csv : CSV with columns: name (or node), vec_0..vec_{D-1}
#   --edges_csv : CSV with columns: src,dst (optional: w,count,weight)
#
# Outputs:
#   outdir/phase31b_nodes_3d.csv
#   outdir/phase31b_edges_3d.csv
#   outdir/phase31b_graph_3d.png
#   outdir/phase31b_pca_summary.txt

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

def _pick_name_col(df):
    for c in ["name", "node", "id"]:
        if c in df.columns:
            return c
    raise ValueError(f"nodes_csv must include a name column like name/node/id. got={list(df.columns)[:20]}...")

def _pick_vec_cols(df):
    vec_cols = [c for c in df.columns if c.startswith("vec_")]
    if not vec_cols:
        # allow raw dims like 0..255 (rare)
        maybe = []
        for c in df.columns:
            try:
                int(c)
                maybe.append(c)
            except:
                pass
        if maybe:
            vec_cols = sorted(maybe, key=lambda x: int(x))
    if not vec_cols:
        raise ValueError("nodes_csv must include vec_0..vec_D-1 columns.")
    return vec_cols

def _read_edges(edges_csv):
    edf = pd.read_csv(edges_csv)
    # normalize column names
    rename = {}
    if "src" not in edf.columns:
        for c in ["from", "u", "source"]:
            if c in edf.columns: rename[c] = "src"
    if "dst" not in edf.columns:
        for c in ["to", "v", "target"]:
            if c in edf.columns: rename[c] = "dst"
    if rename:
        edf = edf.rename(columns=rename)
    if "src" not in edf.columns or "dst" not in edf.columns:
        raise ValueError(f"edges_csv must have src,dst columns (or from,to). got={list(edf.columns)}")

    # weight preference order
    wcol = None
    for c in ["weight", "w", "count", "n", "freq"]:
        if c in edf.columns:
            wcol = c
            break
    if wcol is None:
        edf["weight"] = 1.0
        wcol = "weight"
    else:
        edf["weight"] = pd.to_numeric(edf[wcol], errors="coerce").fillna(1.0).astype(float)

    edf["src"] = edf["src"].astype(str)
    edf["dst"] = edf["dst"].astype(str)
    return edf[["src", "dst", "weight"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes_csv", required=True)
    ap.add_argument("--edges_csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--title", default="31b: expanded graph 3D (PCA)")
    ap.add_argument("--max_edges_plot", type=int, default=2500, help="cap for drawing edges")
    ap.add_argument("--edge_alpha", type=float, default=0.15)
    ap.add_argument("--node_size", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    np.random.seed(args.seed)

    ndf = pd.read_csv(args.nodes_csv)
    name_col = _pick_name_col(ndf)
    vec_cols = _pick_vec_cols(ndf)

    names = ndf[name_col].astype(str).tolist()
    X = ndf[vec_cols].to_numpy(dtype=np.float32)
    N, D = X.shape

    # PCA -> 3D
    pca = PCA(n_components=3, random_state=args.seed)
    X3 = pca.fit_transform(X)  # (N,3)
    evr = pca.explained_variance_ratio_.tolist()

    n3 = pd.DataFrame({
        "name": names,
        "x": X3[:,0],
        "y": X3[:,1],
        "z": X3[:,2],
    })
    # keep metadata columns if present
    for c in ndf.columns:
        if c not in vec_cols and c != name_col and c not in n3.columns:
            n3[c] = ndf[c]

    n3_path = os.path.join(args.outdir, "phase31b_nodes_3d.csv")
    n3.to_csv(n3_path, index=False)

    edf = _read_edges(args.edges_csv)

    # filter to known nodes
    node_set = set(n3["name"].astype(str))
    edf = edf[edf["src"].isin(node_set) & edf["dst"].isin(node_set)].copy()

    # edge coordinates
    idx = {n:i for i,n in enumerate(n3["name"].astype(str).tolist())}
    edf["src_i"] = edf["src"].map(idx)
    edf["dst_i"] = edf["dst"].map(idx)

    e3_path = os.path.join(args.outdir, "phase31b_edges_3d.csv")
    edf.to_csv(e3_path, index=False)

    # Plot
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(n3["x"], n3["y"], n3["z"], s=args.node_size)

    # sample edges if huge
    if len(edf) > args.max_edges_plot:
        edf_plot = edf.sample(args.max_edges_plot, random_state=args.seed)
    else:
        edf_plot = edf

    xs = n3["x"].to_numpy()
    ys = n3["y"].to_numpy()
    zs = n3["z"].to_numpy()

    for _, r in edf_plot.iterrows():
        i = int(r["src_i"]); j = int(r["dst_i"])
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]],
                linewidth=0.7, alpha=args.edge_alpha)

    ax.set_title(args.title)
    ax.set_xlabel(f"PC1 ({evr[0]:.3f})")
    ax.set_ylabel(f"PC2 ({evr[1]:.3f})")
    ax.set_zlabel(f"PC3 ({evr[2]:.3f})")

    out_png = os.path.join(args.outdir, "phase31b_graph_3d.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close(fig)

    summ = os.path.join(args.outdir, "phase31b_pca_summary.txt")
    with open(summ, "w") as f:
        f.write(f"[info] nodes={N} dim={D}\n")
        f.write(f"[info] PCA EVR: {evr[0]:.6f}, {evr[1]:.6f}, {evr[2]:.6f} (sum={sum(evr):.6f})\n")
        f.write(f"[out] {n3_path}\n")
        f.write(f"[out] {e3_path}\n")
        f.write(f"[out] {out_png}\n")

    print(f"[ok] wrote: {n3_path}")
    print(f"[ok] wrote: {e3_path}")
    print(f"[ok] wrote: {out_png}")
    print(f"[info] PCA EVR: {evr[0]:.3f}, {evr[1]:.3f}, {evr[2]:.3f} (sum={sum(evr):.3f})")

if __name__ == "__main__":
    main()
