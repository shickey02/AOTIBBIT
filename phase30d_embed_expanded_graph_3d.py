#!/usr/bin/env python3
# phase30d_embed_expanded_graph_3d.py
#
# Step 3A:
# - Read expanded graph nodes/edges from phase30c (CSV)
# - Pull 256-D base vectors for each node from the transport cache (phase15f schema)
# - Compute a 3D embedding (PCA)
# - Render a 3D graph plot + write coords CSV
#
# Output:
#   outdir/phase30d_nodes_3d.csv
#   outdir/phase30d_graph_3d.png
#   outdir/phase30d_graph_3d_labeled.png (optional)

import os, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


# ------------------------
# Cache helpers (same schema as your 15f cache)
# ------------------------
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def _try_array(x):
    try:
        a = np.array(x)
        if a.dtype == object:
            return None
        if a.size == 0:
            return None
        if a.dtype.kind in ("f", "i", "u"):
            return a.astype(np.float32)
        return None
    except Exception:
        return None

def _find_numeric_lists(obj, min_len=8):
    found = []
    if isinstance(obj, list):
        a = _try_array(obj)
        if a is not None and a.size >= min_len:
            found.append(a)
        else:
            for v in obj:
                found.extend(_find_numeric_lists(v, min_len=min_len))
    elif isinstance(obj, dict):
        for v in obj.values():
            found.extend(_find_numeric_lists(v, min_len=min_len))
    return found

def get_base_vec(cache, node):
    b = cache["bases"][node]
    if isinstance(b, list):
        v = _try_array(b)
        if v is None:
            raise ValueError(f"bases[{node}] list but not numeric")
        v = v.reshape(-1).astype(np.float32)
        return v
    if isinstance(b, dict):
        cands = _find_numeric_lists(b, min_len=200)
        for a in cands:
            if a.ndim == 1 and a.shape[0] == 256:
                return a.astype(np.float32)
            if a.ndim == 2:
                aa = a.reshape(-1)
                if aa.shape[0] == 256:
                    return aa.astype(np.float32)
        raise ValueError(f"bases[{node}] dict but no 256-vector found. keys={list(b.keys())[:10]}")
    raise ValueError(f"bases[{node}] unsupported type {type(b)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="phase15f_transport_cache.json")
    ap.add_argument("--nodes_csv", required=True, help="phase30c_nodes.csv")
    ap.add_argument("--edges_csv", required=True, help="phase30c_edges.csv")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--label", action="store_true", help="render labels (can be cluttered)")
    ap.add_argument("--title", default="Phase30D: Expanded Graph 3D (PCA from 256-D bases)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    cache = load_json(args.cache)
    nodes_df = pd.read_csv(args.nodes_csv)
    edges_df = pd.read_csv(args.edges_csv)

    # Expect nodes_df to include at least 'node'
    if "node" not in nodes_df.columns:
        # tolerate 'name'
        if "name" in nodes_df.columns:
            nodes_df = nodes_df.rename(columns={"name": "node"})
        else:
            raise ValueError(f"nodes_csv must have a 'node' column. got={list(nodes_df.columns)}")

    # Expect edges_df to include src/dst
    # tolerate 'u','v' or 'from','to'
    if not {"src","dst"}.issubset(edges_df.columns):
        if {"u","v"}.issubset(edges_df.columns):
            edges_df = edges_df.rename(columns={"u":"src","v":"dst"})
        elif {"from","to"}.issubset(edges_df.columns):
            edges_df = edges_df.rename(columns={"from":"src","to":"dst"})
        else:
            raise ValueError(f"edges_csv must have src/dst columns. got={list(edges_df.columns)}")

    nodes = nodes_df["node"].astype(str).tolist()

    # Build matrix of base vectors
    X = []
    missing = []
    for n in nodes:
        if n not in cache.get("bases", {}):
            missing.append(n)
            continue
        X.append(get_base_vec(cache, n))
    if missing:
        print(f"[warn] {len(missing)} nodes missing from cache['bases'] (will be skipped). first={missing[:10]}")

    X = np.stack(X, axis=0)  # [N,256]

    # PCA -> 3D
    pca = PCA(n_components=3, random_state=0)
    Y = pca.fit_transform(X)  # [N,3]
    evr = pca.explained_variance_ratio_
    print(f"[info] PCA EVR: {evr[0]:.3f}, {evr[1]:.3f}, {evr[2]:.3f} (sum={evr.sum():.3f})")

    # Map node->coord
    kept_nodes = [n for n in nodes if n in cache.get("bases", {})]
    coord = {n: Y[i] for i, n in enumerate(kept_nodes)}

    # write coords csv
    out_nodes = os.path.join(args.outdir, "phase30d_nodes_3d.csv")
    out_df = pd.DataFrame({
        "node": kept_nodes,
        "x": [coord[n][0] for n in kept_nodes],
        "y": [coord[n][1] for n in kept_nodes],
        "z": [coord[n][2] for n in kept_nodes],
    })
    out_df.to_csv(out_nodes, index=False)
    print(f"[ok] wrote: {out_nodes}")

    # Plot 3D graph
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # nodes
    ax.scatter(out_df["x"], out_df["y"], out_df["z"], s=25)

    # edges
    # only draw edges where both endpoints have coords
    drawn = 0
    for _, r in edges_df.iterrows():
        s = str(r["src"])
        t = str(r["dst"])
        if s not in coord or t not in coord:
            continue
        xs = [coord[s][0], coord[t][0]]
        ys = [coord[s][1], coord[t][1]]
        zs = [coord[s][2], coord[t][2]]
        ax.plot(xs, ys, zs, linewidth=1, alpha=0.35)
        drawn += 1

    ax.set_title(args.title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    out_png = os.path.join(args.outdir, "phase30d_graph_3d.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    print(f"[ok] wrote: {out_png} (edges drawn={drawn})")

    # Optional labeled plot
    if args.label:
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, projection="3d")
        ax2.scatter(out_df["x"], out_df["y"], out_df["z"], s=25)

        for _, r in edges_df.iterrows():
            s = str(r["src"]); t = str(r["dst"])
            if s not in coord or t not in coord:
                continue
            ax2.plot([coord[s][0], coord[t][0]],
                     [coord[s][1], coord[t][1]],
                     [coord[s][2], coord[t][2]],
                     linewidth=1, alpha=0.35)

        for n in kept_nodes:
            ax2.text(coord[n][0], coord[n][1], coord[n][2], n, fontsize=7)

        ax2.set_title(args.title + " (labeled)")
        ax2.set_xlabel("PC1"); ax2.set_ylabel("PC2"); ax2.set_zlabel("PC3")

        out_png2 = os.path.join(args.outdir, "phase30d_graph_3d_labeled.png")
        plt.tight_layout()
        plt.savefig(out_png2, dpi=200)
        print(f"[ok] wrote: {out_png2}")

if __name__ == "__main__":
    main()
