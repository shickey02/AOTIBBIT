#!/usr/bin/env python3
# phase30b_plot_node_graph_3d.py
#
# Visualize extracted node vectors as a 3D graph.
# - Reads phase30a_nodes.csv and phase30a_edges.csv
# - Embeds 256D -> 3D via PCA (numpy SVD; no sklearn needed)
# - Plots 3D scatter with labels + directed edges (arrows)
# - Also writes embedded coords to CSV and makes a 2D PCA plot for sanity

import os, argparse, csv
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def read_nodes_csv(path):
    with open(path, "r", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        # header: node, d0..d255
        rows = []
        for row in r:
            node = row[0]
            vec = np.array([float(x) for x in row[1:]], dtype=np.float32)
            rows.append((node, vec))
    nodes = [n for n,_ in rows]
    X = np.stack([v for _,v in rows], axis=0)  # [N,D]
    return nodes, X

def read_edges_csv(path):
    edges = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            src = row["src"]
            dst = row["dst"]
            plane = row.get("plane", "")
            has_map = int(row.get("has_map", "1"))
            if has_map <= 0:
                continue
            edges.append((src, dst, plane))
    return edges

def pca_embed(X, k=3):
    """
    PCA via SVD on centered data.
    Returns Z [N,k] and explained variance ratio [k].
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    # SVD: Xc = U S Vt
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    # principal axes = Vt[:k]
    Z = Xc @ Vt[:k].T
    # explained variance
    # var along PCi = (S[i]^2) / (N-1)
    N = X.shape[0]
    var = (S**2) / max(N - 1, 1)
    evr = var[:k] / (var.sum() + 1e-12)
    return Z, evr

def write_embed_csv(out_csv, nodes, Z, evr):
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["node", "x", "y", "z", "pca_evr_x", "pca_evr_y", "pca_evr_z"])
        for i, n in enumerate(nodes):
            w.writerow([n, float(Z[i,0]), float(Z[i,1]), float(Z[i,2]),
                        float(evr[0]), float(evr[1]), float(evr[2])])

def plane_style(plane):
    # simple styling: bo solid, lt dashed, bo_lr dotted
    if plane == "bo":
        return dict(linestyle="-", linewidth=2.0, alpha=0.8)
    if plane == "lt":
        return dict(linestyle="--", linewidth=2.0, alpha=0.8)
    if plane == "bo_lr":
        return dict(linestyle=":", linewidth=2.5, alpha=0.8)
    return dict(linestyle="-", linewidth=1.5, alpha=0.6)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes_csv", required=True, help="phase30a_nodes.csv")
    ap.add_argument("--edges_csv", required=True, help="phase30a_edges.csv")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--title", default="30b: 3D relation-graph (PCA)")
    ap.add_argument("--no_arrows", action="store_true", help="draw edges as lines instead of arrows")
    ap.add_argument("--label_offset", type=float, default=0.02, help="text offset in axis units")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    nodes, X = read_nodes_csv(args.nodes_csv)
    edges = read_edges_csv(args.edges_csv)

    Z, evr = pca_embed(X, k=3)

    out_embed = os.path.join(args.outdir, "phase30b_embed_pca3.csv")
    write_embed_csv(out_embed, nodes, Z, evr)

    # index map
    idx = {n:i for i,n in enumerate(nodes)}

    # ---------- 3D plot ----------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"{args.title}\nEVR: {evr[0]:.3f}, {evr[1]:.3f}, {evr[2]:.3f}")

    xs, ys, zs = Z[:,0], Z[:,1], Z[:,2]
    ax.scatter(xs, ys, zs, s=120)

    # labels
    for i, n in enumerate(nodes):
        ax.text(xs[i] + args.label_offset,
                ys[i] + args.label_offset,
                zs[i] + args.label_offset,
                n, fontsize=9)

    # edges
    # We’ll draw a vector from src->dst. For arrows in 3D we use quiver.
    for (src, dst, plane) in edges:
        if src not in idx or dst not in idx:
            continue
        i = idx[src]; j = idx[dst]
        x0,y0,z0 = Z[i]
        x1,y1,z1 = Z[j]
        dx,dy,dz = (x1-x0, y1-y0, z1-z0)
        st = plane_style(plane)

        if args.no_arrows:
            ax.plot([x0,x1],[y0,y1],[z0,z1], **st)
        else:
            # quiver arrow; length=1 uses dx/dy/dz directly
            ax.quiver(x0,y0,z0, dx,dy,dz, arrow_length_ratio=0.12, **st)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    out_png3d = os.path.join(args.outdir, "phase30b_graph_3d.png")
    plt.tight_layout()
    plt.savefig(out_png3d, dpi=180)
    plt.close(fig)

    # ---------- 2D plot (PC1 vs PC2) ----------
    fig2 = plt.figure(figsize=(9, 7))
    ax2 = fig2.add_subplot(111)
    ax2.set_title(f"{args.title} (2D)\nEVR: {evr[0]:.3f}, {evr[1]:.3f}")
    ax2.scatter(Z[:,0], Z[:,1], s=140)

    for i, n in enumerate(nodes):
        ax2.text(Z[i,0] + args.label_offset,
                 Z[i,1] + args.label_offset,
                 n, fontsize=9)

    for (src, dst, plane) in edges:
        if src not in idx or dst not in idx:
            continue
        i = idx[src]; j = idx[dst]
        x0,y0 = Z[i,0], Z[i,1]
        x1,y1 = Z[j,0], Z[j,1]
        st = plane_style(plane)
        ax2.plot([x0,x1],[y0,y1], **st)

    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.grid(True, alpha=0.25)

    out_png2d = os.path.join(args.outdir, "phase30b_graph_2d.png")
    plt.tight_layout()
    plt.savefig(out_png2d, dpi=180)
    plt.close(fig2)

    # summary
    print(f"[ok] wrote: {out_embed}")
    print(f"[ok] wrote: {out_png3d}")
    print(f"[ok] wrote: {out_png2d}")
    print(f"[info] nodes={len(nodes)} edges={len(edges)} evr3={evr[0]:.3f},{evr[1]:.3f},{evr[2]:.3f}")

if __name__ == "__main__":
    main()
