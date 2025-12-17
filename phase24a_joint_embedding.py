#!/usr/bin/env python3
# phase24a_joint_embedding.py
#
# Build a "joint manifold" by stacking per-plane embeddings (lt/bo/bo_lr),
# producing an explicit 6D/7D object, then projecting to 3D for visualization.
#
# Fixes:
# - cluster JSON files are OPTIONAL; missing files won't crash.
# - picks first existing cluster map in priority order: bo_lr -> lt -> bo

import os, json, argparse, math
import numpy as np

# ---------------------------- utils ----------------------------

def safe_mkdir(p):
    if p and not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def exists(path):
    return (path is not None) and os.path.isfile(path)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def zscore(X, eps=1e-12):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return (X - mu) / sd, mu.squeeze().tolist(), sd.squeeze().tolist()

# ---------------------- phase21b reading ----------------------

def get_coords_from_phase21b(edge_axis_json):
    data = load_json(edge_axis_json)
    coords = data.get("embedding", {}).get("coords", {})
    out = {}
    for k, v in coords.items():
        vv = list(v)
        while len(vv) < 3:
            vv.append(0.0)
        out[k] = vv[:3]
    return out, data

# ---------------------- phase22a cluster reading ----------------------

def get_node_cluster_map_from_phase22a(clusters_json):
    """
    phase22a_edge_axis_clusters_<plane>.json contains:
      edge_clusters: dict[str] -> list[edge dict with cluster_id, a, b]
    We assign each node a cluster label by majority vote over incident edges.
    """
    data = load_json(clusters_json)
    edge_clusters = data.get("edge_clusters", {})
    node_votes = {}

    def vote(n, cid):
        if n not in node_votes:
            node_votes[n] = {}
        node_votes[n][cid] = node_votes[n].get(cid, 0) + 1

    for _, edges in edge_clusters.items():
        for e in edges:
            cid = int(e.get("cluster_id", -1))
            a = e.get("a", None)
            b = e.get("b", None)
            if a is not None:
                vote(a, cid)
            if b is not None:
                vote(b, cid)

    node_cluster = {}
    for n, counts in node_votes.items():
        best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        node_cluster[n] = int(best)
    return node_cluster

# ---------------------- cache edges for drawing ----------------------

def sim_to_dist(sim, mode="log", eps=1e-12):
    s = max(eps, min(1.0, float(sim)))
    if mode == "linear":
        return max(0.0, 1.0 - s)
    if mode == "sqrt":
        return math.sqrt(max(0.0, 1.0 - s))
    return -math.log(s)

def read_edges_from_cache(cache_json, plane, min_sim=0.0):
    cache = load_json(cache_json)
    tmap = cache.get("transport_maps", {})
    best = {}
    for key, per in tmap.items():
        if "__to__" not in key:
            continue
        if plane not in per:
            continue
        a, b = key.split("__to__", 1)
        entry = per.get(plane, {})
        sim = entry.get("similarity", entry.get("sim", None))
        if sim is None:
            continue
        sim = float(sim)
        if sim < float(min_sim):
            continue
        u = tuple(sorted([a, b]))
        best[u] = max(best.get(u, -1.0), sim)

    edges = [(u[0], u[1], best[u]) for u in best.keys()]
    edges.sort(key=lambda x: x[2], reverse=True)
    return edges

# ---------------------- projection ----------------------

def pca_project(X, out_dim=3):
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:out_dim].T
    Y = Xc @ W
    return Y, S.tolist(), W.tolist()

def isomap_project(X, out_dim=3, n_neighbors=6):
    try:
        from sklearn.manifold import Isomap
    except Exception:
        return None, "sklearn_missing", None
    iso = Isomap(n_neighbors=n_neighbors, n_components=out_dim)
    Y = iso.fit_transform(X)
    return Y, {"n_neighbors": int(n_neighbors)}, None

# ---------------------- plotting ----------------------

def try_plotly_html(out_html, nodes, coords3, edges, node_color, title):
    try:
        import importlib
        go = importlib.import_module("plotly.graph_objects")
    except Exception:
        return False

    xs = [coords3[n][0] for n in nodes]
    ys = [coords3[n][1] for n in nodes]
    zs = [coords3[n][2] for n in nodes]

    ex, ey, ez = [], [], []
    for a, b, sim in edges:
        if a not in coords3 or b not in coords3:
            continue
        xa, ya, za = coords3[a]
        xb, yb, zb = coords3[b]
        ex += [xa, xb, None]
        ey += [ya, yb, None]
        ez += [za, zb, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines", line=dict(width=2), name="edges"))
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="markers+text",
        text=nodes, textposition="top center",
        marker=dict(size=6, color=node_color),
        name="nodes"
    ))
    fig.update_layout(title=title, margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
    fig.write_html(out_html)
    return True

def plot_png(out_png, nodes, coords3, edges, title):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    xs = [coords3[n][0] for n in nodes]
    ys = [coords3[n][1] for n in nodes]
    zs = [coords3[n][2] for n in nodes]
    ax.scatter(xs, ys, zs, s=70)

    for n in nodes:
        x, y, z = coords3[n]
        ax.text(x, y, z, n, fontsize=8)

    for a, b, sim in edges:
        if a not in coords3 or b not in coords3:
            continue
        xa, ya, za = coords3[a]
        xb, yb, zb = coords3[b]
        ax.plot([xa, xb], [ya, yb], [za, zb], linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

# ---------------------------- main ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--edge_axis_json_lt", required=True)
    ap.add_argument("--edge_axis_json_bo_lr", default=None)
    ap.add_argument("--edge_axis_json_bo", default=None)

    ap.add_argument("--clusters_json_lt", default=None)
    ap.add_argument("--clusters_json_bo_lr", default=None)
    ap.add_argument("--clusters_json_bo", default=None)

    ap.add_argument("--project", choices=["pca", "isomap"], default="pca")
    ap.add_argument("--isomap_k", type=int, default=6)

    ap.add_argument("--min_edge_sim_draw", type=float, default=0.40)
    ap.add_argument("--max_edges_draw", type=int, default=400)

    args = ap.parse_args()
    safe_mkdir(args.outdir)

    # --- Load per-plane coords
    coords_lt, meta_lt = get_coords_from_phase21b(args.edge_axis_json_lt)
    coords_bo_lr, meta_bo_lr = ({}, None)
    coords_bo, meta_bo = ({}, None)

    if args.edge_axis_json_bo_lr and exists(args.edge_axis_json_bo_lr):
        coords_bo_lr, meta_bo_lr = get_coords_from_phase21b(args.edge_axis_json_bo_lr)
    elif args.edge_axis_json_bo_lr:
        print(f"[warn] bo_lr edge_axis_json not found: {args.edge_axis_json_bo_lr}")

    if args.edge_axis_json_bo and exists(args.edge_axis_json_bo):
        coords_bo, meta_bo = get_coords_from_phase21b(args.edge_axis_json_bo)
    elif args.edge_axis_json_bo:
        print(f"[warn] bo edge_axis_json not found: {args.edge_axis_json_bo}")

    # Nodes = union of all coords keys
    nodes = sorted(set(coords_lt.keys()) | set(coords_bo_lr.keys()) | set(coords_bo.keys()))
    if not nodes:
        raise SystemExit("[error] no nodes found in provided edge_axis json files")

    # --- Build stacked vector per node
    # Layout:
    #   lt: x,y
    #   bo: x,y
    #   bo_lr: x,y,z
    def vec_for(n):
        lt = coords_lt.get(n, [0, 0, 0])
        bo = coords_bo.get(n, [0, 0, 0])
        bl = coords_bo_lr.get(n, [0, 0, 0])
        return np.array([lt[0], lt[1], bo[0], bo[1], bl[0], bl[1], bl[2]], dtype=np.float64)

    X = np.vstack([vec_for(n) for n in nodes])
    Xz, mu, sd = zscore(X)

    # --- Project to 3D
    proj_info = {}
    if args.project == "pca":
        Y, S, W = pca_project(Xz, out_dim=3)
        proj_info = {"method": "pca_svd", "singular_values": S, "components": W}
    else:
        Y, iso_info, _ = isomap_project(Xz, out_dim=3, n_neighbors=args.isomap_k)
        if Y is None:
            print("[warn] sklearn not available; falling back to PCA")
            Y, S, W = pca_project(Xz, out_dim=3)
            proj_info = {"method": "pca_svd_fallback", "singular_values": S, "components": W}
        else:
            proj_info = {"method": "isomap", "isomap": iso_info}

    coords3 = {nodes[i]: [float(Y[i, 0]), float(Y[i, 1]), float(Y[i, 2])] for i in range(len(nodes))}

    # --- Build edges to draw (prefer bo_lr if provided else lt)
    plane_for_edges = "bo_lr" if (args.edge_axis_json_bo_lr and exists(args.edge_axis_json_bo_lr)) else "lt"
    edges = read_edges_from_cache(args.cache, plane_for_edges, min_sim=args.min_edge_sim_draw)
    edges = edges[: max(0, int(args.max_edges_draw))]

    # --- Node clustering (optional)
    node_cluster = {}
    cluster_source = None
    # priority bo_lr -> lt -> bo, but only if file exists
    for cand, tag in [
        (args.clusters_json_bo_lr, "bo_lr"),
        (args.clusters_json_lt, "lt"),
        (args.clusters_json_bo, "bo"),
    ]:
        if cand and exists(cand):
            try:
                node_cluster = get_node_cluster_map_from_phase22a(cand)
                cluster_source = cand
                break
            except Exception as e:
                print(f"[warn] failed reading clusters from {cand}: {e}")

    if (args.clusters_json_bo_lr and not exists(args.clusters_json_bo_lr)):
        print(f"[warn] clusters_json_bo_lr missing: {args.clusters_json_bo_lr} (continuing)")
    if (args.clusters_json_lt and not exists(args.clusters_json_lt)):
        print(f"[warn] clusters_json_lt missing: {args.clusters_json_lt} (continuing)")
    if (args.clusters_json_bo and not exists(args.clusters_json_bo)):
        print(f"[warn] clusters_json_bo missing: {args.clusters_json_bo} (continuing)")

    node_color = [int(node_cluster.get(n, -1)) for n in nodes]

    # --- Save json
    out_json = os.path.join(args.outdir, "phase24a_joint_embedding.json")
    out = {
        "phase": "24a",
        "status": "ok",
        "args": vars(args),
        "nodes": nodes,
        "stacked_dim": int(X.shape[1]),
        "stacked_layout": ["lt_x","lt_y","bo_x","bo_y","bo_lr_x","bo_lr_y","bo_lr_z"],
        "stacked_mu": mu,
        "stacked_sd": sd,
        "coords_plane": {
            "lt": coords_lt,
            "bo": coords_bo,
            "bo_lr": coords_bo_lr,
        },
        "projection": {
            "coords3": coords3,
            "info": proj_info,
        },
        "edges_draw_plane": plane_for_edges,
        "edges_draw": [{"a": a, "b": b, "sim": float(sim)} for (a,b,sim) in edges],
        "node_cluster": node_cluster,
        "node_cluster_source": cluster_source,
    }
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)

    # --- Plot outputs
    out_png = os.path.join(args.outdir, "phase24a_joint_3d.png")
    title = f"Phase24a Joint Manifold (stacked→3D via {proj_info.get('method','?')}) | edges={plane_for_edges}"
    plot_png(out_png, nodes, coords3, edges, title)

    out_html = os.path.join(args.outdir, "phase24a_joint_3d.html")
    html_ok = try_plotly_html(out_html, nodes, coords3, edges, node_color, title)

    print("[ok] wrote:", out_json)
    print("[ok] wrote:", out_png)
    if html_ok:
        print("[ok] wrote:", out_html)
    else:
        print("[note] plotly not available; skipped html")

if __name__ == "__main__":
    main()
