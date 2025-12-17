#!/usr/bin/env python3
# phase19a_reasoning_graph_embed_3d.py
#
# Phase 19a: Reasoning graph embedding in 3D (robust MDS).
# - Builds a graph from Phase15f transport cache (anchors + transport_maps)
# - Converts similarity -> distance (configurable sim_mode)
# - Computes all-pairs shortest paths to get a graph distance matrix
# - Runs classical MDS robustly:
#     * keeps ONLY positive eigenvalues (>= eps)
#     * uses up to 3 dimensions from the positive spectrum
#     * reports eigenvalues + counts
# - Optionally overlays holonomy loops from Phase16a (legacy or threshold_results)
#
# Example:
#   python phase19a_reasoning_graph_embed_3d.py \
#     --cache outputs_edges_relternary256_phase15/phase15f_transport_cache.json \
#     --outdir outputs_edges_relternary256_phase15/phase19a_bo_lr \
#     --plane bo_lr --min_edge_sim 0.40 --edge_metric sim --sim_mode sqrt \
#     --draw_transport --annotate \
#     --loops_json outputs_edges_relternary256_phase15/phase16a_holonomy_bo_lr.json \
#     --threshold 0.35 --min_sim 0.35 --top_loops 10 --draw_loops

import os, json, math, argparse
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------- small utils ----------------------------

def safe_mkdir(p):
    if p and not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def _as_float(x):
    try:
        return float(x)
    except Exception:
        return None

def sim_to_dist(sim, sim_mode="linear", eps=1e-9):
    """
    Convert similarity in [0,1] (approximately) to a nonnegative distance.
    sim_mode:
      - linear: dist = 1 - sim
      - sqrt:   dist = sqrt(max(0, 1 - sim))
      - log:    dist = -log(max(eps, sim))
    """
    if sim is None or (isinstance(sim, float) and np.isnan(sim)):
        return float("inf")
    s = float(sim)
    if sim_mode == "log":
        return float(-math.log(max(eps, s)))
    # linear / sqrt use (1 - sim) baseline
    t = max(0.0, 1.0 - s)
    if sim_mode == "sqrt":
        return float(math.sqrt(t))
    return float(t)

# ---------------------------- load loops overlay ----------------------------

def pick_threshold_key(threshold_results: dict, threshold: float):
    if threshold is None:
        return None
    s = f"{threshold:.2f}"
    if s in threshold_results:
        return s
    s2 = str(threshold)
    if s2 in threshold_results:
        return s2
    best_k, best_d = None, None
    for k in threshold_results.keys():
        fk = _as_float(k)
        if fk is None:
            continue
        d = abs(fk - threshold)
        if best_d is None or d < best_d:
            best_d, best_k = d, k
    return best_k

def load_phase16a_loops(path, threshold=None, min_sim=None, top_loops=10):
    """
    Supports:
      legacy: {"plane":"bo_lr","top":[...],"all_count":N}
      threshold_results:
        {"plane":"bo_lr","threshold_results":{"0.35":{"top":[...],...},...}, "total_valid":N}
    Returns: (loops_list, meta_dict)
    """
    if not path:
        return [], {}
    with open(path, "r") as f:
        data = json.load(f)

    meta = {}
    loops = []

    if "threshold_results" in data and isinstance(data["threshold_results"], dict):
        tr = data["threshold_results"]
        chosen_key = pick_threshold_key(tr, threshold)
        if chosen_key is None:
            # choose highest numeric if possible
            numeric_keys = [(k, _as_float(k)) for k in tr.keys()]
            numeric_keys = [(k, v) for (k, v) in numeric_keys if v is not None]
            if numeric_keys:
                numeric_keys.sort(key=lambda kv: kv[1], reverse=True)
                chosen_key = numeric_keys[0][0]
            else:
                chosen_key = list(tr.keys())[0]

        bucket = tr.get(chosen_key, {})
        if isinstance(bucket, dict):
            loops = list(bucket.get("top", []))
            kept = bucket.get("kept_count", len(loops))
            bucket_min_sim = bucket.get("min_sim", None)
        elif isinstance(bucket, list):
            loops = list(bucket)
            kept = len(loops)
            bucket_min_sim = None
        else:
            loops = []
            kept = 0
            bucket_min_sim = None

        meta = {
            "format": "threshold_results",
            "chosen_threshold": threshold,
            "chosen_threshold_key": chosen_key,
            "kept_count": kept,
            "total_valid": data.get("total_valid", data.get("all_count", None)),
            "bucket_min_sim": bucket_min_sim,
            "min_sim_arg": min_sim,
        }
    else:
        loops = list(data.get("top", data.get("results", [])))
        meta = {
            "format": "legacy",
            "total_valid": data.get("all_count", len(loops)),
            "min_sim_arg": min_sim,
        }

    if min_sim is not None:
        loops = [L for L in loops if float(L.get("edge_sim_min", -1e9)) >= float(min_sim)]

    loops = loops[:max(0, int(top_loops))] if top_loops is not None else loops
    return loops, meta

# ---------------------------- graph construction ----------------------------

def build_graph_from_cache(cache, plane, min_edge_sim=0.0):
    """
    Returns:
      names: list[str] anchor names
      edges: list[(i,j,sim,dist,how,a,b)] where i,j index into names
      dist_mat: NxN with inf for missing (0 on diagonal), using direct edge dist
    """
    anchors = cache.get("anchors", {})
    names = sorted(list(anchors.keys()))
    name_to_i = {n: idx for idx, n in enumerate(names)}
    N = len(names)

    maps = cache.get("transport_maps", {})
    edges = []

    def add_edge(a, b, sim, dist, how):
        # robust: always compute dist if None
        if dist is None:
            dist = sim_to_dist(sim, sim_mode="linear")  # placeholder; caller usually overwrites later
        i = name_to_i.get(a, None)
        j = name_to_i.get(b, None)
        if i is None or j is None:
            return
        edges.append((i, j, float(sim), float(dist), how, a, b))

    # collect all directed edges that exist for this plane
    for key, by_plane in maps.items():
        if "__to__" not in key:
            continue
        if not isinstance(by_plane, dict):
            continue
        if plane not in by_plane:
            continue
        entry = by_plane[plane]
        if not isinstance(entry, dict):
            continue
        sim = entry.get("similarity", None)
        if sim is None:
            continue
        if float(sim) < float(min_edge_sim):
            continue
        a, b = key.split("__to__")
        add_edge(a, b, sim, dist=None, how="forward_raw")

    # init direct distance matrix
    dist_mat = np.full((N, N), np.inf, dtype=np.float64)
    np.fill_diagonal(dist_mat, 0.0)

    return names, edges, dist_mat

def all_pairs_shortest_paths(dist_mat):
    """
    Floyd-Warshall (N is small for anchors; super stable and simple).
    """
    D = dist_mat.copy()
    N = D.shape[0]
    for k in range(N):
        # vectorized "min(D, D[:,k]+D[k,:])"
        D = np.minimum(D, D[:, [k]] + D[[k], :])
    return D

# ---------------------------- robust classical MDS ----------------------------

def classical_mds(D, ndim=3, eps=1e-12):
    """
    Classical MDS from a distance matrix D.
    Returns:
      coords: (N, ndim_used)
      eigvals_sorted: list float (descending)
      ndim_used: int
    """
    D = np.array(D, dtype=np.float64)
    N = D.shape[0]

    # Replace any remaining inf distances with a large finite value (so MDS can proceed).
    finite = D[np.isfinite(D)]
    if finite.size == 0:
        raise ValueError("Distance matrix has no finite entries.")
    big = float(np.max(finite)) * 1.25 + 1e-6
    D2 = D.copy()
    D2[~np.isfinite(D2)] = big

    # Double-centering: B = -0.5 * J * D^2 * J
    J = np.eye(N) - (1.0 / N) * np.ones((N, N), dtype=np.float64)
    B = -0.5 * J @ (D2 ** 2) @ J

    # Symmetrize for numerical stability
    B = 0.5 * (B + B.T)

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(B)  # ascending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    eigvals_list = [float(x) for x in eigvals.tolist()]

    # Keep positive eigenvalues only
    pos_mask = eigvals > float(eps)
    pos_vals = eigvals[pos_mask]
    pos_vecs = eigvecs[:, pos_mask]

    ndim_used = min(int(ndim), int(pos_vals.shape[0]))
    if ndim_used <= 0:
        # No positive spectrum: return zeros
        return np.zeros((N, 1), dtype=np.float64), eigvals_list, 0

    L = np.diag(np.sqrt(pos_vals[:ndim_used]))
    V = pos_vecs[:, :ndim_used]
    X = V @ L
    return X, eigvals_list, ndim_used

# ---------------------------- plotting ----------------------------

def plot_3d(coords3, names, out_png, annotate=False, loops=None):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    xs, ys, zs = coords3[:, 0], coords3[:, 1], coords3[:, 2]
    ax.scatter(xs, ys, zs)

    if annotate:
        for i, name in enumerate(names):
            ax.text(xs[i], ys[i], zs[i], name, fontsize=8)

    # Optional loop overlay: draw polyline through the anchors in loop order
    if loops:
        name_to_i = {n: i for i, n in enumerate(names)}
        for L in loops:
            nodes = L.get("loop", None)
            if not nodes or not isinstance(nodes, list):
                continue
            idxs = []
            ok = True
            for n in nodes + [nodes[0]]:
                if n not in name_to_i:
                    ok = False
                    break
                idxs.append(name_to_i[n])
            if not ok:
                continue
            P = coords3[np.array(idxs, dtype=np.int64)]
            ax.plot(P[:, 0], P[:, 1], P[:, 2], linewidth=1.0)

    ax.set_title("Phase19a: Reasoning Graph Embed (MDS)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

# ---------------------------- main ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="phase15f_transport_cache.json")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--plane", default="bo_lr", choices=["bo", "lt", "bo_lr"])
    ap.add_argument("--min_edge_sim", type=float, default=0.40)
    ap.add_argument("--edge_metric", default="sim", choices=["sim", "dist"], help="Transport edge weight source.")
    ap.add_argument("--sim_mode", default="sqrt", choices=["linear", "sqrt", "log"], help="Similarity->distance mapping.")
    ap.add_argument("--draw_transport", action="store_true")
    ap.add_argument("--annotate", action="store_true")
    ap.add_argument("--loops_json", default="", help="Phase16a output to overlay loops.")
    ap.add_argument("--threshold", type=float, default=None, help="If loops_json is threshold_results, choose this threshold.")
    ap.add_argument("--min_sim", type=float, default=None, help="Loop filter: edge_sim_min >= min_sim.")
    ap.add_argument("--top_loops", type=int, default=10)
    ap.add_argument("--draw_loops", action="store_true")
    args = ap.parse_args()

    safe_mkdir(args.outdir)

    with open(args.cache, "r") as f:
        cache = json.load(f)

    names, edges_raw, dist_mat = build_graph_from_cache(cache, args.plane, min_edge_sim=args.min_edge_sim)

    # Convert edges to chosen metric distances
    # If edge_metric == "dist" and cache ever contains dist, you'd read it; for now we treat dist as derived.
    edges = []
    for (i, j, sim, _dist_placeholder, how, a, b) in edges_raw:
        if args.edge_metric == "sim":
            dist = sim_to_dist(sim, sim_mode=args.sim_mode)
        else:
            # no native dist field in cache entries here; fallback to sim->dist too
            dist = sim_to_dist(sim, sim_mode=args.sim_mode)
        edges.append((i, j, float(sim), float(dist), how, a, b))

        # keep the *best* (smallest) direct distance if multiple edges exist
        if dist < dist_mat[i, j]:
            dist_mat[i, j] = dist

    N = len(names)
    if N < 3:
        raise SystemExit("[error] Not enough anchors to embed (need >=3).")

    # all-pairs shortest path for a graph distance
    Dsp = all_pairs_shortest_paths(dist_mat)

    # Robust classical MDS (use only positive eigvals)
    X, eigvals_list, ndim_used = classical_mds(Dsp, ndim=3, eps=1e-12)

    # Ensure 3 columns for plotting (pad with zeros if ndim_used < 3)
    if X.shape[1] < 3:
        X = np.pad(X, ((0, 0), (0, 3 - X.shape[1])), mode="constant")

    loops = []
    loops_meta = {}
    if args.loops_json and args.draw_loops:
        loops, loops_meta = load_phase16a_loops(
            args.loops_json,
            threshold=args.threshold,
            min_sim=args.min_sim,
            top_loops=args.top_loops
        )

    out_png = os.path.join(args.outdir, f"phase19a_reasoning_graph_embed_{args.plane}.png")
    out_json = os.path.join(args.outdir, f"phase19a_reasoning_graph_embed_{args.plane}.json")

    # Save JSON
    coords = {names[i]: [float(X[i, 0]), float(X[i, 1]), float(X[i, 2])] for i in range(N)}
    out = {
        "phase": "19a",
        "plane": args.plane,
        "status": "ok",
        "args": vars(args),
        "counts": {
            "anchors": N,
            "transport_edges": len(edges),
            "loops_overlayed": len(loops) if (args.draw_loops and args.loops_json) else 0
        },
        "mds": {
            "eigvals": eigvals_list,
            "ndim_used_positive": int(ndim_used),
            "coords": coords
        },
        "loops_meta": loops_meta
    }
    with open(out_json, "w") as g:
        json.dump(out, g, indent=2)

    # Plot
    plot_3d(X, names, out_png, annotate=args.annotate, loops=loops if args.draw_loops else None)

    # Console summary (includes the key thing you asked about)
    pos_count = sum(1 for v in eigvals_list if v > 1e-12)
    neg_count = sum(1 for v in eigvals_list if v < -1e-12)
    near0_count = len(eigvals_list) - pos_count - neg_count
    print(f"[plane={args.plane}] anchors={N} edges={len(edges)} min_edge_sim={args.min_edge_sim}")
    print(f"[MDS] eigvals(desc)={eigvals_list}")
    print(f"[MDS] positive={pos_count} near0={near0_count} negative={neg_count}  ndim_used={ndim_used}")
    print(f"[saved] {out_json}")
    print(f"[saved] {out_png}")

if __name__ == "__main__":
    main()
