#!/usr/bin/env python3
# phase20a_reasoning_graph_stress.py
#
# Phase 20a: Graph-distance -> 3D MDS + Stress
#
# - Build a weighted graph from phase15f transport cache (edges filtered by sim threshold)
# - Convert similarity -> distance using modes: linear, sqrt, log
# - Compute all-pairs shortest path distances
# - Classical MDS to 3D (positive eigen spectrum)
# - Compute Kruskal Stress-1 between shortest-path distances and embedded Euclidean distances
# - Optionally overlay Phase16a holonomy loops on the 3D plot
#
# Examples:
#   python phase20a_reasoning_graph_stress.py ^
#     --cache outputs_edges_relternary256_phase15/phase15f_transport_cache.json ^
#     --outdir outputs_edges_relternary256_phase15/phase20a_bo_lr ^
#     --plane bo_lr --min_edge_sim 0.40 --modes "linear,sqrt,log" ^
#     --draw_transport --annotate ^
#     --loops_json outputs_edges_relternary256_phase15/phase16a_holonomy_bo_lr.json --threshold 0.35 --min_sim 0.35 --top_loops 10 --draw_loops
#
# Notes:
# - Stress is computed over all finite shortest-path pairs (excluding i==j).
# - If the graph is disconnected, stress is computed on the connected pairs only.

import os, json, math, argparse
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------- utilities ----------------------------

def safe_mkdir(p):
    if p and not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def fro_norm(A):
    return float(np.linalg.norm(A, ord="fro"))

def _as_float(x):
    try:
        return float(x)
    except Exception:
        return None

# ---------------------------- Phase16a loader (robust) ----------------------------

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
            best_d = d
            best_k = k
    return best_k

def load_phase16a_loops(in_path, plane_arg=None, threshold=None, min_sim=None, top_k=None):
    with open(in_path, "r") as f:
        data = json.load(f)

    meta = {"format": "unknown"}
    loops = []

    if "threshold_results" in data and isinstance(data["threshold_results"], dict):
        tr = data["threshold_results"]
        chosen_key = pick_threshold_key(tr, threshold)

        if chosen_key is None:
            # default: highest numeric key
            numeric_keys = [(k, _as_float(k)) for k in tr.keys()]
            numeric_keys = [(k, v) for (k, v) in numeric_keys if v is not None]
            if numeric_keys:
                numeric_keys.sort(key=lambda kv: kv[1], reverse=True)
                chosen_key = numeric_keys[0][0]
            else:
                chosen_key = list(tr.keys())[0]

        bucket = tr.get(chosen_key)
        if bucket is None:
            raise SystemExit(f"[error] threshold key not found: {chosen_key}. keys={list(tr.keys())}")

        if isinstance(bucket, dict):
            loops = bucket.get("top", [])
            kept_count = bucket.get("kept_count", len(loops))
            bucket_min_sim = bucket.get("min_sim", None)
        elif isinstance(bucket, list):
            loops = bucket
            kept_count = len(loops)
            bucket_min_sim = None
        else:
            loops = []
            kept_count = 0
            bucket_min_sim = None

        plane = data.get("plane", plane_arg or "unknown")
        meta = {
            "format": "threshold_results",
            "chosen_threshold": threshold,
            "chosen_threshold_key": chosen_key,
            "kept_count": kept_count,
            "total_valid": data.get("total_valid", data.get("all_count", None)),
            "bucket_min_sim": bucket_min_sim,
            "min_sim_arg": min_sim
        }
    else:
        loops = data.get("top", data.get("results", []))
        plane = data.get("plane", plane_arg or (loops[0].get("plane", "unknown") if loops else "unknown"))
        meta = {
            "format": "legacy",
            "total_valid": data.get("all_count", len(loops)),
            "min_sim_arg": min_sim
        }

    if plane_arg:
        loops = [L for L in loops if str(L.get("plane", plane)) == plane_arg]
        plane = plane_arg

    if min_sim is not None:
        loops = [L for L in loops if float(L.get("edge_sim_min", -1e9)) >= float(min_sim)]

    if top_k is not None and top_k > 0:
        loops = loops[:top_k]

    return plane, loops, meta

# ---------------------------- cache -> graph ----------------------------

def sim_from_entry(entry):
    if entry is None:
        return None
    s = entry.get("similarity", None)
    try:
        return float(s)
    except Exception:
        return None

def build_nodes_from_cache(cache):
    names = sorted(list(cache.get("anchors", {}).keys()))
    if names:
        return names
    # fallback: infer from keys
    maps = cache.get("transport_maps", {})
    seen = set()
    for k in maps.keys():
        if "__to__" in k:
            a, b = k.split("__to__")
            seen.add(a); seen.add(b)
    return sorted(list(seen))

def build_graph_edges(cache, plane, min_edge_sim=0.0):
    """
    Returns:
      names: list[str]
      edges: list[(i,j,sim,how)]
      idx: dict[name->i]
    Adds undirected edges (i<->j) using forward maps where available;
    if only reverse exists, still include that as an undirected edge with sim.
    """
    names = build_nodes_from_cache(cache)
    idx = {n:i for i,n in enumerate(names)}
    maps = cache.get("transport_maps", {})

    def add_undirected(a, b, sim, how):
        if a not in idx or b not in idx:
            return
        if sim is None:
            return
        if sim < min_edge_sim:
            return
        i, j = idx[a], idx[b]
        if i == j:
            return
        edges.append((min(i,j), max(i,j), float(sim), str(how)))

    edges = []
    for key, planes in maps.items():
        if "__to__" not in key:
            continue
        a, b = key.split("__to__")
        if plane not in planes:
            continue
        sim = sim_from_entry(planes[plane])
        # store as undirected “available connection”
        add_undirected(a, b, sim, how="forward_or_cached")

    # de-dup (keep max sim if repeated)
    best = {}
    for i, j, sim, how in edges:
        k = (i, j)
        if k not in best or sim > best[k][0]:
            best[k] = (sim, how)
    edges = [(i, j, best[(i,j)][0], best[(i,j)][1]) for (i,j) in best.keys()]
    edges.sort(key=lambda t: (-t[2], t[0], t[1]))

    return names, edges, idx

def sim_to_dist(sim, mode, eps=1e-9):
    sim = float(sim)
    sim = max(eps, min(1.0, sim))
    if mode == "linear":
        return 1.0 - sim
    if mode == "sqrt":
        return math.sqrt(max(0.0, 1.0 - sim))
    if mode == "log":
        return -math.log(sim + eps)
    raise ValueError(f"unknown sim_mode: {mode}")

def all_pairs_shortest_paths(n, edges_undirected, mode):
    """
    Floyd-Warshall on small graphs. n is small (anchors count).
    edges_undirected: [(i,j,sim,how)]
    """
    INF = 1e18
    D = np.full((n, n), INF, dtype=np.float64)
    np.fill_diagonal(D, 0.0)
    for i, j, sim, how in edges_undirected:
        dist = sim_to_dist(sim, mode)
        if dist < D[i, j]:
            D[i, j] = dist
            D[j, i] = dist

    # Floyd-Warshall
    for k in range(n):
        dk = D[:, k][:, None] + D[k, :][None, :]
        D = np.minimum(D, dk)
    return D

# ---------------------------- classical MDS + stress ----------------------------

def classical_mds(D, ndim=3):
    """
    Classical MDS from a distance matrix D (n x n).
    Returns coords (n x ndim_used), eigvals, ndim_used, and Gram matrix B.
    Uses only positive eigenvalues.
    """
    n = D.shape[0]
    # double-center squared distances
    D2 = D**2
    J = np.eye(n) - (1.0/n) * np.ones((n,n))
    B = -0.5 * J @ D2 @ J

    w, V = np.linalg.eigh(B)  # ascending
    order = np.argsort(w)[::-1]
    w = w[order]
    V = V[:, order]

    pos = w > 1e-12
    w_pos = w[pos]
    V_pos = V[:, pos]

    ndim_used = min(ndim, V_pos.shape[1])
    if ndim_used <= 0:
        return np.zeros((n,0)), w.tolist(), 0, B

    L = np.diag(np.sqrt(w_pos[:ndim_used]))
    X = V_pos[:, :ndim_used] @ L
    return X, w.tolist(), ndim_used, B

def pairwise_euclid(X):
    n = X.shape[0]
    G = np.sum(X*X, axis=1)
    D2 = G[:, None] + G[None, :] - 2.0 * (X @ X.T)
    D2 = np.maximum(D2, 0.0)
    return np.sqrt(D2)

def kruskal_stress1(D_target, D_embed, mask=None):
    """
    Stress-1 = sqrt( sum_ij (d_ij - dhat_ij)^2 / sum_ij d_ij^2 )
    excluding diagonal. mask can be boolean matrix of valid pairs.
    """
    n = D_target.shape[0]
    if mask is None:
        mask = np.ones((n,n), dtype=bool)
    mask = mask & (~np.eye(n, dtype=bool))

    dt = D_target[mask]
    de = D_embed[mask]

    num = np.sum((dt - de)**2)
    den = np.sum((dt)**2) + 1e-18
    return float(math.sqrt(num / den))

# ---------------------------- plotting ----------------------------

def plot_3d_embedding(out_png, names, X, annotate=False, loops=None, loop_weight="hol", title=""):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X[:,0], X[:,1], X[:,2], s=60)

    if annotate:
        for i, name in enumerate(names):
            ax.text(X[i,0], X[i,1], X[i,2], " " + name, fontsize=8)

    # loops overlay
    if loops:
        for L in loops:
            nodes = L.get("loop", None)
            if not nodes:
                continue
            # close
            cyc = list(nodes) + [nodes[0]]
            # weight by holonomy
            hol = float(L.get("fro_norm_R_minus_I", 0.0))
            # also allow angle-based weighting if available
            ang = None
            ta = L.get("trace_angle", {})
            if isinstance(ta, dict) and "angle_deg_proxy" in ta:
                ang = float(ta["angle_deg_proxy"])
            w = hol if loop_weight == "hol" else (ang if ang is not None else hol)
            # convert to linewidth (gentle)
            lw = 1.0 + 4.0 * min(1.0, w / (0.5 + 1e-9))

            for a, b in zip(cyc[:-1], cyc[1:]):
                if a not in names or b not in names:
                    continue
                i = names.index(a)
                j = names.index(b)
                xs = [X[i,0], X[j,0]]
                ys = [X[i,1], X[j,1]]
                zs = [X[i,2], X[j,2]]
                ax.plot(xs, ys, zs, linewidth=lw)

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close(fig)

def plot_stress_compare(out_png, rows):
    # rows: list of dicts with mode + stress
    modes = [r["mode"] for r in rows]
    stress = [r["stress1"] for r in rows]

    plt.figure()
    plt.plot(modes, stress, marker="o")
    plt.ylabel("Kruskal Stress-1 (lower is better)")
    plt.xlabel("sim_mode")
    plt.title("Phase20a: Stress by sim_mode")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

# ---------------------------- main ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="phase15f_transport_cache.json")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--plane", default="bo_lr", choices=["bo","lt","bo_lr"])
    ap.add_argument("--min_edge_sim", type=float, default=0.40, help="Edge include threshold")
    ap.add_argument("--modes", default="linear,sqrt,log", help="Comma list: linear,sqrt,log")
    ap.add_argument("--ndim", type=int, default=3, help="MDS output dims")
    ap.add_argument("--draw_transport", action="store_true", help="(kept for compatibility; transport always used)")
    ap.add_argument("--annotate", action="store_true")

    # loops overlay (optional)
    ap.add_argument("--loops_json", default="", help="Phase16a holonomy JSON (legacy or threshold_results)")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--min_sim", type=float, default=None)
    ap.add_argument("--top_loops", type=int, default=10)
    ap.add_argument("--draw_loops", action="store_true")
    args = ap.parse_args()

    safe_mkdir(args.outdir)

    with open(args.cache, "r") as f:
        cache = json.load(f)

    names, edges, idx = build_graph_edges(cache, args.plane, min_edge_sim=args.min_edge_sim)
    n = len(names)
    if n < 3:
        raise SystemExit(f"[error] need >=3 nodes for MDS; got {n}")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    valid_modes = {"linear","sqrt","log"}
    for m in modes:
        if m not in valid_modes:
            raise SystemExit(f"[error] unknown mode: {m}. allowed={sorted(list(valid_modes))}")

    # load loops (if provided)
    loops = None
    loops_meta = None
    if args.loops_json.strip():
        plane_arg = args.plane
        p2, loops, loops_meta = load_phase16a_loops(
            args.loops_json,
            plane_arg=plane_arg,
            threshold=args.threshold,
            min_sim=args.min_sim,
            top_k=args.top_loops
        )
        if not loops:
            loops = None

    results = []
    best = None  # (stress, mode, X, Dsp, Deuclid, eigvals, ndim_used)
    for mode in modes:
        Dsp = all_pairs_shortest_paths(n, edges, mode=mode)

        # mask finite pairs
        finite = np.isfinite(Dsp) & (Dsp < 1e17)
        # if graph disconnected, we still embed using a filled distance matrix:
        # For classical MDS we need a full matrix; fill INF with a large cap based on finite max.
        Dfill = Dsp.copy()
        finite_vals = Dfill[finite]
        if finite_vals.size == 0:
            continue
        dmax = float(np.max(finite_vals))
        cap = dmax * 1.25 + 1e-6
        Dfill[~finite] = cap

        X, eigvals, ndim_used, B = classical_mds(Dfill, ndim=args.ndim)
        if ndim_used < 3:
            # pad to 3 for plotting
            if X.shape[1] == 0:
                X = np.zeros((n,3), dtype=np.float64)
            elif X.shape[1] == 1:
                X = np.hstack([X, np.zeros((n,2))])
            elif X.shape[1] == 2:
                X = np.hstack([X, np.zeros((n,1))])
        elif X.shape[1] > 3:
            X = X[:, :3]

        De = pairwise_euclid(X)

        # stress computed only on finite shortest-path pairs
        stress = kruskal_stress1(Dsp, De, mask=finite)

        row = {
            "mode": mode,
            "stress1": float(stress),
            "ndim_used_positive": int(min(args.ndim, max(0, sum(np.array(eigvals) > 1e-12)))),
            "eigvals": eigvals[:min(10, len(eigvals))],
            "nodes": n,
            "edges": len(edges),
            "min_edge_sim": float(args.min_edge_sim),
            "plane": args.plane,
        }
        results.append(row)

        if best is None or stress < best[0]:
            best = (stress, mode, X, Dsp, De, eigvals, ndim_used)

    if not results:
        raise SystemExit("[error] no modes produced results (graph may be empty/disconnected)")

    # save CSV
    csv_path = os.path.join(args.outdir, f"phase20a_stress_{args.plane}.csv")
    with open(csv_path, "w") as g:
        g.write("mode,stress1,nodes,edges,min_edge_sim\n")
        for r in results:
            g.write(f"{r['mode']},{r['stress1']:.8f},{r['nodes']},{r['edges']},{r['min_edge_sim']}\n")

    # save JSON
    best_stress, best_mode, best_X, best_Dsp, best_De, best_eig, best_nd = best
    out_json = {
        "phase": "20a",
        "plane": args.plane,
        "status": "ok",
        "args": {
            "cache": args.cache,
            "outdir": args.outdir,
            "plane": args.plane,
            "min_edge_sim": args.min_edge_sim,
            "modes": modes,
            "ndim": args.ndim,
            "annotate": bool(args.annotate),
            "loops_json": args.loops_json if args.loops_json.strip() else None,
            "threshold": args.threshold,
            "min_sim": args.min_sim,
            "top_loops": args.top_loops,
            "draw_loops": bool(args.draw_loops),
        },
        "graph": {
            "nodes": n,
            "edges": len(edges),
            "names": names,
            "edges_list": [
                {"a": names[i], "b": names[j], "sim": float(sim), "how": how}
                for (i,j,sim,how) in edges
            ]
        },
        "stress_compare": results,
        "best": {
            "mode": best_mode,
            "stress1": float(best_stress),
            "mds": {
                "eigvals": best_eig[:min(10, len(best_eig))],
                "coords": {names[i]: best_X[i,:3].tolist() for i in range(n)},
            }
        },
        "loops_meta": loops_meta if loops_meta is not None else None,
    }

    json_path = os.path.join(args.outdir, f"phase20a_stress_{args.plane}.json")
    with open(json_path, "w") as g:
        json.dump(out_json, g, indent=2)

    # plots
    stress_png = os.path.join(args.outdir, f"phase20a_stress_compare_{args.plane}.png")
    plot_stress_compare(stress_png, results)

    embed_png = os.path.join(args.outdir, f"phase20a_embed3d_best_{args.plane}_{best_mode}.png")
    loops_to_draw = loops if (args.draw_loops and loops is not None) else None
    title = f"Phase20a  plane={args.plane}  best_mode={best_mode}  stress={best_stress:.4f}"
    plot_3d_embedding(embed_png, names, best_X[:, :3], annotate=args.annotate, loops=loops_to_draw, title=title)

    print(f"[phase20a] plane={args.plane} nodes={n} edges={len(edges)}")
    for r in sorted(results, key=lambda x: x["stress1"]):
        print(f"  mode={r['mode']:<6} stress1={r['stress1']:.6f}")
    print(f"[best] mode={best_mode} stress1={best_stress:.6f}")
    print(f"[saved] {json_path}")
    print(f"[saved] {csv_path}")
    print(f"[saved] {stress_png}")
    print(f"[saved] {embed_png}")

if __name__ == "__main__":
    main()
