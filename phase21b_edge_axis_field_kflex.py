#!/usr/bin/env python3
# phase21b_edge_axis_field_kflex.py
#
# Phase 21b (k-flex): build an "edge axis field" from holonomy loops for ANY plane:
#   - lt / bo => k=2 (2x2 R_loop)
#   - bo_lr   => k=3 (3x3 R_loop)
#
# Fixes in this drop-in:
#   - Robust transport similarity key support: similarity / sim / edge_sim
#   - Reverse-edge fallback for similarity lookup (b__to__a if a__to__b missing)
#   - Diagnostics when edges are unexpectedly empty
#
# Outputs:
#   - phase21b_edge_axis_field_<plane>.json
#   - phase21b_edge_axis_field_<plane>.png

import os, json, math, argparse
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------- utils ----------------------------

def safe_mkdir(p):
    if p and not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def _as_float(x):
    try:
        return float(x)
    except Exception:
        return None

def normalize(v, eps=1e-12):
    v = np.array(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < eps:
        return v * 0.0
    return v / n

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
            best_d, best_k = d, k
    return best_k

def load_phase16a_loops(path, plane_arg=None, threshold=None, min_sim=None):
    with open(path, "r") as f:
        data = json.load(f)

    if "threshold_results" in data and isinstance(data["threshold_results"], dict):
        tr = data["threshold_results"]
        chosen_key = pick_threshold_key(tr, threshold)
        if chosen_key is None:
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
            loops, kept_count, bucket_min_sim = [], 0, None

        plane = data.get("plane", plane_arg or "unknown")
        meta = {
            "format": "threshold_results",
            "chosen_threshold": threshold,
            "chosen_threshold_key": chosen_key,
            "kept_count": kept_count,
            "total_valid": data.get("total_valid", data.get("all_count", None)),
            "bucket_min_sim": bucket_min_sim,
            "min_sim_arg": min_sim,
        }
    else:
        loops = data.get("top", data.get("results", []))
        plane = data.get("plane", plane_arg or (loops[0].get("plane", "unknown") if loops else "unknown"))
        meta = {
            "format": "legacy",
            "total_valid": data.get("all_count", len(loops)),
            "min_sim_arg": min_sim,
        }

    if plane_arg:
        loops = [L for L in loops if str(L.get("plane", plane)) == plane_arg]
        plane = plane_arg

    if min_sim is not None:
        loops = [L for L in loops if float(L.get("edge_sim_min", -1e9)) >= float(min_sim)]

    return plane, loops, meta

# ---------------------- rotations ----------------------

def rotation_angle_from_trace(R):
    k = R.shape[0]
    tr = float(np.trace(R))
    if k == 2:
        c = max(-1.0, min(1.0, tr / 2.0))
        return math.acos(c)
    if k == 3:
        c = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
        return math.acos(c)
    c = max(-1.0, min(1.0, tr / k))
    return math.acos(c)

def axis_angle_from_R3(R):
    eps = 1e-9
    theta = rotation_angle_from_trace(R)
    if abs(theta) < 1e-8:
        return 0.0, [1.0, 0.0, 0.0]
    S = (R - R.T) / (2.0 * math.sin(theta) + eps)
    axis = np.array([S[2,1], S[0,2], S[1,0]], dtype=np.float64)
    n = float(np.linalg.norm(axis))
    if n < 1e-8:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        n = 1.0
    axis = (axis / n).tolist()
    return float(theta), axis

def angle_from_R2(R):
    return float(math.atan2(float(R[1,0]), float(R[0,0])))

# ---------------------- graph + embedding ----------------------

def sim_to_dist(sim, mode="log", eps=1e-12):
    s = max(eps, min(1.0, float(sim)))
    if mode == "linear":
        return max(0.0, 1.0 - s)
    if mode == "sqrt":
        return math.sqrt(max(0.0, 1.0 - s))
    return -math.log(s)

def all_pairs_shortest_path(dist_mat):
    D = dist_mat.copy()
    n = D.shape[0]
    for k in range(n):
        Dk = D[:, k].reshape(n, 1) + D[k, :].reshape(1, n)
        D = np.minimum(D, Dk)
    return D

def classical_mds(D, ndim=3):
    n = D.shape[0]
    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1]
    w = w[idx]
    V = V[:, idx]
    pos = [(i, w[i]) for i in range(n) if w[i] > 1e-12]
    ndim_use = min(ndim, len(pos))
    if ndim_use == 0:
        return w.tolist(), 0, np.zeros((n, ndim), dtype=np.float64)
    L = np.diag([w[i] for (i, _) in pos[:ndim_use]])
    X = V[:, [i for (i, _) in pos[:ndim_use]]] @ np.sqrt(L)
    if ndim_use < ndim:
        X = np.pad(X, ((0, 0), (0, ndim - ndim_use)))
    return w.tolist(), ndim_use, X

# ---------------------- transport sim handling (FIX) ----------------------

def sim_from_plane_entry(pe):
    if not isinstance(pe, dict):
        return None
    for k in ("similarity", "sim", "edge_sim"):
        v = pe.get(k, None)
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if math.isnan(fv) or math.isinf(fv):
            continue
        return fv
    return None

def diag_transport(cache, plane):
    tmap = cache.get("transport_maps", {})
    n_plane = 0
    key_hits = defaultdict(int)
    sims = []
    for _, per in tmap.items():
        pe = per.get(plane, None) if isinstance(per, dict) else None
        if not isinstance(pe, dict):
            continue
        n_plane += 1
        for k in ("similarity", "sim", "edge_sim"):
            if k in pe:
                key_hits[k] += 1
        s = sim_from_plane_entry(pe)
        if s is not None:
            sims.append(s)
    print(f"[diag] transport_maps plane='{plane}': entries={n_plane} keys={dict(key_hits)}")
    if sims:
        print(f"[diag] sim stats: n={len(sims)} min={min(sims):.6f} max={max(sims):.6f}")

# ---------------------------- main ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--loops_json", required=True)
    ap.add_argument("--plane", required=True, choices=["bo", "lt", "bo_lr"])
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--min_sim", type=float, default=None)
    ap.add_argument("--min_edge_sim", type=float, default=0.40)
    ap.add_argument("--sim_mode", default="log", choices=["linear", "sqrt", "log"])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--annotate", action="store_true")
    ap.add_argument("--max_edges_draw", type=int, default=300)
    ap.add_argument("--tick_scale", type=float, default=0.1)
    ap.add_argument("--top_loops", type=int, default=None)
    args = ap.parse_args()

    safe_mkdir(args.outdir)

    cache = json.load(open(args.cache, "r"))
    tmap = cache.get("transport_maps", {})

    plane, loops, loops_meta = load_phase16a_loops(
        args.loops_json,
        plane_arg=args.plane,
        threshold=args.threshold,
        min_sim=args.min_sim,
    )

    k_expected = 3 if args.plane == "bo_lr" else 2

    usable = []
    for L in loops:
        Rl = L.get("R_loop", None)
        if Rl is None:
            continue
        R = np.array(Rl, dtype=np.float64)
        if R.ndim != 2 or R.shape[0] != R.shape[1]:
            continue
        if R.shape[0] != k_expected:
            continue
        usable.append(L)

    if args.top_loops is not None:
        usable = usable[: max(0, int(args.top_loops))]

    if not usable:
        raise SystemExit(f"[error] No usable k={k_expected} loops with R_loop found (plane={args.plane}).")

    # nodes from loops
    node_set = set()
    for L in usable:
        for n in L.get("loop", []):
            node_set.add(n)
    nodes = sorted(node_set)
    name_to_i = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    # robust similarity lookup with reverse fallback
    def lookup_sim(a, b):
        key = f"{a}__to__{b}"
        per = tmap.get(key, None)
        if isinstance(per, dict):
            pe = per.get(args.plane, None)
            s = sim_from_plane_entry(pe)
            if s is not None:
                return s

        # reverse fallback (still useful for connectivity + inclusion)
        key2 = f"{b}__to__{a}"
        per2 = tmap.get(key2, None)
        if isinstance(per2, dict):
            pe2 = per2.get(args.plane, None)
            s2 = sim_from_plane_entry(pe2)
            if s2 is not None:
                return s2

        return None

    # adjacency distance matrix for shortest path
    INF = 1e18
    dist = np.full((n, n), INF, dtype=np.float64)
    np.fill_diagonal(dist, 0.0)

    edges_in_cache_filtered = 0
    for a in nodes:
        for b in nodes:
            if a == b:
                continue
            sim = lookup_sim(a, b)
            if sim is None:
                continue
            if sim < args.min_edge_sim:
                continue
            dij = sim_to_dist(sim, mode=args.sim_mode)
            ia, ib = name_to_i[a], name_to_i[b]
            if dij < dist[ia, ib]:
                dist[ia, ib] = dij
            edges_in_cache_filtered += 1

    if edges_in_cache_filtered == 0:
        print("[warn] edges_in_cache_filtered == 0. This usually means sim key mismatch or no sims for this plane.")
        diag_transport(cache, args.plane)

    sp = all_pairs_shortest_path(dist)
    eigvals, ndim_pos, X = classical_mds(sp, ndim=3)
    coords = {nodes[i]: [float(X[i, 0]), float(X[i, 1]), float(X[i, 2])] for i in range(n)}

    # global reference axis
    if k_expected == 3:
        acc = np.zeros(3, dtype=np.float64)
        for L in usable:
            R = np.array(L["R_loop"], dtype=np.float64)
            theta, axis = axis_angle_from_R3(R)
            ang_deg = abs(math.degrees(theta))
            acc += normalize(axis) * ang_deg
        gref = normalize(acc) if float(np.linalg.norm(acc)) > 0 else np.array([1.0, 0.0, 0.0])
        global_ref_axis = [float(gref[0]), float(gref[1]), float(gref[2])]
    else:
        global_ref_axis = [0.0, 0.0, 1.0]

    edge_acc = {}
    def ensure_edge(a, b, sim):
        k = (a, b)
        if k not in edge_acc:
            edge_acc[k] = {
                "a": a, "b": b,
                "sim": float(sim) if sim is not None else float("nan"),
                "hits": 0,
                "curv_abs": 0.0,
                "curv_signed": 0.0,
                "axes": [],
            }
        return edge_acc[k]

    loop_summaries = []

    for L in usable:
        loop_nodes = L.get("loop", [])
        if len(loop_nodes) < 2:
            continue
        R = np.array(L["R_loop"], dtype=np.float64)

        if k_expected == 3:
            theta, axis = axis_angle_from_R3(R)
            ang_deg_raw = float(math.degrees(theta))
            axis_u = normalize(axis)
            sgn = 1.0 if float(np.dot(axis_u, global_ref_axis)) >= 0 else -1.0
            axis_vec = axis_u.tolist()
            ang_abs = abs(ang_deg_raw)
        else:
            theta = angle_from_R2(R)
            ang_deg_raw = float(math.degrees(theta))
            sgn = 1.0 if ang_deg_raw >= 0 else -1.0
            axis_vec = [0.0, 0.0, float(sgn)]
            ang_abs = abs(ang_deg_raw)

        loop_summaries.append({
            "loop": loop_nodes,
            "rank_score": float(L.get("rank_score", float("nan"))),
            "edge_sim_min": float(L.get("edge_sim_min", float("nan"))),
            "edge_sim_mean": float(L.get("edge_sim_mean", float("nan"))),
            "angle_deg": float(ang_abs),
            "axis": [float(axis_vec[0]), float(axis_vec[1]), float(axis_vec[2])],
        })

        cyc = loop_nodes + [loop_nodes[0]]
        for i in range(len(loop_nodes)):
            a = cyc[i]
            b = cyc[i + 1]
            if a not in name_to_i or b not in name_to_i:
                continue
            sim = lookup_sim(a, b)
            if sim is None or sim < args.min_edge_sim:
                continue

            E = ensure_edge(a, b, sim)
            E["hits"] += 1
            E["curv_abs"] += ang_abs
            E["curv_signed"] += (sgn * ang_abs)
            E["axes"].append(axis_vec)

    edge_axis_field = []
    for (a, b), E in edge_acc.items():
        axes = np.array(E["axes"], dtype=np.float64)
        if axes.shape[0] == 0:
            continue
        mean_axis = normalize(np.mean(axes, axis=0))
        consistency = float(np.linalg.norm(np.mean(axes, axis=0)))
        edge_axis_field.append({
            "a": a,
            "b": b,
            "sim": float(E["sim"]),
            "hits": int(E["hits"]),
            "curv_abs": float(E["curv_abs"]),
            "curv_signed": float(E["curv_signed"]),
            "axis_mean": [float(mean_axis[0]), float(mean_axis[1]), float(mean_axis[2])],
            "axis_consistency_proxy": float(consistency),
        })

    node_abs = defaultdict(float)
    node_signed = defaultdict(float)
    for e in edge_axis_field:
        node_abs[e["a"]] += e["curv_abs"]
        node_abs[e["b"]] += e["curv_abs"]
        node_signed[e["a"]] += e["curv_signed"]
        node_signed[e["b"]] += e["curv_signed"]

    node_scores = [{"name": nm, "curv_abs": float(node_abs[nm]), "curv_signed": float(node_signed[nm])} for nm in nodes]

    out_json = os.path.join(args.outdir, f"phase21b_edge_axis_field_{args.plane}.json")
    out_png  = os.path.join(args.outdir, f"phase21b_edge_axis_field_{args.plane}.png")

    out = {
        "phase": "21b",
        "plane": args.plane,
        "status": "ok",
        "args": vars(args),
        "loops_meta": loops_meta,
        "embedding": {
            "method": "classical_mds_shortest_path",
            "sim_mode": args.sim_mode,
            "eigvals": eigvals,
            "ndim_used_positive": int(ndim_pos),
            "coords": coords,
        },
        "global_reference_axis": [float(x) for x in global_ref_axis],
        "counts": {
            "anchors": int(n),
            "loops_used": int(len(usable)),
            "edges_in_cache_filtered": int(edges_in_cache_filtered),
            "edges_with_curvature": int(len(edge_axis_field)),
        },
        "loop_summaries": loop_summaries,
        "edge_axis_field": edge_axis_field,
        "node_scores": node_scores,
        "paths": {"png": out_png},
        "note": f"k={k_expected}; k=2 uses pseudo axis [0,0,sign(theta)]."
    }

    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)

    # ---------------- plot ----------------
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    xs = [coords[nm][0] for nm in nodes]
    ys = [coords[nm][1] for nm in nodes]
    zs = [coords[nm][2] for nm in nodes]
    ax.scatter(xs, ys, zs)

    if args.annotate:
        for nm in nodes:
            x, y, z = coords[nm]
            ax.text(x, y, z, nm, fontsize=8)

    edge_axis_field_sorted = sorted(edge_axis_field, key=lambda e: e["curv_abs"], reverse=True)
    draw_edges = edge_axis_field_sorted[: max(0, int(args.max_edges_draw))]

    for e in draw_edges:
        a, b = e["a"], e["b"]
        xa, ya, za = coords[a]
        xb, yb, zb = coords[b]
        ax.plot([xa, xb], [ya, yb], [za, zb], linewidth=1)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Phase21b k-flex edge axis field ({args.plane}, sim_mode={args.sim_mode})")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)

    print(f"[ok] plane={args.plane} k={k_expected} loops={len(usable)} edges={len(edge_axis_field)}")
    print(f"[counts] edges_in_cache_filtered={edges_in_cache_filtered} min_edge_sim={args.min_edge_sim}")
    print(f"[saved] {out_json}")
    print(f"[saved] {out_png}")

if __name__ == "__main__":
    main()
