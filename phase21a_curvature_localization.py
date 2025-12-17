#!/usr/bin/env python3
# phase21a_curvature_localization.py
#
# Phase 21a: Curvature Localization
# - Uses Phase16a holonomy loops (+ transport cache) to localize where holonomy “lives”
# - Produces:
#   (1) per-directed-edge curvature score (abs + signed)
#   (2) per-node curvature score (sum of incident edges)
#   (3) 3D embedding (classical MDS on shortest-path distances) with edges colored by curvature
#
# Works with Phase16a formats:
#   (A) legacy: {"top":[...], "all_count":N, "plane":"bo_lr"}
#   (B) threshold_results:
#       {"plane":"bo_lr", "threshold_results": {"0.35":{"min_sim":0.35,"kept_count":10,"top":[...]}, ...}, ...}
#
# Example:
#   python phase21a_curvature_localization.py \
#     --cache outputs_edges_relternary256_phase15/phase15f_transport_cache.json \
#     --plane bo_lr \
#     --loops_json outputs_edges_relternary256_phase15/phase16a_holonomy_bo_lr.json \
#     --threshold 0.35 --min_sim 0.35 \
#     --min_edge_sim 0.40 \
#     --sim_mode log \
#     --outdir outputs_edges_relternary256_phase15/phase21a_bo_lr
#
# Notes on scoring:
# - Each loop has a holonomy angle (deg) from its stored R_loop (k=3 assumed for axis/angle sign).
# - Each directed edge in the loop receives:
#     contrib_abs   += angle_deg / L * w_edge
#     contrib_signed+= sign(loop_axis · edge_axis) * angle_deg / L * w_edge
#   where w_edge defaults to 1.0, or can be sim-weighted (see --weight_mode).
#
# This is a “localization heuristic”: it answers “which transitions participate most in curvature”
# rather than trying to uniquely attribute curvature to a single edge.

import os, json, math, argparse, heapq
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------- small utils ----------------------------

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

    if "threshold_results" in data and isinstance(data["threshold_results"], dict):
        tr = data["threshold_results"]
        chosen_key = pick_threshold_key(tr, threshold)
        if chosen_key is None:
            # choose highest numeric key if possible
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

    if top_k is not None:
        loops = loops[:max(0, int(top_k))]

    return plane, loops, meta

# ---------------------------- rotation helpers ----------------------------

def rotation_angle_from_trace(R):
    # assumes near-rotation
    k = R.shape[0]
    tr = float(np.trace(R))
    if k == 3:
        c = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
        return math.acos(c)
    if k == 2:
        c = max(-1.0, min(1.0, tr / 2.0))
        return math.acos(c)
    c = max(-1.0, min(1.0, tr / k))
    return math.acos(c)

def axis_angle_from_R3(R):
    eps = 1e-9
    theta = rotation_angle_from_trace(R)
    if abs(theta) < 1e-10:
        return {"angle_rad": 0.0, "angle_deg": 0.0, "axis": np.array([1.0, 0.0, 0.0], dtype=np.float64), "note": "near_identity"}
    S = (R - R.T) / (2.0 * math.sin(theta) + eps)
    axis = np.array([S[2,1], S[0,2], S[1,0]], dtype=np.float64)
    n = float(np.linalg.norm(axis))
    if n < 1e-10:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        n = 1.0
    axis = axis / n
    return {"angle_rad": float(theta), "angle_deg": float(math.degrees(theta)), "axis": axis, "note": "axis_angle"}

# ---------------------------- cache edge access ----------------------------

def mat_from_entry(entry):
    if entry is None:
        return None
    if "R" in entry:
        return np.array(entry["R"], dtype=np.float64)
    return None

def sim_from_entry(entry):
    if entry is None:
        return None
    return float(entry.get("similarity", float("nan")))

def get_edge(cache, a, b, plane):
    """
    Fetch A__to__B plane entry, else try reverse and invert (transpose).
    Returns (R, sim, how)
    """
    maps = cache.get("transport_maps", {})
    key = f"{a}__to__{b}"
    if key in maps and plane in maps[key]:
        entry = maps[key][plane]
        R = mat_from_entry(entry)
        return R, sim_from_entry(entry), "forward"
    key2 = f"{b}__to__{a}"
    if key2 in maps and plane in maps[key2]:
        entry = maps[key2][plane]
        R = mat_from_entry(entry)
        if R is None:
            return None, None, "missing"
        return R.T, sim_from_entry(entry), "reverse_used_transpose"
    return None, None, "missing"

# ---------------------------- graph embedding (classical MDS) ----------------------------

def sim_to_dist(sim, mode="log", eps=1e-9):
    """
    Convert similarity in (0..1-ish) to a positive distance.
    mode:
      - linear: dist = 1 - sim
      - sqrt:   dist = sqrt(max(0, 1 - sim))
      - log:    dist = -log(sim + eps)
    """
    if sim is None or (isinstance(sim, float) and (math.isnan(sim) or math.isinf(sim))):
        return None
    s = float(sim)
    if mode == "linear":
        return max(0.0, 1.0 - s)
    if mode == "sqrt":
        return math.sqrt(max(0.0, 1.0 - s))
    # log
    return max(0.0, -math.log(max(eps, s)))

def dijkstra_all_pairs(names, edges, mode="log"):
    """
    names: list[str]
    edges: list of (u_name, v_name, sim) directed edges
    returns: (D, index) where D is NxN shortest-path matrix
    """
    idx = {n:i for i,n in enumerate(names)}
    N = len(names)
    adj = [[] for _ in range(N)]
    for u, v, sim in edges:
        duv = sim_to_dist(sim, mode=mode)
        if duv is None:
            continue
        i, j = idx[u], idx[v]
        adj[i].append((j, duv))

    D = np.full((N, N), np.inf, dtype=np.float64)
    for s in range(N):
        D[s, s] = 0.0
        pq = [(0.0, s)]
        seen = set()
        while pq:
            d, u = heapq.heappop(pq)
            if u in seen:
                continue
            seen.add(u)
            for v, w in adj[u]:
                nd = d + w
                if nd < D[s, v]:
                    D[s, v] = nd
                    heapq.heappush(pq, (nd, v))

    # symmetrize for MDS (use min of (i->j) and (j->i) if graph is directed)
    Ds = np.minimum(D, D.T)
    # replace unreachable pairs with max finite * 1.5 (keeps MDS stable)
    finite = Ds[np.isfinite(Ds)]
    if finite.size == 0:
        raise SystemExit("[error] Graph has no finite distances.")
    fill = float(np.max(finite)) * 1.5
    Ds[~np.isfinite(Ds)] = fill
    return Ds, idx

def classical_mds(D, ndim=3):
    """
    Classical (Torgerson) MDS on distance matrix D (NxN).
    Returns coords Nxndim and eigenvalues.
    """
    N = D.shape[0]
    D2 = D**2
    J = np.eye(N) - np.ones((N, N))/N
    B = -0.5 * (J @ D2 @ J)
    w, V = np.linalg.eigh(B)  # ascending
    order = np.argsort(w)[::-1]
    w = w[order]
    V = V[:, order]
    # keep positive eigenvalues
    pos = w > 1e-12
    wpos = w[pos]
    Vpos = V[:, pos]
    ndim_eff = min(ndim, Vpos.shape[1])
    L = np.diag(np.sqrt(wpos[:ndim_eff]))
    X = Vpos[:, :ndim_eff] @ L
    # pad if needed
    if ndim_eff < ndim:
        X = np.hstack([X, np.zeros((N, ndim-ndim_eff), dtype=np.float64)])
    return X, w.tolist()

# ---------------------------- phase21a core ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="Path to phase15f_transport_cache.json")
    ap.add_argument("--plane", default="bo_lr", choices=["bo", "lt", "bo_lr"], help="Plane.")
    ap.add_argument("--loops_json", required=True, help="Phase16a holonomy JSON (legacy or threshold_results).")
    ap.add_argument("--threshold", type=float, default=None, help="If loops_json is threshold_results, choose threshold (e.g. 0.35).")
    ap.add_argument("--min_sim", type=float, default=None, help="Require loop edge_sim_min >= min_sim (bucket filter).")
    ap.add_argument("--top_loops", type=int, default=None, help="Optional: only use first K loops from bucket/top.")
    ap.add_argument("--min_edge_sim", type=float, default=0.0, help="When building graph & edge axes, require transport sim>=this.")
    ap.add_argument("--weight_mode", default="none", choices=["none", "sim", "sim_over_mean"],
                    help="Edge contribution weight inside loop.")
    ap.add_argument("--sim_mode", default="log", choices=["linear", "sqrt", "log"], help="Distance mapping for MDS embedding.")
    ap.add_argument("--outdir", required=True, help="Output directory.")
    ap.add_argument("--annotate", action="store_true", help="Annotate node labels in plot.")
    ap.add_argument("--max_edges_draw", type=int, default=300, help="Max edges drawn (highest curvature first).")
    args = ap.parse_args()

    safe_mkdir(args.outdir)

    with open(args.cache, "r") as f:
        cache = json.load(f)

    plane, loops, loops_meta = load_phase16a_loops(
        args.loops_json,
        plane_arg=args.plane,
        threshold=args.threshold,
        min_sim=args.min_sim,
        top_k=args.top_loops,
    )
    if not loops:
        raise SystemExit("[error] No loops available after selection. Check --threshold/--min_sim/--plane.")

    # Build list of anchor names from cache anchors if present, else infer from transport keys
    names = sorted(list(cache.get("anchors", {}).keys()))
    if not names:
        maps = cache.get("transport_maps", {})
        seen = set()
        for k in maps.keys():
            if "__to__" in k:
                a, b = k.split("__to__")
                seen.add(a); seen.add(b)
        names = sorted(list(seen))
    if len(names) < 3:
        raise SystemExit(f"[error] Need >=3 anchors for a 3D map; found {len(names)}")

    # Precompute per-directed-edge (a->b) sim and axis-angle (if k==3)
    edge_sim = {}
    edge_axis = {}
    all_directed_edges = []
    for a in names:
        for b in names:
            if a == b:
                continue
            R, sim, how = get_edge(cache, a, b, plane)
            if R is None:
                continue
            if sim is None or (isinstance(sim, float) and math.isnan(sim)):
                continue
            if float(sim) < float(args.min_edge_sim):
                continue
            edge_sim[(a, b)] = float(sim)
            all_directed_edges.append((a, b, float(sim)))
            if R.shape[0] == 3:
                aa = axis_angle_from_R3(np.array(R, dtype=np.float64))
                edge_axis[(a, b)] = aa["axis"]

    # Graph embedding (MDS on shortest-path distances)
    D, idx = dijkstra_all_pairs(names, all_directed_edges, mode=args.sim_mode)
    X, eigvals = classical_mds(D, ndim=3)
    coords = {n: [float(X[idx[n],0]), float(X[idx[n],1]), float(X[idx[n],2])] for n in names}

    # Curvature localization: accumulate edge contributions across loops
    edge_abs = defaultdict(float)
    edge_signed = defaultdict(float)
    edge_hits = defaultdict(int)

    loop_summaries = []
    for L in loops:
        loop_nodes = L.get("loop", None)
        R_loop = L.get("R_loop", None)

        if not loop_nodes or not R_loop:
            continue

        Rl = np.array(R_loop, dtype=np.float64)
        if Rl.shape[0] != 3:
            # for now we localize only for k==3 (your bo_lr loops are k=3)
            continue

        aa_loop = axis_angle_from_R3(Rl)
        ang_deg = float(aa_loop["angle_deg"])
        axis_loop = aa_loop["axis"]

        nodes_closed = list(loop_nodes) + [loop_nodes[0]]
        Llen = len(loop_nodes)

        # mean sim for weighting if desired
        sims_in_loop = []
        directed_edges_in_loop = []
        for i in range(len(nodes_closed)-1):
            a = nodes_closed[i]; b = nodes_closed[i+1]
            s = edge_sim.get((a,b), None)
            if s is not None:
                sims_in_loop.append(float(s))
            directed_edges_in_loop.append((a,b))

        sim_mean = float(np.mean(sims_in_loop)) if sims_in_loop else float("nan")

        for (a,b) in directed_edges_in_loop:
            s = edge_sim.get((a,b), None)
            if s is None:
                # if loop used a reverse edge, 16a stored edges[]; still, for localization we skip missing
                continue

            w = 1.0
            if args.weight_mode == "sim":
                w = float(s)
            elif args.weight_mode == "sim_over_mean":
                if sim_mean == sim_mean and sim_mean > 1e-9:  # not nan
                    w = float(s) / float(sim_mean)

            # signed contribution uses loop axis · edge axis if edge axis known; else treat as positive
            sign = 1.0
            ea = edge_axis.get((a,b), None)
            if ea is not None:
                dot = float(np.dot(axis_loop, ea))
                sign = 1.0 if dot >= 0 else -1.0

            contrib = (ang_deg / max(1, Llen)) * w
            edge_abs[(a,b)] += abs(contrib)
            edge_signed[(a,b)] += sign * abs(contrib)
            edge_hits[(a,b)] += 1

        loop_summaries.append({
            "loop": loop_nodes,
            "angle_deg": ang_deg,
            "axis": [float(axis_loop[0]), float(axis_loop[1]), float(axis_loop[2])],
            "edge_sim_min": float(L.get("edge_sim_min", float("nan"))),
            "edge_sim_mean": float(L.get("edge_sim_mean", float("nan"))),
            "rank_score": float(L.get("rank_score", float("nan"))),
        })

    if not edge_abs:
        raise SystemExit("[error] No curvature contributions computed. (Do loops have R_loop and do edges meet min_edge_sim?)")

    # Node scores (incident directed edges)
    node_abs = defaultdict(float)
    node_signed = defaultdict(float)
    for (a,b), v in edge_abs.items():
        node_abs[a] += v
        node_abs[b] += v
    for (a,b), v in edge_signed.items():
        node_signed[a] += v
        node_signed[b] += v

    # Prepare ranked edges for plotting
    ranked = sorted(edge_abs.items(), key=lambda kv: kv[1], reverse=True)
    if args.max_edges_draw is not None:
        ranked = ranked[:max(0, int(args.max_edges_draw))]

    # Plot: edges colored by abs curvature
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"Phase21a Curvature Localization (plane={plane}, sim_mode={args.sim_mode}, weight={args.weight_mode})")

    # node scatter
    Xn = np.array([coords[n] for n in names], dtype=np.float64)
    ax.scatter(Xn[:,0], Xn[:,1], Xn[:,2], s=60)

    if args.annotate:
        for n in names:
            x,y,z = coords[n]
            ax.text(x, y, z, n, fontsize=8)

    # edge colors
    vals = np.array([v for (_,v) in ranked], dtype=np.float64)
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    if vmax <= vmin + 1e-12:
        vmax = vmin + 1.0
    norm = (vals - vmin) / (vmax - vmin + 1e-12)
    cmap = plt.get_cmap("viridis")

    for (k,(a,b)) in enumerate([kv[0] for kv in ranked]):
        v = edge_abs[(a,b)]
        t = float((v - vmin) / (vmax - vmin + 1e-12))
        col = cmap(t)
        x1,y1,z1 = coords[a]
        x2,y2,z2 = coords[b]
        ax.plot([x1,x2],[y1,y2],[z1,z2], color=col, linewidth=2.0, alpha=0.9)

    # write a colorbar
    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_array(vals)
    cb = plt.colorbar(mappable, ax=ax, fraction=0.03, pad=0.08)
    cb.set_label("edge curvature score (abs)")

    ax.set_xlabel("MDS-1")
    ax.set_ylabel("MDS-2")
    ax.set_zlabel("MDS-3")

    out_png = os.path.join(args.outdir, f"phase21a_curvature_localization_{plane}.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close(fig)

    # Save JSON summary
    edge_rows = []
    for (a,b), v in sorted(edge_abs.items(), key=lambda kv: kv[1], reverse=True):
        edge_rows.append({
            "a": a,
            "b": b,
            "curv_abs": float(v),
            "curv_signed": float(edge_signed[(a,b)]),
            "hits": int(edge_hits[(a,b)]),
            "sim": float(edge_sim.get((a,b), float("nan"))),
        })

    node_rows = []
    for n in sorted(names):
        node_rows.append({
            "name": n,
            "curv_abs": float(node_abs.get(n, 0.0)),
            "curv_signed": float(node_signed.get(n, 0.0)),
        })

    out = {
        "phase": "21a",
        "plane": plane,
        "status": "ok",
        "args": {
            "cache": args.cache,
            "loops_json": args.loops_json,
            "threshold": args.threshold,
            "min_sim": args.min_sim,
            "top_loops": args.top_loops,
            "min_edge_sim": args.min_edge_sim,
            "weight_mode": args.weight_mode,
            "sim_mode": args.sim_mode,
            "annotate": bool(args.annotate),
            "max_edges_draw": args.max_edges_draw,
        },
        "loops_meta": loops_meta,
        "counts": {
            "anchors": len(names),
            "loops_used": len(loop_summaries),
            "edges_scored": len(edge_rows),
        },
        "embedding": {
            "method": "classical_mds_shortest_path",
            "eigvals": eigvals,
            "coords": coords,
        },
        "loop_summaries": loop_summaries,
        "edge_scores": edge_rows,
        "node_scores": node_rows,
        "paths": {
            "png": out_png,
        }
    }

    out_json = os.path.join(args.outdir, f"phase21a_curvature_localization_{plane}.json")
    with open(out_json, "w") as g:
        json.dump(out, g, indent=2)

    # Console summary: top edges + top nodes
    print(f"[phase21a] plane={plane} loops_used={len(loop_summaries)} edges_scored={len(edge_rows)}")
    print(f"[saved] {out_png}")
    print(f"[saved] {out_json}")

    print("\nTop directed edges by curvature(abs):")
    for i, r in enumerate(edge_rows[:10], 1):
        print(f" {i:02d}. {r['a']} -> {r['b']}   curv_abs={r['curv_abs']:.6f}  hits={r['hits']}  sim={r['sim']:.4f}")

    node_rows_sorted = sorted(node_rows, key=lambda r: r["curv_abs"], reverse=True)
    print("\nTop nodes by incident curvature(abs):")
    for i, r in enumerate(node_rows_sorted[:10], 1):
        print(f" {i:02d}. {r['name']}   curv_abs={r['curv_abs']:.6f}  curv_signed={r['curv_signed']:.6f}")

if __name__ == "__main__":
    main()
