#!/usr/bin/env python3
# phase21b_edge_axis_field.py
#
# Phase 21b: Build an "axis field" over transport edges from holonomy loops.
# - Loads Phase16a holonomy loops (legacy or threshold_results formats)
# - Extracts loop axis-angle from R_loop (k=3) and uses it as curvature generator
# - Aggregates per-edge:
#     * curv_abs (sum |angle|)
#     * curv_signed (signed by global reference axis)
#     * hits (#loops containing edge)
#     * axis_mean (mean loop axis, aligned to global ref)
#     * axis_consistency (mean |dot(axis_i, axis_mean)|)
# - Builds a 3D graph embedding from edge distances (sim->dist with sim_mode)
# - Plots: nodes + transport edges + axis ticks at edge midpoints
#
# Example (PowerShell):
#   python bbit_geomlang/phase21b_edge_axis_field.py `
#     --cache outputs_edges_relternary256_phase15/phase15f_transport_cache.json `
#     --loops_json outputs_edges_relternary256_phase15/phase16a_holonomy_bo_lr.json `
#     --plane bo_lr `
#     --threshold 0.35 --min_sim 0.35 --min_edge_sim 0.40 --sim_mode log `
#     --outdir outputs_edges_relternary256_phase15/phase21b_bo_lr `
#     --annotate --max_edges_draw 300 --tick_scale 0.1

import os, json, math, argparse
import numpy as np

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

def unit(v, eps=1e-12):
    v = np.array(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
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

def load_phase16a_loops(path, plane_arg=None, threshold=None, min_sim=None, top_loops=None):
    with open(path, "r") as f:
        data = json.load(f)

    meta = {}
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
            bucket_min_sim = bucket.get("min_sim", bucket.get("min_sim_threshold", bucket.get("min_sim", None)))
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

    # plane filter
    if plane_arg:
        loops = [L for L in loops if str(L.get("plane", plane)) == plane_arg]
        plane = plane_arg

    # min_sim filter (belt+suspenders)
    if min_sim is not None:
        loops = [L for L in loops if float(L.get("edge_sim_min", -1e9)) >= float(min_sim)]

    # top_loops clamp
    if top_loops is not None and top_loops > 0:
        loops = loops[:top_loops]

    return plane, loops, meta

# ---------------------------- transport cache reading ----------------------------

def mat_from_entry(entry):
    if entry is None:
        return None
    if "R" in entry:
        return np.array(entry["R"], dtype=np.float64)
    return None

def sim_from_entry(entry):
    """
    Robust sim reader:
    - supports multiple keys (similarity/sim/edge_sim)
    - rejects NaN/inf
    - returns None if unusable
    """
    if entry is None or not isinstance(entry, dict):
        return None

    for k in ("similarity", "sim", "edge_sim"):
        v = entry.get(k, None)
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

def get_edge(cache, a, b, plane):
    """
    Fetch A__to__B plane entry; else try reverse and use transpose as inverse.
    Returns (R, sim, how) or (None,None,"missing")
    """
    maps = cache.get("transport_maps", {})
    key = f"{a}__to__{b}"
    if key in maps and plane in maps[key]:
        entry = maps[key][plane]
        R = mat_from_entry(entry)
        sim = sim_from_entry(entry)
        if R is None:
            return None, None, "missing"
        return R, sim, "forward"

    key2 = f"{b}__to__{a}"
    if key2 in maps and plane in maps[key2]:
        entry = maps[key2][plane]
        R = mat_from_entry(entry)
        sim = sim_from_entry(entry)
        if R is None:
            return None, None, "missing"
        return R.T, sim, "reverse_used_transpose"

    return None, None, "missing"

def build_edges_from_cache(cache, plane, min_edge_sim=0.0):
    """
    Return list of directed edges: (a,b,sim,R,how)
    (Only edges with usable sim and sim>=min_edge_sim)
    """
    maps = cache.get("transport_maps", {})
    out = []
    for key, per in maps.items():
        if "__to__" not in key:
            continue
        if plane not in per:
            continue
        a, b = key.split("__to__", 1)
        entry = per[plane]
        R = mat_from_entry(entry)
        sim = sim_from_entry(entry)
        if R is None or sim is None:
            continue
        if sim < float(min_edge_sim):
            continue
        out.append((a, b, float(sim), R, "forward_raw"))
    return out

def diag_sim_stats(cache, plane):
    maps = cache.get("transport_maps", {})
    sims = []
    missing = 0
    for key, per in maps.items():
        if "__to__" not in key:
            continue
        if plane not in per:
            continue
        s = sim_from_entry(per[plane])
        if s is None:
            missing += 1
        else:
            sims.append(float(s))
    if sims:
        print(f"[diag] plane={plane} cache sims: n={len(sims)} min={min(sims):.6f} max={max(sims):.6f} missing={missing}")
    else:
        print(f"[diag] plane={plane} NO usable sims found (missing={missing})")

# ---------------------------- axis-angle (k=3) ----------------------------

def rotation_angle_from_trace(R):
    tr = float(np.trace(R))
    c = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
    return math.acos(c)

def axis_angle_from_R3(R):
    eps = 1e-12
    theta = rotation_angle_from_trace(R)
    if abs(theta) < 1e-10:
        return {"angle_rad": 0.0, "angle_deg": 0.0, "axis": [1.0, 0.0, 0.0], "note": "near_identity"}
    S = (R - R.T) / (2.0 * math.sin(theta) + eps)
    axis = np.array([S[2,1], S[0,2], S[1,0]], dtype=np.float64)
    axis = unit(axis)
    return {"angle_rad": float(theta), "angle_deg": float(math.degrees(theta)), "axis": axis.tolist(), "note": "axis_angle"}

# ---------------------------- graph distances + classical MDS ----------------------------

def sim_to_dist(sim, sim_mode="log", eps=1e-9):
    """
    Convert similarity in (0,1] to distance >=0.
    Modes:
      - linear: dist = 1 - sim
      - sqrt:   dist = sqrt(1 - sim)
      - log:    dist = -log(sim)
    """
    s = float(sim)
    s = max(eps, min(1.0, s))
    if sim_mode == "linear":
        return 1.0 - s
    if sim_mode == "sqrt":
        return math.sqrt(max(0.0, 1.0 - s))
    if sim_mode == "log":
        return -math.log(s)
    raise ValueError(f"unknown sim_mode={sim_mode}")

def floyd_warshall(D):
    """
    All-pairs shortest paths for small N.
    D is NxN with inf for missing edges, 0 diag.
    """
    N = D.shape[0]
    dist = D.copy()
    for k in range(N):
        dk = dist[:, k][:, None]
        kk = dist[k, :][None, :]
        dist = np.minimum(dist, dk + kk)
    return dist

def classical_mds_from_dist(D, dim=3):
    """
    Classical MDS (Torgerson) from full distance matrix.
    """
    D = np.array(D, dtype=np.float64)
    N = D.shape[0]
    D2 = D ** 2
    J = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * J @ D2 @ J

    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    pos = eigvals > 1e-12
    ndim = min(dim, int(np.sum(pos)))
    if ndim <= 0:
        return np.zeros((N, dim)), eigvals.tolist(), 0

    L = np.diag(np.sqrt(eigvals[:ndim]))
    X = eigvecs[:, :ndim] @ L
    if ndim < dim:
        X = np.concatenate([X, np.zeros((N, dim-ndim))], axis=1)
    return X, eigvals.tolist(), ndim

# ---------------------------- main aggregation ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--loops_json", required=True)
    ap.add_argument("--plane", default="bo_lr", choices=["bo", "lt", "bo_lr"])
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--min_sim", type=float, default=None)
    ap.add_argument("--top_loops", type=int, default=None)
    ap.add_argument("--min_edge_sim", type=float, default=0.0)
    ap.add_argument("--sim_mode", default="log", choices=["linear", "sqrt", "log"])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--annotate", action="store_true")
    ap.add_argument("--max_edges_draw", type=int, default=300)
    ap.add_argument("--tick_scale", type=float, default=0.10, help="Axis tick length in embedding units.")
    args = ap.parse_args()

    safe_mkdir(args.outdir)

    with open(args.cache, "r") as f:
        cache = json.load(f)

    plane, loops, loops_meta = load_phase16a_loops(
        args.loops_json,
        plane_arg=args.plane,
        threshold=args.threshold,
        min_sim=args.min_sim,
        top_loops=args.top_loops
    )
    if not loops:
        raise SystemExit("[error] No loops selected. Check --threshold/--min_sim/--plane.")

    # collect anchor names
    names = sorted(list(cache.get("anchors", {}).keys()))
    if not names:
        maps = cache.get("transport_maps", {})
        seen = set()
        for k in maps.keys():
            if "__to__" in k:
                a, b = k.split("__to__", 1)
                seen.add(a); seen.add(b)
        names = sorted(list(seen))
    name_to_i = {n:i for i,n in enumerate(names)}
    N = len(names)

    # build directed edge list from cache
    edges_raw = build_edges_from_cache(cache, plane, min_edge_sim=args.min_edge_sim)
    if not edges_raw:
        print("[warn] No edges passed sim/min_edge_sim filtering. Printing cache sim stats:")
        diag_sim_stats(cache, plane)

    # build graph dist matrix from sims (directed -> undirected min-dist)
    INF = 1e18
    D = np.full((N, N), INF, dtype=np.float64)
    np.fill_diagonal(D, 0.0)

    kept_edges = []
    for (a,b,sim,R,how) in edges_raw:
        if a not in name_to_i or b not in name_to_i:
            continue
        i, j = name_to_i[a], name_to_i[b]
        dist = sim_to_dist(sim, sim_mode=args.sim_mode)
        if dist < D[i,j]:
            D[i,j] = dist
        if dist < D[j,i]:
            D[j,i] = dist
        kept_edges.append((a,b,sim,dist,how))

    # shortest path completion
    Dsp = floyd_warshall(D)

    if np.any(Dsp >= INF/10):
        print("[warn] graph appears disconnected at given min_edge_sim; embedding may be unstable.")

    X, eigvals, ndim_used = classical_mds_from_dist(Dsp, dim=3)
    coords = {names[i]: [float(X[i,0]), float(X[i,1]), float(X[i,2])] for i in range(N)}

    # ------------------ compute loop axes + global reference ------------------

    loop_axes = []
    loop_infos = []

    for L in loops:
        R_loop = L.get("R_loop", None)
        if R_loop is None:
            continue
        R = np.array(R_loop, dtype=np.float64)
        if R.shape != (3,3):
            continue
        aa = axis_angle_from_R3(R)
        axis = unit(aa["axis"])
        ang = float(aa["angle_deg"])
        loop_axes.append(axis)
        loop_infos.append({
            "loop": L.get("loop", []),
            "rank_score": float(L.get("rank_score", float("nan"))),
            "edge_sim_min": float(L.get("edge_sim_min", float("nan"))),
            "edge_sim_mean": float(L.get("edge_sim_mean", float("nan"))),
            "angle_deg": ang,
            "axis": axis.tolist(),
        })

    if not loop_axes:
        raise SystemExit("[error] No usable k=3 loops with R_loop found.")

    # build global reference axis with sign alignment
    ref = np.zeros(3, dtype=np.float64)
    for ax in loop_axes:
        if float(np.linalg.norm(ref)) < 1e-12:
            ref = ax.copy()
        else:
            ref += ax if float(np.dot(ax, ref)) >= 0 else (-ax)
    ref = unit(ref)
    if float(np.linalg.norm(ref)) < 1e-12:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    # ------------------ per-edge aggregation from loops ------------------

    edge_stats = {}

    def ensure_edge(a,b):
        k = (a,b)
        if k not in edge_stats:
            edge_stats[k] = {
                "a": a, "b": b,
                "sim": None,
                "curv_abs": 0.0,
                "curv_signed": 0.0,
                "hits": 0,
                "axis_sum": np.zeros(3, dtype=np.float64),
            }
        return edge_stats[k]

    def fetch_sim(a,b):
        R, sim, how = get_edge(cache, a, b, plane)
        if sim is None:
            return None
        return float(sim)

    for info in loop_infos:
        loop = info["loop"]
        if not loop or len(loop) < 2:
            continue
        ax = np.array(info["axis"], dtype=np.float64)
        ang = float(info["angle_deg"])

        # align axis to global ref for signed angle
        sgn = 1.0 if float(np.dot(ax, ref)) >= 0 else -1.0
        ax_aligned = ax * sgn
        ang_signed = ang * sgn

        nodes = list(loop) + [loop[0]]
        for i in range(len(nodes)-1):
            a, b = nodes[i], nodes[i+1]
            st = ensure_edge(a,b)

            if st["sim"] is None:
                st["sim"] = fetch_sim(a,b)

            st["hits"] += 1
            st["curv_abs"] += abs(ang)
            st["curv_signed"] += ang_signed

            w = abs(ang)
            if st["sim"] is not None:
                w *= float(st["sim"])
            st["axis_sum"] += w * ax_aligned

    # finalize edge axis_mean + consistency proxy
    edge_list = []
    for (a,b), st in edge_stats.items():
        axis_mean = unit(st["axis_sum"])
        denom = (st["curv_abs"] * (float(st["sim"]) if st["sim"] is not None else 1.0)) + 1e-12
        consistency_proxy = float(np.linalg.norm(st["axis_sum"])) / float(denom)
        consistency_proxy = max(0.0, min(1.0, consistency_proxy))

        edge_list.append({
            "a": a, "b": b,
            "sim": (None if st["sim"] is None else float(st["sim"])),
            "hits": int(st["hits"]),
            "curv_abs": float(st["curv_abs"]),
            "curv_signed": float(st["curv_signed"]),
            "axis_mean": axis_mean.tolist(),
            "axis_consistency_proxy": float(consistency_proxy),
        })

    edge_list.sort(key=lambda e: e["curv_abs"], reverse=True)

    # node scores = sum of incident directed edges
    node_scores = {n: {"name": n, "curv_abs": 0.0, "curv_signed": 0.0} for n in names}
    for e in edge_list:
        a, b = e["a"], e["b"]
        if a in node_scores:
            node_scores[a]["curv_abs"] += e["curv_abs"]
            node_scores[a]["curv_signed"] += e["curv_signed"]
        if b in node_scores:
            node_scores[b]["curv_abs"] += e["curv_abs"]
            node_scores[b]["curv_signed"] += e["curv_signed"]
    node_scores_list = sorted(node_scores.values(), key=lambda d: d["curv_abs"], reverse=True)

    # ------------------ plot ------------------

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(10, 8))
    axp = fig.add_subplot(111, projection="3d")

    xs = [coords[n][0] for n in names]
    ys = [coords[n][1] for n in names]
    zs = [coords[n][2] for n in names]
    axp.scatter(xs, ys, zs, s=60)

    if args.annotate:
        for n in names:
            x,y,z = coords[n]
            axp.text(x, y, z, n, fontsize=8)

    # draw transport edges
    drawn = 0
    for (a,b,sim,dist,how) in kept_edges:
        if drawn >= args.max_edges_draw:
            break
        if a not in coords or b not in coords:
            continue
        xa,ya,za = coords[a]
        xb,yb,zb = coords[b]
        axp.plot([xa,xb], [ya,yb], [za,zb], linewidth=1)
        drawn += 1

    # draw axis ticks for top curvature edges
    top_ticks = min(len(edge_list), 50)
    for e in edge_list[:top_ticks]:
        a, b = e["a"], e["b"]
        if a not in coords or b not in coords:
            continue
        xa,ya,za = coords[a]
        xb,yb,zb = coords[b]
        mx,my,mz = (0.5*(xa+xb), 0.5*(ya+yb), 0.5*(za+zb))
        axis = np.array(e["axis_mean"], dtype=np.float64)
        if float(np.linalg.norm(axis)) < 1e-12:
            continue

        if float(e["curv_signed"]) < 0:
            axis = -axis

        s = args.tick_scale * (0.25 + 0.75 * float(e["axis_consistency_proxy"]))
        dx,dy,dz = (axis[0]*s, axis[1]*s, axis[2]*s)
        axp.plot([mx-dx, mx+dx], [my-dy, my+dy], [mz-dz, mz+dz], linewidth=2)

    axp.set_title(f"Phase21b Edge Axis Field ({plane}) | sim_mode={args.sim_mode} | min_edge_sim={args.min_edge_sim}")
    axp.set_xlabel("X")
    axp.set_ylabel("Y")
    axp.set_zlabel("Z")

    out_png = os.path.join(args.outdir, f"phase21b_edge_axis_field_{plane}.png")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    # ------------------ save json ------------------

    out = {
        "phase": "21b",
        "plane": plane,
        "status": "ok",
        "args": vars(args),
        "loops_meta": loops_meta,
        "embedding": {
            "method": "classical_mds_shortest_path",
            "sim_mode": args.sim_mode,
            "eigvals": eigvals,
            "ndim_used_positive": int(ndim_used),
            "coords": coords,
        },
        "global_reference_axis": ref.tolist(),
        "counts": {
            "anchors": int(N),
            "loops_used": int(len(loop_infos)),
            "edges_in_cache_filtered": int(len(kept_edges)),
            "edges_with_curvature": int(len(edge_list)),
        },
        "loop_summaries": loop_infos,
        "edge_axis_field": edge_list,
        "node_scores": node_scores_list,
        "paths": {"png": out_png},
        "note": "axis_mean lives in bo_lr k=3 basis; drawn as a glyph in embedding space (qualitative).",
    }

    out_json = os.path.join(args.outdir, f"phase21b_edge_axis_field_{plane}.json")
    with open(out_json, "w") as g:
        json.dump(out, g, indent=2)

    print(f"[phase21b] plane={plane} anchors={N} loops_used={len(loop_infos)} edges_with_curvature={len(edge_list)}")
    print(f"[edges] kept_from_cache={len(kept_edges)}  min_edge_sim={args.min_edge_sim}")
    if edge_list:
        e0 = edge_list[0]
        print(f"top_edge: {e0['a']} -> {e0['b']}  curv_abs={e0['curv_abs']:.3f}  curv_signed={e0['curv_signed']:.3f}  hits={e0['hits']}")
    print(f"[saved] {out_png}")
    print(f"[saved] {out_json}")

if __name__ == "__main__":
    main()
