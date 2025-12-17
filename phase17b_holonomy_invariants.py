#!/usr/bin/env python3
# phase17b_holonomy_invariants.py
#
# Phase 17b: Holonomy invariants for a chosen loop.
# Robust to Phase16a output formats:
#   (A) legacy: {"top":[...], "all_count":N, "plane":"bo_lr"}
#   (B) threshold_results:
#       {"plane":"bo_lr", "threshold_results": {"0.35":{"min_sim":0.35,"kept_count":10,"top":[...]}, ...}, ...}
#
# Example:
#   python phase17b_holonomy_invariants.py --cache .../phase15f_transport_cache.json \
#     --in_json .../phase16a_holonomy_bo_lr.json --plane bo_lr --threshold 0.35 --loop_index 1 \
#     --outdir .../phase17b_bo_lr --min_sim 0.35

import os, json, math, argparse
import numpy as np

# ---------------------------- utils ----------------------------

def fro_norm(A):
    return float(np.linalg.norm(A, ord="fro"))

def safe_mkdir(p):
    if p and not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def _as_float(x):
    try:
        return float(x)
    except Exception:
        return None

def pick_threshold_key(threshold_results: dict, threshold: float):
    """
    Return an existing key (string) from threshold_results that best matches `threshold`.
    Prefers exact string match ("0.35"), else nearest numeric key.
    """
    if threshold is None:
        return None
    # exact string matches first
    s = f"{threshold:.2f}"
    if s in threshold_results:
        return s
    s2 = str(threshold)
    if s2 in threshold_results:
        return s2

    # nearest numeric
    best_k = None
    best_d = None
    for k in threshold_results.keys():
        fk = _as_float(k)
        if fk is None:
            continue
        d = abs(fk - threshold)
        if best_d is None or d < best_d:
            best_d = d
            best_k = k
    return best_k

def load_phase16a_loops(in_path, plane_arg=None, threshold=None, min_sim=None):
    """
    Returns: (plane_str, loops_list, meta_dict)
    """
    with open(in_path, "r") as f:
        data = json.load(f)

    # Detect format
    if "threshold_results" in data and isinstance(data["threshold_results"], dict):
        tr = data["threshold_results"]
        chosen_key = pick_threshold_key(tr, threshold)
        if chosen_key is None:
            # no threshold provided; pick the *highest* threshold key numerically
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

        # bucket can be dict with 'top', or already a list
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
            "min_sim_arg": min_sim,
        }

    else:
        # legacy: {"top":[...], "all_count":N, "plane":"..."}
        loops = data.get("top", data.get("results", []))
        plane = data.get("plane", plane_arg or (loops[0].get("plane", "unknown") if loops else "unknown"))
        meta = {
            "format": "legacy",
            "total_valid": data.get("all_count", len(loops)),
            "min_sim_arg": min_sim,
        }

    # Optional filtering by plane
    if plane_arg:
        loops = [L for L in loops if str(L.get("plane", plane)) == plane_arg]
        plane = plane_arg

    # Optional filtering by min_sim (belt + suspenders)
    if min_sim is not None:
        loops = [L for L in loops if float(L.get("edge_sim_min", -1e9)) >= float(min_sim)]

    return plane, loops, meta

def loop_index_to_zero_based(loop_index, n):
    """
    Accept both 1-based (preferred) and 0-based.
    If loop_index in [1..n] => treat as 1-based.
    Else if loop_index in [0..n-1] => treat as 0-based.
    """
    if n <= 0:
        return None
    if 1 <= loop_index <= n:
        return loop_index - 1
    if 0 <= loop_index < n:
        return loop_index
    return None

def rotation_angle_from_trace(R):
    k = R.shape[0]
    tr = float(np.trace(R))
    if k == 2:
        c = max(-1.0, min(1.0, tr / 2.0))
        return math.acos(c)
    if k == 3:
        c = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
        return math.acos(c)
    # generic proxy
    c = max(-1.0, min(1.0, tr / k))
    return math.acos(c)

def axis_angle_from_R3(R):
    """
    Proper rotation matrix -> axis (unit) and angle in radians.
    Handles near-identity gracefully.
    """
    eps = 1e-9
    theta = rotation_angle_from_trace(R)
    if abs(theta) < 1e-8:
        return {"angle_rad": 0.0, "angle_deg": 0.0, "axis": [1.0, 0.0, 0.0], "note": "near_identity"}

    # axis from skew-symmetric part
    S = (R - R.T) / (2.0 * math.sin(theta) + eps)
    axis = np.array([S[2,1], S[0,2], S[1,0]], dtype=np.float64)
    n = float(np.linalg.norm(axis))
    if n < 1e-8:
        # fallback (numerical)
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        n = 1.0
    axis = (axis / n).tolist()
    return {"angle_rad": float(theta), "angle_deg": float(math.degrees(theta)), "axis": axis, "note": "axis_angle"}

def angle_from_R2(R):
    # for 2D rotation, theta = atan2(sin, cos) = atan2(R21, R11)
    theta = math.atan2(float(R[1,0]), float(R[0,0]))
    return {"angle_rad": float(theta), "angle_deg": float(math.degrees(theta)), "note": "2D_atan2"}

def eig_summary(R):
    vals = np.linalg.eigvals(R)
    out = []
    for z in vals:
        out.append({"re": float(np.real(z)), "im": float(np.imag(z)), "abs": float(np.abs(z))})
    return out

# ---------------------------- main ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="Path to phase15f_transport_cache.json (not strictly required if R_loop already in JSON).")
    ap.add_argument("--in_json", required=True, help="Phase16a holonomy JSON (legacy or threshold_results format).")
    ap.add_argument("--plane", default="", choices=["", "bo", "lt", "bo_lr"], help="Optional plane filter.")
    ap.add_argument("--threshold", type=float, default=None, help="If in_json is threshold_results, choose this threshold (e.g. 0.35).")
    ap.add_argument("--min_sim", type=float, default=None, help="Optional extra filter: require edge_sim_min >= min_sim.")
    ap.add_argument("--loop_index", type=int, default=1, help="Loop index to analyze. 1-based (loop 001 => 1).")
    ap.add_argument("--outdir", required=True, help="Output directory.")
    args = ap.parse_args()

    safe_mkdir(args.outdir)

    plane_arg = args.plane.strip() or None
    plane, loops, meta = load_phase16a_loops(args.in_json, plane_arg=plane_arg, threshold=args.threshold, min_sim=args.min_sim)

    if not loops:
        # dump helpful diagnostics
        with open(args.in_json, "r") as f:
            raw = json.load(f)
        keys = list(raw.get("threshold_results", {}).keys()) if isinstance(raw.get("threshold_results", None), dict) else []
        raise SystemExit(
            "[error] No loops found after selection.\n"
            f"  plane_filter={plane_arg}\n"
            f"  threshold={args.threshold}\n"
            f"  min_sim={args.min_sim}\n"
            f"  in_json_format={meta.get('format')}\n"
            f"  available_threshold_keys={keys}\n"
        )

    idx0 = loop_index_to_zero_based(args.loop_index, len(loops))
    if idx0 is None:
        raise SystemExit(f"[error] loop_index={args.loop_index} out of range. available_loops={len(loops)} (1-based).")

    chosen = loops[idx0]

    # Prefer stored R_loop; fall back to recomputing is possible, but not necessary here.
    if "R_loop" not in chosen:
        raise SystemExit("[error] chosen loop has no R_loop field (unexpected). Re-run Phase16a with full results.")
    R = np.array(chosen["R_loop"], dtype=np.float64)

    k = int(R.shape[0])
    I = np.eye(k, dtype=np.float64)
    delta = R - I

    det = float(np.linalg.det(R))
    tr = float(np.trace(R))
    fro = fro_norm(delta)
    spec = float(np.linalg.norm(delta, ord=2))

    invariants = {
        "k": k,
        "trace": tr,
        "det": det,
        "fro_norm_R_minus_I": fro,
        "spectral_norm_R_minus_I": spec,
        "eigvals": eig_summary(R),
    }

    if k == 2:
        invariants["rotation"] = angle_from_R2(R)
    elif k == 3:
        invariants["rotation"] = axis_angle_from_R3(R)
    else:
        invariants["rotation"] = {
            "angle_rad_proxy": float(rotation_angle_from_trace(R)),
            "angle_deg_proxy": float(math.degrees(rotation_angle_from_trace(R))),
            "note": f"{k}D trace-based proxy only"
        }

    out = {
        "phase": "17b",
        "plane": plane,
        "meta": meta,
        "selection": {
            "loop_index_arg": args.loop_index,
            "loop_index_zero_based": idx0,
            "loop": chosen.get("loop"),
            "edge_sim_min": float(chosen.get("edge_sim_min", float("nan"))),
            "edge_sim_mean": float(chosen.get("edge_sim_mean", float("nan"))),
            "edge_sim_max": float(chosen.get("edge_sim_max", float("nan"))),
            "rank_score": float(chosen.get("rank_score", float("nan"))),
        },
        "invariants": invariants,
        "R_loop": chosen["R_loop"],
        "edges": chosen.get("edges", []),
    }

    out_json = os.path.join(args.outdir, f"phase17b_invariants_{plane}_loop{idx0+1:03d}.json")
    with open(out_json, "w") as g:
        json.dump(out, g, indent=2)

    # print a tight console summary
    loop_nodes = chosen.get("loop", [])
    loop_str = " -> ".join(loop_nodes + [loop_nodes[0]]) if loop_nodes else "(unknown)"
    ang = invariants.get("rotation", {}).get("angle_deg", invariants.get("rotation", {}).get("angle_deg_proxy", None))
    if ang is None:
        ang = float("nan")

    print(f"[plane={plane}] loop_index={idx0+1:03d}/{len(loops)}  "
          f"||R-I||_F={fro:.6f}  det={det:.6f}  trace={tr:.6f}  angle_deg={ang:.3f}")
    print(f"loop: {loop_str}")
    print(f"[saved] {out_json}")

if __name__ == "__main__":
    main()
