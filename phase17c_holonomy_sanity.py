#!/usr/bin/env python3
# phase17c_holonomy_sanity.py
#
# Phase 17c: Sanity checks on holonomy loops.
# - Reverse-loop pairing: verifies R_reverse ≈ R_forward^{-1}
# - Nested composition: for 4-cycles, compares R_quad to products of two triangles
#
# Robust to Phase16a output formats:
#   (A) legacy: {"top":[...], "all_count":N, "plane":"bo_lr"}
#   (B) threshold_results:
#       {"plane":"bo_lr", "threshold_results": {"0.35":{"min_sim":0.35,"kept_count":10,"top":[...]}, ...}, ...}
#
# Example:
#   python phase17c_holonomy_sanity.py --in_json outputs_edges_relternary256_phase15/phase16a_holonomy_bo_lr.json \
#     --plane bo_lr --threshold 0.35 --min_sim 0.35 --outdir outputs_edges_relternary256_phase15/phase17c_bo_lr

import os, json, math, argparse
import numpy as np

# ---------------------------- utils ----------------------------

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

def load_phase16a_loops(in_path, plane_arg=None, threshold=None, min_sim=None):
    with open(in_path, "r") as f:
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

    if plane_arg:
        loops = [L for L in loops if str(L.get("plane", plane)) == plane_arg]
        plane = plane_arg

    if min_sim is not None:
        loops = [L for L in loops if float(L.get("edge_sim_min", -1e9)) >= float(min_sim)]

    return plane, loops, meta

def rotation_angle_deg(R):
    k = R.shape[0]
    tr = float(np.trace(R))
    if k == 2:
        c = max(-1.0, min(1.0, tr / 2.0))
        return math.degrees(math.acos(c))
    if k == 3:
        c = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
        return math.degrees(math.acos(c))
    c = max(-1.0, min(1.0, tr / k))
    return math.degrees(math.acos(c))

def axis_from_R3(R):
    # axis from skew-symmetric part; sign flips are allowed in comparisons
    theta = math.radians(rotation_angle_deg(R))
    if abs(theta) < 1e-10:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64), 0.0
    s = math.sin(theta)
    if abs(s) < 1e-12:
        # near pi; fallback via eigenvector of eigenvalue 1
        vals, vecs = np.linalg.eig(R)
        idx = int(np.argmin(np.abs(vals - 1.0)))
        v = np.real(vecs[:, idx]).astype(np.float64)
        n = np.linalg.norm(v)
        if n < 1e-12:
            v = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            n = 1.0
        return v / n, theta
    S = (R - R.T) / (2.0 * s)
    axis = np.array([S[2,1], S[0,2], S[1,0]], dtype=np.float64)
    n = float(np.linalg.norm(axis))
    if n < 1e-12:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        n = 1.0
    return axis / n, theta

def canonical_cycle(loop_nodes):
    """
    Canonical representative of the cycle ignoring rotation.
    We use minimal-rotation among all cyclic shifts.
    """
    L = list(loop_nodes)
    best = None
    for s in range(len(L)):
        rot = tuple(L[s:] + L[:s])
        if best is None or rot < best:
            best = rot
    return best

def reversed_cycle(loop_nodes):
    """
    Reverse traversal: A->B->C->A becomes A->C->B->A, i.e. reverse order.
    For node list [A,B,C,D], reversed should be [A,D,C,B] (keeping same start A).
    """
    L = list(loop_nodes)
    A = L[0]
    rest = L[1:]
    rev = [A] + list(reversed(rest))
    return rev

def key_for_pairing(loop_nodes):
    """
    Pair loops that are same node set and same start (up to rotation)
    by using canonical cycle.
    """
    return canonical_cycle(loop_nodes)

def get_R(loop_entry):
    if "R_loop" not in loop_entry:
        return None
    return np.array(loop_entry["R_loop"], dtype=np.float64)

def find_triangle(loop_map, tri_nodes):
    """
    Find triangle loop entry in loops by canonical cycle.
    tri_nodes is [A,B,C]. We look for any rotation match among available loops.
    """
    key = canonical_cycle(tri_nodes)
    return loop_map.get(key)

def compose_triangle_pair(R1, R2):
    # transport composition in loop scripts was: R_loop = R_last @ ... @ R_first
    # Here we compare loop holonomies as standalone matrices.
    # If quad can be decomposed as "go around tri1 then tri2", the net should be R2 @ R1.
    return R2 @ R1

# ---------------------------- checks ----------------------------

def reverse_pair_checks(Ra, Rb):
    k = Ra.shape[0]
    I = np.eye(k, dtype=np.float64)
    inv_resid = fro_norm((Ra @ Rb) - I)
    inv_resid_T = fro_norm((Ra @ Rb.T) - I)  # sometimes user stored transpose accidentally
    det_a = float(np.linalg.det(Ra))
    det_b = float(np.linalg.det(Rb))
    ang_a = rotation_angle_deg(Ra)
    ang_b = rotation_angle_deg(Rb)
    out = {
        "k": int(k),
        "det_a": det_a,
        "det_b": det_b,
        "angle_deg_a": ang_a,
        "angle_deg_b": ang_b,
        "abs_angle_diff_deg": abs(ang_a - ang_b),
        "inv_resid_fro": inv_resid,
        "inv_resid_fro_using_RbT": inv_resid_T,
    }
    if k == 3:
        ax_a, _ = axis_from_R3(Ra)
        ax_b, _ = axis_from_R3(Rb)
        dot = float(np.clip(np.dot(ax_a, ax_b), -1.0, 1.0))
        out["axis_a"] = ax_a.tolist()
        out["axis_b"] = ax_b.tolist()
        out["axis_dot"] = dot
        out["axis_dot_abs"] = abs(dot)
    return out

def quad_decomposition_checks(loop_map, quad_nodes):
    """
    quad_nodes: [A,B,C,D]
    Try 2 decompositions:
      (A,B,C) + (A,C,D)
      (A,B,D) + (B,C,D)
    Return best residual among those possible, with details.
    """
    Rq_entry = loop_map.get(canonical_cycle(quad_nodes))
    if Rq_entry is None:
        return None
    Rq = get_R(Rq_entry)
    if Rq is None:
        return None
    A,B,C,D = quad_nodes

    candidates = []

    # Decomp 1: tri1=(A,B,C), tri2=(A,C,D)
    t1 = find_triangle(loop_map, [A,B,C])
    t2 = find_triangle(loop_map, [A,C,D])
    if t1 is not None and t2 is not None:
        R1 = get_R(t1); R2 = get_R(t2)
        if R1 is not None and R2 is not None:
            Rpred = compose_triangle_pair(R1, R2)
            resid = fro_norm(Rq - Rpred)
            candidates.append({
                "decomposition": [[A,B,C],[A,C,D]],
                "resid_fro": resid,
            })

    # Decomp 2: tri1=(A,B,D), tri2=(B,C,D)
    t1 = find_triangle(loop_map, [A,B,D])
    t2 = find_triangle(loop_map, [B,C,D])
    if t1 is not None and t2 is not None:
        R1 = get_R(t1); R2 = get_R(t2)
        if R1 is not None and R2 is not None:
            Rpred = compose_triangle_pair(R1, R2)
            resid = fro_norm(Rq - Rpred)
            candidates.append({
                "decomposition": [[A,B,D],[B,C,D]],
                "resid_fro": resid,
            })

    if not candidates:
        return {"quad": quad_nodes, "available": False, "reason": "missing_triangles_for_decomposition"}

    candidates.sort(key=lambda x: x["resid_fro"])
    best = candidates[0]
    return {
        "quad": quad_nodes,
        "available": True,
        "best": best,
        "all_candidates": candidates,
    }

# ---------------------------- main ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True, help="Phase16a holonomy JSON (legacy or threshold_results).")
    ap.add_argument("--plane", default="", choices=["", "bo", "lt", "bo_lr"], help="Optional plane filter.")
    ap.add_argument("--threshold", type=float, default=None, help="Threshold bucket to select (e.g. 0.35) if threshold_results format.")
    ap.add_argument("--min_sim", type=float, default=None, help="Extra filter: edge_sim_min >= min_sim.")
    ap.add_argument("--outdir", required=True, help="Output directory for Phase17c report.")
    ap.add_argument("--max_pairs", type=int, default=9999, help="Max reverse pairs to report (sorted by inverse residual).")
    ap.add_argument("--max_quads", type=int, default=9999, help="Max quad decompositions to report (sorted by residual).")
    args = ap.parse_args()

    safe_mkdir(args.outdir)

    plane_arg = args.plane.strip() or None
    plane, loops, meta = load_phase16a_loops(args.in_json, plane_arg=plane_arg, threshold=args.threshold, min_sim=args.min_sim)

    if not loops:
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

    # Build map from canonical cycle -> representative entry
    loop_map = {}
    for L in loops:
        nodes = L.get("loop", [])
        if not nodes:
            continue
        key = canonical_cycle(nodes)
        # keep highest rank_score if duplicates
        if key not in loop_map:
            loop_map[key] = L
        else:
            a = float(loop_map[key].get("rank_score", -1e9))
            b = float(L.get("rank_score", -1e9))
            if b > a:
                loop_map[key] = L

    # ---------------- Reverse pairing ----------------
    pairs = []
    used = set()
    for key, entry in loop_map.items():
        nodes = list(key)
        if key in used:
            continue
        rev_nodes = reversed_cycle(nodes)
        rev_key = canonical_cycle(rev_nodes)
        if rev_key == key:
            continue
        if rev_key in loop_map:
            used.add(key)
            used.add(rev_key)
            A = loop_map[key]
            B = loop_map[rev_key]
            Ra = get_R(A)
            Rb = get_R(B)
            if Ra is None or Rb is None:
                continue
            chk = reverse_pair_checks(Ra, Rb)
            pairs.append({
                "loop_a": A.get("loop"),
                "loop_b": B.get("loop"),
                "edge_sim_min_a": float(A.get("edge_sim_min", float("nan"))),
                "edge_sim_min_b": float(B.get("edge_sim_min", float("nan"))),
                "rank_score_a": float(A.get("rank_score", float("nan"))),
                "rank_score_b": float(B.get("rank_score", float("nan"))),
                "checks": chk,
            })

    pairs.sort(key=lambda p: p["checks"]["inv_resid_fro"])
    pairs = pairs[:max(0, args.max_pairs)]

    # ---------------- Quad decomposition ----------------
    quad_reports = []
    for key, entry in loop_map.items():
        nodes = list(key)
        if len(nodes) != 4:
            continue
        rep = quad_decomposition_checks(loop_map, nodes)
        if rep is None:
            continue
        if rep.get("available", False):
            quad_reports.append(rep)

    quad_reports.sort(key=lambda r: r["best"]["resid_fro"] if r.get("available") else 1e9)
    quad_reports = quad_reports[:max(0, args.max_quads)]

    # ---------------- Save report ----------------
    report = {
        "phase": "17c",
        "plane": plane,
        "meta": meta,
        "counts": {
            "loops_in_bucket": len(loops),
            "unique_cycles": len(loop_map),
            "reverse_pairs_found": len(pairs),
            "quad_decompositions_found": len(quad_reports),
        },
        "reverse_pairs": pairs,
        "quad_decompositions": quad_reports,
    }

    out_json = os.path.join(args.outdir, f"phase17c_sanity_{plane}.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    # ---------------- Console summary ----------------
    print(f"[plane={plane}] loops={len(loops)} unique_cycles={len(loop_map)}")
    print(f"reverse_pairs_found={len(pairs)} quad_decompositions_found={len(quad_reports)}")
    if pairs:
        best = pairs[0]
        chk = best["checks"]
        print("\n[best reverse-pair]")
        print("  A:", " -> ".join(best["loop_a"] + [best["loop_a"][0]]))
        print("  B:", " -> ".join(best["loop_b"] + [best["loop_b"][0]]))
        print(f"  inv_resid_fro={chk['inv_resid_fro']:.6e}  angle_diff={chk['abs_angle_diff_deg']:.6e}  det_a={chk['det_a']:.6f} det_b={chk['det_b']:.6f}")
        if "axis_dot_abs" in chk:
            print(f"  axis_dot_abs={chk['axis_dot_abs']:.6f} (want ~1.0; sign flip allowed)")
    if quad_reports:
        bestq = quad_reports[0]
        print("\n[best quad decomposition]")
        q = bestq["quad"]
        b = bestq["best"]
        print("  quad:", " -> ".join(q + [q[0]]))
        print("  best decomposition:", b["decomposition"], f"resid_fro={b['resid_fro']:.6e}")

    print(f"\n[saved] {out_json}")

if __name__ == "__main__":
    main()
