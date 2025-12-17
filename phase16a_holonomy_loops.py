#!/usr/bin/env python3
# phase16a_holonomy_loops.py
#
# Phase 16a: Discrete connection / holonomy test.
# Compose transport maps around closed loops and measure deviation from identity.
#
# NEW (multi-threshold mode):
#   --min_sim_list "0.30,0.35,0.40" filters loops by edge_sim_min >= threshold,
#   producing per-threshold "top" results and a combined JSON report.

import os, json, math, argparse
import numpy as np
from itertools import combinations, permutations

# ---------------------------- utilities ----------------------------

def fro_norm(A):
    return float(np.linalg.norm(A, ord="fro"))

def safe_trace_angle(R):
    """
    If R is (approximately) a proper rotation, angle can be approximated from trace:
      cos(theta) = (tr(R) - 1) / 2   (3D)
      cos(theta) = tr(R)/2          (2D)
    For general k, we still compute a "trace angle proxy" but treat it cautiously.
    """
    k = R.shape[0]
    tr = float(np.trace(R))
    det = float(np.linalg.det(R))
    if k == 2:
        c = max(-1.0, min(1.0, tr / 2.0))
        theta = math.degrees(math.acos(c))
        return {"trace": tr, "det": det, "angle_deg_proxy": theta, "note": "2D trace angle"}
    elif k == 3:
        c = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
        theta = math.degrees(math.acos(c))
        return {"trace": tr, "det": det, "angle_deg_proxy": theta, "note": "3D trace angle (assumes det≈+1)"}
    else:
        c = max(-1.0, min(1.0, (tr / k)))
        theta = math.degrees(math.acos(c))
        return {"trace": tr, "det": det, "angle_deg_proxy": theta, "note": f"{k}D proxy from tr/k"}

def eig_summary(R):
    vals = np.linalg.eigvals(R)
    out = []
    for z in vals:
        out.append({
            "re": float(np.real(z)),
            "im": float(np.imag(z)),
            "abs": float(np.abs(z))
        })
    return out

def mat_from_entry(entry):
    """
    Supports both 15e-style and 15f-style caches.
    Prefer 'R' if present.
    """
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
    Fetch A__to__B plane entry, else try reverse and invert (transpose) if it exists.
    Returns (R, sim, direction_used)
    """
    maps = cache.get("transport_maps", {})
    key = f"{a}__to__{b}"
    if key in maps and plane in maps[key]:
        entry = maps[key][plane]
        R = mat_from_entry(entry)
        return R, sim_from_entry(entry), "forward"

    # try reverse
    key2 = f"{b}__to__{a}"
    if key2 in maps and plane in maps[key2]:
        entry = maps[key2][plane]
        R = mat_from_entry(entry)
        if R is None:
            return None, None, "missing"
        Rinv = R.T  # orthonormal inverse approx
        sim = sim_from_entry(entry)
        return Rinv, sim, "reverse_used_transpose"

    return None, None, "missing"

def compose_loop(cache, loop_nodes, plane):
    """
    loop_nodes: list like [A,B,C] meaning closed loop A->B->C->A
    Returns dict with R_loop and diagnostics, or None if missing edges.
    """
    assert len(loop_nodes) >= 2
    nodes = list(loop_nodes)
    nodes_closed = nodes + [nodes[0]]

    R_loop = None
    edge_info = []

    for i in range(len(nodes_closed) - 1):
        a = nodes_closed[i]
        b = nodes_closed[i+1]
        R, sim, how = get_edge(cache, a, b, plane)
        if R is None:
            return None
        edge_info.append({"a": a, "b": b, "sim": sim, "how": how, "k": int(R.shape[0])})
        if R_loop is None:
            R_loop = R.copy()
        else:
            # compose in path order: R_{a->b} then R_{b->c} etc
            R_loop = R @ R_loop

    k = int(R_loop.shape[0])
    I = np.eye(k, dtype=np.float64)
    delta = R_loop - I

    angle = safe_trace_angle(R_loop)
    det = float(np.linalg.det(R_loop))

    sims = [e["sim"] for e in edge_info]
    return {
        "plane": plane,
        "loop": nodes,
        "k": k,
        "R_loop": R_loop.tolist(),
        "fro_norm_R_minus_I": fro_norm(delta),
        "det_R_loop": det,
        "trace_angle": angle,
        "eigvals": eig_summary(R_loop),
        "edges": edge_info,
        "edge_sim_min": float(np.nanmin(sims)),
        "edge_sim_mean": float(np.nanmean(sims)),
        "edge_sim_max": float(np.nanmax(sims)),
    }

def enumerate_candidate_loops(names, max_len=3):
    """
    Generates loops of length 3..max_len (unique node lists).
    For length=3, loops are (A,B,C). For length=4, (A,B,C,D), etc.
    """
    out = []
    if max_len < 3:
        return out
    for L in range(3, max_len + 1):
        for comb in combinations(names, L):
            first = comb[0]     # fix first to avoid rotational duplicates
            rest = comb[1:]
            for perm in permutations(rest):
                out.append([first] + list(perm))
    return out

def parse_min_sim_list(s):
    if s is None:
        return []
    s = s.strip()
    if not s:
        return []
    vals = []
    for part in s.split(","):
        p = part.strip()
        if not p:
            continue
        vals.append(float(p))
    # stable, sorted unique
    vals = sorted(set(vals))
    return vals

# ---------------------------- main ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="Path to phase15f_transport_cache.json")
    ap.add_argument("--plane", default="bo", choices=["bo", "lt", "bo_lr"], help="Which frame to test")

    ap.add_argument("--loop", default="", help='Comma list: "A,B,C" (implies A->B->C->A). If set, runs single-loop mode.')
    ap.add_argument("--max_len", type=int, default=3, help="When auto-searching, max loop length (>=3).")
    ap.add_argument("--top", type=int, default=25, help="Top N loops to save/print per mode.")

    # Single-threshold filter (legacy) + multi-threshold filter (new)
    ap.add_argument("--min_sim", type=float, default=float("nan"),
                    help="(Optional) In auto mode, filter loops by edge_sim_min >= this value.")
    ap.add_argument("--min_sim_list", default="",
                    help='(Optional) Multi-threshold auto mode. Example: --min_sim_list "0.30,0.35,0.40"')

    ap.add_argument("--out", default="", help="Optional output JSON path (default next to cache).")
    args = ap.parse_args()

    with open(args.cache, "r") as f:
        cache = json.load(f)

    # Build anchor/name list
    names = sorted(list(cache.get("anchors", {}).keys()))
    if not names:
        maps = cache.get("transport_maps", {})
        seen = set()
        for k in maps.keys():
            if "__to__" in k:
                a, b = k.split("__to__")
                seen.add(a); seen.add(b)
        names = sorted(list(seen))

    # ---------------- single-loop mode ----------------
    if args.loop.strip():
        loop_nodes = [s.strip() for s in args.loop.split(",") if s.strip()]
        res = compose_loop(cache, loop_nodes, args.plane)
        if res is None:
            raise SystemExit(f"[error] Loop invalid / missing edges for plane={args.plane}: {loop_nodes}")

        print(json.dumps({
            "plane": res["plane"],
            "loop": res["loop"],
            "k": res["k"],
            "fro_norm_R_minus_I": res["fro_norm_R_minus_I"],
            "det_R_loop": res["det_R_loop"],
            "trace_angle": res["trace_angle"],
            "edge_sim_min": res["edge_sim_min"],
            "edge_sim_mean": res["edge_sim_mean"],
            "edge_sim_max": res["edge_sim_max"],
        }, indent=2))

        outpath = args.out or os.path.join(os.path.dirname(args.cache), f"phase16a_holonomy_{args.plane}.json")
        with open(outpath, "w") as g:
            json.dump({"results": [res]}, g, indent=2)
        print(f"[saved] {outpath}")
        return

    # ---------------- auto-search (base enumeration) ----------------
    cand = enumerate_candidate_loops(names, max_len=max(3, args.max_len))
    base_results = []
    for loop_nodes in cand:
        res = compose_loop(cache, loop_nodes, args.plane)
        if res is None:
            continue
        sim_mean = res["edge_sim_mean"]
        hol = res["fro_norm_R_minus_I"]
        score = (sim_mean + 1e-9) * (hol + 1e-9)
        res["rank_score"] = float(score)
        base_results.append(res)

    if not base_results:
        raise SystemExit(f"[error] No valid loops found for plane={args.plane}. (Edges missing?)")

    base_results.sort(key=lambda r: r["rank_score"], reverse=True)

    # ---------------- multi-threshold mode ----------------
    thresholds = parse_min_sim_list(args.min_sim_list)
    if thresholds:
        thresh_out = {}
        for t in thresholds:
            kept = [r for r in base_results if r["edge_sim_min"] >= t]
            topN = kept[:max(1, args.top)] if kept else []
            key = f"{t:.2f}"
            thresh_out[key] = {
                "min_sim": float(t),
                "kept_count": int(len(kept)),
                "top": topN
            }

        # Print quick summary
        print(f"[plane={args.plane}] total_valid={len(base_results)}  max_len={args.max_len}")
        for k, v in thresh_out.items():
            print(f"  threshold>={k}: kept={v['kept_count']}  showing_top={min(len(v['top']), args.top)}")
            if v["top"]:
                r0 = v["top"][0]
                loop_str = " -> ".join(r0["loop"] + [r0["loop"][0]])
                print(f"    best: score={r0['rank_score']:.6f} ||R-I||_F={r0['fro_norm_R_minus_I']:.6f} "
                      f"det={r0['det_R_loop']:.4f} sim_min={r0['edge_sim_min']:.4f} sim_mean={r0['edge_sim_mean']:.4f} "
                      f"loop: {loop_str}")

        out_obj = {
            "plane": args.plane,
            "max_len": int(args.max_len),
            "top": int(args.top),
            "total_valid": int(len(base_results)),
            "threshold_results": thresh_out,
        }

        outpath = args.out or os.path.join(os.path.dirname(args.cache), f"phase16a_holonomy_{args.plane}.json")
        with open(outpath, "w") as g:
            json.dump(out_obj, g, indent=2)
        print(f"\n[saved] {outpath}")
        return

    # ---------------- single-threshold (legacy) OR no-threshold ----------------
    filtered = base_results
    if not math.isnan(args.min_sim):
        filtered = [r for r in base_results if r["edge_sim_min"] >= args.min_sim]

    if not filtered:
        raise SystemExit(f"[error] No loops remain after filtering for plane={args.plane} min_sim={args.min_sim}")

    topN = filtered[:max(1, args.top)]

    print(f"[plane={args.plane}] valid_loops={len(base_results)} kept={len(filtered)} showing_top={len(topN)}\n")
    for i, r in enumerate(topN, 1):
        loop_str = " -> ".join(r["loop"] + [r["loop"][0]])
        print(f"{i:02d}. score={r['rank_score']:.6f}  ||R-I||_F={r['fro_norm_R_minus_I']:.6f}  "
              f"det={r['det_R_loop']:.4f}  sim_min={r['edge_sim_min']:.4f}  sim_mean={r['edge_sim_mean']:.4f}  loop: {loop_str}")

    outpath = args.out or os.path.join(os.path.dirname(args.cache), f"phase16a_holonomy_{args.plane}.json")
    with open(outpath, "w") as g:
        json.dump({
            "plane": args.plane,
            "max_len": int(args.max_len),
            "top": topN,
            "all_count": int(len(base_results)),
            "kept_count": int(len(filtered)),
            "min_sim": None if math.isnan(args.min_sim) else float(args.min_sim),
        }, g, indent=2)
    print(f"\n[saved] {outpath}")

if __name__ == "__main__":
    main()
