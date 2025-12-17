#!/usr/bin/env python3
# phase26e_geodesic_field_stability_sweep.py
#
# Sweep (w_curv, w_switch) settings and measure attractor stability vs baseline.
#
# Uses phase15f transport cache:
#   transport_maps[edge_key][plane] -> payload
# payload expected to include:
#   similarity (float)
#   principal_cosines (list[float]) optional
#   k (int) optional
#   match_score_abs_diag_sum (float) optional
#
# Edge cost (per step):
#   cost = w_transport*transport + w_curv*curv_norm + w_switch*switch + w_incoh*incoh + w_unknown*unknown
#
# transport = 1 - similarity
# curv = sum(acos(principal_cosines))
# curv_norm = curv / curv_scale[plane]  (median/mean across all edges in that plane)
# cons in [0..1] = match_score_abs_diag_sum/k if present else similarity
# incoh = 1 - cons (or 1.0 if unknown)
# unknown = 1 if no principal_cosines else 0
#
# Rollouts:
#   - For each start node (all nodes in cache), do greedy rollout choosing the lowest-cost next edge
#   - Optional no-backtrack constraint
#   - Cycle detection:
#       if repetition happens at node nb, cycle is path[j:-1] (exclude final repeated nb)
#       then reduce to smallest period
#       canonicalize by rotation (lexicographically smallest rotation)
#
# Outputs:
#   phase26e_stability_sweep.csv
#   phase26e_stability_sweep.json
#
import os, json, math, argparse
from collections import defaultdict

# -------------------------
# Basic I/O
# -------------------------

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

# -------------------------
# Cache parsing
# -------------------------

def parse_edge_key(k):
    """
    Supports:
      a__to__b
      a->b
      a|b
      a__b
    Returns (a,b) or None.
    """
    if "__to__" in k:
        a, b = k.split("__to__", 1)
        return a, b
    if "->" in k:
        a, b = k.split("->", 1)
        return a, b
    if "|" in k:
        a, b = k.split("|", 1)
        return a, b
    if "__" in k:
        a, b = k.split("__", 1)
        return a, b
    return None

def collect_nodes(transport_maps):
    nodes = set()
    for k in transport_maps.keys():
        parsed = parse_edge_key(k)
        if not parsed:
            continue
        a, b = parsed
        nodes.add(a); nodes.add(b)
    return sorted(nodes)

# -------------------------
# Geometry terms
# -------------------------

def curvature_from_principal_cosines(pc):
    s = 0.0
    for c in pc:
        c = clamp(float(c), -1.0, 1.0)
        s += math.acos(c)
    return s

def consistency_from_payload(payload):
    """
    Returns a 0..1 consistency.
    Prefer match_score_abs_diag_sum / k if present; else similarity if present; else None.
    """
    if not isinstance(payload, dict):
        return None
    if "match_score_abs_diag_sum" in payload and "k" in payload:
        k = payload.get("k", None)
        if k and k > 0:
            return float(payload["match_score_abs_diag_sum"]) / float(k)
    if "similarity" in payload:
        return float(payload["similarity"])
    return None

def edge_transport(sim):
    return 1.0 - float(sim)

def edge_incoherence(cons):
    if cons is None:
        return 1.0
    return clamp(1.0 - float(cons), 0.0, 1.0)

def compute_plane_curv_scale(transport_maps, planes, norm):
    """
    norm in {"none","median","mean"}.
    Scale computed over ALL edges in cache for each plane that have principal_cosines.
    """
    if norm == "none":
        return {p: 1.0 for p in planes}

    vals_by_plane = {p: [] for p in planes}
    for _, by_plane in transport_maps.items():
        if not isinstance(by_plane, dict):
            continue
        for p in planes:
            payload = by_plane.get(p, None)
            if not isinstance(payload, dict):
                continue
            pcs = payload.get("principal_cosines", None)
            if not pcs:
                continue
            vals_by_plane[p].append(curvature_from_principal_cosines(pcs))

    scales = {}
    for p, vals in vals_by_plane.items():
        if not vals:
            scales[p] = 1.0
            continue
        vals = sorted(vals)
        if norm == "median":
            mid = len(vals) // 2
            if len(vals) % 2 == 1:
                scales[p] = float(vals[mid])
            else:
                scales[p] = 0.5 * (float(vals[mid-1]) + float(vals[mid]))
        elif norm == "mean":
            scales[p] = float(sum(vals) / len(vals))
        else:
            scales[p] = 1.0
        if scales[p] <= 1e-12:
            scales[p] = 1.0
    return scales

# -------------------------
# Graph building
# -------------------------

def build_adjacency(transport_maps, planes, min_edge_sim):
    """
    adj[a] -> list of dict edges:
      {a,b,plane,sim,cons,transport,curv,curv_norm,incoh,unknown}
    """
    adj = defaultdict(list)
    for k, by_plane in transport_maps.items():
        parsed = parse_edge_key(k)
        if not parsed:
            continue
        a, b = parsed
        if not isinstance(by_plane, dict):
            continue

        for plane in planes:
            payload = by_plane.get(plane, None)
            if not isinstance(payload, dict):
                continue
            sim = float(payload.get("similarity", -1.0))
            if sim < min_edge_sim:
                continue

            pcs = payload.get("principal_cosines", None)
            cons = consistency_from_payload(payload)
            transport = edge_transport(sim)
            incoh = edge_incoherence(cons)

            if pcs:
                curv = curvature_from_principal_cosines(pcs)
                unknown = 0
            else:
                curv = None
                unknown = 1

            adj[a].append({
                "a": a, "b": b, "plane": plane,
                "sim": sim, "cons": cons,
                "transport": transport,
                "curv": curv,
                "incoh": incoh,
                "unknown": unknown,
            })
    return adj

# -------------------------
# Minimal-cycle attractor fix (FULL)
# -------------------------

def _minimal_period(seq):
    """
    Return smallest-period version of a cyclic sequence.
    Example: [A,B,A,B] -> [A,B]
    """
    n = len(seq)
    if n <= 1:
        return seq
    for p in range(1, n + 1):
        if n % p != 0:
            continue
        ok = True
        for i in range(n):
            if seq[i] != seq[i % p]:
                ok = False
                break
        if ok:
            return seq[:p]
    return seq

def extract_minimal_cycle(path_nodes, repeated_node):
    """
    If repetition happens at nb, and nb was first seen at index j:
      cycle = path_nodes[j:-1]   (exclude the final repeated nb)
    Then reduce to smallest period.
    """
    if not path_nodes:
        return []
    try:
        j = path_nodes.index(repeated_node)
    except ValueError:
        return []
    cycle = path_nodes[j:-1]  # critical fix
    if not cycle:
        cycle = [repeated_node]
    cycle = _minimal_period(cycle)
    return cycle

def canonical_cycle_key(cycle_nodes):
    """
    Rotation-invariant ordered key: choose lexicographically smallest rotation.
    """
    if not cycle_nodes:
        return ""
    n = len(cycle_nodes)
    best = None
    for r in range(n):
        rot = cycle_nodes[r:] + cycle_nodes[:r]
        tup = tuple(rot)
        if best is None or tup < best:
            best = tup
    return "|".join(best)

# -------------------------
# Rollout / attractors
# -------------------------

def edge_cost(edge, last_plane, curv_scale, w_transport, w_curv, w_switch, w_incoh, w_unknown):
    # switch penalty depends on last_plane
    sw = 0
    if last_plane is not None and edge["plane"] != last_plane:
        sw = 1

    # curvature normalized if available; else 0 + unknown penalty
    if edge["curv"] is None:
        curv_norm = 0.0
        unknown = 1
    else:
        denom = float(curv_scale.get(edge["plane"], 1.0))
        if denom <= 1e-12:
            denom = 1.0
        curv_norm = float(edge["curv"]) / denom
        unknown = 0

    return (w_transport * edge["transport"]
            + w_curv * curv_norm
            + w_switch * sw
            + w_incoh * edge["incoh"]
            + w_unknown * unknown)

def greedy_rollout(start, adj, curv_scale,
                  w_transport, w_curv, w_switch, w_incoh, w_unknown,
                  max_steps, topk_per_node,
                  no_backtrack):
    """
    Returns dict with:
      status: "cycle" or "deadend" or "max_steps"
      path_nodes: list[str]
      attractor_key: str or None
      attractor_size: int or None
    """
    path_nodes = [start]
    seen = {start: 0}  # node -> first index in path_nodes
    last_plane = None

    for _ in range(max_steps):
        cur = path_nodes[-1]
        cands = adj.get(cur, [])
        if not cands:
            return {"status": "deadend", "path_nodes": path_nodes,
                    "attractor_key": None, "attractor_size": None}

        # score candidates; optionally forbid immediate backtrack
        prev_node = path_nodes[-2] if len(path_nodes) >= 2 else None

        scored = []
        for e in cands:
            if no_backtrack and prev_node is not None and e["b"] == prev_node:
                continue
            c = edge_cost(e, last_plane, curv_scale, w_transport, w_curv, w_switch, w_incoh, w_unknown)
            scored.append((c, e))

        if not scored:
            return {"status": "deadend", "path_nodes": path_nodes,
                    "attractor_key": None, "attractor_size": None}

        scored.sort(key=lambda x: x[0])
        scored = scored[:max(1, int(topk_per_node))]

        # greedy: pick best
        best_cost, best_edge = scored[0]
        nb = best_edge["b"]

        # cycle detection
        if nb in seen:
            path_nodes.append(nb)  # include the repeated node once for slicing rule
            cycle_nodes = extract_minimal_cycle(path_nodes, nb)
            key = canonical_cycle_key(cycle_nodes)
            return {"status": "cycle", "path_nodes": path_nodes,
                    "attractor_key": key, "attractor_size": len(cycle_nodes)}

        # advance
        seen[nb] = len(path_nodes)
        path_nodes.append(nb)
        last_plane = best_edge["plane"]

    return {"status": "max_steps", "path_nodes": path_nodes,
            "attractor_key": None, "attractor_size": None}

# -------------------------
# Sweep
# -------------------------

def attractor_signature(attractor_counts):
    """
    Build a stable signature string like:
      key1:count|key2:count|...
    keys sorted.
    """
    items = sorted(attractor_counts.items(), key=lambda kv: kv[0])
    return "|".join([f"{k}:{v}" for (k, v) in items])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--planes", nargs="+", required=True)
    ap.add_argument("--min_edge_sim", type=float, default=0.10)

    ap.add_argument("--max_steps", type=int, default=30)
    ap.add_argument("--topk_per_node", type=int, default=5)
    ap.add_argument("--no_backtrack", action="store_true")

    ap.add_argument("--w_transport", type=float, default=1.0)
    ap.add_argument("--w_incoh", type=float, default=0.5)
    ap.add_argument("--w_unknown", type=float, default=0.75)

    # normalization
    ap.add_argument("--curv_norm", choices=["none", "median", "mean"], default="median")

    # baseline for agreement
    ap.add_argument("--baseline_w_curv", type=float, default=1.0)
    ap.add_argument("--baseline_w_switch", type=float, default=0.25)

    # sweep grids (defaults match what you used)
    ap.add_argument("--sweep_w_curv", nargs="+", type=float,
                    default=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    ap.add_argument("--sweep_w_switch", nargs="+", type=float,
                    default=[0.0, 0.125, 0.25, 0.5, 1.0])

    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()
    ensure_dir(args.outdir)

    d = load_json(args.cache)
    if "transport_maps" not in d:
        raise SystemExit("[error] cache missing transport_maps")
    tm = d["transport_maps"]

    nodes = collect_nodes(tm)
    planes = args.planes

    curv_scale = compute_plane_curv_scale(tm, planes, args.curv_norm)
    adj = build_adjacency(tm, planes, args.min_edge_sim)

    if args.debug:
        print("[debug] nodes:", nodes)
        for p in planes:
            print(f"[debug] curv_scale[{p}] =", curv_scale.get(p, None))
        print("[debug] total nodes with outgoing edges:", sum(1 for n in nodes if adj.get(n)))

    # --- baseline attractors per start ---
    baseline_map = {}
    baseline_counts = defaultdict(int)
    for s in nodes:
        res = greedy_rollout(
            s, adj, curv_scale,
            args.w_transport, args.baseline_w_curv, args.baseline_w_switch, args.w_incoh, args.w_unknown,
            args.max_steps, args.topk_per_node,
            args.no_backtrack
        )
        ak = res["attractor_key"] if res["status"] == "cycle" else None
        baseline_map[s] = ak
        if ak:
            baseline_counts[ak] += 1

    baseline_sig = attractor_signature(baseline_counts)

    # --- sweep ---
    rows = []
    sweep_results = []

    for w_curv in args.sweep_w_curv:
        for w_switch in args.sweep_w_switch:
            attr_counts = defaultdict(int)
            this_map = {}
            for s in nodes:
                res = greedy_rollout(
                    s, adj, curv_scale,
                    args.w_transport, w_curv, w_switch, args.w_incoh, args.w_unknown,
                    args.max_steps, args.topk_per_node,
                    args.no_backtrack
                )
                ak = res["attractor_key"] if res["status"] == "cycle" else None
                this_map[s] = ak
                if ak:
                    attr_counts[ak] += 1

            # agreement vs baseline
            agree = 0
            for s in nodes:
                if this_map.get(s) == baseline_map.get(s):
                    agree += 1
            agree_frac = float(agree) / float(len(nodes)) if nodes else 0.0

            sig = attractor_signature(attr_counts)
            n_attr = len(attr_counts)

            row = {
                "w_curv": float(w_curv),
                "w_switch": float(w_switch),
                "agree_frac_vs_base": float(agree_frac),
                "n_attractors": int(n_attr),
                "attractor_sig": sig,
            }
            rows.append(row)
            sweep_results.append({
                "params": {"w_curv": float(w_curv), "w_switch": float(w_switch)},
                "agree_frac_vs_base": float(agree_frac),
                "attractor_counts": dict(attr_counts),
                "attractor_sig": sig,
                "per_start_attractor": this_map,
            })

    # sort for printing
    rows_sorted = sorted(rows, key=lambda r: (-r["agree_frac_vs_base"], r["w_curv"], r["w_switch"]))

    # write CSV
    out_csv = os.path.join(args.outdir, "phase26e_stability_sweep.csv")
    with open(out_csv, "w", newline="") as f:
        f.write("w_curv,w_switch,agree_frac_vs_base,n_attractors,attractor_sig\n")
        for r in rows:
            f.write(f"{r['w_curv']},{r['w_switch']},{r['agree_frac_vs_base']},{r['n_attractors']},{r['attractor_sig']}\n")

    # write JSON
    out_json = os.path.join(args.outdir, "phase26e_stability_sweep.json")
    out = {
        "phase": "26e",
        "status": "ok",
        "args": vars(args),
        "planes": planes,
        "curv_norm": args.curv_norm,
        "curv_scale": curv_scale,
        "nodes": nodes,
        "baseline": {
            "w_curv": args.baseline_w_curv,
            "w_switch": args.baseline_w_switch,
            "signature": baseline_sig,
            "counts": dict(baseline_counts),
            "per_start_attractor": baseline_map,
        },
        "sweep": sweep_results,
    }
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)

    # console
    print("Phase26e Stability Sweep")
    print(f"planes={planes} curv_norm={args.curv_norm} min_edge_sim={args.min_edge_sim} topk_per_node={args.topk_per_node}")
    print("curv_scale:", curv_scale)
    print(f"baseline: w_curv={args.baseline_w_curv} w_switch={args.baseline_w_switch}")
    print(f"baseline_sig: {baseline_sig}")

    print("\nTOP 10 MOST STABLE SETTINGS (agreement vs baseline):")
    for i, r in enumerate(rows_sorted[:10], 1):
        print(f"[{i}] agree={r['agree_frac_vs_base']:.3f}  w_curv={r['w_curv']:.4f}  w_switch={r['w_switch']:.4f}  n_attr={r['n_attractors']}  sig={r['attractor_sig']}")

    print(f"\n[ok] wrote: {out_csv}")
    print(f"[ok] wrote: {out_json}")

if __name__ == "__main__":
    main()
