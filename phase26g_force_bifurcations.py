#!/usr/bin/env python3
# phase26g_force_bifurcations.py
#
# Novelty rollouts to force bifurcations in the geodesic field.
# Robust edge loading from cache JSON:
#   - supports transport_maps keyed like "A__to__B"
#   - supports plane dicts under:
#       transport_maps[pair][plane]
#       transport_maps[pair]["planes"][plane]
#   - SIM is stored as "similarity" in your cache (not "sim")
#
# Cycle fix:
#   when repetition happens at node nb, minimal recurrent cycle is path[j:] (exclude final repeated nb),
#   then reduce to smallest period.

import os, json, csv, argparse
from collections import defaultdict, Counter

# --------------------------
# Cycle canonicalization
# --------------------------
def _min_rotation(seq):
    n = len(seq)
    if n == 0:
        return seq
    best = None
    for s in range(n):
        rot = tuple(seq[s:] + seq[:s])
        if best is None or rot < best:
            best = rot
    return list(best)

def _reduce_smallest_period(cycle):
    n = len(cycle)
    if n <= 1:
        return cycle
    for k in range(1, n + 1):
        if n % k != 0:
            continue
        base = cycle[:k]
        ok = True
        for i in range(n):
            if cycle[i] != base[i % k]:
                ok = False
                break
        if ok:
            return base
    return cycle

def canonical_cycle_key(cycle_nodes):
    c = _reduce_smallest_period(list(cycle_nodes))
    c = _min_rotation(c)
    return "|".join(c), c

# --------------------------
# JSON + edge helpers
# --------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _extract_metric(d, name):
    """
    Extract metric 'name' from possibly nested dicts:
      d[name]
      d["stats"][name]
      d["metrics"][name]
      d["summary"][name]
    """
    if not isinstance(d, dict):
        return None
    if name in d:
        return d.get(name)
    for box in ("stats", "metrics", "summary"):
        sub = d.get(box)
        if isinstance(sub, dict) and name in sub:
            return sub.get(name)
    return None

def _edge_dict(a, b, plane, sim, cons=None, transport=None, curv=None, curv_norm=None, incoh=None, unknown=None):
    sim = float(sim)
    cons = float(cons) if cons is not None else sim
    if transport is None:
        transport = 1.0 - sim
    incoh = float(incoh) if incoh is not None else (1.0 - sim)
    unknown = float(unknown) if unknown is not None else 0.0
    curv = float(curv) if curv is not None else 0.0
    curv_norm = float(curv_norm) if curv_norm is not None else 0.0
    return {
        "a": a, "b": b, "plane": plane,
        "sim": sim,
        "cons": cons,
        "transport": float(transport),
        "curv": curv,
        "curv_norm": curv_norm,
        "incoh": incoh,
        "unknown": unknown,
    }

def _parse_pair_key(k):
    if "__to__" in k:
        a, b = k.split("__to__", 1)
        return a, b
    if "->" in k:
        a, b = k.split("->", 1)
        return a, b
    return None, None

def derive_edges_from_transport_maps(data, planes, min_edge_sim, debug=False):
    tm = data.get("transport_maps", {})
    edges = []

    for pair_key, v in tm.items():
        if not isinstance(v, dict):
            continue
        a, b = _parse_pair_key(pair_key)
        if a is None or b is None:
            continue

        plane_block = v.get("planes") if isinstance(v.get("planes"), dict) else v

        for plane in planes:
            d = plane_block.get(plane) if isinstance(plane_block, dict) else None
            if not isinstance(d, dict):
                continue

            # ---- SIM extraction (your cache uses "similarity") ----
            sim = (
                _extract_metric(d, "sim")
                or _extract_metric(d, "cos_sim")
                or _extract_metric(d, "similarity")          # <-- KEY FIX
            )
            if sim is None:
                continue

            sim = float(sim)
            if sim < min_edge_sim:
                continue

            # best-effort extras (usually absent in this cache)
            cons = _extract_metric(d, "cons")
            transport = _extract_metric(d, "transport")
            curv = _extract_metric(d, "curv")
            curv_norm = _extract_metric(d, "curv_norm") or _extract_metric(d, "curvNorm")
            incoh = _extract_metric(d, "incoh")
            unknown = _extract_metric(d, "unknown")

            edges.append(_edge_dict(
                a=a, b=b, plane=plane,
                sim=sim, cons=cons, transport=transport,
                curv=curv, curv_norm=curv_norm,
                incoh=incoh, unknown=unknown
            ))

    if debug:
        print(f"[debug] derive_edges_from_transport_maps: produced {len(edges)} edges")
    return edges

def load_edges(cache_path, planes, min_edge_sim, debug=False):
    data = load_json(cache_path)

    # We primarily derive from transport_maps for this cache format.
    edges = derive_edges_from_transport_maps(data, planes, min_edge_sim, debug=debug)

    edges_by_a = defaultdict(list)
    for e in edges:
        edges_by_a[e["a"]].append(e)

    if debug:
        nodes_with_outgoing = sum(1 for a, lst in edges_by_a.items() if lst)
        print(f"[debug] derived edges: {len(edges)} | nodes_with_outgoing: {nodes_with_outgoing}")
        print(f"[debug] cache top keys: {list(data.keys())[:20]}")
        if "transport_maps" in data:
            tm_keys = list(data["transport_maps"].keys())[:8]
            print(f"[debug] transport_maps top keys sample: {tm_keys}")

    return edges_by_a, data

# --------------------------
# Scoring + rollout
# --------------------------
def base_cost(edge, prev_plane, w_transport, w_curv, w_switch, w_incoh, w_unknown):
    switch = 0.0
    if prev_plane is not None and edge["plane"] != prev_plane:
        switch = 1.0
    return (
        w_transport * edge["transport"]
        + w_curv * edge.get("curv_norm", 0.0)
        + w_switch * switch
        + w_incoh * edge.get("incoh", 0.0)
        + w_unknown * edge.get("unknown", 0.0)
    )

def rollout(edges_by_a, start, max_steps, topk_per_node,
            w_transport, w_curv, w_switch, w_incoh, w_unknown,
            w_novelty, no_backtrack=True):
    """
    Greedy rollout:
      score = base_cost + w_novelty * nov
      nov = 1 if dst already visited else 0  (penalize revisits)
    Cycle detection:
      if nb repeats at index j, cycle_nodes = path[j:]  (exclude final repeated nb)
      then reduce to smallest period
    """
    path = [start]
    idx = {start: 0}
    prev_plane = None

    for _t in range(max_steps):
        cur = path[-1]
        cands = edges_by_a.get(cur, [])
        if not cands:
            return {"status": "deadend", "path": path, "cycle": None}

        back = None
        if no_backtrack and len(path) >= 2:
            back = path[-2]

        scored = []
        for e in cands:
            if back is not None and e["b"] == back:
                continue
            nov = 1.0 if (e["b"] in idx) else 0.0
            bc = base_cost(e, prev_plane, w_transport, w_curv, w_switch, w_incoh, w_unknown)
            sc = bc + w_novelty * nov
            scored.append((sc, bc, nov, e))

        if not scored:
            return {"status": "deadend", "path": path, "cycle": None}

        scored.sort(key=lambda x: x[0])
        scored = scored[:max(1, int(topk_per_node))]

        sc, bc, nov, e = scored[0]
        nb = e["b"]

        if nb in idx:
            j = idx[nb]
            cycle_nodes = path[j:]   # EXCLUDE the final repeated nb
            cycle_key, canon_cycle = canonical_cycle_key(cycle_nodes)
            return {"status": "cycle", "path": path + [nb], "cycle": canon_cycle, "cycle_key": cycle_key}

        path.append(nb)
        idx[nb] = len(path) - 1
        prev_plane = e["plane"]

    return {"status": "max_steps", "path": path, "cycle": None}

# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--planes", nargs="+", required=True)
    ap.add_argument("--min_edge_sim", type=float, default=0.10)
    ap.add_argument("--topk_per_node", type=int, default=5)
    ap.add_argument("--max_steps", type=int, default=30)

    ap.add_argument("--w_transport", type=float, default=1.0)
    ap.add_argument("--w_curv", type=float, default=1.0)
    ap.add_argument("--w_switch", type=float, default=0.25)
    ap.add_argument("--w_incoh", type=float, default=0.5)
    ap.add_argument("--w_unknown", type=float, default=0.75)

    ap.add_argument("--curv_norm", default="median")  # interface parity
    ap.add_argument("--baseline_w_curv", type=float, default=1.0)
    ap.add_argument("--baseline_w_switch", type=float, default=0.25)

    ap.add_argument("--w_novelty_list", nargs="+", type=float, required=True)
    ap.add_argument("--no_backtrack", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    edges_by_a, data = load_edges(args.cache, args.planes, args.min_edge_sim, debug=args.debug)
    starts = sorted([a for a, lst in edges_by_a.items() if lst])

    if args.debug:
        print(f"[debug] starts: {starts}")

    # baseline (novelty=0) with baseline w_curv/w_switch
    baseline_cycle_by_start = {}
    for s in starts:
        r = rollout(
            edges_by_a, s, args.max_steps, args.topk_per_node,
            args.w_transport, args.baseline_w_curv, args.baseline_w_switch, args.w_incoh, args.w_unknown,
            w_novelty=0.0,
            no_backtrack=args.no_backtrack
        )
        baseline_cycle_by_start[s] = r.get("cycle_key") if r.get("cycle_key") else "NONE"

    baseline_sig_counts = Counter(baseline_cycle_by_start.values())
    baseline_sig = "|".join([f"{k}:{baseline_sig_counts[k]}" for k in sorted(baseline_sig_counts.keys())])

    if args.debug:
        print("Phase26g Force Bifurcations (Novelty Rollouts)")
        print(f"planes={args.planes} curv_norm={args.curv_norm} min_edge_sim={args.min_edge_sim} topk_per_node={args.topk_per_node}")
        print(f"no_backtrack={args.no_backtrack} max_steps={args.max_steps}")
        print(f"baseline: w_curv={args.baseline_w_curv} w_switch={args.baseline_w_switch}")
        print(f"baseline_sig: {baseline_sig}")

    sweep_rows = []
    for wN in args.w_novelty_list:
        cycle_by_start = {}
        rollouts = []
        for s in starts:
            r = rollout(
                edges_by_a, s, args.max_steps, args.topk_per_node,
                args.w_transport, args.w_curv, args.w_switch, args.w_incoh, args.w_unknown,
                w_novelty=wN,
                no_backtrack=args.no_backtrack
            )
            cycle_key = r.get("cycle_key") if r.get("cycle_key") else "NONE"
            cycle_by_start[s] = cycle_key
            rollouts.append({
                "start": s,
                "status": r["status"],
                "steps": max(0, len(r["path"]) - 1),
                "cycle_len": len(r["cycle"]) if r.get("cycle") else 0,
                "cycle_key": cycle_key,
                "path": "->".join(r["path"]),
            })

        agree = sum(1 for s in starts if cycle_by_start[s] == baseline_cycle_by_start[s])
        agree_frac = agree / max(1, len(starts))

        sig_counts = Counter(cycle_by_start.values())
        sig = "|".join([f"{k}:{sig_counts[k]}" for k in sorted(sig_counts.keys())])
        n_attr = len([k for k in sig_counts.keys() if k != "NONE"])

        sweep_rows.append({
            "w_novelty": wN,
            "agree_frac_vs_base": agree_frac,
            "n_attractors": n_attr,
            "signature": sig
        })

        per_csv = os.path.join(args.outdir, f"phase26g_rollouts_wN={wN:.3f}.csv")
        with open(per_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["start", "status", "steps", "cycle_len", "cycle_key", "path"])
            w.writeheader()
            w.writerows(rollouts)

    print("\nSWEEP RESULTS:")
    for r in sweep_rows:
        print(f"  wN={r['w_novelty']:.3f}  agree={r['agree_frac_vs_base']:.3f}  n_attr={r['n_attractors']}  sig={r['signature']}")

    out_csv = os.path.join(args.outdir, "phase26g_bifurcation_sweep.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["w_novelty", "agree_frac_vs_base", "n_attractors", "signature"])
        w.writeheader()
        w.writerows(sweep_rows)

    out_json = os.path.join(args.outdir, "phase26g_bifurcation_sweep.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "phase": "26g",
            "cache": args.cache,
            "outdir": args.outdir,
            "planes": args.planes,
            "min_edge_sim": args.min_edge_sim,
            "topk_per_node": args.topk_per_node,
            "max_steps": args.max_steps,
            "weights": {
                "w_transport": args.w_transport, "w_curv": args.w_curv, "w_switch": args.w_switch,
                "w_incoh": args.w_incoh, "w_unknown": args.w_unknown
            },
            "baseline": {"w_curv": args.baseline_w_curv, "w_switch": args.baseline_w_switch, "sig": baseline_sig},
            "rows": sweep_rows
        }, f, indent=2)

    print(f"\n[ok] wrote: {out_csv}")
    print(f"[ok] wrote: {out_json}")
    print("[ok] wrote: phase26g_rollouts_wN=*.csv (one per novelty weight)")

if __name__ == "__main__":
    main()
