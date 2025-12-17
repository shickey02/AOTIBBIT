#!/usr/bin/env python3
# explain_novelty_flip.py
#
# Explain why novelty rollouts "flip" a decision relative to baseline.
#
# FIX (node-novelty):
#   novelty = 1 if candidate edge goes to a node already visited in the current path
#            = 0 otherwise
#
# Cache format support:
# - Primary: cache["transport_maps"] dict keyed like "{a}__to__{b}"
#   where each value contains similarity per plane in some form (we search common fields).
#
# Output:
# - base rollout path (w_novelty=0)
# - novelty rollout path (w_novelty>0)
# - divergence point, plus candidate table with base + novelty scores.

import os, json, math, argparse
from collections import defaultdict

# -------------------------
# helpers
# -------------------------

def _safe_float(x):
    try:
        if x is None: return None
        if isinstance(x, (int, float)): return float(x)
        if isinstance(x, str):
            return float(x.strip())
        return None
    except Exception:
        return None

def parse_ab_from_key(k):
    # expected: "{a}__to__{b}" (as seen in your cache keys)
    if "__to__" in k:
        a, b = k.split("__to__", 1)
        return a, b
    return None, None

def extract_sim_for_plane(v, plane):
    """
    Try to extract similarity for a given plane from a transport_map entry.
    We look in several common locations:
      - v["sim"][plane] / v["sims"][plane] / v["similarity"][plane]
      - v["planes"][plane]["sim"] or ["similarity"]
      - v[plane] if v is plane->dict and dict contains sim
      - nested "metrics" dicts
    Return None if not found.
    """
    # direct dicts keyed by plane
    for key in ("sim", "sims", "similarity", "similarities"):
        if isinstance(v, dict) and key in v and isinstance(v[key], dict):
            s = _safe_float(v[key].get(plane))
            if s is not None:
                return s

    # planes -> plane -> {sim: ...}
    if isinstance(v, dict) and "planes" in v and isinstance(v["planes"], dict):
        pv = v["planes"].get(plane)
        if isinstance(pv, dict):
            for key in ("sim", "similarity"):
                s = _safe_float(pv.get(key))
                if s is not None:
                    return s
            # sometimes nested metrics
            if "metrics" in pv and isinstance(pv["metrics"], dict):
                for key in ("sim", "similarity"):
                    s = _safe_float(pv["metrics"].get(key))
                    if s is not None:
                        return s

    # v might itself be dict keyed by plane
    if isinstance(v, dict) and plane in v and isinstance(v[plane], dict):
        pv = v[plane]
        for key in ("sim", "similarity"):
            s = _safe_float(pv.get(key))
            if s is not None:
                return s
        if "metrics" in pv and isinstance(pv["metrics"], dict):
            for key in ("sim", "similarity"):
                s = _safe_float(pv["metrics"].get(key))
                if s is not None:
                    return s

    # very last: search one-level deep for plane->value
    if isinstance(v, dict):
        for k2, v2 in v.items():
            if isinstance(v2, dict) and plane in v2 and isinstance(v2[plane], (int, float, str)):
                s = _safe_float(v2[plane])
                if s is not None:
                    return s

    return None

def load_cache_edges(cache_path, planes, min_edge_sim):
    """
    Build edges list from cache["transport_maps"].
    Each (a,b,plane) becomes an edge with sim + derived costs.
    """
    with open(cache_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "transport_maps" not in data or not isinstance(data["transport_maps"], dict):
        raise RuntimeError(f"Cache JSON missing 'transport_maps' dict. Top-level keys: {list(data.keys())[:20]}")

    tmap = data["transport_maps"]

    edges = []
    nodes = set()

    for k, v in tmap.items():
        a, b = parse_ab_from_key(k)
        if not a or not b:
            continue

        for plane in planes:
            sim = extract_sim_for_plane(v, plane)
            if sim is None:
                continue
            if sim < min_edge_sim:
                continue
            # cost components used across your phase26* scripts
            transport = 1.0 - sim
            incoh = 1.0 - sim
            unknown = 0.0

            edges.append({
                "a": a,
                "b": b,
                "plane": plane,
                "sim": sim,
                "transport": transport,
                "incoh": incoh,
                "unknown": unknown,
            })
            nodes.add(a); nodes.add(b)

    edges_by_a = defaultdict(list)
    for e in edges:
        edges_by_a[e["a"]].append(e)

    # sort each adjacency list by highest sim first (stable)
    for a in list(edges_by_a.keys()):
        edges_by_a[a].sort(key=lambda x: (-x["sim"], x["plane"], x["b"]))

    n_edges = len(edges)
    return edges_by_a, data, n_edges, sorted(nodes)

def edge_cost(e, w_transport, w_incoh, w_unknown):
    # NOTE: no curvature/switch here; this tool is specifically for novelty explanation
    return (
        w_transport * float(e.get("transport", 0.0)) +
        w_incoh    * float(e.get("incoh", 0.0)) +
        w_unknown  * float(e.get("unknown", 0.0))
    )

def choose_edge(cands, visited_nodes, w_transport, w_incoh, w_unknown, w_novelty):
    """
    Pick the best edge among candidates, using node-novelty.
    novelty = 1 if stepping into an already-visited node, else 0
    """
    best = None
    best_score = 1e18
    for e in cands:
        base = edge_cost(e, w_transport, w_incoh, w_unknown)
        novelty = 1.0 if (e["b"] in visited_nodes) else 0.0
        score = base + w_novelty * novelty
        if score < best_score:
            best_score = score
            best = (e, base, novelty, score)
    return best

def cycle_from_path(path):
    """
    Detect first repetition and return minimal recurrent cycle nodes.
    When repetition happens at nb, cycle is path[j:-1] excluding repeated final nb.
    Also reduce to smallest period if possible.
    Returns (cycle_nodes_list, j_index, repeated_node) or (None, None, None).
    """
    seen = {}
    for i, n in enumerate(path):
        if n in seen:
            j = seen[n]
            # path[j:i] is cycle nodes (exclude repeated node at i)
            cyc = path[j:i]
            # reduce to smallest period
            cyc = reduce_cycle_period(cyc)
            return cyc, j, n
        seen[n] = i
    return None, None, None

def reduce_cycle_period(cyc):
    """
    If cyc repeats a smaller pattern, return smallest period.
    e.g. [a,b,a,b] -> [a,b]
    """
    L = len(cyc)
    if L <= 1:
        return cyc
    for p in range(1, L):
        if L % p != 0:
            continue
        ok = True
        for i in range(L):
            if cyc[i] != cyc[i % p]:
                ok = False
                break
        if ok:
            return cyc[:p]
    return cyc

def rollout(edges_by_a, start, planes, min_edge_sim, topk_per_node, max_steps,
            w_transport, w_incoh, w_unknown, w_novelty, no_backtrack):
    """
    Deterministic greedy rollout with (optional) no-backtrack and node-novelty.
    Returns:
      path (list of nodes),
      trace (list of dicts per step),
      cycle_nodes (or None)
    """
    path = [start]
    visited_nodes = set([start])

    trace = []
    prev = None

    for step in range(max_steps):
        cur = path[-1]
        cands = list(edges_by_a.get(cur, []))

        # filter by allowed planes + sim
        cands = [e for e in cands if e["plane"] in planes and e["sim"] >= min_edge_sim]

        # optional no-backtrack (forbid immediate reversal)
        if no_backtrack and prev is not None:
            cands = [e for e in cands if not (e["b"] == prev)]

        # take top-k by sim (already sorted)
        cands = cands[:topk_per_node]

        if not cands:
            trace.append({
                "step": step,
                "node": cur,
                "status": "dead_end",
                "chosen": None,
                "cands": []
            })
            break

        chosen = choose_edge(cands, visited_nodes, w_transport, w_incoh, w_unknown, w_novelty)
        e, base, nov, score = chosen

        trace.append({
            "step": step,
            "node": cur,
            "status": "ok",
            "chosen": {"edge": e, "base": base, "novelty": nov, "score": score},
            "cands": [
                {
                    "edge": ce,
                    "base": edge_cost(ce, w_transport, w_incoh, w_unknown),
                    "novelty": 1.0 if (ce["b"] in visited_nodes) else 0.0,
                    "score": edge_cost(ce, w_transport, w_incoh, w_unknown) + w_novelty*(1.0 if (ce["b"] in visited_nodes) else 0.0),
                }
                for ce in cands
            ]
        })

        prev = cur
        path.append(e["b"])
        visited_nodes.add(e["b"])

        cyc, j, rep = cycle_from_path(path)
        if cyc is not None:
            return path, trace, cyc

    return path, trace, None

def fmt_path(path):
    return "->".join(path)

def print_step_table(steprec, w_novelty):
    cur = steprec["node"]
    step = steprec["step"]
    print(f"\n[step {step}] node={cur}")
    if steprec["status"] == "dead_end":
        print("  (dead end: no candidates)")
        return
    chosen = steprec["chosen"]
    if chosen is None:
        print("  (no choice)")
        return
    ce = chosen["edge"]
    print(f"  CHOSEN: {ce['a']} -> {ce['b']}  plane={ce['plane']}  sim={ce['sim']:.4f}  base={chosen['base']:.4f}  nov={chosen['novelty']:.1f}  score={chosen['score']:.4f}  (wN={w_novelty})")

    # print sorted by score (then sim desc)
    rows = list(steprec["cands"])
    rows.sort(key=lambda r: (r["score"], -r["edge"]["sim"], r["edge"]["plane"], r["edge"]["b"]))

    print("  candidates (sorted by score):")
    for r in rows:
        e = r["edge"]
        print(f"    {e['a']} -> {e['b']:<22} plane={e['plane']:<6} sim={e['sim']:.4f}  base={r['base']:.4f}  nov={r['novelty']:.1f}  score={r['score']:.4f}")

def find_divergence(base_path, nov_path):
    m = min(len(base_path), len(nov_path))
    for i in range(m):
        if base_path[i] != nov_path[i]:
            return i
    return None

# -------------------------
# main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="phase15f_transport_cache.json")
    ap.add_argument("--planes", nargs="+", required=True, help="planes to consider (e.g. lt bo_lr)")
    ap.add_argument("--start", required=True, help="start node")
    ap.add_argument("--min_edge_sim", type=float, default=0.10)
    ap.add_argument("--topk_per_node", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=10)
    ap.add_argument("--w_transport", type=float, default=1.0)
    ap.add_argument("--w_incoh", type=float, default=0.5)
    ap.add_argument("--w_unknown", type=float, default=0.75)
    ap.add_argument("--w_novelty", type=float, default=0.35)
    ap.add_argument("--no_backtrack", action="store_true")
    args = ap.parse_args()

    edges_by_a, data, n_edges, nodes = load_cache_edges(args.cache, args.planes, args.min_edge_sim)
    nodes_with_outgoing = sum(1 for k,v in edges_by_a.items() if len(v) > 0)

    print(f"[ok] derived edges: {n_edges} | nodes_with_outgoing: {nodes_with_outgoing}")
    print(f"[info] cache top keys: {list(data.keys())}")
    if "transport_maps" in data:
        sample = list(data["transport_maps"].keys())[:8]
        print(f"[info] transport_maps top keys sample: {sample}")

    if args.start not in nodes:
        print(f"[warn] start '{args.start}' not found in node list extracted from edges.")
        print(f"       nodes: {nodes}")
        return

    # baseline: w_novelty=0
    base_path, base_trace, base_cycle = rollout(
        edges_by_a,
        start=args.start,
        planes=args.planes,
        min_edge_sim=args.min_edge_sim,
        topk_per_node=args.topk_per_node,
        max_steps=args.max_steps,
        w_transport=args.w_transport,
        w_incoh=args.w_incoh,
        w_unknown=args.w_unknown,
        w_novelty=0.0,
        no_backtrack=args.no_backtrack,
    )

    # novelty
    nov_path, nov_trace, nov_cycle = rollout(
        edges_by_a,
        start=args.start,
        planes=args.planes,
        min_edge_sim=args.min_edge_sim,
        topk_per_node=args.topk_per_node,
        max_steps=args.max_steps,
        w_transport=args.w_transport,
        w_incoh=args.w_incoh,
        w_unknown=args.w_unknown,
        w_novelty=args.w_novelty,
        no_backtrack=args.no_backtrack,
    )

    print("\n[base path]", fmt_path(base_path))
    if base_cycle is not None:
        print("  base cycle:", "|".join(base_cycle))
    print("[nov  path]", fmt_path(nov_path))
    if nov_cycle is not None:
        print("  nov  cycle:", "|".join(nov_cycle))

    # divergence check
    div = None
    m = min(len(base_path), len(nov_path))
    for i in range(m):
        if base_path[i] != nov_path[i]:
            div = i
            break

    if div is None:
        print("\n[divergence] none (identical decisions under these settings)")
        return

    # divergence means different node reached at same index; the flip occurs at previous step decision
    flip_step = div - 1
    print(f"\n[divergence] first different node at index={div}")
    print(f"  base[{div}]={base_path[div]}   nov[{div}]={nov_path[div]}")
    if flip_step >= 0:
        print(f"  flip step was decision made at step={flip_step} from node={base_path[flip_step]}")

    # Print tables around flip step
    print("\n=== BASE decision table at flip step ===")
    if 0 <= flip_step < len(base_trace):
        print_step_table(base_trace[flip_step], w_novelty=0.0)
    else:
        print("  (no trace row at that step)")

    print("\n=== NOVELTY decision table at flip step ===")
    if 0 <= flip_step < len(nov_trace):
        print_step_table(nov_trace[flip_step], w_novelty=args.w_novelty)
    else:
        print("  (no trace row at that step)")

if __name__ == "__main__":
    main()
