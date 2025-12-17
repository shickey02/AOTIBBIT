#!/usr/bin/env python3
# phase26b_geodesic_field.py
#
# Phase26b: Geodesic Field + Deliverables
#
# Based on 26a, with upgrades:
#   - Top-K paths per node (not just best)
#   - Start-edge diagnostic table (sorted by total edge cost)
#   - Optional detailed per-node path printing (top N nodes)
#
# Outputs:
#   * phase26b_geodesic_field.json (field with top-k paths per node)
#   * phase26b_geodesic_field.csv  (summary table + alt totals)
#
# Scoring consistent with Phase25:
#   edge_cost = w_transport*transport + w_curv*curv_norm + w_switch*switch + w_incoh*incoh + w_unknown*unknown
#
# Notes:
# - transport uses (1 - similarity)
# - curv uses sum(arccos(principal_cosines)) in radians, optionally normalized by plane median/mean
# - consistency uses match_score_abs_diag_sum/k when present; fallback to mean principal cosine
#
import os, json, math, argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

# -------------------------
# Utilities
# -------------------------

def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def parse_edge_key(k: str) -> Optional[Tuple[str, str]]:
    """
    Supports:
      a__to__b
      a->b
      a|b
      a__b
    Returns (a,b) or None if unknown.
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
    # NOTE: this one is ambiguous in some codebases, but you used it in earlier phases.
    if "__" in k:
        a, b = k.split("__", 1)
        return a, b
    return None

def curvature_from_principal_cosines(pc: List[float]) -> float:
    # Sum of principal angles (radians)
    s = 0.0
    for c in pc:
        c = clamp(float(c), -1.0, 1.0)
        s += math.acos(c)
    return s

def consistency_from_payload(payload: Dict[str, Any]) -> Optional[float]:
    """
    Extract a 0..1 consistency score.
    Prefer: match_score_abs_diag_sum (0..k), normalized by k.
    Fallback: mean principal cosine (rough proxy).
    """
    if not isinstance(payload, dict):
        return None
    if "match_score_abs_diag_sum" in payload and "k" in payload:
        k = payload.get("k", None)
        if k and k > 0:
            return float(payload["match_score_abs_diag_sum"]) / float(k)
    if "principal_cosines" in payload:
        pcs = payload.get("principal_cosines", None)
        if pcs:
            return float(sum(pcs) / max(1, len(pcs)))
    return None

# -------------------------
# Edge scoring model
# -------------------------

@dataclass
class EdgeScore:
    transport: float
    curv: float
    curv_norm: float
    switch: int
    incoh: float
    unknown: int
    sim: float
    cons: Optional[float]
    plane: str
    a: str
    b: str

def edge_transport(sim: float) -> float:
    return 1.0 - float(sim)

def edge_incoherence(cons: Optional[float]) -> float:
    # If cons missing => max incoherence (1.0)
    if cons is None:
        return 1.0
    return clamp(1.0 - float(cons), 0.0, 1.0)

# -------------------------
# Build graph from cache
# -------------------------

def build_adjacency(
    transport_maps: Dict[str, Any],
    planes: List[str],
    min_edge_sim: float
) -> Dict[str, List[Tuple[str, str, Dict[str, Any]]]]:
    """
    Returns adjacency: adj[a] -> list of (b, plane, payload)
    Uses directed edges as stored in cache keys.
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
            if sim < float(min_edge_sim):
                continue
            adj[a].append((b, plane, payload))
    return adj

def collect_nodes(transport_maps: Dict[str, Any]) -> List[str]:
    nodes = set()
    for k in transport_maps.keys():
        parsed = parse_edge_key(k)
        if not parsed:
            continue
        a, b = parsed
        nodes.add(a); nodes.add(b)
    return sorted(nodes)

# -------------------------
# Curvature normalization per-plane
# -------------------------

def compute_plane_curv_scale(
    transport_maps: Dict[str, Any],
    planes: List[str],
    norm: str
) -> Dict[str, float]:
    """
    norm in {"none","median","mean"}.
    Scale computed over ALL edges for that plane present in the cache.
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
        vals_sorted = sorted(vals)
        if norm == "median":
            mid = len(vals_sorted) // 2
            if len(vals_sorted) % 2 == 1:
                scales[p] = float(vals_sorted[mid])
            else:
                scales[p] = 0.5 * (float(vals_sorted[mid-1]) + float(vals_sorted[mid]))
        elif norm == "mean":
            scales[p] = float(sum(vals_sorted) / len(vals_sorted))
        else:
            scales[p] = 1.0

        if scales[p] <= 1e-12:
            scales[p] = 1.0
    return scales

# -------------------------
# Dijkstra over (node, last_plane, known_count)
# -------------------------

@dataclass
class State:
    node: str
    last_plane: Optional[str]
    known_k: int

def state_key(s: State) -> Tuple[str, Optional[str], int]:
    return (s.node, s.last_plane, s.known_k)

def dijkstra_field_topk(
    adj: Dict[str, List[Tuple[str, str, Dict[str, Any]]]],
    start: str,
    planes: List[str],
    w_transport: float, w_curv: float, w_switch: float, w_incoh: float, w_unknown: float,
    require_known_k: int,
    curv_scale: Dict[str, float],
    max_steps: int,
    topk_per_node: int,
    debug: bool = False
):
    """
    Dijkstra over expanded state space. Then:
      - collect ALL states per node with known_k >= require_known_k
      - keep top-k by total cost
      - reconstruct top-k edge paths for each node
    """
    import heapq

    dist: Dict[Tuple[str, Optional[str], int], float] = {}
    prev: Dict[Tuple[str, Optional[str], int], Tuple[Tuple[str, Optional[str], int], EdgeScore]] = {}
    pq = []

    s0 = State(start, None, 0)
    sk0 = state_key(s0)
    dist[sk0] = 0.0
    heapq.heappush(pq, (0.0, 0, s0))  # (cost, steps, state)

    while pq:
        cost, steps, s = heapq.heappop(pq)
        sk = state_key(s)
        if dist.get(sk, float("inf")) < cost:
            continue
        if steps >= max_steps:
            continue

        for (nb, plane, payload) in adj.get(s.node, []):
            sim = float(payload.get("similarity", 0.0))
            pcs = payload.get("principal_cosines", None)
            cons = consistency_from_payload(payload)

            known_edge = bool(pcs)  # treat curvature-defined as "known"
            curv = curvature_from_principal_cosines(pcs) if pcs else None
            curv_norm = (curv / curv_scale.get(plane, 1.0)) if curv is not None else None

            transport = edge_transport(sim)
            sw = 0
            if s.last_plane is not None and plane != s.last_plane:
                sw = 1

            unknown = 0
            if curv_norm is None:
                unknown = 1
                curv_norm_val = 0.0
                curv_val = float("nan")
            else:
                curv_norm_val = float(curv_norm)
                curv_val = float(curv)

            incoh = edge_incoherence(cons)

            edge_cost = (w_transport * transport) + (w_curv * curv_norm_val) + (w_switch * sw) + (w_incoh * incoh) + (w_unknown * unknown)

            nk = s.known_k + (1 if known_edge else 0)
            s2 = State(nb, plane, nk)
            sk2 = state_key(s2)
            c2 = cost + edge_cost

            if c2 < dist.get(sk2, float("inf")):
                dist[sk2] = c2
                prev[sk2] = (sk, EdgeScore(
                    transport=transport,
                    curv=curv_val,
                    curv_norm=curv_norm_val,
                    switch=sw,
                    incoh=incoh,
                    unknown=unknown,
                    sim=sim,
                    cons=cons,
                    plane=plane,
                    a=s.node,
                    b=nb
                ))
                heapq.heappush(pq, (c2, steps + 1, s2))

    # collect candidate states per node meeting known_k constraint
    cand_by_node: Dict[str, List[Tuple[float, Tuple[str, Optional[str], int]]]] = defaultdict(list)
    for sk, c in dist.items():
        node, last_plane, known_k = sk
        if known_k < require_known_k:
            continue
        cand_by_node[node].append((c, sk))

    # keep top-k states per node
    best_state_keys_by_node: Dict[str, List[Tuple[str, Optional[str], int]]] = {}
    for node, lst in cand_by_node.items():
        lst.sort(key=lambda x: x[0])
        best_state_keys_by_node[node] = [sk for _, sk in lst[:max(1, int(topk_per_node))]]

    # reconstruct top-k paths per node
    paths_k: Dict[str, List[List[EdgeScore]]] = {}
    costs_k: Dict[str, List[float]] = {}
    for node, sk_list in best_state_keys_by_node.items():
        costs_k[node] = [dist[sk] for sk in sk_list]
        paths_k[node] = []
        for sk_best in sk_list:
            edges: List[EdgeScore] = []
            cur = sk_best
            while cur in prev:
                prev_sk, es = prev[cur]
                edges.append(es)
                cur = prev_sk
            edges.reverse()
            paths_k[node].append(edges)

    return best_state_keys_by_node, costs_k, paths_k

# -------------------------
# Output helpers
# -------------------------

def summarize_path(edges: List[EdgeScore], weights):
    w_transport, w_curv, w_switch, w_incoh, w_unknown = weights
    if not edges:
        return {
            "steps": 0,
            "total": 0.0,
            "transport": 0.0,
            "curv_norm_sum": 0.0,
            "switch": 0,
            "incoh": 0.0,
            "unknown": 0,
            "mean_sim": 0.0,
            "mean_cons": None
        }

    transport = sum(e.transport for e in edges)
    curv_norm_sum = sum(e.curv_norm for e in edges)
    switch = sum(e.switch for e in edges)
    incoh = sum(e.incoh for e in edges) / len(edges)
    unknown = sum(e.unknown for e in edges)
    mean_sim = sum(e.sim for e in edges) / len(edges)

    cons_vals = [e.cons for e in edges if e.cons is not None]
    mean_cons = (sum(cons_vals) / len(cons_vals)) if cons_vals else None

    total = (w_transport * transport) + (w_curv * curv_norm_sum) + (w_switch * switch) + (w_incoh * incoh) + (w_unknown * unknown)

    return {
        "steps": len(edges),
        "total": float(total),
        "transport": float(transport),
        "curv_norm_sum": float(curv_norm_sum),
        "switch": int(switch),
        "incoh": float(incoh),
        "unknown": int(unknown),
        "mean_sim": float(mean_sim),
        "mean_cons": (float(mean_cons) if mean_cons is not None else None)
    }

def print_start_edge_table(args, adj, curv_scale):
    rows = []
    for (b, plane, payload) in adj.get(args.start, []):
        sim = float(payload.get("similarity", 0.0))
        pcs = payload.get("principal_cosines", None)
        cons = consistency_from_payload(payload)

        curv = curvature_from_principal_cosines(pcs) if pcs else None
        curv_norm = (curv / curv_scale.get(plane, 1.0)) if curv is not None else None

        transport = edge_transport(sim)
        incoh = edge_incoherence(cons)

        unknown = 0
        if curv_norm is None:
            unknown = 1
            curv_norm_val = 0.0
        else:
            curv_norm_val = float(curv_norm)

        sw = 0  # start has no last_plane
        edge_cost = (args.w_transport * transport) + (args.w_curv * curv_norm_val) + (args.w_switch * sw) + (args.w_incoh * incoh) + (args.w_unknown * unknown)
        rows.append((edge_cost, plane, b, sim, curv_norm_val, incoh, unknown, cons))

    rows.sort(key=lambda x: x[0])

    print("\nSTART EDGE TABLE (sorted by total edge cost):")
    for (c, plane, b, sim, cn, incoh, unk, cons) in rows:
        cs = "None" if cons is None else f"{cons:.3f}"
        print(f"  cost={c:.4f}  {args.start} -> {b:22s} plane={plane:5s} sim={sim:.3f} curvN={cn:.3f} incoh={incoh:.3f} unk={unk} cons={cs}")

def format_path_edges(edges: List[EdgeScore]) -> str:
    if not edges:
        return "(start)"
    parts = []
    for e in edges:
        c = "None" if e.cons is None else f"{e.cons:.2f}"
        parts.append(f"{e.a}-[{e.plane} sim={e.sim:.3f} curvN={e.curv_norm:.3f} cons={c}]->{e.b}")
    return " | ".join(parts)

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--cache", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--planes", nargs="+", required=True)
    ap.add_argument("--start", required=True)

    ap.add_argument("--max_steps", type=int, default=6)
    ap.add_argument("--min_edge_sim", type=float, default=0.10)

    ap.add_argument("--w_transport", type=float, default=1.0)
    ap.add_argument("--w_curv", type=float, default=1.0)
    ap.add_argument("--w_switch", type=float, default=0.25)
    ap.add_argument("--w_incoh", type=float, default=0.5)
    ap.add_argument("--w_unknown", type=float, default=0.75)

    ap.add_argument("--require_known_k", type=int, default=1)
    ap.add_argument("--curv_norm", choices=["none", "median", "mean"], default="median")

    # 26b deliverables
    ap.add_argument("--topk_per_node", type=int, default=5)
    ap.add_argument("--print_paths", type=int, default=0, help="Print detailed top-k paths for the top N closest nodes (0 disables).")
    ap.add_argument("--print_start_edges", action="store_true")

    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()
    ensure_dir(args.outdir)

    d = load_json(args.cache)
    if "transport_maps" not in d:
        raise SystemExit("[error] cache missing transport_maps")

    tm = d["transport_maps"]
    nodes = collect_nodes(tm)
    if args.start not in nodes:
        raise SystemExit(f"[error] start node '{args.start}' not found in cache nodes: {nodes}")

    planes = args.planes

    # curvature scale
    curv_scale = compute_plane_curv_scale(tm, planes, args.curv_norm)

    # adjacency
    adj = build_adjacency(tm, planes, args.min_edge_sim)

    if args.debug:
        print("[debug] nodes:", nodes)
        for p in planes:
            print(f"[debug] curv_scale[{p}] = {curv_scale.get(p, None)}")
        print("[debug] start outgoing edges:", len(adj.get(args.start, [])))

    if args.print_start_edges:
        print_start_edge_table(args, adj, curv_scale)

    best_states, costs_k, paths_k = dijkstra_field_topk(
        adj, args.start, planes,
        args.w_transport, args.w_curv, args.w_switch, args.w_incoh, args.w_unknown,
        args.require_known_k,
        curv_scale,
        args.max_steps,
        args.topk_per_node,
        debug=args.debug
    )

    weights = (args.w_transport, args.w_curv, args.w_switch, args.w_incoh, args.w_unknown)

    # assemble output
    out = {
        "phase": "26b",
        "status": "ok",
        "args": vars(args),
        "planes": planes,
        "start": args.start,
        "curv_norm": args.curv_norm,
        "curv_scale": curv_scale,
        "nodes": nodes,
        "field": {}
    }

    # CSV summary:
    # include best, steps, plus alt totals 2..K
    alt_cols = []
    for i in range(2, max(2, args.topk_per_node + 1)):
        alt_cols.append(f"alt{i}_total")
    csv_lines = ["node,best_total,best_steps,best_transport,best_curv_norm_sum,best_switch,best_incoh,best_unknown,best_mean_sim,best_mean_cons," + ",".join(alt_cols)]

    # build field per node
    for n in nodes:
        if n == args.start:
            out["field"][n] = {
                "reachable": True,
                "topk": [
                    {
                        "rank": 1,
                        "summary": summarize_path([], weights),
                        "path": [args.start],
                        "edges": []
                    }
                ]
            }
            csv_lines.append(f"{n},0,0,0,0,0,0,0,0,,{','.join(['' for _ in alt_cols])}")
            continue

        if n not in paths_k or not paths_k[n]:
            out["field"][n] = {"reachable": False}
            csv_lines.append(f"{n},,,,,,,,,,{','.join(['' for _ in alt_cols])}")
            continue

        topk_entries = []
        for rank_idx, edges in enumerate(paths_k[n], start=1):
            summary = summarize_path(edges, weights)
            path_nodes = [args.start] + [e.b for e in edges]
            topk_entries.append({
                "rank": rank_idx,
                "summary": summary,
                "path": path_nodes,
                "edges": [
                    {
                        "a": e.a, "b": e.b, "plane": e.plane,
                        "sim": e.sim, "cons": e.cons,
                        "transport": e.transport,
                        "curv": e.curv,
                        "curv_norm": e.curv_norm,
                        "switch": e.switch,
                        "incoh": e.incoh,
                        "unknown": e.unknown
                    } for e in edges
                ]
            })

        out["field"][n] = {
            "reachable": True,
            "topk": topk_entries
        }

        # CSV best + alt totals
        best = topk_entries[0]["summary"]
        mc = "" if best["mean_cons"] is None else f"{best['mean_cons']}"
        alt_totals = []
        for i in range(1, args.topk_per_node):
            if i < len(topk_entries):
                alt_totals.append(f"{topk_entries[i]['summary']['total']}")
            else:
                alt_totals.append("")
        # pad to alt_cols length
        while len(alt_totals) < len(alt_cols):
            alt_totals.append("")

        csv_lines.append(
            f"{n},{best['total']},{best['steps']},{best['transport']},{best['curv_norm_sum']},{best['switch']},{best['incoh']},{best['unknown']},{best['mean_sim']},{mc},{','.join(alt_totals)}"
        )

    # write files
    out_json = os.path.join(args.outdir, "phase26b_geodesic_field.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)

    out_csv = os.path.join(args.outdir, "phase26b_geodesic_field.csv")
    with open(out_csv, "w", newline="") as f:
        f.write("\n".join(csv_lines) + "\n")

    # console summary
    reachable = [n for n in nodes if out["field"].get(n, {}).get("reachable")]
    print("\nPhase26b Geodesic Field")
    print(f"start={args.start} planes={planes} curv_norm={args.curv_norm} min_edge_sim={args.min_edge_sim}")
    print("curv_scale:", curv_scale)
    print(f"reachable: {len(reachable)}/{len(nodes)}")
    print(f"topk_per_node: {args.topk_per_node}")

    # Print top-10 closest (excluding start), using best path total
    scored = []
    for n in nodes:
        if n == args.start:
            continue
        fld = out["field"].get(n, {})
        if not fld.get("reachable"):
            continue
        best_total = fld["topk"][0]["summary"]["total"]
        best_steps = fld["topk"][0]["summary"]["steps"]
        scored.append((best_total, n, best_steps))
    scored.sort(key=lambda x: x[0])

    print("\nTOP CLOSEST NODES (best path):")
    for rank, (tot, n, steps) in enumerate(scored[:10], 1):
        print(f"[{rank}] total={tot:.4f} steps={steps} -> {n}")

    # Optional detailed printing for top N closest nodes
    if args.print_paths and args.print_paths > 0:
        print(f"\nDETAILED TOP-K PATHS FOR TOP {args.print_paths} NODES:")
        for rank, (tot, n, _) in enumerate(scored[:args.print_paths], 1):
            print(f"\n== [{rank}] {n} (best_total={tot:.4f}) ==")
            topk = out["field"][n]["topk"]
            for entry in topk:
                s = entry["summary"]
                print(f"  (k={entry['rank']}) total={s['total']:.4f} steps={s['steps']} transport={s['transport']:.4f} curvNsum={s['curv_norm_sum']:.4f} switch={s['switch']} incoh={s['incoh']:.3f} unknown={s['unknown']}")
                # print edge chain compactly
                if entry["edges"]:
                    for e in entry["edges"]:
                        cons = e["cons"]
                        cs = "None" if cons is None else f"{cons:.2f}"
                        print(f"      {e['a']} -[{e['plane']} sim={e['sim']:.3f} curvN={e['curv_norm']:.3f} cons={cs} unk={e['unknown']}]-> {e['b']}")
                else:
                    print("      (start)")

    print(f"\n[ok] wrote: {out_json}")
    print(f"[ok] wrote: {out_csv}")

if __name__ == "__main__":
    main()
