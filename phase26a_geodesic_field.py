#!/usr/bin/env python3
# phase26a_geodesic_field.py
#
# Multi-goal geodesic field:
# - Given start node and planes, compute best geodesic cost to ALL nodes
# - Uses transport from phase15f cache and curvature (principal angles) per plane
# - Optional curvature normalization per plane (median/mean)
# - Outputs:
#   * phase26a_geodesic_field.json (distances + paths + per-edge breakdown)
#   * phase26a_geodesic_field.csv  (summary table)
#
# Designed to be consistent with Phase25 scoring:
#   total = w_transport*transport + w_curv*curv_norm + w_switch*switch + w_incoh*incoh + w_unknown*unknown
#
# Notes:
# - "transport" uses (1 - similarity) by default (lower is better)
# - "curv" uses sum(arccos(principal_cosines)) (radians) by default (lower is straighter)
# - "consistency" uses match_score_abs_diag_sum if present (0..k), mapped to 0..1
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
    if "__" in k:
        a, b = k.split("__", 1)
        return a, b
    return None

def curvature_from_principal_cosines(pc: List[float]) -> float:
    # Sum of principal angles (radians)
    # pc values should be within [-1,1], but we clamp for safety.
    s = 0.0
    for c in pc:
        c = clamp(c, -1.0, 1.0)
        s += math.acos(c)
    return s

def consistency_from_payload(payload: Dict[str, Any]) -> Optional[float]:
    """
    Attempts to extract a 0..1 consistency score.
    In your cache, we saw: match_score_abs_diag_sum (0..k).
    We map to [0..1] by dividing by k if available.
    """
    if not isinstance(payload, dict):
        return None
    if "match_score_abs_diag_sum" in payload and "k" in payload:
        k = payload.get("k", None)
        if k and k > 0:
            return float(payload["match_score_abs_diag_sum"]) / float(k)
    # Fallback: if "principal_cosines" exists, use mean cosine as rough consistency
    if "principal_cosines" in payload:
        pcs = payload["principal_cosines"]
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
    # Lower is better. (1-sim) behaves like a distance.
    return 1.0 - sim

def edge_incoherence(cons: Optional[float]) -> float:
    # Your Phase25 prints incoh like 1.000 when cons==0.
    # We'll define incoh = 1 - cons when cons is known; if cons missing => 1.0 (max incoherence).
    if cons is None:
        return 1.0
    return clamp(1.0 - float(cons), 0.0, 1.0)

# -------------------------
# Build graph from cache
# -------------------------

def build_adjacency(transport_maps: Dict[str, Any], planes: List[str], min_edge_sim: float) -> Dict[str, List[Tuple[str, str, Dict[str, Any]]]]:
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
            if plane not in by_plane:
                continue
            payload = by_plane[plane]
            if not isinstance(payload, dict):
                continue
            sim = float(payload.get("similarity", -1.0))
            if sim < min_edge_sim:
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

def compute_plane_curv_scale(transport_maps: Dict[str, Any], planes: List[str], norm: str) -> Dict[str, float]:
    """
    norm in {"none","median","mean"}.
    Scale is computed over ALL edges available for that plane in the cache.
    """
    if norm == "none":
        return {p: 1.0 for p in planes}

    vals_by_plane = {p: [] for p in planes}
    for k, by_plane in transport_maps.items():
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

        # Avoid divide-by-zero
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

def dijkstra_field(adj, start: str, planes: List[str],
                   w_transport: float, w_curv: float, w_switch: float, w_incoh: float, w_unknown: float,
                   require_known_k: int,
                   curv_scale: Dict[str, float],
                   max_steps: int,
                   debug: bool = False):
    """
    We compute best cost to all nodes with a constraint: must have >= require_known_k "known" edges along path,
    where an edge is "known" if it has curvature+consistency available (principal_cosines exist).
    Unknown edges are allowed but penalized by w_unknown and counted.
    """
    import heapq

    # Dist per state; we’ll later pick best state per node that meets require_known_k.
    dist = {}
    prev = {}  # state_key -> (prev_state_key, EdgeScore)
    pq = []

    s0 = State(start, None, 0)
    dist[state_key(s0)] = 0.0
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

            # known edge if pcs exists (curv defined) AND cons defined (optional, but we treat pcs as main)
            known_edge = bool(pcs)

            curv = curvature_from_principal_cosines(pcs) if pcs else None
            curv_norm = (curv / curv_scale.get(plane, 1.0)) if curv is not None else None

            # penalties
            transport = edge_transport(sim)
            sw = 0
            if s.last_plane is not None and plane != s.last_plane:
                sw = 1

            # If no curvature info, we treat it as unknown and penalize.
            unknown = 0
            if curv_norm is None:
                unknown = 1
                curv_norm_val = 0.0  # don’t add curvature term if unknown
            else:
                curv_norm_val = float(curv_norm)

            incoh = edge_incoherence(cons)

            edge_cost = (w_transport * transport) + (w_curv * curv_norm_val) + (w_switch * sw) + (w_incoh * incoh) + (w_unknown * unknown)

            nk = s.known_k + (1 if known_edge else 0)

            s2 = State(nb, plane, nk)
            sk2 = state_key(s2)
            c2 = cost + edge_cost

            # optional step pruning: keep only best known_k up to require_known_k+2 per node/plane
            if c2 < dist.get(sk2, float("inf")):
                dist[sk2] = c2
                prev[sk2] = (sk, EdgeScore(
                    transport=transport,
                    curv=(float(curv) if curv is not None else float("nan")),
                    curv_norm=(curv_norm_val),
                    switch=sw,
                    incoh=incoh,
                    unknown=unknown,
                    sim=sim,
                    cons=(cons if cons is not None else None),
                    plane=plane,
                    a=s.node,
                    b=nb
                ))
                heapq.heappush(pq, (c2, steps + 1, s2))

    # For each node, pick best state that satisfies require_known_k
    best_per_node = {}
    best_state_key_per_node = {}

    for sk, c in dist.items():
        node, last_plane, known_k = sk
        if known_k < require_known_k:
            continue
        if c < best_per_node.get(node, float("inf")):
            best_per_node[node] = c
            best_state_key_per_node[node] = sk

    # reconstruct paths
    paths = {}
    for node, sk_best in best_state_key_per_node.items():
        # walk back
        edges = []
        cur = sk_best
        while cur in prev:
            prev_sk, es = prev[cur]
            edges.append(es)
            cur = prev_sk
        edges.reverse()
        paths[node] = edges

    return best_per_node, paths

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
            "curv": 0.0,
            "switch": 0,
            "incoh": 0.0,
            "unknown": 0,
            "mean_sim": 0.0,
            "mean_cons": None
        }

    transport = sum(e.transport for e in edges)
    curv_norm = sum(e.curv_norm for e in edges)
    switch = sum(e.switch for e in edges)
    incoh = sum(e.incoh for e in edges) / len(edges)
    unknown = sum(e.unknown for e in edges)
    mean_sim = sum(e.sim for e in edges) / len(edges)

    cons_vals = [e.cons for e in edges if e.cons is not None]
    mean_cons = (sum(cons_vals) / len(cons_vals)) if cons_vals else None

    total = (w_transport * transport) + (w_curv * curv_norm) + (w_switch * switch) + (w_incoh * incoh) + (w_unknown * unknown)

    return {
        "steps": len(edges),
        "total": float(total),
        "transport": float(transport),
        "curv_norm_sum": float(curv_norm),
        "switch": int(switch),
        "incoh": float(incoh),
        "unknown": int(unknown),
        "mean_sim": float(mean_sim),
        "mean_cons": (float(mean_cons) if mean_cons is not None else None)
    }

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
            print(f"[debug] curv_scale[{p}] =", curv_scale.get(p, None))
        print("[debug] start outgoing edges:", len(adj.get(args.start, [])))

    best_cost, paths = dijkstra_field(
        adj, args.start, planes,
        args.w_transport, args.w_curv, args.w_switch, args.w_incoh, args.w_unknown,
        args.require_known_k,
        curv_scale,
        args.max_steps,
        debug=args.debug
    )

    weights = (args.w_transport, args.w_curv, args.w_switch, args.w_incoh, args.w_unknown)

    # assemble output
    out = {
        "phase": "26a",
        "status": "ok",
        "args": vars(args),
        "planes": planes,
        "start": args.start,
        "curv_norm": args.curv_norm,
        "curv_scale": curv_scale,
        "nodes": nodes,
        "field": {}
    }

    # CSV summary
    csv_lines = ["node,total,steps,transport,curv_norm_sum,switch,incoh,unknown,mean_sim,mean_cons"]

    for n in nodes:
        if n == args.start:
            out["field"][n] = {
                "reachable": True,
                "summary": summarize_path([], weights),
                "path": [args.start],
                "edges": []
            }
            csv_lines.append(f"{n},0,0,0,0,0,0,0,0,")
            continue

        if n not in paths:
            out["field"][n] = {"reachable": False}
            csv_lines.append(f"{n},, , , , , , , ,")
            continue

        edges = paths[n]
        summary = summarize_path(edges, weights)
        path_nodes = [args.start] + [e.b for e in edges]

        out["field"][n] = {
            "reachable": True,
            "summary": summary,
            "path": path_nodes,
            "edges": [
                {
                    "a": e.a, "b": e.b, "plane": e.plane,
                    "sim": e.sim, "cons": e.cons,
                    "transport": e.transport,
                    "curv_norm": e.curv_norm,
                    "switch": e.switch,
                    "incoh": e.incoh,
                    "unknown": e.unknown
                } for e in edges
            ]
        }

        mc = "" if summary["mean_cons"] is None else f"{summary['mean_cons']}"
        csv_lines.append(
            f"{n},{summary['total']},{summary['steps']},{summary['transport']},{summary['curv_norm_sum']},{summary['switch']},{summary['incoh']},{summary['unknown']},{summary['mean_sim']},{mc}"
        )

    # write files
    out_json = os.path.join(args.outdir, "phase26a_geodesic_field.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)

    out_csv = os.path.join(args.outdir, "phase26a_geodesic_field.csv")
    with open(out_csv, "w", newline="") as f:
        f.write("\n".join(csv_lines) + "\n")

    # console summary
    reachable = [n for n in nodes if out["field"].get(n, {}).get("reachable")]
    print("Phase26a Geodesic Field")
    print(f"start={args.start} planes={planes} curv_norm={args.curv_norm} min_edge_sim={args.min_edge_sim}")
    print("curv_scale:", curv_scale)
    print(f"reachable: {len(reachable)}/{len(nodes)}")

    # Print top-10 closest (excluding start)
    scored = []
    for n in nodes:
        if n == args.start:
            continue
        fld = out["field"].get(n, {})
        if not fld.get("reachable"):
            continue
        scored.append((fld["summary"]["total"], n, fld["summary"]["steps"]))
    scored.sort(key=lambda x: x[0])

    print("\nTOP CLOSEST NODES:")
    for rank, (tot, n, steps) in enumerate(scored[:10], 1):
        print(f"[{rank}] total={tot:.4f} steps={steps}  -> {n}")

    print(f"\n[ok] wrote: {out_json}")
    print(f"[ok] wrote: {out_csv}")

if __name__ == "__main__":
    main()
