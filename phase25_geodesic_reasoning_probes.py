#!/usr/bin/env python3
# phase25_geodesic_reasoning_probes.py
#
# Phase25: Geodesic Reasoning Probes over concept-graphs derived from transport cache.
# NEW (Step 25 upgrades):
#   - Curvature normalization per plane (median/mean of known-edge curvatures)
#   - Deliverables:
#       * phase25_paths.json
#       * phase25_graph.png (graph + highlighted best path)
#
# Scoring (unchanged except curvature normalization):
#   transport_cost = sum(-log(sim))
#   curv_cost      = sum(curv_abs) or sum(curv_abs / plane_scale)
#   switch_cost    = number of plane changes
#   incoh_cost     = (1 - mean_consistency) over KNOWN edges only (if none -> 1.0)
#   unknown_cost   = number of UNKNOWN edges
#
# total = w_transport*transport + w_curv*curv + w_switch*switch + w_incoh*incoh + w_unknown*unknown

import os, json, math, argparse, statistics
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any

# optional graph viz
try:
    import networkx as nx
except Exception:
    nx = None

import matplotlib.pyplot as plt


# ----------------------------
# Utilities
# ----------------------------

def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def split_edge_key(k: str) -> Optional[Tuple[str, str]]:
    """
    Supports a bunch of historical key formats.
    Current cache shows: a__to__b
    Also supports: a->b, a|b, a__b
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

def edge_id(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)

def safe_log_sim(sim: float, eps: float = 1e-9) -> float:
    sim = max(eps, min(1.0, sim))
    return -math.log(sim)

def parse_planes(planes: List[str]) -> List[str]:
    out = []
    for p in planes:
        p = p.strip()
        if p:
            out.append(p)
    return out


# ----------------------------
# Known-edge map from phase22a clusters output
# ----------------------------

@dataclass
class KnownEdge:
    a: str
    b: str
    sim: float
    curv_abs: float
    cons: float  # axis_consistency_proxy or similar (0..1)

def load_known_edges_from_clusters_json(path: str) -> Dict[Tuple[str, str], KnownEdge]:
    """
    Expects your phase22a_*_edge_axis_clusters_*.json structure:
      - "edge_clusters": { "cluster_0_...": [ {a,b,sim,curv_abs,axis_consistency_proxy,...}, ... ], ... }
    """
    data = load_json(path)
    out: Dict[Tuple[str, str], KnownEdge] = {}
    edge_clusters = data.get("edge_clusters", {})
    for _, edges in edge_clusters.items():
        if not isinstance(edges, list):
            continue
        for e in edges:
            try:
                a = e["a"]; b = e["b"]
                sim = float(e.get("sim", e.get("similarity", 0.0)))
                curv_abs = float(e.get("curv_abs", 0.0))
                cons = float(e.get("axis_consistency_proxy", e.get("consistency", e.get("cons", 0.0))))
                out[edge_id(a, b)] = KnownEdge(a=a, b=b, sim=sim, curv_abs=curv_abs, cons=cons)
            except Exception:
                continue
    return out


# ----------------------------
# Transport cache -> adjacency
# ----------------------------

@dataclass
class CacheEdge:
    a: str
    b: str
    plane: str
    sim: float

def build_adjacency_from_cache(cache_path: str, planes: List[str], min_edge_sim: float, debug_cache: bool=False):
    cache = load_json(cache_path)
    tm = cache.get("transport_maps", cache.get("transport_map", None))
    if tm is None or not isinstance(tm, dict):
        raise RuntimeError("[error] Could not find transport_maps in cache.")

    if debug_cache:
        print("\n[debug] transport_maps top-level keys sample:")
        for i, k in enumerate(list(tm.keys())[:10]):
            print("   ", k)

    # adjacency[node] = list of CacheEdge
    adj: Dict[str, List[CacheEdge]] = {}

    for k, v in tm.items():
        sp = split_edge_key(k)
        if sp is None:
            continue
        a, b = sp
        if not isinstance(v, dict):
            continue

        for plane in planes:
            if plane not in v:
                continue
            pv = v[plane]
            if not isinstance(pv, dict):
                continue
            sim = pv.get("similarity", pv.get("sim", None))
            if sim is None:
                continue
            sim = float(sim)
            if sim < min_edge_sim:
                continue

            adj.setdefault(a, []).append(CacheEdge(a=a, b=b, plane=plane, sim=sim))
            adj.setdefault(b, []).append(CacheEdge(a=b, b=a, plane=plane, sim=sim))  # undirected traversal

    return adj


# ----------------------------
# Path search
# ----------------------------

@dataclass
class Step:
    a: str
    b: str
    plane: str
    sim: float
    known: bool
    curv_abs: Optional[float]
    curv_norm: Optional[float]
    cons: Optional[float]

@dataclass
class PathResult:
    nodes: List[str]
    steps: List[Step]
    total: float
    transport: float
    curv: float
    switch: int
    incoh: float
    mean_sim: float
    mean_cons: float
    unknown: int
    known: int

def curvature_scale_for_plane(known_map: Dict[Tuple[str,str], KnownEdge], mode: str, eps: float) -> float:
    vals = [ke.curv_abs for ke in known_map.values() if ke.curv_abs is not None]
    if not vals:
        return 1.0
    if mode == "mean":
        s = sum(vals) / max(1, len(vals))
    else:
        # median default
        s = statistics.median(vals)
    return max(eps, float(s))

def score_path(
    steps: List[Step],
    w_transport: float,
    w_curv: float,
    w_switch: float,
    w_incoh: float,
    w_unknown: float,
) -> Tuple[float, dict]:

    transport = sum(safe_log_sim(s.sim) for s in steps)

    # Curvature: if curv_norm exists, use it; else 0.
    curv = 0.0
    for st in steps:
        if st.curv_norm is not None:
            curv += st.curv_norm
        elif st.curv_abs is not None:
            # fallback if something forgot to set norm
            curv += st.curv_abs

    # plane switches
    switch = 0
    for i in range(1, len(steps)):
        if steps[i].plane != steps[i-1].plane:
            switch += 1

    # known/unknown counts and consistency
    known_steps = [st for st in steps if st.known]
    known_n = len(known_steps)
    unknown_n = len(steps) - known_n

    if known_n > 0:
        mean_cons = sum((st.cons or 0.0) for st in known_steps) / known_n
    else:
        mean_cons = 0.0

    incoh = 1.0 - mean_cons

    mean_sim = sum(st.sim for st in steps) / max(1, len(steps))

    total = (
        w_transport * transport
        + w_curv * curv
        + w_switch * switch
        + w_incoh * incoh
        + w_unknown * unknown_n
    )

    info = dict(
        transport=transport,
        curv=curv,
        switch=switch,
        incoh=incoh,
        mean_sim=mean_sim,
        mean_cons=mean_cons,
        unknown=unknown_n,
        known=known_n
    )
    return total, info

def enumerate_paths(adj, start, goal, max_len):
    """
    Enumerate all simple paths up to length max_len (edges) using DFS.
    """
    results = []
    stack = [(start, [start], [])]  # (node, path_nodes, path_edges(CacheEdge list))
    while stack:
        node, path_nodes, path_edges = stack.pop()
        if node == goal:
            results.append((path_nodes, path_edges))
            continue
        if len(path_edges) >= max_len:
            continue
        for e in adj.get(node, []):
            if e.b in path_nodes:
                continue
            stack.append((e.b, path_nodes + [e.b], path_edges + [e]))
    return results


# ----------------------------
# Visualization
# ----------------------------

def draw_graph_png(
    out_png: str,
    nodes: List[str],
    all_edges: List[CacheEdge],
    best_path_steps: Optional[List[Step]],
    title: str
):
    if nx is None:
        print("[warn] networkx not available; skipping graph PNG.")
        return

    G = nx.Graph()
    for n in nodes:
        G.add_node(n)

    # collapse multi-plane edges to (a,b,plane) as separate edges by creating edge keys
    # but networkx Graph supports one edge per pair; so we store the BEST sim and plane label.
    # For visualization we keep the max sim edge per pair and annotate plane.
    best_per_pair: Dict[Tuple[str,str], Tuple[float,str]] = {}
    for e in all_edges:
        k = edge_id(e.a, e.b)
        cur = best_per_pair.get(k)
        if (cur is None) or (e.sim > cur[0]):
            best_per_pair[k] = (e.sim, e.plane)

    for (a,b), (sim, plane) in best_per_pair.items():
        G.add_edge(a, b, weight=sim, plane=plane)

    pos = nx.spring_layout(G, seed=7, k=1.2)  # deterministic-ish layout

    plt.figure(figsize=(10, 8))
    plt.title(title)

    # draw edges (base)
    edge_labels = {}
    for (u, v, d) in G.edges(data=True):
        edge_labels[(u, v)] = f"{d.get('plane','?')}:{d.get('weight',0):.2f}"

    nx.draw_networkx_nodes(G, pos, node_size=1500)
    nx.draw_networkx_labels(G, pos, font_size=9)

    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

    # highlight best path
    if best_path_steps:
        hp = [(st.a, st.b) for st in best_path_steps]
        nx.draw_networkx_edges(G, pos, edgelist=hp, width=4.0, alpha=0.9)

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--cache", required=True)

    ap.add_argument("--edge_axis_json_lt", default=None)
    ap.add_argument("--edge_axis_json_bo_lr", default=None)
    ap.add_argument("--edge_axis_json_bo", default=None)

    ap.add_argument("--clusters_json_lt", default=None)
    ap.add_argument("--clusters_json_bo_lr", default=None)
    ap.add_argument("--clusters_json_bo", default=None)

    ap.add_argument("--planes", nargs="+", required=True)

    ap.add_argument("--start", required=True)
    ap.add_argument("--goal", required=True)

    ap.add_argument("--max_len", type=int, default=5)
    ap.add_argument("--topk", type=int, default=25)

    ap.add_argument("--min_edge_sim", type=float, default=0.25)
    ap.add_argument("--min_edge_sim_draw", type=float, default=0.25)

    # weights
    ap.add_argument("--w_transport", type=float, default=1.0)
    ap.add_argument("--w_curv", type=float, default=1.0)
    ap.add_argument("--w_switch", type=float, default=0.25)
    ap.add_argument("--w_incoh", type=float, default=0.5)
    ap.add_argument("--w_unknown", type=float, default=0.75)

    ap.add_argument("--require_known_k", type=int, default=0)

    # NEW: curvature normalization
    ap.add_argument("--curv_norm", choices=["none", "median", "mean"], default="median",
                    help="Normalize curv_abs per-plane by median/mean of known edges in that plane.")
    ap.add_argument("--curv_norm_eps", type=float, default=1e-6)

    # deliverables
    ap.add_argument("--outdir", default=None, help="If set, saves phase25_paths.json and phase25_graph.png")
    ap.add_argument("--tag", default="phase25", help="Filename prefix tag inside outdir")

    ap.add_argument("--debug_cache", action="store_true")

    args = ap.parse_args()

    planes = parse_planes(args.planes)

    # Load known edges per plane (from clusters_json_*)
    known_by_plane: Dict[str, Dict[Tuple[str,str], KnownEdge]] = {}
    plane_to_clusters = {
        "lt": args.clusters_json_lt,
        "bo_lr": args.clusters_json_bo_lr,
        "bo": args.clusters_json_bo,
    }
    for p in planes:
        cj = plane_to_clusters.get(p, None)
        if cj and os.path.exists(cj):
            known_by_plane[p] = load_known_edges_from_clusters_json(cj)
        else:
            known_by_plane[p] = {}

    # curvature scale per plane
    curv_scale = {}
    for p in planes:
        if args.curv_norm == "none":
            curv_scale[p] = 1.0
        else:
            curv_scale[p] = curvature_scale_for_plane(known_by_plane[p], args.curv_norm, args.curv_norm_eps)

    # adjacency from cache (filtered by min_edge_sim)
    adj = build_adjacency_from_cache(args.cache, planes, args.min_edge_sim, debug_cache=args.debug_cache)

    # gather node universe from adjacency
    all_nodes = sorted(adj.keys())

    print(f"Phase25 Geodesic Reasoning Probes")
    print(f"start={args.start} goal={args.goal} max_len={args.max_len} planes={planes}")
    print(f"weights: w_transport={args.w_transport} w_curv={args.w_curv} w_switch={args.w_switch} w_incoh={args.w_incoh} w_unknown={args.w_unknown}")
    print(f"require_known_k={args.require_known_k}")
    if args.curv_norm != "none":
        print(f"curv_norm={args.curv_norm} per-plane scales: " + ", ".join([f"{p}:{curv_scale[p]:.4f}" for p in planes]))
    else:
        print("curv_norm=none")

    if args.start not in adj:
        print(f"\n[warn] start node '{args.start}' not in cache adjacency. Available nodes sample: {all_nodes[:10]}")
    if args.goal not in adj:
        print(f"\n[warn] goal node '{args.goal}' not in cache adjacency. Available nodes sample: {all_nodes[:10]}")

    # enumerate candidate paths
    raw_paths = enumerate_paths(adj, args.start, args.goal, args.max_len)

    scored: List[PathResult] = []
    for path_nodes, path_edges in raw_paths:
        steps: List[Step] = []
        for e in path_edges:
            a, b, plane, sim = e.a, e.b, e.plane, e.sim
            kid = edge_id(a, b)
            ke = known_by_plane.get(plane, {}).get(kid, None)
            known = ke is not None
            if known:
                curv_abs = ke.curv_abs
                cons = ke.cons
                if args.curv_norm == "none":
                    curv_norm = curv_abs
                else:
                    curv_norm = curv_abs / curv_scale[plane]
            else:
                curv_abs = None
                curv_norm = None
                cons = None

            steps.append(Step(
                a=a, b=b, plane=plane, sim=sim,
                known=known, curv_abs=curv_abs, curv_norm=curv_norm, cons=cons
            ))

        total, info = score_path(
            steps,
            w_transport=args.w_transport,
            w_curv=args.w_curv,
            w_switch=args.w_switch,
            w_incoh=args.w_incoh,
            w_unknown=args.w_unknown
        )

        if info["known"] < args.require_known_k:
            continue

        scored.append(PathResult(
            nodes=path_nodes,
            steps=steps,
            total=total,
            transport=info["transport"],
            curv=info["curv"],
            switch=info["switch"],
            incoh=info["incoh"],
            mean_sim=info["mean_sim"],
            mean_cons=info["mean_cons"],
            unknown=info["unknown"],
            known=info["known"]
        ))

    scored.sort(key=lambda x: x.total)

    if not scored:
        print("\n[warn] No candidate paths found. Likely causes:")
        print("  - min_edge_sim too high")
        print("  - start/goal not connected in cache for selected planes")
        print("  - start/goal names typo")
        print("\nTry lowering --min_edge_sim (e.g. 0.10) or use a single plane to debug.")
        return

    print("\nTOP PATHS:\n")
    top = scored[:args.topk]
    for i, pr in enumerate(top, 1):
        print(f"[{i}] total={pr.total:.4f}  transport={pr.transport:.4f}  curv={pr.curv:.4f}  switch={pr.switch}  incoh={pr.incoh:.3f}  mean_sim={pr.mean_sim:.3f}  mean_cons={pr.mean_cons:.3f}  unknown={pr.unknown} known={pr.known}")
        for st in pr.steps:
            curv_s = "None" if st.curv_abs is None else f"{st.curv_abs:.3f}"
            cons_s = "None" if st.cons is None else f"{st.cons:.2f}"
            print(f"    {st.a} -[{st.plane} sim={st.sim:.3f} curv={curv_s} cons={cons_s}]-> {st.b}")
        print("")

    # Deliverables
    if args.outdir:
        ensure_dir(args.outdir)
        out_json = os.path.join(args.outdir, f"{args.tag}_paths.json")
        out_png = os.path.join(args.outdir, f"{args.tag}_graph.png")

        # collect a deduped set of edges for drawing
        draw_edges: List[CacheEdge] = []
        for n, elist in adj.items():
            for e in elist:
                if e.sim >= args.min_edge_sim_draw:
                    # avoid duplicating both directions by only taking canonical a<=b
                    if e.a <= e.b:
                        draw_edges.append(e)

        payload = {
            "phase": "25",
            "start": args.start,
            "goal": args.goal,
            "planes": planes,
            "args": vars(args),
            "curv_norm": args.curv_norm,
            "curv_scale": curv_scale,
            "min_edge_sim": args.min_edge_sim,
            "min_edge_sim_draw": args.min_edge_sim_draw,
            "topk": args.topk,
            "paths": [
                {
                    "rank": idx+1,
                    "total": pr.total,
                    "transport": pr.transport,
                    "curv": pr.curv,
                    "switch": pr.switch,
                    "incoh": pr.incoh,
                    "mean_sim": pr.mean_sim,
                    "mean_cons": pr.mean_cons,
                    "unknown": pr.unknown,
                    "known": pr.known,
                    "nodes": pr.nodes,
                    "steps": [
                        {
                            "a": st.a, "b": st.b, "plane": st.plane, "sim": st.sim,
                            "known": st.known,
                            "curv_abs": st.curv_abs,
                            "curv_norm": st.curv_norm,
                            "cons": st.cons
                        } for st in pr.steps
                    ]
                }
                for idx, pr in enumerate(top)
            ]
        }

        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)

        best_steps = top[0].steps if top else None
        draw_graph_png(
            out_png=out_png,
            nodes=sorted(set(all_nodes + [args.start, args.goal])),
            all_edges=draw_edges,
            best_path_steps=best_steps,
            title=f"Phase25 Graph ({args.start} -> {args.goal}) | planes={','.join(planes)} | curv_norm={args.curv_norm}"
        )

        print(f"[ok] wrote: {out_json}")
        if nx is None:
            print(f"[warn] networkx not installed; skipped: {out_png}")
        else:
            print(f"[ok] wrote: {out_png}")


if __name__ == "__main__":
    main()
