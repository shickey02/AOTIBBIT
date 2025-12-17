#!/usr/bin/env python3
# phase26c_geodesic_vector_field.py
#
# Phase26c — Geodesic Vector Field Visualization
#
# Goal:
#   Show the "arrow field" implied by your geodesic edge metric.
#   For each node, pick the best outgoing edge (lowest edge-cost) and draw an arrow.
#
# Inputs:
#   - phase15f_transport_cache.json (transport_maps with per-plane payloads)
#
# Scoring (edge cost) matches Phase26b local edge cost:
#   edge_cost = w_transport*(1-sim) + w_curv*(curv/curv_scale[plane]) + w_switch*switch + w_incoh*incoh + w_unknown*unknown
#   NOTE: For the local vector field, switch is 0 because we’re not chaining (no last_plane).
#
# Curvature normalization:
#   --curv_norm {none,median,mean} computes curv_scale per plane over ALL edges with principal_cosines.
#
# Layout:
#   - If you provide --coords_json (node -> [x,y]) we use it.
#   - Otherwise we compute a simple force-directed layout (FR) so the arrows are drawable.
#
# Outputs (in --outdir):
#   - phase26c_vector_field.png
#   - phase26c_vector_field.svg
#   - phase26c_vector_field.json  (chosen best outgoing edge per node + costs + layout coords)
#
# Example:
#   python bbit_geomlang/phase26c_geodesic_vector_field.py `
#     --cache outputs_edges_relternary256_phase15/phase15f_transport_cache.json `
#     --outdir outputs_edges_relternary256_phase15/phase26c_field `
#     --planes lt bo_lr `
#     --min_edge_sim 0.10 `
#     --w_transport 1.0 --w_curv 1.0 --w_incoh 0.5 --w_unknown 0.75 `
#     --curv_norm median `
#     --label_nodes
#
import os, json, math, argparse, random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

import matplotlib.pyplot as plt

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

def curvature_from_principal_cosines(pc: List[float]) -> float:
    s = 0.0
    for c in pc:
        s += math.acos(clamp(float(c), -1.0, 1.0))
    return s

def consistency_from_payload(payload: Dict[str, Any]) -> Optional[float]:
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

def edge_transport(sim: float) -> float:
    return 1.0 - sim

def edge_incoherence(cons: Optional[float]) -> float:
    if cons is None:
        return 1.0
    return clamp(1.0 - float(cons), 0.0, 1.0)

# -------------------------
# Curvature normalization per-plane
# -------------------------

def compute_plane_curv_scale(transport_maps: Dict[str, Any], planes: List[str], norm: str) -> Dict[str, float]:
    """
    norm in {"none","median","mean"}.
    Scale computed over ALL edges available for that plane in the cache.
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
# Build adjacency
# -------------------------

def collect_nodes(transport_maps: Dict[str, Any]) -> List[str]:
    nodes = set()
    for k in transport_maps.keys():
        parsed = parse_edge_key(k)
        if not parsed:
            continue
        a, b = parsed
        nodes.add(a); nodes.add(b)
    return sorted(nodes)

def build_adjacency(transport_maps: Dict[str, Any], planes: List[str], min_edge_sim: float):
    adj = defaultdict(list)  # a -> list of (b, plane, payload)
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
            adj[a].append((b, plane, payload))
    return adj

# -------------------------
# Local best-edge selection (vector field)
# -------------------------

@dataclass
class BestEdge:
    a: str
    b: str
    plane: str
    sim: float
    cons: Optional[float]
    transport: float
    curv: Optional[float]
    curv_norm: float
    incoh: float
    unknown: int
    cost: float

def best_outgoing_edge_for_node(
    node: str,
    adj,
    curv_scale: Dict[str, float],
    w_transport: float, w_curv: float, w_incoh: float, w_unknown: float,
) -> Optional[BestEdge]:
    best = None
    for (nb, plane, payload) in adj.get(node, []):
        sim = float(payload.get("similarity", 0.0))
        pcs = payload.get("principal_cosines", None)
        cons = consistency_from_payload(payload)

        curv = curvature_from_principal_cosines(pcs) if pcs else None
        if curv is None:
            unknown = 1
            curv_norm = 0.0
        else:
            unknown = 0
            denom = curv_scale.get(plane, 1.0) or 1.0
            curv_norm = float(curv) / float(denom)

        transport = edge_transport(sim)
        incoh = edge_incoherence(cons)

        cost = (w_transport * transport) + (w_curv * curv_norm) + (w_incoh * incoh) + (w_unknown * unknown)

        be = BestEdge(
            a=node, b=nb, plane=plane,
            sim=sim, cons=cons,
            transport=transport,
            curv=(float(curv) if curv is not None else None),
            curv_norm=float(curv_norm),
            incoh=float(incoh),
            unknown=int(unknown),
            cost=float(cost),
        )
        if best is None or be.cost < best.cost:
            best = be
    return best

# -------------------------
# Layout
# -------------------------

def load_coords_json(path: str) -> Dict[str, Tuple[float, float]]:
    d = load_json(path)
    coords = {}
    # Accept formats:
    #  { "node": [x,y], ... }
    #  { "coords": { "node": [x,y], ... } }
    if isinstance(d, dict) and "coords" in d and isinstance(d["coords"], dict):
        d = d["coords"]
    if not isinstance(d, dict):
        raise SystemExit("[error] coords_json must be a dict mapping node-> [x,y] (or wrapped in {'coords': ...})")
    for k, v in d.items():
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            coords[k] = (float(v[0]), float(v[1]))
    return coords

def fr_layout(nodes: List[str], edges: List[Tuple[str, str]], iters: int = 500, seed: int = 0) -> Dict[str, Tuple[float, float]]:
    """
    Tiny Fruchterman–Reingold style layout (no external deps).
    Nodes initialized randomly in unit square; iteratively relax.
    """
    rng = random.Random(seed)
    pos = {n: [rng.random(), rng.random()] for n in nodes}
    idx = {n: i for i, n in enumerate(nodes)}
    n = max(1, len(nodes))
    area = 1.0
    k = math.sqrt(area / n)

    # adjacency for attractive forces
    edge_list = [(a, b) for (a, b) in edges if a in idx and b in idx]

    def cool(t):  # simple linear cooling
        return 0.1 * (1.0 - t)

    for t in range(iters):
        temp = cool(t / max(1, iters - 1))
        disp = {v: [0.0, 0.0] for v in nodes}

        # repulsive
        for i, v in enumerate(nodes):
            for u in nodes[i+1:]:
                dx = pos[v][0] - pos[u][0]
                dy = pos[v][1] - pos[u][1]
                dist = math.hypot(dx, dy) + 1e-9
                force = (k * k) / dist
                rx = (dx / dist) * force
                ry = (dy / dist) * force
                disp[v][0] += rx; disp[v][1] += ry
                disp[u][0] -= rx; disp[u][1] -= ry

        # attractive
        for (v, u) in edge_list:
            dx = pos[v][0] - pos[u][0]
            dy = pos[v][1] - pos[u][1]
            dist = math.hypot(dx, dy) + 1e-9
            force = (dist * dist) / k
            ax = (dx / dist) * force
            ay = (dy / dist) * force
            disp[v][0] -= ax; disp[v][1] -= ay
            disp[u][0] += ax; disp[u][1] += ay

        # update
        for v in nodes:
            dx, dy = disp[v]
            dist = math.hypot(dx, dy) + 1e-9
            # limit step by temperature
            pos[v][0] += (dx / dist) * min(dist, temp)
            pos[v][1] += (dy / dist) * min(dist, temp)
            # keep in bounds
            pos[v][0] = clamp(pos[v][0], 0.0, 1.0)
            pos[v][1] = clamp(pos[v][1], 0.0, 1.0)

    return {n: (pos[n][0], pos[n][1]) for n in nodes}

# -------------------------
# Plotting
# -------------------------

def plot_vector_field(
    coords: Dict[str, Tuple[float, float]],
    best_edges: Dict[str, BestEdge],
    planes: List[str],
    out_png: str,
    out_svg: str,
    start: Optional[str] = None,
    label_nodes: bool = False,
    title: str = "Phase26c Geodesic Vector Field"
):
    # Use default matplotlib cycle; map plane -> color index
    plane_to_idx = {p: i for i, p in enumerate(planes)}

    xs = [coords[n][0] for n in coords.keys()]
    ys = [coords[n][1] for n in coords.keys()]

    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")

    # nodes
    ax.scatter(xs, ys, s=60, alpha=0.85)

    # labels
    if label_nodes:
        for n, (x, y) in coords.items():
            ax.text(x, y, n, fontsize=9, ha="left", va="bottom")

    # arrows (best outgoing edge per node)
    for a, be in best_edges.items():
        if be is None:
            continue
        if be.a not in coords or be.b not in coords:
            continue
        x1, y1 = coords[be.a]
        x2, y2 = coords[be.b]
        dx, dy = (x2 - x1), (y2 - y1)

        # line width scales with "confidence": lower cost => thicker
        # keep it stable:
        lw = 1.0 + 2.5 * (1.0 / (1.0 + be.cost))

        ci = plane_to_idx.get(be.plane, 0)
        # color from default cycle
        color = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0'])[ci % 10]

        ax.arrow(
            x1, y1, dx, dy,
            length_includes_head=True,
            head_width=0.015,
            head_length=0.02,
            linewidth=lw,
            alpha=0.75,
            color=color
        )

    # highlight start if provided
    if start and start in coords:
        sx, sy = coords[start]
        ax.scatter([sx], [sy], s=180, marker="*", alpha=0.95)
        ax.text(sx, sy, f"  START:{start}", fontsize=11, ha="left", va="center")

    # legend proxy artists
    proxies = []
    labels = []
    for p in planes:
        ci = plane_to_idx.get(p, 0)
        color = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0'])[ci % 10]
        proxies.append(plt.Line2D([0], [0], color=color, lw=3))
        labels.append(p)
    ax.legend(proxies, labels, title="plane", loc="best")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()

    fig.savefig(out_png, dpi=200)
    fig.savefig(out_svg)
    plt.close(fig)

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--planes", nargs="+", required=True)

    ap.add_argument("--start", default=None)
    ap.add_argument("--min_edge_sim", type=float, default=0.10)

    ap.add_argument("--w_transport", type=float, default=1.0)
    ap.add_argument("--w_curv", type=float, default=1.0)
    ap.add_argument("--w_incoh", type=float, default=0.5)
    ap.add_argument("--w_unknown", type=float, default=0.75)

    ap.add_argument("--curv_norm", choices=["none", "median", "mean"], default="median")

    ap.add_argument("--coords_json", default=None, help="Optional: node -> [x,y] mapping; otherwise FR layout is used")
    ap.add_argument("--layout_iters", type=int, default=600)
    ap.add_argument("--layout_seed", type=int, default=0)

    ap.add_argument("--label_nodes", action="store_true")
    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()
    ensure_dir(args.outdir)

    d = load_json(args.cache)
    if "transport_maps" not in d:
        raise SystemExit("[error] cache missing transport_maps")
    tm = d["transport_maps"]

    planes = args.planes
    nodes = collect_nodes(tm)

    if args.start and args.start not in nodes:
        raise SystemExit(f"[error] start node '{args.start}' not found. nodes={nodes}")

    curv_scale = compute_plane_curv_scale(tm, planes, args.curv_norm)
    adj = build_adjacency(tm, planes, args.min_edge_sim)

    # Choose best outgoing edge per node
    best_edges: Dict[str, Optional[BestEdge]] = {}
    edge_pairs = []
    for n in nodes:
        be = best_outgoing_edge_for_node(
            n, adj, curv_scale,
            args.w_transport, args.w_curv, args.w_incoh, args.w_unknown
        )
        best_edges[n] = be
        if be is not None:
            edge_pairs.append((be.a, be.b))

    # Layout
    if args.coords_json:
        coords = load_coords_json(args.coords_json)
        # ensure all nodes exist; add missing nodes to avoid KeyError
        missing = [n for n in nodes if n not in coords]
        if missing:
            if args.debug:
                print("[debug] coords_json missing nodes, adding FR positions for:", missing)
            fr = fr_layout(missing, edge_pairs, iters=max(200, args.layout_iters // 2), seed=args.layout_seed)
            coords.update(fr)
    else:
        coords = fr_layout(nodes, edge_pairs, iters=args.layout_iters, seed=args.layout_seed)

    # Save JSON deliverable
    out_json = os.path.join(args.outdir, "phase26c_vector_field.json")
    out = {
        "phase": "26c",
        "status": "ok",
        "args": vars(args),
        "planes": planes,
        "curv_norm": args.curv_norm,
        "curv_scale": curv_scale,
        "nodes": nodes,
        "coords": {n: [coords[n][0], coords[n][1]] for n in nodes if n in coords},
        "best_outgoing": {}
    }
    for n, be in best_edges.items():
        if be is None:
            out["best_outgoing"][n] = None
        else:
            out["best_outgoing"][n] = {
                "a": be.a, "b": be.b, "plane": be.plane,
                "cost": be.cost,
                "sim": be.sim, "cons": be.cons,
                "transport": be.transport,
                "curv": be.curv,
                "curv_norm": be.curv_norm,
                "incoh": be.incoh,
                "unknown": be.unknown,
            }

    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)

    # Plot
    out_png = os.path.join(args.outdir, "phase26c_vector_field.png")
    out_svg = os.path.join(args.outdir, "phase26c_vector_field.svg")
    plot_vector_field(
        coords=coords,
        best_edges={k: v for k, v in best_edges.items() if v is not None},
        planes=planes,
        out_png=out_png,
        out_svg=out_svg,
        start=args.start,
        label_nodes=args.label_nodes,
        title=f"Phase26c Geodesic Vector Field (min_sim={args.min_edge_sim}, curv_norm={args.curv_norm})"
    )

    # Console summary
    if args.debug:
        print("[debug] nodes:", nodes)
        print("[debug] curv_scale:", curv_scale)
        print("[debug] best outgoing edges:")
        for n in nodes:
            be = best_edges[n]
            if be is None:
                print(f"  {n}: (none)")
            else:
                print(f"  {n}: {be.a} -> {be.b} plane={be.plane} cost={be.cost:.4f} sim={be.sim:.3f} curvN={be.curv_norm:.3f} incoh={be.incoh:.3f} unk={be.unknown}")

    print("Phase26c Geodesic Vector Field")
    print(f"planes={planes} min_edge_sim={args.min_edge_sim} curv_norm={args.curv_norm}")
    print("curv_scale:", curv_scale)
    print(f"[ok] wrote: {out_json}")
    print(f"[ok] wrote: {out_png}")
    print(f"[ok] wrote: {out_svg}")

if __name__ == "__main__":
    main()
