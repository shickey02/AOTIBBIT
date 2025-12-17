#!/usr/bin/env python3
# phase26d_geodesic_vector_rollouts.py
#
# Phase26d: Rollouts + attractor detection on the Phase26c-style geodesic vector field.
#
# Builds a "policy" over nodes: choose the lowest-cost outgoing edge (per your weighted geodesic edge cost),
# then runs deterministic rollouts to see where each node flows.
#
# Key deliverables:
# - Attractor / cycle discovery (including 2-cycles that appear in small graphs)
# - Basin assignment: which nodes flow into which attractor
# - Optional "no immediate backtrack" policy rule (avoid a->b->a pingpong by choosing 2nd-best edge)
#
# Outputs:
#   outdir/phase26d_vector_rollouts.json
#   outdir/phase26d_vector_rollouts.csv
#   (optional) outdir/phase26d_vector_rollouts.png  if --coords_json provided
#
# Cost model (matches 26c):
#   edge_cost = w_transport*transport + w_curv*curv_norm + w_incoh*incoh + w_unknown*unknown (+ optional w_switch*switch)
#
# Notes:
# - transport = (1 - similarity)
# - curv = sum(arccos(principal_cosines)) in radians
# - curv_norm = curv / curv_scale[plane] where curv_scale is median/mean per plane
# - incoh = 1 - cons, where cons is match_score_abs_diag_sum/k if present, else mean(principal_cosines) fallback
# - unknown = 1 if principal_cosines missing, else 0
#
import os, json, math, argparse
from collections import defaultdict, Counter
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
    s = 0.0
    for c in pc:
        c = clamp(c, -1.0, 1.0)
        s += math.acos(c)
    return s

def consistency_from_payload(payload: Dict[str, Any]) -> Optional[float]:
    """
    Returns 0..1 consistency estimate.
    Primary: match_score_abs_diag_sum / k
    Fallback: mean(principal_cosines)
    """
    if not isinstance(payload, dict):
        return None
    if "match_score_abs_diag_sum" in payload and "k" in payload:
        k = payload.get("k", None)
        if k and k > 0:
            return float(payload["match_score_abs_diag_sum"]) / float(k)
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
    Scale computed over ALL edges for that plane (from cache).
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
# Build outgoing candidates
# -------------------------

@dataclass
class EdgeCand:
    a: str
    b: str
    plane: str
    cost: float
    sim: float
    cons: Optional[float]
    transport: float
    curv: float
    curv_norm: float
    incoh: float
    unknown: int
    switch: int

def build_outgoing_candidates(transport_maps: Dict[str, Any],
                              planes: List[str],
                              min_edge_sim: float,
                              curv_scale: Dict[str, float],
                              w_transport: float, w_curv: float, w_switch: float, w_incoh: float, w_unknown: float,
                              last_plane_for_switch: Optional[str] = None) -> Dict[str, List[EdgeCand]]:
    """
    Build outgoing candidates per node, across selected planes, sorted by edge cost.
    Note: switch is 0 here by default unless you later compute it during rollout (we do).
    """
    outgoing = defaultdict(list)

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

            sim = float(payload.get("similarity", 0.0))
            if sim < min_edge_sim:
                continue

            pcs = payload.get("principal_cosines", None)
            cons = consistency_from_payload(payload)

            unknown = 0
            curv = 0.0
            curv_norm = 0.0
            if pcs:
                curv = curvature_from_principal_cosines(pcs)
                curv_norm = float(curv) / float(curv_scale.get(plane, 1.0))
            else:
                unknown = 1
                curv = 0.0
                curv_norm = 0.0  # no curvature term if unknown

            transport = edge_transport(sim)
            incoh = edge_incoherence(cons)

            # switch cost is computed during rollout (depends on last_plane); placeholder here
            sw = 0

            cost = (w_transport * transport) + (w_curv * curv_norm) + (w_switch * sw) + (w_incoh * incoh) + (w_unknown * unknown)

            outgoing[a].append(EdgeCand(
                a=a, b=b, plane=plane,
                cost=float(cost), sim=sim, cons=cons,
                transport=float(transport),
                curv=float(curv),
                curv_norm=float(curv_norm),
                incoh=float(incoh),
                unknown=int(unknown),
                switch=int(sw)
            ))

    # sort by cost then tie-break deterministically
    for a in list(outgoing.keys()):
        outgoing[a].sort(key=lambda e: (e.cost, -e.sim, e.plane, e.b))

    return outgoing

# -------------------------
# Policy + rollouts
# -------------------------

def choose_next(out_edges: List[EdgeCand],
                prev_node: Optional[str],
                last_plane: Optional[str],
                w_switch: float,
                no_backtrack: bool,
                backtrack_mode: str,
                backtrack_penalty: float) -> Optional[EdgeCand]:
    """
    Select next edge:
    - base ranking is by cost, but we may need to add switch/backtrack penalties dynamically.
    - if no_backtrack: avoid choosing edge that goes immediately back to prev_node, if an alternative exists.
    - backtrack_mode:
        * "forbid"  : treat as disallowed (if alternatives exist)
        * "penalize": add backtrack_penalty to that edge's effective cost
    """
    if not out_edges:
        return None

    best = None
    best_eff = float("inf")

    for e in out_edges:
        eff = e.cost

        # dynamic switch penalty
        sw = 0
        if last_plane is not None and e.plane != last_plane:
            sw = 1
        eff += (w_switch * sw)

        # dynamic backtrack handling
        is_back = (prev_node is not None and e.b == prev_node)
        if no_backtrack and is_back:
            if backtrack_mode == "forbid":
                # skip for now; if everything is backtrack, we'll fall back later
                continue
            elif backtrack_mode == "penalize":
                eff += float(backtrack_penalty)

        if eff < best_eff:
            best_eff = eff
            best = EdgeCand(**{**e.__dict__, "switch": sw, "cost": float(best_eff)})

    # if forbid caused no choice, fall back to best ignoring backtrack
    if best is None:
        e0 = out_edges[0]
        sw = 0
        if last_plane is not None and e0.plane != last_plane:
            sw = 1
        eff = e0.cost + (w_switch * sw)
        best = EdgeCand(**{**e0.__dict__, "switch": sw, "cost": float(eff)})

    return best

def rollout_from(start: str,
                 outgoing: Dict[str, List[EdgeCand]],
                 max_steps: int,
                 no_backtrack: bool,
                 backtrack_mode: str,
                 backtrack_penalty: float,
                 w_switch: float) -> Dict[str, Any]:
    """
    Deterministic rollout until:
      - dead end (no outgoing edges)
      - repeat node -> cycle detected
      - step limit reached
    """
    path_nodes = [start]
    path_edges: List[EdgeCand] = []

    seen_index = {start: 0}
    prev_node = None
    last_plane = None

    for t in range(max_steps):
        cur = path_nodes[-1]
        cand = choose_next(outgoing.get(cur, []), prev_node, last_plane,
                           w_switch=w_switch,
                           no_backtrack=no_backtrack,
                           backtrack_mode=backtrack_mode,
                           backtrack_penalty=backtrack_penalty)
        if cand is None:
            return {
                "start": start,
                "status": "dead_end",
                "steps": len(path_edges),
                "path": path_nodes,
                "edges": [cand_to_dict(e) for e in path_edges],
                "cycle": None
            }

        # advance
        path_edges.append(cand)
        nxt = cand.b
        prev_node = cur
        last_plane = cand.plane
        path_nodes.append(nxt)

        if nxt in seen_index:
            i0 = seen_index[nxt]
            cycle_nodes = path_nodes[i0:]
            return {
                "start": start,
                "status": "cycle",
                "steps": len(path_edges),
                "path": path_nodes,
                "edges": [cand_to_dict(e) for e in path_edges],
                "cycle": {
                    "entry_index": i0,
                    "cycle_nodes": cycle_nodes
                }
            }
        seen_index[nxt] = len(path_nodes) - 1

    return {
        "start": start,
        "status": "max_steps",
        "steps": len(path_edges),
        "path": path_nodes,
        "edges": [cand_to_dict(e) for e in path_edges],
        "cycle": None
    }

def cand_to_dict(e: EdgeCand) -> Dict[str, Any]:
    return {
        "a": e.a, "b": e.b, "plane": e.plane,
        "cost": e.cost,
        "sim": e.sim, "cons": e.cons,
        "transport": e.transport,
        "curv": e.curv,
        "curv_norm": e.curv_norm,
        "switch": e.switch,
        "incoh": e.incoh,
        "unknown": e.unknown
    }

def canonical_cycle_key(cycle_nodes: List[str]) -> Tuple[str, ...]:
    """
    Make a stable, comparable representation of a cycle.
    cycle_nodes is like [X, Y, X] or [X, Y, Z, X] depending on path slice.
    We'll remove the trailing repeat if present, then rotate so lexicographically smallest node starts the tuple.
    """
    if len(cycle_nodes) >= 2 and cycle_nodes[0] == cycle_nodes[-1]:
        cycle_nodes = cycle_nodes[:-1]

    if not cycle_nodes:
        return tuple()

    # rotate to smallest lexicographic start for stability
    m = min(range(len(cycle_nodes)), key=lambda i: cycle_nodes[i])
    rot = cycle_nodes[m:] + cycle_nodes[:m]
    return tuple(rot)

# -------------------------
# Optional plotting
# -------------------------

def plot_vector_field(coords: Dict[str, List[float]],
                      best_edge: Dict[str, EdgeCand],
                      out_png: str,
                      title: str = "Phase26d Vector Rollouts"):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[warn] matplotlib not available; skipping png")
        return

    xs = []
    ys = []
    for n, (x, y) in coords.items():
        xs.append(x); ys.append(y)

    plt.figure(figsize=(8, 6))
    plt.scatter(xs, ys)

    # labels
    for n, (x, y) in coords.items():
        plt.text(x, y, f" {n}", fontsize=9)

    # arrows
    for a, e in best_edge.items():
        if a not in coords or e.b not in coords:
            continue
        x1, y1 = coords[a]
        x2, y2 = coords[e.b]
        dx, dy = (x2 - x1), (y2 - y1)
        plt.arrow(x1, y1, dx, dy, length_includes_head=True, head_width=0.02, alpha=0.7)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--planes", nargs="+", required=True)

    ap.add_argument("--min_edge_sim", type=float, default=0.10)
    ap.add_argument("--topk_per_node", type=int, default=5)

    ap.add_argument("--max_steps", type=int, default=20)
    ap.add_argument("--starts", nargs="*", default=None,
                    help="Optional list of start nodes. If omitted, rollouts run from ALL nodes.")

    # weights (match 26c defaults)
    ap.add_argument("--w_transport", type=float, default=1.0)
    ap.add_argument("--w_curv", type=float, default=1.0)
    ap.add_argument("--w_incoh", type=float, default=0.5)
    ap.add_argument("--w_unknown", type=float, default=0.75)
    ap.add_argument("--w_switch", type=float, default=0.0, help="Optional switch penalty (plane changes). Default 0.0 to match 26c.")

    ap.add_argument("--curv_norm", choices=["none", "median", "mean"], default="median")

    # backtrack behavior
    ap.add_argument("--no_backtrack", action="store_true",
                    help="Avoid immediate a->b->a pingpong by selecting next-best outgoing when possible.")
    ap.add_argument("--backtrack_mode", choices=["forbid", "penalize"], default="forbid")
    ap.add_argument("--backtrack_penalty", type=float, default=0.5,
                    help="Used only when --backtrack_mode penalize.")

    # optional plot
    ap.add_argument("--coords_json", default=None,
                    help="Optional coords json (e.g., from 26c) to draw arrows. If provided, writes a PNG.")
    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()
    ensure_dir(args.outdir)

    d = load_json(args.cache)
    if "transport_maps" not in d:
        raise SystemExit("[error] cache missing transport_maps")
    tm = d["transport_maps"]

    # collect nodes
    nodes = set()
    for k in tm.keys():
        parsed = parse_edge_key(k)
        if parsed:
            a, b = parsed
            nodes.add(a); nodes.add(b)
    nodes = sorted(nodes)

    planes = args.planes
    curv_scale = compute_plane_curv_scale(tm, planes, args.curv_norm)

    # outgoing candidates
    outgoing = build_outgoing_candidates(
        tm, planes, args.min_edge_sim, curv_scale,
        args.w_transport, args.w_curv, args.w_switch, args.w_incoh, args.w_unknown
    )

    # apply topk cut
    for a in list(outgoing.keys()):
        outgoing[a] = outgoing[a][:max(1, args.topk_per_node)]

    # pick starts
    if args.starts is None or len(args.starts) == 0:
        starts = nodes
    else:
        starts = args.starts
        missing = [s for s in starts if s not in nodes]
        if missing:
            raise SystemExit(f"[error] starts not in cache nodes: {missing}")

    if args.debug:
        print("[debug] nodes:", nodes)
        print("[debug] curv_scale:", curv_scale)
        print("[debug] outgoing_counts:", {n: len(outgoing.get(n, [])) for n in nodes})

    # rollouts
    rollouts = []
    cycle_key_for_start = {}
    for s in starts:
        r = rollout_from(
            s, outgoing,
            max_steps=args.max_steps,
            no_backtrack=args.no_backtrack,
            backtrack_mode=args.backtrack_mode,
            backtrack_penalty=args.backtrack_penalty,
            w_switch=args.w_switch
        )
        rollouts.append(r)
        if r["status"] == "cycle" and r["cycle"] is not None:
            ck = canonical_cycle_key(r["cycle"]["cycle_nodes"])
            cycle_key_for_start[s] = ck
        else:
            cycle_key_for_start[s] = None

    # aggregate attractors
    attractor_counts = Counter([ck for ck in cycle_key_for_start.values() if ck is not None])
    attractors = []
    for ck, cnt in attractor_counts.most_common():
        attractors.append({
            "cycle_key": list(ck),
            "size": len(ck),
            "count_starts_reaching": int(cnt)
        })

    # basin assignment (starts -> attractor_key or None)
    basin = {}
    for s in starts:
        ck = cycle_key_for_start.get(s, None)
        basin[s] = (list(ck) if ck is not None else None)

    # best-edge summary (first candidate per node under the policy, for reference/plot)
    best_edge = {}
    for n in nodes:
        if outgoing.get(n):
            # choose best under policy assuming no prev/last_plane at node start
            chosen = choose_next(outgoing[n], prev_node=None, last_plane=None,
                                 w_switch=args.w_switch,
                                 no_backtrack=False,  # policy base
                                 backtrack_mode="forbid",
                                 backtrack_penalty=0.0)
            if chosen is not None:
                best_edge[n] = chosen

    out = {
        "phase": "26d",
        "status": "ok",
        "args": vars(args),
        "planes": planes,
        "curv_norm": args.curv_norm,
        "curv_scale": curv_scale,
        "nodes": nodes,
        "outgoing_topk": {n: [cand_to_dict(e) for e in outgoing.get(n, [])] for n in nodes},
        "best_edge_policy_base": {n: cand_to_dict(e) for n, e in best_edge.items()},
        "attractors": attractors,
        "basin": basin,
        "rollouts": rollouts
    }

    out_json = os.path.join(args.outdir, "phase26d_vector_rollouts.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)

    # CSV: one row per start
    csv_lines = ["start,status,steps,attractor_size,attractor_key,path"]
    for r in rollouts:
        s = r["start"]
        status = r["status"]
        steps = r["steps"]
        ak = basin.get(s, None)
        ak_str = "" if ak is None else "|".join(ak)
        ak_size = "" if ak is None else str(len(ak))
        path_str = "->".join(r["path"])
        csv_lines.append(f"{s},{status},{steps},{ak_size},{ak_str},{path_str}")

    out_csv = os.path.join(args.outdir, "phase26d_vector_rollouts.csv")
    with open(out_csv, "w", newline="") as f:
        f.write("\n".join(csv_lines) + "\n")

    # optional plot using coords_json
    if args.coords_json:
        coords_obj = load_json(args.coords_json)
        # accept either a raw {"node":[x,y]} or a wrapper {"coords":{...}}
        coords = coords_obj.get("coords", coords_obj)
        out_png = os.path.join(args.outdir, "phase26d_vector_rollouts.png")
        plot_vector_field(coords, best_edge, out_png, title="Phase26d Vector Policy (base)")

        print(f"[ok] wrote: {out_png}")

    # console summary
    print("Phase26d Vector Rollouts")
    print(f"planes={planes} curv_norm={args.curv_norm} min_edge_sim={args.min_edge_sim} topk_per_node={args.topk_per_node}")
    print(f"no_backtrack={args.no_backtrack} backtrack_mode={args.backtrack_mode}")
    print(f"starts={len(starts)} max_steps={args.max_steps}")
    print("curv_scale:", curv_scale)
    if attractors:
        print("\nATTRACTORS (cycles) found:")
        for i, a in enumerate(attractors, 1):
            print(f"  [{i}] size={a['size']} count_starts_reaching={a['count_starts_reaching']} key={a['cycle_key']}")
    else:
        print("\nATTRACTORS: none (no cycles detected within max_steps)")

    print(f"\n[ok] wrote: {out_json}")
    print(f"[ok] wrote: {out_csv}")

if __name__ == "__main__":
    main()
