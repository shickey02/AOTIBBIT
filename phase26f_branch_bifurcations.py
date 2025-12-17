#!/usr/bin/env python3
# phase26f_branch_bifurcations.py
#
# Branching bifurcation probe:
# - For each start node, branch at EVERY step up to depth D (prefix search)
# - For each prefix, optionally continue with greedy rollout to a terminal attractor (cycle) or dead-end
# - Attractor definition:
#     when repetition occurs at node nb, cycle = path[j:-1] (exclude final repeated nb),
#     then reduce to smallest period, then canonicalize by rotation (lexicographically smallest).
#
# Outputs:
#   * phase26f_branch_bifurcations.csv
#   * phase26f_branch_bifurcations.json
#
# Designed to be consistent with Phase25/26 cost model:
#   edge_cost = w_transport*(1-sim) + w_curv*curv_norm + w_switch*switch + w_incoh*incoh + w_unknown*unknown
#
import os, json, math, argparse, itertools
from collections import defaultdict, Counter, deque
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

# -------------------------
# IO / utils
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
    if not isinstance(payload, dict):
        return None
    if "match_score_abs_diag_sum" in payload and "k" in payload:
        k = payload.get("k", None)
        if k and k > 0:
            return float(payload["match_score_abs_diag_sum"]) / float(k)
    if "principal_cosines" in payload:
        pcs = payload["principal_cosines"]
        if pcs:
            return float(sum(pcs) / max(1, len(pcs)))
    return None

def edge_transport(sim: float) -> float:
    return 1.0 - sim

def edge_incoherence(cons: Optional[float]) -> float:
    if cons is None:
        return 1.0
    return clamp(1.0 - float(cons), 0.0, 1.0)

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
# Candidate edges per node
# -------------------------

@dataclass
class CandEdge:
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
    switch: int
    cost: float

def build_candidate_edges(
    transport_maps: Dict[str, Any],
    nodes: List[str],
    planes: List[str],
    min_edge_sim: float,
    curv_scale: Dict[str, float],
    w_transport: float,
    w_curv: float,
    w_switch: float,
    w_incoh: float,
    w_unknown: float,
) -> Dict[str, List[CandEdge]]:
    """
    Build and sort all candidate outgoing edges per node, scored under weights.
    switch is computed later (depends on last_plane), so we store base_cost_no_switch and set switch/cost later.
    We'll handle switch dynamically by recomputing per query.
    """
    # adjacency raw per node: list of (b, plane, payload)
    raw = defaultdict(list)
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
            raw[a].append((b, plane, payload))

    # build partial edge objects (without switch)
    out = {n: [] for n in nodes}
    for a, lst in raw.items():
        for (b, plane, payload) in lst:
            sim = float(payload.get("similarity", 0.0))
            pcs = payload.get("principal_cosines", None)
            cons = consistency_from_payload(payload)
            curv = curvature_from_principal_cosines(pcs) if pcs else None
            if curv is None:
                unknown = 1
                curv_norm = 0.0
            else:
                unknown = 0
                curv_norm = float(curv) / float(curv_scale.get(plane, 1.0))

            transport = edge_transport(sim)
            incoh = edge_incoherence(cons)

            # switch/cost computed later; store with placeholders
            base_cost = (w_transport * transport) + (w_curv * curv_norm) + (w_incoh * incoh) + (w_unknown * unknown)
            out[a].append(CandEdge(
                a=a, b=b, plane=plane,
                sim=sim, cons=cons,
                transport=transport,
                curv=curv,
                curv_norm=curv_norm,
                incoh=incoh,
                unknown=unknown,
                switch=0,
                cost=base_cost
            ))

    return out

def edge_cost_with_switch(e: CandEdge, last_plane: Optional[str], w_switch: float) -> Tuple[float, int]:
    sw = 0
    if last_plane is not None and e.plane != last_plane:
        sw = 1
    return (e.cost + w_switch * sw), sw

# -------------------------
# Cycle detection + canonicalization
# -------------------------

def smallest_period(seq: List[str]) -> List[str]:
    """
    Reduce to smallest period if seq is repeated pattern.
    """
    L = len(seq)
    if L <= 1:
        return seq[:]
    for p in range(1, L + 1):
        if L % p != 0:
            continue
        ok = True
        for i in range(L):
            if seq[i] != seq[i % p]:
                ok = False
                break
        if ok:
            return seq[:p]
    return seq[:]

def canonical_cycle(seq: List[str]) -> List[str]:
    """
    Canonicalize a cycle up to rotation AND reversal:
    - compute best rotation of seq
    - compute best rotation of reversed seq
    - pick lexicographically smallest of the two
    """
    if not seq:
        return seq

    def best_rotation(tup):
        best = tup
        L = len(tup)
        for r in range(1, L):
            rot = tup[r:] + tup[:r]
            if rot < best:
                best = rot
        return best

    fwd = tuple(seq)
    rev = tuple(reversed(seq))

    best_fwd = best_rotation(fwd)
    best_rev = best_rotation(rev)

    best = best_fwd if best_fwd < best_rev else best_rev
    return list(best)


def detect_cycle_minimal(path_nodes: List[str], next_node: str) -> Optional[List[str]]:
    """
    Implements: when repetition happens at node nb (next_node),
    cycle is path[j:-1] excluding the final repeated nb.

    Here, path_nodes is the current path node list BEFORE adding next_node,
    with current node being path_nodes[-1]. If next_node is already in path_nodes,
    let j = first index of next_node, then the closed cycle nodes are:
        cycle = path_nodes[j:]   (from that first occurrence up to current node)
    (and the edge current->next_node closes it).
    """
    if next_node not in path_nodes:
        return None
    j = path_nodes.index(next_node)
    cycle = path_nodes[j:]  # exclude the repeated next_node itself
    cycle = smallest_period(cycle)
    cycle = canonical_cycle(cycle)
    return cycle

# -------------------------
# Greedy rollout from a prefix
# -------------------------

@dataclass
class RolloutResult:
    status: str  # "cycle" | "deadend" | "truncated"
    steps: int
    attractor: Optional[List[str]]
    path: List[str]

def greedy_rollout(
    candidates_by_node: Dict[str, List[CandEdge]],
    start_node: str,
    start_path_nodes: List[str],
    start_last_plane: Optional[str],
    topk_per_node: int,
    w_switch: float,
    max_steps: int,
    no_backtrack: bool,
) -> RolloutResult:
    path_nodes = start_path_nodes[:]  # includes start_node as last
    last_plane = start_last_plane
    prev_node = path_nodes[-2] if len(path_nodes) >= 2 else None

    for step in range(max_steps):
        a = path_nodes[-1]
        cands = candidates_by_node.get(a, [])
        if not cands:
            return RolloutResult(status="deadend", steps=step, attractor=None, path=path_nodes)

        # sort dynamically by (cost+switch)
        scored = []
        for e in cands:
            if no_backtrack and prev_node is not None and e.b == prev_node:
                continue
            c, sw = edge_cost_with_switch(e, last_plane, w_switch)
            scored.append((c, sw, e))
        if not scored:
            return RolloutResult(status="deadend", steps=step, attractor=None, path=path_nodes)

        scored.sort(key=lambda x: x[0])
        best = scored[0][2]
        next_node = best.b

        cyc = detect_cycle_minimal(path_nodes, next_node)
        if cyc is not None:
            # include the closing node for path display, but attractor excludes repeat
            return RolloutResult(status="cycle", steps=step+1, attractor=cyc, path=path_nodes + [next_node])

        # advance
        prev_node = path_nodes[-1]
        path_nodes.append(next_node)
        last_plane = best.plane

    return RolloutResult(status="truncated", steps=max_steps, attractor=None, path=path_nodes)

# -------------------------
# Branching prefixes up to depth D
# -------------------------

@dataclass
class Prefix:
    nodes: List[str]                # path nodes
    last_plane: Optional[str]       # last plane taken
    prev_node: Optional[str]        # for no-backtrack

def branch_prefixes(
    candidates_by_node: Dict[str, List[CandEdge]],
    start: str,
    depth: int,
    topk_per_node: int,
    w_switch: float,
    no_backtrack: bool,
) -> List[Prefix]:
    """
    Enumerate all prefixes by branching at every step up to depth.
    Includes depth=0 prefix (just the start).
    Stops early on cycle within prefix.
    """
    prefixes = []
    q = deque()
    q.append(Prefix(nodes=[start], last_plane=None, prev_node=None))

    while q:
        pfx = q.popleft()
        prefixes.append(pfx)
        if len(pfx.nodes) - 1 >= depth:
            continue

        a = pfx.nodes[-1]
        cands = candidates_by_node.get(a, [])
        if not cands:
            continue

        scored = []
        for e in cands:
            if no_backtrack and pfx.prev_node is not None and e.b == pfx.prev_node:
                continue
            c, sw = edge_cost_with_switch(e, pfx.last_plane, w_switch)
            scored.append((c, sw, e))
        if not scored:
            continue
        scored.sort(key=lambda x: x[0])

        # branch among top-k
        for _, _, e in scored[:topk_per_node]:
            next_node = e.b
            cyc = detect_cycle_minimal(pfx.nodes, next_node)
            if cyc is not None:
                # we could keep cycle prefixes too, but they don't need further expansion
                prefixes.append(Prefix(nodes=pfx.nodes + [next_node], last_plane=e.plane, prev_node=pfx.nodes[-1]))
                continue
            q.append(Prefix(nodes=pfx.nodes + [next_node], last_plane=e.plane, prev_node=pfx.nodes[-1]))

    return prefixes

def cycle_signature(cyc: List[str]) -> str:
    return "|".join(cyc)

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

    ap.add_argument("--depth", type=int, default=3, help="Branch depth D (prefix length in edges).")
    ap.add_argument("--rollout_steps", type=int, default=30, help="Greedy rollout steps after each prefix.")
    ap.add_argument("--include_rollout", action="store_true", help="If set, each prefix continues with greedy rollout to attractor/deadend.")

    ap.add_argument("--w_transport", type=float, default=1.0)
    ap.add_argument("--w_curv", type=float, default=1.0)
    ap.add_argument("--w_switch", type=float, default=0.25)
    ap.add_argument("--w_incoh", type=float, default=0.5)
    ap.add_argument("--w_unknown", type=float, default=0.75)

    ap.add_argument("--curv_norm", choices=["none", "median", "mean"], default="median")

    ap.add_argument("--no_backtrack", action="store_true")
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

    # build scored candidates (without switch)
    candidates_by_node = build_candidate_edges(
        tm, nodes, planes, args.min_edge_sim, curv_scale,
        args.w_transport, args.w_curv, args.w_switch, args.w_incoh, args.w_unknown
    )

    # sanity: ensure each node list is present
    for n in nodes:
        candidates_by_node.setdefault(n, [])

    # pre-sort base candidates by their base cost (no switch) just for stable debug printing
    if args.debug:
        print("[debug] nodes:", nodes)
        for p in planes:
            print(f"[debug] curv_scale[{p}] = {curv_scale.get(p, None)}")
        nnz = sum(1 for n in nodes if len(candidates_by_node.get(n, [])) > 0)
        print("[debug] total nodes with outgoing edges:", nnz)

    # Run per-start branching
    per_start = {}
    global_attr_counts = Counter()

    for start in nodes:
        prefixes = branch_prefixes(
            candidates_by_node,
            start=start,
            depth=args.depth,
            topk_per_node=args.topk_per_node,
            w_switch=args.w_switch,
            no_backtrack=args.no_backtrack,
        )

        # Evaluate each prefix into an outcome signature
        attr_counts = Counter()
        outcome_counts = Counter()
        sample_paths = []  # keep a few examples

        for pfx in prefixes:
            # If the prefix itself ends with a repeated closure (we appended next_node), detect if last is repetition.
            # We treat it as a cycle if last node appears earlier in prefix nodes[:-1].
            cyc = None
            if len(pfx.nodes) >= 2 and pfx.nodes[-1] in pfx.nodes[:-1]:
                # interpret as closure; compute minimal cycle
                cyc = detect_cycle_minimal(pfx.nodes[:-1], pfx.nodes[-1])

            if cyc is not None:
                sig = cycle_signature(cyc)
                attr_counts[sig] += 1
                outcome_counts["cycle(prefix)"] += 1
                if len(sample_paths) < 5:
                    sample_paths.append({"kind": "cycle(prefix)", "prefix": pfx.nodes, "attractor": cyc})
                continue

            if args.include_rollout:
                rr = greedy_rollout(
                    candidates_by_node,
                    start_node=pfx.nodes[-1],
                    start_path_nodes=pfx.nodes,
                    start_last_plane=pfx.last_plane,
                    topk_per_node=args.topk_per_node,
                    w_switch=args.w_switch,
                    max_steps=args.rollout_steps,
                    no_backtrack=args.no_backtrack,
                )
                if rr.status == "cycle" and rr.attractor:
                    sig = cycle_signature(rr.attractor)
                    attr_counts[sig] += 1
                    outcome_counts["cycle(rollout)"] += 1
                    if len(sample_paths) < 5:
                        sample_paths.append({"kind": "cycle(rollout)", "prefix": pfx.nodes, "rollout_path": rr.path, "attractor": rr.attractor})
                else:
                    outcome_counts[rr.status] += 1
                    if len(sample_paths) < 5:
                        sample_paths.append({"kind": rr.status, "prefix": pfx.nodes, "rollout_path": rr.path})
            else:
                outcome_counts["prefix_only"] += 1

        # pick dominant attractor
        dom_sig, dom_count = (None, 0)
        if attr_counts:
            dom_sig, dom_count = attr_counts.most_common(1)[0]

        per_start[start] = {
            "n_prefixes": len(prefixes),
            "attractor_counts": dict(attr_counts),
            "outcome_counts": dict(outcome_counts),
            "dominant_attractor": dom_sig,
            "dominant_count": dom_count,
            "samples": sample_paths,
        }

        for sig, c in attr_counts.items():
            global_attr_counts[sig] += c

    # Build overall summary
    summary = {
        "phase": "26f",
        "status": "ok",
        "args": vars(args),
        "planes": planes,
        "curv_norm": args.curv_norm,
        "curv_scale": curv_scale,
        "nodes": nodes,
        "global_attractor_counts": dict(global_attr_counts),
        "per_start": per_start,
    }

    out_json = os.path.join(args.outdir, "phase26f_branch_bifurcations.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    # CSV: one row per start with a compact attractor signature list
    csv_lines = ["start,n_prefixes,dominant_attractor,dominant_count,n_unique_attractors,attractor_sig,outcome_counts"]
    for start in nodes:
        ps = per_start[start]
        n_pref = ps["n_prefixes"]
        dom = ps["dominant_attractor"] or ""
        domc = ps["dominant_count"]
        nuniq = len(ps["attractor_counts"])
        # compact sig: attractor1:count|attractor2:count...
        sig_parts = []
        for sig, c in sorted(ps["attractor_counts"].items(), key=lambda x: (-x[1], x[0])):
            sig_parts.append(f"{sig}:{c}")
        sig_str = "|".join(sig_parts)
        # outcome counts compact
        oc_parts = [f"{k}:{v}" for k, v in sorted(ps["outcome_counts"].items())]
        oc_str = "|".join(oc_parts)
        csv_lines.append(f"{start},{n_pref},{dom},{domc},{nuniq},{sig_str},{oc_str}")

    out_csv = os.path.join(args.outdir, "phase26f_branch_bifurcations.csv")
    with open(out_csv, "w", newline="") as f:
        f.write("\n".join(csv_lines) + "\n")

    # Console summary
    print("Phase26f Branch Bifurcations")
    print(f"planes={planes} curv_norm={args.curv_norm} min_edge_sim={args.min_edge_sim}")
    print(f"depth={args.depth} topk_per_node={args.topk_per_node} include_rollout={args.include_rollout} rollout_steps={args.rollout_steps}")
    print(f"no_backtrack={args.no_backtrack}")
    print("curv_scale:", curv_scale)

    # show top global attractors
    if global_attr_counts:
        print("\nTOP GLOBAL ATTRACTORS:")
        for i, (sig, c) in enumerate(global_attr_counts.most_common(10), 1):
            print(f"[{i}] count={c}  cycle={sig}")
    else:
        print("\n(no cycles found)")

    print(f"\n[ok] wrote: {out_json}")
    print(f"[ok] wrote: {out_csv}")

if __name__ == "__main__":
    main()
