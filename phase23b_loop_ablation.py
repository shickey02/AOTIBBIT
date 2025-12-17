#!/usr/bin/env python3
# phase23b_loop_ablation.py
#
# Phase 23b: Loop edge-ablation analysis driven by an edge-axis field.
# - Reads Phase21b edge-axis field JSON (edge_axis_field + loop_summaries).
# - Clusters edges by axis_mean cosine similarity (greedy reps; deterministic).
# - For each loop, counts hits per cluster and computes coverage after dropping
#   edges in an ablated cluster.
#
# Why this exists:
#   Phase21b JSON does NOT contain cluster_id, so we MUST (re)cluster edges here
#   (or load a Phase22a clustering file). This script re-clusters by default.
#
# Example:
#   python phase23b_loop_ablation.py \
#     --edge_axis_json outputs.../phase21b_edge_axis_field_bo_lr.json \
#     --outdir outputs.../phase23b_bo_lr \
#     --cos_thresh 0.92 \
#     --ablate_cluster 1
#
# Outputs:
#   - phase23b_loop_ablation_<plane>_dropC{cluster}.csv
#   - phase23b_loop_ablation_<plane>_dropC{cluster}.json

import os, json, math, argparse, csv
from typing import Dict, List, Tuple, Any
import numpy as np

def safe_mkdir(p: str):
    if p and not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def _norm(v):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n <= 0:
        return v
    return v / n

def cos_sim(a, b):
    a = _norm(a); b = _norm(b)
    return float(np.clip(np.dot(a, b), -1.0, 1.0))

def edge_key(a: str, b: str) -> str:
    return f"{a}||{b}"

def parse_loop_nodes(loop_nodes_field):
    if isinstance(loop_nodes_field, list):
        return [str(x) for x in loop_nodes_field]
    if isinstance(loop_nodes_field, str):
        parts = [p.strip() for p in loop_nodes_field.split("->")]
        return [p for p in parts if p]
    return []

def cycle_edges(nodes: List[str]) -> List[Tuple[str,str]]:
    if not nodes:
        return []
    out = []
    for i in range(len(nodes)):
        a = nodes[i]
        b = nodes[(i+1) % len(nodes)]
        out.append((a, b))
    return out

def greedy_axis_clusters(edges: List[Dict[str,Any]], cos_thresh: float):
    # Sort high curvature first, then hits, then sim (stable/deterministic)
    edges_sorted = sorted(
        edges,
        key=lambda e: (float(e.get("curv_abs", 0.0)),
                       float(e.get("hits", 0.0)),
                       float(e.get("sim", 0.0))),
        reverse=True
    )

    clusters: List[Dict[str,Any]] = []
    assign: Dict[str,int] = {}

    for e in edges_sorted:
        a = str(e.get("a",""))
        b = str(e.get("b",""))
        if not a or not b:
            continue

        axis = e.get("axis_mean", None)
        if axis is None:
            continue
        axis = _norm(axis)

        k = edge_key(a,b)
        placed = False

        for cid, C in enumerate(clusters):
            rep = np.asarray(C["rep_axis"], dtype=np.float64)
            c = abs(cos_sim(axis, rep))  # axis direction ambiguous -> abs cosine
            if c >= cos_thresh:
                C["edges"].append(e)
                C["sum_curv_abs"] += float(e.get("curv_abs", 0.0))
                C["sum_hits"] += int(e.get("hits", 0))
                C["mean_axis_accum"] += axis
                assign[k] = cid
                placed = True
                break

        if not placed:
            clusters.append({
                "id": len(clusters),
                "rep_axis": axis.copy(),
                "sum_curv_abs": float(e.get("curv_abs", 0.0)),
                "sum_hits": int(e.get("hits", 0)),
                "mean_axis_accum": axis.copy(),
                "edges": [e],
            })
            assign[k] = len(clusters) - 1

    # finalize mean_axis / rep_axis into JSON-safe lists
    for C in clusters:
        C["mean_axis"] = _norm(C["mean_axis_accum"]).tolist()
        del C["mean_axis_accum"]
        C["rep_axis"] = np.asarray(C["rep_axis"], dtype=np.float64).tolist()

    return clusters, assign

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge_axis_json", required=True, help="Phase21b edge-axis-field JSON on disk.")
    ap.add_argument("--outdir", required=True, help="Output directory.")
    ap.add_argument("--cos_thresh", type=float, default=0.92, help="Axis clustering threshold (abs cosine). Use 0.92 to match Phase22a.")
    ap.add_argument("--ablate_cluster", type=int, default=1, help="Cluster id to DROP.")
    args = ap.parse_args()

    safe_mkdir(args.outdir)

    with open(args.edge_axis_json, "r") as f:
        data = json.load(f)

    plane = data.get("plane", "unknown")
    edge_axis_field = data.get("edge_axis_field", [])
    loop_summaries = data.get("loop_summaries", [])

    if not edge_axis_field:
        raise SystemExit("[error] edge_axis_field empty in JSON (did you point at phase21b_edge_axis_field_*.json?)")
    if not loop_summaries:
        raise SystemExit("[error] loop_summaries empty in JSON (Phase21b should include it).")

    clusters, assign = greedy_axis_clusters(edge_axis_field, float(args.cos_thresh))
    if args.ablate_cluster < 0 or args.ablate_cluster >= len(clusters):
        raise SystemExit(f"[error] ablate_cluster={args.ablate_cluster} out of range (clusters={len(clusters)}). "
                         f"Lower --cos_thresh if you only got 1 cluster.")

    # sim lookup
    sim_lookup: Dict[str,float] = {}
    for e in edge_axis_field:
        sim_lookup[edge_key(str(e["a"]), str(e["b"]))] = float(e.get("sim", float("nan")))

    rows = []
    details = []

    for li, L in enumerate(loop_summaries, start=1):
        nodes = parse_loop_nodes(L.get("loop", []))
        ecycle = cycle_edges(nodes)

        hits = {}
        unknown = 0
        kept_edges = []
        dropped_edges = []
        sims_kept = []

        for (a,b) in ecycle:
            k = edge_key(a,b)
            cid = assign.get(k, None)

            if cid is None:
                unknown += 1
                kept_edges.append((a,b,None))
                s = sim_lookup.get(k, float("nan"))
                if not (isinstance(s, float) and math.isnan(s)):
                    sims_kept.append(float(s))
                continue

            hits[cid] = hits.get(cid, 0) + 1

            if cid == args.ablate_cluster:
                dropped_edges.append((a,b,cid))
            else:
                kept_edges.append((a,b,cid))
                s = sim_lookup.get(k, float("nan"))
                if not (isinstance(s, float) and math.isnan(s)):
                    sims_kept.append(float(s))

        total_edges = len(ecycle)
        kept_n = len(kept_edges)
        dropped_n = len(dropped_edges)
        coverage = (kept_n / total_edges) if total_edges > 0 else float("nan")

        kept_sim_min = float(np.min(sims_kept)) if sims_kept else float("nan")
        kept_sim_mean = float(np.mean(sims_kept)) if sims_kept else float("nan")

        loop_nodes_str = " -> ".join(nodes + ([nodes[0]] if nodes else []))
        row = {
            "loop_index": li,
            "loop_nodes": loop_nodes_str,
            "angle_deg": float(L.get("angle_deg", float("nan"))),
            "rank_score": float(L.get("rank_score", float("nan"))),
            "edge_sim_min": float(L.get("edge_sim_min", float("nan"))),
            "edge_sim_mean": float(L.get("edge_sim_mean", float("nan"))),
            "unknown_edges": int(unknown),
        }
        for cid in range(len(clusters)):
            row[f"hits_cluster_{cid}"] = int(hits.get(cid, 0))

        row.update({
            f"ablate_{args.ablate_cluster}_kept_edges": int(kept_n),
            f"ablate_{args.ablate_cluster}_dropped_edges": int(dropped_n),
            f"ablate_{args.ablate_cluster}_kept_sim_min": kept_sim_min,
            f"ablate_{args.ablate_cluster}_kept_sim_mean": kept_sim_mean,
            f"ablate_{args.ablate_cluster}_coverage_ratio": float(coverage),
        })

        rows.append(row)

        details.append({
            "loop_index": li,
            "loop": nodes,
            "cycle_edges": [{"a":a,"b":b,"cluster_id":assign.get(edge_key(a,b), None)} for (a,b) in ecycle],
            "kept_edges": [{"a":a,"b":b,"cluster_id":cid} for (a,b,cid) in kept_edges],
            "dropped_edges": [{"a":a,"b":b,"cluster_id":cid} for (a,b,cid) in dropped_edges],
        })

    out_base = f"phase23b_loop_ablation_{plane}_dropC{args.ablate_cluster}"
    out_csv = os.path.join(args.outdir, out_base + ".csv")
    out_json = os.path.join(args.outdir, out_base + ".json")

    core = ["loop_index","loop_nodes","angle_deg","rank_score","edge_sim_min","edge_sim_mean","unknown_edges"]
    hits_cols = sorted([k for k in rows[0].keys() if k.startswith("hits_cluster_")],
                       key=lambda s: int(s.split("_")[-1]))
    ab_cols = [k for k in rows[0].keys() if k.startswith(f"ablate_{args.ablate_cluster}_")]
    header = core + hits_cols + ab_cols

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})

    out = {
        "phase": "23b",
        "plane": plane,
        "source": args.edge_axis_json,
        "args": {"cos_thresh": float(args.cos_thresh), "ablate_cluster": int(args.ablate_cluster)},
        "counts": {"clusters": len(clusters), "loops": len(rows), "edge_axis_field": len(edge_axis_field)},
        "clusters": clusters,
        "rows": rows,
        "details": details,
        "paths": {"csv": out_csv, "json": out_json},
        "note": "Cluster ids are computed from phase21b axis_mean with abs-cos threshold; use --cos_thresh 0.92 to match phase22a."
    }
    with open(out_json, "w") as g:
        json.dump(out, g, indent=2)

    print(f"[ok] plane={plane} clusters={len(clusters)} loops={len(rows)} cos_thresh={args.cos_thresh}")
    print(f"[saved] {out_csv}")
    print(f"[saved] {out_json}")

if __name__ == "__main__":
    main()
