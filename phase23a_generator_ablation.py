#!/usr/bin/env python3
# phase23a_generator_ablation.py
#
# Phase 23a: Generator ablation study using Phase21b + Phase22a.
#
# Inputs:
#   - Phase21b JSON (embedding + loop_summaries + edge_axis_field)
#   - Phase22a JSON (clusters + edge->cluster mapping implicitly via clusters[*].edges)
#
# Outputs:
#   - phase23a_ablation_<plane>.json
#   - phase23a_ablation_<plane>.csv
#
# What it does:
#   - Reconstruct edge->cluster_id mapping from Phase22a clusters
#   - For each loop, compute cluster-hit composition (#edges in each cluster)
#   - Provide ablation "scores" for each loop:
#       * remove cluster k edges => remaining_edges_count, remaining_min_sim, remaining_mean_sim
#       * (optionally) estimate "expected angle retention" by edge coverage ratio
#
# Notes:
# - This does NOT recompute holonomy matrices (that would require transport composition),
#   but it gives the *graph/loop-level* causal attribution you need to justify next step.
# - Phase23b can do full R_loop recompute under ablation if you want (more expensive).

import os, json, argparse, csv
import numpy as np

def safe_mkdir(p):
    if p and not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def build_loop_edges(loop_nodes):
    if not loop_nodes or len(loop_nodes) < 2:
        return []
    nodes = list(loop_nodes)
    closed = nodes + [nodes[0]]
    return [(closed[i], closed[i+1]) for i in range(len(closed)-1)]

def build_edge_cluster_map(phase22a):
    m = {}
    clusters = phase22a.get("clusters", []) or []
    for cl in clusters:
        cid = int(cl.get("id", -1))
        for e in cl.get("edges", []) or []:
            a = e.get("a"); b = e.get("b")
            if a is None or b is None:
                continue
            m[(a,b)] = cid
    return m

def build_edge_sim_map(phase21b):
    m = {}
    for e in phase21b.get("edge_axis_field", []) or []:
        a = e.get("a"); b = e.get("b")
        if a is None or b is None:
            continue
        m[(a,b)] = float(e.get("sim", float("nan")))
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase21b", required=True, help="Phase21b JSON from disk.")
    ap.add_argument("--phase22a", required=True, help="Phase22a generators JSON from disk.")
    ap.add_argument("--outdir", required=True, help="Output directory.")
    ap.add_argument("--ablate", default="1", help="Comma list of cluster ids to ablate, e.g. '1' or '0,1'.")
    args = ap.parse_args()

    safe_mkdir(args.outdir)

    p21 = load_json(args.phase21b)
    p22 = load_json(args.phase22a)

    plane = p21.get("plane", p22.get("plane", "unknown"))
    loops = p21.get("loop_summaries", []) or []
    edge_to_cluster = build_edge_cluster_map(p22)
    edge_to_sim = build_edge_sim_map(p21)

    clusters = p22.get("clusters", []) or []
    cluster_ids = sorted([int(c.get("id", -1)) for c in clusters if "id" in c and int(c.get("id",-1)) >= 0])
    ablate_ids = []
    for tok in str(args.ablate).split(","):
        tok = tok.strip()
        if tok == "":
            continue
        ablate_ids.append(int(tok))
    ablate_ids = sorted(list(set(ablate_ids)))

    # analyze loops
    loop_rows = []
    for i, L in enumerate(loops, 1):
        nodes = L.get("loop", [])
        edges = build_loop_edges(nodes)

        # cluster hit counts
        hit = {cid: 0 for cid in cluster_ids}
        unknown = 0
        sims = []
        for (a,b) in edges:
            cid = edge_to_cluster.get((a,b), None)
            if cid is None:
                unknown += 1
            else:
                hit[cid] = hit.get(cid, 0) + 1
            s = edge_to_sim.get((a,b), float("nan"))
            if np.isfinite(s):
                sims.append(float(s))

        # ablated versions
        ablated = {}
        for cid in ablate_ids:
            kept_edges = [(a,b) for (a,b) in edges if edge_to_cluster.get((a,b), None) != cid]
            kept_sims = []
            for (a,b) in kept_edges:
                s = edge_to_sim.get((a,b), float("nan"))
                if np.isfinite(s):
                    kept_sims.append(float(s))

            ablated[str(cid)] = {
                "kept_edges": len(kept_edges),
                "dropped_edges": len(edges) - len(kept_edges),
                "kept_sim_min": float(np.min(kept_sims)) if kept_sims else float("nan"),
                "kept_sim_mean": float(np.mean(kept_sims)) if kept_sims else float("nan"),
                "coverage_ratio": (len(kept_edges) / max(1, len(edges))),
            }

        row = {
            "loop_index": i,
            "loop": nodes,
            "angle_deg": float(L.get("angle_deg", float("nan"))),
            "rank_score": float(L.get("rank_score", float("nan"))),
            "edge_sim_min": float(L.get("edge_sim_min", float("nan"))),
            "edge_sim_mean": float(L.get("edge_sim_mean", float("nan"))),
            "cluster_hits": hit,
            "unknown_edges": int(unknown),
            "ablation": ablated,
        }
        loop_rows.append(row)

    out = {
        "phase": "23a",
        "plane": plane,
        "inputs": {
            "phase21b": args.phase21b,
            "phase22a": args.phase22a,
        },
        "clusters": {
            "cluster_ids": cluster_ids,
            "ablate_ids": ablate_ids
        },
        "loops": loop_rows,
    }

    out_json = os.path.join(args.outdir, f"phase23a_ablation_{plane}.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)

    out_csv = os.path.join(args.outdir, f"phase23a_ablation_{plane}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = [
            "loop_index","loop_nodes","angle_deg","rank_score","edge_sim_min","edge_sim_mean",
            "unknown_edges"
        ]
        for cid in cluster_ids:
            header.append(f"hits_cluster_{cid}")
        for cid in ablate_ids:
            header += [
                f"ablate_{cid}_kept_edges",
                f"ablate_{cid}_dropped_edges",
                f"ablate_{cid}_kept_sim_min",
                f"ablate_{cid}_kept_sim_mean",
                f"ablate_{cid}_coverage_ratio",
            ]
        w.writerow(header)

        for r in loop_rows:
            nodes = r["loop"]
            base = [
                r["loop_index"],
                " -> ".join(nodes + [nodes[0]]) if nodes else "",
                r["angle_deg"],
                r["rank_score"],
                r["edge_sim_min"],
                r["edge_sim_mean"],
                r["unknown_edges"],
            ]
            for cid in cluster_ids:
                base.append(int(r["cluster_hits"].get(cid, 0)))
            for cid in ablate_ids:
                ab = r["ablation"].get(str(cid), {})
                base += [
                    int(ab.get("kept_edges", 0)),
                    int(ab.get("dropped_edges", 0)),
                    float(ab.get("kept_sim_min", float("nan"))),
                    float(ab.get("kept_sim_mean", float("nan"))),
                    float(ab.get("coverage_ratio", float("nan"))),
                ]
            w.writerow(base)

    print(f"[phase23a] plane={plane}")
    print(f"  loops={len(loop_rows)} clusters={len(cluster_ids)} ablate={ablate_ids}")
    print(f"[saved] {out_json}")
    print(f"[saved] {out_csv}")

if __name__ == "__main__":
    main()
