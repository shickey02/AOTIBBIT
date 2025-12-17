#!/usr/bin/env python3
# phase22a_edge_axis_clusters_lt.py
#
# Phase 22a (lt): cluster edge axis field into coherent vs mixed edges.
#
# Input:
#   phase21b_edge_axis_field_lt.json
#
# Output:
#   phase22a_edge_axis_clusters_lt.json
#
# Clusters:
#   cluster 0 = coherent orientation (axis_consistency >= tau)
#   cluster 1 = mixed / boundary edges (axis_consistency < tau)

import json, argparse, os

def safe_mkdir(p):
    if p and not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge_axis_json", required=True,
                    help="phase21b_edge_axis_field_lt.json")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--consistency_tau", type=float, default=0.5,
                    help="Threshold for coherent vs mixed edges")
    args = ap.parse_args()

    safe_mkdir(args.outdir)

    with open(args.edge_axis_json, "r") as f:
        data = json.load(f)

    assert data["plane"] == "lt", "[error] This script is for lt only."

    edges = data["edge_axis_field"]
    loops_meta = data.get("loops_meta", {})
    embedding = data.get("embedding", {})
    global_ref = data.get("global_reference_axis", [0,0,1])

    clusters = {
        0: [],  # coherent
        1: []   # mixed
    }

    for e in edges:
        c = e.get("axis_consistency_proxy", 0.0)
        if c >= args.consistency_tau:
            cid = 0
        else:
            cid = 1

        e2 = dict(e)
        e2["cluster_id"] = cid
        clusters[cid].append(e2)

    # summaries
    cluster_summaries = {}
    for cid, elist in clusters.items():
        cluster_summaries[str(cid)] = {
            "cluster_id": cid,
            "edge_count": len(elist),
            "total_curv_abs": sum(e["curv_abs"] for e in elist),
            "mean_consistency": (
                sum(e["axis_consistency_proxy"] for e in elist) / len(elist)
                if elist else 0.0
            ),
            "top_edges": sorted(
                elist,
                key=lambda e: e["curv_abs"],
                reverse=True
            )[:5]
        }

    out = {
        "phase": "22a",
        "plane": "lt",
        "status": "ok",
        "args": vars(args),
        "source": os.path.basename(args.edge_axis_json),
        "global_reference_axis": global_ref,
        "loops_meta": loops_meta,
        "embedding": embedding,
        "clusters": cluster_summaries,
        "edge_clusters": {
            "cluster_0_coherent": clusters[0],
            "cluster_1_mixed": clusters[1]
        },
        "note": (
            "lt clustering by axis_consistency_proxy. "
            "Cluster 0 = coherent orientation transport. "
            "Cluster 1 = mixed polarity / boundary shuttling."
        )
    }

    out_json = os.path.join(
        args.outdir,
        "phase22a_edge_axis_clusters_lt.json"
    )

    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)

    print("[phase22a lt] clustering complete")
    for cid in sorted(cluster_summaries):
        s = cluster_summaries[cid]
        print(
            f"  cluster {cid}: "
            f"edges={s['edge_count']} "
            f"total_curv={s['total_curv_abs']:.3f} "
            f"mean_consistency={s['mean_consistency']:.3f}"
        )
    print(f"[saved] {out_json}")

if __name__ == "__main__":
    main()
