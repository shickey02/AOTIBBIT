#!/usr/bin/env python3
# phase22a_edge_axis_clusters_kflex.py
#
# Phase 22a (k-flex): cluster edges from phase21b_edge_axis_field_<plane>.json.
# Works for lt/bo (k=2 pseudo-axis) and bo_lr (k=3 axis).
#
# Output:
#   <outdir>/phase22a_edge_axis_clusters_<plane>.json
#
# Clustering rule:
#   - "coherent" edges: axis_consistency_proxy >= tau
#   - "mixed" edges:    axis_consistency_proxy <  tau
#
# Then we cluster within each bucket using a simple graph-connected-components
# over undirected edge adjacency (a--b), producing cluster_id values.

import os, json, argparse
from collections import defaultdict, deque

def safe_mkdir(p):
    if p and not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def load_json(p):
    with open(p, "r") as f:
        return json.load(f)

def save_json(p, obj):
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)

def undirected_components(edges):
    """
    edges: list of dicts with keys: a,b
    returns: list[list[int]] of indices into edges for each component
    """
    # build adjacency over nodes -> edge indices
    node_to_edges = defaultdict(list)
    for i, e in enumerate(edges):
        node_to_edges[e["a"]].append(i)
        node_to_edges[e["b"]].append(i)

    seen_edge = set()
    comps = []
    for i in range(len(edges)):
        if i in seen_edge:
            continue
        # BFS/DFS over edges via shared nodes
        q = deque([i])
        seen_edge.add(i)
        comp = [i]
        while q:
            ei = q.popleft()
            a = edges[ei]["a"]
            b = edges[ei]["b"]
            for n in (a, b):
                for ej in node_to_edges.get(n, []):
                    if ej not in seen_edge:
                        seen_edge.add(ej)
                        q.append(ej)
                        comp.append(ej)
        comps.append(comp)
    return comps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge_axis_json", required=True, help="phase21b_edge_axis_field_<plane>.json")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--consistency_tau", type=float, default=0.5)
    args = ap.parse_args()

    safe_mkdir(args.outdir)
    data = load_json(args.edge_axis_json)

    plane = data.get("plane", "unknown")
    edge_axis_field = data.get("edge_axis_field", [])
    if not edge_axis_field:
        raise SystemExit("[error] edge_axis_field empty in JSON (did you point at phase21b_edge_axis_field_*.json?)")

    # split coherent vs mixed
    coherent = []
    mixed = []
    for e in edge_axis_field:
        c = float(e.get("axis_consistency_proxy", 0.0))
        if c >= float(args.consistency_tau):
            coherent.append(dict(e))
        else:
            mixed.append(dict(e))

    # cluster each bucket into connected components, assign cluster_id
    clusters = {}
    edge_clusters = {}

    next_cluster_id = 0

    def assign_bucket(bucket_edges, bucket_name):
        nonlocal next_cluster_id
        if not bucket_edges:
            edge_clusters[bucket_name] = []
            return

        comps = undirected_components(bucket_edges)
        # largest components first (optional)
        comps.sort(key=len, reverse=True)

        out_edges = []
        for comp in comps:
            cid = next_cluster_id
            next_cluster_id += 1
            total_curv = 0.0
            mean_cons = 0.0

            for idx in comp:
                ee = bucket_edges[idx]
                ee["cluster_id"] = cid
                out_edges.append(ee)
                total_curv += float(ee.get("curv_abs", 0.0))
                mean_cons += float(ee.get("axis_consistency_proxy", 0.0))

            mean_cons /= max(1, len(comp))

            # top edges for summary
            top = sorted([bucket_edges[i] for i in comp], key=lambda x: float(x.get("curv_abs", 0.0)), reverse=True)[:8]
            clusters[str(cid)] = {
                "cluster_id": int(cid),
                "edge_count": int(len(comp)),
                "total_curv_abs": float(total_curv),
                "mean_consistency": float(mean_cons),
                "bucket": bucket_name,
                "top_edges": top,
            }

        edge_clusters[bucket_name] = sorted(out_edges, key=lambda x: float(x.get("curv_abs", 0.0)), reverse=True)

    assign_bucket(coherent, "cluster_0_coherent")
    assign_bucket(mixed, "cluster_1_mixed")

    out = {
        "phase": "22a",
        "plane": plane,
        "status": "ok",
        "args": {
            "edge_axis_json": args.edge_axis_json,
            "outdir": args.outdir,
            "consistency_tau": float(args.consistency_tau),
        },
        "source": os.path.basename(args.edge_axis_json),
        "global_reference_axis": data.get("global_reference_axis", None),
        "loops_meta": data.get("loops_meta", None),
        "embedding": data.get("embedding", None),
        "clusters": clusters,
        "edge_clusters": edge_clusters,
        "note": f"{plane} clustering by axis_consistency_proxy>=tau, then connected-components within buckets."
    }

    out_json = os.path.join(args.outdir, f"phase22a_edge_axis_clusters_{plane}.json")
    save_json(out_json, out)
    print(f"[ok] wrote {out_json}")

if __name__ == "__main__":
    main()
