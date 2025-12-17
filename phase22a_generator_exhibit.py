#!/usr/bin/env python3
# phase22a_generator_exhibit.py
#
# Phase 22a: Generator extraction + minimal exhibit plot from Phase21b output.
#
# Reads Phase21b JSON (edge axis field + embedding coords + loop summaries),
# clusters edges into "generator families" by holonomy axis direction, and produces:
#   - phase22a_generators_<plane>.json
#   - phase22a_generators_<plane>.png
#   - phase22a_loop_attribution_<plane>.csv
#
# Usage:
#   python phase22a_generator_exhibit.py \
#     --in_json outputs_edges_relternary256_phase15/phase21b_bo_lr/phase21b_edge_axis_field_bo_lr.json \
#     --outdir outputs_edges_relternary256_phase15/phase22a_bo_lr \
#     --top_edges 10 --cos_thresh 0.92 --max_edges_draw 10
#
# Notes:
# - Clustering uses abs(cosine(axis_i, axis_j)) >= cos_thresh (sign-invariant).
# - The plot uses the Phase21b embedding coords (3D) and draws top edges by curvature.

import os, json, math, argparse, csv
import numpy as np

# ---------------------------- utils ----------------------------

def safe_mkdir(p):
    if p and not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def norm(v):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v, 0.0
    return v / n, n

def cos_sim(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def get_embedding_coords(data):
    emb = data.get("embedding", {}) or {}
    coords = (emb.get("coords", {}) or {})
    # coords: name -> [x,y,z]
    return coords

def get_edge_axis_field(data):
    return data.get("edge_axis_field", []) or []

def get_loop_summaries(data):
    return data.get("loop_summaries", []) or []

def build_loop_edges(loop_nodes):
    """
    loop_nodes is like [A,B,C] representing A->B->C->A
    return list of directed edges (a,b) in order.
    """
    if not loop_nodes or len(loop_nodes) < 2:
        return []
    nodes = list(loop_nodes)
    closed = nodes + [nodes[0]]
    out = []
    for i in range(len(closed) - 1):
        out.append((closed[i], closed[i+1]))
    return out

# ---------------------------- clustering ----------------------------

def cluster_edges_by_axis(edges, cos_thresh=0.92):
    """
    Greedy clustering by sign-invariant cosine similarity:
      abs(dot(axis_i, axis_j)) >= cos_thresh -> same cluster
    Returns: clusters list, each cluster dict:
      {
        "id": int,
        "rep_axis": [..] unit,
        "edges": [edge_dicts...],
        "sum_curv_abs": float,
        "mean_axis": [..] unit (sign-aligned to rep)
      }
    """
    # keep only edges with valid axis_mean
    cleaned = []
    for e in edges:
        axis = e.get("axis_mean", None)
        if axis is None:
            continue
        axis_u, axis_n = norm(axis)
        if axis_n < 1e-12:
            continue
        ee = dict(e)
        ee["_axis_u"] = axis_u
        cleaned.append(ee)

    # sort by importance (curv_abs desc, then hits desc)
    cleaned.sort(key=lambda x: (float(x.get("curv_abs", 0.0)), float(x.get("hits", 0))), reverse=True)

    clusters = []

    def add_to_cluster(cl, ee):
        # sign-align ee axis to cl rep axis
        rep = cl["rep_axis"]
        ax = ee["_axis_u"]
        if np.dot(rep, ax) < 0:
            ax = -ax
        ee["_axis_u_aligned"] = ax
        cl["edges"].append(ee)
        cl["sum_curv_abs"] += float(ee.get("curv_abs", 0.0))

    for ee in cleaned:
        ax = ee["_axis_u"]
        placed = False
        for cl in clusters:
            rep = cl["rep_axis"]
            if abs(float(np.dot(rep, ax))) >= cos_thresh:
                add_to_cluster(cl, ee)
                placed = True
                break
        if not placed:
            cl = {
                "id": len(clusters),
                "rep_axis": ax.copy(),
                "edges": [],
                "sum_curv_abs": 0.0,
            }
            add_to_cluster(cl, ee)
            clusters.append(cl)

    # finalize mean axis per cluster
    for cl in clusters:
        if not cl["edges"]:
            cl["mean_axis"] = cl["rep_axis"].tolist()
            continue
        A = np.stack([e["_axis_u_aligned"] for e in cl["edges"]], axis=0)
        mu = np.mean(A, axis=0)
        mu_u, _ = norm(mu)
        cl["mean_axis"] = mu_u.tolist()
        cl["rep_axis"] = cl["rep_axis"].tolist()

    # sort clusters by total curvature load
    clusters.sort(key=lambda c: float(c["sum_curv_abs"]), reverse=True)
    # re-id in sorted order
    for i, cl in enumerate(clusters):
        cl["id"] = i

    return clusters

# ---------------------------- plotting ----------------------------

def make_exhibit_plot(coords, edges_to_draw, out_png, annotate=True):
    """
    coords: dict name -> [x,y,z]
    edges_to_draw: list of edge dicts with fields a,b,curv_abs
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    names = list(coords.keys())
    if len(names) < 2:
        raise SystemExit("[error] Not enough nodes in embedding coords to plot.")

    X = np.array([coords[n] for n in names], dtype=np.float64)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(X[:,0], X[:,1], X[:,2], s=80, depthshade=True)

    if annotate:
        for i, n in enumerate(names):
            ax.text(X[i,0], X[i,1], X[i,2], " " + str(n), fontsize=9)

    # draw directed edges (arrows as quivers)
    # linewidth scales by curvature magnitude
    curvs = [float(e.get("curv_abs", 0.0)) for e in edges_to_draw]
    cmax = max(curvs) if curvs else 1.0
    cmax = max(cmax, 1e-9)

    for e in edges_to_draw:
        a = e.get("a"); b = e.get("b")
        if a not in coords or b not in coords:
            continue
        pa = np.array(coords[a], dtype=np.float64)
        pb = np.array(coords[b], dtype=np.float64)
        d = pb - pa
        mag = float(e.get("curv_abs", 0.0))
        lw = 0.5 + 3.5 * (mag / cmax)

        ax.quiver(
            pa[0], pa[1], pa[2],
            d[0], d[1], d[2],
            arrow_length_ratio=0.15,
            linewidth=lw,
            alpha=0.85,
            normalize=False,
        )

        if annotate:
            mid = (pa + pb) / 2.0
            ax.text(mid[0], mid[1], mid[2], f" {mag:.1f}", fontsize=8)

    ax.set_title("Phase22a: Holonomy Generators Exhibit (embedding + top curvature edges)")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

# ---------------------------- loop attribution ----------------------------

def write_loop_attribution_csv(loop_summaries, edge_to_cluster, out_csv):
    """
    loop_summaries: list from Phase21b (each has "loop", "rank_score", "angle_deg", ...)
    edge_to_cluster: dict (a,b)->cluster_id
    Writes rows:
      loop_index, loop_nodes, rank_score, angle_deg, edge_sim_min, edge_sim_mean,
      clusters_used (semicolon list), cluster_hits (counts per cluster)
    """
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "loop_index",
            "loop_nodes",
            "rank_score",
            "angle_deg",
            "edge_sim_min",
            "edge_sim_mean",
            "clusters_used",
            "cluster_hit_counts"
        ])

        for idx, L in enumerate(loop_summaries, 1):
            nodes = L.get("loop", [])
            edges = build_loop_edges(nodes)
            hit = {}
            for (a,b) in edges:
                cid = edge_to_cluster.get((a,b), None)
                if cid is None:
                    continue
                hit[cid] = hit.get(cid, 0) + 1

            clusters_used = ";".join([str(k) for k in sorted(hit.keys())])
            cluster_counts = ";".join([f"{k}:{hit[k]}" for k in sorted(hit.keys())])

            w.writerow([
                idx,
                " -> ".join(nodes + [nodes[0]]) if nodes else "",
                float(L.get("rank_score", float("nan"))),
                float(L.get("angle_deg", float("nan"))),
                float(L.get("edge_sim_min", float("nan"))),
                float(L.get("edge_sim_mean", float("nan"))),
                clusters_used,
                cluster_counts
            ])

# ---------------------------- main ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True, help="Phase21b JSON path (edge_axis_field + embedding + loop_summaries).")
    ap.add_argument("--outdir", required=True, help="Output directory for Phase22a artifacts.")
    ap.add_argument("--top_edges", type=int, default=10, help="How many strongest edges (by curv_abs) to include in report/plot.")
    ap.add_argument("--max_edges_draw", type=int, default=10, help="How many edges to actually draw on the plot.")
    ap.add_argument("--cos_thresh", type=float, default=0.92, help="Axis clustering threshold on abs(cosine).")
    ap.add_argument("--annotate", action="store_true", help="Annotate nodes/edge magnitudes on plot.")
    args = ap.parse_args()

    safe_mkdir(args.outdir)

    data = load_json(args.in_json)
    plane = data.get("plane", "unknown")

    coords = get_embedding_coords(data)
    edge_axis_field = get_edge_axis_field(data)
    loop_summaries = get_loop_summaries(data)

    if not coords or len(coords) < 3:
        raise SystemExit(f"[error] Need >=3 nodes in embedding coords to build exhibit. found={len(coords) if coords else 0}")

    if not edge_axis_field:
        raise SystemExit("[error] No edge_axis_field found in Phase21b JSON.")

    # Sort edges by curvature magnitude
    edges_sorted = sorted(edge_axis_field, key=lambda e: float(e.get("curv_abs", 0.0)), reverse=True)
    top_edges = edges_sorted[:max(1, args.top_edges)]
    draw_edges = top_edges[:max(1, args.max_edges_draw)]

    # Cluster edges into generator families
    clusters = cluster_edges_by_axis(edge_axis_field, cos_thresh=args.cos_thresh)

    # Build edge->cluster map (directed)
    edge_to_cluster = {}
    for cl in clusters:
        cid = int(cl["id"])
        for e in cl.get("edges", []):
            a = e.get("a"); b = e.get("b")
            if a is None or b is None:
                continue
            edge_to_cluster[(a,b)] = cid

    # Summarize clusters (trim bulky internals for JSON)
    clusters_out = []
    for cl in clusters:
        # edges sorted by curv_abs within cluster
        ed = list(cl.get("edges", []))
        ed.sort(key=lambda e: float(e.get("curv_abs", 0.0)), reverse=True)
        edges_compact = []
        for e in ed:
            edges_compact.append({
                "a": e.get("a"),
                "b": e.get("b"),
                "sim": float(e.get("sim", float("nan"))),
                "hits": int(e.get("hits", 0)),
                "curv_abs": float(e.get("curv_abs", 0.0)),
                "curv_signed": float(e.get("curv_signed", 0.0)),
                "axis_consistency_proxy": float(e.get("axis_consistency_proxy", float("nan"))),
                "axis_mean": e.get("axis_mean"),
            })

        clusters_out.append({
            "id": int(cl["id"]),
            "sum_curv_abs": float(cl.get("sum_curv_abs", 0.0)),
            "rep_axis": cl.get("rep_axis"),
            "mean_axis": cl.get("mean_axis"),
            "edges": edges_compact,
        })

    # Write JSON report
    out_json = os.path.join(args.outdir, f"phase22a_generators_{plane}.json")
    out = {
        "phase": "22a",
        "plane": plane,
        "source": args.in_json,
        "args": {
            "top_edges": args.top_edges,
            "max_edges_draw": args.max_edges_draw,
            "cos_thresh": args.cos_thresh,
            "annotate": bool(args.annotate),
        },
        "counts": {
            "nodes": int(len(coords)),
            "edge_axis_field": int(len(edge_axis_field)),
            "loop_summaries": int(len(loop_summaries)),
            "clusters": int(len(clusters_out)),
        },
        "top_edges_by_curvature": [
            {
                "a": e.get("a"),
                "b": e.get("b"),
                "sim": float(e.get("sim", float("nan"))),
                "hits": int(e.get("hits", 0)),
                "curv_abs": float(e.get("curv_abs", 0.0)),
                "curv_signed": float(e.get("curv_signed", 0.0)),
                "axis_consistency_proxy": float(e.get("axis_consistency_proxy", float("nan"))),
                "cluster_id": int(edge_to_cluster.get((e.get("a"), e.get("b")), -1)),
            } for e in top_edges
        ],
        "clusters": clusters_out,
    }
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)

    # Write exhibit plot
    out_png = os.path.join(args.outdir, f"phase22a_generators_{plane}.png")
    make_exhibit_plot(coords, draw_edges, out_png, annotate=bool(args.annotate))

    # Loop attribution CSV
    out_csv = os.path.join(args.outdir, f"phase22a_loop_attribution_{plane}.csv")
    write_loop_attribution_csv(loop_summaries, edge_to_cluster, out_csv)

    # Console summary
    print(f"[phase22a] plane={plane}")
    print(f"  nodes={len(coords)}  edges={len(edge_axis_field)}  loops={len(loop_summaries)}  clusters={len(clusters_out)}")
    if clusters_out:
        print("  top clusters by sum_curv_abs:")
        for cl in clusters_out[:min(5, len(clusters_out))]:
            print(f"    cluster {cl['id']}: sum_curv_abs={cl['sum_curv_abs']:.3f}  edges={len(cl['edges'])}")
    print(f"[saved] {out_json}")
    print(f"[saved] {out_png}")
    print(f"[saved] {out_csv}")

if __name__ == "__main__":
    main()
