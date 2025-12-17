#!/usr/bin/env python3
# phase18b_reasoning_shape_3d.py
#
# Phase 18B: Reasoning-shape visualization in 3D.
# - Finds anchor center vectors (best effort)
# - Also supports bases/frames per anchor (cache["bases"][anchor][...]=vec)
# - Projects everything into a single shared PCA(3) space
# - Plots:
#     (A) anchor centers as points
#     (B) optional basis vectors per anchor as "spikes"
#     (C) optional transport edges by similarity
#     (D) optional holonomy loops (from phase16a JSON; legacy or threshold_results)
#
# Example:
#   python phase18b_reasoning_shape_3d.py ^
#     --cache outputs_edges_relternary256_phase15/phase15f_transport_cache.json ^
#     --outdir outputs_edges_relternary256_phase15/phase18b_bo_lr ^
#     --plane bo_lr ^
#     --loops_json outputs_edges_relternary256_phase15/phase16a_holonomy_bo_lr.json ^
#     --threshold 0.35 --min_sim 0.35 --top_loops 10 ^
#     --draw_loops --draw_transport --min_edge_sim 0.40 --max_edges 300 ^
#     --draw_bases --max_bases_per_anchor 8

import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA


# ---------------------------- utils ----------------------------

def safe_mkdir(p):
    if p and not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def _is_num(x):
    return isinstance(x, (int, float)) and np.isfinite(x)

def _is_vec_list(v, min_dim=2):
    return isinstance(v, list) and len(v) >= min_dim and all(_is_num(x) for x in v)

def _extract_vec(obj):
    # common vector fields
    if _is_vec_list(obj):
        return obj, "(raw_list)"
    if isinstance(obj, dict):
        for fld in ["t","z","latent","vec","embedding","emb","code","mu","mean","center"]:
            if fld in obj and _is_vec_list(obj[fld]):
                return obj[fld], fld
    return None, None

def group_label(name: str):
    s = name.lower()
    for pref in ["between", "overlap", "left", "right", "above", "below", "inside", "contain", "touch", "disjoint"]:
        if s.startswith(pref):
            return pref
    return s.split("_", 1)[0] if "_" in s else s

def load_loops_any_format(path, threshold=None, min_sim=None, top_loops=None):
    if not path:
        return []

    data = json.load(open(path, "r"))

    loops = []
    if isinstance(data, dict) and "threshold_results" in data and isinstance(data["threshold_results"], dict):
        tr = data["threshold_results"]

        def _as_float(x):
            try: return float(x)
            except: return None

        chosen_key = None
        if threshold is not None:
            k1 = f"{threshold:.2f}"
            if k1 in tr: chosen_key = k1
            elif str(threshold) in tr: chosen_key = str(threshold)

        if chosen_key is None:
            numeric = [(k, _as_float(k)) for k in tr.keys()]
            numeric = [(k, v) for (k, v) in numeric if v is not None]
            if numeric:
                numeric.sort(key=lambda kv: kv[1], reverse=True)
                chosen_key = numeric[0][0]
            else:
                chosen_key = list(tr.keys())[0]

        bucket = tr.get(chosen_key, {})
        loops = bucket.get("top", []) if isinstance(bucket, dict) else (bucket if isinstance(bucket, list) else [])
    else:
        loops = data.get("top", data.get("results", []))

    if min_sim is not None:
        loops = [L for L in loops if float(L.get("edge_sim_min", -1e9)) >= float(min_sim)]

    if top_loops is not None and top_loops > 0:
        loops = loops[:top_loops]

    return loops

def transport_edges(cache, plane, min_edge_sim=0.40, max_edges=300):
    """
    Extract transport edges from cache["transport_maps"] with similarity field.
    Returns list of (a,b,sim).
    """
    maps = cache.get("transport_maps", {})
    edges = []
    for k, planes in maps.items():
        if "__to__" not in k:
            continue
        a, b = k.split("__to__", 1)
        if not isinstance(planes, dict) or plane not in planes:
            continue
        entry = planes[plane]
        if not isinstance(entry, dict):
            continue
        sim = entry.get("similarity", None)
        if sim is None:
            continue
        try:
            sim = float(sim)
        except Exception:
            continue
        if sim >= float(min_edge_sim):
            edges.append((a, b, sim))
    edges.sort(key=lambda t: t[2], reverse=True)
    return edges[:max_edges]

def discover_anchor_centers(cache):
    """
    Best-effort: attempt to pull one vector per anchor from cache["anchors"][name][field].
    Returns dict: name -> vec
    """
    out = {}
    anchors = cache.get("anchors", {})
    if not isinstance(anchors, dict):
        return out
    for name, obj in anchors.items():
        vec, fld = _extract_vec(obj)
        if vec is not None:
            out[name] = vec
    return out

def discover_bases(cache):
    """
    Discover cache["bases"] structure:
      bases[anchor] can be:
        - dict: {basis_name: vec_list, ...}
        - list: [vec_list, vec_list, ...]
    Returns dict: anchor -> list_of_vecs
    """
    out = {}
    bases = cache.get("bases", {})
    if not isinstance(bases, dict):
        return out
    for a, obj in bases.items():
        vecs = []
        if isinstance(obj, dict):
            for _, v in obj.items():
                if _is_vec_list(v):
                    vecs.append(v)
                elif isinstance(v, dict):
                    vv, _ = _extract_vec(v)
                    if vv is not None:
                        vecs.append(vv)
        elif isinstance(obj, list):
            for v in obj:
                if _is_vec_list(v):
                    vecs.append(v)
                else:
                    vv, _ = _extract_vec(v)
                    if vv is not None:
                        vecs.append(vv)
        if vecs:
            out[a] = vecs
    return out


# ---------------------------- plotting ----------------------------

def plot_reasoning_shape_3d(
    plane_tag,
    anchor_centers,   # dict name->vec
    bases,            # dict name->list(vec)
    loops,            # list of loop dicts
    edges,            # list (a,b,sim)
    out_png,
    draw_bases=False,
    max_bases_per_anchor=8,
    draw_transport=False,
    draw_loops=False,
    annotate=False
):
    # Build one big matrix for PCA so everything shares the same projection
    names_cent = sorted(anchor_centers.keys())
    X_cent = [anchor_centers[n] for n in names_cent]

    # bases points: represent each basis as (center + basis) so it “spikes” from center
    base_points = []   # list of (anchor, vec_endpoint)
    if draw_bases:
        for a, vecs in bases.items():
            if a not in anchor_centers:
                continue
            c = np.array(anchor_centers[a], dtype=np.float64)
            for v in vecs[:max_bases_per_anchor]:
                vv = np.array(v, dtype=np.float64)
                # endpoint in latent space
                base_points.append((a, (c + vv).tolist()))

    X_all = X_cent + [p[1] for p in base_points]
    if len(X_all) < 3:
        raise SystemExit("[error] Not enough points for PCA.")

    D = min(len(v) for v in X_all)
    X_all = np.array([v[:D] for v in X_all], dtype=np.float64)

    pca = PCA(n_components=3)
    Y_all = pca.fit_transform(X_all)

    # split back
    Y_cent = Y_all[:len(X_cent)]
    Y_base = Y_all[len(X_cent):] if base_points else np.zeros((0,3), dtype=np.float64)

    coord_cent = {n: Y_cent[i] for i, n in enumerate(names_cent)}

    # scatter anchor centers
    labels = [group_label(n) for n in names_cent]
    uniq = sorted(set(labels))
    lab2idx = {lab:i for i,lab in enumerate(uniq)}
    cvals = np.array([lab2idx[lab] for lab in labels], dtype=np.float64)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(Y_cent[:,0], Y_cent[:,1], Y_cent[:,2], c=cvals, s=70, alpha=0.95)

    # optional: annotate anchor names
    if annotate:
        for i, n in enumerate(names_cent):
            x,y,z = Y_cent[i]
            ax.text(x, y, z, n, fontsize=7)

    # draw bases as spikes from anchor center
    if draw_bases and base_points:
        for i, (a, _) in enumerate(base_points):
            if a not in coord_cent:
                continue
            p0 = coord_cent[a]
            p1 = Y_base[i]
            ax.plot([p0[0], p1[0]],[p0[1], p1[1]],[p0[2], p1[2]], linewidth=1.2, alpha=0.8)

    # draw transport edges (between anchor centers)
    if draw_transport and edges:
        for (a,b,sim) in edges:
            if a not in coord_cent or b not in coord_cent:
                continue
            p0 = coord_cent[a]; p1 = coord_cent[b]
            ax.plot([p0[0], p1[0]],[p0[1], p1[1]],[p0[2], p1[2]], linewidth=1.0, alpha=0.45)

    # draw holonomy loops (polyline over anchor centers)
    if draw_loops and loops:
        for L in loops:
            nodes = L.get("loop", None)
            if not nodes or not isinstance(nodes, list):
                continue
            nodes2 = nodes + [nodes[0]]
            ok = True
            pts = []
            for nm in nodes2:
                if nm not in coord_cent:
                    ok = False
                    break
                pts.append(coord_cent[nm])
            if not ok:
                continue
            pts = np.array(pts, dtype=np.float64)
            ax.plot(pts[:,0], pts[:,1], pts[:,2], linewidth=2.0, alpha=0.95)

    # legend proxies (no fixed colors)
    from matplotlib.lines import Line2D
    proxies = [Line2D([0],[0], marker='o', linestyle='None', label=lab) for lab in uniq]
    ax.legend(handles=proxies, title="anchor groups", loc="best")

    ax.set_title(f"Phase18B Reasoning Shape 3D — plane={plane_tag}")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close(fig)

    return {
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "N_anchor_centers": int(len(names_cent)),
        "N_basis_endpoints": int(len(base_points)),
        "D_used": int(D),
    }


# ---------------------------- main ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--plane", default="bo_lr", choices=["bo","lt","bo_lr"])
    ap.add_argument("--loops_json", default="")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--min_sim", type=float, default=None)
    ap.add_argument("--top_loops", type=int, default=10)

    ap.add_argument("--draw_bases", action="store_true")
    ap.add_argument("--max_bases_per_anchor", type=int, default=8)

    ap.add_argument("--draw_transport", action="store_true")
    ap.add_argument("--min_edge_sim", type=float, default=0.40)
    ap.add_argument("--max_edges", type=int, default=300)

    ap.add_argument("--draw_loops", action="store_true")
    ap.add_argument("--annotate", action="store_true")
    args = ap.parse_args()

    safe_mkdir(args.outdir)
    cache = json.load(open(args.cache, "r"))

    # anchors + bases
    anchor_centers = discover_anchor_centers(cache)
    bases = discover_bases(cache)

    # loops
    loops = load_loops_any_format(args.loops_json, threshold=args.threshold, min_sim=args.min_sim, top_loops=args.top_loops) if args.loops_json else []

    # edges
    edges = transport_edges(cache, args.plane, min_edge_sim=args.min_edge_sim, max_edges=args.max_edges) if args.draw_transport else []

    out_png = os.path.join(args.outdir, f"phase18b_reasoning_shape_{args.plane}.png")
    out_meta = os.path.join(args.outdir, f"phase18b_reasoning_shape_{args.plane}.json")

    if len(anchor_centers) < 3:
        meta = {
            "phase": "18b",
            "plane": args.plane,
            "status": "no_anchor_centers_found",
            "note": "cache['anchors'] did not contain vectors under common fields (t/z/vec/etc).",
            "anchors_count_in_cache": len(cache.get("anchors", {})) if isinstance(cache.get("anchors", {}), dict) else None,
            "bases_found_for_anchors": sorted(list(bases.keys()))
        }
        json.dump(meta, open(out_meta, "w"), indent=2)
        print("[error] Not enough anchor centers to build reasoning map (need >=3).")
        print(f"[saved] {out_meta}")
        return

    pca_meta = plot_reasoning_shape_3d(
        plane_tag=args.plane,
        anchor_centers=anchor_centers,
        bases=bases,
        loops=loops,
        edges=edges,
        out_png=out_png,
        draw_bases=args.draw_bases,
        max_bases_per_anchor=args.max_bases_per_anchor,
        draw_transport=args.draw_transport,
        draw_loops=args.draw_loops,
        annotate=args.annotate
    )

    meta = {
        "phase": "18b",
        "plane": args.plane,
        "status": "ok",
        "counts": {
            "anchor_centers": len(anchor_centers),
            "anchors_with_bases": len(bases),
            "loops_loaded": len(loops),
            "transport_edges_drawn": len(edges),
        },
        "pca": pca_meta,
        "args": vars(args),
        "paths": {"png": out_png, "meta": out_meta}
    }
    json.dump(meta, open(out_meta, "w"), indent=2)

    print(f"[ok] saved: {out_png}")
    print(f"[saved] {out_meta}")

if __name__ == "__main__":
    main()
