#!/usr/bin/env python3
# phase18c_reasoning_shape_3d.py
#
# Phase 18C: Reasoning-shape visualization in 3D (robust anchor center discovery).
#
# Goal:
#   Build a 3D map of "anchor points" in latent space and overlay:
#     - transport edges
#     - holonomy loops
#     - optional basis spikes
#
# Problem solved:
#   Many caches don't store anchor center vectors under cache["anchors"][name]["t"/"z"/...].
#   This script tries, in order:
#     (1) deep-scan each anchors[name] recursively for a 1D numeric list of dim>=--min_dim
#     (2) fallback: proxy center from bases[anchor] via deterministic "center proxy"
#         (mean of basis endpoints in a fixed, stable construction)
#
# Example:
#   python phase18c_reasoning_shape_3d.py ^
#     --cache outputs_edges_relternary256_phase15/phase15f_transport_cache.json ^
#     --outdir outputs_edges_relternary256_phase15/phase18c_bo_lr ^
#     --plane bo_lr ^
#     --loops_json outputs_edges_relternary256_phase15/phase16a_holonomy_bo_lr.json ^
#     --threshold 0.35 --min_sim 0.35 --top_loops 10 ^
#     --draw_loops --draw_transport --min_edge_sim 0.40 --max_edges 300 ^
#     --draw_bases --max_bases_per_anchor 8 ^
#     --min_dim 64
#
# Output:
#   - phase18c_reasoning_shape_<plane>.png
#   - phase18c_reasoning_shape_<plane>.json (meta + diagnostics)

import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# ---------------------------- utils ----------------------------

def safe_mkdir(p):
    if p and not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def _is_num(x):
    return isinstance(x, (int, float)) and np.isfinite(x)

def _is_vec_list(v, min_dim):
    return isinstance(v, list) and len(v) >= min_dim and all(_is_num(x) for x in v)

def group_label(name: str):
    s = name.lower()
    for pref in ["between", "overlap", "left", "right", "above", "below", "inside", "contain", "touch", "disjoint"]:
        if s.startswith(pref):
            return pref
    return s.split("_", 1)[0] if "_" in s else s

def deep_find_vec(obj, min_dim, max_depth=8, _depth=0, _path="$"):
    """
    Recursively search for the first numeric 1D list with len>=min_dim.
    Returns (vec, path_string) or (None, None).
    """
    if _depth > max_depth:
        return None, None

    if _is_vec_list(obj, min_dim):
        return obj, f"{_path}:(raw_list)"

    if isinstance(obj, dict):
        # Prefer common fields first if present
        for fld in ["t","z","latent","vec","embedding","emb","code","mu","mean","center"]:
            if fld in obj and _is_vec_list(obj[fld], min_dim):
                return obj[fld], f"{_path}.{fld}"

        # Otherwise recurse
        for k, v in obj.items():
            vec, p = deep_find_vec(v, min_dim, max_depth=max_depth, _depth=_depth+1, _path=f"{_path}.{k}")
            if vec is not None:
                return vec, p

    if isinstance(obj, list):
        for i, v in enumerate(obj):
            vec, p = deep_find_vec(v, min_dim, max_depth=max_depth, _depth=_depth+1, _path=f"{_path}[{i}]")
            if vec is not None:
                return vec, p

    return None, None

def discover_bases(cache, min_dim):
    """
    bases[anchor] may be dict or list; we extract vec lists with len>=min_dim.
    Returns dict anchor -> list(vec)
    """
    out = {}
    bases = cache.get("bases", {})
    if not isinstance(bases, dict):
        return out

    for a, obj in bases.items():
        vecs = []
        if isinstance(obj, dict):
            for _, v in obj.items():
                if _is_vec_list(v, min_dim):
                    vecs.append(v)
                elif isinstance(v, (dict, list)):
                    vv, _ = deep_find_vec(v, min_dim)
                    if vv is not None:
                        vecs.append(vv)
        elif isinstance(obj, list):
            for v in obj:
                if _is_vec_list(v, min_dim):
                    vecs.append(v)
                else:
                    vv, _ = deep_find_vec(v, min_dim)
                    if vv is not None:
                        vecs.append(vv)
        if vecs:
            out[a] = vecs
    return out

def discover_anchor_centers_deepscan(cache, min_dim):
    """
    Try to find one vector per anchor by deep scan of cache["anchors"][name].
    Returns:
      centers: dict name->vec
      paths: dict name->path_string
    """
    centers = {}
    paths = {}
    anchors = cache.get("anchors", {})
    if not isinstance(anchors, dict):
        return centers, paths

    for name, obj in anchors.items():
        vec, path = deep_find_vec(obj, min_dim=min_dim)
        if vec is not None:
            centers[name] = vec
            paths[name] = path
    return centers, paths

def proxy_centers_from_bases(bases_dict):
    """
    If we have no true anchor centers, create a stable proxy center per anchor
    from its bases vectors. This gives a consistent "map" that reflects
    frame structure even without explicit centers.

    Construction:
      center_proxy = mean( normalize(v_i) )  (then scaled back to typical magnitude)
    """
    centers = {}
    notes = {}
    for a, vecs in bases_dict.items():
        V = np.array(vecs, dtype=np.float64)
        # normalize each basis vector
        norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
        Vn = V / norms
        c = Vn.mean(axis=0)
        # scale by median norm so magnitudes stay reasonable
        med = float(np.median(norms))
        c = c * med
        centers[a] = c.tolist()
        notes[a] = "proxy_from_bases(mean(normalized_basis))*median_norm"
    return centers, notes

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


# ---------------------------- plotting ----------------------------

def plot_3d(
    plane,
    centers,                 # dict name->vec
    bases,                   # dict anchor->list(vec)
    loops,                   # list of loop dicts
    edges,                   # list (a,b,sim)
    out_png,
    draw_bases=False,
    max_bases_per_anchor=8,
    draw_transport=False,
    draw_loops=False,
    annotate=False
):
    names = sorted(centers.keys())
    X_cent = [centers[n] for n in names]
    if len(X_cent) < 3:
        raise SystemExit("[error] Need >=3 centers to plot.")

    # Create basis endpoints in latent space so spikes are meaningful in PCA space
    base_points = []
    if draw_bases:
        for a, vecs in bases.items():
            if a not in centers:
                continue
            c = np.array(centers[a], dtype=np.float64)
            for v in vecs[:max_bases_per_anchor]:
                vv = np.array(v, dtype=np.float64)
                base_points.append((a, (c + vv).tolist()))

    X_all = X_cent + [p[1] for p in base_points]
    D = min(len(v) for v in X_all)
    X_all = np.array([v[:D] for v in X_all], dtype=np.float64)

    pca = PCA(n_components=3)
    Y_all = pca.fit_transform(X_all)

    Y_cent = Y_all[:len(X_cent)]
    Y_base = Y_all[len(X_cent):] if base_points else np.zeros((0,3), dtype=np.float64)

    coord = {n: Y_cent[i] for i, n in enumerate(names)}

    labels = [group_label(n) for n in names]
    uniq = sorted(set(labels))
    lab2idx = {lab:i for i,lab in enumerate(uniq)}
    cvals = np.array([lab2idx[lab] for lab in labels], dtype=np.float64)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(Y_cent[:,0], Y_cent[:,1], Y_cent[:,2], c=cvals, s=70, alpha=0.95)

    if annotate:
        for i, n in enumerate(names):
            x,y,z = Y_cent[i]
            ax.text(x, y, z, n, fontsize=7)

    if draw_bases and base_points:
        for i, (a, _) in enumerate(base_points):
            if a not in coord:
                continue
            p0 = coord[a]
            p1 = Y_base[i]
            ax.plot([p0[0], p1[0]],[p0[1], p1[1]],[p0[2], p1[2]], linewidth=1.2, alpha=0.8)

    if draw_transport and edges:
        for (a,b,sim) in edges:
            if a not in coord or b not in coord:
                continue
            p0 = coord[a]; p1 = coord[b]
            ax.plot([p0[0], p1[0]],[p0[1], p1[1]],[p0[2], p1[2]], linewidth=1.0, alpha=0.45)

    if draw_loops and loops:
        for L in loops:
            nodes = L.get("loop", None)
            if not nodes or not isinstance(nodes, list):
                continue
            nodes2 = nodes + [nodes[0]]
            pts = []
            ok = True
            for nm in nodes2:
                if nm not in coord:
                    ok = False
                    break
                pts.append(coord[nm])
            if not ok:
                continue
            pts = np.array(pts, dtype=np.float64)
            ax.plot(pts[:,0], pts[:,1], pts[:,2], linewidth=2.0, alpha=0.95)

    ax.set_title(f"Phase18C Reasoning Shape 3D — plane={plane}")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close(fig)

    return {
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "N": int(len(names)),
        "D_used": int(D),
        "basis_endpoints": int(len(base_points))
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

    ap.add_argument("--min_dim", type=int, default=64, help="Minimum vector dimension to consider as a latent vector.")
    ap.add_argument("--max_depth", type=int, default=8, help="Recursion depth for deep scan.")
    args = ap.parse_args()

    safe_mkdir(args.outdir)
    cache = json.load(open(args.cache, "r"))

    # 1) Bases first (we know you have these)
    bases = discover_bases(cache, min_dim=args.min_dim)

    # 2) Try deep-scan anchor centers
    centers, paths = discover_anchor_centers_deepscan(cache, min_dim=args.min_dim)

    strategy = "anchors_deep_scan"
    proxy_notes = {}

    # 3) If insufficient centers, fall back to proxy centers from bases
    if len(centers) < 3:
        centers, proxy_notes = proxy_centers_from_bases(bases)
        strategy = "proxy_centers_from_bases"

    # loops + edges
    loops = load_loops_any_format(args.loops_json, threshold=args.threshold, min_sim=args.min_sim, top_loops=args.top_loops) if args.loops_json else []
    edges = transport_edges(cache, args.plane, min_edge_sim=args.min_edge_sim, max_edges=args.max_edges) if args.draw_transport else []

    out_png  = os.path.join(args.outdir, f"phase18c_reasoning_shape_{args.plane}.png")
    out_meta = os.path.join(args.outdir, f"phase18c_reasoning_shape_{args.plane}.json")

    meta = {
        "phase": "18c",
        "plane": args.plane,
        "status": "ok",
        "strategy": strategy,
        "counts": {
            "centers": len(centers),
            "bases_anchors": len(bases),
            "loops_loaded": len(loops),
            "transport_edges": len(edges)
        },
        "diagnostics": {
            "anchors_key_present": isinstance(cache.get("anchors", None), dict),
            "anchors_count": len(cache.get("anchors", {})) if isinstance(cache.get("anchors", {}), dict) else None,
            "center_paths_found_examples": dict(list(paths.items())[:10]),
            "proxy_notes_examples": dict(list(proxy_notes.items())[:10]),
        },
        "args": vars(args),
        "paths": {"png": out_png, "meta": out_meta}
    }

    if len(centers) < 3:
        meta["status"] = "error"
        meta["error"] = "Still not enough centers after proxy step (unexpected)."
        json.dump(meta, open(out_meta, "w"), indent=2)
        print("[error] Still not enough centers to plot.")
        print(f"[saved] {out_meta}")
        return

    pca_meta = plot_3d(
        plane=args.plane,
        centers=centers,
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
    meta["pca"] = pca_meta

    json.dump(meta, open(out_meta, "w"), indent=2)
    print(f"[ok] saved: {out_png}")
    print(f"[saved] {out_meta}")
    print(f"[note] center_strategy={strategy}")

if __name__ == "__main__":
    main()
