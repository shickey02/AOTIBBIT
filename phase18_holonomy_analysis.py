#!/usr/bin/env python3
# phase18_holonomy_analysis.py
#
# Phase 18: Anchor latent 3D visualization + optional loop overlays.
#
# Robustly discovers anchor vectors even if cache["anchors"] does NOT contain vectors,
# by deep-scanning the cache for (anchor_name -> vector) tables or (anchor_name -> {"vec":[...]}) style maps.
#
# Outputs:
#   - phase18_anchor3d_pca.png
#   - phase18_anchor3d_meta.json
#
# Example:
#   python phase18_holonomy_analysis.py ^
#     --cache outputs_edges_relternary256_phase15/phase15f_transport_cache.json ^
#     --outdir outputs_edges_relternary256_phase15/phase18_bo_lr ^
#     --plane bo_lr ^
#     --loops_json outputs_edges_relternary256_phase15/phase16a_holonomy_bo_lr.json ^
#     --threshold 0.35 --min_sim 0.35 --top_loops 10

import os, json, argparse, math
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

def _extract_vec_from_obj(obj):
    """
    Return (vec, fieldname) if obj contains a vector.
    Accepts:
      - raw list of numbers
      - dict with common vector fields
    """
    if _is_vec_list(obj):
        return obj, "(raw_list)"
    if isinstance(obj, dict):
        for fld in ["t","z","latent","vec","embedding","emb","code","mu","mean","center"]:
            if fld in obj and _is_vec_list(obj[fld]):
                return obj[fld], fld
    return None, None

def find_anchor_vectors(cache, max_depth=6):
    """
    Tries multiple strategies:
      1) cache["anchors"][name] contains vector-ish fields
      2) deep scan for dicts shaped like {anchor_name: vector_or_dict_with_vector}
    Returns (names, X, note, diagnostics)
    """
    diagnostics = {
        "strategy": None,
        "note": None,
        "found_count": 0,
        "dim": None,
        "candidate_count": 0,
        "anchors_key_present": "anchors" in cache,
        "anchors_count": len(cache.get("anchors", {})) if isinstance(cache.get("anchors", {}), dict) else None
    }

    # ---------- Strategy 1: anchors dict ----------
    anchors = cache.get("anchors", {})
    if isinstance(anchors, dict) and anchors:
        names, X = [], []
        field_counts = {}
        for name, obj in anchors.items():
            vec, fld = _extract_vec_from_obj(obj)
            if vec is not None:
                names.append(name); X.append(vec)
                field_counts[fld] = field_counts.get(fld, 0) + 1

        if len(X) >= 3:
            d = min(len(v) for v in X)
            X = np.array([v[:d] for v in X], dtype=np.float64)
            best_fld = max(field_counts.items(), key=lambda kv: kv[1])[0]
            diagnostics.update({
                "strategy": "anchors",
                "note": f"anchors.{best_fld}",
                "found_count": len(names),
                "dim": int(X.shape[1]),
            })
            return names, X, diagnostics["note"], diagnostics

    # ---------- Strategy 2: deep scan ----------
    anchor_names = set(anchors.keys()) if isinstance(anchors, dict) else set()
    candidates = []  # (score, note, names, X, overlap)

    def consider_table(path, table: dict):
        names, vecs, fld_counts = [], [], {}
        for k, obj in table.items():
            vec, fld = _extract_vec_from_obj(obj)
            if vec is None:
                continue
            names.append(k); vecs.append(vec)
            fld_counts[fld] = fld_counts.get(fld, 0) + 1

        if len(vecs) < 3:
            return

        d = min(len(v) for v in vecs)
        X = np.array([v[:d] for v in vecs], dtype=np.float64)

        overlap = len(set(names) & anchor_names) if anchor_names else 0
        best_fld = max(fld_counts.items(), key=lambda kv: kv[1])[0]
        score = len(names) + 2.0 * overlap
        candidates.append((score, f"{path}.*:{best_fld}", names, X, overlap))

    def walk(obj, path="$", depth=0):
        if depth > max_depth:
            return
        if isinstance(obj, dict):
            if len(obj) >= 3:
                vec_hits = 0
                for _, v in list(obj.items())[:60]:
                    vv, _ = _extract_vec_from_obj(v)
                    if vv is not None:
                        vec_hits += 1
                    if vec_hits >= 3:
                        consider_table(path, obj)
                        break
            for k, v in obj.items():
                walk(v, path=f"{path}.{k}", depth=depth+1)
        elif isinstance(obj, list):
            for i, v in enumerate(obj[:80]):
                walk(v, path=f"{path}[{i}]", depth=depth+1)

    walk(cache, "$", 0)
    diagnostics["candidate_count"] = len(candidates)

    if not candidates:
        diagnostics.update({
            "strategy": "none",
            "note": "no_vector_field_found_in_anchors_or_elsewhere",
            "found_count": 0,
            "dim": None
        })
        return None, None, diagnostics["note"], diagnostics

    # Prefer overlap with cache["anchors"] names if available; else highest score
    if anchor_names:
        candidates.sort(key=lambda t: (t[4], t[0]), reverse=True)  # (overlap, score)
    else:
        candidates.sort(key=lambda t: t[0], reverse=True)

    score, note, names, X, overlap = candidates[0]
    diagnostics.update({
        "strategy": "deep_scan",
        "note": f"deep_scan:{note}",
        "found_count": len(names),
        "dim": int(X.shape[1]),
        "overlap_with_anchors_names": int(overlap)
    })
    return names, X, diagnostics["note"], diagnostics


def group_label(name: str):
    """
    Simple semantic grouping for coloring.
    You can tune this as your anchor naming evolves.
    """
    s = name.lower()
    for pref in ["between", "overlap", "left", "right", "above", "below", "inside", "contain", "touch", "disjoint"]:
        if s.startswith(pref):
            return pref
    # fallback: first token
    return s.split("_", 1)[0] if "_" in s else s


def load_loops_any_format(path, threshold=None, min_sim=None, top_loops=None):
    """
    Loads Phase16a JSON in either:
      - legacy: {"top":[...], "all_count":N, "plane":"..."}
      - threshold_results: {"plane":..., "threshold_results":{"0.35":{...,"top":[...]}, ...}, ...}

    Returns loops list (each loop entry is dict with 'loop' list of node names).
    """
    if not path:
        return []

    data = json.load(open(path, "r"))

    loops = []
    meta = {"format": "unknown"}

    if isinstance(data, dict) and "threshold_results" in data and isinstance(data["threshold_results"], dict):
        tr = data["threshold_results"]

        def _as_float(x):
            try: return float(x)
            except: return None

        chosen_key = None
        if threshold is not None:
            # exact match attempts
            k1 = f"{threshold:.2f}"
            if k1 in tr: chosen_key = k1
            elif str(threshold) in tr: chosen_key = str(threshold)

        if chosen_key is None:
            # choose highest numeric threshold key
            numeric = [(k, _as_float(k)) for k in tr.keys()]
            numeric = [(k, v) for (k, v) in numeric if v is not None]
            if numeric:
                numeric.sort(key=lambda kv: kv[1], reverse=True)
                chosen_key = numeric[0][0]
            else:
                chosen_key = list(tr.keys())[0]

        bucket = tr.get(chosen_key, {})
        loops = bucket.get("top", []) if isinstance(bucket, dict) else (bucket if isinstance(bucket, list) else [])
        meta = {
            "format": "threshold_results",
            "chosen_key": chosen_key,
            "threshold_arg": threshold,
            "min_sim_arg": min_sim,
            "bucket_min_sim": bucket.get("min_sim", None) if isinstance(bucket, dict) else None
        }

    else:
        loops = data.get("top", data.get("results", []))
        meta = {"format": "legacy", "min_sim_arg": min_sim}

    # optional min_sim filter
    if min_sim is not None:
        keep = []
        for L in loops:
            try:
                if float(L.get("edge_sim_min", -1e9)) >= float(min_sim):
                    keep.append(L)
            except Exception:
                pass
        loops = keep

    # optional truncate
    if top_loops is not None and top_loops > 0:
        loops = loops[:top_loops]

    return loops


def plot_anchor_pca_3d(names, X, out_png, title, loops=None, overlay_loops=True, overlay_top_k=None, annotate=False):
    """
    PCA -> 3D scatter. Optionally overlays loop edges.
    """
    pca = PCA(n_components=3)
    Y = pca.fit_transform(X)

    # map name -> coords
    coord = {n: Y[i] for i, n in enumerate(names)}

    # group coloring
    labels = [group_label(n) for n in names]
    uniq = sorted(set(labels))
    label_to_idx = {lab: i for i, lab in enumerate(uniq)}
    c = np.array([label_to_idx[lab] for lab in labels], dtype=np.float64)

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(Y[:,0], Y[:,1], Y[:,2], c=c, s=40, alpha=0.9)

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    # legend using proxy artists (keep it simple)
    # NOTE: matplotlib will auto-assign a colormap; we don't set specific colors.
    from matplotlib.lines import Line2D
    proxies = []
    for lab in uniq:
        proxies.append(Line2D([0],[0], marker='o', linestyle='None', label=lab))
    ax.legend(handles=proxies, title="anchor groups", loc="best")

    if annotate:
        for i, n in enumerate(names):
            x,y,z = Y[i]
            ax.text(x, y, z, n, fontsize=7)

    # loop overlays
    if overlay_loops and loops:
        # choose which loops to overlay
        Ls = loops
        if overlay_top_k is not None and overlay_top_k > 0:
            Ls = loops[:overlay_top_k]

        for L in Ls:
            nodes = L.get("loop", None)
            if not nodes or not isinstance(nodes, list):
                continue
            nodes2 = nodes + [nodes[0]]
            ok = True
            pts = []
            for nm in nodes2:
                if nm not in coord:
                    ok = False
                    break
                pts.append(coord[nm])
            if not ok:
                continue
            pts = np.array(pts, dtype=np.float64)
            ax.plot(pts[:,0], pts[:,1], pts[:,2], linewidth=1.5, alpha=0.8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close(fig)

    return {
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pca_singular_values": pca.singular_values_.tolist(),
        "N": int(len(names)),
        "D_used": int(X.shape[1]),
    }


# ---------------------------- main ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="Path to phase15f_transport_cache.json")
    ap.add_argument("--outdir", required=True, help="Output directory for Phase18 artifacts")
    ap.add_argument("--plane", default="", choices=["", "bo", "lt", "bo_lr"], help="Optional label only (for filenames/titles)")
    ap.add_argument("--loops_json", default="", help="Optional Phase16a holonomy json to overlay loops")
    ap.add_argument("--threshold", type=float, default=None, help="If loops_json is threshold_results, choose threshold bucket (e.g. 0.35)")
    ap.add_argument("--min_sim", type=float, default=None, help="Optional: require loop edge_sim_min >= min_sim")
    ap.add_argument("--top_loops", type=int, default=10, help="How many loops to load/overlay (if loops_json provided)")
    ap.add_argument("--overlay_loops", action="store_true", help="If set, draw loop polylines on the 3D scatter")
    ap.add_argument("--overlay_top_k", type=int, default=None, help="Draw at most this many loops (default: same as top_loops)")
    ap.add_argument("--annotate", action="store_true", help="If set, write anchor names next to points (can be cluttered)")
    args = ap.parse_args()

    safe_mkdir(args.outdir)

    cache = json.load(open(args.cache, "r"))

    # find vectors
    names, X, note, diag = find_anchor_vectors(cache, max_depth=6)

    plane_tag = args.plane.strip() or "unknown"
    out_png = os.path.join(args.outdir, f"phase18_anchor3d_pca_{plane_tag}.png")
    out_meta = os.path.join(args.outdir, f"phase18_anchor3d_meta_{plane_tag}.json")

    if names is None or X is None or len(names) < 3:
        meta = {
            "phase": "18",
            "plane": plane_tag,
            "status": "no_vectors_found",
            "note": note,
            "diagnostics": diag
        }
        with open(out_meta, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[anchor3d] not generated: {note}")
        print(f"[saved] {out_meta}")
        return

    # load loops (optional)
    loops = []
    loops_meta = None
    if args.loops_json:
        loops = load_loops_any_format(
            args.loops_json,
            threshold=args.threshold,
            min_sim=args.min_sim,
            top_loops=args.top_loops
        )
        loops_meta = {
            "loops_json": args.loops_json,
            "threshold": args.threshold,
            "min_sim": args.min_sim,
            "loaded_loops": len(loops),
            "overlay_loops": bool(args.overlay_loops),
            "overlay_top_k": args.overlay_top_k if args.overlay_top_k is not None else args.top_loops
        }

    title = f"Phase18 Anchor PCA 3D — plane={plane_tag} — vectors={len(names)} ({note})"
    pca_meta = plot_anchor_pca_3d(
        names, X,
        out_png=out_png,
        title=title,
        loops=loops,
        overlay_loops=args.overlay_loops,
        overlay_top_k=(args.overlay_top_k if args.overlay_top_k is not None else args.top_loops),
        annotate=args.annotate
    )

    meta = {
        "phase": "18",
        "plane": plane_tag,
        "status": "ok",
        "vector_source_note": note,
        "diagnostics": diag,
        "pca": pca_meta,
        "loops": loops_meta
    }
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[ok] anchor3d generated: {out_png}")
    print(f"[saved] {out_meta}")


if __name__ == "__main__":
    main()
