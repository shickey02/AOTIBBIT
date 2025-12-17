#!/usr/bin/env python3
# geomlang_phase15e_global_matrices_and_heatmaps.py
#
# Phase 15E: Build GLOBAL anchor-to-anchor similarity / diagonal-cos matrices
# from your Phase15d cache, and render heatmaps.
#
# Input:
#   outputs_edges_relternary256_phase15/phase15d_transport_cache.json
#
# Output dir:
#   outputs_edges_relternary256_phase15/phase15e_viz/
# Produces:
#   - sim_bo.png, sim_lt.png
#   - diag1_bo.png, diag2_bo.png, diag1_lt.png, diag2_lt.png
#   - matrices_{plane}.npz (sim, diag1, diag2, names)
#   - axis_strengths.json (per-anchor median diag cos by plane)
#
# Only deps: numpy, matplotlib

import os, json
import numpy as np
import matplotlib.pyplot as plt

INDIR  = "outputs_edges_relternary256_phase15"
INJSON = os.path.join(INDIR, "phase15d_transport_cache.json")
OUTDIR = os.path.join(INDIR, "phase15e_viz")

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

def load_cache(path):
    with open(path, "r") as f:
        return json.load(f)

def heatmap(mat, names, title, outpath, vmin=None, vmax=None):
    mat = np.asarray(mat, dtype=np.float64)

    plt.figure(figsize=(max(6, 0.55*len(names)), max(5, 0.55*len(names))))
    im = plt.imshow(mat, interpolation="nearest", aspect="equal", vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.xticks(range(len(names)), names, rotation=90)
    plt.yticks(range(len(names)), names)

    # light grid for readability
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def build_matrices(cache, plane):
    anchors = cache.get("anchors", {})
    names = list(anchors.keys())
    n = len(names)

    # Map anchor name -> index
    idx_of = {name: i for i, name in enumerate(names)}

    sim   = np.full((n, n), np.nan, dtype=np.float64)
    diag1 = np.full((n, n), np.nan, dtype=np.float64)
    diag2 = np.full((n, n), np.nan, dtype=np.float64)

    # Set diagonal to 1 (self similarity; diag cos)
    np.fill_diagonal(sim, 1.0)
    np.fill_diagonal(diag1, 1.0)
    np.fill_diagonal(diag2, 1.0)

    tmap = cache.get("transport_maps", {})

    # Fill from A__to__B entries if present
    for key, per in tmap.items():
        if "__to__" not in key:
            continue
        A, B = key.split("__to__", 1)
        A = A.strip()
        B = B.strip()
        if A not in idx_of or B not in idx_of:
            continue
        if plane not in per:
            continue

        entry = per[plane]
        i = idx_of[A]
        j = idx_of[B]

        sim[i, j] = _safe_float(entry.get("similarity", np.nan))

        after = entry.get("transported_col_cos_after_sign", None)
        if isinstance(after, list) and len(after) >= 2:
            diag1[i, j] = _safe_float(after[0])
            diag2[i, j] = _safe_float(after[1])

    return names, sim, diag1, diag2

def per_anchor_axis_strengths(names, diag1, diag2):
    # For each anchor A, take median across B!=A of diag cos
    n = len(names)
    out = {}
    for i, name in enumerate(names):
        row1 = np.array(diag1[i, :], dtype=np.float64)
        row2 = np.array(diag2[i, :], dtype=np.float64)

        # exclude self
        row1[i] = np.nan
        row2[i] = np.nan

        out[name] = {
            "median_diag1": _safe_float(np.nanmedian(row1)),
            "median_diag2": _safe_float(np.nanmedian(row2)),
            "min_diag1": _safe_float(np.nanmin(row1)),
            "min_diag2": _safe_float(np.nanmin(row2)),
            "max_diag1": _safe_float(np.nanmax(row1)),
            "max_diag2": _safe_float(np.nanmax(row2)),
        }
    return out

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    cache = load_cache(INJSON)

    summary = {
        "input_json": INJSON,
        "outdir": OUTDIR,
        "notes": [
            "sim_* is cache['transport_maps'][A__to__B][plane]['similarity']",
            "diag1/diag2 are cache['transport_maps'][A__to__B][plane]['transported_col_cos_after_sign'][0/1]",
            "diagonal entries set to 1.0 by convention",
        ],
        "planes": {}
    }

    for plane in ["bo", "lt"]:
        names, sim, d1, d2 = build_matrices(cache, plane)

        # Save matrices
        npz_path = os.path.join(OUTDIR, f"matrices_{plane}.npz")
        np.savez_compressed(npz_path, names=np.array(names, dtype=object), sim=sim, diag1=d1, diag2=d2)

        # Heatmaps
        heatmap(sim, names, f"Phase15e {plane.upper()} similarity (principal cosine mean)", os.path.join(OUTDIR, f"sim_{plane}.png"), vmin=0.0, vmax=1.0)
        heatmap(d1,  names, f"Phase15e {plane.upper()} diag1 cos (after sign)",            os.path.join(OUTDIR, f"diag1_{plane}.png"), vmin=-1.0, vmax=1.0)
        heatmap(d2,  names, f"Phase15e {plane.upper()} diag2 cos (after sign)",            os.path.join(OUTDIR, f"diag2_{plane}.png"), vmin=-1.0, vmax=1.0)

        # Per-anchor axis strength summary
        strengths = per_anchor_axis_strengths(names, d1, d2)
        summary["planes"][plane] = {
            "names": names,
            "matrices_npz": npz_path,
            "heatmaps": {
                "sim":   os.path.join(OUTDIR, f"sim_{plane}.png"),
                "diag1": os.path.join(OUTDIR, f"diag1_{plane}.png"),
                "diag2": os.path.join(OUTDIR, f"diag2_{plane}.png"),
            },
            "axis_strengths": strengths,
        }

    # Write summary JSON
    out_json = os.path.join(OUTDIR, "phase15e_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("[phase15e] loaded:", INJSON)
    print("[phase15e] wrote :", OUTDIR)
    print("[phase15e] summary:", out_json)
    print("[phase15e] open PNGs:", os.path.join(OUTDIR, "sim_bo.png"), "and", os.path.join(OUTDIR, "sim_lt.png"))

if __name__ == "__main__":
    main()
