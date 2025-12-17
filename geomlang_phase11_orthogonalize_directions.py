#!/usr/bin/env python3
# geomlang_phase11_orthogonalize_directions.py
#
# Phase 11A: Orthogonalize v_between vs v_overlap (and optionally others),
# then save "clean" directions for plane maps / walks.

import os, json
import numpy as np

OUTDIR = "outputs_edges_relternary256_phase11"
os.makedirs(OUTDIR, exist_ok=True)

PH10_DIRS = "outputs_edges_relternary256_phase10/phase10_directions.npz"

def unit(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + eps)

def gram_schmidt(v, u):
    """
    Remove the component of v along u: v_perp = v - (v·u)u
    Assumes u is unit. Returns unit(v_perp) and dot(v,u) before removal.
    """
    u = unit(u)
    dot = float(np.dot(v, u))
    v_perp = v - dot * u
    return unit(v_perp), dot

dirs = np.load(PH10_DIRS)
v_lr      = unit(dirs["v_lr"])
v_between = unit(dirs["v_between"])
v_overlap = unit(dirs["v_overlap"])
v_tproj   = unit(dirs["v_tproj"])

# Key operation: make a "between_clean" axis perpendicular to overlap
v_between_clean, dot_bo = gram_schmidt(v_between, v_overlap)

# Also optional: make overlap_clean perpendicular to between (symmetry)
v_overlap_clean, dot_ob = gram_schmidt(v_overlap, v_between)

report = {
    "dot_before": {
        "v_between·v_overlap": float(np.dot(v_between, v_overlap)),
        "v_between_clean·v_overlap": float(np.dot(v_between_clean, v_overlap)),
        "v_overlap_clean·v_between": float(np.dot(v_overlap_clean, v_between)),
    },
    "removed_components": {
        "proj_coeff_between_on_overlap": dot_bo,
        "proj_coeff_overlap_on_between": dot_ob,
    },
    "notes": [
        "Use v_between_clean when you want between changes without overlap spillover.",
        "Use v_overlap_clean when you want overlap changes without between spillover.",
    ]
}

np.savez(
    os.path.join(OUTDIR, "phase11_directions_ortho.npz"),
    v_lr=v_lr,
    v_between=v_between,
    v_overlap=v_overlap,
    v_tproj=v_tproj,
    v_between_clean=v_between_clean,
    v_overlap_clean=v_overlap_clean,
)

with open(os.path.join(OUTDIR, "phase11_ortho_report.json"), "w") as f:
    json.dump(report, f, indent=2)

print("[phase11A] saved ->", OUTDIR)
print(json.dumps(report, indent=2))
