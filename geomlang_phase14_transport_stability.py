#!/usr/bin/env python3
# geomlang_phase14_transport_stability.py
#
# Phase 14A: Transport stability under larger neighborhoods.
# - Ridge local directions at anchors with radius ~6 and min_pts ~2500
# - Clean between/overlap via Gram-Schmidt
# - Report pairwise alignment with cos + abs_cos (sign-invariant)
# - Report principal-angle subspace similarity:
#     BO subspace = span{between_clean, overlap_clean}
#     LT subspace = span{lr, tproj}

import os, json
import numpy as np

PHASE7_DIR = "outputs_edges_relternary256_phase7"
PHASE8_DIR = "outputs_edges_relternary256_phase8"
OUTDIR     = "outputs_edges_relternary256_phase14"
os.makedirs(OUTDIR, exist_ok=True)

LATENTS = os.path.join(PHASE7_DIR, "encoded_latents_seed123_N6000.npy")
TARGETS = os.path.join(PHASE7_DIR, "encoded_targets_seed123_N6000.npz")
PREDS   = os.path.join(PHASE7_DIR, "encoded_preds_seed123_N6000.npz")
THRESH  = os.path.join(PHASE8_DIR, "phase8_thresholds.json")

# ----------------------------
# knobs
# ----------------------------
DEFAULT_RADIUS = 6.0
DEFAULT_MINPTS = 2500
DEFAULT_LAM    = 1e-2

# Set to [] to disable sweeping, or e.g. [2.0, 4.0, 6.0, 8.0]
RADIUS_SWEEP = []  # recommended: [] for one run; set to [2,4,6,8] when exploring


def sigmoid_np(x):
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-x))

def unit(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n

def cos(a, b):
    return float(np.dot(unit(a), unit(b)))

def abs_cos(a, b):
    return float(abs(cos(a, b)))

def load_thresholds():
    with open(THRESH, "r") as f:
        d = json.load(f)
    return float(d["Tb"]), float(d["To"])

def ridge_direction_local(Z, y, center_z, radius=6.0, lam=1e-2, min_pts=2500):
    """
    Fit ridge direction in a local neighborhood around center_z (Euclidean in latent space).
    Returns (unit direction, count used).
    """
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    d = np.linalg.norm(Z - center_z[None, :], axis=1)
    idx = np.where(d <= radius)[0]
    if len(idx) < min_pts:
        K = min_pts
        idx = np.argsort(d)[:K]

    X = Z[idx]
    t = y[idx]

    # standardize X
    Xm = X.mean(axis=0, keepdims=True)
    Xs = X.std(axis=0, keepdims=True) + 1e-9
    Xn = (X - Xm) / Xs

    tm = t.mean()
    ts = t.std() + 1e-9
    tn = (t - tm) / ts

    A = Xn.T @ Xn
    A.flat[::A.shape[0] + 1] += lam
    b = Xn.T @ tn

    w = np.linalg.solve(A, b)

    # unstandardize
    w = w / Xs.reshape(-1)
    return unit(w), int(len(idx))

def gram_schmidt_clean(v_main, v_remove):
    v_remove = unit(v_remove)
    coeff = float(np.dot(v_main, v_remove))
    v_clean = v_main - coeff * v_remove
    return unit(v_clean), coeff

def choose_seed(mask, score):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None
    return int(idx[np.argmin(score[idx])])

def pick_anchor_seeds(between_pred, overlap_prob, lr_prob, Tb, To):
    """
    Pick anchors spanning regimes (same logic as your Phase 13, but stable).
    """
    b  = between_pred.reshape(-1)
    o  = overlap_prob.reshape(-1)
    lr = lr_prob.reshape(-1)

    LOWO = 0.35
    anchors = {}

    # 1) between boundary clear (near Tb, low overlap)
    for eps in [0.005,0.01,0.02,0.04,0.06,0.08,0.10]:
        m = (np.abs(b - Tb) < eps) & (o < LOWO)
        s = np.abs(b - Tb) + 0.50*o
        j = choose_seed(m, s)
        if j is not None:
            anchors["between_boundary_clear"] = j
            break

    # 2) overlap boundary only-ish (near To, below between)
    m = (np.abs(o - To) < 0.06) & (b < Tb - 0.06)
    s = np.abs(o - To) + 0.15*(Tb - b)
    j = choose_seed(m, s)
    if j is not None:
        anchors["overlap_boundary_only"] = j

    # 3) deep between clear
    m = (b > 0.85) & (o < LOWO)
    s = -b + 0.5*o
    j = choose_seed(m, s)
    if j is not None:
        anchors["between_deep_clear"] = j

    # 4) deep overlap only
    m = (o > 0.85) & (b < 0.30)
    s = -o + 0.25*b
    j = choose_seed(m, s)
    if j is not None:
        anchors["overlap_deep_only"] = j

    # 5) left clean
    m = (b < 0.30) & (o < LOWO) & (lr < 0.5)
    s = b + o
    j = choose_seed(m, s)
    if j is not None:
        anchors["left_clean"] = j

    return anchors

def _orthonormal_basis_from_two(v1, v2, eps=1e-12):
    """
    Return Q (D,2) with orthonormal columns spanning {v1,v2}.
    Handles near-collinearity by re-orthogonalizing.
    """
    a = unit(v1)
    b = np.asarray(v2, dtype=np.float64)
    b = b - np.dot(b, a) * a
    nb = np.linalg.norm(b)
    if nb < eps:
        # fallback: pick an arbitrary orthogonal direction
        # choose coordinate axis least aligned with a
        k = int(np.argmin(np.abs(a)))
        e = np.zeros_like(a); e[k] = 1.0
        b = e - np.dot(e, a) * a
        nb = np.linalg.norm(b) + eps
    b = b / nb
    Q = np.stack([a, b], axis=1)  # (D,2)
    return Q

def principal_angles_deg(Q1, Q2):
    """
    Q1,Q2: (D,k) orthonormal columns
    Returns angles (deg) and similarity = mean(cos(theta)).
    """
    M = Q1.T @ Q2
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    angles = np.degrees(np.arccos(s))
    similarity = float(np.mean(s))
    return similarity, angles.tolist()

def compute_for_radius(Z, targets, preds, Tb, To, radius, min_pts, lam):
    between_true = targets["between_score"].reshape(-1)
    overlap_true = targets["overlap_any"].reshape(-1)
    tproj_true   = targets["t_on_BC"].reshape(-1)
    lr_true      = targets["lr_sign"].reshape(-1)

    between_pred = preds["between_pred"].reshape(-1)
    overlap_prob = sigmoid_np(preds["overlap_logit"]).reshape(-1)
    lr_prob      = sigmoid_np(preds["lr_logit"]).reshape(-1)
    tproj_pred   = preds["tproj_pred"].reshape(-1)

    anchors = pick_anchor_seeds(between_pred, overlap_prob, lr_prob, Tb, To)
    if "between_boundary_clear" not in anchors:
        anchors["between_boundary_clear"] = 0

    bases = {}
    meta  = {}

    for name, idx in anchors.items():
        z0 = Z[idx].astype(np.float64)

        vb, nb = ridge_direction_local(Z, between_true, z0, radius=radius, min_pts=min_pts, lam=lam)
        vo, no = ridge_direction_local(Z, overlap_true, z0, radius=radius, min_pts=min_pts, lam=lam)
        vt, nt = ridge_direction_local(Z, tproj_true,   z0, radius=radius, min_pts=min_pts, lam=lam)
        vl, nl = ridge_direction_local(Z, lr_true,      z0, radius=radius, min_pts=min_pts, lam=lam)

        vb_clean, cb = gram_schmidt_clean(vb, vo)
        vo_clean, co = gram_schmidt_clean(vo, vb)

        bases[name] = {
            "v_between_clean": vb_clean,
            "v_overlap_clean": vo_clean,
            "v_lr": vl,
            "v_tproj": vt,
        }

        meta[name] = {
            "seed": int(idx),
            "counts": {"between": nb, "overlap": no, "tproj": nt, "lr": nl},
            "clean_coeffs": {"between_on_overlap": cb, "overlap_on_between": co},
            "signals_at_seed": {
                "between_pred": float(between_pred[idx]),
                "tproj_pred": float(tproj_pred[idx]),
                "overlap_prob": float(overlap_prob[idx]),
                "lr_prob": float(lr_prob[idx]),
            }
        }

    names = list(bases.keys())

    # pairwise alignment (sign-aware + sign-invariant)
    align = {}
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a = names[i]; b = names[j]
            align[f"{a}__vs__{b}"] = {
                "cos_between_clean": cos(bases[a]["v_between_clean"], bases[b]["v_between_clean"]),
                "abs_between_clean": abs_cos(bases[a]["v_between_clean"], bases[b]["v_between_clean"]),
                "cos_overlap_clean": cos(bases[a]["v_overlap_clean"], bases[b]["v_overlap_clean"]),
                "abs_overlap_clean": abs_cos(bases[a]["v_overlap_clean"], bases[b]["v_overlap_clean"]),
                "cos_lr":            cos(bases[a]["v_lr"], bases[b]["v_lr"]),
                "abs_lr":            abs_cos(bases[a]["v_lr"], bases[b]["v_lr"]),
                "cos_tproj":         cos(bases[a]["v_tproj"], bases[b]["v_tproj"]),
                "abs_tproj":         abs_cos(bases[a]["v_tproj"], bases[b]["v_tproj"]),
            }

    # principal angles: BO and LT subspaces
    bo = {}
    lt = {}
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a = names[i]; b = names[j]

            Qbo_a = _orthonormal_basis_from_two(bases[a]["v_between_clean"], bases[a]["v_overlap_clean"])
            Qbo_b = _orthonormal_basis_from_two(bases[b]["v_between_clean"], bases[b]["v_overlap_clean"])
            sim_bo, ang_bo = principal_angles_deg(Qbo_a, Qbo_b)

            Qlt_a = _orthonormal_basis_from_two(bases[a]["v_lr"], bases[a]["v_tproj"])
            Qlt_b = _orthonormal_basis_from_two(bases[b]["v_lr"], bases[b]["v_tproj"])
            sim_lt, ang_lt = principal_angles_deg(Qlt_a, Qlt_b)

            key = f"{a}__vs__{b}"
            bo[key] = {"similarity": float(sim_bo), "angles_deg": ang_bo}
            lt[key] = {"similarity": float(sim_lt), "angles_deg": ang_lt}

    return {
        "Tb": Tb, "To": To,
        "radius": float(radius),
        "min_pts": int(min_pts),
        "lam": float(lam),
        "anchors": meta,
        "pairwise_alignment": align,
        "principal_angles": {
            "bo": bo,
            "lt": lt,
            "notes": [
                "Similarity is mean cos(principal angles). 1.0=identical subspace, 0.0=orthogonal.",
                "Angles are in degrees; smaller is more stable under transport.",
                "If subspaces are stable but single vectors are not, the factor exists globally but twists within that subspace."
            ]
        }
    }

def main():
    Tb, To = load_thresholds()

    Z = np.load(LATENTS)
    targets = np.load(TARGETS)
    preds   = np.load(PREDS)

    radii = RADIUS_SWEEP if len(RADIUS_SWEEP) else [DEFAULT_RADIUS]

    all_runs = {}
    for r in radii:
        rep = compute_for_radius(
            Z, targets, preds, Tb, To,
            radius=float(r),
            min_pts=DEFAULT_MINPTS,
            lam=DEFAULT_LAM
        )
        all_runs[f"radius_{float(r):.2f}"] = rep

        out_path = os.path.join(OUTDIR, f"phase14_transport_report_radius{float(r):.2f}.json")
        with open(out_path, "w") as f:
            json.dump(rep, f, indent=2)
        print("[phase14] saved ->", out_path)

    if len(radii) > 1:
        out_path = os.path.join(OUTDIR, "phase14_transport_report_ALL.json")
        with open(out_path, "w") as f:
            json.dump(all_runs, f, indent=2)
        print("[phase14] saved combined ->", out_path)

if __name__ == "__main__":
    main()
