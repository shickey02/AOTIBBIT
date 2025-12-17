#!/usr/bin/env python3
# geomlang_phase15b_store_perm_signedcos.py
#
# Phase 15B: "Store perm" cache WITH raw signed cosines.
#
# For each anchor pair and each 2D factor plane (BO and LT):
#   - Build 2D orthonormal frames QA (at A) and QB (at B)
#   - Compute best-fit 2x2 rotation R (orthogonal Procrustes) s.t. QA @ R ~= QB
#   - Transport: Qt = QA @ R
#   - Find best column permutation (identity/swap) and per-column sign flips (+1/-1)
#     that maximize total ABS cosine with QB columns.
#   - Store BOTH raw signed cosines (before abs) and abs cosines, plus chosen perm/signs.
#
# Output: outputs_edges_relternary256_phase15/phase15b_transport_cache.json
#
# Notes:
# - BO plane uses v_between_clean, v_overlap_clean
# - LT plane uses v_lr, v_tproj
# - Ridge directions are fit locally via ridge regression on nearby latent samples.
# - This script is self-contained and safe to run once to sanity-check sign flips.

import os, json
import numpy as np

PHASE7_DIR = "outputs_edges_relternary256_phase7"
PHASE8_DIR = "outputs_edges_relternary256_phase8"
OUTDIR     = "outputs_edges_relternary256_phase15"
os.makedirs(OUTDIR, exist_ok=True)

LATENTS = os.path.join(PHASE7_DIR, "encoded_latents_seed123_N6000.npy")
TARGETS = os.path.join(PHASE7_DIR, "encoded_targets_seed123_N6000.npz")
PREDS   = os.path.join(PHASE7_DIR, "encoded_preds_seed123_N6000.npz")
THRESH  = os.path.join(PHASE8_DIR, "phase8_thresholds.json")

# ----------------------------
# Helpers
# ----------------------------

def sigmoid_np(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))

def unit(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < eps:
        return v
    return v / n

def ortho2(a, b, eps=1e-12):
    """
    Make an orthonormal 2D frame (D x 2) from two (D,) vectors.
    Column 0 = unit(a)
    Column 1 = unit(b - proj_a(b))
    If degenerate, falls back to any orthonormal completion.
    """
    a0 = unit(a)
    b1 = b - float(np.dot(b, a0)) * a0
    nb = float(np.linalg.norm(b1))
    if nb < eps:
        # fallback: pick a coordinate axis not aligned with a0
        # choose index of smallest abs component to build a basis direction
        idx = int(np.argmin(np.abs(a0)))
        e = np.zeros_like(a0)
        e[idx] = 1.0
        b1 = e - float(np.dot(e, a0)) * a0
        nb = float(np.linalg.norm(b1)) + eps
    b0 = b1 / nb
    Q = np.stack([a0, b0], axis=1)  # D x 2
    return Q

def gram_schmidt_clean(v_main, v_remove):
    v_remove = unit(v_remove)
    coeff = float(np.dot(v_main, v_remove))
    v_clean = v_main - coeff * v_remove
    return unit(v_clean), coeff

def ridge_direction_local(Z, y, center_z, radius=6.0, lam=1e-2, min_pts=2500):
    """
    Local ridge regression direction around center_z.
    Returns unit direction + count used.
    """
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    d = np.linalg.norm(Z - center_z[None, :], axis=1)
    idx = np.where(d <= radius)[0]
    if len(idx) < min_pts:
        idx = np.argsort(d)[:min_pts]

    X = Z[idx]
    t = y[idx]

    Xm = X.mean(axis=0, keepdims=True)
    Xs = X.std(axis=0, keepdims=True) + 1e-9
    Xn = (X - Xm) / Xs

    tm = float(t.mean())
    ts = float(t.std()) + 1e-9
    tn = (t - tm) / ts

    A = Xn.T @ Xn
    A.flat[::A.shape[0] + 1] += lam
    b = Xn.T @ tn
    w = np.linalg.solve(A, b)

    w = w / Xs.reshape(-1)
    return unit(w), int(len(idx))

def load_thresholds():
    with open(THRESH, "r") as f:
        d = json.load(f)
    return float(d["Tb"]), float(d["To"])

def choose_seed(mask, score):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None
    return int(idx[np.argmin(score[idx])])

def pick_anchor_seeds(between_pred, overlap_prob, lr_prob, Tb, To):
    b  = between_pred.reshape(-1)
    o  = overlap_prob.reshape(-1)
    lr = lr_prob.reshape(-1)

    LOWO = 0.35
    anchors = {}

    # 1) between boundary clear
    for eps in [0.005,0.01,0.02,0.04,0.06,0.08,0.10]:
        m = (np.abs(b - Tb) < eps) & (o < LOWO)
        s = np.abs(b - Tb) + 0.50*o
        j = choose_seed(m, s)
        if j is not None:
            anchors["between_boundary_clear"] = j
            break

    # 2) overlap boundary only-ish
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

def procrustes_R_2x2(QA, QB):
    """
    QA, QB: D x 2 orthonormal frames.
    Find R (2x2 orthogonal) minimizing ||QA R - QB||_F.
    """
    M = QA.T @ QB  # 2x2
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    # ensure det=+1 (proper rotation); if reflection slips in, fix
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R

def best_perm_and_signs(Qt, QB):
    """
    Qt, QB: D x 2 orthonormal frames. We want to match columns.
    Try perm in {identity, swap}. For each perm, choose signs for each column
    based on raw cosine sign. Score by sum(abs(cos)).
    Returns:
      perm_name, perm_idx (list), signs (list of +/-1),
      raw_cos (list), abs_cos (list), score
    """
    perms = [
        ("identity", [0, 1]),
        ("swap",     [1, 0]),
    ]

    best = None
    for name, p in perms:
        raw = []
        ab  = []
        signs = []
        score = 0.0
        for k in range(2):
            q = Qt[:, p[k]]
            b = QB[:, k]
            c = float(np.dot(unit(q), unit(b)))  # signed cosine
            raw.append(c)
            ab.append(abs(c))
            s = 1 if c >= 0 else -1
            signs.append(int(s))
            score += abs(c)
        cand = (score, name, p, signs, raw, ab)
        if best is None or cand[0] > best[0]:
            best = cand

    score, name, p, signs, raw, ab = best
    return name, p, signs, raw, ab, float(score)

def principal_angles_2d(QA, QB):
    """
    Principal angles between 2D subspaces spanned by QA and QB.
    Since QA, QB are D x 2 with orthonormal cols, cos(theta_i) are svd of QA^T QB.
    Returns mean cos, angles_deg list.
    """
    M = QA.T @ QB
    _, s, _ = np.linalg.svd(M)
    s = np.clip(s, 0.0, 1.0)
    angles = np.degrees(np.arccos(s))
    return float(np.mean(s)), [float(angles[0]), float(angles[1])]

# ----------------------------
# Main
# ----------------------------

def main():
    Tb, To = load_thresholds()

    Z = np.load(LATENTS)
    targets = np.load(TARGETS)
    preds   = np.load(PREDS)

    # supervised signals (use TRUE targets for ridge direction)
    between_true = targets["between_score"].reshape(-1)
    overlap_true = targets["overlap_any"].reshape(-1)
    tproj_true   = targets["t_on_BC"].reshape(-1)
    lr_true      = targets["lr_sign"].reshape(-1)

    # predictions only for picking anchors / metadata
    between_pred = preds["between_pred"].reshape(-1)
    overlap_prob = sigmoid_np(preds["overlap_logit"]).reshape(-1)
    lr_prob      = sigmoid_np(preds["lr_logit"]).reshape(-1)
    tproj_pred   = preds["tproj_pred"].reshape(-1)

    anchors = pick_anchor_seeds(between_pred, overlap_prob, lr_prob, Tb, To)
    if "between_boundary_clear" not in anchors:
        anchors["between_boundary_clear"] = 0

    # hyperparams (match your run)
    RADIUS = 6.0
    MINPTS = 2500
    LAM    = 0.01

    # compute local basis vectors for each anchor
    bases = {}
    meta  = {}

    for name, idx in anchors.items():
        z0 = Z[idx].astype(np.float64)

        vb, nb = ridge_direction_local(Z, between_true, z0, radius=RADIUS, lam=LAM, min_pts=MINPTS)
        vo, no = ridge_direction_local(Z, overlap_true, z0, radius=RADIUS, lam=LAM, min_pts=MINPTS)
        vt, nt = ridge_direction_local(Z, tproj_true,   z0, radius=RADIUS, lam=LAM, min_pts=MINPTS)
        vl, nl = ridge_direction_local(Z, lr_true,      z0, radius=RADIUS, lam=LAM, min_pts=MINPTS)

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

    # build frames for each anchor
    frames = {}
    for a in names:
        QA_bo = ortho2(bases[a]["v_between_clean"], bases[a]["v_overlap_clean"])  # D x 2
        QA_lt = ortho2(bases[a]["v_lr"],           bases[a]["v_tproj"])          # D x 2
        frames[a] = {"bo": QA_bo, "lt": QA_lt}

    # compute transport cache for ALL ordered pairs (A->B, A!=B)
    transport_maps = {}
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j:
                continue
            A = names[i]
            B = names[j]
            key = f"{A}__to__{B}"
            transport_maps[key] = {}

            for plane in ["bo", "lt"]:
                QA = frames[A][plane]
                QB = frames[B][plane]

                # Procrustes transport rotation
                R = procrustes_R_2x2(QA, QB)
                Qt = QA @ R

                # principal angles (subspace stability)
                sim, ang = principal_angles_2d(QA, QB)

                # best perm/sign to match columns
                perm_name, perm_idx, signs, raw_cos, abs_cos, score = best_perm_and_signs(Qt, QB)

                # also compute "cos after sign" (should be nonnegative)
                cos_after = [float(signs[k] * raw_cos[k]) for k in range(2)]

                transport_maps[key][plane] = {
                    "similarity": sim,
                    "angles_deg": ang,
                    "R": [[float(R[0,0]), float(R[0,1])],
                          [float(R[1,0]), float(R[1,1])]],
                    "column_permutation": perm_name,
                    "column_signs": [int(signs[0]), int(signs[1])],
                    "transported_col_cos_raw": [float(raw_cos[0]), float(raw_cos[1])],
                    "transported_col_abs_cos": [float(abs_cos[0]), float(abs_cos[1])],
                    "transported_col_cos_after_sign": [float(cos_after[0]), float(cos_after[1])],
                    "match_score_abs_sum": score,
                }

    out = {
        "Tb": Tb, "To": To,
        "radius": float(RADIUS),
        "min_pts": int(MINPTS),
        "lam": float(LAM),
        "anchors": meta,
        "transport_maps": transport_maps,
        "notes": [
            "This cache stores BOTH Procrustes R and the post-transport column permutation + sign flips.",
            "transported_col_cos_raw are SIGNED cosines between transported columns (after perm) and B columns (before sign fix).",
            "column_signs are chosen to flip each transported column so it aligns positively with B; cos_after_sign should be >= 0.",
            "If you expect sign flips, look for transported_col_cos_raw being negative and column_signs = -1.",
        ]
    }

    out_path = os.path.join(OUTDIR, "phase15b_transport_cache.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("[phase15b] saved ->", out_path)
    print(json.dumps({"anchors": {k:v["seed"] for k,v in meta.items()}}, indent=2))

if __name__ == "__main__":
    main()
