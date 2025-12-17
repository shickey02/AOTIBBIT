#!/usr/bin/env python3
# geomlang_phase15f_store_3d_frames_and_3x3_cos.py
#
# Phase 15F: store FULL anchor direction vectors + build a TRUE 3D frame per anchor,
# then compute anchor-to-anchor 3x3 signed cosine matrices (with best perm+sign match).
#
# Why 15F:
# - Your 15E shows BO is fairly coherent, but LT is weak/unstable (diag cos often ~0.1–0.3).
# - Next step toward “3-D”: lock BO as a plane, then extract a THIRD axis (LR residual)
#   orthogonal to that plane and see if it transports consistently.
#
# Inputs:
#   outputs_edges_relternary256_phase7/encoded_latents_seed123_N6000.npy
#   outputs_edges_relternary256_phase7/encoded_targets_seed123_N6000.npz
#   outputs_edges_relternary256_phase7/encoded_preds_seed123_N6000.npz
#   outputs_edges_relternary256_phase8/phase8_thresholds.json
#
# Outputs:
#   outputs_edges_relternary256_phase15/phase15f_transport_cache.json
#
# Notes:
# - Builds a 3D frame per anchor: [between_clean, overlap_clean, lr_perp_to_BO] then ortho.
# - Also keeps your existing BO + LT frames.
# - Stores vectors (so you can do real 3D viz next without recomputing).
#
# Deps: numpy only

import os, json
import numpy as np
from itertools import permutations, product

PHASE7_DIR = "outputs_edges_relternary256_phase7"
PHASE8_DIR = "outputs_edges_relternary256_phase8"
OUTDIR     = "outputs_edges_relternary256_phase15"
os.makedirs(OUTDIR, exist_ok=True)

LATENTS = os.path.join(PHASE7_DIR, "encoded_latents_seed123_N6000.npy")
TARGETS = os.path.join(PHASE7_DIR, "encoded_targets_seed123_N6000.npz")
PREDS   = os.path.join(PHASE7_DIR, "encoded_preds_seed123_N6000.npz")
THRESH  = os.path.join(PHASE8_DIR, "phase8_thresholds.json")

def sigmoid_np(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))

def unit(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < eps:
        return v
    return v / n

def ortho_k(cols, eps=1e-12):
    """
    cols: list of D-vectors. returns Q: D x k (orthonormal)
    simple Gram-Schmidt with fallback.
    """
    Q = []
    D = cols[0].shape[0]
    for i, v in enumerate(cols):
        w = np.array(v, dtype=np.float64)
        for q in Q:
            w = w - float(np.dot(w, q)) * q
        nw = float(np.linalg.norm(w))
        if nw < eps:
            # fallback: pick basis vector least aligned with existing Q
            if len(Q) == 0:
                e = np.zeros(D); e[int(np.argmax(np.abs(v)))] = 1.0
                w = e
            else:
                # choose coordinate axis with minimal projection on existing Q
                best = None
                for j in range(D):
                    e = np.zeros(D); e[j] = 1.0
                    proj = 0.0
                    for q in Q:
                        proj += abs(float(np.dot(e, q)))
                    if best is None or proj < best[0]:
                        best = (proj, e)
                w = best[1]
                for q in Q:
                    w = w - float(np.dot(w, q)) * q
                nw = float(np.linalg.norm(w)) + eps
        Q.append(w / (nw + eps))
    return np.stack(Q, axis=1)  # D x k

def gram_schmidt_clean(v_main, v_remove):
    v_remove = unit(v_remove)
    coeff = float(np.dot(v_main, v_remove))
    v_clean = v_main - coeff * v_remove
    return unit(v_clean), coeff

def ridge_direction_local(Z, y, center_z, radius=6.0, lam=1e-2, min_pts=2500):
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

    for eps in [0.005,0.01,0.02,0.04,0.06,0.08,0.10]:
        m = (np.abs(b - Tb) < eps) & (o < LOWO)
        s = np.abs(b - Tb) + 0.50*o
        j = choose_seed(m, s)
        if j is not None:
            anchors["between_boundary_clear"] = j
            break

    m = (np.abs(o - To) < 0.06) & (b < Tb - 0.06)
    s = np.abs(o - To) + 0.15*(Tb - b)
    j = choose_seed(m, s)
    if j is not None:
        anchors["overlap_boundary_only"] = j

    m = (b > 0.85) & (o < LOWO)
    s = -b + 0.5*o
    j = choose_seed(m, s)
    if j is not None:
        anchors["between_deep_clear"] = j

    m = (o > 0.85) & (b < 0.30)
    s = -o + 0.25*b
    j = choose_seed(m, s)
    if j is not None:
        anchors["overlap_deep_only"] = j

    m = (b < 0.30) & (o < LOWO) & (lr < 0.5)
    s = b + o
    j = choose_seed(m, s)
    if j is not None:
        anchors["left_clean"] = j

    return anchors

def procrustes_R_k(QA, QB):
    """
    QA,QB: D x k orthonormal.
    Return k x k rotation aligning QA to QB in least squares.
    """
    M = QA.T @ QB
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R

def principal_cosines(QA, QB):
    M = QA.T @ QB
    _, s, _ = np.linalg.svd(M)
    s = np.clip(s, 0.0, 1.0)
    return float(np.mean(s)), s.tolist()

def best_perm_signs_from_C_k(C):
    """
    C is k x k (signed). Choose permutation of columns + per-column sign flips
    to maximize sum(abs(diagonal)).
    Returns best mapping plus diag values.
    """
    k = C.shape[0]
    best = None
    for p in permutations(range(k)):
        Cp = C[:, p]  # permute columns
        for signs in product([-1, 1], repeat=k):
            S = np.diag(signs)
            Cps = Cp @ S
            diag = np.diag(Cps)
            score = float(np.sum(np.abs(diag)))
            cand = (score, p, list(signs), diag.tolist())
            if best is None or cand[0] > best[0]:
                best = cand
    score, p, signs, diag = best
    return {
        "column_permutation_idx": [int(x) for x in p],
        "column_signs": [int(s) for s in signs],
        "diag_after_perm_sign": [float(x) for x in diag],
        "match_score_abs_diag_sum": float(score),
    }

def main():
    Tb, To = load_thresholds()

    Z = np.load(LATENTS)
    targets = np.load(TARGETS)
    preds   = np.load(PREDS)

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

    RADIUS = 6.0
    MINPTS = 2500
    LAM    = 0.01

    bases = {}
    meta  = {}

    for name, idx in anchors.items():
        z0 = Z[idx].astype(np.float64)

        vb, nb = ridge_direction_local(Z, between_true, z0, radius=RADIUS, lam=LAM, min_pts=MINPTS)
        vo, no = ridge_direction_local(Z, overlap_true, z0, radius=RADIUS, lam=LAM, min_pts=MINPTS)
        vt, nt = ridge_direction_local(Z, tproj_true,   z0, radius=RADIUS, lam=LAM, min_pts=MINPTS)
        vl, nl = ridge_direction_local(Z, lr_true,      z0, radius=RADIUS, lam=LAM, min_pts=MINPTS)

        # --- Clean BO pair (as before) ---
        vb_clean, cb = gram_schmidt_clean(vb, vo)
        vo_clean, co = gram_schmidt_clean(vo, vb)

        # --- NEW: make LR/Tproj perpendicular to BO plane ---
        # Remove components along vb_clean and vo_clean
        def remove_plane(v, a, b):
            w = np.array(v, dtype=np.float64)
            w = w - float(np.dot(w, a)) * a
            w = w - float(np.dot(w, b)) * b
            return unit(w)

        vl_bo = remove_plane(vl, vb_clean, vo_clean)
        vt_bo = remove_plane(vt, vb_clean, vo_clean)

        # Then mutually clean within that residual subspace
        vl_bo_clean, cl_on_t = gram_schmidt_clean(vl_bo, vt_bo)
        vt_bo_clean, ct_on_l = gram_schmidt_clean(vt_bo, vl_bo)

        bases[name] = {
            "v_between": vb,
            "v_overlap": vo,
            "v_tproj": vt,
            "v_lr": vl,
            "v_between_clean": vb_clean,
            "v_overlap_clean": vo_clean,
            "v_lr_bo_clean": vl_bo_clean,
            "v_tproj_bo_clean": vt_bo_clean,
        }

        meta[name] = {
            "seed": int(idx),
            "counts": {"between": nb, "overlap": no, "tproj": nt, "lr": nl},
            "clean_coeffs": {
                "between_on_overlap": cb,
                "overlap_on_between": co,
                "lr_on_tproj": cl_on_t,
                "tproj_on_lr": ct_on_l,
            },
            "signals_at_seed": {
                "between_pred": float(between_pred[idx]),
                "tproj_pred": float(tproj_pred[idx]),
                "overlap_prob": float(overlap_prob[idx]),
                "lr_prob": float(lr_prob[idx]),
            }
        }

    names = list(bases.keys())

    # Build orthonormal frames for:
    # - bo: [between_clean, overlap_clean]
    # - lt: [lr_bo_clean, tproj_bo_clean]   (LT after BO-removal)
    # - bo_lr: 3D frame [between_clean, overlap_clean, lr_bo_clean]
    frames = {}
    for a in names:
        QA_bo    = ortho_k([bases[a]["v_between_clean"], bases[a]["v_overlap_clean"]])
        QA_lt    = ortho_k([bases[a]["v_lr_bo_clean"],  bases[a]["v_tproj_bo_clean"]])
        QA_bo_lr = ortho_k([bases[a]["v_between_clean"], bases[a]["v_overlap_clean"], bases[a]["v_lr_bo_clean"]])
        frames[a] = {"bo": QA_bo, "lt": QA_lt, "bo_lr": QA_bo_lr}

    transport_maps = {}

    for A in names:
        for B in names:
            if A == B:
                continue
            key = f"{A}__to__{B}"
            transport_maps[key] = {}

            for plane in ["bo", "lt", "bo_lr"]:
                QA = frames[A][plane]
                QB = frames[B][plane]

                k = QA.shape[1]
                R = procrustes_R_k(QA, QB)
                Qt = QA @ R

                sim, princ = principal_cosines(QA, QB)

                C = (Qt.T @ QB).astype(np.float64)  # k x k signed
                C_abs = np.abs(C)

                match = best_perm_signs_from_C_k(C)

                transport_maps[key][plane] = {
                    "k": int(k),
                    "similarity": float(sim),
                    "principal_cosines": [float(x) for x in princ],
                    "R": [[float(R[i, j]) for j in range(k)] for i in range(k)],
                    "coskxk_raw": [[float(C[i, j]) for j in range(k)] for i in range(k)],
                    "coskxk_abs": [[float(C_abs[i, j]) for j in range(k)] for i in range(k)],
                    **match,
                }

    # Store the actual vectors (small N of anchors, so JSON is fine)
    bases_out = {}
    for a in names:
        bases_out[a] = {k: [float(x) for x in unit(v)] for k, v in bases[a].items()}

    out = {
        "phase": "15f",
        "Tb": Tb, "To": To,
        "radius": float(RADIUS),
        "min_pts": int(MINPTS),
        "lam": float(LAM),
        "anchors": meta,
        "bases": bases_out,
        "transport_maps": transport_maps,
        "notes": [
            "15F removes BO components from LR and Tproj first (so LT is truly 'orthogonal complement'-ish).",
            "Adds bo_lr 3D frame = [between_clean, overlap_clean, lr_bo_clean] (orthonormalized).",
            "coskxk_raw = (Qt^T QB) signed; bo_lr gives a 3x3 matrix.",
            "best perm+sign maximizes sum(abs(diagonal)) (k=2 or k=3).",
        ]
    }

    out_path = os.path.join(OUTDIR, "phase15f_transport_cache.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("[phase15f] saved ->", out_path)
    print(json.dumps({"anchors": {k:v["seed"] for k,v in meta.items()}}, indent=2))

if __name__ == "__main__":
    main()
