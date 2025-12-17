#!/usr/bin/env python3
# geomlang_phase15_transport_cache.py
#
# Phase 15: "store perm" (and more) — build a reusable transport cache.
#
# What this does:
#   1) Loads Phase7 exports (Z, targets, preds) + Phase8 thresholds (Tb, To)
#   2) Picks anchor seeds spanning regimes (between/overlap boundary + deep + left_clean)
#   3) Computes local factor frames at each anchor via local ridge regression:
#        - BO plane:  [between_clean, overlap_clean]   (Gram–Schmidt cleaned)
#        - LT plane:  [lr, tproj]
#   4) For every ordered anchor pair A->B, computes a Procrustes rotation R (2x2)
#      and ALSO computes/stores:
#        - column permutation ("identity" or "swap")
#        - column sign flips (+1/-1 per column)
#        - transported column abs-cosines (after perm+sign)
#      IMPORTANT: This computes perm+sign separately for BO and LT (no reuse bug).
#   5) Saves a single JSON cache you can reuse in later phases:
#        outputs_edges_relternary256_phase15/phase15_transport_cache.json
#      and an NPZ with the anchor frames:
#        outputs_edges_relternary256_phase15/phase15_anchor_frames.npz
#
# Notes:
# - No dependencies on any "phase11 direction cleaning" modules.
# - Pure numpy + json + stdlib.
#
# Run:
#   python bbit_geomlang/geomlang_phase15_transport_cache.py
#
# Then in later scripts you can "load perm" by reading the JSON:
#   cache["transport_maps"][f"{A}__to__{B}"]["bo"]["perm"], ["signs"], ["R"], ...

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

OUT_JSON = os.path.join(OUTDIR, "phase15_transport_cache.json")
OUT_NPZ  = os.path.join(OUTDIR, "phase15_anchor_frames.npz")

# -----------------------------
# Small utilities
# -----------------------------
def sigmoid_np(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))

def unit(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    if n < eps:
        return v
    return v / n

def orthonormalize2(u, v):
    """Return (q1,q2) as an orthonormal 2-frame in R^D, plus GS coefficients."""
    u = unit(u)
    proj = float(np.dot(v, u))
    v2 = v - proj * u
    v2 = unit(v2)
    return u, v2, proj

def gram_schmidt_clean(v_main, v_remove):
    v_remove = unit(v_remove)
    coeff = float(np.dot(v_main, v_remove))
    v_clean = v_main - coeff * v_remove
    return unit(v_clean), coeff

def load_thresholds():
    with open(THRESH, "r") as f:
        d = json.load(f)
    return float(d["Tb"]), float(d["To"])

def choose_seed(mask, score):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None
    return int(idx[np.argmin(score[idx])])

# -----------------------------
# Local direction estimator
# -----------------------------
def ridge_direction_local(Z, y, center_z, radius=6.0, lam=1e-2, min_pts=2500):
    """
    Fit ridge direction in a local neighborhood around center_z (Euclidean in latent space).
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

# -----------------------------
# Anchor selection
# -----------------------------
def pick_anchor_seeds(between_pred, overlap_prob, lr_prob, Tb, To):
    """
    Pick anchors spanning manifold regimes.
    """
    b = between_pred.reshape(-1)
    o = overlap_prob.reshape(-1)
    lr = lr_prob.reshape(-1)

    LOWO = 0.35

    anchors = {}

    # 1) between boundary clear (near Tb, low overlap)
    for eps in [0.003,0.005,0.008,0.012,0.02,0.03,0.05,0.07,0.10]:
        m = (np.abs(b - Tb) < eps) & (o < LOWO)
        s = np.abs(b - Tb) + 0.50*o
        j = choose_seed(m, s)
        if j is not None:
            anchors["between_boundary_clear"] = j
            break

    # 2) overlap boundary only-ish (near To, below Tb)
    m = (np.abs(o - To) < 0.06) & (b < Tb - 0.06)
    s = np.abs(o - To) + 0.15*(Tb - b)
    j = choose_seed(m, s)
    if j is not None:
        anchors["overlap_boundary_only"] = j

    # 3) deep between clear (high between, low overlap)
    m = (b > 0.85) & (o < LOWO)
    s = -b + 0.5*o
    j = choose_seed(m, s)
    if j is not None:
        anchors["between_deep_clear"] = j

    # 4) deep overlap only (high overlap, low between)
    m = (o > 0.85) & (b < 0.30)
    s = -o + 0.25*b
    j = choose_seed(m, s)
    if j is not None:
        anchors["overlap_deep_only"] = j

    # 5) left clean (low between, low overlap, lr<0.5)
    m = (b < 0.30) & (o < LOWO) & (lr < 0.5)
    s = b + o
    j = choose_seed(m, s)
    if j is not None:
        anchors["left_clean"] = j

    return anchors

# -----------------------------
# Procrustes transport + perm/sign storage
# -----------------------------
def procrustes_R(QA, QB):
    """
    Best-fit 2x2 rotation R so that QA @ R aligns to QB in least squares,
    where QA, QB are (D,2) orthonormal frames.
    """
    # Solve min ||QA R - QB||_F with R orthogonal => R = UV^T for SVD(QA^T QB)
    M = QA.T @ QB  # (2,2)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    # force det=+1 (proper rotation)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R

def best_column_match_abs_cos(Q_trans, QB):
    """
    Given transported frame Q_trans (D,2) and target frame QB (D,2),
    choose permutation + sign flips maximizing diagonal abs cosine.
    Returns:
      abs_cos_cols: [abs cos col0, abs cos col1] after perm+sign
      perm_name: "identity" or "swap"
      signs: [s0, s1] where each is +1 or -1 applied AFTER permutation
    """
    # Candidate perms: identity, swap
    perms = {
        "identity": np.array([0,1], dtype=int),
        "swap":     np.array([1,0], dtype=int),
    }

    best = None
    for pname, p in perms.items():
        Qt = Q_trans[:, p]  # permuted
        # compute cosines columnwise with QB columns 0,1
        c0 = float(np.dot(Qt[:,0], QB[:,0]))
        c1 = float(np.dot(Qt[:,1], QB[:,1]))
        # choose signs to maximize alignment (abs -> make positive)
        s0 = 1 if c0 >= 0 else -1
        s1 = 1 if c1 >= 0 else -1
        ac0 = abs(c0)
        ac1 = abs(c1)
        score = ac0 + ac1
        cand = (score, pname, [ac0, ac1], [int(s0), int(s1)])
        if (best is None) or (cand[0] > best[0]):
            best = cand

    _, pname, abs_cols, signs = best
    return abs_cols, pname, signs

def principal_angles_similarity(QA, QB):
    """
    Similarity = mean cos(theta_k) where theta_k are principal angles between subspaces.
    For 2D vs 2D: singular values of QA^T QB are cos(theta_1), cos(theta_2).
    """
    M = QA.T @ QB
    _, s, _ = np.linalg.svd(M)
    s = np.clip(s, 0.0, 1.0)
    angles = np.degrees(np.arccos(s))
    return float(np.mean(s)), [float(a) for a in angles]

# -----------------------------
# Main
# -----------------------------
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

    # You can tune these to match what you liked best
    RADIUS = 6.0
    MINPTS = 2500
    LAM    = 1e-2

    anchors = pick_anchor_seeds(between_pred, overlap_prob, lr_prob, Tb, To)
    if "between_boundary_clear" not in anchors:
        anchors["between_boundary_clear"] = 0  # hard fallback

    # Compute anchor frames
    meta = {}
    frames = {}  # store numpy frames for later reuse

    for name, idx in anchors.items():
        z0 = Z[idx].astype(np.float64)

        vb, nb = ridge_direction_local(Z, between_true, z0, radius=RADIUS, lam=LAM, min_pts=MINPTS)
        vo, no = ridge_direction_local(Z, overlap_true, z0, radius=RADIUS, lam=LAM, min_pts=MINPTS)
        vt, nt = ridge_direction_local(Z, tproj_true,   z0, radius=RADIUS, lam=LAM, min_pts=MINPTS)
        vl, nl = ridge_direction_local(Z, lr_true,      z0, radius=RADIUS, lam=LAM, min_pts=MINPTS)

        vb_clean, cb = gram_schmidt_clean(vb, vo)
        vo_clean, co = gram_schmidt_clean(vo, vb)

        # Build orthonormal 2-frames (D,2)
        bo_q1, bo_q2, bo_proj = orthonormalize2(vb_clean, vo_clean)
        lt_q1, lt_q2, lt_proj = orthonormalize2(vl, vt)

        Qbo = np.stack([bo_q1, bo_q2], axis=1)
        Qlt = np.stack([lt_q1, lt_q2], axis=1)

        frames[name] = {
            "Qbo": Qbo,
            "Qlt": Qlt,
        }

        meta[name] = {
            "seed": int(idx),
            "counts": {"between": nb, "overlap": no, "tproj": nt, "lr": nl},
            "clean_coeffs": {
                "between_on_overlap": float(cb),
                "overlap_on_between": float(co),
            },
            "signals_at_seed": {
                "between_pred": float(between_pred[idx]),
                "tproj_pred": float(tproj_pred[idx]),
                "overlap_prob": float(overlap_prob[idx]),
                "lr_prob": float(lr_prob[idx]),
            }
        }

    # Save frames NPZ (so later phases can load without recomputing)
    npz_dict = {}
    for name in frames.keys():
        npz_dict[f"{name}__Qbo"] = frames[name]["Qbo"].astype(np.float32)
        npz_dict[f"{name}__Qlt"] = frames[name]["Qlt"].astype(np.float32)
    np.savez(OUT_NPZ, **npz_dict)

    # Build transport maps for every ordered pair
    names = list(frames.keys())
    transport_maps = {}

    for a in names:
        for b in names:
            if a == b:
                continue

            Qbo_A = frames[a]["Qbo"]
            Qbo_B = frames[b]["Qbo"]
            Qlt_A = frames[a]["Qlt"]
            Qlt_B = frames[b]["Qlt"]

            # BO
            bo_sim, bo_angles = principal_angles_similarity(Qbo_A, Qbo_B)
            Rbo = procrustes_R(Qbo_A, Qbo_B)
            Qbo_trans = Qbo_A @ Rbo
            bo_cols, bo_perm, bo_signs = best_column_match_abs_cos(Qbo_trans, Qbo_B)

            # LT (computed independently — no reuse)
            lt_sim, lt_angles = principal_angles_similarity(Qlt_A, Qlt_B)
            Rlt = procrustes_R(Qlt_A, Qlt_B)
            Qlt_trans = Qlt_A @ Rlt
            lt_cols, lt_perm, lt_signs = best_column_match_abs_cos(Qlt_trans, Qlt_B)

            key = f"{a}__to__{b}"
            transport_maps[key] = {
                "bo": {
                    "similarity": float(bo_sim),
                    "angles_deg": bo_angles,
                    "R": Rbo.tolist(),
                    "transported_col_abs_cos": [float(x) for x in bo_cols],
                    "column_permutation": bo_perm,
                    "column_signs": bo_signs,  # +1/-1 per column after permutation
                },
                "lt": {
                    "similarity": float(lt_sim),
                    "angles_deg": lt_angles,
                    "R": Rlt.tolist(),
                    "transported_col_abs_cos": [float(x) for x in lt_cols],
                    "column_permutation": lt_perm,
                    "column_signs": lt_signs,
                }
            }

    out = {
        "Tb": Tb,
        "To": To,
        "radius": float(RADIUS),
        "min_pts": int(MINPTS),
        "lam": float(LAM),
        "anchors": meta,
        "transport_maps": transport_maps,
        "notes": [
            "This cache stores BOTH Procrustes R and the post-transport column permutation + sign flips.",
            "Use this to 'store perm': later phases can load the JSON and apply perm+sign consistently.",
            "Perm+sign are computed independently for BO and LT (no reuse).",
            "If column_permutation is always identity everywhere, that can be real — but if LT values mirror BO, it's a bug (this script prevents that).",
        ]
    }

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    print("[phase15] saved ->", OUT_JSON)
    print("[phase15] saved ->", OUT_NPZ)
    print(json.dumps({
        "Tb": Tb, "To": To,
        "anchors": {k: v["seed"] for k, v in meta.items()},
        "pairs": len(transport_maps),
    }, indent=2))

if __name__ == "__main__":
    main()
