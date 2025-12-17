#!/usr/bin/env python3
# geomlang_phase15_parallel_transport.py
#
# Phase 15: Parallel transport / connection estimate for factor subspaces.
# - Build local BO frame (between_clean, overlap_clean) and LT frame (lr, tproj) at anchors
# - Compute best-fit transport rotation between frames via Procrustes (SVD)
# - Report:
#    * principal angles (already)
#    * transport rotation matrices R_BO, R_LT
#    * transported vector alignment: cos( Q_A[:,k] -> Q_B R[:,k] ) ~ 1
#
# Output JSON: outputs_edges_relternary256_phase15/phase15_transport_maps.json

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

# knobs (match your Phase 14 run)
RADIUS = 6.0
MINPTS = 2500
LAM    = 1e-2

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

def best_column_match_abs_cos(Qa_trans, Qb):
    """
    Return best matching between columns of Qa_trans and Qb in 2D,
    allowing swap + sign flips (abs cos).
    """
    a0, a1 = Qa_trans[:,0], Qa_trans[:,1]
    b0, b1 = Qb[:,0], Qb[:,1]

    # match = [(a0->b0, a1->b1), (a0->b1, a1->b0)]
    m00 = abs_cos(a0, b0); m11 = abs_cos(a1, b1)
    m01 = abs_cos(a0, b1); m10 = abs_cos(a1, b0)

    if (m00 + m11) >= (m01 + m10):
        return [float(m00), float(m11)], "identity"
    else:
        return [float(m01), float(m10)], "swap"


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

    tm = t.mean()
    ts = t.std() + 1e-9
    tn = (t - tm) / ts

    A = Xn.T @ Xn
    A.flat[::A.shape[0] + 1] += lam
    b = Xn.T @ tn

    w = np.linalg.solve(A, b)
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

def orthonormal_frame(v1, v2, eps=1e-12):
    a = unit(v1)
    b = np.asarray(v2, dtype=np.float64)
    b = b - np.dot(b, a) * a
    nb = np.linalg.norm(b)
    if nb < eps:
        k = int(np.argmin(np.abs(a)))
        e = np.zeros_like(a); e[k] = 1.0
        b = e - np.dot(e, a) * a
        nb = np.linalg.norm(b) + eps
    b = b / nb
    Q = np.stack([a, b], axis=1)  # (D,2)
    return Q

def principal_angles_deg(Q1, Q2):
    M = Q1.T @ Q2
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    ang = np.degrees(np.arccos(s))
    sim = float(np.mean(s))
    return sim, ang.tolist(), M

def procrustes_rotation(M):
    """
    Given M = Q1^T Q2 (k x k), find R s.t. Q1 R ~ Q2.
    R = UV^T where M = U S V^T
    """
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    # enforce det +1 for proper rotation (avoid reflection)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R

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

    # local frames at each anchor
    frames = {}
    meta = {}

    for name, idx in anchors.items():
        z0 = Z[idx].astype(np.float64)

        vb, nb = ridge_direction_local(Z, between_true, z0, radius=RADIUS, min_pts=MINPTS, lam=LAM)
        vo, no = ridge_direction_local(Z, overlap_true, z0, radius=RADIUS, min_pts=MINPTS, lam=LAM)
        vt, nt = ridge_direction_local(Z, tproj_true,   z0, radius=RADIUS, min_pts=MINPTS, lam=LAM)
        vl, nl = ridge_direction_local(Z, lr_true,      z0, radius=RADIUS, min_pts=MINPTS, lam=LAM)

        vb_clean, cb = gram_schmidt_clean(vb, vo)
        vo_clean, co = gram_schmidt_clean(vo, vb)

        Qbo = orthonormal_frame(vb_clean, vo_clean)
        Qlt = orthonormal_frame(vl, vt)

        frames[name] = {"Qbo": Qbo, "Qlt": Qlt}

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

    names = list(frames.keys())
    maps = {}

    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a = names[i]; b = names[j]

            # BO
            sim_bo, ang_bo, Mbo = principal_angles_deg(frames[a]["Qbo"], frames[b]["Qbo"])
            Rbo = procrustes_rotation(Mbo)  # 2x2
            # transported frame: Qbo_a @ Rbo should align with Qbo_b
            Qbo_trans = frames[a]["Qbo"] @ Rbo
            # column-wise alignment (sign-invariant)
            cols, perm = best_column_match_abs_cos(Qbo_trans, frames[b]["Qbo"])
            col0, col1 = cols
            # LT
            sim_lt, ang_lt, Mlt = principal_angles_deg(frames[a]["Qlt"], frames[b]["Qlt"])
            Rlt = procrustes_rotation(Mlt)
            Qlt_trans = frames[a]["Qlt"] @ Rlt
            lcol0 = abs_cos(Qlt_trans[:,0], frames[b]["Qlt"][:,0])
            lcol1 = abs_cos(Qlt_trans[:,1], frames[b]["Qlt"][:,1])

            maps[f"{a}__to__{b}"] = {
                "bo": {
                    "similarity": float(sim_bo),
                    "angles_deg": ang_bo,
                    "R": Rbo.tolist(),
                    "transported_col_abs_cos": [col0, col1],
                    "column_permutation": perm

                },
                "lt": {
                    "similarity": float(sim_lt),
                    "angles_deg": ang_lt,
                    "R": Rlt.tolist(),
                    "transported_col_abs_cos": [col0, col1],
                    "column_permutation": perm
                }
            }

    out = {
        "Tb": Tb, "To": To,
        "radius": RADIUS, "min_pts": MINPTS, "lam": LAM,
        "anchors": meta,
        "transport_maps": maps,
        "notes": [
            "R is the best-fit rotation that transports frame at A into frame at B (Procrustes).",
            "transported_col_abs_cos near 1.0 means the transported basis vectors match B's basis up to sign.",
            "If subspace similarity is moderate but transported_col_abs_cos is high, the factor exists but twists; transport corrects it."
        ]
    }

    out_path = os.path.join(OUTDIR, "phase15_transport_maps.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("[phase15] saved ->", out_path)
    print(json.dumps({"anchors": {k:v["seed"] for k,v in meta.items()}}, indent=2))

if __name__ == "__main__":
    main()
