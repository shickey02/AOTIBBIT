#!/usr/bin/env python3
# geomlang_phase13_direction_transport_v2.py
#
# Improvements:
# - Fix sign of each local direction so proj correlates positively with the target in the local neighborhood.
# - Report BOTH signed cosine and abs(cos) (abs is the right “global-axis similarity” metric).
# - Save local vectors per anchor to .npz so Phase13B can decode TRUE local factor walks.

import os, json
import numpy as np

PHASE7_DIR = "outputs_edges_relternary256_phase7"
PHASE8_DIR = "outputs_edges_relternary256_phase8"
OUTDIR     = "outputs_edges_relternary256_phase13"
os.makedirs(OUTDIR, exist_ok=True)

LATENTS = os.path.join(PHASE7_DIR, "encoded_latents_seed123_N6000.npy")
TARGETS = os.path.join(PHASE7_DIR, "encoded_targets_seed123_N6000.npz")
PREDS   = os.path.join(PHASE7_DIR, "encoded_preds_seed123_N6000.npz")
THRESH  = os.path.join(PHASE8_DIR, "phase8_thresholds.json")

def sigmoid_np(x):
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-x))

def unit(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n

def cos(a,b):
    return float(np.dot(unit(a), unit(b)))

def gram_schmidt_clean(v_main, v_remove):
    v_remove = unit(v_remove)
    coeff = float(np.dot(v_main, v_remove))
    v_clean = v_main - coeff * v_remove
    return unit(v_clean), coeff

def load_thresholds():
    with open(THRESH, "r") as f:
        d = json.load(f)
    return float(d["Tb"]), float(d["To"])

def ridge_direction_local(Z, y, center_z, radius=3.0, lam=1e-3, min_pts=900):
    """
    Ridge direction in local neighborhood around center_z.
    Returns: unit w, idx used.
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

    tm = t.mean()
    ts = t.std() + 1e-9
    tn = (t - tm) / ts

    A = Xn.T @ Xn
    A.flat[::A.shape[0] + 1] += lam
    b = Xn.T @ tn
    w = np.linalg.solve(A, b)

    # unstandardize
    w = w / Xs.reshape(-1)
    return unit(w), idx

def orient_sign_by_corr(Z, y, idx, center_z, v):
    """
    Choose sign so that projection correlates positively with y in neighborhood.
    """
    X = Z[idx]
    t = y[idx].reshape(-1)

    proj = (X - center_z[None, :]) @ v
    # corr sign via covariance
    cov = float(np.mean((proj - proj.mean()) * (t - t.mean())))
    if cov < 0:
        return -v, -1
    return v, +1

def choose_seed(mask, score):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None
    return int(idx[np.argmin(score[idx])])

def pick_anchor_seeds(between_pred, overlap_prob, lr_prob, Tb, To):
    b = between_pred.reshape(-1)
    o = overlap_prob.reshape(-1)
    lr = lr_prob.reshape(-1)

    LOWO = 0.35

    anchors = {}

    # between boundary clear
    for eps in [0.005,0.01,0.02,0.04,0.06,0.08,0.10]:
        m = (np.abs(b - Tb) < eps) & (o < LOWO)
        s = np.abs(b - Tb) + 0.50*o
        j = choose_seed(m, s)
        if j is not None:
            anchors["between_boundary_clear"] = j
            break

    # overlap boundary (only-ish)
    m = (np.abs(o - To) < 0.06) & (b < Tb - 0.06)
    s = np.abs(o - To) + 0.15*(Tb - b)
    j = choose_seed(m, s)
    if j is not None:
        anchors["overlap_boundary_only"] = j

    # deep between clear
    m = (b > 0.85) & (o < LOWO)
    s = -b + 0.5*o
    j = choose_seed(m, s)
    if j is not None:
        anchors["between_deep_clear"] = j

    # deep overlap only
    m = (o > 0.85) & (b < 0.30)
    s = -o + 0.25*b
    j = choose_seed(m, s)
    if j is not None:
        anchors["overlap_deep_only"] = j

    # left clean
    m = (b < 0.30) & (o < LOWO) & (lr < 0.5)
    s = b + o
    j = choose_seed(m, s)
    if j is not None:
        anchors["left_clean"] = j

    return anchors

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

    # more stable neighborhood than v1
    RADIUS = 3.0
    MINPTS = 900
    LAM    = 1e-3

    bases = {}
    meta  = {}

    for name, idx in anchors.items():
        z0 = Z[idx].astype(np.float64)

        vb, ib = ridge_direction_local(Z, between_true, z0, radius=RADIUS, min_pts=MINPTS, lam=LAM)
        vo, io = ridge_direction_local(Z, overlap_true, z0, radius=RADIUS, min_pts=MINPTS, lam=LAM)
        vt, it = ridge_direction_local(Z, tproj_true,   z0, radius=RADIUS, min_pts=MINPTS, lam=LAM)
        vl, il = ridge_direction_local(Z, lr_true,      z0, radius=RADIUS, min_pts=MINPTS, lam=LAM)

        vb, sb = orient_sign_by_corr(Z, between_true, ib, z0, vb)
        vo, so = orient_sign_by_corr(Z, overlap_true, io, z0, vo)
        vt, st = orient_sign_by_corr(Z, tproj_true,   it, z0, vt)
        vl, sl = orient_sign_by_corr(Z, lr_true,      il, z0, vl)

        vb_clean, cb = gram_schmidt_clean(vb, vo)
        vo_clean, co = gram_schmidt_clean(vo, vb)

        bases[name] = {
            "v_between": vb,
            "v_overlap": vo,
            "v_tproj": vt,
            "v_lr": vl,
            "v_between_clean": vb_clean,
            "v_overlap_clean": vo_clean,
        }

        meta[name] = {
            "seed": int(idx),
            "counts": {"between": int(len(ib)), "overlap": int(len(io)), "tproj": int(len(it)), "lr": int(len(il))},
            "signs": {"between": int(sb), "overlap": int(so), "tproj": int(st), "lr": int(sl)},
            "clean_coeffs": {"between_on_overlap": float(cb), "overlap_on_between": float(co)},
            "signals_at_seed": {
                "between_pred": float(between_pred[idx]),
                "tproj_pred": float(tproj_pred[idx]),
                "overlap_prob": float(overlap_prob[idx]),
                "lr_prob": float(lr_prob[idx]),
            }
        }

    names = list(bases.keys())
    align = {}
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a = names[i]; b = names[j]
            align[f"{a}__vs__{b}"] = {
                "cos_between_clean": cos(bases[a]["v_between_clean"], bases[b]["v_between_clean"]),
                "abs_between_clean": abs(cos(bases[a]["v_between_clean"], bases[b]["v_between_clean"])),
                "cos_overlap_clean": cos(bases[a]["v_overlap_clean"], bases[b]["v_overlap_clean"]),
                "abs_overlap_clean": abs(cos(bases[a]["v_overlap_clean"], bases[b]["v_overlap_clean"])),
                "cos_lr":            cos(bases[a]["v_lr"],           bases[b]["v_lr"]),
                "abs_lr":            abs(cos(bases[a]["v_lr"],       bases[b]["v_lr"])),
                "cos_tproj":         cos(bases[a]["v_tproj"],        bases[b]["v_tproj"]),
                "abs_tproj":         abs(cos(bases[a]["v_tproj"],    bases[b]["v_tproj"])),
            }

    # save vectors
    vec_path = os.path.join(OUTDIR, "phase13_local_vectors.npz")
    npz_kwargs = {}
    for name in bases:
        for k, v in bases[name].items():
            npz_kwargs[f"{name}__{k}"] = v.astype(np.float32)
    np.savez(vec_path, **npz_kwargs)

    out = {
        "Tb": Tb, "To": To,
        "radius": RADIUS, "min_pts": MINPTS, "lam": LAM,
        "anchors": meta,
        "pairwise_alignment": align,
        "vectors_npz": vec_path,
        "interpretation_notes": [
            "Use abs_* cosines to judge global consistency (direction sign is arbitrary).",
            "If abs_between_clean stays high across anchors, 'between' is a global axis.",
            "If abs_between_clean drops low across anchors even after sign-fixing & stabilization, that indicates real twisting/curvature."
        ]
    }

    out_path = os.path.join(OUTDIR, "phase13_transport_report_v2.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("[phase13A-v2] saved ->", out_path)
    print("[phase13A-v2] saved vectors ->", vec_path)

if __name__ == "__main__":
    main()
