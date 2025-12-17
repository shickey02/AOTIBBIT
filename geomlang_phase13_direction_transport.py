#!/usr/bin/env python3
# geomlang_phase13_direction_transport.py
#
# Phase 13A: Direction transport / twist check.
# Compute local directions v_between, v_overlap, v_lr, v_tproj at multiple anchor seeds,
# Gram-Schmidt clean (between/overlap), then compare alignment across anchors.
#
# Outputs JSON with:
# - chosen anchors (boundary + interior)
# - pairwise cos similarities between local bases
# - how much "between" and "overlap" twist across regions

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

def ridge_direction_local(Z, y, center_z, radius=2.0, lam=1e-3, min_pts=400):
    """
    Fit ridge direction in a local neighborhood around center_z (Euclidean in latent space).
    Returns unit direction + count used.
    """
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    d = np.linalg.norm(Z - center_z[None, :], axis=1)
    idx = np.where(d <= radius)[0]
    if len(idx) < min_pts:
        # fallback: take nearest K
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
    """
    Pick 5 anchors spanning manifold regimes.
    """
    b = between_pred.reshape(-1)
    o = overlap_prob.reshape(-1)
    lr = lr_prob.reshape(-1)

    LOWO = 0.35
    HIO  = 0.65

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

def cos(a,b):
    return float(np.dot(unit(a), unit(b)))

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
        anchors["between_boundary_clear"] = 0  # hard fallback

    # local bases per anchor
    bases = {}
    meta  = {}

    # neighborhood radius: adjust if you want (smaller = more local, more twist)
    RADIUS = 2.0
    MINPTS = 450

    for name, idx in anchors.items():
        z0 = Z[idx].astype(np.float64)

        vb, nb = ridge_direction_local(Z, between_true, z0, radius=RADIUS, min_pts=MINPTS)
        vo, no = ridge_direction_local(Z, overlap_true, z0, radius=RADIUS, min_pts=MINPTS)
        vt, nt = ridge_direction_local(Z, tproj_true,   z0, radius=RADIUS, min_pts=MINPTS)
        vl, nl = ridge_direction_local(Z, lr_true,      z0, radius=RADIUS, min_pts=MINPTS)

        vb_clean, cb = gram_schmidt_clean(vb, vo)
        vo_clean, co = gram_schmidt_clean(vo, vb)

        bases[name] = {
            "v_between": vb, "v_overlap": vo, "v_tproj": vt, "v_lr": vl,
            "v_between_clean": vb_clean, "v_overlap_clean": vo_clean,
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

    # pairwise alignment report
    names = list(bases.keys())
    align = {}
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a = names[i]; b = names[j]
            align[f"{a}__vs__{b}"] = {
                "cos_between_clean": cos(bases[a]["v_between_clean"], bases[b]["v_between_clean"]),
                "cos_overlap_clean": cos(bases[a]["v_overlap_clean"], bases[b]["v_overlap_clean"]),
                "cos_lr":            cos(bases[a]["v_lr"],           bases[b]["v_lr"]),
                "cos_tproj":         cos(bases[a]["v_tproj"],        bases[b]["v_tproj"]),
            }

    out = {
        "Tb": Tb, "To": To,
        "radius": RADIUS, "min_pts": MINPTS,
        "anchors": meta,
        "pairwise_alignment": align,
        "interpretation_notes": [
            "Cosine ~1 means the direction is globally consistent between those two regions.",
            "If between_clean or overlap_clean cosines drop far below 1, that factor twists across the manifold (curvature / transport effects).",
            "Try smaller radius (e.g., 1.0) to emphasize local geometry; larger radius to emphasize global direction."
        ]
    }

    out_path = os.path.join(OUTDIR, "phase13_transport_report.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("[phase13A] saved ->", out_path)
    print(json.dumps({"anchors": {k:v["seed"] for k,v in meta.items()}}, indent=2))

if __name__ == "__main__":
    main()
