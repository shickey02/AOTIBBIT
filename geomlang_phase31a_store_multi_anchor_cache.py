#!/usr/bin/env python3
# geomlang_phase31a_store_multi_anchor_cache.py
#
# Phase 31A: Build a BIGGER transport_cache with multiple anchors per relation,
# including RIGHT / ABOVE / BELOW (shared latent space, more nodes).
#
# - Reuses Phase15F ridge + BO-clean + BO-removal for LR/Tproj, and frame transport.
# - Adds anchor types: right_clean, above_clean, below_clean
# - Supports K seeds per anchor with a simple diversity constraint (min latent distance).
#
# Outputs:
#   outputs_edges_relternary256_phase15/phase31a_transport_cache.json
#
# Deps: numpy only

import os, json, argparse
import numpy as np
from itertools import permutations, product

# ------------------- defaults -------------------
PHASE7_DIR = "outputs_edges_relternary256_phase7"
PHASE8_DIR = "outputs_edges_relternary256_phase8"
OUTDIR     = "outputs_edges_relternary256_phase15"
os.makedirs(OUTDIR, exist_ok=True)

LATENTS = os.path.join(PHASE7_DIR, "encoded_latents_seed123_N6000.npy")
TARGETS = os.path.join(PHASE7_DIR, "encoded_targets_seed123_N6000.npz")
PREDS   = os.path.join(PHASE7_DIR, "encoded_preds_seed123_N6000.npz")
THRESH  = os.path.join(PHASE8_DIR, "phase8_thresholds.json")


# ------------------- math utils -------------------
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
    Q = []
    D = cols[0].shape[0]
    for v in cols:
        w = np.array(v, dtype=np.float64)
        for q in Q:
            w = w - float(np.dot(w, q)) * q
        nw = float(np.linalg.norm(w))
        if nw < eps:
            if len(Q) == 0:
                e = np.zeros(D); e[int(np.argmax(np.abs(v)))] = 1.0
                w = e
            else:
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


# ------------------- transport helpers -------------------
def procrustes_R_k(QA, QB):
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
    k = C.shape[0]
    best = None
    for p in permutations(range(k)):
        Cp = C[:, p]
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


# ------------------- anchor selection (NEW) -------------------
def choose_topk_diverse(mask, score, Z, K, min_latent_dist):
    """Pick up to K indices satisfying mask, ranked by score (low is good),
    enforcing that chosen points are at least min_latent_dist apart in latent."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    order = idx[np.argsort(score[idx])]
    chosen = []
    for j in order:
        if len(chosen) >= K:
            break
        if len(chosen) == 0:
            chosen.append(int(j))
            continue
        # diversity check
        dz = np.linalg.norm(Z[j] - Z[np.array(chosen)], axis=1)
        if float(np.min(dz)) >= float(min_latent_dist):
            chosen.append(int(j))
    return chosen

def pick_anchor_seedsets(Z, between_pred, overlap_prob, lr_prob, tproj_pred, Tb, To, K, min_latent_dist):
    """
    Returns dict name->list[int] seeds.
    We keep your original 5 plus new: right_clean, above_clean, below_clean.
    """
    b  = between_pred.reshape(-1)
    o  = overlap_prob.reshape(-1)
    lr = lr_prob.reshape(-1)
    tp = tproj_pred.reshape(-1)

    LOWO = 0.95
    anchors = {}

    # --- originals (same spirit as 15f, but allow K seeds) ---
    # between_boundary_clear: near Tb, low overlap (15f-style eps sweep)
    anchors["between_boundary_clear"] = []
    for eps in [0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10]:
        m = (np.abs(b - Tb) < eps) & (o < LOWO)
        s = np.abs(b - Tb) + 0.50*o
        picks = choose_topk_diverse(m, s, Z, K, min_latent_dist)
        if len(picks) > 0:
            anchors["between_boundary_clear"] = picks
            break


    # overlap_boundary_only: near To, low between
    m = (np.abs(o - To) < 0.06) & (b < Tb - 0.06)
    s = np.abs(o - To) + 0.15*(Tb - b)
    anchors["overlap_boundary_only"] = choose_topk_diverse(m, s, Z, K, min_latent_dist)

    # between_deep_clear
    m = (b > 0.85) & (o < LOWO)
    s = -b + 0.5*o
    anchors["between_deep_clear"] = choose_topk_diverse(m, s, Z, K, min_latent_dist)

    # overlap_deep_only
    m = (o > 0.85) & (b < 0.30)
    s = -o + 0.25*b
    anchors["overlap_deep_only"] = choose_topk_diverse(m, s, Z, K, min_latent_dist)

    # left_clean
    m = (b < 0.30) & (o < LOWO) & (lr < 0.5)
    s = b + o + (lr)  # prefer strong "left" (low lr_prob)
    anchors["left_clean"] = choose_topk_diverse(m, s, Z, K, min_latent_dist)

    # --- NEW: right_clean ---
    m = (b < 0.30) & (o < LOWO) & (lr > 0.5)
    s = b + o + (1.0 - lr)  # prefer strong "right" (high lr_prob)
    anchors["right_clean"] = choose_topk_diverse(m, s, Z, K, min_latent_dist)

    # --- NEW: above_clean / below_clean ---
    # We treat tproj_pred as the "vertical" scalar. We don't assume sigmoid.
    # "above" = high tproj_pred, "below" = low tproj_pred, while staying clean of overlap/between.
    # These thresholds are adjustable; start with strong extremes.
    m = (b < 0.35) & (o < LOWO) & (tp > 0.85)
    s = (b + o) - tp
    anchors["above_clean"] = choose_topk_diverse(m, s, Z, K, min_latent_dist)

    m = (b < 0.35) & (o < LOWO) & (tp < 0.15)
    s = (b + o) + tp
    anchors["below_clean"] = choose_topk_diverse(m, s, Z, K, min_latent_dist)

    # If anything is empty, we keep it empty (script will skip).
    return anchors


# ------------------- main -------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=20, help="seeds per anchor type")
    ap.add_argument("--min_latent_dist", type=float, default=2.0, help="diversity constraint in latent")
    ap.add_argument("--radius", type=float, default=6.0)
    ap.add_argument("--min_pts", type=int, default=2500)
    ap.add_argument("--lam", type=float, default=0.01)
    ap.add_argument("--out", type=str, default=os.path.join(OUTDIR, "phase31a_transport_cache.json"))
    args = ap.parse_args()

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

    anchor_sets = pick_anchor_seedsets(
        Z, between_pred, overlap_prob, lr_prob, tproj_pred,
        Tb, To, K=args.K, min_latent_dist=args.min_latent_dist
    )

    # Flatten into unique node names like "left_clean__07"
    anchor_nodes = []
    for k, seeds in anchor_sets.items():
        for i, idx in enumerate(seeds):
            anchor_nodes.append((f"{k}__{i:02d}", k, int(idx)))

    if len(anchor_nodes) == 0:
        raise RuntimeError("No anchors selected. Loosen thresholds or reduce min_latent_dist.")

    RADIUS = float(args.radius)
    MINPTS = int(args.min_pts)
    LAM    = float(args.lam)

    bases = {}
    meta  = {}

    # build bases per node
    for node_name, anchor_type, idx in anchor_nodes:
        z0 = Z[idx].astype(np.float64)

        vb, nb = ridge_direction_local(Z, between_true, z0, radius=RADIUS, lam=LAM, min_pts=MINPTS)
        vo, no = ridge_direction_local(Z, overlap_true, z0, radius=RADIUS, lam=LAM, min_pts=MINPTS)
        vt, nt = ridge_direction_local(Z, tproj_true,   z0, radius=RADIUS, lam=LAM, min_pts=MINPTS)
        vl, nl = ridge_direction_local(Z, lr_true,      z0, radius=RADIUS, lam=LAM, min_pts=MINPTS)

        vb_clean, cb = gram_schmidt_clean(vb, vo)
        vo_clean, co = gram_schmidt_clean(vo, vb)

        def remove_plane(v, a, b):
            w = np.array(v, dtype=np.float64)
            w = w - float(np.dot(w, a)) * a
            w = w - float(np.dot(w, b)) * b
            return unit(w)

        vl_bo = remove_plane(vl, vb_clean, vo_clean)
        vt_bo = remove_plane(vt, vb_clean, vo_clean)

        vl_bo_clean, cl_on_t = gram_schmidt_clean(vl_bo, vt_bo)
        vt_bo_clean, ct_on_l = gram_schmidt_clean(vt_bo, vl_bo)

        bases[node_name] = {
            "v_between": vb,
            "v_overlap": vo,
            "v_tproj": vt,
            "v_lr": vl,
            "v_between_clean": vb_clean,
            "v_overlap_clean": vo_clean,
            "v_lr_bo_clean": vl_bo_clean,
            "v_tproj_bo_clean": vt_bo_clean,
        }

        meta[node_name] = {
            "seed": int(idx),
            "anchor_type": str(anchor_type),
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

    # frames per node
    frames = {}
    for a in names:
        QA_bo    = ortho_k([bases[a]["v_between_clean"], bases[a]["v_overlap_clean"]])
        QA_lt    = ortho_k([bases[a]["v_lr_bo_clean"],  bases[a]["v_tproj_bo_clean"]])
        QA_bo_lr = ortho_k([bases[a]["v_between_clean"], bases[a]["v_overlap_clean"], bases[a]["v_lr_bo_clean"]])
        frames[a] = {"bo": QA_bo, "lt": QA_lt, "bo_lr": QA_bo_lr}

    # transport maps for all pairs
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
                C = (Qt.T @ QB).astype(np.float64)
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

    # store vectors to JSON
    bases_out = {}
    for a in names:
        bases_out[a] = {k: [float(x) for x in unit(v)] for k, v in bases[a].items()}

    out = {
        "phase": "31a",
        "Tb": Tb, "To": To,
        "radius": float(RADIUS),
        "min_pts": int(MINPTS),
        "lam": float(LAM),
        "anchors": meta,
        "bases": bases_out,
        "transport_maps": transport_maps,
        "notes": [
            "31A expands anchors to include right_clean / above_clean / below_clean, with K seeds per type.",
            "Still uses 15F: BO-clean then remove BO components from LR/Tproj to define LT.",
            "Node names are like left_clean__07; meta[node]['anchor_type'] stores original group.",
        ],
        "anchor_sets_summary": {k: int(len(v)) for k, v in anchor_sets.items()},
    }

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("[phase31a] saved ->", args.out)
    print("[phase31a] nodes =", len(names))
    print("[phase31a] anchor_sets_summary =", out["anchor_sets_summary"])


if __name__ == "__main__":
    main()
