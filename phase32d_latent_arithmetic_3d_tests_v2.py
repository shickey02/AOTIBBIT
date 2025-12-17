#!/usr/bin/env python3
# phase32d_latent_arithmetic_3d_tests_v2.py
#
# Reports linear R2 AND retrieval-translate error in:
#  (A) raw latent space
#  (B) whitened latent space (z-scored + PCA-whiten)
#  (C) predicted-xyz space (fit xyz <- z, retrieve in that space)
#
# This makes retrieval benefit from improved linear structure.

import os, argparse
import numpy as np

def r2_score(y, yhat):
    y = np.asarray(y, np.float64)
    yhat = np.asarray(yhat, np.float64)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean(axis=0, keepdims=True)) ** 2) + 1e-12
    return 1.0 - ss_res / ss_tot

def fit_linear_ridge(X, Y, lam=1.0):
    # Solve (X^T X + lam I)W = X^T Y
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    XtX = X.T @ X
    XtX.flat[::XtX.shape[0] + 1] += lam
    W = np.linalg.solve(XtX, X.T @ Y)    # [D,3]
    b = Y.mean(axis=0) - X.mean(axis=0) @ W
    return W, b

def predict_linear(X, W, b):
    return X @ W + b

def pca_whiten_fit(X, eps=1e-6, max_dim=None):
    # z-score then PCA whiten
    mu = X.mean(axis=0, keepdims=True)
    sig = X.std(axis=0, keepdims=True) + eps
    Xn = (X - mu) / sig
    C = (Xn.T @ Xn) / max(1, Xn.shape[0]-1)
    # eig
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    if max_dim is not None:
        evals = evals[:max_dim]
        evecs = evecs[:, :max_dim]
    inv_sqrt = 1.0 / np.sqrt(evals + eps)
    # Whitening transform: Xw = Xn @ (evecs * inv_sqrt)
    T = evecs * inv_sqrt.reshape(1, -1)  # [D,k]
    return mu, sig, T

def pca_whiten_apply(X, mu, sig, T):
    Xn = (X - mu) / sig
    return Xn @ T

def knn_pick(query, Xcand):
    # returns index in Xcand of closest L2
    d = np.sum((Xcand - query) ** 2, axis=1)
    return int(np.argmin(d))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--pairs", type=int, default=60000)
    ap.add_argument("--cand", type=int, default=8000, help="candidate pool size for approximate retrieval")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ridge_lam", type=float, default=1.0)
    ap.add_argument("--whiten_dim", type=int, default=64, help="PCA dims for whitening (smaller=more stable)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    Z = np.load(args.latents).astype(np.float32)
    lab = np.load(args.labels)

    dxyz = np.stack([lab["dx"], lab["dy"], lab["dz"]], axis=1).astype(np.float32)  # [N,3]
    N, D = Z.shape
    print(f"N={N} D={D}")

    # --- Linear fits ---
    # xyz <- z
    W, b = fit_linear_ridge(Z, dxyz, lam=args.ridge_lam)
    dxyz_hat = predict_linear(Z, W, b)
    r2 = r2_score(dxyz, dxyz_hat)
    print(f"[linear latent->dxyz] R2_total={r2:.4f}")

    # delta_xyz <- delta_z
    M = min(args.pairs, N)
    i = rng.integers(0, N, size=M)
    j = rng.integers(0, N, size=M)
    dZ = (Z[j] - Z[i]).astype(np.float32)
    dD = (dxyz[j] - dxyz[i]).astype(np.float32)
    Wd, bd = fit_linear_ridge(dZ, dD, lam=args.ridge_lam)
    dD_hat = predict_linear(dZ, Wd, bd)
    r2d = r2_score(dD, dD_hat)
    print(f"[delta latent->delta_dxyz] R2_total={r2d:.4f}")

    # --- Retrieval translate tests (VECTORIZED) ---
    # approximate retrieval via random candidate pool (fast + consistent across runs)
    C = min(args.cand, N)
    cand = rng.integers(0, N, size=C)

    Zcand = Z[cand].astype(np.float32)
    Zwh_mu, Zwh_sig, Zwh_T = pca_whiten_fit(Z[cand], max_dim=min(args.whiten_dim, D))
    Zcand_w = pca_whiten_apply(Zcand, Zwh_mu, Zwh_sig, Zwh_T).astype(np.float32)

    Dcand = dxyz_hat[cand].astype(np.float32)  # predicted xyz space candidates

    K = min(args.pairs, N)
    a = rng.integers(0, N, size=K)      # source
    bidx = rng.integers(0, N, size=K)   # target for delta
    base = rng.integers(0, N, size=K)   # base to apply delta to

    deltaZ = (Z[bidx] - Z[a]).astype(np.float32)
    deltaD = (dxyz[bidx] - dxyz[a]).astype(np.float32)

    # Precompute norms for fast L2
    Zcand_T = Zcand.T
    Zcand_norm = np.sum(Zcand * Zcand, axis=1).astype(np.float32)

    Zcand_w_T = Zcand_w.T
    Zcand_w_norm = np.sum(Zcand_w * Zcand_w, axis=1).astype(np.float32)

    Dcand_T = Dcand.T
    Dcand_norm = np.sum(Dcand * Dcand, axis=1).astype(np.float32)

    def batched_knn_argmin(Q, C_T, C_norm, bs=256):
        # returns indices into candidate array (0..C-1) for each row of Q
        Q = Q.astype(np.float32, copy=False)
        out = np.empty((Q.shape[0],), dtype=np.int64)
        for i in range(0, Q.shape[0], bs):
            qb = Q[i:i+bs]
            qb_norm = np.sum(qb * qb, axis=1, keepdims=True)  # [B,1]
            # dist^2 = ||q||^2 + ||c||^2 - 2 q.c
            dots = qb @ C_T  # [B,C]
            d2 = qb_norm + C_norm.reshape(1, -1) - 2.0 * dots
            out[i:i+bs] = np.argmin(d2, axis=1)
        return out

    # mode A: raw latent translate (vectorized)
    print("[retrieval] mode=rawZ building queries...")
    Q_raw = (Z[base].astype(np.float32) + deltaZ).astype(np.float32)
    print("[retrieval] mode=rawZ knn...")
    pick_idx_raw = batched_knn_argmin(Q_raw, Zcand_T, Zcand_norm, bs=256)
    pick_raw = cand[pick_idx_raw]
    target = (dxyz[base] + deltaD).astype(np.float32)
    errs_raw = np.linalg.norm(dxyz[pick_raw] - target, axis=1)

    # mode B: whitened latent translate (vectorized)
    print("[retrieval] mode=whiten building queries...")
    Zbase_w = pca_whiten_apply(Z[base], Zwh_mu, Zwh_sig, Zwh_T).astype(np.float32)
    Ztrans_w = pca_whiten_apply(Z[base] + deltaZ, Zwh_mu, Zwh_sig, Zwh_T).astype(np.float32)
    Q_w = (Zbase_w + (Ztrans_w - Zbase_w)).astype(np.float32)  # same as Ztrans_w, kept explicit
    print("[retrieval] mode=whiten knn...")
    pick_idx_wh = batched_knn_argmin(Q_w, Zcand_w_T, Zcand_w_norm, bs=256)
    pick_wh = cand[pick_idx_wh]
    errs_wh = np.linalg.norm(dxyz[pick_wh] - target, axis=1)

    # mode C: predicted-xyz translate (vectorized, cheap)
    print("[retrieval] mode=predXYZ building queries...")
    Q_p = (dxyz_hat[base] + deltaD).astype(np.float32)
    print("[retrieval] mode=predXYZ knn...")
    pick_idx_p = batched_knn_argmin(Q_p, Dcand_T, Dcand_norm, bs=2048)
    pick_pred = cand[pick_idx_p]
    errs_pred = np.linalg.norm(dxyz[pick_pred] - target, axis=1)

    def stats(x):
        x = np.asarray(x, np.float32)
        return float(x.mean()), float(np.median(x)), float(np.quantile(x, 0.90))

    mr = stats(errs_raw)
    mw = stats(errs_wh)
    mp = stats(errs_pred)

    print(f"[retrieval translate | raw Z ] mean={mr[0]:.4f} median={mr[1]:.4f} p90={mr[2]:.4f}")
    print(f"[retrieval translate | whit] mean={mw[0]:.4f} median={mw[1]:.4f} p90={mw[2]:.4f}")
    print(f"[retrieval translate | pred] mean={mp[0]:.4f} median={mp[1]:.4f} p90={mp[2]:.4f}")


    report = [
        f"N={N} D={D}",
        f"[linear latent->dxyz] R2_total={r2:.6f}",
        f"[delta latent->delta_dxyz] R2_total={r2d:.6f}",
        f"[retrieval translate | raw Z ] mean={mr[0]:.6f} median={mr[1]:.6f} p90={mr[2]:.6f}",
        f"[retrieval translate | whit] mean={mw[0]:.6f} median={mw[1]:.6f} p90={mw[2]:.6f}",
        f"[retrieval translate | pred] mean={mp[0]:.6f} median={mp[1]:.6f} p90={mp[2]:.6f}",
    ]
    out_path = os.path.join(args.outdir, "report_v2.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(report) + "\n")
    np.savez(os.path.join(args.outdir, "linear_xyz_fit.npz"), W=W.astype(np.float32), b=b.astype(np.float32))
    print("[ok] wrote:", out_path)

if __name__ == "__main__":
    main()
