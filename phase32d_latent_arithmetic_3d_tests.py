#!/usr/bin/env python3
# phase32d_latent_arithmetic_3d_tests.py
#
# Tests whether frozen latents support 3D arithmetic / translation consistency.
#
# Inputs:
#   --latents latents.npy [N,D]
#   --labels  labels.npz (must contain dx,dy,dz,dist AND optionally x1,y1,z1,x2,y2,z2)
#
# Outputs (outdir):
#   report.txt
#   delta_fit.npz (linear map from dz -> dxyz)
#
import os, argparse
import numpy as np

def r2_score(y, yhat, eps=1e-12):
    y = np.asarray(y); yhat = np.asarray(yhat)
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean(axis=0, keepdims=True))**2) + eps
    return 1.0 - ss_res/ss_tot

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--pairs", type=int, default=40000, help="random pairs for delta tests")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k_retrieve", type=int, default=1, help="nearest neighbor k (1 is fine)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    Z = np.load(args.latents).astype(np.float32)   # [N,D]
    lab = np.load(args.labels)

    # We need dxyz per sample.
    # phase32a labels likely store dx,dy,dz directly.
    if not all(k in lab for k in ["dx","dy","dz"]):
        raise ValueError("labels.npz must contain dx,dy,dz")
    dxyz = np.stack([lab["dx"], lab["dy"], lab["dz"]], axis=1).astype(np.float32)  # [N,3]

    N, D = Z.shape
    # center/whiten latents for stable linear regression
    Zm = Z.mean(axis=0, keepdims=True)
    Zc = Z - Zm
    Zs = Zc.std(axis=0, keepdims=True) + 1e-6
    Zw = Zc / Zs

    # -----------------------
    # 1) Fit linear map: latent -> dxyz (simple baseline)
    # -----------------------
    # ridge regression: W = (X^T X + lam I)^-1 X^T Y
    lam = 1e-2
    X = Zw
    Y = dxyz
    XtX = X.T @ X
    XtX.flat[::D+1] += lam
    W = np.linalg.solve(XtX, X.T @ Y)     # [D,3]
    Yhat = X @ W
    r2 = r2_score(Y, Yhat)

    # -----------------------
    # 2) Delta consistency: do latent differences correspond to dxyz differences?
    # Pick random i,j -> deltaZ; compare to delta(dxyz)
    # Fit linear map from deltaZ -> delta(dxyz)
    # -----------------------
    P = args.pairs
    ii = rng.integers(0, N, size=P)
    jj = rng.integers(0, N, size=P)
    dz = (Zw[jj] - Zw[ii]).astype(np.float32)          # [P,D]
    dd = (dxyz[jj] - dxyz[ii]).astype(np.float32)      # [P,3]

    # Fit delta map with ridge
    XtX2 = dz.T @ dz
    XtX2.flat[::D+1] += lam
    Wd = np.linalg.solve(XtX2, dz.T @ dd)              # [D,3]
    ddhat = dz @ Wd
    r2_delta = r2_score(dd, ddhat)

    # -----------------------
    # 3) Retrieval test: apply a desired translation in (dx,dy,dz) space.
    # We use Wd^T as "decoder" from desired dd -> latent delta via least squares:
    # Solve for v: (dz @ Wd) ≈ dd  => want dz that yields dd. We'll compute
    # dz = dd @ (Wd^T (Wd Wd^T)^-1)  (minimum-norm in feature space)
    # Then apply to a random anchor and NN-retrieve.
    # -----------------------
    # pseudo-inverse in 3D:
    A = (Wd.T @ Wd)  # [3,3]
    Ainv = np.linalg.inv(A + 1e-6*np.eye(3))
    # map desired dd -> latent delta (in whitened latent space)
    # dd: [M,3] -> dz: [M,D]
    def dd_to_dz(dd_vec):
        # dd_vec shape (3,)
        g = (Wd @ (Ainv @ dd_vec.reshape(3,1))).reshape(D)  # [D]
        return g.astype(np.float32)

    # build NN index (brute force is ok for small M; N=60k is still fine for M~200)
    def nn1(query):
        # query shape [D]
        # return idx of closest in Zw
        diff = Zw - query[None,:]
        dist2 = np.sum(diff*diff, axis=1)
        return int(np.argmin(dist2))

    M = 200
    # choose random starting samples
    anchors = rng.integers(0, N, size=M)
    # choose random desired translations in dxyz-space
    # (scale this to your dataset typical magnitude; start small)
    dd_targets = rng.normal(0, 0.5, size=(M,3)).astype(np.float32)

    errs = []
    for a, dd_t in zip(anchors, dd_targets):
        zq = Zw[a]
        dz_t = dd_to_dz(dd_t)
        zq2 = zq + dz_t
        b = nn1(zq2)
        dd_real = (dxyz[b] - dxyz[a])
        errs.append(np.linalg.norm(dd_real - dd_t))

    errs = np.asarray(errs, dtype=np.float32)
    report = []
    report.append(f"N={N} D={D}")
    report.append(f"[linear latent->dxyz] R2_total={r2:.4f}")
    report.append(f"[delta latent->delta_dxyz] R2_total={r2_delta:.4f}")
    report.append(f"[retrieval translate] mean_err={errs.mean():.4f} median_err={np.median(errs):.4f} p90_err={np.quantile(errs,0.90):.4f}")
    txt = "\n".join(report) + "\n"
    print(txt)

    with open(os.path.join(args.outdir, "report.txt"), "w") as f:
        f.write(txt)

    np.savez(os.path.join(args.outdir, "delta_fit.npz"),
             Z_mean=Zm.astype(np.float32),
             Z_std=Zs.astype(np.float32),
             W_latent_to_dxyz=W.astype(np.float32),
             W_delta_to_delta_dxyz=Wd.astype(np.float32),
             lam=np.float32(lam))

    print("[ok] wrote report.txt and delta_fit.npz")

if __name__ == "__main__":
    main()
