#!/usr/bin/env python3
# phase32h_operator_group_tests_v2.py
#
# Operator "group-ish" tests on latent space:
#  - Mine operator prototypes (+x,-x,+y,-y,+z,-z) from random pairs that match
#    a target step in (dx,dy,dz) within tolerance.
#  - Test additivity / commutativity / cycle / step magnitude in:
#       (A) predicted xyz space via affine fit: xyz_hat = A z + b
#       (B) "snap to dataset" space: apply operator(s) in latent, then snap to
#           nearest real dataset latent (or nearest in predxyz space), and evaluate
#           achieved displacement in predicted xyz.
#
# Notes:
#  - Additivity/commutativity in predxyz are "tautological" for an affine map.
#    Snap-based tests are the real check.
#  - Writes UTF-8 report to avoid Windows cp1252 UnicodeEncodeError.

import os, argparse, math
import numpy as np

# optional sklearn (recommended)
try:
    from sklearn.neighbors import NearestNeighbors
except Exception:
    NearestNeighbors = None

# -----------------------------
# utils
# -----------------------------
def fit_affine(Z, XYZ):
    """
    Fit affine XYZ_hat = A Z + b via least squares.
    Z: [N,D], XYZ: [N,3]
    Returns A: [3,D], b: [3]
    """
    N, D = Z.shape
    X = np.concatenate([Z, np.ones((N, 1), dtype=np.float32)], axis=1)  # [N,D+1]
    # Solve X W ~= XYZ, W: [D+1,3]
    W, *_ = np.linalg.lstsq(X, XYZ, rcond=None)
    W = W.astype(np.float32)
    A = W[:D, :].T  # [3,D]
    b = W[D, :].astype(np.float32)  # [3]
    return A, b

def affine_predict(A, b, Z):
    # A: [3,D], Z: [...,D] -> [...,3]
    return (Z @ A.T) + b

def r2_score(y_true, y_pred):
    # y_true/y_pred: [N,3] or [N,k]
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2, axis=0) + 1e-12
    r2 = 1.0 - (ss_res / ss_tot)
    r2_total = float(np.mean(r2))
    return r2, r2_total

def corr(a, b):
    a = np.asarray(a).reshape(-1).astype(np.float64)
    b = np.asarray(b).reshape(-1).astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.sqrt((a*a).sum()) * np.sqrt((b*b).sum())) + 1e-12
    return float((a*b).sum() / denom)

def stats(x):
    x = np.asarray(x).reshape(-1)
    return dict(
        mean=float(np.mean(x)),
        med=float(np.median(x)),
        p90=float(np.percentile(x, 90)),
        p99=float(np.percentile(x, 99)),
        max=float(np.max(x)),
    )

def fmt_stats(st):
    return f"mean={st['mean']:.4f} med={st['med']:.4f} p90={st['p90']:.4f} p99={st['p99']:.4f} max={st['max']:.4f}"

def build_nn_index(points, metric="euclidean"):
    if NearestNeighbors is None:
        return None
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto", metric=metric)
    nn.fit(points)
    return nn

def nn_query(nn, points, queries):
    # returns indices into points
    if nn is None:
        # slow fallback: brute force
        # points: [M,d], queries: [Q,d]
        # compute argmin ||p-q||^2
        pts = points.astype(np.float64)
        q = queries.astype(np.float64)
        # (Q,M) = q^2 + p^2 -2 q p^T
        q2 = np.sum(q*q, axis=1, keepdims=True)
        p2 = np.sum(pts*pts, axis=1, keepdims=True).T
        dist2 = q2 + p2 - 2.0*(q @ pts.T)
        return np.argmin(dist2, axis=1)
    dists, idx = nn.kneighbors(queries, n_neighbors=1, return_distance=True)
    return idx.reshape(-1)

# -----------------------------
# mining operators
# -----------------------------
def mine_ops(Z, POS, step, tol, n_pairs, rng):
    """
    Mine operator deltas by sampling random i,j and selecting those where
    dpos approx equals (+step,0,0), (-step,0,0), etc within tol per-axis.
    Returns dict: op_name -> list of dz vectors (each [D])
    """
    N, D = Z.shape
    ops = {k: [] for k in ["+x","-x","+y","-y","+z","-z"]}

    # sample i,j
    i = rng.integers(0, N, size=n_pairs, endpoint=False)
    j = rng.integers(0, N, size=n_pairs, endpoint=False)
    # avoid i==j a bit (optional)
    same = (i == j)
    if np.any(same):
        j[same] = (j[same] + 1) % N

    dpos = (POS[j] - POS[i]).astype(np.float32)   # [P,3]
    dz   = (Z[j]   - Z[i]).astype(np.float32)     # [P,D]

    # axis-wise windows: target vector, and require other axes near 0
    def select(target):
        tx, ty, tz = target
        m = (
            (np.abs(dpos[:,0] - tx) <= tol) &
            (np.abs(dpos[:,1] - ty) <= tol) &
            (np.abs(dpos[:,2] - tz) <= tol)
        )
        return m

    masks = {
        "+x": select((+step, 0.0, 0.0)),
        "-x": select((-step, 0.0, 0.0)),
        "+y": select((0.0, +step, 0.0)),
        "-y": select((0.0, -step, 0.0)),
        "+z": select((0.0, 0.0, +step)),
        "-z": select((0.0, 0.0, -step)),
    }

    for k, m in masks.items():
        if np.any(m):
            ops[k] = dz[m].copy()  # store all samples
    return ops

def op_proto(ops):
    """
    ops: dict op_name -> array [K,D] (or empty list)
    returns dict op_name -> (count, mean_delta[D], dz_norm)
    """
    out = {}
    for k, arr in ops.items():
        if isinstance(arr, list) or (isinstance(arr, np.ndarray) and arr.size == 0):
            out[k] = (0, None, None)
            continue
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            out[k] = (0, None, None)
            continue
        mu = arr.mean(axis=0)
        out[k] = (arr.shape[0], mu, float(np.linalg.norm(mu)))
    return out

# -----------------------------
# evaluation
# -----------------------------
def apply_op(Z, dz):
    return Z + dz

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--step", type=float, default=0.35)
    ap.add_argument("--tol", type=float, default=0.12)

    ap.add_argument("--n_pairs", type=int, default=2500000)
    ap.add_argument("--n_eval", type=int, default=12000)

    ap.add_argument("--snap", action="store_true", help="snap applied latents back to nearest dataset point")
    ap.add_argument("--snap_metric", choices=["latent", "predxyz"], default="latent",
                    help="distance used for snapping: latent L2 or predicted-xyz L2")
    ap.add_argument("--cand", type=int, default=60000, help="candidate pool size for snapping (<=N)")
    ap.add_argument("--force_antisym", action="store_true",
                    help="force -ops to be negative of +ops (kills most cycle drift)")

    ap.add_argument("--zscore", action="store_true", help="zscore latents before everything (recommended if scales drift)")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    Z = np.load(args.latents).astype(np.float32)  # [N,D]
    lab = np.load(args.labels)
    dx = lab["dx"].astype(np.float32).reshape(-1)
    dy = lab["dy"].astype(np.float32).reshape(-1)
    dz = lab["dz"].astype(np.float32).reshape(-1)
    POS = np.stack([dx, dy, dz], axis=1).astype(np.float32)  # [N,3]

    N, D = Z.shape
    lines = []
    lines.append(f"PHASE 32H v2 — Operator group-ish tests")
    lines.append(f"N={N} D={D}  step={args.step} tol={args.tol}  n_pairs={args.n_pairs} n_eval={args.n_eval}  snap={args.snap} snap_metric={args.snap_metric} cand={args.cand}")
    lines.append(f"zscore={'ENABLED' if args.zscore else 'disabled'}  force_antisym={'YES' if args.force_antisym else 'no'}  seed={args.seed}")

    # zscore
    if args.zscore:
        mu = Z.mean(axis=0, keepdims=True)
        sd = Z.std(axis=0, keepdims=True) + 1e-6
        Z0 = (Z - mu) / sd
        np.savez(os.path.join(args.outdir, "zscore_latent_stats.npz"), mean=mu.astype(np.float32), std=sd.astype(np.float32))
        Z = Z0.astype(np.float32)
        lines.append("[zscore] wrote zscore_latent_stats.npz")

    # affine fit z->xyz
    A, b = fit_affine(Z, POS)
    POS_hat = affine_predict(A, b, Z)
    r2_dim, r2_total = r2_score(POS, POS_hat)
    lines.append(f"[affine z->xyz] R2_dim=({r2_dim[0]:.4f},{r2_dim[1]:.4f},{r2_dim[2]:.4f}) R2_total={r2_total:.4f}")

    # mine operators
    ops = mine_ops(Z, POS, step=args.step, tol=args.tol, n_pairs=args.n_pairs, rng=rng)
    proto = op_proto(ops)

    # optionally force antisymmetry
    if args.force_antisym:
        # if + exists, set - = -(+). If + missing but - exists, set + = -(-)
        for axis in ["x","y","z"]:
            p = f"+{axis}"
            m = f"-{axis}"
            cp, mup, _ = proto[p]
            cm, mum, _ = proto[m]
            if cp > 0 and mup is not None:
                proto[m] = (cp, -mup, float(np.linalg.norm(mup)))  # count mirrors
            elif cm > 0 and mum is not None:
                proto[p] = (cm, -mum, float(np.linalg.norm(mum)))

    lines.append("")
    lines.append("Operator prototypes (counts / ||dz||):")
    missing = []
    for k in ["+x","-x","+y","-y","+z","-z"]:
        c, mu, nrm = proto[k]
        if c <= 0 or mu is None:
            lines.append(f"  {k}: MISSING (count={c}) -> increase --n_pairs or loosen --tol")
            missing.append(k)
        else:
            lines.append(f"  {k}: count={c}  dz_norm={nrm:.4f}")

    if any(k in missing for k in ["+x","-x","+y","-y"]):
        lines.append("")
        lines.append("ERROR: missing required x/y operators. Try: --n_pairs 1200000 and/or --tol 0.08 (or bigger).")
        # still write report and exit
        with open(os.path.join(args.outdir, "report_v2.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines))
        print(f"[ok] wrote: {os.path.join(args.outdir,'report_v2.txt')}")
        return

    # helper to get dz vectors
    def dz_op(name):
        return proto[name][1].astype(np.float32)

    dxp = dz_op("+x"); dxm = dz_op("-x")
    dyp = dz_op("+y"); dym = dz_op("-y")
    dzp = dz_op("+z"); dzm = dz_op("-z")

    # evaluate set
    eval_idx = rng.choice(N, size=min(args.n_eval, N), replace=False)
    Z_eval = Z[eval_idx]
    POS_eval_hat = affine_predict(A, b, Z_eval)

    # ---------------- predxyz tests ----------------
    lines.append("")
    lines.append("=== Tests in predicted xyz (affine) ===")

    # additivity (+x then +y) vs (+x+y)
    Z_xy = apply_op(apply_op(Z_eval, dxp), dyp)
    Z_xpy = apply_op(Z_eval, dxp + dyp)
    diff_add = np.linalg.norm(affine_predict(A, b, Z_xy) - affine_predict(A, b, Z_xpy), axis=1)
    lines.append("Additivity: apply(+x then +y) vs apply(+(x+y) combined)")
    lines.append(f"  norm(diff): {fmt_stats(stats(diff_add))}")

    # commutativity (+x then +y) vs (+y then +x)
    Z_xy2 = apply_op(apply_op(Z_eval, dyp), dxp)
    diff_comm = np.linalg.norm(affine_predict(A, b, Z_xy) - affine_predict(A, b, Z_xy2), axis=1)
    lines.append("Commutativity: apply(+x then +y) vs apply(+y then +x)")
    lines.append(f"  norm(diff): {fmt_stats(stats(diff_comm))}")

    # cycle (+x then -x) vs identity in predicted xyz
    Z_cycle = apply_op(apply_op(Z_eval, dxp), dxm)
    diff_cyc = np.linalg.norm(affine_predict(A, b, Z_cycle) - POS_eval_hat, axis=1)
    lines.append("Cycle: apply(+x then -x) vs identity (in predicted xyz)")
    lines.append(f"  norm(diff): {fmt_stats(stats(diff_cyc))}")

    # step magnitude for +x (pred xyz displacement)
    dxyz_hat = affine_predict(A, b, apply_op(Z_eval, dxp)) - POS_eval_hat
    step_norm = np.linalg.norm(dxyz_hat, axis=1)
    lines.append("Step magnitude consistency for +x (predicted xyz displacement)")
    lines.append(f"  norm(dxyz_hat): {fmt_stats(stats(step_norm))}")
    lines.append(f"  target step={args.step}")

    # metric correlation on random pairs (use a smaller sample)
    Pm = min(200000, N * 3)
    ii = rng.integers(0, N, size=Pm, endpoint=False)
    jj = rng.integers(0, N, size=Pm, endpoint=False)
    jj[ii == jj] = (jj[ii == jj] + 1) % N
    dz_rand = np.linalg.norm(Z[jj] - Z[ii], axis=1)
    dxyz_rand = np.linalg.norm(POS[jj] - POS[ii], axis=1)
    lines.append("")
    lines.append("Metric correlation on random pairs: corr(||dz||, ||dxyz||)")
    lines.append(f"  corr={corr(dz_rand, dxyz_rand):.4f}")

    # ---------------- SNAP tests ----------------
    if args.snap:
        lines.append("")
        lines.append("=== Tests with SNAP to dataset ===")

        # candidate pool for snapping
        cand = int(min(max(1000, args.cand), N))
        cand_idx = rng.choice(N, size=cand, replace=False)
        Z_cand = Z[cand_idx]
        POS_cand_hat = affine_predict(A, b, Z_cand)

        if args.snap_metric == "latent":
            nn = build_nn_index(Z_cand)
            snap_space = "latent"
        else:
            nn = build_nn_index(POS_cand_hat)
            snap_space = "predxyz"

        lines.append(f"[snap] using {snap_space} distance; candidates={cand} (sklearn={'YES' if NearestNeighbors is not None else 'NO(bruteforce)'})")

        def snap_latents(Zq):
            if args.snap_metric == "latent":
                idx0 = nn_query(nn, Z_cand, Zq)
            else:
                pq = affine_predict(A, b, Zq)
                idx0 = nn_query(nn, POS_cand_hat, pq)
            # map back to full dataset index
            snapped_full_idx = cand_idx[idx0]
            return Z[snapped_full_idx], snapped_full_idx

        # additivity snap: snap(apply(apply(z,+x),+y)) vs snap(apply(z,+x+y))
        Z1, _ = snap_latents(apply_op(apply_op(Z_eval, dxp), dyp))
        Z2, _ = snap_latents(apply_op(Z_eval, dxp + dyp))
        diff_add_s = np.linalg.norm(affine_predict(A, b, Z1) - affine_predict(A, b, Z2), axis=1)
        lines.append("Additivity (snap): snap(+x then +y) vs snap(+(x+y))")
        lines.append(f"  norm(diff): {fmt_stats(stats(diff_add_s))}")

        # commutativity snap
        Z1c, _ = snap_latents(apply_op(apply_op(Z_eval, dxp), dyp))
        Z2c, _ = snap_latents(apply_op(apply_op(Z_eval, dyp), dxp))
        diff_comm_s = np.linalg.norm(affine_predict(A, b, Z1c) - affine_predict(A, b, Z2c), axis=1)
        lines.append("Commutativity (snap): snap(+x then +y) vs snap(+y then +x)")
        lines.append(f"  norm(diff): {fmt_stats(stats(diff_comm_s))}")

        # cycle snap: snap(+x then -x) vs original z snapped-to-itself (identity)
        Zcy, _ = snap_latents(apply_op(apply_op(Z_eval, dxp), dxm))
        # identity reference: snap the original Z_eval too (so we're comparing on same snapped manifold)
        Zid, _ = snap_latents(Z_eval)
        diff_cyc_s = np.linalg.norm(affine_predict(A, b, Zcy) - affine_predict(A, b, Zid), axis=1)
        lines.append("Cycle (snap): snap(+x then -x) vs snap(identity)")
        lines.append(f"  norm(diff): {fmt_stats(stats(diff_cyc_s))}")

        # step magnitude snap: achieved displacement after snap(+x)
        Zsx, _ = snap_latents(apply_op(Z_eval, dxp))
        dxyz_s = affine_predict(A, b, Zsx) - affine_predict(A, b, Zid)
        step_s = np.linalg.norm(dxyz_s, axis=1)
        lines.append("Step magnitude (+x) after snap (predxyz displacement)")
        lines.append(f"  norm(dxyz): {fmt_stats(stats(step_s))}")
        lines.append(f"  target step={args.step}")

        # direction consistency: cosine similarity between intended (+x) direction and achieved dxyz
        # Intended direction unit in predxyz is A*dxp
        intended = (A @ dxp).astype(np.float32)  # [3]
        intended_norm = float(np.linalg.norm(intended) + 1e-12)
        intended_u = intended / intended_norm

        d = dxyz_s.astype(np.float32)
        dnorm = np.linalg.norm(d, axis=1) + 1e-12
        cos = (d @ intended_u) / dnorm
        lines.append("Direction consistency (+x) after snap (cosine with intended A*dz)")
        lines.append(f"  cos: {fmt_stats(stats(cos))}")

    # write report
    report_path = os.path.join(args.outdir, "report_v2.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"[ok] wrote: {report_path}")

if __name__ == "__main__":
    main()
