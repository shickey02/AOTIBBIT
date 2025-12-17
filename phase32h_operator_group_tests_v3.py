#!/usr/bin/env python3
# phase32h_operator_group_tests_v3.py
#
# Operator "group-ish" tests in latent space with optional SNAP-to-dataset.
# Key fix: prefer SNAP in predicted-xyz space (predxyz), not raw latent distance.
#
# Adds:
#  - UTF-8 report writing (fixes Windows cp1252 UnicodeEncodeError)
#  - --snap_k (query KNN with k>1)
#  - --snap_exclude_self (avoid trivial self-snap)
#  - --snap_min_step (reject snaps that don't actually move)
#  - --snap_metric {predxyz,latent}
#
# Inputs:
#   --latents latents.npy [N,D]
#   --labels  labels.npz with dx,dy,dz (and possibly dist, left/above/front/ov2d)
#
# Output:
#   outdir/report_v3.txt

import os, argparse, math
import numpy as np

def fit_affine(Z, XYZ):
    """
    Fit XYZ ~= Z @ A + b  (A: D x 3, b: 1 x 3)
    Returns A, b, predXYZ, R2_dim, R2_total
    """
    Z = Z.astype(np.float32)
    XYZ = XYZ.astype(np.float32)
    N, D = Z.shape

    X = np.concatenate([Z, np.ones((N, 1), dtype=np.float32)], axis=1)  # [N, D+1]
    # Solve least squares: X @ W ~= XYZ, W is (D+1, 3)
    W, *_ = np.linalg.lstsq(X, XYZ, rcond=None)
    A = W[:-1, :]              # [D,3]
    b = W[-1:, :]              # [1,3]
    pred = X @ W               # [N,3]

    # R2 per dim
    r2_dim = []
    for k in range(3):
        y = XYZ[:, k]
        yhat = pred[:, k]
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-9
        r2_dim.append(1.0 - ss_res / ss_tot)

    # "total" R2 on concatenated dims
    ss_res = float(np.sum((XYZ - pred) ** 2))
    ss_tot = float(np.sum((XYZ - XYZ.mean(axis=0, keepdims=True)) ** 2)) + 1e-9
    r2_total = 1.0 - ss_res / ss_tot
    return A.astype(np.float32), b.astype(np.float32), pred.astype(np.float32), tuple(r2_dim), float(r2_total)

def build_knn(points, use_sklearn=True):
    """
    points: [N,M]
    Returns a query function knn(q, k) -> (dist, idx)
    """
    points = points.astype(np.float32)

    if use_sklearn:
        try:
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean")
            nn.fit(points)
            def query(q, k=1):
                q = np.asarray(q, dtype=np.float32)
                if q.ndim == 1:
                    q2 = q[None, :]
                else:
                    q2 = q
                nn.n_neighbors = int(k)
                d, idx = nn.kneighbors(q2, n_neighbors=int(k), return_distance=True)
                return d, idx
            return query, True
        except Exception:
            pass

    # fallback brute-force
    def query(q, k=1):
        q = np.asarray(q, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        # compute squared distances
        # (N,M) - (B,M) -> (B,N)
        d2 = np.sum((points[None, :, :] - q[:, None, :]) ** 2, axis=2)
        idx = np.argsort(d2, axis=1)[:, :k]
        d = np.take_along_axis(np.sqrt(d2 + 1e-9), idx, axis=1)
        return d, idx
    return query, False

def mine_operator_deltas(Z, POS, step, tol, n_pairs, seed):
    """
    Randomly sample pairs (i,j), compute dpos = POS[j]-POS[i], and if dpos is close
    to (+step,0,0), (-step,0,0), etc. then collect dz = Z[j]-Z[i].
    Return dict op_name -> mean_dz, counts
    """
    rng = np.random.default_rng(seed)
    N, D = Z.shape

    # targets in xyz space
    targets = {
        "+x": np.array([ step, 0.0, 0.0], dtype=np.float32),
        "-x": np.array([-step, 0.0, 0.0], dtype=np.float32),
        "+y": np.array([0.0,  step, 0.0], dtype=np.float32),
        "-y": np.array([0.0, -step, 0.0], dtype=np.float32),
        "+z": np.array([0.0, 0.0,  step], dtype=np.float32),
        "-z": np.array([0.0, 0.0, -step], dtype=np.float32),
    }

    acc = {k: [] for k in targets.keys()}
    counts = {k: 0 for k in targets.keys()}

    # sample indices
    ii = rng.integers(0, N, size=(n_pairs,), dtype=np.int64)
    jj = rng.integers(0, N, size=(n_pairs,), dtype=np.int64)

    for i, j in zip(ii, jj):
        if i == j:
            continue
        dpos = POS[j] - POS[i]  # [3]
        # quick reject if too big overall
        # but we want near one axis target
        for name, t in targets.items():
            if np.linalg.norm(dpos - t) <= tol:
                dz = Z[j] - Z[i]
                acc[name].append(dz.astype(np.float32))
                counts[name] += 1

    ops = {}
    for name, lst in acc.items():
        if len(lst) > 0:
            ops[name] = np.mean(np.stack(lst, axis=0), axis=0).astype(np.float32)
        else:
            ops[name] = None

    return ops, counts

def apply_op(z, dz):
    return (z + dz).astype(np.float32)

def cosine(a, b):
    na = float(np.linalg.norm(a)) + 1e-9
    nb = float(np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / (na * nb))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--step", type=float, default=0.35)
    ap.add_argument("--tol", type=float, default=0.12)
    ap.add_argument("--n_pairs", type=int, default=2_500_000)
    ap.add_argument("--n_eval", type=int, default=12_000)

    ap.add_argument("--zscore", action="store_true", help="z-score latents before mining/testing (usually leave off for finetuned)")
    ap.add_argument("--force_antisym", action="store_true", help="force -axis op = - (+axis op)")

    # snapping
    ap.add_argument("--snap", action="store_true")
    ap.add_argument("--snap_metric", choices=["predxyz", "latent"], default="predxyz")
    ap.add_argument("--cand", type=int, default=60000, help="candidate pool size (<=N); 60000 means all")
    ap.add_argument("--snap_k", type=int, default=50, help="KNN K for snapping (k>1 helps avoid self-snap)")
    ap.add_argument("--snap_exclude_self", action="store_true")
    ap.add_argument("--snap_min_step", type=float, default=0.10, help="reject snapped moves smaller than this in predxyz norm")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    Z = np.load(args.latents).astype(np.float32)
    lab = np.load(args.labels)

    dx = lab["dx"].astype(np.float32).reshape(-1)
    dy = lab["dy"].astype(np.float32).reshape(-1)
    dz = lab["dz"].astype(np.float32).reshape(-1)
    POS = np.stack([dx, dy, dz], axis=1).astype(np.float32)  # [N,3]

    N, D = Z.shape
    print(f"N={N} D={D}")

    if args.zscore:
        mu = Z.mean(axis=0, keepdims=True)
        sd = Z.std(axis=0, keepdims=True) + 1e-6
        Z = (Z - mu) / sd
        print("[zscore] enabled")
    else:
        print("[zscore] disabled")

    # affine map z->xyz (for predxyz metric + sanity)
    A, b, predXYZ, r2_dim, r2_total = fit_affine(Z, POS)
    print(f"[affine z->xyz] R2_dim=({r2_dim[0]:.4f},{r2_dim[1]:.4f},{r2_dim[2]:.4f}) R2_total={r2_total:.4f}")

    # mine operator prototypes
    ops, counts = mine_operator_deltas(Z, POS, step=float(args.step), tol=float(args.tol),
                                       n_pairs=int(args.n_pairs), seed=int(args.seed))

    # enforce antisym if requested
    if args.force_antisym:
        for ax in ["x", "y", "z"]:
            p = ops.get(f"+{ax}", None)
            if p is not None:
                ops[f"-{ax}"] = (-p).astype(np.float32)

    # report op norms
    lines = []
    lines.append("PHASE 32H v3 — Operator group-ish tests")
    lines.append(f"N={N} D={D}  step={args.step} tol={args.tol}  n_pairs={args.n_pairs} n_eval={args.n_eval}  snap={bool(args.snap)} snap_metric={args.snap_metric} cand={args.cand}")
    lines.append(f"zscore={'YES' if args.zscore else 'NO'}  force_antisym={'YES' if args.force_antisym else 'NO'}  seed={args.seed}")
    lines.append(f"[affine z->xyz] R2_dim=({r2_dim[0]:.4f},{r2_dim[1]:.4f},{r2_dim[2]:.4f}) R2_total={r2_total:.4f}")
    lines.append("")
    lines.append("Operator prototypes (counts / ||dz||):")
    for name in ["+x","-x","+y","-y","+z","-z"]:
        if ops.get(name, None) is None:
            lines.append(f"  {name}: MISSING (count={counts.get(name,0)})")
        else:
            lines.append(f"  {name}: count={counts.get(name,0)}  dz_norm={float(np.linalg.norm(ops[name])):.4f}")

    # check required ops
    required = ["+x","-x","+y","-y","+z","-z"]
    missing = [k for k in required if ops.get(k, None) is None]
    if missing:
        lines.append("")
        lines.append("ERROR: missing required operators: " + ",".join(missing))
        lines.append("Try increasing --n_pairs and/or loosening --tol")
        # still write report
        outpath = os.path.join(args.outdir, "report_v3.txt")
        with open(outpath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print("[ok] wrote:", outpath)
        return

    # evaluation seeds
    eval_idx = rng.integers(0, N, size=(int(args.n_eval),), dtype=np.int64)

    def z_to_predxyz(z):
        return (z @ A + b.reshape(3,)).astype(np.float32)

    # === tests in predicted xyz (affine) ===
    lines.append("")
    lines.append("=== Tests in predicted xyz (affine) ===")
    # additivity + commutativity in predicted xyz always looks perfect for affine, but keep for logging
    diffs_add = []
    diffs_comm = []
    diffs_cycle = []
    step_mags = []

    dx_op = ops["+x"]; dy_op = ops["+y"]; nx_op = ops["-x"]

    for i in eval_idx:
        z0 = Z[i]
        p0 = z_to_predxyz(z0)

        p_xy = z_to_predxyz(apply_op(apply_op(z0, dx_op), dy_op))
        p_xpy = z_to_predxyz(apply_op(z0, (dx_op + dy_op)))

        diffs_add.append(float(np.linalg.norm(p_xy - p_xpy)))

        p_yx = z_to_predxyz(apply_op(apply_op(z0, dy_op), dx_op))
        diffs_comm.append(float(np.linalg.norm(p_xy - p_yx)))

        p_cycle = z_to_predxyz(apply_op(apply_op(z0, dx_op), nx_op))
        diffs_cycle.append(float(np.linalg.norm(p_cycle - p0)))

        p_step = z_to_predxyz(apply_op(z0, dx_op))
        step_mags.append(float(np.linalg.norm(p_step - p0)))

    def stats(arr):
        arr = np.asarray(arr, dtype=np.float32)
        return dict(
            mean=float(np.mean(arr)),
            med=float(np.median(arr)),
            p90=float(np.percentile(arr, 90)),
            p99=float(np.percentile(arr, 99)),
            mx=float(np.max(arr)),
        )

    s_add = stats(diffs_add)
    s_com = stats(diffs_comm)
    s_cyc = stats(diffs_cycle)
    s_mag = stats(step_mags)

    lines.append("Additivity: apply(+x then +y) vs apply(+(x+y) combined)")
    lines.append(f"  norm(diff): mean={s_add['mean']:.4f} med={s_add['med']:.4f} p90={s_add['p90']:.4f} p99={s_add['p99']:.4f} max={s_add['mx']:.4f}")
    lines.append("Commutativity: apply(+x then +y) vs apply(+y then +x)")
    lines.append(f"  norm(diff): mean={s_com['mean']:.4f} med={s_com['med']:.4f} p90={s_com['p90']:.4f} p99={s_com['p99']:.4f} max={s_com['mx']:.4f}")
    lines.append("Cycle: apply(+x then -x) vs identity (in predicted xyz)")
    lines.append(f"  norm(diff): mean={s_cyc['mean']:.4f} med={s_cyc['med']:.4f} p90={s_cyc['p90']:.4f} p99={s_cyc['p99']:.4f} max={s_cyc['mx']:.4f}")
    lines.append("Step magnitude consistency for +x (predicted xyz displacement)")
    lines.append(f"  norm(dxyz_hat): mean={s_mag['mean']:.4f} med={s_mag['med']:.4f} p90={s_mag['p90']:.4f} p99={s_mag['p99']:.4f} max={s_mag['mx']:.4f}")
    lines.append(f"  target step={args.step:.4f}")

    # metric correlation on random pairs
    rp = min(100000, N)
    ii = rng.integers(0, N, size=(rp,), dtype=np.int64)
    jj = rng.integers(0, N, size=(rp,), dtype=np.int64)
    dz_norms = np.linalg.norm(Z[jj] - Z[ii], axis=1)
    dxyz_norms = np.linalg.norm(POS[jj] - POS[ii], axis=1)
    corr = float(np.corrcoef(dz_norms, dxyz_norms)[0, 1])
    lines.append("")
    lines.append("Metric correlation on random pairs: corr(||dz||, ||dxyz||)")
    lines.append(f"  corr={corr:.4f}")

    # === SNAP tests ===
    if args.snap:
        lines.append("")
        lines.append("=== Tests with SNAP to dataset ===")

        # candidate pool
        cand = int(min(args.cand, N))
        cand_idx = np.arange(N, dtype=np.int64) if cand == N else rng.choice(N, size=cand, replace=False).astype(np.int64)

        if args.snap_metric == "predxyz":
            snap_pts = predXYZ[cand_idx]
        else:
            snap_pts = Z[cand_idx]

        knn, used_sklearn = build_knn(snap_pts, use_sklearn=True)
        lines.append(f"[snap] using {args.snap_metric} distance; candidates={cand} (sklearn={'YES' if used_sklearn else 'NO'})")
        lines.append(f"[snap] snap_k={args.snap_k} exclude_self={'YES' if args.snap_exclude_self else 'NO'} snap_min_step={args.snap_min_step:.3f}")

        def snap(z_query, original_global_index=None, predxyz0=None):
            """
            Return snapped latent (from full Z), snapped global index.
            We query top-K and pick first acceptable candidate.
            """
            if args.snap_metric == "predxyz":
                q = z_to_predxyz(z_query)
            else:
                q = z_query

            d, idx = knn(q, k=int(args.snap_k))
            idx = idx[0]  # indices into cand_idx
            for local in idx:
                gi = int(cand_idx[int(local)])

                if args.snap_exclude_self and (original_global_index is not None) and (gi == int(original_global_index)):
                    continue

                # optional reject: too small move in predxyz
                if predxyz0 is not None and args.snap_min_step > 0:
                    p_new = z_to_predxyz(Z[gi])
                    if float(np.linalg.norm(p_new - predxyz0)) < float(args.snap_min_step):
                        continue

                return Z[gi], gi

            # if all rejected, fall back to best
            gi = int(cand_idx[int(idx[0])])
            return Z[gi], gi

        # snap-additivity/commutativity/cycle (+x,+y)
        diffs_add_s = []
        diffs_com_s = []
        diffs_cyc_s = []

        # snap step magnitude + direction (for +x)
        mags_s = []
        coss_s = []

        # intended direction in predxyz for +x is (A * dzx) displacement
        intended_dir = z_to_predxyz(apply_op(np.zeros((D,), np.float32), ops["+x"])) - z_to_predxyz(np.zeros((D,), np.float32))
        # if near-zero, just use unit x
        if float(np.linalg.norm(intended_dir)) < 1e-6:
            intended_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        for i in eval_idx:
            z0 = Z[i]
            p0 = z_to_predxyz(z0)

            # additivity: snap( snap(z0+dx)+dy ) vs snap(z0+(dx+dy))
            z1, i1 = snap(apply_op(z0, ops["+x"]), original_global_index=i, predxyz0=p0)
            p1 = z_to_predxyz(z1)
            z2, i2 = snap(apply_op(z1, ops["+y"]), original_global_index=i1, predxyz0=p1)

            z3, i3 = snap(apply_op(z0, ops["+x"] + ops["+y"]), original_global_index=i, predxyz0=p0)

            diffs_add_s.append(float(np.linalg.norm(z_to_predxyz(z2) - z_to_predxyz(z3))))

            # commutativity
            z1b, i1b = snap(apply_op(z0, ops["+y"]), original_global_index=i, predxyz0=p0)
            p1b = z_to_predxyz(z1b)
            z2b, i2b = snap(apply_op(z1b, ops["+x"]), original_global_index=i1b, predxyz0=p1b)

            diffs_com_s.append(float(np.linalg.norm(z_to_predxyz(z2) - z_to_predxyz(z2b))))

            # cycle: +x then -x should return (approximately) identity under snap
            zcx, icx = snap(apply_op(z0, ops["+x"]), original_global_index=i, predxyz0=p0)
            pcx = z_to_predxyz(zcx)
            zback, iback = snap(apply_op(zcx, ops["-x"]), original_global_index=icx, predxyz0=pcx)

            diffs_cyc_s.append(float(np.linalg.norm(z_to_predxyz(zback) - p0)))

            # +x step magnitude + direction
            p_after = z_to_predxyz(zcx)
            disp = (p_after - p0).astype(np.float32)
            mags_s.append(float(np.linalg.norm(disp)))
            coss_s.append(cosine(disp, intended_dir))

        s_add2 = stats(diffs_add_s)
        s_com2 = stats(diffs_com_s)
        s_cyc2 = stats(diffs_cyc_s)
        s_mag2 = stats(mags_s)
        s_cos2 = stats(coss_s)

        lines.append("Additivity (snap): snap(+x then +y) vs snap(+(x+y))")
        lines.append(f"  norm(diff): mean={s_add2['mean']:.4f} med={s_add2['med']:.4f} p90={s_add2['p90']:.4f} p99={s_add2['p99']:.4f} max={s_add2['mx']:.4f}")
        lines.append("Commutativity (snap): snap(+x then +y) vs snap(+y then +x)")
        lines.append(f"  norm(diff): mean={s_com2['mean']:.4f} med={s_com2['med']:.4f} p90={s_com2['p90']:.4f} p99={s_com2['p99']:.4f} max={s_com2['mx']:.4f}")
        lines.append("Cycle (snap): snap(+x then -x) vs snap(identity)")
        lines.append(f"  norm(diff): mean={s_cyc2['mean']:.4f} med={s_cyc2['med']:.4f} p90={s_cyc2['p90']:.4f} p99={s_cyc2['p99']:.4f} max={s_cyc2['mx']:.4f}")

        lines.append("Step magnitude (+x) after snap (predxyz displacement)")
        lines.append(f"  norm(dxyz): mean={s_mag2['mean']:.4f} med={s_mag2['med']:.4f} p90={s_mag2['p90']:.4f} p99={s_mag2['p99']:.4f} max={s_mag2['mx']:.4f}")
        lines.append(f"  target step={args.step:.4f}")

        lines.append("Direction consistency (+x) after snap (cosine with intended direction)")
        lines.append(f"  cos: mean={s_cos2['mean']:.4f} med={s_cos2['med']:.4f} p90={s_cos2['p90']:.4f} p99={s_cos2['p99']:.4f} max={s_cos2['mx']:.4f}")

    # write report (UTF-8)
    outpath = os.path.join(args.outdir, "report_v3.txt")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("[ok] wrote:", outpath)

if __name__ == "__main__":
    main()
