#!/usr/bin/env python3
# phase32i_rollout_control_eval.py
#
# Phase 32I — Control / reachability rollouts in latent space using learned operators
# + snapping to nearest dataset sample (in predxyz space).
#
# Pipeline:
#  1) Load latents Z [N,D] and labels.npz (dx,dy,dz)
#  2) Fit affine z->predxyz via least squares: xyz = A z + b
#  3) Estimate operator prototypes dz_{+x,-x,+y,-y,+z,-z} by mining random pairs
#     where (predxyz_j - predxyz_i) is near +/- step * axis within tol
#  4) Evaluate greedy rollouts: from random start, target = start + random delta
#     Choose best action among 6 ops each step, apply, snap-to-dataset (predxyz metric),
#     stop when within success_tol or hit max_steps.
#
# Outputs:
#   outdir/report_32i.txt
#   outdir/rollouts_32i.csv
#   outdir/ops_32i.npz

import os, argparse, math, csv
import numpy as np

def fit_affine(Z, XYZ):
    """
    Fit XYZ ≈ A Z + b  (A: 3xD, b: 3)
    Returns A, b
    """
    N, D = Z.shape
    X = np.concatenate([Z, np.ones((N, 1), dtype=np.float32)], axis=1)  # [N, D+1]
    # Solve X W ≈ XYZ, W = [D+1, 3]
    W, *_ = np.linalg.lstsq(X, XYZ, rcond=None)
    W = W.astype(np.float32)
    A = W[:D, :].T          # [3, D]
    b = W[D, :].reshape(3)  # [3]
    return A, b

def predxyz(Z, A, b):
    # Z: [N,D] or [D]
    if Z.ndim == 1:
        return (A @ Z.astype(np.float32)) + b
    return (Z.astype(np.float32) @ A.T) + b  # [N,3]

def r2_score(y_true, y_pred):
    # y_true,y_pred: [N,3]
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2, axis=0) + 1e-12
    r2_dim = 1.0 - ss_res / ss_tot
    # overall
    ss_res_all = float(np.sum((y_true - y_pred) ** 2))
    ss_tot_all = float(np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2) + 1e-12)
    r2_total = 1.0 - ss_res_all / ss_tot_all
    return r2_dim, r2_total

def mine_operator_prototypes(Z, XYZhat, step, tol, n_pairs, seed=0, force_antisym=True):
    """
    Mine dz operator prototypes by selecting pairs (i,j) s.t. dxyz ≈ +/- step*axis within tol.
    Return dict name->dz_mean and counts.
    """
    rng = np.random.default_rng(seed)
    N, D = Z.shape

    # Predefine operator bins
    ops = {
        "+x": [], "-x": [],
        "+y": [], "-y": [],
        "+z": [], "-z": [],
    }

    # Helper: classify a displacement into operator bin (or None)
    def classify(d):
        # d: [3]
        # Require the target axis component near +/- step and other components small
        ax = abs(d[0] - step) <= tol and abs(d[1]) <= tol and abs(d[2]) <= tol
        bx = abs(d[0] + step) <= tol and abs(d[1]) <= tol and abs(d[2]) <= tol
        ay = abs(d[1] - step) <= tol and abs(d[0]) <= tol and abs(d[2]) <= tol
        by = abs(d[1] + step) <= tol and abs(d[0]) <= tol and abs(d[2]) <= tol
        az = abs(d[2] - step) <= tol and abs(d[0]) <= tol and abs(d[1]) <= tol
        bz = abs(d[2] + step) <= tol and abs(d[0]) <= tol and abs(d[1]) <= tol
        if ax: return "+x"
        if bx: return "-x"
        if ay: return "+y"
        if by: return "-y"
        if az: return "+z"
        if bz: return "-z"
        return None

    # Sample random pairs
    # We allow i!=j; using vectorized blocks for speed
    block = 250000  # tune for memory
    done = 0
    while done < n_pairs:
        m = min(block, n_pairs - done)
        i = rng.integers(0, N, size=m, dtype=np.int64)
        j = rng.integers(0, N, size=m, dtype=np.int64)
        mask = (i != j)
        if not np.any(mask):
            continue
        i = i[mask]; j = j[mask]
        dxyz = XYZhat[j] - XYZhat[i]  # [m',3]
        dz   = Z[j] - Z[i]            # [m',D]

        for k in range(dxyz.shape[0]):
            op = classify(dxyz[k])
            if op is not None:
                ops[op].append(dz[k])

        done += m

    # Compute means
    op_mean = {}
    op_count = {}
    for name, lst in ops.items():
        op_count[name] = len(lst)
        if len(lst) == 0:
            op_mean[name] = None
        else:
            op_mean[name] = np.mean(np.stack(lst, axis=0), axis=0).astype(np.float32)

    # Force antisymmetry if requested
    if force_antisym:
        pairs = [("+x","-x"),("+y","-y"),("+z","-z")]
        for a,bn in pairs:
            if op_mean[a] is not None and op_mean[bn] is not None:
                m = 0.5*(op_mean[a] - op_mean[bn])
                op_mean[a]  = m.astype(np.float32)
                op_mean[bn] = (-m).astype(np.float32)

    return op_mean, op_count

def build_knn(points, metric="euclidean"):
    # points: [N,3] float32
    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception as e:
        raise RuntimeError("scikit-learn is required for snapping. pip install scikit-learn") from e

    nn = NearestNeighbors(n_neighbors=50, algorithm="auto", metric=metric)
    nn.fit(points)
    return nn

def snap_to_dataset(z_query, z_current_idx, Z, XYZhat, knn, A, b,
                    snap_k=50, exclude_self=True, snap_min_step=0.10):
    """
    Snap z_query to a dataset latent, using nearest neighbors in predxyz space.
    We query knn on predxyz(z_query) and choose first candidate that:
      - not self (if exclude_self)
      - distance from current predxyz >= snap_min_step (if >0)
    Returns (snapped_idx, snapped_z, snapped_xyzhat, snapped_dist)
    """
    xq = predxyz(z_query, A, b).reshape(1,3).astype(np.float32)
    dists, idxs = knn.kneighbors(xq, n_neighbors=snap_k, return_distance=True)
    dists = dists.reshape(-1)
    idxs = idxs.reshape(-1)

    cur_xyz = XYZhat[z_current_idx]
    for dist, idx in zip(dists, idxs):
        if exclude_self and int(idx) == int(z_current_idx):
            continue
        if snap_min_step > 0:
            step = float(np.linalg.norm(XYZhat[int(idx)] - cur_xyz))
            if step < snap_min_step:
                continue
        return int(idx), Z[int(idx)], XYZhat[int(idx)], float(dist)

    # fallback: best neighbor
    idx0 = int(idxs[0])
    return idx0, Z[idx0], XYZhat[idx0], float(dists[0])

def random_delta(rng, radius, mode="ball"):
    """
    Sample a target delta in 3D.
      mode="ball": uniform direction, radius uniform [0,r]
      mode="shell": fixed radius
    """
    v = rng.normal(size=3).astype(np.float32)
    n = float(np.linalg.norm(v) + 1e-12)
    v = v / n
    if mode == "shell":
        r = radius
    else:
        r = float(rng.random()) * radius
    return v * r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--outdir", required=True)

    # operator mining
    ap.add_argument("--step", type=float, default=0.35)
    ap.add_argument("--tol", type=float, default=0.12)
    ap.add_argument("--n_pairs", type=int, default=2500000)
    ap.add_argument("--force_antisym", action="store_true")

    # snapping
    ap.add_argument("--snap_k", type=int, default=50)
    ap.add_argument("--snap_min_step", type=float, default=0.10)
    ap.add_argument("--no_exclude_self", action="store_true")

    # rollout eval
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--max_steps", type=int, default=40)
    ap.add_argument("--target_radius", type=float, default=1.5)
    ap.add_argument("--target_mode", choices=["ball","shell"], default="ball")
    ap.add_argument("--success_tol", type=float, default=0.10)
    ap.add_argument("--eval_steps", default="5,10,20,40")

    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    Z = np.load(args.latents).astype(np.float32)  # [N,D]
    lab = np.load(args.labels)
    dx = lab["dx"].astype(np.float32).reshape(-1)
    dy = lab["dy"].astype(np.float32).reshape(-1)
    dz = lab["dz"].astype(np.float32).reshape(-1)
    XYZ = np.stack([dx,dy,dz], axis=1).astype(np.float32)  # [N,3]

    N, D = Z.shape
    print(f"N={N} D={D}")
    print("[data] labels xyz:", XYZ.shape)

    # 1) affine fit
    A, b = fit_affine(Z, XYZ)
    XYZhat = predxyz(Z, A, b).astype(np.float32)
    r2_dim, r2_total = r2_score(XYZ, XYZhat)
    print(f"[affine z->xyz] R2_dim=({r2_dim[0]:.4f},{r2_dim[1]:.4f},{r2_dim[2]:.4f}) R2_total={r2_total:.4f}")

    # 2) mine operators
    print(f"[ops] mining n_pairs={args.n_pairs} step={args.step} tol={args.tol} force_antisym={'YES' if args.force_antisym else 'NO'}")
    op_mean, op_count = mine_operator_prototypes(
        Z, XYZhat,
        step=args.step, tol=args.tol,
        n_pairs=args.n_pairs,
        seed=args.seed,
        force_antisym=args.force_antisym
    )

    missing = [k for k,v in op_mean.items() if v is None]
    if missing:
        print("[ops] MISSING:", missing)
        raise SystemExit("ERROR: missing required operators. Increase --n_pairs and/or loosen --tol.")

    # save ops
    np.savez(os.path.join(args.outdir, "ops_32i.npz"),
             A=A, b=b,
             step=np.float32(args.step), tol=np.float32(args.tol),
             dz_px=op_mean["+x"], dz_nx=op_mean["-x"],
             dz_py=op_mean["+y"], dz_ny=op_mean["-y"],
             dz_pz=op_mean["+z"], dz_nz=op_mean["-z"],
             count_px=np.int64(op_count["+x"]), count_nx=np.int64(op_count["-x"]),
             count_py=np.int64(op_count["+y"]), count_ny=np.int64(op_count["-y"]),
             count_pz=np.int64(op_count["+z"]), count_nz=np.int64(op_count["-z"])
            )
    print("[ok] wrote ops_32i.npz")

    # 3) KNN for snapping in predxyz space
    print("[snap] building KNN in predxyz space...")
    knn = build_knn(XYZhat, metric="euclidean")
    exclude_self = (not args.no_exclude_self)

    # 4) rollouts
    eval_steps = [int(s.strip()) for s in args.eval_steps.split(",") if s.strip()]
    eval_steps = sorted(set(eval_steps))
    maxK = max([args.max_steps] + eval_steps)

    actions = [
        ("+x", op_mean["+x"]),
        ("-x", op_mean["-x"]),
        ("+y", op_mean["+y"]),
        ("-y", op_mean["-y"]),
        ("+z", op_mean["+z"]),
        ("-z", op_mean["-z"]),
    ]

    # metrics accumulators
    success_at = {K:0 for K in eval_steps}
    dist0_list = []
    distf_list = []
    steps_used_list = []
    efficiency_list = []
    drift_list = []

    csv_path = os.path.join(args.outdir, "rollouts_32i.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow([
            "ep","start_idx",
            "target_x","target_y","target_z",
            "start_x","start_y","start_z",
            "final_x","final_y","final_z",
            "dist_start","dist_final",
            "steps_used","success",
            "path_len","straight_len","efficiency",
            "avg_step","snap_failures"
        ])

        for ep in range(args.episodes):
            start_idx = int(rng.integers(0, N))
            z_cur = Z[start_idx].copy()
            idx_cur = start_idx
            xyz_start = XYZhat[idx_cur].copy()

            delta = random_delta(rng, args.target_radius, mode=args.target_mode)
            xyz_target = xyz_start + delta

            # rollout
            path_len = 0.0
            snap_failures = 0
            prev_xyz = xyz_start.copy()

            dist_start = float(np.linalg.norm(xyz_start - xyz_target))
            dist = dist_start

            success_step = None
            steps = 0

            for t in range(1, maxK+1):
                # greedy choose best among 6 actions
                best = None
                best_dist = None
                best_idx = None
                best_z = None
                best_xyz = None

                for name, dzop in actions:
                    z_try = z_cur + dzop
                    try:
                        idx_s, z_s, xyz_s, _ = snap_to_dataset(
                            z_try, idx_cur, Z, XYZhat, knn, A, b,
                            snap_k=args.snap_k,
                            exclude_self=exclude_self,
                            snap_min_step=args.snap_min_step
                        )
                    except Exception:
                        snap_failures += 1
                        continue

                    d = float(np.linalg.norm(xyz_s - xyz_target))
                    if (best_dist is None) or (d < best_dist):
                        best_dist = d
                        best = name
                        best_idx = idx_s
                        best_z = z_s
                        best_xyz = xyz_s

                if best is None:
                    # no valid snap (very unlikely)
                    snap_failures += 1
                    break

                # apply best
                z_cur = best_z
                idx_cur = int(best_idx)

                # path update
                step_len = float(np.linalg.norm(best_xyz - prev_xyz))
                path_len += step_len
                prev_xyz = best_xyz

                dist = float(best_dist)
                steps = t

                if (success_step is None) and (dist <= args.success_tol):
                    success_step = t

                if t >= args.max_steps:
                    break

            xyz_final = XYZhat[idx_cur].copy()
            dist_final = float(np.linalg.norm(xyz_final - xyz_target))

            straight_len = float(np.linalg.norm(xyz_start - xyz_final))
            eff = float(straight_len / (path_len + 1e-12))
            avg_step = float(path_len / max(1, steps))

            success = int(dist_final <= args.success_tol)

            # success@K based on first time entered tol
            for K in eval_steps:
                if success_step is not None and success_step <= K:
                    success_at[K] += 1

            dist0_list.append(dist_start)
            distf_list.append(dist_final)
            steps_used_list.append(steps)
            efficiency_list.append(eff)

            # drift: how far final is from the ideal target displacement magnitude
            # (i.e., distance from target)
            drift_list.append(dist_final)

            w.writerow([
                ep, start_idx,
                float(xyz_target[0]), float(xyz_target[1]), float(xyz_target[2]),
                float(xyz_start[0]), float(xyz_start[1]), float(xyz_start[2]),
                float(xyz_final[0]), float(xyz_final[1]), float(xyz_final[2]),
                dist_start, dist_final,
                steps, success,
                path_len, straight_len, eff,
                avg_step, snap_failures
            ])

            if (ep+1) % 200 == 0:
                print(f"[rollout] {ep+1}/{args.episodes}")

    # summary report
    def qstats(a):
        a = np.asarray(a, dtype=np.float32)
        return (float(a.mean()), float(np.median(a)), float(np.quantile(a, 0.90)), float(np.quantile(a, 0.99)))

    out_txt = os.path.join(args.outdir, "report_32i.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("PHASE 32I — Control / reachability rollouts\n")
        f.write(f"N={N} D={D}\n")
        f.write(f"step={args.step} tol={args.tol} n_pairs={args.n_pairs}\n")
        f.write(f"episodes={args.episodes} max_steps={args.max_steps} target_radius={args.target_radius} target_mode={args.target_mode}\n")
        f.write(f"success_tol={args.success_tol} snap_metric=predxyz snap_k={args.snap_k} exclude_self={exclude_self} snap_min_step={args.snap_min_step}\n")
        f.write(f"seed={args.seed}\n\n")

        f.write("[affine z->xyz]\n")
        f.write(f"R2_dim=({r2_dim[0]:.4f},{r2_dim[1]:.4f},{r2_dim[2]:.4f}) R2_total={r2_total:.4f}\n\n")

        f.write("Operator prototypes (counts):\n")
        for k in ["+x","-x","+y","-y","+z","-z"]:
            f.write(f"  {k}: count={op_count[k]}\n")
        f.write("\n")

        f.write("Success@K:\n")
        for K in eval_steps:
            f.write(f"  K={K:>3d}: {success_at[K]}/{args.episodes} = {success_at[K]/max(1,args.episodes):.3f}\n")
        f.write("\n")

        m0, med0, p90_0, p99_0 = qstats(dist0_list)
        mf, medf, p90_f, p99_f = qstats(distf_list)
        ms, meds, p90_s, p99_s = qstats(steps_used_list)
        me, mede, p90_e, p99_e = qstats(efficiency_list)

        f.write("Distance to target:\n")
        f.write(f"  start: mean={m0:.4f} med={med0:.4f} p90={p90_0:.4f} p99={p99_0:.4f}\n")
        f.write(f"  final: mean={mf:.4f} med={medf:.4f} p90={p90_f:.4f} p99={p99_f:.4f}\n\n")

        f.write("Steps used:\n")
        f.write(f"  mean={ms:.2f} med={meds:.2f} p90={p90_s:.2f} p99={p99_s:.2f}\n\n")

        f.write("Path efficiency (straight/path):\n")
        f.write(f"  mean={me:.4f} med={mede:.4f} p90={p90_e:.4f} p99={p99_e:.4f}\n\n")

        f.write(f"[ok] wrote rollouts_32i.csv and ops_32i.npz\n")

    print("[ok] wrote:", out_txt)
    print("[ok] wrote:", csv_path)

if __name__ == "__main__":
    main()
