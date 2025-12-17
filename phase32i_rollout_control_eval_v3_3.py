#!/usr/bin/env python3
# phase32i_rollout_control_eval_v3_3.py
#
# v3.3:
#  - true sphere target sampling (fixed) OR legacy sampling for comparison
#  - multi-scale operator prototypes: coarse step + fine step (default fine=step/2)
#  - policy: best1 lookahead over ALL moves (6 dirs * 2 scales) after snap
#  - early termination on success + true per-K accounting
#  - optional no-progress stop
#
# Drop-in successor to v3_2

import os, argparse
import numpy as np
from sklearn.neighbors import NearestNeighbors

def unit(v, eps=1e-12):
    n = float(np.linalg.norm(v))
    return v * 0.0 if n < eps else (v / n)

def build_protos(rng, Z, XYZ, step, tol, n_pairs, force_antisym, dot_thr=0.95, dz_min=1e-6):
    N, D = Z.shape
    dirs = {
        "+x": np.array([ 1, 0, 0], np.float32),
        "-x": np.array([-1, 0, 0], np.float32),
        "+y": np.array([ 0, 1, 0], np.float32),
        "-y": np.array([ 0,-1, 0], np.float32),
        "+z": np.array([ 0, 0, 1], np.float32),
        "-z": np.array([ 0, 0,-1], np.float32),
    }
    ops = {k: [] for k in dirs}

    idx_i = rng.integers(0, N, size=n_pairs, endpoint=False)
    idx_j = rng.integers(0, N, size=n_pairs, endpoint=False)

    for i, j in zip(idx_i, idx_j):
        dxyz = XYZ[j] - XYZ[i]
        norm = float(np.linalg.norm(dxyz))
        if abs(norm - step) > tol:
            continue
        dz = Z[j] - Z[i]
        if float(np.linalg.norm(dz)) < dz_min:
            continue
        u = dxyz / (norm + 1e-8)
        for k, d in dirs.items():
            if float(np.dot(u, d)) > dot_thr:
                ops[k].append(dz)

    if force_antisym:
        for ax in ["x", "y", "z"]:
            p, n = "+" + ax, "-" + ax
            if ops[p] and ops[n]:
                m = min(len(ops[p]), len(ops[n]))
                ops[p] = ops[p][:m]
                ops[n] = [-dz for dz in ops[p]]

    proto = {}
    counts = {}
    for k, v in ops.items():
        if len(v) > 0:
            proto[k] = np.mean(np.stack(v, axis=0), axis=0).astype(np.float32)
            counts[k] = len(v)

    return dirs, proto, counts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--step", type=float, default=0.35)
    ap.add_argument("--fine_step", type=float, default=None, help="default = step/2")
    ap.add_argument("--tol", type=float, default=0.12)
    ap.add_argument("--n_pairs", type=int, default=2500000)

    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--max_steps", type=int, default=40)
    ap.add_argument("--target_radius", type=float, default=1.5)
    ap.add_argument("--target_mode", default="ball", choices=["ball"])
    ap.add_argument("--success_tol", type=float, default=0.10)

    ap.add_argument("--target_sampling", choices=["fixed", "legacy"], default="fixed")

    ap.add_argument("--policy", choices=["best1"], default="best1",
                    help="best1 = 1-step lookahead over all moves (scales included)")

    ap.add_argument("--no_progress_stop", action="store_true")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--improve_eps", type=float, default=1e-3)
    ap.add_argument("--min_steps", type=int, default=4)

    ap.add_argument("--snap", action="store_true")
    ap.add_argument("--snap_metric", choices=["latent", "predxyz"], default="predxyz")
    ap.add_argument("--snap_k", type=int, default=50)
    ap.add_argument("--exclude_self", action="store_true")
    ap.add_argument("--snap_min_step", type=float, default=0.1)

    ap.add_argument("--force_antisym", action="store_true")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    fine_step = args.fine_step if args.fine_step is not None else (args.step * 0.25)

    rng = np.random.default_rng(args.seed)

    # ----------------------------
    # Load
    # ----------------------------
    Z = np.load(args.latents).astype(np.float32)
    lab = np.load(args.labels)
    XYZ = np.stack([lab["dx"], lab["dy"], lab["dz"]], axis=1).astype(np.float32)

    N, D = Z.shape
    print("PHASE 32I v3.3 — Multi-scale best1 rollouts")
    print(f"N={N} D={D}")
    print(f"step={args.step} fine_step={fine_step} tol={args.tol} n_pairs={args.n_pairs}")
    print(f"episodes={args.episodes} max_steps={args.max_steps} success_tol={args.success_tol}")
    print(f"target_radius={args.target_radius} target_sampling={args.target_sampling}")
    print(f"snap={args.snap} metric={args.snap_metric} snap_k={args.snap_k} snap_min_step={args.snap_min_step}")
    print(f"no_progress_stop={args.no_progress_stop} patience={args.patience} eps={args.improve_eps} min_steps={args.min_steps}")

    # ----------------------------
    # Affine z -> xyz
    # ----------------------------
    Z1 = np.concatenate([Z, np.ones((N, 1), np.float32)], axis=1)
    A, *_ = np.linalg.lstsq(Z1, XYZ, rcond=None)

    def z_to_xyz(z):
        z1 = np.concatenate([z, np.ones((1,), np.float32)])
        return (z1 @ A).astype(np.float32)

    # ----------------------------
    # Build operator prototypes (coarse + fine)
    # ----------------------------
    dirs, proto_c, counts_c = build_protos(
        rng, Z, XYZ, step=args.step, tol=args.tol, n_pairs=args.n_pairs,
        force_antisym=args.force_antisym
    )
    dirs2, proto_f, counts_f = build_protos(
        rng, Z, XYZ, step=float(fine_step), tol=args.tol, n_pairs=args.n_pairs,
        force_antisym=args.force_antisym
    )

    missing = [k for k in dirs.keys() if k not in proto_c or k not in proto_f]
    if missing:
        raise RuntimeError(f"Missing operator prototypes for: {missing}. Increase --n_pairs or loosen --tol.")

    print("\nOperator counts (coarse):")
    for k in sorted(dirs.keys()):
        print(f"  {k}: {counts_c.get(k,0)}")
    print("Operator counts (fine):")
    for k in sorted(dirs.keys()):
        print(f"  {k}: {counts_f.get(k,0)}")

    # Action set: 12 moves
    actions = []
    for k in sorted(dirs.keys()):
        actions.append((f"{k}@C", proto_c[k]))
        actions.append((f"{k}@F", proto_f[k]))

    # ----------------------------
    # Snap index
    # ----------------------------
    nbrs = None
    if args.snap:
        snap_space = Z if args.snap_metric == "latent" else XYZ
        nbrs = NearestNeighbors(n_neighbors=args.snap_k + 1).fit(snap_space)

    def do_snap(z_cur, z_prev=None):
        if not args.snap:
            return z_cur
        q = z_cur if args.snap_metric == "latent" else z_to_xyz(z_cur)
        _, inds = nbrs.kneighbors(q.reshape(1, -1))
        for idx in inds[0]:
            if args.exclude_self and z_prev is not None:
                if np.allclose(Z[idx], z_prev):
                    continue
            if float(np.linalg.norm(z_cur - Z[idx])) >= args.snap_min_step:
                return Z[idx]
        return z_cur

    # ----------------------------
    # Rollouts
    # ----------------------------
    K_LIST = [5, 10, 20, 40]
    success_at = {K: 0 for K in K_LIST}

    rows = []
    start_dists = []
    final_dists = []
    steps_used = []
    path_eff = []

    for ep in range(args.episodes):
        s = int(rng.integers(0, N, endpoint=False))
        z = Z[s].copy()
        xyz = z_to_xyz(z)

        # target
        if args.target_mode == "ball":
            if args.target_sampling == "fixed":
                direction = unit(rng.normal(size=3).astype(np.float32))
                tgt = xyz + direction * float(args.target_radius)
            else:
                # legacy buggy mode (for fair comparison to old numbers)
                tmp = xyz + rng.normal(size=3).astype(np.float32)
                tgt = xyz + unit(tmp) * float(args.target_radius)
        else:
            raise ValueError("Unsupported target_mode")

        start_dist = float(np.linalg.norm(xyz - tgt))
        start_dists.append(start_dist)

        hit_step = None
        path_len = 0.0
        best_dist = start_dist
        no_improve = 0

        for t in range(1, args.max_steps + 1):
            dist = float(np.linalg.norm(xyz - tgt))
            if dist <= args.success_tol:
                hit_step = t - 1
                break

            # best1 over 12 actions
            best_next = 1e9
            best_z = None
            best_xyz = None

            for name, dz in actions:
                z_cand = z + dz
                z_cand = do_snap(z_cand, z_prev=z)
                xyz_cand = z_to_xyz(z_cand)
                d = float(np.linalg.norm(xyz_cand - tgt))
                if d < best_next:
                    best_next = d
                    best_z = z_cand
                    best_xyz = xyz_cand

            step_len = float(np.linalg.norm(best_xyz - xyz))
            path_len += step_len
            z_prev = z
            z, xyz = best_z, best_xyz

            new_dist = float(np.linalg.norm(xyz - tgt))
            if new_dist <= args.success_tol:
                hit_step = t
                break

            if args.no_progress_stop:
                if new_dist < best_dist - args.improve_eps:
                    best_dist = new_dist
                    no_improve = 0
                else:
                    no_improve += 1
                if t >= args.min_steps and no_improve >= args.patience:
                    break

        final_dist = float(np.linalg.norm(xyz - tgt))
        final_dists.append(final_dist)

        used = hit_step if hit_step is not None else t
        steps_used.append(float(used))

        if path_len > 1e-6:
            path_eff.append(start_dist / path_len)
        else:
            path_eff.append(0.0)

        if hit_step is not None:
            for K in K_LIST:
                if hit_step <= K:
                    success_at[K] += 1

        rows.append([
            ep,
            start_dist,
            final_dist,
            hit_step if hit_step is not None else -1,
            used,
            path_len,
            1 if hit_step is not None else 0
        ])

    # ----------------------------
    # Report + save
    # ----------------------------
    sd = np.array(start_dists, np.float32)
    fd = np.array(final_dists, np.float32)
    su = np.array(steps_used, np.float32)
    pe = np.array(path_eff, np.float32)

    print("\nStart distance:")
    print(f"  mean={sd.mean():.4f} med={np.median(sd):.4f} p90={np.percentile(sd,90):.4f}")

    print("\nSuccess@K:")
    for K in K_LIST:
        print(f"  K={K:>3}: {success_at[K]}/{args.episodes} = {success_at[K]/args.episodes:.3f}")

    print("\nFinal distance:")
    print(f"  mean={fd.mean():.4f} med={np.median(fd):.4f} p90={np.percentile(fd,90):.4f}")

    print("\nSteps used:")
    print(f"  mean={su.mean():.2f} med={np.median(su):.2f} p90={np.percentile(su,90):.2f}")

    print("\nPath efficiency (straight/path):")
    print(f"  mean={pe.mean():.4f} med={np.median(pe):.4f} p90={np.percentile(pe,90):.4f}")

    out_csv = os.path.join(args.outdir, "rollouts_32i_v3_3.csv")
    header = "ep,start_dist,final_dist,hit_step,steps_used,path_len,success"
    np.savetxt(out_csv, np.array(rows, np.float32), delimiter=",", header=header, comments="")
    print(f"\n[ok] wrote {out_csv}")

if __name__ == "__main__":
    main()
