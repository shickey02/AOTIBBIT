#!/usr/bin/env python3
# phase32i_rollout_control_eval_v3_2.py
#
# Fixes:
#  - option to reproduce legacy (buggy/easier) target sampling for fair comparison
#  - policy option: best1-step lookahead over all 6 operators (after snap)
#  - early termination: success always; no-progress stopping optional
#  - true per-K accounting

import os, argparse
import numpy as np
from sklearn.neighbors import NearestNeighbors

def unit(v, eps=1e-12):
    n = float(np.linalg.norm(v))
    return v * 0.0 if n < eps else (v / n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--step", type=float, default=0.35)
    ap.add_argument("--tol", type=float, default=0.12)
    ap.add_argument("--n_pairs", type=int, default=2500000)

    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--max_steps", type=int, default=40)
    ap.add_argument("--target_radius", type=float, default=1.5)
    ap.add_argument("--target_mode", default="ball", choices=["ball"])
    ap.add_argument("--success_tol", type=float, default=0.10)

    # Target sampling mode
    ap.add_argument("--target_sampling", choices=["fixed", "legacy"], default="fixed",
                    help="fixed = true sphere at radius; legacy = reproduce old buggy/easier sampling")

    # Control policy
    ap.add_argument("--policy", choices=["greedy_axis", "best1"], default="best1",
                    help="greedy_axis = choose axis by dot(delta, axis); best1 = 1-step lookahead over all moves")

    # Optional no-progress early stop
    ap.add_argument("--no_progress_stop", action="store_true")
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--improve_eps", type=float, default=1e-3)
    ap.add_argument("--min_steps", type=int, default=3)

    ap.add_argument("--snap", action="store_true")
    ap.add_argument("--snap_metric", choices=["latent", "predxyz"], default="predxyz")
    ap.add_argument("--snap_k", type=int, default=50)
    ap.add_argument("--exclude_self", action="store_true")
    ap.add_argument("--snap_min_step", type=float, default=0.1)

    ap.add_argument("--force_antisym", action="store_true")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # ----------------------------
    # Load
    # ----------------------------
    Z = np.load(args.latents).astype(np.float32)
    lab = np.load(args.labels)
    XYZ = np.stack([lab["dx"], lab["dy"], lab["dz"]], axis=1).astype(np.float32)

    N, D = Z.shape
    print("PHASE 32I v3.2 — Control rollouts (policy + target_sampling)")
    print(f"N={N} D={D}")
    print(f"target_radius={args.target_radius} target_sampling={args.target_sampling}")
    print(f"policy={args.policy} snap={args.snap} snap_metric={args.snap_metric} snap_k={args.snap_k}")
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
    # Operator prototypes
    # ----------------------------
    dirs = {
        "+x": np.array([ 1, 0, 0], np.float32),
        "-x": np.array([-1, 0, 0], np.float32),
        "+y": np.array([ 0, 1, 0], np.float32),
        "-y": np.array([ 0,-1, 0], np.float32),
        "+z": np.array([ 0, 0, 1], np.float32),
        "-z": np.array([ 0, 0,-1], np.float32),
    }

    ops = {k: [] for k in dirs}

    idx_i = rng.integers(0, N, size=args.n_pairs, endpoint=False)
    idx_j = rng.integers(0, N, size=args.n_pairs, endpoint=False)

    for i, j in zip(idx_i, idx_j):
        dxyz = XYZ[j] - XYZ[i]
        norm = float(np.linalg.norm(dxyz))
        if abs(norm - args.step) > args.tol:
            continue
        dz = Z[j] - Z[i]
        if float(np.linalg.norm(dz)) < 1e-6:
            continue
        u = dxyz / (norm + 1e-8)
        for k, d in dirs.items():
            if float(np.dot(u, d)) > 0.95:
                ops[k].append(dz)

    if args.force_antisym:
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

    if not proto:
        raise RuntimeError("No operator prototypes built. Increase --n_pairs or loosen --tol.")

    print("\nOperator counts:")
    for k in sorted(dirs.keys()):
        print(f"  {k}: {counts.get(k,0)}")

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
    final_dists = []
    steps_used = []
    path_eff = []
    start_dists = []

    for ep in range(args.episodes):
        s = int(rng.integers(0, N, endpoint=False))
        z = Z[s].copy()
        xyz = z_to_xyz(z)

        # target sampling
        if args.target_mode == "ball":
            if args.target_sampling == "fixed":
                direction = unit(rng.normal(size=3).astype(np.float32))
                tgt = xyz + direction * float(args.target_radius)
            else:
                # legacy (reproduce old behavior): matches your previous script’s structure
                tgt = xyz + rng.normal(size=3).astype(np.float32)
                tgt = xyz + unit(tgt) * float(args.target_radius)
        else:
            raise ValueError("Unsupported target_mode")

        start_dist = float(np.linalg.norm(xyz - tgt))
        start_dists.append(start_dist)

        best_dist = start_dist
        no_improve = 0
        hit_step = None
        path_len = 0.0
        steps_taken = 0

        for t in range(1, args.max_steps + 1):
            steps_taken = t
            dist = float(np.linalg.norm(xyz - tgt))

            if dist <= args.success_tol:
                hit_step = t - 1  # already at success before applying another move
                break

            # choose action
            if args.policy == "greedy_axis":
                delta = (tgt - xyz).astype(np.float32)
                best_k = None
                best_dot = -1e9
                for k in proto.keys():
                    v = float(np.dot(delta, dirs[k]))
                    if v > best_dot:
                        best_dot = v
                        best_k = k
                chosen = best_k
                z_next = z + proto[chosen]
                z_next = do_snap(z_next, z_prev=z)
                xyz_next = z_to_xyz(z_next)

            else:
                # best1: try all moves (after snap), pick the one minimizing next distance
                best_k = None
                best_next_dist = 1e9
                best_z = None
                best_xyz = None

                for k, dz in proto.items():
                    z_cand = z + dz
                    z_cand = do_snap(z_cand, z_prev=z)
                    xyz_cand = z_to_xyz(z_cand)
                    d = float(np.linalg.norm(xyz_cand - tgt))
                    if d < best_next_dist:
                        best_next_dist = d
                        best_k = k
                        best_z = z_cand
                        best_xyz = xyz_cand

                z_next, xyz_next = best_z, best_xyz

            step_len = float(np.linalg.norm(xyz_next - xyz))
            path_len += step_len
            z, xyz = z_next, xyz_next

            new_dist = float(np.linalg.norm(xyz - tgt))

            if new_dist <= args.success_tol:
                hit_step = t
                break

            # optional no-progress early stop
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

        # steps_used = hit_step if success else steps_taken
        used = hit_step if hit_step is not None else steps_taken
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
    print("\nStart distance:")
    sd = np.array(start_dists, np.float32)
    print(f"  mean={sd.mean():.4f} med={np.median(sd):.4f} p90={np.percentile(sd,90):.4f}")

    print("\nSuccess@K:")
    for K in K_LIST:
        print(f"  K={K:>3}: {success_at[K]}/{args.episodes} = {success_at[K]/args.episodes:.3f}")

    fd = np.array(final_dists, np.float32)
    print("\nFinal distance:")
    print(f"  mean={fd.mean():.4f} med={np.median(fd):.4f} p90={np.percentile(fd,90):.4f}")

    su = np.array(steps_used, np.float32)
    print("\nSteps used:")
    print(f"  mean={su.mean():.2f} med={np.median(su):.2f} p90={np.percentile(su,90):.2f}")

    pe = np.array(path_eff, np.float32)
    print("\nPath efficiency (straight/path):")
    print(f"  mean={pe.mean():.4f} med={np.median(pe):.4f} p90={np.percentile(pe,90):.4f}")

    out_csv = os.path.join(args.outdir, "rollouts_32i_v3_2.csv")
    header = "ep,start_dist,final_dist,hit_step,steps_used,path_len,success"
    np.savetxt(out_csv, np.array(rows, np.float32), delimiter=",", header=header, comments="")
    print(f"\n[ok] wrote {out_csv}")

if __name__ == "__main__":
    main()
