#!/usr/bin/env python3
# phase32i_rollout_control_eval_v3_1.py
#
# Control / reachability rollouts with:
#  - early termination on success
#  - patience-based early termination on lack of progress (FIXED)
#  - true per-K success accounting
#  - fixed target sampling for ball mode (direction-only normalization)
#  - robust axis selection if some operator prototypes are missing
#
# Drop-in replacement for phase32i_rollout_control_eval_v3.py

import os, argparse
import numpy as np
from sklearn.neighbors import NearestNeighbors

# ----------------------------
# Utilities
# ----------------------------

def l2(a, b):
    return float(np.linalg.norm(a - b))

def unit(v, eps=1e-12):
    n = float(np.linalg.norm(v))
    if n < eps:
        return v * 0.0
    return v / n

# ----------------------------
# Main
# ----------------------------

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

    # --- Early termination controls (NEW)
    ap.add_argument("--patience", type=int, default=6,
                    help="How many consecutive non-improving steps before early-stop.")
    ap.add_argument("--improve_eps", type=float, default=1e-3,
                    help="Minimum distance improvement to count as progress.")
    ap.add_argument("--min_steps", type=int, default=3,
                    help="Do not early-stop for no-progress before this many steps.")

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
    # Load data
    # ----------------------------

    Z = np.load(args.latents).astype(np.float32)
    lab = np.load(args.labels)

    # dx,dy,dz are assumed present (as in your pipeline)
    XYZ = np.stack([lab["dx"], lab["dy"], lab["dz"]], axis=1).astype(np.float32)

    N, D = Z.shape
    print("PHASE 32I v3.1 — Early termination rollouts (patience-based)")
    print(f"N={N} D={D}")
    print(f"episodes={args.episodes} max_steps={args.max_steps} success_tol={args.success_tol}")
    print(f"early_stop: patience={args.patience} improve_eps={args.improve_eps} min_steps={args.min_steps}")
    print(f"snap={bool(args.snap)} metric={args.snap_metric} snap_k={args.snap_k} snap_min_step={args.snap_min_step}")

    # ----------------------------
    # Fit affine z -> xyz
    # ----------------------------

    Z1 = np.concatenate([Z, np.ones((N, 1), np.float32)], axis=1)
    A, *_ = np.linalg.lstsq(Z1, XYZ, rcond=None)

    def z_to_xyz(z):
        z1 = np.concatenate([z, np.ones((1,), np.float32)])
        return z1 @ A

    # ----------------------------
    # Build operator prototypes
    # ----------------------------

    dirs = {
        "+x": np.array([ 1, 0, 0], dtype=np.float32),
        "-x": np.array([-1, 0, 0], dtype=np.float32),
        "+y": np.array([ 0, 1, 0], dtype=np.float32),
        "-y": np.array([ 0,-1, 0], dtype=np.float32),
        "+z": np.array([ 0, 0, 1], dtype=np.float32),
        "-z": np.array([ 0, 0,-1], dtype=np.float32),
    }

    ops = {k: [] for k in dirs}

    idx_i = rng.integers(0, N, size=args.n_pairs, endpoint=False)
    idx_j = rng.integers(0, N, size=args.n_pairs, endpoint=False)

    # Collect dz examples for near-step pairs aligned with axes
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
    proto_counts = {}
    for k, v in ops.items():
        if len(v) > 0:
            proto[k] = np.mean(np.stack(v, axis=0), axis=0).astype(np.float32)
            proto_counts[k] = len(v)

    if not proto:
        raise RuntimeError("No operator prototypes were built. Try increasing --n_pairs or loosening --tol.")

    print("\nOperator prototype counts:")
    for k in sorted(dirs.keys()):
        c = proto_counts.get(k, 0)
        print(f"  {k}: {c}")

    # ----------------------------
    # Snap index
    # ----------------------------

    nbrs = None
    snap_space = None
    if args.snap:
        snap_space = Z if args.snap_metric == "latent" else XYZ
        nbrs = NearestNeighbors(n_neighbors=args.snap_k + 1).fit(snap_space)

    def snap(z_cur, z_prev=None):
        # Query in either latent or predicted xyz space
        if args.snap_metric == "latent":
            q = z_cur
        else:
            q = z_to_xyz(z_cur).astype(np.float32)

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

    for ep in range(args.episodes):
        s = int(rng.integers(0, N, endpoint=False))
        z = Z[s].copy()
        xyz = z_to_xyz(z)

        # FIXED target sampling: normalize direction only
        if args.target_mode == "ball":
            direction = rng.normal(size=3).astype(np.float32)
            direction = unit(direction)
            tgt = xyz + direction * float(args.target_radius)
        else:
            raise ValueError("Unsupported target_mode")

        start_dist = float(np.linalg.norm(xyz - tgt))
        best_dist = start_dist
        no_improve = 0
        hit_step = None
        path_len = 0.0

        for t in range(1, args.max_steps + 1):
            delta = (tgt - xyz).astype(np.float32)

            # Choose an axis among *available* prototypes
            # (robust if, say, +z missing)
            best_k = None
            best_dot = -1e9
            for k in proto.keys():
                d = dirs[k]
                v = float(np.dot(delta, d))
                if v > best_dot:
                    best_dot = v
                    best_k = k

            dz = proto[best_k]
            z_next = z + dz

            if args.snap:
                z_next = snap(z_next, z_prev=z)

            xyz_next = z_to_xyz(z_next)
            step_len = float(np.linalg.norm(xyz_next - xyz))
            path_len += step_len

            z, xyz = z_next, xyz_next
            dist = float(np.linalg.norm(xyz - tgt))

            # Success early termination
            if dist <= args.success_tol:
                hit_step = t
                break

            # Patience-based early termination (NEW, FIXED)
            if dist < best_dist - args.improve_eps:
                best_dist = dist
                no_improve = 0
            else:
                no_improve += 1

            if t >= args.min_steps and no_improve >= args.patience:
                # stop only if we are not already basically successful
                if best_dist > args.success_tol:
                    break

        final_dist = float(np.linalg.norm(xyz - tgt))
        final_dists.append(final_dist)
        steps_taken = hit_step if hit_step is not None else t
        steps_used.append(steps_taken)

        if path_len > 1e-6:
            path_eff.append(start_dist / path_len)
        else:
            path_eff.append(0.0)

        # per-K success accounting
        if hit_step is not None:
            for K in K_LIST:
                if hit_step <= K:
                    success_at[K] += 1

        rows.append([
            ep,
            start_dist,
            final_dist,
            hit_step if hit_step is not None else -1,
            steps_taken,
            path_len,
            1 if hit_step is not None else 0
        ])

    # ----------------------------
    # Report
    # ----------------------------

    print("\nSuccess@K:")
    for K in K_LIST:
        print(f"  K={K:>3}: {success_at[K]}/{args.episodes} = {success_at[K]/args.episodes:.3f}")

    fd = np.array(final_dists, dtype=np.float32)
    print("\nFinal distance:")
    print(f"  mean={fd.mean():.4f} med={np.median(fd):.4f} p90={np.percentile(fd,90):.4f}")

    su = np.array(steps_used, dtype=np.float32)
    print("\nSteps used (hit_step if success else steps until stop):")
    print(f"  mean={su.mean():.2f} med={np.median(su):.2f} p90={np.percentile(su,90):.2f}")

    pe = np.array(path_eff, dtype=np.float32)
    print("\nPath efficiency (straight/path):")
    print(f"  mean={pe.mean():.4f} med={np.median(pe):.4f} p90={np.percentile(pe,90):.4f}")

    # Save CSV
    out_csv = os.path.join(args.outdir, "rollouts_32i_v3_1.csv")
    header = "ep,start_dist,final_dist,hit_step,steps_taken,path_len,success"
    np.savetxt(out_csv, np.array(rows, dtype=np.float32), delimiter=",", header=header, comments="")
    print(f"\n[ok] wrote {out_csv}")

if __name__ == "__main__":
    main()
