#!/usr/bin/env python3
# phase32i_rollout_control_eval_v3.py
#
# Control / reachability rollouts with:
#  - early termination on success
#  - true per-K success accounting
#
# Drop-in replacement for phase32i_rollout_control_eval_v2.py

import os, argparse
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

# ----------------------------
# Utilities
# ----------------------------

def l2(x, y):
    return np.linalg.norm(x - y, axis=-1)

def cosine(a, b, eps=1e-8):
    na = np.linalg.norm(a) + eps
    nb = np.linalg.norm(b) + eps
    return float(np.dot(a, b) / (na * nb))

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
    ap.add_argument("--target_mode", default="ball")
    ap.add_argument("--success_tol", type=float, default=0.1)

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
    XYZ = np.stack([lab["dx"], lab["dy"], lab["dz"]], axis=1).astype(np.float32)

    N, D = Z.shape
    print(f"PHASE 32I v3 — Early termination rollouts")
    print(f"N={N} D={D}")

    # ----------------------------
    # Fit affine z -> xyz
    # ----------------------------

    Z1 = np.concatenate([Z, np.ones((N,1), np.float32)], axis=1)
    A, *_ = np.linalg.lstsq(Z1, XYZ, rcond=None)

    def z_to_xyz(z):
        z1 = np.concatenate([z, np.ones((1,), np.float32)])
        return z1 @ A

    # ----------------------------
    # Build operator prototypes
    # ----------------------------

    dirs = {
        "+x": np.array([1,0,0]),
        "-x": np.array([-1,0,0]),
        "+y": np.array([0,1,0]),
        "-y": np.array([0,-1,0]),
        "+z": np.array([0,0,1]),
        "-z": np.array([0,0,-1]),
    }

    ops = {k: [] for k in dirs}

    idx_i = rng.integers(0, N, size=args.n_pairs)
    idx_j = rng.integers(0, N, size=args.n_pairs)

    for i, j in zip(idx_i, idx_j):
        dxyz = XYZ[j] - XYZ[i]
        norm = np.linalg.norm(dxyz)
        if abs(norm - args.step) > args.tol:
            continue
        dz = Z[j] - Z[i]
        if np.linalg.norm(dz) < 1e-6:
            continue
        u = dxyz / (norm + 1e-8)
        for k, d in dirs.items():
            if np.dot(u, d) > 0.95:
                ops[k].append(dz)

    if args.force_antisym:
        for ax in ["x","y","z"]:
            p, n = "+"+ax, "-"+ax
            if ops[p] and ops[n]:
                m = min(len(ops[p]), len(ops[n]))
                ops[p] = ops[p][:m]
                ops[n] = [-dz for dz in ops[p]]

    proto = {k: np.mean(np.stack(v), axis=0) for k,v in ops.items() if len(v) > 0}

    # ----------------------------
    # Snap index
    # ----------------------------

    if args.snap:
        snap_space = Z if args.snap_metric == "latent" else XYZ
        nbrs = NearestNeighbors(n_neighbors=args.snap_k+1).fit(snap_space)

    def snap(z_cur, z_prev=None):
        q = z_cur if args.snap_metric == "latent" else z_to_xyz(z_cur)
        _, inds = nbrs.kneighbors(q.reshape(1,-1))
        for idx in inds[0]:
            if args.exclude_self and z_prev is not None:
                if np.allclose(Z[idx], z_prev): continue
            if np.linalg.norm(z_cur - Z[idx]) >= args.snap_min_step:
                return Z[idx]
        return z_cur

    # ----------------------------
    # Rollouts
    # ----------------------------

    success_at = {5:0, 10:0, 20:0, 40:0}
    final_dists = []
    steps_used = []
    path_eff = []

    for ep in range(args.episodes):
        s = rng.integers(0, N)
        z = Z[s].copy()
        xyz = z_to_xyz(z)

        if args.target_mode == "ball":
            tgt = xyz + rng.normal(size=3)
            tgt = xyz + tgt / np.linalg.norm(tgt) * args.target_radius

        hit_step = None
        start_dist = np.linalg.norm(xyz - tgt)
        path_len = 0.0

        for t in range(1, args.max_steps+1):
            delta = tgt - xyz
            axis = max(dirs, key=lambda k: np.dot(delta, dirs[k]))
            dz = proto[axis]
            z_next = z + dz

            if args.snap:
                z_next = snap(z_next, z_prev=z)

            xyz_next = z_to_xyz(z_next)
            step_len = np.linalg.norm(xyz_next - xyz)
            path_len += step_len

            z, xyz = z_next, xyz_next
            dist = np.linalg.norm(xyz - tgt)

            if dist <= args.success_tol:
                hit_step = t
                break

        final_dists.append(dist)
        steps_used.append(hit_step if hit_step is not None else args.max_steps)
        if path_len > 1e-6:
            path_eff.append(start_dist / path_len)

        if hit_step is not None:
            for K in success_at:
                if hit_step <= K:
                    success_at[K] += 1

    # ----------------------------
    # Report
    # ----------------------------

    print("\nSuccess@K:")
    for K in sorted(success_at):
        print(f"  K={K:>3}: {success_at[K]}/{args.episodes} = {success_at[K]/args.episodes:.3f}")

    fd = np.array(final_dists)
    print("\nFinal distance:")
    print(f"  mean={fd.mean():.4f} med={np.median(fd):.4f} p90={np.percentile(fd,90):.4f}")

    su = np.array(steps_used)
    print("\nSteps to success:")
    print(f"  mean={su.mean():.2f} med={np.median(su):.2f}")

    pe = np.array(path_eff)
    print("\nPath efficiency:")
    print(f"  mean={pe.mean():.4f} med={np.median(pe):.4f}")

    # Save CSV
    out_csv = os.path.join(args.outdir, "rollouts_32i_v3.csv")
    np.savetxt(
        out_csv,
        np.stack([final_dists, steps_used], axis=1),
        delimiter=",",
        header="final_dist,steps_used",
        comments=""
    )
    print(f"\n[ok] wrote {out_csv}")

if __name__ == "__main__":
    main()
