#!/usr/bin/env python3
# phase32i_rollout_control_eval_v3_5.py
#
# PHASE 32I v3.5 — Beam-search planning rollouts (snap-graph control)
#
# What changes vs v3.3/v3.4:
#   - Instead of greedy/best1, we do short-horizon planning with beam search.
#   - Planning is done in the snapped dataset world (each action -> snap to dataset).
#   - Receding horizon: plan each step, execute first action of best sequence.
#
# Why this matters:
#   - Snap introduces discrete "state graph" effects and local minima.
#   - 1-step policies can't see around snap traps; beam search can.
#
# Outputs:
#   - prints Success@K, distances, steps, path efficiency
#   - writes rollouts_32i_v3_5.csv with episode stats

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

    # operator mining
    ap.add_argument("--step", type=float, default=0.35)
    ap.add_argument("--tol", type=float, default=0.12)
    ap.add_argument("--n_pairs", type=int, default=2500000)

    # optional scaled fine operators (like v3.4)
    ap.add_argument("--use_fine", action="store_true")
    ap.add_argument("--fine_step", type=float, default=0.0875)

    # rollout
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--max_steps", type=int, default=40)
    ap.add_argument("--target_radius", type=float, default=1.5)
    ap.add_argument("--target_mode", default="ball", choices=["ball"])
    ap.add_argument("--target_sampling", choices=["fixed", "legacy"], default="fixed")
    ap.add_argument("--success_tol", type=float, default=0.10)

    # planning
    ap.add_argument("--plan_depth", type=int, default=4, help="beam search depth per decision")
    ap.add_argument("--beam_width", type=int, default=64, help="beam width")
    ap.add_argument("--score", choices=["final_dist", "min_dist"], default="final_dist",
                    help="final_dist = minimize distance after depth; min_dist = minimize best distance reached within plan")

    # snap
    ap.add_argument("--snap", action="store_true")
    ap.add_argument("--snap_metric", choices=["latent", "predxyz"], default="predxyz")
    ap.add_argument("--snap_k", type=int, default=200)
    ap.add_argument("--exclude_self", action="store_true")
    ap.add_argument("--snap_min_step", type=float, default=0.10)

    # near-target snap loosening
    ap.add_argument("--near_radius", type=float, default=0.50)
    ap.add_argument("--snap_min_step_near", type=float, default=0.0)

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
    print("PHASE 32I v3.5 — Beam-search planning rollouts")
    print(f"N={N} D={D}")
    print(f"step={args.step} tol={args.tol} n_pairs={args.n_pairs}")
    print(f"episodes={args.episodes} max_steps={args.max_steps} success_tol={args.success_tol}")
    print(f"target_radius={args.target_radius} sampling={args.target_sampling}")
    print(f"plan_depth={args.plan_depth} beam_width={args.beam_width} score={args.score}")
    print(f"snap={args.snap} metric={args.snap_metric} snap_k={args.snap_k} snap_min_step={args.snap_min_step}")
    print(f"near_radius={args.near_radius} snap_min_step_near={args.snap_min_step_near}")
    print(f"use_fine={args.use_fine} fine_step={args.fine_step}")
    print(f"seed={args.seed}")

    # ----------------------------
    # Affine z -> xyz
    # ----------------------------
    Z1 = np.concatenate([Z, np.ones((N, 1), np.float32)], axis=1)
    A, *_ = np.linalg.lstsq(Z1, XYZ, rcond=None)

    def z_to_xyz(z):
        z1 = np.concatenate([z, np.ones((1,), np.float32)])
        return (z1 @ A).astype(np.float32)

    # ----------------------------
    # Operator prototypes (coarse)
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

    proto_coarse = {}
    counts = {}
    for k, v in ops.items():
        if len(v) > 0:
            proto_coarse[k] = np.mean(np.stack(v, axis=0), axis=0).astype(np.float32)
            counts[k] = len(v)

    required = ["+x","-x","+y","-y","+z","-z"]
    missing = [k for k in required if k not in proto_coarse]
    if missing:
        raise RuntimeError(f"Missing coarse operator prototypes: {missing}. Increase --n_pairs or loosen --tol.")

    # optional fine operators by scaling coarse (v3.4 approach)
    actions = []
    for k in required:
        actions.append((k, proto_coarse[k], "coarse"))

    if args.use_fine:
        scale = float(args.fine_step / args.step)
        for k in required:
            actions.append((k, (proto_coarse[k] * scale).astype(np.float32), "fine"))
        print("\nFine operators: scaled from coarse by factor=%.6f" % scale)

    print("\nOperator counts (coarse):")
    for k in required:
        print(f"  {k}: {counts.get(k,0)}")

    # ----------------------------
    # Snap index (dataset)
    # ----------------------------
    nbrs = None
    if args.snap:
        snap_space = Z if args.snap_metric == "latent" else XYZ
        nbrs = NearestNeighbors(n_neighbors=args.snap_k + 1).fit(snap_space)

    def do_snap(z_cur, z_prev=None, min_step=0.0):
        if not args.snap:
            return z_cur, None
        q = z_cur if args.snap_metric == "latent" else z_to_xyz(z_cur)
        _, inds = nbrs.kneighbors(q.reshape(1, -1))
        for idx in inds[0]:
            if args.exclude_self and z_prev is not None:
                if np.allclose(Z[idx], z_prev):
                    continue
            if float(np.linalg.norm(z_cur - Z[idx])) >= float(min_step):
                return Z[idx], int(idx)
        return z_cur, None

    def snap_idx_from_latent(z_query, z_prev=None, min_step=0.0):
        # returns snapped idx and snapped latent
        z_s, idx = do_snap(z_query, z_prev=z_prev, min_step=min_step)
        if idx is None:
            # fallback: find exact match index if possible; else nearest by latent
            # (rare; only happens if min_step blocks everything)
            idx = int(np.argmin(np.sum((Z - z_s)**2, axis=1)))
        return idx, z_s

    # ----------------------------
    # Planning: Beam search from current idx
    # ----------------------------
    def plan_action(cur_idx, tgt_xyz, min_step):
        # beam entries: (score, idx, z, xyz, path_len, best_dist_seen, first_action_id)
        z0 = Z[cur_idx]
        xyz0 = z_to_xyz(z0)
        d0 = float(np.linalg.norm(xyz0 - tgt_xyz))

        # initialize beam with "no move yet"
        beam = [(d0, cur_idx, z0, xyz0, 0.0, d0, None)]

        for depth in range(1, args.plan_depth + 1):
            cand = []
            for (sc, idx, z, xyz, path_len, best_seen, first_a) in beam:
                # expand all actions
                for ai, (k, dz, kind) in enumerate(actions):
                    z_prop = z + dz
                    nxt_idx, z_snapped = snap_idx_from_latent(z_prop, z_prev=z, min_step=min_step)
                    xyz_snapped = z_to_xyz(z_snapped)
                    step_len = float(np.linalg.norm(xyz_snapped - xyz))
                    new_path = path_len + step_len
                    dist = float(np.linalg.norm(xyz_snapped - tgt_xyz))
                    new_best = min(best_seen, dist)
                    # choose objective
                    if args.score == "final_dist":
                        score = dist
                    else:
                        score = new_best
                    # preserve the first action of the whole plan
                    fa = ai if first_a is None else first_a
                    cand.append((score, nxt_idx, z_snapped, xyz_snapped, new_path, new_best, fa))

            if not cand:
                break
            # keep top beam_width by score
            cand.sort(key=lambda x: x[0])
            beam = cand[:args.beam_width]

        # pick best in final beam, return its first action
        best = min(beam, key=lambda x: x[0])
        fa = best[6]
        if fa is None:
            return 0  # fallback
        return fa

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
        cur_idx = int(rng.integers(0, N, endpoint=False))
        z = Z[cur_idx].copy()
        xyz = z_to_xyz(z)

        # target sampling
        if args.target_mode == "ball":
            if args.target_sampling == "fixed":
                direction = unit(rng.normal(size=3).astype(np.float32))
                tgt = xyz + direction * float(args.target_radius)
            else:
                tgt = xyz + rng.normal(size=3).astype(np.float32)
                tgt = xyz + unit(tgt) * float(args.target_radius)
        else:
            raise ValueError("Unsupported target_mode")

        start_dist = float(np.linalg.norm(xyz - tgt))
        start_dists.append(start_dist)

        hit_step = None
        path_len = 0.0
        steps_taken = 0

        for t in range(1, args.max_steps + 1):
            steps_taken = t
            dist = float(np.linalg.norm(xyz - tgt))

            # success check
            if dist <= args.success_tol:
                hit_step = t - 1
                break

            # near-target snap loosening
            min_step = args.snap_min_step_near if dist < args.near_radius else args.snap_min_step

            # choose action via beam planning
            ai = plan_action(cur_idx, tgt, min_step=min_step)
            k, dz, kind = actions[ai]

            z_prop = z + dz
            nxt_idx, z_next = snap_idx_from_latent(z_prop, z_prev=z, min_step=min_step)
            xyz_next = z_to_xyz(z_next)

            step_len = float(np.linalg.norm(xyz_next - xyz))
            path_len += step_len

            # advance
            cur_idx = nxt_idx
            z = z_next
            xyz = xyz_next

            # success after move
            if float(np.linalg.norm(xyz - tgt)) <= args.success_tol:
                hit_step = t
                break

        final_dist = float(np.linalg.norm(xyz - tgt))
        final_dists.append(final_dist)

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

    out_csv = os.path.join(args.outdir, "rollouts_32i_v3_5.csv")
    header = "ep,start_dist,final_dist,hit_step,steps_used,path_len,success"
    np.savetxt(out_csv, np.array(rows, np.float32), delimiter=",", header=header, comments="")
    print(f"\n[ok] wrote {out_csv}")

if __name__ == "__main__":
    main()
