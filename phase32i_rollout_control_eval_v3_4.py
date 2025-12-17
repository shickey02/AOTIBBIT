#!/usr/bin/env python3
import os, argparse
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm  # <--- NEW IMPORT

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--step", type=float, default=0.35)
    ap.add_argument("--fine_step", type=float, default=0.0875)
    ap.add_argument("--tol", type=float, default=0.12)
    ap.add_argument("--n_pairs", type=int, default=2500000)
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--max_steps", type=int, default=40)
    ap.add_argument("--target_radius", type=float, default=1.5)
    ap.add_argument("--target_mode", default="ball", choices=["ball"])
    ap.add_argument("--success_tol", type=float, default=0.10)
    ap.add_argument("--target_sampling", choices=["fixed", "legacy"], default="fixed")
    ap.add_argument("--no_progress_stop", action="store_true")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--improve_eps", type=float, default=1e-3)
    ap.add_argument("--min_steps", type=int, default=4)
    ap.add_argument("--snap", action="store_true")
    ap.add_argument("--snap_metric", choices=["latent", "predxyz"], default="predxyz")
    ap.add_argument("--snap_k", type=int, default=50)
    ap.add_argument("--exclude_self", action="store_true")
    ap.add_argument("--snap_min_step", type=float, default=0.1)
    ap.add_argument("--near_radius", type=float, default=0.50)
    ap.add_argument("--snap_min_step_near", type=float, default=0.0)
    ap.add_argument("--force_antisym", action="store_true")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Load Data
    print(f"Loading data from {args.latents}...")
    Z = np.load(args.latents).astype(np.float32)
    lab = np.load(args.labels)
    XYZ = np.stack([lab["dx"], lab["dy"], lab["dz"]], axis=1).astype(np.float32)
    N, D = Z.shape

    # ----------------------------
    # 1. OPTIMIZED: Pre-compute Weights/Bias
    # ----------------------------
    Z1 = np.concatenate([Z, np.ones((N, 1), np.float32)], axis=1)
    A, *_ = np.linalg.lstsq(Z1, XYZ, rcond=None)
    W_affine = A[:D]
    b_affine = A[D]

    def z_to_xyz_batch(z_batch):
        return z_batch @ W_affine + b_affine

    # ----------------------------
    # 2. OPTIMIZED: Vectorized Operator Mining
    # ----------------------------
    print("Mining operators (Vectorized)...")
    
    # Sample indices
    idx_i = rng.integers(0, N, size=args.n_pairs, endpoint=False)
    idx_j = rng.integers(0, N, size=args.n_pairs, endpoint=False)

    # Compute XYZ diffs
    dXYZ = XYZ[idx_j] - XYZ[idx_i]
    norms = np.linalg.norm(dXYZ, axis=1)

    # Filter by step size
    valid_step_mask = np.abs(norms - args.step) <= args.tol
    dXYZ = dXYZ[valid_step_mask]
    norms = norms[valid_step_mask]
    
    # Normalize for direction check
    u = dXYZ / (norms[:, None] + 1e-8)

    # Filter by direction
    dir_keys = ["+x", "-x", "+y", "-y", "+z", "-z"]
    dir_vecs = np.array([
        [ 1, 0, 0], [-1, 0, 0],
        [ 0, 1, 0], [ 0,-1, 0],
        [ 0, 0, 1], [ 0, 0,-1]
    ], dtype=np.float32)

    dots = u @ dir_vecs.T
    best_dir_idx = np.argmax(dots, axis=1)
    best_dir_val = np.max(dots, axis=1)
    valid_dir_mask = best_dir_val > 0.95
    
    # Final indices
    final_indices_i = idx_i[valid_step_mask][valid_dir_mask]
    final_indices_j = idx_j[valid_step_mask][valid_dir_mask]
    final_dir_indices = best_dir_idx[valid_dir_mask]

    # Compute Z diffs
    dZ_valid = Z[final_indices_j] - Z[final_indices_i]

    proto_coarse = {}
    counts = {}
    
    for i, k in enumerate(dir_keys):
        mask_k = (final_dir_indices == i)
        dz_k = dZ_valid[mask_k]
        
        # Filter tiny latent moves
        dz_norms = np.linalg.norm(dz_k, axis=1)
        dz_k = dz_k[dz_norms > 1e-6]

        counts[k] = len(dz_k)
        if len(dz_k) > 0:
            proto_coarse[k] = np.mean(dz_k, axis=0)

    # Anti-symmetry
    if args.force_antisym:
        for ax in ["x", "y", "z"]:
            p, n = "+" + ax, "-" + ax
            if p in proto_coarse and n in proto_coarse:
                avg = (proto_coarse[p] - proto_coarse[n]) / 2.0
                proto_coarse[p] = avg
                proto_coarse[n] = -avg

    # Generate Fine Operators
    scale = float(args.fine_step / args.step)
    proto_fine = {k: proto_coarse[k] * scale for k in dir_keys}

    print("Operator counts:", counts)

    # ----------------------------
    # Snap Setup
    # ----------------------------
    nbrs = None
    if args.snap:
        print(f"Building KDTree for snapping (k={args.snap_k})...")
        snap_space = Z if args.snap_metric == "latent" else XYZ
        nbrs = NearestNeighbors(n_neighbors=args.snap_k + 1, algorithm='auto', n_jobs=-1).fit(snap_space)

    def do_snap_batch(z_batch, min_step_val):
        if not args.snap:
            return z_batch
        
        B = z_batch.shape[0]
        q = z_batch if args.snap_metric == "latent" else z_to_xyz_batch(z_batch)
        
        dists, inds = nbrs.kneighbors(q) # (B, K+1)
        z_out = z_batch.copy()
        
        for i in range(B):
            found = False
            for k_idx in range(inds.shape[1]):
                idx = inds[i, k_idx]
                dist_z = np.linalg.norm(z_batch[i] - Z[idx])
                if dist_z >= min_step_val:
                    z_out[i] = Z[idx]
                    found = True
                    break
        return z_out

    # ----------------------------
    # 3. OPTIMIZED: Rollouts with Progress Bar
    # ----------------------------
    ops_list = [proto_coarse[k] for k in dir_keys] + [proto_fine[k] for k in dir_keys]
    ops_matrix = np.stack(ops_list) 

    rows = []
    start_dists, final_dists, steps_used, path_eff = [], [], [], []
    K_LIST = [5, 10, 20, 40]
    success_at = {K: 0 for K in K_LIST}

    print(f"\nStarting {args.episodes} episodes...")

    # --- WRAPPED ITERATOR WITH TQDM ---
    iterator = tqdm(range(args.episodes), desc="Running Rollouts", unit="ep")
    
    for ep in iterator:
        s = int(rng.integers(0, N))
        z = Z[s].copy()
        xyz = z_to_xyz_batch(z[None, :])[0]

        if args.target_sampling == "fixed":
            vec = rng.normal(size=3).astype(np.float32)
            vec /= np.linalg.norm(vec)
            tgt = xyz + vec * args.target_radius
        else:
            tgt = xyz + rng.normal(size=3).astype(np.float32)
            tgt = xyz + (tgt / np.linalg.norm(tgt)) * args.target_radius

        start_dist = float(np.linalg.norm(xyz - tgt))
        start_dists.append(start_dist)

        best_dist = start_dist
        no_improve_count = 0
        hit_step = None
        path_len = 0.0
        used_steps = 0

        for t in range(1, args.max_steps + 1):
            used_steps = t
            dist_to_tgt = np.linalg.norm(xyz - tgt)

            if dist_to_tgt <= args.success_tol:
                hit_step = t - 1
                break

            cur_min_step = args.snap_min_step_near if dist_to_tgt < args.near_radius else args.snap_min_step

            # Batch Ops
            z_cands = z + ops_matrix
            if args.snap:
                z_cands = do_snap_batch(z_cands, cur_min_step)
            xyz_cands = z_to_xyz_batch(z_cands)
            dists = np.linalg.norm(xyz_cands - tgt, axis=1)

            best_idx = np.argmin(dists)
            z_next = z_cands[best_idx]
            xyz_next = xyz_cands[best_idx]
            new_dist = dists[best_idx]

            path_len += np.linalg.norm(xyz_next - xyz)
            z, xyz = z_next, xyz_next

            if new_dist <= args.success_tol:
                hit_step = t
                break

            if args.no_progress_stop:
                if new_dist < best_dist - args.improve_eps:
                    best_dist = new_dist
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                if t >= args.min_steps and no_improve_count >= args.patience:
                    break

        # Episode End
        final_dist = np.linalg.norm(xyz - tgt)
        final_dists.append(final_dist)
        final_steps = hit_step if hit_step is not None else used_steps
        steps_used.append(final_steps)
        path_eff.append(start_dist / path_len if path_len > 1e-6 else 0.0)

        if hit_step is not None:
            for K in K_LIST:
                if hit_step <= K:
                    success_at[K] += 1

        rows.append([
            ep, start_dist, final_dist, 
            hit_step if hit_step is not None else -1, 
            final_steps, path_len, 
            1 if hit_step is not None else 0
        ])
        
        # Update progress bar description with current success rate
        current_succ_rate = success_at[40] / (ep + 1)
        iterator.set_postfix({"Succ@40": f"{current_succ_rate:.2%}"})

    # Stats
    sd = np.array(start_dists)
    fd = np.array(final_dists)
    
    print(f"\nStart dist: mean={sd.mean():.3f}")
    print(f"Success@40: {success_at[40]}/{args.episodes} ({success_at[40]/args.episodes:.3f})")
    print(f"Final dist: mean={fd.mean():.3f} med={np.median(fd):.3f}")
    
    out_csv = os.path.join(args.outdir, "rollouts_32i_v3_4_optimized.csv")
    np.savetxt(out_csv, np.array(rows), delimiter=",", header="ep,start,final,hit,steps,len,succ", comments="")
    print(f"[ok] saved {out_csv}")

if __name__ == "__main__":
    main()