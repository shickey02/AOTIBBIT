#!/usr/bin/env python3
# PHASE 32I v3.5 — Beam Search Planning Control
#
# Changes vs v3.4:
#   - Replaces greedy "Best-1" logic with Short-Horizon Beam Search.
#   - "Receding Horizon": Plans H steps ahead, executes 1, re-plans.
#   - Massive boost in topological navigation (avoids dead ends).

import os, argparse
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    # Data Paths
    ap.add_argument("--latents", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--outdir", required=True)
    
    # Mining Params
    ap.add_argument("--step", type=float, default=0.35)
    ap.add_argument("--fine_step", type=float, default=0.0875)
    ap.add_argument("--tol", type=float, default=0.12)
    ap.add_argument("--n_pairs", type=int, default=2500000)
    
    # Beam Search Params (THE NEW STUFF)
    ap.add_argument("--beam_width", type=int, default=5, help="Number of paths to keep")
    ap.add_argument("--beam_horizon", type=int, default=3, help="Steps to look ahead")
    
    # Rollout Params
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--max_steps", type=int, default=40)
    ap.add_argument("--target_radius", type=float, default=1.5)
    ap.add_argument("--target_mode", default="ball", choices=["ball"])
    ap.add_argument("--success_tol", type=float, default=0.10)
    ap.add_argument("--target_sampling", choices=["fixed", "legacy"], default="fixed")
    
    # Snapping
    ap.add_argument("--snap", action="store_true")
    ap.add_argument("--snap_metric", choices=["latent", "predxyz"], default="predxyz")
    ap.add_argument("--snap_k", type=int, default=50)
    ap.add_argument("--snap_min_step", type=float, default=0.1)
    ap.add_argument("--near_radius", type=float, default=0.50)
    ap.add_argument("--snap_min_step_near", type=float, default=0.0)
    
    # Misc
    ap.add_argument("--force_antisym", action="store_true")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # 1. Load Data & Precompute Affine
    print(f"Loading data from {args.latents}...")
    Z = np.load(args.latents).astype(np.float32)
    lab = np.load(args.labels)
    XYZ = np.stack([lab["dx"], lab["dy"], lab["dz"]], axis=1).astype(np.float32)
    N, D = Z.shape

    Z1 = np.concatenate([Z, np.ones((N, 1), np.float32)], axis=1)
    A, *_ = np.linalg.lstsq(Z1, XYZ, rcond=None)
    W_affine = A[:D]
    b_affine = A[D]

    def z_to_xyz_batch(z_batch):
        return z_batch @ W_affine + b_affine

    # 2. Mine Operators (Vectorized)
    print("Mining operators...")
    idx_i = rng.integers(0, N, size=args.n_pairs, endpoint=False)
    idx_j = rng.integers(0, N, size=args.n_pairs, endpoint=False)
    dXYZ = XYZ[idx_j] - XYZ[idx_i]
    norms = np.linalg.norm(dXYZ, axis=1)
    valid_mask = np.abs(norms - args.step) <= args.tol
    
    dXYZ = dXYZ[valid_mask]
    u = dXYZ / (norms[valid_mask, None] + 1e-8)
    
    dir_vecs = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=np.float32)
    dir_keys = ["+x", "-x", "+y", "-y", "+z", "-z"]
    
    dots = u @ dir_vecs.T
    best_dir = np.argmax(dots, axis=1)
    valid_dir = np.max(dots, axis=1) > 0.95
    
    final_i = idx_i[valid_mask][valid_dir]
    final_j = idx_j[valid_mask][valid_dir]
    final_dir = best_dir[valid_dir]
    
    dZ = Z[final_j] - Z[final_i]
    
    proto_coarse = {}
    for i, k in enumerate(dir_keys):
        dz_k = dZ[final_dir == i]
        dz_k = dz_k[np.linalg.norm(dz_k, axis=1) > 1e-6]
        if len(dz_k) > 0:
            proto_coarse[k] = np.mean(dz_k, axis=0)

    if args.force_antisym:
        for ax in ["x", "y", "z"]:
            p, n = "+" + ax, "-" + ax
            if p in proto_coarse and n in proto_coarse:
                avg = (proto_coarse[p] - proto_coarse[n]) / 2.0
                proto_coarse[p], proto_coarse[n] = avg, -avg

    scale = float(args.fine_step / args.step)
    proto_fine = {k: proto_coarse[k] * scale for k in dir_keys}
    
    # 3. Stack Operators for Beam Search
    # 12 operators: [coarse_0..5, fine_0..5]
    ops_matrix = np.stack([proto_coarse[k] for k in dir_keys] + [proto_fine[k] for k in dir_keys])
    NUM_OPS = 12

    # 4. Snap Setup
    nbrs = None
    if args.snap:
        print(f"Building KDTree (k={args.snap_k})...")
        snap_space = Z if args.snap_metric == "latent" else XYZ
        nbrs = NearestNeighbors(n_neighbors=args.snap_k + 1, algorithm='auto', n_jobs=-1).fit(snap_space)

    def do_snap_batch(z_batch, min_step):
        if not args.snap: return z_batch
        B = len(z_batch)
        q = z_batch if args.snap_metric == "latent" else z_to_xyz_batch(z_batch)
        _, inds = nbrs.kneighbors(q)
        z_out = z_batch.copy()
        for i in range(B):
            for k_idx in range(inds.shape[1]):
                idx = inds[i, k_idx]
                if np.linalg.norm(z_batch[i] - Z[idx]) >= min_step:
                    z_out[i] = Z[idx]; break
        return z_out

    # ---------------------------------------------------------
    # BEAM SEARCH LOGIC
    # ---------------------------------------------------------
    def run_beam_search(z_start, xyz_start, target, beam_width, horizon, snap_min_step):
        """
        Returns: (z_next, xyz_next) corresponding to the FIRST STEP of the best path found.
        """
        # Beam items: (score, z_state, first_action_index)
        # Score = negative distance (so higher is better, or just minimize dist)
        # Actually let's store (dist, z, first_op_idx). We want to minimize dist.
        
        # Initialize beam: Apply all 12 ops to start
        # This determines the "first action" for the path
        
        # 1. Expand Root
        z_cands = z_start + ops_matrix # (12, D)
        if args.snap:
            z_cands = do_snap_batch(z_cands, snap_min_step)
        xyz_cands = z_to_xyz_batch(z_cands) # (12, 3)
        dists = np.linalg.norm(xyz_cands - target, axis=1) # (12,)
        
        # Sort and pick top K to start the beam
        # We store tuples: (current_z, first_op_idx)
        # We assume the "cost" is the heuristic distance at the leaf
        
        sorted_idx = np.argsort(dists)
        beam = []
        for i in range(min(beam_width, NUM_OPS)):
            idx = sorted_idx[i]
            beam.append({
                "z": z_cands[idx], 
                "first_op": idx, 
                "dist": dists[idx]
            })
            
        # 2. Expand Beam for H-1 steps
        for h in range(horizon - 1):
            next_beam_candidates = []
            
            # Form batch of all expansions: (Width * 12, D)
            current_zs = np.stack([b["z"] for b in beam]) # (K, D)
            
            # Expand: (K, 1, D) + (1, 12, D) -> (K, 12, D)
            # This broadcasts to add every op to every beam item
            expanded_zs = current_zs[:, None, :] + ops_matrix[None, :, :]
            expanded_zs = expanded_zs.reshape(-1, D) # Flatten
            
            # Snap all at once
            if args.snap:
                expanded_zs = do_snap_batch(expanded_zs, snap_min_step)
            
            # Eval all
            expanded_xyz = z_to_xyz_batch(expanded_zs)
            expanded_dists = np.linalg.norm(expanded_xyz - target, axis=1)
            
            # Collect candidates with their provenance
            # We need to map back to which 'first_op' they came from
            num_beam = len(beam)
            for i in range(num_beam * NUM_OPS):
                parent_idx = i // NUM_OPS
                op_idx = i % NUM_OPS
                
                # Provenance: Keep the first_op of the parent
                origin_op = beam[parent_idx]["first_op"]
                
                next_beam_candidates.append({
                    "z": expanded_zs[i],
                    "first_op": origin_op,
                    "dist": expanded_dists[i]
                })
            
            # Sort all candidates by distance and prune to K
            next_beam_candidates.sort(key=lambda x: x["dist"])
            beam = next_beam_candidates[:beam_width]
            
        # 3. Return action of the winner
        best_path = beam[0]
        best_op_idx = best_path["first_op"]
        
        # Re-compute the immediate next step (deterministic)
        # We simply re-apply the best first op to z_start
        # (Alternatively we could have cached it, but this is cheap)
        z_next_step = z_start + ops_matrix[best_op_idx]
        if args.snap:
            z_next_step = do_snap_batch(z_next_step[None, :], snap_min_step)[0]
        xyz_next_step = z_to_xyz_batch(z_next_step[None, :])[0]
        
        return z_next_step, xyz_next_step, best_path["dist"]

    # ---------------------------------------------------------
    # EPISODE LOOP
    # ---------------------------------------------------------
    rows = []
    success_at = {k:0 for k in [5,10,20,40]}
    
    print(f"\nStarting {args.episodes} episodes (Beam Width={args.beam_width}, Horizon={args.beam_horizon})...")
    iterator = tqdm(range(args.episodes), desc="Rollouts")
    
    for ep in iterator:
        s = int(rng.integers(0, N))
        z = Z[s].copy()
        xyz = z_to_xyz_batch(z[None, :])[0]

        # Target
        if args.target_sampling == "fixed":
            vec = rng.normal(size=3).astype(np.float32); vec /= np.linalg.norm(vec)
            tgt = xyz + vec * args.target_radius
        else:
            tgt = xyz + rng.normal(size=3).astype(np.float32)
            tgt = xyz + (tgt/np.linalg.norm(tgt)) * args.target_radius

        start_dist = np.linalg.norm(xyz - tgt)
        hit_step = None
        path_len = 0.0
        
        for t in range(1, args.max_steps + 1):
            dist = np.linalg.norm(xyz - tgt)
            
            # Success check
            if dist <= args.success_tol:
                hit_step = t - 1
                break
                
            # Params
            min_step = args.snap_min_step_near if dist < args.near_radius else args.snap_min_step
            
            # PLAN & ACT (Receding Horizon)
            # We treat the greedy step as a beam search with W=1, H=1.
            # Here we use the user's W and H.
            z_next, xyz_next, projected_dist = run_beam_search(
                z, xyz, tgt, 
                args.beam_width, 
                args.beam_horizon, 
                min_step
            )
            
            # Update state
            path_len += np.linalg.norm(xyz_next - xyz)
            z, xyz = z_next, xyz_next
            
            # Immediate success check after move
            if np.linalg.norm(xyz - tgt) <= args.success_tol:
                hit_step = t
                break
                
        # Stats
        final_dist = np.linalg.norm(xyz - tgt)
        if hit_step:
            for k in success_at: 
                if hit_step <= k: success_at[k] += 1
        
        rows.append([ep, start_dist, final_dist, hit_step if hit_step else -1, path_len, 1 if hit_step else 0])
        iterator.set_postfix({"Succ": f"{success_at[40]/(ep+1):.1%}"})

    # Save
    out_csv = os.path.join(args.outdir, f"rollouts_beam_w{args.beam_width}_h{args.beam_horizon}.csv")
    np.savetxt(out_csv, np.array(rows), delimiter=",", header="ep,start,final,hit,len,succ", comments="")
    print(f"\n[ok] Saved {out_csv}")

if __name__ == "__main__":
    main()