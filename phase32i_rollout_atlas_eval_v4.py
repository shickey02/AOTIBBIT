#!/usr/bin/env python3
import argparse, pickle, os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--atlas", required=True, help="Path to atlas.pkl")
    ap.add_argument("--outdir", required=True)
    
    # Rollout Params
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--max_steps", type=int, default=40)
    ap.add_argument("--target_radius", type=float, default=1.5)
    ap.add_argument("--success_tol", type=float, default=0.10)
    ap.add_argument("--target_sampling", default="fixed")
    
    # Snap
    ap.add_argument("--snap", action="store_true")
    ap.add_argument("--snap_k", type=int, default=50)
    ap.add_argument("--snap_min_step", type=float, default=0.1)
    
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # 1. Load Data & Atlas
    print("Loading Data & Atlas...")
    Z = np.load(args.latents).astype(np.float32)
    lab = np.load(args.labels)
    XYZ = np.stack([lab["dx"], lab["dy"], lab["dz"]], axis=1).astype(np.float32)
    
    with open(args.atlas, "rb") as f:
        atlas = pickle.load(f)
    
    centroids = atlas["centroids"] # (K, D)
    charts = atlas["charts"]       # List of dicts
    
    # 2. Precompute Affine (Global)
    N, D = Z.shape
    Z1 = np.concatenate([Z, np.ones((N, 1), np.float32)], axis=1)
    A, *_ = np.linalg.lstsq(Z1, XYZ, rcond=None)
    W_aff, b_aff = A[:D], A[D]
    
    def z_to_xyz(z_vec): return z_vec @ W_aff + b_aff

    # 3. Snap Tree
    nbrs = None
    if args.snap:
        print("Building Snap Tree...")
        nbrs = NearestNeighbors(n_neighbors=args.snap_k+1, algorithm='auto', n_jobs=1).fit(XYZ)

    def do_snap(z_cand, xyz_cand, min_step):
        if not args.snap: return z_cand, xyz_cand
        _, inds = nbrs.kneighbors(xyz_cand[None,:])
        for idx in inds[0]:
            z_neighbor = Z[idx]
            if np.linalg.norm(z_cand - z_neighbor) >= min_step:
                return z_neighbor, XYZ[idx]
        return z_cand, xyz_cand

    # 4. Atlas Lookup Helper
    # We use a simple dot product or distance to find nearest centroid
    def get_local_ops(z_curr):
        # Find nearest centroid
        dists = np.linalg.norm(centroids - z_curr, axis=1)
        k = np.argmin(dists)
        chart = charts[k]
        
        # Collect available ops (some might be missing in sparse clusters)
        ops = []
        # Coarse
        for d in ["+x","-x","+y","-y","+z","-z"]:
            if d in chart["coarse"]: ops.append(chart["coarse"][d])
        # Fine
        for d in ["+x","-x","+y","-y","+z","-z"]:
            if d in chart["fine"]: ops.append(chart["fine"][d])
            
        if len(ops) == 0: return np.zeros((1, D), dtype=np.float32)
        return np.stack(ops)

    # 5. Rollout Loop (Greedy Best-1 for Speed/Proof of Concept)
    print("Starting Atlas Rollouts...")
    success_count = 0
    final_dists = []
    
    iterator = tqdm(range(args.episodes))
    for ep in iterator:
        s = int(rng.integers(0, N))
        z, xyz = Z[s].copy(), z_to_xyz(Z[s])
        
        # Target
        if args.target_sampling == "fixed":
            v = rng.normal(size=3).astype(np.float32); v /= np.linalg.norm(v)
            tgt = xyz + v * args.target_radius
        else:
            tgt = xyz + rng.normal(size=3).astype(np.float32)
            tgt = xyz + (tgt/np.linalg.norm(tgt)) * args.target_radius
            
        hit = False
        
        for t in range(args.max_steps):
            dist = np.linalg.norm(xyz - tgt)
            if dist <= args.success_tol:
                hit = True; break
            
            # 1. GET LOCAL OPS
            ops = get_local_ops(z)
            
            # 2. Expand
            z_cands = z + ops
            xyz_cands = z_to_xyz(z_cands) # (NumOps, 3)
            dists = np.linalg.norm(xyz_cands - tgt, axis=1)
            
            best_idx = np.argmin(dists)
            
            # 3. Snap & Move
            # Note: We snap AFTER picking the best direction to save compute
            z_next_raw = z_cands[best_idx]
            xyz_next_raw = xyz_cands[best_idx]
            
            z_next, xyz_next = do_snap(z_next_raw, xyz_next_raw, args.snap_min_step)
            
            z, xyz = z_next, xyz_next
            
            if np.linalg.norm(xyz - tgt) <= args.success_tol:
                hit = True; break
        
        if hit: success_count += 1
        final_dists.append(np.linalg.norm(xyz - tgt))
        iterator.set_postfix({"Succ": f"{success_count/(ep+1):.1%}"})

    print(f"\nFinal Success: {success_count}/{args.episodes} ({success_count/args.episodes:.3f})")
    print(f"Median Dist: {np.median(final_dists):.3f}")

if __name__ == "__main__":
    main()