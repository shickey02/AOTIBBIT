#!/usr/bin/env python3
import argparse
import numpy as np
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--step", type=float, default=0.35)
    ap.add_argument("--tol", type=float, default=0.12)
    ap.add_argument("--n_pairs", type=int, default=100000)
    args = ap.parse_args()

    # Load
    print("Loading...")
    Z = np.load(args.latents).astype(np.float32)
    lab = np.load(args.labels)
    XYZ = np.stack([lab["dx"], lab["dy"], lab["dz"]], axis=1).astype(np.float32)
    N = len(Z)
    rng = np.random.default_rng(42)

    # 1. Mine Global Operators (The "Old" Way)
    print("Mining Global Operators...")
    idx_i = rng.integers(0, N, size=args.n_pairs)
    idx_j = rng.integers(0, N, size=args.n_pairs)
    
    dXYZ = XYZ[idx_j] - XYZ[idx_i]
    norms = np.linalg.norm(dXYZ, axis=1)
    mask = np.abs(norms - args.step) <= args.tol
    
    dZ = Z[idx_j] - Z[idx_i]
    dXYZ = dXYZ[mask]
    dZ = dZ[mask]
    
    # Classify directions
    dirs = {
        "+x": [1,0,0], "-x": [-1,0,0],
        "+y": [0,1,0], "-y": [0,-1,0],
        "+z": [0,0,1], "-z": [0,0,-1]
    }
    
    global_ops = {}
    local_samples = {k: [] for k in dirs}
    
    u = dXYZ / (np.linalg.norm(dXYZ, axis=1, keepdims=True) + 1e-8)
    
    for k, v in dirs.items():
        vec = np.array(v, dtype=np.float32)
        dots = u @ vec
        # Get all pairs that match this direction
        matches = dZ[dots > 0.95]
        if len(matches) > 0:
            # The Global Operator is the MEAN
            global_ops[k] = np.mean(matches, axis=0)
            # Save the individual samples to check variance
            local_samples[k] = matches

    # 2. Measure Variance (The "Delusion" Check)
    print("\n--- DIAGNOSIS: GLOBAL OPERATOR QUALITY ---")
    print(f"{'Op':<4} | {'Count':<6} | {'Cos Sim (Mean)':<14} | {'Variance':<10}")
    print("-" * 45)
    
    for k in sorted(dirs.keys()):
        if k not in global_ops:
            print(f"{k:<4} | MISSING")
            continue
            
        op = global_ops[k]
        samples = local_samples[k]
        
        # Cosine Similarity between the Global Average and the Actual Local Moves
        # Logic: If the manifold is flat, every local move should look exactly like the average (Cos Sim = 1.0)
        # If curved, local moves will point in different directions (Cos Sim < 1.0)
        
        op_norm = op / np.linalg.norm(op)
        sample_norms = samples / (np.linalg.norm(samples, axis=1, keepdims=True) + 1e-8)
        
        sims = sample_norms @ op_norm
        mean_sim = np.mean(sims)
        
        # Variance of the vectors (how "spread out" are the local meanings of 'Left'?)
        var = np.var(samples)
        
        print(f"{k:<4} | {len(samples):<6} | {mean_sim:.4f}         | {var:.4f}")

    print("\nINTERPRETATION:")
    print("Cos Sim > 0.90:  Manifold is Flat. The operators are good. (Bug is elsewhere)")
    print("Cos Sim < 0.80:  Manifold is Curved. Global operators are meaningless.")
    print("                 YOU NEED LOCAL CHARTS (ATLAS).")

if __name__ == "__main__":
    main()