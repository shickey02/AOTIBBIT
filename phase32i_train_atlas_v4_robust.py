#!/usr/bin/env python3
# PHASE 32I v4.1 — ROBUST Atlas Builder
# 
# Fixes the "Cracked Atlas" problem by:
# 1. Mining GLOBAL operators first as a baseline.
# 2. Filling "empty slots" in local charts with Global operators.
# 3. Enforcing ANTI-SYMMETRY locally (reduces drift).

import argparse, pickle
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True, help="Path to save atlas_robust.pkl")
    
    ap.add_argument("--k_clusters", type=int, default=64)
    ap.add_argument("--n_pairs", type=int, default=5000000)
    ap.add_argument("--min_chart_count", type=int, default=50, help="Min pairs to trust local op")
    
    ap.add_argument("--step", type=float, default=0.35)
    ap.add_argument("--fine_step", type=float, default=0.0875)
    ap.add_argument("--tol", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # 1. Load Data
    print(f"Loading {args.latents}...")
    Z = np.load(args.latents).astype(np.float32)
    lab = np.load(args.labels)
    XYZ = np.stack([lab["dx"], lab["dy"], lab["dz"]], axis=1).astype(np.float32)
    N = len(Z)

    # 2. Cluster (The Neighborhoods)
    print(f"Clustering (K={args.k_clusters})...")
    kmeans = MiniBatchKMeans(n_clusters=args.k_clusters, random_state=args.seed, batch_size=2048)
    cluster_labels = kmeans.fit_predict(Z)
    centroids = kmeans.cluster_centers_.astype(np.float32)

    # 3. Mine Pairs (Massive Batch)
    print(f"Mining {args.n_pairs} pairs...")
    idx_i = rng.integers(0, N, size=args.n_pairs)
    idx_j = rng.integers(0, N, size=args.n_pairs)

    dXYZ = XYZ[idx_j] - XYZ[idx_i]
    norms = np.linalg.norm(dXYZ, axis=1)
    mask = np.abs(norms - args.step) <= args.tol
    
    idx_i, idx_j = idx_i[mask], idx_j[mask]
    dXYZ = dXYZ[mask]
    dZ = Z[idx_j] - Z[idx_i]
    
    # Classify Directions
    u = dXYZ / (norms[mask, None] + 1e-8)
    dir_vecs = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=np.float32)
    dir_keys = ["+x", "-x", "+y", "-y", "+z", "-z"]
    
    dots = u @ dir_vecs.T
    best_dir = np.argmax(dots, axis=1)
    valid_dir = np.max(dots, axis=1) > 0.95
    
    final_i = idx_i[valid_dir]
    final_d = best_dir[valid_dir]
    final_dz = dZ[valid_dir]
    
    # 4. Calculate GLOBAL Baseline (The "Safety Net")
    print("Computing Global Fallbacks...")
    global_ops = {}
    for i, k in enumerate(dir_keys):
        ops = final_dz[final_d == i]
        if len(ops) > 0:
            global_ops[k] = np.mean(ops, axis=0)
        else:
            global_ops[k] = np.zeros(Z.shape[1], dtype=np.float32)

    # Enforce Global Anti-Symmetry
    for ax in ["x","y","z"]:
        p, n = f"+{ax}", f"-{ax}"
        avg = (global_ops[p] - global_ops[n]) / 2.0
        global_ops[p], global_ops[n] = avg, -avg

    # 5. Build Local Charts (The "Atlas")
    print("Building Local Charts...")
    atlas = {"k": args.k_clusters, "centroids": centroids, "charts": []}
    scale_factor = args.fine_step / args.step
    labels_i = cluster_labels[final_i]

    fallback_count = 0

    for k in tqdm(range(args.k_clusters)):
        chart = {"coarse": {}, "fine": {}, "counts": {}}
        
        # Get pairs starting in this cluster
        mask_k = (labels_i == k)
        dz_k = final_dz[mask_k]
        dir_k = final_d[mask_k]
        
        # Raw Local Ops
        local_raw = {}
        for i, key in enumerate(dir_keys):
            moves = dz_k[dir_k == i]
            moves = moves[np.linalg.norm(moves, axis=1) > 1e-6] # Filter noise
            
            count = len(moves)
            chart["counts"][key] = count
            
            # RULE 1: If insufficient data, use Global Fallback
            if count < args.min_chart_count:
                local_raw[key] = global_ops[key]
                fallback_count += 1
            else:
                local_raw[key] = np.mean(moves, axis=0)

        # RULE 2: Enforce Local Anti-Symmetry
        for ax in ["x","y","z"]:
            p, n = f"+{ax}", f"-{ax}"
            # Average the positive and inverted negative
            avg = (local_raw[p] - local_raw[n]) / 2.0
            
            # Store Coarse
            chart["coarse"][p] = avg
            chart["coarse"][n] = -avg
            
            # Store Fine (Scaled)
            chart["fine"][p] = avg * scale_factor
            chart["fine"][n] = -avg * scale_factor

        atlas["charts"].append(chart)

    print(f"\nAtlas built. Fallbacks used: {fallback_count} times (across {args.k_clusters * 6} slots).")
    print(f"Saving to {args.out}...")
    with open(args.out, "wb") as f:
        pickle.dump(atlas, f)

if __name__ == "__main__":
    main()