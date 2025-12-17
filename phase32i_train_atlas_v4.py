#!/usr/bin/env python3
import argparse, pickle
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True, help="Path to save atlas.pkl")
    
    # Hyperparams
    ap.add_argument("--k_clusters", type=int, default=64)
    ap.add_argument("--n_pairs", type=int, default=5000000) # Need high count to fill clusters
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

    # 2. Build Clusters (The "Regions" of the Atlas)
    print(f"Clustering Latent Space (K={args.k_clusters})...")
    kmeans = MiniBatchKMeans(n_clusters=args.k_clusters, random_state=args.seed, batch_size=2048)
    cluster_labels = kmeans.fit_predict(Z)
    centroids = kmeans.cluster_centers_.astype(np.float32)

    # 3. Mine Pairs (Global Search, Local Assignment)
    print(f"Mining {args.n_pairs} pairs to populate charts...")
    idx_i = rng.integers(0, N, size=args.n_pairs)
    idx_j = rng.integers(0, N, size=args.n_pairs)

    # Fast XYZ Filter
    dXYZ = XYZ[idx_j] - XYZ[idx_i]
    norms = np.linalg.norm(dXYZ, axis=1)
    mask = np.abs(norms - args.step) <= args.tol
    
    idx_i, idx_j = idx_i[mask], idx_j[mask]
    dXYZ = dXYZ[mask]
    norms = norms[mask]

    # Direction Check
    u = dXYZ / (norms[:, None] + 1e-8)
    dir_vecs = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=np.float32)
    dir_keys = ["+x", "-x", "+y", "-y", "+z", "-z"]
    
    dots = u @ dir_vecs.T
    best_dir = np.argmax(dots, axis=1)
    valid_dir = np.max(dots, axis=1) > 0.95

    # Filtered lists
    final_i = idx_i[valid_dir]
    final_j = idx_j[valid_dir]
    final_d = best_dir[valid_dir]
    
    # 4. Assign Operators to Clusters
    # Logic: An operator belongs to Cluster K if the START point (z_i) is in Cluster K
    print("Assigning vectors to local charts...")
    dZ = Z[final_j] - Z[final_i]
    labels_i = cluster_labels[final_i] # Which cluster does the move start from?

    atlas = {
        "k": args.k_clusters,
        "centroids": centroids,
        "charts": []
    }

    # Build Charts
    scale_factor = args.fine_step / args.step
    empty_clusters = 0

    for k in tqdm(range(args.k_clusters)):
        chart = {"coarse": {}, "fine": {}, "counts": {}}
        
        # Get all moves starting in this cluster
        mask_k = (labels_i == k)
        dz_k_all = dZ[mask_k]
        dirs_k_all = final_d[mask_k]

        for di, d_key in enumerate(dir_keys):
            # Moves in specific direction
            moves = dz_k_all[dirs_k_all == di]
            
            # Filter noise
            moves = moves[np.linalg.norm(moves, axis=1) > 1e-6]
            
            if len(moves) > 0:
                # The "Local Operator" is the average of moves within this cluster
                op = np.mean(moves, axis=0)
                chart["coarse"][d_key] = op
                chart["fine"][d_key] = op * scale_factor
                chart["counts"][d_key] = len(moves)
            else:
                # Fallback: If a cluster has NO data for "+x", what do we do?
                # For now, we leave it empty. The rollout will need to handle "missing ops".
                pass
        
        if len(chart["coarse"]) < 6:
            # Heuristic: If missing directions, borrow from global average or nearest neighbor?
            # For v4.0, we just log it.
            pass

        atlas["charts"].append(chart)

    print(f"\nAtlas built. Saving to {args.out}...")
    with open(args.out, "wb") as f:
        pickle.dump(atlas, f)
    print("Done.")

if __name__ == "__main__":
    main()