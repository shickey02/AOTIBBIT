#!/usr/bin/env python3
# VISUAL ARITHMETIC (ATLAS) EVALUATION v3.0
# 
# Hypothesis: Logic is Multimodal. There isn't one way to "Add One."
# Task: Learn K distinct "strategies" for each transition and test via search.

import argparse, os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    # USE THE STANDARD AE LATENTS (Sharper geometry)
    ap.add_argument("--latents", default="./data/mnist_bbit/latents_mnist.npy")
    ap.add_argument("--labels", default="./data/mnist_bbit/labels_mnist.npy")
    ap.add_argument("--k_strategies", type=int, default=5, help="Number of distinct vectors to learn per transition")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    rng = np.random.default_rng(args.seed)

    # 1. Load Data
    print(f"Loading {args.latents}...")
    Z = np.load(args.latents).astype(np.float32)
    Y = np.load(args.labels).astype(np.int32)
    
    Z_train, Z_test, Y_train, Y_test = train_test_split(Z, Y, test_size=0.2, random_state=args.seed)
    
    # 2. Build Oracle (KNN)
    print("Building Oracle...")
    knn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
    knn.fit(Z_test, Y_test)

    print(f"\n--- EVALUATION: ATLAS ARITHMETIC (K={args.k_strategies}) ---")
    print(f"{'Transition':<12} | {'Single Vec':<10} | {'Atlas (Best-of-K)':<18}")
    print("-" * 50)
    
    global_accs = []
    atlas_accs = []
    
    for start_digit in range(9):
        end_digit = start_digit + 1
        
        # --- TRAINING ---
        # 1. Get all pairs
        src_idxs = np.where(Y_train == start_digit)[0]
        tgt_idxs = np.where(Y_train == end_digit)[0]
        
        # We need a lot of pairs to cluster strategies efficiently
        # Let's sample 20,000 pairs randomly
        n_pairs = 20000
        i_s = rng.choice(src_idxs, size=n_pairs)
        j_s = rng.choice(tgt_idxs, size=n_pairs)
        
        # 2. Calculate ALL raw vectors
        raw_vectors = Z_train[j_s] - Z_train[i_s]
        
        # 3. Strategy A: The Global Average (Baseline)
        global_op = np.mean(raw_vectors, axis=0)
        
        # 4. Strategy B: The Atlas (Cluster the vectors)
        kmeans = MiniBatchKMeans(n_clusters=args.k_strategies, random_state=args.seed, batch_size=1024, n_init="auto")
        kmeans.fit(raw_vectors)
        strategies = kmeans.cluster_centers_ # (K, D)
        
        # --- TESTING ---
        test_idxs = np.where(Y_test == start_digit)[0]
        z_input = Z_test[test_idxs]
        
        # 1. Test Baseline (Single Vector)
        z_pred_global = z_input + global_op
        y_pred_global = knn.predict(z_pred_global)
        acc_global = np.mean(y_pred_global == end_digit)
        
        # 2. Test Atlas (Try ALL K strategies)
        # We want to see if *any* strategy worked for a given input.
        # Shape: (N_test, K, D)
        z_input_expanded = z_input[:, None, :] # (N, 1, D)
        strategies_expanded = strategies[None, :, :] # (1, K, D)
        
        # Apply all K strategies to all N inputs
        z_pred_atlas = z_input_expanded + strategies_expanded # (N, K, D)
        
        # Flatten to query KNN: (N*K, D)
        N_test = len(z_input)
        z_pred_flat = z_pred_atlas.reshape(-1, Z.shape[1])
        
        # Batch Predict
        y_pred_flat = knn.predict(z_pred_flat)
        y_pred_atlas = y_pred_flat.reshape(N_test, args.k_strategies) # (N, K)
        
        # Check if Target exists in the K predictions
        # (Did ANY strategy produce the number we wanted?)
        hits = np.any(y_pred_atlas == end_digit, axis=1)
        acc_atlas = np.mean(hits)
        
        global_accs.append(acc_global)
        atlas_accs.append(acc_atlas)
        
        print(f"{start_digit} -> {end_digit:<6} | {acc_global:.2%}     | {acc_atlas:.2%}")

    print("-" * 50)
    print(f"AVERAGE      | {np.mean(global_accs):.2%}     | {np.mean(atlas_accs):.2%}")

if __name__ == "__main__":
    main()