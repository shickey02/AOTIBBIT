#!/usr/bin/env python3
# VISUAL ARITHMETIC (LOCAL) EVALUATION v2.0
# 
# Hypothesis: The manifold is locally consistent even if globally twisted.
# Task: Learn specific transitions (e.g., "1->2") and test on unseen data.

import argparse, os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--n_pairs", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    rng = np.random.default_rng(args.seed)

    # 1. Load Data
    print("Loading Data...")
    Z = np.load(args.latents).astype(np.float32)
    Y = np.load(args.labels).astype(np.int32)
    
    Z_train, Z_test, Y_train, Y_test = train_test_split(Z, Y, test_size=0.2, random_state=args.seed)
    
    # 2. Build Classifier (Oracle)
    print("Building Oracle...")
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(Z_test, Y_test) # We verify against the test set geometry

    print("\n--- EVALUATION: LOCAL ARITHMETIC ---")
    print(f"{'Transition':<12} | {'Vector Mag':<10} | {'Accuracy':<10}")
    print("-" * 40)
    
    accuracies = []
    
    # Mine and Test each transition separately
    for start_digit in range(9):
        end_digit = start_digit + 1
        
        # --- TRAINING (Mining the specific vector) ---
        src_idxs = np.where(Y_train == start_digit)[0]
        tgt_idxs = np.where(Y_train == end_digit)[0]
        
        # Sample pairs
        i_s = rng.choice(src_idxs, size=args.n_pairs)
        j_s = rng.choice(tgt_idxs, size=args.n_pairs)
        
        # Calculate the specific "Transformation Vector"
        # e.g., The average move required to turn a 0 into a 1
        vectors = Z_train[j_s] - Z_train[i_s]
        local_op = np.mean(vectors, axis=0)
        mag = np.linalg.norm(local_op)
        
        # --- TESTING (Reasoning on unseen data) ---
        # Get all "Start Digits" from the Test Set
        test_idxs = np.where(Y_test == start_digit)[0]
        z_input = Z_test[test_idxs]
        
        # Apply the learned logic
        z_pred = z_input + local_op
        
        # Check result
        y_pred = knn.predict(z_pred)
        acc = np.mean(y_pred == end_digit)
        accuracies.append(acc)
        
        print(f"{start_digit} -> {end_digit:<6} | {mag:.4f}     | {acc:.2%}")

    print("-" * 40)
    print(f"Average Local Reasoning Accuracy: {np.mean(accuracies):.2%}")

if __name__ == "__main__":
    main()