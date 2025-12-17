#!/usr/bin/env python3
# VISUAL ARITHMETIC EVALUATION v1.0
# 
# Hypothesis: Reasoning is geometric navigation.
# Task: Learn the concept of "Adding One" as a vector in latent space.
# Data: MNIST Latents (Z) and Labels (Y).
#
# Process:
# 1. Mine the "+1" Operator from training data (e.g., 2->3, 8->9).
# 2. Apply it to test data (e.g., take a "5", add vector, see if it becomes a "6").
# 3. Success = The nearest neighbor of (z_5 + vec) is a "6".

import argparse, os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True, help="Path to MNIST latents .npy (N, D)")
    ap.add_argument("--labels", required=True, help="Path to MNIST labels .npy (N,)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n_pairs", type=int, default=100000, help="Pairs to mine for the operator")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # 1. Load Data
    print("Loading Data...")
    Z = np.load(args.latents).astype(np.float32)
    Y = np.load(args.labels).astype(np.int32)
    N, D = Z.shape
    
    # Split Train/Test (We learn the math on Train, test on Test)
    # This ensures we aren't just memorizing pairs.
    Z_train, Z_test, Y_train, Y_test = train_test_split(Z, Y, test_size=0.2, random_state=args.seed)
    
    print(f"Train: {len(Z_train)}, Test: {len(Z_test)}")

    # 2. Mine the "+1" Operator (The "Logic Vector")
    print("Mining the '+1' Operator...")
    
    # We want pairs (i, j) where Label(j) - Label(i) == 1
    # Optimization: Filter indices by class first to avoid N^2 scan
    class_idxs = [np.where(Y_train == c)[0] for c in range(10)]
    
    deltas = []
    
    # We sample balanced pairs from 0->1, 1->2, ... 8->9
    # Note: We do NOT use 9->0 (that's modular arithmetic, we want linear)
    pairs_per_transition = args.n_pairs // 9
    
    for c in tqdm(range(9), desc="Mining Transitions"):
        # Source digits (c) and Target digits (c+1)
        src_idxs = class_idxs[c]
        tgt_idxs = class_idxs[c+1]
        
        # Randomly pair them up
        i_s = rng.choice(src_idxs, size=pairs_per_transition)
        j_s = rng.choice(tgt_idxs, size=pairs_per_transition)
        
        # Calculate vector: z_target - z_source
        # This is the "direction of adding one"
        dz = Z_train[j_s] - Z_train[i_s]
        deltas.append(dz)
        
    # Stack and Average to get the Global "+1" Operator
    all_deltas = np.vstack(deltas)
    plus_one_op = np.mean(all_deltas, axis=0)
    
    # Optional: Anti-Symmetry (Adding 1 is opposite of Subtracting 1)
    # We could mine "-1" and average them, but let's stick to simple addition first.
    
    print(f"Operator Magnitude: {np.linalg.norm(plus_one_op):.4f}")

    # 3. The Reasoning Test
    # We build a Classifier on the Test Set to check "Where did we land?"
    # K=1 Nearest Neighbor is the strictest geometric test.
    print("Building Geometric Oracle (KNN)...")
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(Z_test, Y_test)
    
    print("\n--- EVALUATION: VISUAL ARITHMETIC ---")
    
    total_acc = 0
    class_acc = {}
    
    # Test on each digit 0..8 (9+1 is out of bounds for single digits)
    for c in range(9):
        # Get all instances of number 'c' in Test Set
        idxs = np.where(Y_test == c)[0]
        z_start = Z_test[idxs]
        
        # APPLY THE OPERATOR: z_new = z_old + (+1_vec)
        z_pred = z_start + plus_one_op
        
        # Check where we landed
        # We expect the nearest neighbor to be class 'c+1'
        y_pred = knn.predict(z_pred)
        y_true = c + 1
        
        # Accuracy
        correct = np.sum(y_pred == y_true)
        acc = correct / len(idxs)
        
        class_acc[c] = acc
        print(f"Input {c} + 1  ->  Expected {c+1}  | Accuracy: {acc:.2%}")
        
    # Global Accuracy
    avg_acc = np.mean(list(class_acc.values()))
    print("-" * 40)
    print(f"Overall Arithmetic Reasoning Accuracy: {avg_acc:.2%}")
    print("-" * 40)
    
    # 4. Save Results
    results = {
        "operator": plus_one_op,
        "accuracy": avg_acc,
        "per_digit": class_acc
    }
    np.save(os.path.join(args.outdir, "arithmetic_results.npy"), results)

if __name__ == "__main__":
    main()