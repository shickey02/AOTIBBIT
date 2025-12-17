#!/usr/bin/env python3
# VISUAL ALGEBRA: ZERO-SHOT SUBTRACTION
# Task: Solve "x + 1 = 3". (i.e., Take '3', apply '-1', see if we land on '2')

import argparse
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", default="./data/mnist_bbit/latents_mnist.npy")
    ap.add_argument("--labels", default="./data/mnist_bbit/labels_mnist.npy")
    ap.add_argument("--k_strategies", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    Z = np.load(args.latents).astype(np.float32)
    Y = np.load(args.labels).astype(np.int32)
    Z_train, Z_test, Y_train, Y_test = train_test_split(Z, Y, test_size=0.2, random_state=args.seed)

    print("Mining Forward Strategies (+1)...")
    strategies = {}
    rng = np.random.default_rng(args.seed)
    
    for start_digit in range(9):
        end_digit = start_digit + 1
        src = Z_train[Y_train == start_digit]
        tgt = Z_train[Y_train == end_digit]
        
        # Mine vectors
        i_s = rng.choice(len(src), 10000)
        j_s = rng.choice(len(tgt), 10000)
        vecs = tgt[j_s] - src[i_s]
        
        # Cluster
        kmeans = MiniBatchKMeans(n_clusters=args.k_strategies, batch_size=1024, n_init="auto").fit(vecs)
        strategies[f"{start_digit}->{end_digit}"] = kmeans.cluster_centers_

    print("\n--- ZERO-SHOT SUBTRACTION (x + 1 = N) ---")
    print("We apply the INVERSE of the '+1' vectors to 'N'. Do we find 'N-1'?")
    
    knn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1).fit(Z_test, Y_test)
    
    for target_digit in range(1, 10): # 1..9
        expected_digit = target_digit - 1
        
        # We need to reverse the logic of "Expected -> Target"
        # i.e., to go 2->1, we use -(Vectors for 1->2)
        key = f"{expected_digit}->{target_digit}"
        forward_ops = strategies[key] # The "Add One" vectors
        backward_ops = -forward_ops   # The "Subtract One" vectors (Zero-Shot)
        
        # Get inputs (Target Digit)
        z_input = Z_test[Y_test == target_digit]
        if len(z_input) > 200: z_input = z_input[:200]
        
        # Apply Inverse Atlas
        # Try all K reverse strategies
        z_expanded = z_input[:, None, :] 
        ops_expanded = backward_ops[None, :, :]
        z_pred = z_expanded + ops_expanded
        
        # Check if ANY strategy lands on "N-1"
        preds = knn.predict(z_pred.reshape(-1, Z.shape[1])).reshape(len(z_input), -1)
        hits = np.any(preds == expected_digit, axis=1)
        acc = np.mean(hits)
        
        print(f"Solve 'x + 1 = {target_digit}' (Exp: {expected_digit}) | Accuracy: {acc:.2%}")

if __name__ == "__main__":
    main()