#!/usr/bin/env python3
# BBIT IQ TEST: FLUID INTELLIGENCE EVALUATION
# 
# Tests the system's ability to solve geometric logic puzzles:
# 1. Extrapolation (Sequence Completion)
# 2. Interpolation (Finding the Missing Link)
# 3. Analogy (Zero-Shot Rule Transfer)

import argparse, os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def get_mean_vector(Z, Y, start_class, end_class, limit=500):
    """Mines the average vector between two classes."""
    src = Z[Y == start_class]
    tgt = Z[Y == end_class]
    
    # Safety check
    if len(src) == 0 or len(tgt) == 0:
        return np.zeros(Z.shape[1])
        
    # Random sampling to save time
    rng = np.random.default_rng(42)
    i_s = rng.choice(len(src), size=min(len(src), limit))
    j_s = rng.choice(len(tgt), size=min(len(tgt), limit))
    
    vecs = tgt[j_s] - src[i_s]
    return np.mean(vecs, axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", default="./data/mnist_bbit/latents_mnist.npy")
    ap.add_argument("--labels", default="./data/mnist_bbit/labels_mnist.npy")
    args = ap.parse_args()
    
    # 1. Load Brain
    print("// SYSTEM: LOADING CORTEX...")
    Z = np.load(args.latents).astype(np.float32)
    Y = np.load(args.labels).astype(np.int32)
    
    # Split: We learn rules on Train, solve puzzles on Test
    Z_train, Z_test, Y_train, Y_test = train_test_split(Z, Y, test_size=0.2, random_state=42)
    
    # Oracle (To verify answers)
    knn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1).fit(Z_test, Y_test)

    print("\n=== PUZZLE 1: THE 'SKIP' SEQUENCE (0, 2, 4 -> ?) ===")
    print("Task: Infer the '+2' rule from 0->2 and apply it to 4.")
    
    # Learn the "+2" vector from 0->2
    vec_skip = get_mean_vector(Z_train, Y_train, 0, 2)
    
    # Apply to 4
    z_starts = Z_test[Y_test == 4]
    z_preds = z_starts + vec_skip
    
    # Check if we landed on 6
    y_preds = knn.predict(z_preds)
    acc = np.mean(y_preds == 6)
    
    print(f">> LOGIC:  Rule_Inferred [Mag: {np.linalg.norm(vec_skip):.2f}]")
    print(f"<< SOLVE:  Accuracy on '4 -> 6': {acc:.2%}")

    
    print("\n=== PUZZLE 2: THE MISSING LINK (1, ?, 3) ===")
    print("Task: Find the geometric midpoint between 1 and 3. Is it 2?")
    
    # Get 1s and 3s
    idx_1 = np.where(Y_test == 1)[0]
    idx_3 = np.where(Y_test == 3)[0]
    
    # Pair them up arbitrarily to form "bookends"
    limit = min(len(idx_1), len(idx_3), 1000)
    z_1 = Z_test[idx_1[:limit]]
    z_3 = Z_test[idx_3[:limit]]
    
    # Interpolate: Midpoint = (A + B) / 2
    z_mid = (z_1 + z_3) / 2.0
    
    # Check identity of the midpoint
    y_mid = knn.predict(z_mid)
    acc_mid = np.mean(y_mid == 2)
    
    print(f">> LOGIC:  Calculating Midpoints...")
    print(f"<< SOLVE:  Accuracy (Is it 2?): {acc_mid:.2%}")
    # Drift check: What else did it think it was?
    if acc_mid < 0.8:
        common = np.bincount(y_mid).argmax()
        print(f"!! DRIFT:  System confused it with '{common}'")


    print("\n=== PUZZLE 3: ANALOGY (1:2 :: 8:?) ===")
    print("Task: '1 is to 2' as '8 is to X'. Solve for X.")
    
    # Learn relation 1->2
    vec_relation = get_mean_vector(Z_train, Y_train, 1, 2)
    
    # Apply to 8
    z_8 = Z_test[Y_test == 8]
    z_analogy = z_8 + vec_relation
    
    # Expect 9
    y_analogy = knn.predict(z_analogy)
    acc_analogy = np.mean(y_analogy == 9)
    
    print(f">> LOGIC:  Transporting Relation Vector...")
    print(f"<< SOLVE:  Accuracy on '8 -> 9': {acc_analogy:.2%}")

    
    print("\n=== PUZZLE 4: THE 'IMPOSSIBLE' JUMP (0 -> 9) ===")
    print("Task: Can we jump across the entire manifold in one shot?")
    
    vec_long = get_mean_vector(Z_train, Y_train, 0, 9)
    z_0 = Z_test[Y_test == 0]
    z_long = z_0 + vec_long
    y_long = knn.predict(z_long)
    acc_long = np.mean(y_long == 9)
    
    print(f"<< SOLVE:  Accuracy: {acc_long:.2%}")

if __name__ == "__main__":
    main()