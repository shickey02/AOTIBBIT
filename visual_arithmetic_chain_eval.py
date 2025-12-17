#!/usr/bin/env python3
# VISUAL ARITHMETIC (CHAIN) EVALUATION v4.0
# 
# Hypothesis: Complex reasoning is just a sequence of local geometric moves.
# Task: Perform multi-step arithmetic (e.g., "1 + 1 + 1") without seeing the intermediate images.

import argparse, os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

class GeometricReasoningEngine:
    def __init__(self, n_strategies=5, seed=42):
        self.k = n_strategies
        self.rng = np.random.default_rng(seed)
        self.strategies = {} # Stores { "0->1": [vec1, vec2...], "1->2": ... }

    def learn(self, Z, Y):
        """Mines geometric strategies for every digit transition."""
        print("Learning Reasoning Strategies...")
        
        for start_digit in tqdm(range(9), desc="Mining Logic"):
            end_digit = start_digit + 1
            key = f"{start_digit}->{end_digit}"
            
            src_idxs = np.where(Y == start_digit)[0]
            tgt_idxs = np.where(Y == end_digit)[0]
            
            # Sample pairs to find the "Hidden Rules"
            n_pairs = 20000
            i_s = self.rng.choice(src_idxs, size=n_pairs)
            j_s = self.rng.choice(tgt_idxs, size=n_pairs)
            
            vectors = Z[j_s] - Z[i_s]
            
            # Cluster into K distinct strategies
            kmeans = MiniBatchKMeans(n_clusters=self.k, batch_size=1024, n_init="auto", random_state=42)
            kmeans.fit(vectors)
            self.strategies[key] = kmeans.cluster_centers_

    def reason_chain(self, z_start, start_digit, steps=2, beam_width=3):
        """
        Performs multi-step reasoning using Beam Search.
        Input: A single latent vector 'z' (image of a number).
        Output: The latent vector after 'steps' additions.
        """
        # Beam: List of tuples (current_z, current_digit)
        # We start with the input image
        beam = [(z_start, start_digit)]
        
        for _ in range(steps):
            next_beam_candidates = []
            
            for (z_curr, digit) in beam:
                # Stop if we hit 9 (can't go higher in this dataset)
                if digit >= 9:
                    continue
                    
                # 1. Get Strategies for this step (e.g., "2->3")
                key = f"{digit}->{digit+1}"
                if key not in self.strategies:
                    continue
                    
                moves = self.strategies[key] # (K, D)
                
                # 2. Apply ALL strategies (Expand Hypotheses)
                # z_curr is (D,), moves is (K, D) -> candidates is (K, D)
                candidates = z_curr + moves
                
                # 3. Add to pool
                for z_cand in candidates:
                    next_beam_candidates.append((z_cand, digit + 1))
            
            # 4. Pruning (The "Focus" Step)
            # In a real system, we'd use a critic/discriminator to score these.
            # Here, we lack a "truth" metric mid-air, so we keep ALL (or random subset).
            # To simulate "Geometry Holding Together", we keep the Beam Width.
            
            if not next_beam_candidates:
                break
                
            # Random selection for now (mimics exploration)
            # In v5, we would check "Is this a valid digit?" density.
            indices = self.rng.choice(len(next_beam_candidates), size=min(len(next_beam_candidates), beam_width), replace=False)
            beam = [next_beam_candidates[i] for i in indices]
            
        return beam

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", default="./data/mnist_bbit/latents_mnist.npy")
    ap.add_argument("--labels", default="./data/mnist_bbit/labels_mnist.npy")
    ap.add_argument("--k_strategies", type=int, default=5)
    ap.add_argument("--beam_width", type=int, default=5)
    ap.add_argument("--chain_len", type=int, default=2, help="Steps to add (e.g. 2 means +1+1)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 1. Load Data
    print(f"Loading {args.latents}...")
    Z = np.load(args.latents).astype(np.float32)
    Y = np.load(args.labels).astype(np.int32)
    Z_train, Z_test, Y_train, Y_test = train_test_split(Z, Y, test_size=0.2, random_state=args.seed)

    # 2. Train the Brain
    engine = GeometricReasoningEngine(n_strategies=args.k_strategies, seed=args.seed)
    engine.learn(Z_train, Y_train)

    # 3. Build Oracle
    print("Building Oracle...")
    knn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
    knn.fit(Z_test, Y_test)

    # 4. Run Chain Test
    print(f"\n--- EVALUATION: CHAIN OF THOUGHT (Length={args.chain_len}) ---")
    print(f"Task: Take image of 'N', add 1 {args.chain_len} times. Is result 'N+{args.chain_len}'?")
    print("-" * 60)
    
    total_hits = 0
    total_samples = 0
    
    # We can only test starts where start + chain_len <= 9
    valid_starts = range(10 - args.chain_len)
    
    for start_digit in valid_starts:
        target_digit = start_digit + args.chain_len
        
        # Get test examples
        idxs = np.where(Y_test == start_digit)[0]
        # Limit samples to speed up
        if len(idxs) > 200: idxs = idxs[:200]
        
        hits = 0
        
        for idx in idxs:
            z_input = Z_test[idx]
            
            # Run the Mental Simulation
            final_hypotheses = engine.reason_chain(z_input, start_digit, steps=args.chain_len, beam_width=args.beam_width)
            
            # Check results
            # Success = If ANY of the final hypotheses maps to the Target Digit
            example_success = False
            
            if len(final_hypotheses) > 0:
                # Extract Z vectors from list of tuples
                final_zs = np.stack([h[0] for h in final_hypotheses])
                preds = knn.predict(final_zs)
                
                if np.any(preds == target_digit):
                    example_success = True
            
            if example_success: hits += 1
            
        acc = hits / len(idxs)
        total_hits += hits
        total_samples += len(idxs)
        
        print(f"Input {start_digit} + {args.chain_len} -> Expected {target_digit} | Accuracy: {acc:.2%}")

    print("-" * 60)
    print(f"OVERALL CHAIN ACCURACY: {total_hits/total_samples:.2%}")

if __name__ == "__main__":
    main()