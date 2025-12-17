#!/usr/bin/env python3
import argparse, os, json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/semantic_v2")
    args = ap.parse_args()

    # 1. Load Data
    print("Loading Semantic Field...")
    Z = np.load(os.path.join(args.data, "latents.npy"))
    with open(os.path.join(args.data, "manifest.json"), "r") as f:
        manifest = json.load(f)
    
    # Extract sentences for labeling
    # Handle list-of-lists format from v2 setup
    sentences = [" ".join(s) for s in manifest["sentences"]]
    meta = manifest["metadata"]
    
    # 2. Compute 2D Projection
    pca = PCA(n_components=2)
    Z_2d = pca.fit_transform(Z)
    
    # 3. Mine the "Future" Vector (Global Average)
    print("Mining Logic Vector...")
    future_vecs = []
    # Identify pairs
    grouped = {}
    for i, m in enumerate(meta):
        root = m["root"]
        typ = m["type"]
        if root not in grouped: grouped[root] = {}
        grouped[root][typ] = i
        
    for root, variants in grouped.items():
        if "declarative" in variants and "future" in variants:
            v_start = Z[variants["declarative"]]
            v_end = Z[variants["future"]]
            future_vecs.append(v_end - v_start)
            
    avg_future = np.mean(future_vecs, axis=0)
    # Project vector into 2D space (relative to center)
    center = np.mean(Z, axis=0)
    vec_start_2d = pca.transform([center])[0]
    vec_end_2d = pca.transform([center + avg_future])[0]
    vec_2d = vec_end_2d - vec_start_2d

    # 4. PLOT
    plt.figure(figsize=(14, 10))
    
    # Color by Tense
    types = [m["type"] for m in meta]
    colors = {'declarative': 'blue', 'future': 'green', 'negation': 'red'}
    c_list = [colors.get(t, 'gray') for t in types]
    
    plt.scatter(Z_2d[:,0], Z_2d[:,1], c=c_list, alpha=0.1, s=10)
    
    # Highlight Specific Trajectory: "I create data" -> Future
    target_root = "i create data"
    if target_root in grouped and "declarative" in grouped[target_root]:
        idx_start = grouped[target_root]["declarative"]
        z_start = Z[idx_start]
        p_start = Z_2d[idx_start]
        
        # Predicted Point
        z_pred = z_start + avg_future
        p_pred = pca.transform([z_pred])[0]
        
        # Plot Arrow
        plt.arrow(p_start[0], p_start[1], vec_2d[0], vec_2d[1], 
                  head_width=0.2, color='black', linewidth=2, label="Applied Logic")
        
        plt.text(p_start[0], p_start[1], "Start: 'I create'", fontsize=12, fontweight='bold')
        plt.text(p_pred[0], p_pred[1], "Projected Thought", fontsize=12, color='purple', fontweight='bold')
        
        # Where did it actually land? (Nearest Neighbor)
        # Find nearest point in Z to z_pred
        dists = np.linalg.norm(Z - z_pred, axis=1)
        idx_nearest = np.argmin(dists)
        p_nearest = Z_2d[idx_nearest]
        nearest_sent = sentences[idx_nearest]
        
        plt.scatter(p_nearest[0], p_nearest[1], c='orange', s=150, marker='*', label="Retrieval")
        plt.text(p_nearest[0], p_nearest[1], f"Snap: '{nearest_sent}'", fontsize=12, color='orange', fontweight='bold')

    plt.title("Semantic Drift: Why 'I' became 'System'")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("semantic_drift.png")
    print("[ok] Saved semantic_drift.png")

if __name__ == "__main__":
    main()