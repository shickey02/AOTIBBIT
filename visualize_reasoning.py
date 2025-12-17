#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", default="./data/mnist_bbit/latents_mnist.npy")
    ap.add_argument("--labels", default="./data/mnist_bbit/labels_mnist.npy")
    args = ap.parse_args()

    # 1. Load & Compress to 2D
    print("Compressing Brain to 2D...")
    Z = np.load(args.latents)
    Y = np.load(args.labels)
    
    # We use only a subset to keep the plot clean
    idx = np.random.choice(len(Z), 2000, replace=False)
    Z_sub, Y_sub = Z[idx], Y[idx]
    
    pca = PCA(n_components=2)
    Z_2d = pca.fit_transform(Z_sub) # Fit on subset
    
    # 2. Mine the "+1" Vector (Global Average for visualization)
    print("Mining Logic Vector...")
    plus_one_vectors = []
    for d in range(9):
        # simple global mining
        src = Z[Y == d]
        tgt = Z[Y == d+1]
        # minimal sampling
        if len(src)>0 and len(tgt)>0:
            v = np.mean(tgt, axis=0) - np.mean(src, axis=0)
            plus_one_vectors.append(v)
    global_vec = np.mean(plus_one_vectors, axis=0)
    
    vec_2d = pca.transform([global_vec + np.mean(Z, axis=0)]) - pca.transform([np.mean(Z, axis=0)])
    vec_2d = vec_2d[0]

    # 3. Simulate a Chain: 1 -> 2 -> 3
    print("Simulating Chain: 1 -> 2 -> 3...")
    # Find a good '1'
    start_idxs = np.where(Y == 1)[0]
    start_z = Z[start_idxs[0]] # Just pick first one
    
    # Path: Start -> Step1 -> Step2
    # NOTE: In reality, we used 'Local' vectors. Here we visualize Global to see the "Flow"
    # To visualize the REAL path, we'd need the Atlas code here. 
    # Let's just project the Global Path to show "Ideal Trajectory" vs "Manifold"
    
    path_z = [start_z]
    curr = start_z
    for _ in range(2):
        curr = curr + global_vec
        path_z.append(curr)
        
    path_2d = pca.transform(path_z)

    # 4. PLOT
    plt.figure(figsize=(12, 10))
    
    # Background: The Manifold
    scatter = plt.scatter(Z_2d[:,0], Z_2d[:,1], c=Y_sub, cmap='tab10', alpha=0.5, s=10)
    plt.colorbar(scatter, label="Digit Class")
    
    # The "River of Logic" (Vector Field)
    # Draw arrows showing where "+1" pushes you from various points
    grid_x, grid_y = np.meshgrid(np.linspace(Z_2d[:,0].min(), Z_2d[:,0].max(), 10),
                                 np.linspace(Z_2d[:,1].min(), Z_2d[:,1].max(), 10))
    plt.quiver(grid_x, grid_y, np.ones_like(grid_x)*vec_2d[0], np.ones_like(grid_y)*vec_2d[1], 
               color='black', alpha=0.2, label="Global '+1' Flow")

    # The Chain Path
    plt.plot(path_2d[:,0], path_2d[:,1], 'r-', linewidth=3, label="Reasoning Path (1->2->3)")
    plt.scatter(path_2d[:,0], path_2d[:,1], c='red', s=100, marker='x')
    
    # Annotate Path
    plt.text(path_2d[0,0], path_2d[0,1], "Start (1)", fontsize=12, fontweight='bold')
    plt.text(path_2d[1,0], path_2d[1,1], "Step 1", fontsize=12)
    plt.text(path_2d[2,0], path_2d[2,1], "End (3)", fontsize=12, fontweight='bold')

    plt.title("Visual Arithmetic: The Geometry of 'Adding One'")
    plt.xlabel("Latent Dim 1")
    plt.ylabel("Latent Dim 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_file = "visual_reasoning.png"
    plt.savefig(out_file)
    print(f"Saved visualization to {out_file}")

if __name__ == "__main__":
    main()