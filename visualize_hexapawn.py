#!/usr/bin/env python3
import argparse, os, numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/hexapawn_bbit")
    args = ap.parse_args()
    
    Z = np.load(os.path.join(args.data, "latents.npy"))
    Y = np.load(os.path.join(args.data, "outcomes.npy")) # 1, -1, 0
    
    pca = PCA(n_components=2)
    Z_2d = pca.fit_transform(Z)
    
    plt.figure(figsize=(10, 8))
    
    # Plot Ongoing
    plt.scatter(Z_2d[Y==0, 0], Z_2d[Y==0, 1], c='gray', alpha=0.3, label="Ongoing")
    # Plot White Wins
    plt.scatter(Z_2d[Y==1, 0], Z_2d[Y==1, 1], c='white', edgecolors='black', s=50, label="White Wins")
    # Plot Black Wins
    plt.scatter(Z_2d[Y==-1, 0], Z_2d[Y==-1, 1], c='black', alpha=0.8, s=50, label="Black Wins")
    
    # Calculate Centroids (Gravity Wells)
    c_white = np.mean(Z_2d[Y==1], axis=0)
    c_black = np.mean(Z_2d[Y==-1], axis=0)
    
    plt.scatter(c_white[0], c_white[1], c='gold', marker='*', s=300, edgecolors='black', label="White Gravity")
    plt.scatter(c_black[0], c_black[1], c='red', marker='*', s=300, edgecolors='black', label="Black Gravity")

    plt.title("The Hexapawn Manifold: Victory as Gravity")
    plt.legend()
    plt.savefig("hexapawn_manifold.png")
    print("[ok] Saved hexapawn_manifold.png")

if __name__ == "__main__":
    main()