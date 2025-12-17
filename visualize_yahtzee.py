#!/usr/bin/env python3
import argparse, os, numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/yahtzee_bbit")
    args = ap.parse_args()
    
    Z = np.load(os.path.join(args.data, "latents.npy"))
    Y = np.load(os.path.join(args.data, "labels.npy"))
    
    # Filter for interesting hands
    mask = Y >= 4
    Z_sub = Z[mask]
    Y_sub = Y[mask]
    
    pca = PCA(n_components=2)
    Z_2d = pca.fit_transform(Z_sub)
    
    plt.figure(figsize=(10, 8))
    labels = {4: "Full House", 5: "Sm Str", 6: "Lg Str", 7: "4-Kind", 8: "YAHTZEE"}
    
    for lid, name in labels.items():
        plt.scatter(Z_2d[Y_sub==lid, 0], Z_2d[Y_sub==lid, 1], label=name, alpha=0.6)
        
    plt.legend()
    plt.title("The Geometry of Dice")
    plt.savefig("yahtzee_map.png")
    print("[ok] Saved yahtzee_map.png")

if __name__ == "__main__":
    main()