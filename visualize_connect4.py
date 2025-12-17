#!/usr/bin/env python3
import argparse, os, numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/connect4_bbit")
    args = ap.parse_args()
    
    Z = np.load(os.path.join(args.data, "latents.npy"))
    Y = np.load(os.path.join(args.data, "outcomes.npy"))
    
    # Subsample for speed
    idx = np.random.choice(len(Z), 2000, replace=False)
    Z_sub = Z[idx]
    Y_sub = Y[idx]
    
    pca = PCA(n_components=2)
    Z_2d = pca.fit_transform(Z_sub)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(Z_2d[Y_sub==-1, 0], Z_2d[Y_sub==-1, 1], c='red', alpha=0.5, label="Losing States")
    plt.scatter(Z_2d[Y_sub==1, 0], Z_2d[Y_sub==1, 1], c='blue', alpha=0.5, label="Winning States")
    
    # Plot Vector
    c_win = np.mean(Z_2d[Y_sub==1], axis=0)
    c_loss = np.mean(Z_2d[Y_sub==-1], axis=0)
    
    plt.arrow(c_loss[0], c_loss[1], c_win[0]-c_loss[0], c_win[1]-c_loss[1], 
              color='black', width=0.05, label="The Path to Victory")
              
    plt.title("Connect 4: The Geometry of Winning")
    plt.legend()
    plt.savefig("connect4_map.png")
    print("[ok] Map saved.")

if __name__ == "__main__":
    main()