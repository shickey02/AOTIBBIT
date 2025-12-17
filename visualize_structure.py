#!/usr/bin/env python3
import argparse, os, json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/semantic_structured")
    args = ap.parse_args()

    print(f"Loading Crystal Lattice from {args.data}...")
    Z = np.load(os.path.join(args.data, "latents.npy"))
    with open(os.path.join(args.data, "manifest.json"), "r") as f:
        manifest = json.load(f)
    sentences = manifest["sentences"]
    
    # We want to plot Dimension 0 (Gender) vs Dimension 1 (Power)
    # This is the "Forced Physics" we created.
    gender = Z[:, 0]
    power = Z[:, 1]
    
    plt.figure(figsize=(12, 10))
    
    # Plot all points
    plt.scatter(gender, power, alpha=0.1, c='gray', s=10, label="The People")
    
    # Highlight Key Archetypes (Average position of specific words)
    archetypes = ["king", "queen", "prince", "princess", "man", "woman", "father", "mother"]
    colors = ['gold', 'purple', 'orange', 'pink', 'blue', 'red', 'cyan', 'magenta']
    
    found_archetypes = {}
    
    print("Locating Archetypes...")
    for word, color in zip(archetypes, colors):
        # Find indices where the sentence contains the word
        # (Naive check: "the king..." contains "king")
        indices = [i for i, s in enumerate(sentences) if f" {word} " in s or f" {word}" in s]
        
        if indices:
            z_group = Z[indices]
            mean_z = np.mean(z_group, axis=0)
            
            # Plot the center of the concept
            plt.scatter(mean_z[0], mean_z[1], c=color, s=200, edgecolors='black', label=word.upper())
            plt.text(mean_z[0]+0.05, mean_z[1]+0.05, word.upper(), fontsize=12, fontweight='bold', color=color)
            
            found_archetypes[word] = mean_z
            
    # VISUALIZE THE ALGEBRA
    # King - Man + Woman
    if "king" in found_archetypes and "man" in found_archetypes and "woman" in found_archetypes:
        v_king = found_archetypes["king"]
        v_man = found_archetypes["man"]
        v_woman = found_archetypes["woman"]
        
        # Calculate ideal result
        v_result = v_king - v_man + v_woman
        
        # Draw the trajectory
        # 1. Start at King
        # 2. Vector "Minus Man" (This removes "Maleness" and "Low Status"?)
        # 3. Vector "Plus Woman" (Adds "Femaleness")
        
        # Simply plot the Result Point
        plt.scatter(v_result[0], v_result[1], c='black', marker='x', s=300, linewidth=3, label="CALCULATED RESULT")
        plt.text(v_result[0], v_result[1]-0.1, "Algebraic Target", fontsize=12, fontweight='bold')
        
        # Draw arrow from King to Result
        plt.arrow(v_king[0], v_king[1], v_result[0]-v_king[0], v_result[1]-v_king[1], 
                  color='black', linestyle='--', alpha=0.5, head_width=0.05)

    plt.title("The Structured Manifold: Gender (X) vs Power (Y)")
    plt.xlabel("Dim 0: Gender (-1 Female, +1 Male)")
    plt.ylabel("Dim 1: Power (-1 Low, +1 Royal)")
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    
    out_path = "structure_plot.png"
    plt.savefig(out_path)
    print(f"[ok] Saved {out_path}")

if __name__ == "__main__":
    main()