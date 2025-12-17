#!/usr/bin/env python3
import argparse, os, json, torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Model Definition
import torch.nn as nn
class StructuredAE(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, latent_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder_rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc_z = nn.Linear(hidden_dim, latent_dim)
        self.decoder_rnn = nn.LSTM(embed_dim + latent_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
    def encode(self, x):
        embed = self.embedding(x)
        _, (h_n, _) = self.encoder_rnn(embed)
        return self.fc_z(h_n.squeeze(0))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/semantic_structured")
    args = ap.parse_args()

    # Load Brain
    device = torch.device("cpu")
    with open(os.path.join(args.data, "manifest.json"), "r") as f:
        manifest = json.load(f)
    vocab = manifest["vocab"]
    
    model = StructuredAE(len(vocab), latent_dim=32)
    model.load_state_dict(torch.load(os.path.join(args.data, "model.pth"), map_location=device))
    model.eval()

    def get_vec(word):
        text = f"the {word} rules the kingdom"
        seq = [vocab.get(w, 0) for w in text.split()]
        with torch.no_grad():
            return model.encode(torch.tensor([seq])).numpy()[0]

    # The Puzzle Group
    words = ["king", "prince", "man", "queen"]
    vecs = np.array([get_vec(w) for w in words])
    
    # Extract Axes (0=Gender, 1=Power)
    X = vecs[:, 0]
    Y = vecs[:, 1]
    
    plt.figure(figsize=(10, 8))
    
    # Plot Points
    plt.scatter(X, Y, c=['blue', 'cyan', 'gray', 'red'], s=300, edgecolors='black')
    
    for i, w in enumerate(words):
        plt.text(X[i]+0.05, Y[i]+0.05, w.upper(), fontsize=12, fontweight='bold')

    # Draw the "Male Cluster"
    male_x = X[:3] # King, Prince, Man
    male_y = Y[:3]
    
    # Ellipse for cluster
    ellipse = patches.Ellipse((np.mean(male_x), np.mean(male_y)), width=0.4, height=1.5, 
                              angle=0, fill=False, edgecolor='blue', linestyle='--', linewidth=2, label="Male Cluster")
    plt.gca().add_patch(ellipse)

    # Highlight Outlier
    plt.scatter(X[3], Y[3], s=400, facecolors='none', edgecolors='red', linewidth=3, label="Outlier")
    
    plt.title("Visualizing the Logic: 'Odd One Out'")
    plt.xlabel("Gender Axis (-1 Female ... +1 Male)")
    plt.ylabel("Power Axis (-1 Low ... +1 Royal)")
    plt.grid(True, alpha=0.3)
    plt.axvline(0, color='black', alpha=0.2)
    plt.axhline(0, color='black', alpha=0.2)
    plt.legend()
    
    plt.savefig("puzzle_viz.png")
    print("[ok] Saved puzzle_viz.png")

if __name__ == "__main__":
    main()