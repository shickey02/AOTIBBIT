#!/usr/bin/env python3
import argparse, os, json, torch
import numpy as np
import matplotlib.pyplot as plt

# Model Definition
import torch.nn as nn
class SphericalAE(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, latent_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder_rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc_z = nn.Linear(hidden_dim, latent_dim)
        
        # FIX: Added Decoder layers to match the saved model structure
        self.decoder_rnn = nn.LSTM(embed_dim + latent_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x):
        embed = self.embedding(x)
        _, (h_n, _) = self.encoder_rnn(embed)
        z = self.fc_z(h_n.squeeze(0))
        return torch.nn.functional.normalize(z, p=2, dim=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/semantic_spherical")
    args = ap.parse_args()

    # Load Brain
    device = torch.device("cpu")
    with open(os.path.join(args.data, "manifest.json"), "r") as f:
        manifest = json.load(f)
    vocab = manifest["vocab"]
    
    # Initialize Model
    model = SphericalAE(len(vocab), latent_dim=32)
    # Load weights
    model.load_state_dict(torch.load(os.path.join(args.data, "model.pth"), map_location=device))
    model.eval()

    def get_vec(word):
        text = f"the {word} rules the kingdom"
        seq = [vocab.get(w, 0) for w in text.split()]
        with torch.no_grad():
            return model.encode(torch.tensor([seq])).numpy()[0]

    # Get Vectors
    v_man = get_vec("man")
    v_king = get_vec("king")
    v_woman = get_vec("woman")
    v_queen = get_vec("queen")
    
    # Calculate the tiny "Royal" delta
    delta_royal = v_king - v_man
    
    # Calculate Result
    v_result = v_woman + delta_royal
    
    # Plot (2D PCA for simplicity)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    # Fit on key points to preserve their geometry
    all_vecs = np.array([v_man, v_king, v_woman, v_queen, v_result])
    pca.fit(all_vecs)
    
    p_man, p_king, p_woman, p_queen, p_res = pca.transform(all_vecs)
    
    plt.figure(figsize=(10, 8))
    
    # Plot Points
    plt.scatter(p_man[0], p_man[1], c='blue', s=200, label="Man")
    plt.scatter(p_king[0], p_king[1], c='gold', s=200, label="King")
    plt.scatter(p_woman[0], p_woman[1], c='pink', s=200, label="Woman")
    plt.scatter(p_queen[0], p_queen[1], c='purple', s=200, label="Queen")
    plt.scatter(p_res[0], p_res[1], c='red', marker='X', s=300, label="Result (Failed)")

    # Draw the Tiny Royal Vector
    plt.arrow(p_man[0], p_man[1], p_king[0]-p_man[0], p_king[1]-p_man[1], 
              color='green', width=0.005, label="Royal Delta (Weak)")
              
    # Draw the Applied Vector
    plt.arrow(p_woman[0], p_woman[1], p_res[0]-p_woman[0], p_res[1]-p_woman[1], 
              color='green', width=0.005, linestyle='--')

    plt.title("Signal Cancellation: Why 'Royal' was ignored")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("cancellation_plot.png")
    print("[ok] Saved cancellation_plot.png")

if __name__ == "__main__":
    main()