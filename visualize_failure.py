#!/usr/bin/env python3
import argparse, os, json, torch
import numpy as np
import matplotlib.pyplot as plt

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

    # Get the Key Players
    v_king = get_vec("king")
    v_man = get_vec("man")
    v_woman = get_vec("woman")
    v_queen = get_vec("queen")
    v_princess = get_vec("princess")
    
    # Calculate the Attempt
    # "The Royal Vector" = King - Man
    v_royal = v_king - v_man
    
    # "The Result" = Woman + Royal
    v_result = v_woman + v_royal
    
    # Plot Gender (Dim 0) vs Power (Dim 1)
    plt.figure(figsize=(10, 8))
    
    # Plot Anchors
    anchors = {
        "KING": v_king, "MAN": v_man, 
        "WOMAN": v_woman, "QUEEN": v_queen, "PRINCESS": v_princess
    }
    
    for name, v in anchors.items():
        color = 'gold' if "KING" in name or "QUEEN" in name else 'gray'
        if name == "PRINCESS": color = 'pink'
        plt.scatter(v[0], v[1], c=color, s=200, edgecolors='black', zorder=10)
        plt.text(v[0]+0.05, v[1]+0.05, name, fontsize=12, fontweight='bold')

    # Draw the "Royal Logic" Arrow (Man -> King)
    plt.arrow(v_man[0], v_man[1], v_royal[0], v_royal[1], 
              color='blue', width=0.02, alpha=0.3, label="Concept: ROYALTY")
    
    # Draw the "Applied Logic" Arrow (Woman -> Result)
    plt.arrow(v_woman[0], v_woman[1], v_royal[0], v_royal[1], 
              color='blue', width=0.02, label="Applied Logic")
    
    # Draw the Result
    plt.scatter(v_result[0], v_result[1], c='red', marker='X', s=300, label="CALCULATED POINT", zorder=20)
    
    # Draw "Gravity" (Why it fell short)
    plt.arrow(v_result[0], v_result[1], v_queen[0]-v_result[0], v_queen[1]-v_result[1], 
              color='red', linestyle='--', width=0.005, label="The Gap (Error)")

    plt.title("The Princess Trap: Why Linear Math Failed")
    plt.xlabel("Gender Axis")
    plt.ylabel("Power Axis")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("princess_trap.png")
    print("[ok] Saved princess_trap.png")

if __name__ == "__main__":
    main()