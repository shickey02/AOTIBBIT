#!/usr/bin/env python3
# LOGIC PUZZLE: ODD ONE OUT
# Uses geometric variance to detect the semantic outlier in a group.

import os, argparse, json, torch
import numpy as np
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

class PuzzleEngine:
    def __init__(self, data_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(os.path.join(data_dir, "manifest.json"), "r") as f:
            self.manifest = json.load(f)
        self.vocab = self.manifest["vocab"]
        
        self.model = StructuredAE(len(self.vocab), latent_dim=32).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(data_dir, "model.pth")))
        self.model.eval()

    def get_vec(self, word):
        # Contextualize to get a robust vector
        text = f"the {word} rules the kingdom"
        seq = [self.vocab.get(w, 0) for w in text.split()]
        t = torch.tensor([seq], dtype=torch.long).to(self.device)
        with torch.no_grad():
            return self.model.encode(t).cpu().numpy()[0]

    def solve_puzzle(self, words):
        print(f"\n// PUZZLE: Which is the odd one out? {words}")
        vecs = np.array([self.get_vec(w) for w in words])
        
        # 1. Analyze Dimensions 0 (Gender) and 1 (Power)
        gender = vecs[:, 0]
        power = vecs[:, 1]
        
        # 2. Check Variance to see which axis matters
        var_gender = np.var(gender)
        var_power = np.var(power)
        
        print(f">> ANALYSIS: Gender Var={var_gender:.2f}, Power Var={var_power:.2f}")
        
        outlier_idx = -1
        reason = ""
        
        # 3. Detect Minority
        if var_gender > var_power:
            # The puzzle is about Gender. Who is the gender minority?
            # E.g. [1, 1, 1, -1] -> Mean is 0.5. -1 is furthest.
            mean_g = np.mean(gender)
            dists = np.abs(gender - mean_g)
            outlier_idx = np.argmax(dists)
            reason = "Gender Mismatch"
        else:
            # The puzzle is about Power.
            mean_p = np.mean(power)
            dists = np.abs(power - mean_p)
            outlier_idx = np.argmax(dists)
            reason = "Status Mismatch"
            
        print(f"<< SOLVE: '{words[outlier_idx]}' is the odd one out. ({reason})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/semantic_structured")
    args = ap.parse_args()
    eng = PuzzleEngine(args.data)
    
    # Puzzle 1: Gender
    eng.solve_puzzle(["king", "prince", "man", "queen"])
    
    # Puzzle 2: Power
    eng.solve_puzzle(["king", "queen", "prince", "actor"])
    
    # Puzzle 3: Mixed (Harder)
    eng.solve_puzzle(["father", "brother", "uncle", "mother"])

if __name__ == "__main__":
    main()