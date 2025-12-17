#!/usr/bin/env python3
# ENGLISH+ META-DECODER
# Decodes the internal state of the structured manifold into "System Speak".

import os, argparse, json, torch
import numpy as np
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors

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

class EnglishPlusEngine:
    def __init__(self, data_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"// SYSTEM: CONNECTING TO CRYSTAL LATTICE AT {data_dir}")
        
        with open(os.path.join(data_dir, "manifest.json"), "r") as f:
            self.manifest = json.load(f)
        self.vocab = self.manifest["vocab"]
        self.sentences = self.manifest["sentences"]
        self.Z = np.load(os.path.join(data_dir, "latents.npy"))
        
        self.model = StructuredAE(len(self.vocab), latent_dim=32).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(data_dir, "model.pth")))
        self.model.eval()
        
        self.nbrs = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(self.Z)

    def get_vec(self, text):
        seq = [self.vocab.get(w, 0) for w in text.split()]
        t = torch.tensor([seq], dtype=torch.long).to(self.device)
        with torch.no_grad():
            return self.model.encode(t).cpu().numpy()[0]

    def solve(self, a, b, c):
        # Algebra
        va = self.get_vec(a)
        vb = self.get_vec(b)
        vc = self.get_vec(c)
        
        v_res = va - vb + vc
        
        # Retrieval
        dists, indices = self.nbrs.kneighbors([v_res])
        res_text = self.sentences[indices[0][0]]
        snap = dists[0][0]
        
        # META-ANALYSIS (Reading the Axes)
        # We know Dim 0 = Gender, Dim 1 = Power
        gender_val = v_res[0]
        power_val = v_res[1]
        
        gender_tag = "NEUTRAL"
        if gender_val > 0.5: gender_tag = "MALE (+)"
        if gender_val < -0.5: gender_tag = "FEMALE (-)"
        
        power_tag = "COMMONER"
        if power_val > 0.8: power_tag = "ROYAL (+++"
        if power_val < -0.8: power_tag = "LOW_STATUS (--)"
        
        self.log_english_plus(a, b, c, res_text, gender_tag, power_tag, snap)

    def log_english_plus(self, a, b, c, res, g_tag, p_tag, snap):
        print("\n" + "="*60)
        print(f"// IN: ({a}) - ({b}) + ({c})")
        print(f">> OPR: [Algebraic Projection]")
        print(f":: AXIS_0 (Gender): {g_tag}")
        print(f":: AXIS_1 (Power):  {p_tag}")
        
        stability = "STABLE"
        if snap > 0.5: stability = "UNSTABLE (Drift Warning)"
        
        print(f"!! SYS: Lattice Snap = {snap:.4f} [{stability}]")
        print(f"<< OUT: '{res}'")
        print("="*60)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/semantic_structured")
    args = ap.parse_args()
    
    eng = EnglishPlusEngine(args.data)
    
    # 1. King - Man + Woman
    eng.solve(
        "the king rules the kingdom",
        "the man rules the kingdom",
        "the woman rules the kingdom"
    )
    
    # 2. Prince - Man + Woman
    eng.solve(
        "the prince commands the people",
        "the man commands the people",
        "the woman commands the people"
    )

    # 3. Father - Man + Woman
    eng.solve(
        "the father loves the throne",
        "the man loves the throne",
        "the woman loves the throne"
    )

if __name__ == "__main__":
    main()