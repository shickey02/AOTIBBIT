#!/usr/bin/env python3
import os, argparse, json, torch
import numpy as np
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors

class SphericalAE(nn.Module):
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
        z_raw = self.fc_z(h_n.squeeze(0))
        return torch.nn.functional.normalize(z_raw, p=2, dim=1)

class SphericalEngine:
    def __init__(self, data_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(os.path.join(data_dir, "manifest.json"), "r") as f:
            self.manifest = json.load(f)
        self.vocab = self.manifest["vocab"]
        self.sentences = self.manifest["sentences"]
        self.Z = np.load(os.path.join(data_dir, "latents.npy"))
        
        self.model = SphericalAE(len(self.vocab), latent_dim=32).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(data_dir, "model.pth")))
        self.model.eval()
        
        # USE COSINE METRIC FOR SPHERES
        self.nbrs = NearestNeighbors(n_neighbors=1, metric='cosine').fit(self.Z)

    def get_vec(self, text):
        seq = [self.vocab.get(w, 0) for w in text.split()]
        t = torch.tensor([seq], dtype=torch.long).to(self.device)
        with torch.no_grad():
            return self.model.encode(t).cpu().numpy()[0]

    def solve(self, a, b, c):
        va = self.get_vec(a)
        vb = self.get_vec(b)
        vc = self.get_vec(c)
        
        # Spherical Algebra: We add vectors, then RE-NORMALIZE
        # Start + (End - Minus)
        v_res_raw = va - vb + vc
        
        # Renormalize to stay on sphere
        v_res = v_res_raw / np.linalg.norm(v_res_raw)
        
        dists, indices = self.nbrs.kneighbors([v_res])
        res_text = self.sentences[indices[0][0]]
        
        # Convert Cosine Distance to Degrees
        # Dist = 1 - Cos(theta) => Cos(theta) = 1 - Dist
        cosine_sim = 1 - dists[0][0]
        # Clip to handle float errors
        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cosine_sim))
        
        self.log_english_plus(a, b, c, res_text, angle_deg)

    def log_english_plus(self, a, b, c, res, angle):
        print("\n" + "="*60)
        print(f"// SPHERE_OP: ({a}) - ({b}) + ({c})")
        
        # Status
        status = "LOCKED"
        if angle > 15: status = "DRIFTING"
        if angle > 45: status = "LOST_IN_VOID"
        
        print(f">> GEOM: [Angle Delta: {angle:.2f}°] [{status}]")
        print(f"<< RET:  '{res}'")
        print("="*60)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/semantic_spherical")
    args = ap.parse_args()
    eng = SphericalEngine(args.data)
    
    # 1. King - Man + Woman
    eng.solve(
        "the king rules the kingdom",
        "the man rules the kingdom",
        "the woman rules the kingdom"
    )

if __name__ == "__main__":
    main()