#!/usr/bin/env python3
import os, argparse, json, torch
import numpy as np
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors

# Re-define Spherical Model
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
        z = self.fc_z(h_n.squeeze(0))
        return torch.nn.functional.normalize(z, p=2, dim=1)

class HighGainEngine:
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
        self.nbrs = NearestNeighbors(n_neighbors=1, metric='cosine').fit(self.Z)

    def get_vec(self, text):
        seq = [self.vocab.get(w, 0) for w in text.split()]
        t = torch.tensor([seq], dtype=torch.long).to(self.device)
        with torch.no_grad():
            return self.model.encode(t).cpu().numpy()[0]

    def solve(self, base_text, minus_text, plus_text, gain=2.0):
        # 1. Get Anchors
        v_base = self.get_vec(base_text)   # King
        v_minus = self.get_vec(minus_text) # Man
        v_plus = self.get_vec(plus_text)   # Woman
        
        # 2. Extract Raw Delta (The "Concept")
        delta = v_base - v_minus
        
        # 3. Orthogonalize (Clean the Concept)
        # Remove any part of 'delta' that is parallel to 'v_plus' (Woman)
        # We want Pure Royalty, not "Male Royalty"
        proj = np.dot(delta, v_plus) * v_plus
        delta_clean = delta - proj
        
        # 4. Amplify (High Gain)
        delta_boosted = delta_clean * gain
        
        # 5. Apply
        v_res_raw = v_plus + delta_boosted
        
        # 6. Re-Normalize (Snap to Sphere)
        v_res = v_res_raw / np.linalg.norm(v_res_raw)
        
        # Retrieval
        dists, indices = self.nbrs.kneighbors([v_res])
        res_text = self.sentences[indices[0][0]]
        
        # Calculate Angular Shift (How much did we rotate?)
        cos_sim = np.dot(v_plus, v_res)
        rotation_deg = np.degrees(np.arccos(np.clip(cos_sim, -1, 1)))
        
        self.log_english_plus(base_text, minus_text, plus_text, res_text, rotation_deg, gain)

    def log_english_plus(self, a, b, c, res, rot, gain):
        print("\n" + "="*60)
        print(f"// HIGH_GAIN_OP: ({a}) - ({b}) + ({c})")
        print(f">> PARAM: [Gain: {gain}x] [Rotation: {rot:.2f}°]")
        
        status = "STABLE"
        if rot > 45: status = "HYPER-JUMP (High Energy)"
        
        print(f":: META:  [Vector Logic Applied] [{status}]")
        print(f"<< RET:   '{res}'")
        print("="*60)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/semantic_spherical")
    args = ap.parse_args()
    eng = HighGainEngine(args.data)
    
    print("--- TUNING THE MIND ---")
    
    # Try increasing gain until we hit the target
    # 1. Standard Gain (likely Princess or Woman)
    eng.solve("the king rules the kingdom", "the man rules the kingdom", "the woman rules the kingdom", gain=1.0)
    
    # 2. High Gain (Should hit Queen)
    eng.solve("the king rules the kingdom", "the man rules the kingdom", "the woman rules the kingdom", gain=3.0)
    
    # 3. Extreme Gain (Might overshoot to something else?)
    eng.solve("the king rules the kingdom", "the man rules the kingdom", "the woman rules the kingdom", gain=5.0)

if __name__ == "__main__":
    main()