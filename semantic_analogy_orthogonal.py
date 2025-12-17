#!/usr/bin/env python3
# SEMANTIC ANALOGY (ORTHOGONAL)
# Uses Gram-Schmidt projection to remove concept leakage (e.g., removing 'Male' from 'Royal').

import os, argparse, json, torch
import numpy as np
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors

# Re-define Model
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

class OrthogonalEngine:
    def __init__(self, data_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    def project_and_remove(self, v_target, v_axis):
        """Removes the component of v_target that lies along v_axis."""
        # Proj_u(v) = (v . u) / (u . u) * u
        dot_product = np.dot(v_target, v_axis)
        norm_sq = np.dot(v_axis, v_axis)
        projection = (dot_product / norm_sq) * v_axis
        return v_target - projection

    def solve(self, base_text, minus_text, plus_text):
        # E.g. Base=King, Minus=Man, Plus=Woman
        
        v_base = self.get_vec(base_text)   # King
        v_minus = self.get_vec(minus_text) # Man
        v_plus = self.get_vec(plus_text)   # Woman
        
        # 1. Define the Gender Axis (Woman - Man)
        v_gender_axis = v_plus - v_minus
        
        # 2. Define the Raw Attribute (King - Man)
        # This is supposed to be "Royalty", but it's dirty.
        v_attrib_raw = v_base - v_minus
        
        # 3. CLEAN THE ATTRIBUTE (Orthogonalization)
        # Remove any "Gender" from "Royalty"
        v_attrib_pure = self.project_and_remove(v_attrib_raw, v_gender_axis)
        
        # 4. Construct Result
        # Woman + Pure_Royalty
        v_result = v_plus + v_attrib_pure
        
        # Retrieve
        dists, indices = self.nbrs.kneighbors([v_result])
        res_text = self.sentences[indices[0][0]]
        
        # Measure contamination removed
        contamination = np.linalg.norm(v_attrib_raw - v_attrib_pure)
        
        print(f"// QUERY: ({base_text}) - ({minus_text}) + ({plus_text})")
        print(f">> MATH:  [Gram-Schmidt Cleaned]")
        print(f":: INFO:  Removed {contamination:.4f} magnitude of Gender Leakage.")
        print(f"<< RET:   '{res_text}'")
        print("-" * 50)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/semantic_structured")
    args = ap.parse_args()
    eng = OrthogonalEngine(args.data)
    
    print("\n--- ORTHOGONAL LOGIC TESTS ---")
    
    # 1. King - Man + Woman
    eng.solve("the king rules the kingdom", "the man rules the kingdom", "the woman rules the kingdom")
    
    # 2. Prince - Man + Woman
    eng.solve("the prince sees the gold", "the man sees the gold", "the woman sees the gold")
    
    # 3. Father - Man + Woman
    eng.solve("the father loves the people", "the man loves the people", "the woman loves the people")

if __name__ == "__main__":
    main()