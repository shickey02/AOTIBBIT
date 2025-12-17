#!/usr/bin/env python3
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

class AmplifiedEngine:
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

    def solve(self, a, b, c, gain=1.5):
        # Amplified Algebra: A - (gain * B) + (gain * C)
        # Or better: A + gain * (C - B)
        # Vector = (C - B) is the "Gender Flip" vector
        
        va = self.get_vec(a)
        vb = self.get_vec(b)
        vc = self.get_vec(c)
        
        # The logic is: "Take A, apply the transformation B->C"
        transform_vec = vc - vb
        v_res = va + (transform_vec * gain)
        
        dists, indices = self.nbrs.kneighbors([v_res])
        res_text = self.sentences[indices[0][0]]
        
        print(f"// QUERY: ({a}) + {gain} * [({c}) - ({b})]")
        print(f"<< RET: '{res_text}' (Dist: {dists[0][0]:.4f})")
        print("-" * 40)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/semantic_structured")
    args = ap.parse_args()
    eng = AmplifiedEngine(args.data)
    
    print("\n--- AMPLIFIED LOGIC TEST ---")
    # King - Man + Woman
    eng.solve("the king rules the kingdom", "the man rules the kingdom", "the woman rules the kingdom", gain=1.2)
    eng.solve("the king rules the kingdom", "the man rules the kingdom", "the woman rules the kingdom", gain=1.5)
    eng.solve("the king rules the kingdom", "the man rules the kingdom", "the woman rules the kingdom", gain=2.0)

if __name__ == "__main__":
    main()