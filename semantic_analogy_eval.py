#!/usr/bin/env python3
# SEMANTIC ANALOGY EVALUATION
# Task: Solve "A - B + C = ?" (e.g., King - Man + Woman)

import os, argparse, json, torch
import numpy as np
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors

# Re-define Model for loading
class RoyalAE(nn.Module):
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

class AnalogyEngine:
    def __init__(self, data_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"// SYSTEM: MOUNTING ROYAL MANIFOLD FROM {data_dir}")
        
        with open(os.path.join(data_dir, "manifest.json"), "r") as f:
            self.manifest = json.load(f)
        self.vocab = self.manifest["vocab"]
        self.sentences = self.manifest["sentences"]
        self.Z = np.load(os.path.join(data_dir, "latents.npy"))
        
        # Load Brain
        self.model = RoyalAE(len(self.vocab), latent_dim=32).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(data_dir, "model.pth")))
        self.model.eval()
        
        # Build Index
        self.nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='cosine').fit(self.Z)
        
        # Cache common vectors for the math
        # We need the vectors for words, but our model encodes SENTENCES.
        # Trick: "King" is context. We use "the king rules..." vs "the man rules..."
        # to isolate the "King-ness" vector.
        # Actually, simpler: Input "the king rules the kingdom", subtract "the man rules the kingdom".
        
    def get_thought_vector(self, text):
        seq = [self.vocab.get(w, 0) for w in text.split()]
        t = torch.tensor([seq], dtype=torch.long).to(self.device)
        with torch.no_grad():
            z = self.model.encode(t).cpu().numpy()[0]
        return z

    def solve_analogy(self, a_text, b_text, c_text):
        # A - B + C = ?
        z_a = self.get_thought_vector(a_text)
        z_b = self.get_thought_vector(b_text)
        z_c = self.get_thought_vector(c_text)
        
        # The Algebra
        z_result = z_a - z_b + z_c
        
        # Drift Check
        mag = np.linalg.norm(z_result)
        
        # Retrieval
        dists, indices = self.nbrs.kneighbors([z_result])
        result_text = self.sentences[indices[0][0]]
        snap_dist = dists[0][0]
        
        self.log_output(a_text, b_text, c_text, result_text, snap_dist, mag)

    def log_output(self, a, b, c, res, drift, mag):
        print("\n" + "="*60)
        print(f"// QUERY: ({a}) - ({b}) + ({c})")
        print(f">> CALC:  [Vector Mag: {mag:.2f}]")
        
        status = "STABLE"
        if drift > 0.1: status = "LOW_CONFIDENCE"
        if drift > 0.3: status = "HIGH_ENTROPY (Drift detected)"
        
        print(f":: META:  [Snap: {drift:.4f}] [{status}]")
        print(f"<< RET:   '{res}'")
        print("="*60)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/semantic_royal")
    args = ap.parse_args()
    
    eng = AnalogyEngine(args.data)
    
    # 1. The Classic: King - Man + Woman
    # We use full sentences to give the model context
    print("\n// TEST 1: THE ROYAL SUCCESSION")
    eng.solve_analogy(
        "the king rules the kingdom", 
        "the man rules the kingdom", 
        "the woman rules the kingdom"
    )

    # 2. Family: Father - Man + Woman
    print("\n// TEST 2: PARENTAL LINEAGE")
    eng.solve_analogy(
        "the father loves the people", 
        "the man loves the people", 
        "the woman loves the people"
    )
    
    # 3. Actor - Man + Woman
    print("\n// TEST 3: CASTING CALL")
    eng.solve_analogy(
        "the actor sees the gold", 
        "the man sees the gold", 
        "the woman sees the gold"
    )

    # 4. Inverted: Queen - Woman + Man -> King?
    print("\n// TEST 4: INVERTED HIERARCHY (Reverse Logic)")
    eng.solve_analogy(
        "the queen commands the throne", 
        "the woman commands the throne", 
        "the man commands the throne"
    )

if __name__ == "__main__":
    main()