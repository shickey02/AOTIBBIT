#!/usr/bin/env python3
# SEMANTIC RETRIEVAL EVALUATION
# Proves Geometric Logic by finding the nearest existing thought to the result vector.

import os, argparse, json, torch
import numpy as np
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors

# (We need the model class just to load the weights for encoding)
class ThoughtAE(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, latent_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder_rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc_z = nn.Linear(hidden_dim, latent_dim)
        self.decoder_rnn = nn.LSTM(embed_dim + latent_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.latent_dim = latent_dim

    def encode(self, x):
        embed = self.embedding(x)
        _, (h_n, _) = self.encoder_rnn(embed)
        z = self.fc_z(h_n.squeeze(0))
        return z

class SemanticEngine:
    def __init__(self, data_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"// SYSTEM: LOADING MANIFOLD FROM {data_dir}")
        
        with open(os.path.join(data_dir, "manifest.json"), "r") as f:
            self.manifest = json.load(f)
        self.vocab = self.manifest["vocab"]
        self.sentences = [" ".join(s) for s in self.manifest["sentences"]] if isinstance(self.manifest["sentences"][0], list) else self.manifest["sentences"]
        self.meta = self.manifest["metadata"]
        self.Z = np.load(os.path.join(data_dir, "latents.npy"))
        
        # Load Encoder
        self.model = ThoughtAE(len(self.vocab), latent_dim=32).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(data_dir, "model.pth")))
        self.model.eval()
        
        # Build Search Index (The "Memory")
        print("// SYSTEM: BUILDING MEMORY INDEX...")
        self.nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(self.Z)
        self.operators = {}

    def mine_operators(self):
        print(">> OPR: MINING LOGIC VECTORS...")
        grouped = {}
        # Simple grouping by root
        for i, m in enumerate(self.meta):
            root = m["root"]
            typ = m["type"]
            if root not in grouped: grouped[root] = {}
            grouped[root][typ] = i
            
        ops = {"future": [], "negation": []}
        for root, variants in grouped.items():
            if "declarative" in variants:
                z_base = self.Z[variants["declarative"]]
                if "future" in variants:
                    ops["future"].append(self.Z[variants["future"]] - z_base)
                if "negation" in variants:
                    ops["negation"].append(self.Z[variants["negation"]] - z_base)

        for k, vecs in ops.items():
            if vecs:
                self.operators[k] = np.mean(vecs, axis=0)
                mag = np.linalg.norm(self.operators[k])
                print(f"   :: OP '{k.upper()}' [Mag={mag:.2f}]")

    def reason(self, text_list, op):
        # 1. Encode Input
        seq = [self.vocab.get(w, 0) for w in text_list]
        t = torch.tensor([seq], dtype=torch.long).to(self.device)
        with torch.no_grad():
            z_start = self.model.encode(t).cpu().numpy()[0]
            
        # 2. Apply Logic
        op_vec = self.operators.get(op, np.zeros_like(z_start))
        z_new = z_start + op_vec
        
        # 3. Retrieve Nearest Thought (The "Snap")
        dists, indices = self.nbrs.kneighbors([z_new])
        idx = indices[0][0]
        result_text = self.sentences[idx]
        snap_dist = dists[0][0]
        
        print("\n" + "="*60)
        print(f"// COG: '{' '.join(text_list)}'")
        print(f">> VEC: [{op.upper()}]")
        print(f":: TRJ: [Snap Dist: {snap_dist:.4f}]")
        print(f"<< RET: '{result_text}'")
        print("="*60)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/semantic_v2")
    args = ap.parse_args()
    
    eng = SemanticEngine(args.data)
    eng.mine_operators()
    
    # Critical Test Cases
    eng.reason(["i", "create", "data"], "future")
    eng.reason(["system", "observe", "entropy"], "negation")
    eng.reason(["you", "ignore", "void"], "future")
    
    # Zero-Shot Test (A sentence not in the guarantee list?)
    eng.reason(["we", "analyze", "pattern"], "negation")

if __name__ == "__main__":
    main()