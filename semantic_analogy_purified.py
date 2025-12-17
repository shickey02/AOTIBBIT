#!/usr/bin/env python3
# SEMANTIC ANALOGY (PURIFIED)
# Uses Monte Carlo averaging to isolate pure concept vectors (Gender, Status).

import os, argparse, json, torch, random
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

class PurifiedEngine:
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

    def get_batch_vec(self, text_list):
        # Encodes a list of sentences at once
        batch = []
        for t in text_list:
            batch.append([self.vocab.get(w, 0) for w in t.split()])
        
        # Pad
        max_len = max(len(s) for s in batch)
        tensor = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, s in enumerate(batch):
            for j, w in enumerate(s):
                tensor[i, j] = w
        
        with torch.no_grad():
            return self.model.encode(tensor.to(self.device)).cpu().numpy()

    def mine_pure_vector(self, word_a, word_b, samples=100):
        # Generates pairs: "The {word_a} rules..." vs "The {word_b} rules..."
        # Returns Avg(B - A)
        
        templates = [
            "the {} rules the kingdom", "the {} sees the gold", 
            "the {} loves the people", "the {} commands the throne",
            "the {} ignores the void", "the {} sees the pattern"
        ]
        
        list_a = []
        list_b = []
        
        # Create parallel lists (context must match perfectly)
        for t in templates:
            list_a.append(t.format(word_a))
            list_b.append(t.format(word_b))
            
        vecs_a = self.get_batch_vec(list_a)
        vecs_b = self.get_batch_vec(list_b)
        
        # Calculate diffs
        diffs = vecs_b - vecs_a
        
        # Average
        pure_vec = np.mean(diffs, axis=0)
        return pure_vec

    def solve(self, start_text, vec_name, pure_vec):
        z_start = self.get_batch_vec([start_text])[0]
        z_res = z_start + pure_vec
        
        dists, indices = self.nbrs.kneighbors([z_res])
        res_text = self.sentences[indices[0][0]]
        
        print(f"// QUERY: ({start_text}) + [PURE_{vec_name}]")
        print(f"<< RET: '{res_text}' (Dist: {dists[0][0]:.4f})")
        print("-" * 50)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/semantic_structured")
    args = ap.parse_args()
    eng = PurifiedEngine(args.data)
    
    print("Mining Pure Concepts...")
    # Mine "Femaleness" (Woman - Man)
    v_female = eng.mine_pure_vector("man", "woman")
    
    # Mine "Royalness" (King - Man)
    v_royal = eng.mine_pure_vector("man", "king")
    
    print("\n--- PURIFIED LOGIC TESTS ---")
    
    # 1. King + Female -> Queen? (The Princess Trap Breaker)
    eng.solve("the king rules the kingdom", "FEMALE", v_female)
    
    # 2. Prince + Female -> Princess?
    eng.solve("the prince sees the gold", "FEMALE", v_female)
    
    # 3. Actor + Royal -> King? (Status Promotion)
    eng.solve("the actor commands the people", "ROYAL", v_royal)
    
    # 4. Constructive Logic: Man + Royal + Female -> Queen?
    # This combines two pure vectors
    v_composite = v_royal + v_female
    eng.solve("the man rules the kingdom", "ROYAL+FEMALE", v_composite)

if __name__ == "__main__":
    main()