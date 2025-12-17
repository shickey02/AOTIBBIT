#!/usr/bin/env python3
import os, argparse, json, torch
import numpy as np
import torch.nn as nn

# --- ARCHITECTURE COPY ---
class SentenceAE(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, latent_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder_rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc_z = nn.Linear(hidden_dim, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder_rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def encode_text(self, x):
        embed = self.embedding(x)
        _, (h_n, _) = self.encoder_rnn(embed)
        h_n = h_n.squeeze(0)
        z = self.fc_z(h_n)
        return z

    def decode_thought(self, z, max_len=50, vocab_inv=None):
        h_0 = self.decoder_input(z).unsqueeze(0).unsqueeze(0)
        c_0 = torch.zeros_like(h_0)
        curr_token = torch.tensor([[0]], device=z.device) # Pad as start
        state = (h_0, c_0)
        output_text = ""
        
        for _ in range(max_len):
            embed = self.embedding(curr_token)
            out, state = self.decoder_rnn(embed, state)
            logits = self.fc_out(out)
            token_idx = torch.argmax(logits, dim=2).item()
            if token_idx == 0: break
            
            output_text += vocab_inv[token_idx]
            curr_token = torch.tensor([[token_idx]], device=z.device)
        return output_text

class SemanticEngine:
    def __init__(self, data_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"// SYSTEM: LOADING SEMANTIC FIELD FROM {data_dir}")
        
        with open(os.path.join(data_dir, "manifest.json"), "r") as f:
            self.manifest = json.load(f)
        self.vocab = self.manifest["vocab"]
        self.vocab_inv = {int(v): k for k, v in self.vocab.items()}
        self.sentences = self.manifest["sentences"]
        self.meta = self.manifest["metadata"]
        self.Z = np.load(os.path.join(data_dir, "latents.npy"))
        
        self.model = SentenceAE(len(self.vocab), latent_dim=32).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(data_dir, "ae_model.pth")))
        self.model.eval()
        
        self.operators = {}
        self.centroids = None # For drift calculation

    def mine_operators(self):
        print(">> OPR: MINING VECTORS...")
        grouped = {}
        for i, m in enumerate(self.meta):
            root = m["root"]
            typ = m["type"]
            if root not in grouped: grouped[root] = {}
            grouped[root][typ] = i
            
        ops = {"future": [], "negation": [], "past": []}
        for root, variants in grouped.items():
            if "declarative" in variants:
                z_base = self.Z[variants["declarative"]]
                for t in ops:
                    if t in variants:
                        z_tgt = self.Z[variants[t]]
                        ops[t].append(z_tgt - z_base)
                        
        for k, vecs in ops.items():
            if vecs:
                mean = np.mean(vecs, axis=0)
                mag = np.linalg.norm(mean)
                self.operators[k] = mean
                print(f"   :: DEF_OP '{k.upper()}' [samples={len(vecs)}, mag={mag:.2f}]")

        # Compute centroid of valid manifold for drift
        self.centroids = np.mean(self.Z, axis=0)

    def encode_live(self, text):
        # Real encoding via LSTM
        seq = [self.vocab.get(c, 0) for c in text]
        tensor = torch.tensor([seq], dtype=torch.long).to(self.device)
        with torch.no_grad():
            z = self.model.encode_text(tensor)
        return z.cpu().numpy()[0]

    def reason(self, start_text, op_key):
        z_start = self.encode_live(start_text)
        op_vec = self.operators.get(op_key, np.zeros_like(z_start))
        
        # 1. Logic
        z_new = z_start + op_vec
        
        # 2. English+ Stats
        op_mag = np.linalg.norm(op_vec)
        # Drift: How far is z_new from the global center compared to z_start?
        dist_start = np.linalg.norm(z_start - self.centroids)
        dist_end = np.linalg.norm(z_new - self.centroids)
        drift = dist_end - dist_start # Positive means moving away from known reality
        
        # 3. Decode
        z_tensor = torch.tensor(z_new, dtype=torch.float32).to(self.device)
        decoded = self.model.decode_thought(z_tensor, vocab_inv=self.vocab_inv)
        
        self.log_english_plus(start_text, op_key, op_mag, drift, decoded)

    def log_english_plus(self, start, op, mag, drift, result):
        print("\n" + "="*60)
        print(f"// COG: '{start}'")
        print(f">> VEC: [{op.upper()}] (Magnitude: {mag:.2f})")
        
        drift_status = "STABLE"
        if drift > 2.0: drift_status = "HIGH_ENTROPY"
        elif drift > 5.0: drift_status = "PARADOX_REGION"
        
        print(f":: TRJ: [Direct] -> [Drift: {drift:.2f}] -> [{drift_status}]")
        
        # Meta-Commentary
        if "not" in result and "not" in start:
            print(f"** WARN: RECURSIVE_NEGATION")
        if result == "":
            print(f"!! CRIT: COLLAPSE (Thought terminated)")
        else:
            print(f"<< OUT: '{result}'")
        print("="*60)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/semantic_bbit_robust")
    args = ap.parse_args()
    
    engine = SemanticEngine(args.data)
    engine.mine_operators()
    
    print("\n// SYSTEM: TOKENLESS REASONING ONLINE...")
    
    # Test Sentences (Some guaranteed, some new)
    tests = [
        "i create data",
        "the system observes entropy", 
        "you ignore the void" # New combination
    ]
    
    for s in tests:
        engine.reason(s, "future")
        engine.reason(s, "negation")
        # Chaining? Let's try manual chaining logic in output
        # (Just simple ops for now)

if __name__ == "__main__":
    main()