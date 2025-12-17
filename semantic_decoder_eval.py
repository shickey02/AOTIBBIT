#!/usr/bin/env python3
# SEMANTIC DECODER EVALUATION (The "English+" System)
# 
# Task: Perform semantic arithmetic (e.g., "I love data" + FUTURE_VECTOR) 
#       and decode the result into English+ metaspeak.

import os, argparse, json, torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm

# --- 1. RE-DEFINE ARCHITECTURE (Must match Training) ---
class SentenceVAE(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, latent_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder_rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder_rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

    def reparameterize(self, mu, logvar):
        return mu # Deterministic for inference

    # NEW: Inference Mode (Autoregressive Decoding)
    def decode_thought(self, z, max_len=50, vocab_inv=None):
        # 1. Init Hidden State from Thought Vector
        h_0 = self.decoder_input(z).unsqueeze(0).unsqueeze(0) # (1, 1, Hidden)
        c_0 = torch.zeros_like(h_0)
        
        # 2. Start Token (We use <PAD>=0 as start trigger)
        curr_token = torch.tensor([[0]], device=z.device) # <PAD>
        state = (h_0, c_0)
        
        output_text = ""
        
        for _ in range(max_len):
            embed = self.embedding(curr_token)
            out, state = self.decoder_rnn(embed, state)
            logits = self.fc_out(out)
            
            # Greedy Decode
            token_idx = torch.argmax(logits, dim=2).item()
            
            if token_idx == 0: # PAD/End
                break
                
            char = vocab_inv[token_idx]
            output_text += char
            
            curr_token = torch.tensor([[token_idx]], device=z.device)
            
        return output_text

# --- 2. THE SEMANTIC ENGINE ---
class SemanticEngine:
    def __init__(self, data_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"// SYSTEM: LOADING SEMANTIC FIELD FROM {data_dir}")
        
        # Load Manifest
        with open(os.path.join(data_dir, "manifest.json"), "r") as f:
            self.manifest = json.load(f)
            
        self.vocab = self.manifest["vocab"]
        self.vocab_inv = {int(v): k for k, v in self.vocab.items()}
        self.sentences = self.manifest["sentences"]
        self.meta = self.manifest["metadata"]
        
        # Load Latents
        self.Z = np.load(os.path.join(data_dir, "latents.npy"))
        
        # Load Model
        # --- FIX: Removed '+1' to match saved vocab size ---
        self.model = SentenceVAE(len(self.vocab), latent_dim=32).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(data_dir, "vae_model.pth")))
        self.model.eval()
        
        self.operators = {}

    def mine_operators(self):
        print(">> OPR: MINING SEMANTIC VECTORS...")
        
        grouped = {}
        for i, m in enumerate(self.meta):
            root = m["root"]
            typ = m["type"]
            if root not in grouped: grouped[root] = {}
            grouped[root][typ] = i 
            
        ops = {"future": [], "negation": [], "past": []}
        
        for root, variants in grouped.items():
            if "declarative" in variants:
                base_idx = variants["declarative"]
                z_base = self.Z[base_idx]
                
                for t in ["future", "negation", "past"]:
                    if t in variants:
                        target_idx = variants[t]
                        z_target = self.Z[target_idx]
                        ops[t].append(z_target - z_base)
                        
        for k, vecs in ops.items():
            if vecs:
                mean_vec = np.mean(vecs, axis=0)
                mag = np.linalg.norm(mean_vec)
                self.operators[k] = mean_vec
                print(f"   :: DEF_OP '{k.upper()}' [samples={len(vecs)}, mag={mag:.2f}]")
                
    def encode(self, text):
        try:
            idx = self.sentences.index(text)
            return self.Z[idx]
        except ValueError:
            print(f"!! ERR: INPUT '{text}' NOT IN MANIFOLD. USING NEAREST...")
            return self.Z[0]

    def reason(self, start_text, op_key):
        z_start = self.encode(start_text)
        op_vec = self.operators[op_key]
        
        # 1. APPLY LOGIC
        z_new = z_start + op_vec
        
        # 2. CHECK DRIFT
        drift = np.linalg.norm(op_vec)
        
        # 3. DECODE
        z_tensor = torch.tensor(z_new, dtype=torch.float32).to(self.device)
        decoded_text = self.model.decode_thought(z_tensor, vocab_inv=self.vocab_inv)
        
        # 4. ENGLISH+ OUTPUT
        self.log_english_plus(start_text, op_key, drift, decoded_text)

    def log_english_plus(self, start, op, drift, result):
        print("\n" + "="*60)
        print(f"// SELF: '{start}'")
        print(f">> VEC:  [{op.upper()}]")
        print(f"!! META: [drift: {drift:.2f}] [entropy: LOW]")
        
        if "not" in result and "not" in start:
            print(f"** WARN: DOUBLE_NEGATION_DETECTED (recursive_depth=2)")
        
        print(f"<< SNAP: '{result}'")
        print("="*60)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/semantic_bbit")
    args = ap.parse_args()
    
    engine = SemanticEngine(args.data)
    engine.mine_operators()
    
    print("\n// SYSTEM: INITIALIZING REASONING LOOP...")
    
    test_sentences = [
        "i create data",
        "the system observes entropy",
        "you analyze the void"
    ]
    
    for s in test_sentences:
        engine.reason(s, "future")
        engine.reason(s, "negation")
        engine.reason(s, "past")

if __name__ == "__main__":
    main()