#!/usr/bin/env python3
import os, argparse, json, torch
import numpy as np
import torch.nn as nn

class WordAE(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, latent_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder_rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc_z = nn.Linear(hidden_dim, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder_rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x):
        embed = self.embedding(x)
        _, (h_n, _) = self.encoder_rnn(embed)
        z = self.fc_z(h_n.squeeze(0))
        return z

    def decode(self, z, max_len=10, vocab_inv=None):
        h_0 = self.decoder_input(z).unsqueeze(0)
        c_0 = torch.zeros_like(h_0)
        
        # Start with a dummy token (0=PAD)
        curr_token = torch.tensor([[0]], device=z.device)
        state = (h_0, c_0)
        
        words = []
        for _ in range(max_len):
            embed = self.embedding(curr_token)
            out, state = self.decoder_rnn(embed, state)
            logits = self.fc_out(out)
            idx = torch.argmax(logits, dim=2).item()
            
            if idx == 0: break # Pad
            words.append(vocab_inv[idx])
            curr_token = torch.tensor([[idx]], device=z.device)
            
        return " ".join(words)

class SemanticEngine:
    def __init__(self, data_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"// SYSTEM: LOADING WORD FIELD FROM {data_dir}")
        
        with open(os.path.join(data_dir, "manifest.json"), "r") as f:
            self.manifest = json.load(f)
        self.vocab = self.manifest["vocab"]
        # Invert vocab: string keys need to be ints
        self.vocab_inv = {int(v): k for k, v in self.vocab.items()}
        self.sentences = self.manifest["sentences"]
        self.meta = self.manifest["metadata"]
        self.Z = np.load(os.path.join(data_dir, "latents.npy"))
        
        # --- FIX IS HERE: Removed '+1' ---
        self.model = WordAE(len(self.vocab), latent_dim=32).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(data_dir, "ae_model.pth")))
        self.model.eval()
        
        self.operators = {}

    def mine_operators(self):
        print(">> OPR: MINING VECTORS...")
        grouped = {}
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
                print(f"   :: OP '{k.upper()}' [Mag={np.linalg.norm(self.operators[k]):.2f}]")

    def reason(self, text_list, op):
        # Encode live
        seq = [self.vocab.get(w, 0) for w in text_list]
        t = torch.tensor([seq], dtype=torch.long).to(self.device)
        with torch.no_grad():
            z_start = self.model.encode(t)
            z_start_np = z_start.cpu().numpy()[0]
            
        op_vec = self.operators.get(op, np.zeros_like(z_start_np))
        z_new = z_start_np + op_vec
        
        # Decode
        z_new_t = torch.tensor([z_new], dtype=torch.float32).to(self.device)
        res = self.model.decode(z_new_t, vocab_inv=self.vocab_inv)
        
        print("\n" + "="*50)
        print(f"// COG: '{' '.join(text_list)}'")
        print(f">> VEC: [{op.upper()}]")
        print(f"<< OUT: '{res}'")
        print("="*50)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/semantic_word_bbit")
    args = ap.parse_args()
    
    eng = SemanticEngine(args.data)
    eng.mine_operators()
    
    # Test: "i create data" -> Future?
    eng.reason(["i", "create", "data"], "future")
    eng.reason(["system", "observe", "entropy"], "negation")
    eng.reason(["you", "ignore", "void"], "future")

if __name__ == "__main__":
    main()