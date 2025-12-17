#!/usr/bin/env python3
import os, argparse, json, torch
import numpy as np
import torch.nn as nn

class ThoughtAE(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, latent_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder_rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc_z = nn.Linear(hidden_dim, latent_dim)
        # Decoder input is now Embedding + Latent
        self.decoder_rnn = nn.LSTM(embed_dim + latent_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.latent_dim = latent_dim

    def encode(self, x):
        embed = self.embedding(x)
        _, (h_n, _) = self.encoder_rnn(embed)
        z = self.fc_z(h_n.squeeze(0))
        return z

    def decode(self, z, max_len=10, vocab_inv=None):
        # Decode loop with injection
        curr_token = torch.tensor([[0]], device=z.device) # PAD/Start
        h_0 = torch.zeros(1, 1, 128, device=z.device)
        c_0 = torch.zeros(1, 1, 128, device=z.device)
        state = (h_0, c_0)
        
        words = []
        for _ in range(max_len):
            embed = self.embedding(curr_token) # (1, 1, Embed)
            z_in = z.unsqueeze(0).unsqueeze(0) # (1, 1, Latent)
            
            # INJECT: Input = Embed + Thought
            decoder_input = torch.cat([embed, z_in], dim=2)
            
            out, state = self.decoder_rnn(decoder_input, state)
            logits = self.fc_out(out)
            idx = torch.argmax(logits, dim=2).item()
            
            if idx == 0: break
            words.append(vocab_inv[idx])
            curr_token = torch.tensor([[idx]], device=z.device)
            
        return " ".join(words)

class SemanticEngine:
    def __init__(self, data_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"// SYSTEM: LOADING V2 FIELD FROM {data_dir}")
        
        with open(os.path.join(data_dir, "manifest.json"), "r") as f:
            self.manifest = json.load(f)
        self.vocab = self.manifest["vocab"]
        self.vocab_inv = {int(v): k for k, v in self.vocab.items()}
        self.meta = self.manifest["metadata"]
        self.Z = np.load(os.path.join(data_dir, "latents.npy"))
        
        self.model = ThoughtAE(len(self.vocab), latent_dim=32).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(data_dir, "model.pth")))
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
        seq = [self.vocab.get(w, 0) for w in text_list]
        t = torch.tensor([seq], dtype=torch.long).to(self.device)
        with torch.no_grad():
            z_start = self.model.encode(t)
            z_start_np = z_start.cpu().numpy()[0]
            
        op_vec = self.operators.get(op, np.zeros_like(z_start_np))
        z_new = z_start_np + op_vec
        
        z_new_t = torch.tensor(z_new, dtype=torch.float32).to(self.device)
        res = self.model.decode(z_new_t, vocab_inv=self.vocab_inv)
        
        print("\n" + "="*50)
        print(f"// COG: '{' '.join(text_list)}'")
        print(f">> VEC: [{op.upper()}]")
        print(f"<< OUT: '{res}'")
        print("="*50)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/semantic_v2")
    args = ap.parse_args()
    
    eng = SemanticEngine(args.data)
    eng.mine_operators()
    
    eng.reason(["i", "create", "data"], "future")
    eng.reason(["system", "observe", "entropy"], "negation")
    eng.reason(["you", "ignore", "void"], "future")

if __name__ == "__main__":
    main()