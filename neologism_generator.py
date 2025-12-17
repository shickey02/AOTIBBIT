#!/usr/bin/env python3
# NEOLOGISM GENERATOR (FIXED)
# Invents new words by interpolating phonetic vectors.

import os, argparse, json, torch
import numpy as np
import torch.nn as nn

class PhoneticVAE(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64, latent_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder_rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc_z = nn.Linear(latent_dim, hidden_dim)
        self.decoder_rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    # FIX: Added 'sos_idx' argument so we don't have to look it up here
    def decode(self, z, vocab_inv, sos_idx, max_len=15):
        # Generative Decoding
        h = self.fc_z(z).unsqueeze(0)
        
        # Start with the numeric ID of SOS
        curr_token = torch.tensor([[sos_idx]], device=z.device)
        
        word = ""
        for _ in range(max_len):
            embed = self.embedding(curr_token)
            out, h = self.decoder_rnn(embed, h)
            logits = self.fc_out(out)
            
            # Sample (Temperature 0.8 for creativity)
            probs = torch.softmax(logits[0, 0] / 0.8, dim=0)
            idx = torch.multinomial(probs, 1).item()
            
            if idx == 0: break # Pad
            
            char = vocab_inv.get(idx, "")
            # Don't print special tokens
            if isinstance(char, str) and len(char) == 1:
                word += char
            
            curr_token = torch.tensor([[idx]], device=z.device)
        return word

    def encode(self, x):
        embed = self.embedding(x)
        _, h_n = self.encoder_rnn(embed)
        mu = self.fc_mu(h_n.squeeze(0))
        return mu

class WordSmith:
    def __init__(self, data_dir):
        self.device = torch.device("cpu")
        with open(os.path.join(data_dir, "manifest.json"), "r") as f:
            self.manifest = json.load(f)
        self.vocab = self.manifest["vocab"]
        # Invert vocab correctly (Int -> Char)
        self.vocab_inv_int = {int(v): k for k, v in self.vocab.items()}
        # Add reverse lookup for encoding (Char -> Int)
        self.vocab_lookup = {k: v for k, v in self.vocab.items()}
        
        self.model = PhoneticVAE(len(self.vocab), latent_dim=16)
        self.model.load_state_dict(torch.load(os.path.join(data_dir, "model.pth"), map_location=self.device))
        self.model.eval()

    def get_vec(self, word):
        # Convert word to indices + SOS
        seq = [self.vocab_lookup.get("<SOS>")] + [self.vocab_lookup.get(c, 0) for c in word]
        t = torch.tensor([seq], dtype=torch.long)
        with torch.no_grad():
            return self.model.encode(t)[0]

    def invent(self, word1, word2, mix=0.5):
        v1 = self.get_vec(word1)
        v2 = self.get_vec(word2)
        
        # Vector Algebra (Interpolation)
        v_new = (v1 * (1 - mix)) + (v2 * mix)
        
        # Decode
        # FIX: Look up SOS ID here and pass it in
        sos_id = self.vocab_lookup["<SOS>"]
        new_word = self.model.decode(v_new.unsqueeze(0), self.vocab_inv_int, sos_idx=sos_id)
        
        print(f"// INVENTING: [{word1}] + [{word2}]")
        print(f">> VECTOR_MIX: {mix*100:.0f}% {word2}")
        print(f"<< NEOLOGISM: '{new_word}'")
        print("-" * 40)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/phonetic_bbit")
    args = ap.parse_args()
    
    smith = WordSmith(args.data)
    
    print("\n--- THE NEOLOGISM TEST ---")
    
    # 1. Tech + Nature (Cyber + Flow)
    smith.invent("cyber", "flow", mix=0.5)
    
    # 2. Light + Dark (Glimmer + Obsidian)
    smith.invent("glimmer", "obsidian", mix=0.5)
    
    # 3. Chaos + Order (Chaos + Logic)
    smith.invent("chaos", "logic", mix=0.5)
    
    # 4. Emotional Paradox (Joy + Doom)
    smith.invent("joy", "doom", mix=0.5)
    
    # 5. The Concept of Thought itself (Mind + Vector)
    smith.invent("mind", "vector", mix=0.5)

if __name__ == "__main__":
    main()