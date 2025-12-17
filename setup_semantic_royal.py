#!/usr/bin/env python3
# SETUP SCRIPT: THE ROYAL FIELD
# Generates a rigid hierarchical dataset to test Semantic Algebra (Analogy).

import os, argparse, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# THE HIERARCHY
# We pair them strictly: (Male, Female)
PAIRS = [
    ("king", "queen"),
    ("man", "woman"),
    ("prince", "princess"),
    ("father", "mother"),
    ("uncle", "aunt"),
    ("brother", "sister"),
    ("actor", "actress"),
    ("hero", "heroine")
]

ACTIONS = ["rules", "commands", "sees", "ignores", "loves"]
OBJECTS = ["the kingdom", "the people", "the gold", "the throne"]

def generate_corpus(size=25000):
    sentences = []
    
    # We generate dense combinations to ensure the manifold is packed
    for _ in range(size):
        # Pick a pair
        male, female = random.choice(PAIRS)
        action = random.choice(ACTIONS)
        obj = random.choice(OBJECTS)
        
        # Generate parallel sentences
        # "the king rules the kingdom"
        sentences.append(["the", male, action, obj])
        sentences.append(["the", female, action, obj])
        
    return sentences

# ROBUST AE (Standard architecture)
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

    def forward(self, x):
        z = self.encode(x)
        seq_len = x.size(1)
        z_expand = z.unsqueeze(1).repeat(1, seq_len, 1)
        embed = self.embedding(x)
        decode_in = torch.cat([embed, z_expand], dim=2)
        out, _ = self.decoder_rnn(decode_in)
        return self.fc_out(out), z

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="./data/semantic_royal")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Generating Royal Corpus...")
    raw_sents = generate_corpus()
    
    all_words = set(w for s in raw_sents for w in s)
    vocab = {w: i+1 for i, w in enumerate(sorted(all_words))}
    vocab["<PAD>"] = 0
    
    max_len = max(len(s) for s in raw_sents)
    data = torch.zeros((len(raw_sents), max_len), dtype=torch.long)
    for i, s in enumerate(raw_sents):
        for j, w in enumerate(s):
            data[i, j] = vocab[w]
            
    loader = DataLoader(torch.utils.data.TensorDataset(data), batch_size=128, shuffle=True)
    
    print("Training Royal Brain...")
    model = RoyalAE(len(vocab), latent_dim=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}", leave=False):
            x = batch[0].to(device)
            optimizer.zero_grad()
            logits, z = model(x)
            loss = criterion(logits.view(-1, len(vocab)), x.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    # Save Manifold
    print("Extracting Manifold...")
    model.eval()
    full_x = data.to(device)
    all_z = []
    with torch.no_grad():
        for i in range(0, len(full_x), 512):
            all_z.append(model.encode(full_x[i:i+512]).cpu().numpy())
    Z = np.vstack(all_z)
    
    # Save Sentences as flat strings for retrieval
    flat_sents = [" ".join(s) for s in raw_sents]
    
    export = {"vocab": vocab, "sentences": flat_sents}
    np.save(os.path.join(args.outdir, "latents.npy"), Z)
    with open(os.path.join(args.outdir, "manifest.json"), "w") as f:
        json.dump(export, f)
    torch.save(model.state_dict(), os.path.join(args.outdir, "model.pth"))
    print("[ok] Royal Field Built.")

if __name__ == "__main__":
    main()