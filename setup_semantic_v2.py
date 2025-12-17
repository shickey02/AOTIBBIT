#!/usr/bin/env python3
# SETUP SCRIPT v2.1: THOUGHT INJECTION (FIXED EXPORT)
# Concatenates the thought vector 'z' to every input token.
# NOW SAVES SENTENCES CORRECTLY.

import os, argparse, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- REALITY GENERATOR ---
SUBJECTS = ["i", "you", "we", "system"]
VERBS = ["create", "observe", "analyze", "destroy", "ignore"]
OBJECTS = ["data", "entropy", "void", "paradox", "pattern"]

def generate_corpus(size=20000):
    sentences = []
    metadata = []
    
    # Hardcoded patterns to ensure the model learns specific rules
    for _ in range(size):
        s = random.choice(SUBJECTS)
        v = random.choice(VERBS)
        o = random.choice(OBJECTS)
        mode = random.choice(["declarative", "future", "negation"])
        
        words = []
        if mode == "declarative": words = [s, v, o]
        elif mode == "future":    words = [s, "will", v, o]
        elif mode == "negation":  words = [s, "not", v, o]
            
        sentences.append(words)
        metadata.append({"root": f"{s} {v} {o}", "type": mode})
        
    return sentences, metadata

# --- ARCHITECTURE WITH INJECTION ---
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

    def forward(self, x):
        z = self.encode(x) # (Batch, Latent)
        
        # Expand z to match sequence length: (Batch, Seq, Latent)
        seq_len = x.size(1)
        z_expanded = z.unsqueeze(1).repeat(1, seq_len, 1)
        
        embed = self.embedding(x) # (Batch, Seq, Embed)
        
        # INJECT THOUGHT: Concatenate Embed + Z
        decoder_input = torch.cat([embed, z_expanded], dim=2)
        
        # Run Decoder
        output, _ = self.decoder_rnn(decoder_input)
        logits = self.fc_out(output)
        return logits, z

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="./data/semantic_v2")
    ap.add_argument("--epochs", type=int, default=25)
    args = ap.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Generating Reality...")
    raw_sents, meta = generate_corpus()
    
    all_words = set(w for s in raw_sents for w in s)
    vocab = {w: i+1 for i, w in enumerate(sorted(all_words))}
    vocab["<PAD>"] = 0
    print(f"Vocab Size: {len(vocab)}")
    
    max_len = max(len(s) for s in raw_sents)
    data = torch.zeros((len(raw_sents), max_len), dtype=torch.long)
    for i, s in enumerate(raw_sents):
        for j, w in enumerate(s):
            data[i, j] = vocab[w]
            
    loader = DataLoader(torch.utils.data.TensorDataset(data), batch_size=128, shuffle=True)
    
    print("Training Thought AE...")
    model = ThoughtAE(len(vocab), latent_dim=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            x = batch[0].to(device)
            optimizer.zero_grad()
            logits, z = model(x)
            loss = criterion(logits.view(-1, len(vocab)), x.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    print("Saving Atlas...")
    model.eval()
    # Save Latents
    full_x = data.to(device)
    all_z = []
    with torch.no_grad():
        for i in range(0, len(full_x), 512):
            batch = full_x[i:i+512]
            z = model.encode(batch)
            all_z.append(z.cpu().numpy())
    Z = np.vstack(all_z)
    
    # --- FIX IS HERE: Added 'sentences' to export ---
    export = {
        "vocab": vocab, 
        "metadata": meta, 
        "sentences": raw_sents
    }
    
    np.save(os.path.join(args.outdir, "latents.npy"), Z)
    with open(os.path.join(args.outdir, "manifest.json"), "w") as f:
        json.dump(export, f)
    torch.save(model.state_dict(), os.path.join(args.outdir, "model.pth"))
    print("[ok] Done.")

if __name__ == "__main__":
    main()