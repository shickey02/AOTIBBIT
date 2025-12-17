#!/usr/bin/env python3
# ROBUST SETUP: SEMANTIC AUTOENCODER (AE)
# Removes variational noise to force a structured, navigable semantic manifold.

import os, argparse, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- DATA GENERATOR ---
SUBJECTS = ["i", "you", "he", "she", "we", "the system"]
VERBS = ["create", "observe", "analyze", "destroy", "ignore"]
OBJECTS = ["data", "entropy", "logic", "the void", "paradox"]

def generate_corpus(size=20000):
    sentences = []
    metadata = []
    
    # Ensure specific test cases exist
    guaranteed = [
        ("i", "create", "data"),
        ("the system", "observe", "entropy"), # lemmatized for generation logic
        ("you", "analyze", "the void")
    ]
    
    for _ in range(size):
        if _ < len(guaranteed):
            s, v, o = guaranteed[_]
        else:
            s = random.choice(SUBJECTS)
            v = random.choice(VERBS)
            o = random.choice(OBJECTS)
            
        # Types
        modes = ["declarative", "future", "negation", "past"]
        mode = random.choice(modes)
        
        text = ""
        root = f"{s} {v} {o}" # Simple root for alignment
        
        if mode == "declarative": text = f"{s} {v}s {o}"
        elif mode == "future":    text = f"{s} will {v} {o}"
        elif mode == "negation":  text = f"{s} does not {v} {o}"
        elif mode == "past":      text = f"{s} {v}d {o}"
            
        sentences.append(text)
        metadata.append({"root": root, "type": mode})
        
    return sentences, metadata

# --- ROBUST AE ARCHITECTURE ---
class SentenceAE(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, latent_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Encoder
        self.encoder_rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc_z = nn.Linear(hidden_dim, latent_dim) # Direct mapping
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder_rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def encode_text(self, x):
        embed = self.embedding(x)
        _, (h_n, _) = self.encoder_rnn(embed)
        h_n = h_n.squeeze(0)
        z = self.fc_z(h_n)
        return z

    def forward(self, x):
        z = self.encode_text(x)
        
        # Decode (Teacher Forcing)
        decoder_h = self.decoder_input(z).unsqueeze(0)
        c_0 = torch.zeros_like(decoder_h)
        
        embed = self.embedding(x)
        output, _ = self.decoder_rnn(embed, (decoder_h, c_0))
        logits = self.fc_out(output)
        return logits, z

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="./data/semantic_bbit_robust")
    ap.add_argument("--epochs", type=int, default=30) # More epochs for text
    args = ap.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Data
    print("Generating Reality...")
    sentences, meta = generate_corpus()
    chars = sorted(list(set("".join(sentences))))
    # 0=PAD, 1=Start/End? Let's keep it simple: 0 is PAD.
    vocab = {c: i+1 for i, c in enumerate(chars)}
    vocab["<PAD>"] = 0
    
    # Vectorize
    max_len = max(len(s) for s in sentences)
    data_tensor = torch.zeros((len(sentences), max_len), dtype=torch.long)
    for i, s in enumerate(sentences):
        for j, c in enumerate(s):
            data_tensor[i, j] = vocab[c]
            
    loader = DataLoader(torch.utils.data.TensorDataset(data_tensor), batch_size=128, shuffle=True)

    # 2. Train
    print("Training Semantic AE...")
    model = SentenceAE(len(vocab), latent_dim=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            x = batch[0].to(device)
            optimizer.zero_grad()
            logits, z = model(x)
            
            logits_flat = logits.view(-1, len(vocab))
            x_flat = x.view(-1)
            loss = criterion(logits_flat, x_flat)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    # 3. Save Latents & Model
    print("Extracting Atlas...")
    model.eval()
    all_z = []
    
    # Encode all in batches
    full_tensor = data_tensor.to(device)
    with torch.no_grad():
        for i in range(0, len(full_tensor), 512):
            batch = full_tensor[i:i+512]
            z = model.encode_text(batch)
            all_z.append(z.cpu().numpy())
            
    Z = np.vstack(all_z)
    
    export = {
        "vocab": vocab,
        "sentences": sentences,
        "metadata": meta
    }
    
    np.save(os.path.join(args.outdir, "latents.npy"), Z)
    with open(os.path.join(args.outdir, "manifest.json"), "w") as f:
        json.dump(export, f)
    torch.save(model.state_dict(), os.path.join(args.outdir, "ae_model.pth"))
    print("[ok] Done.")

if __name__ == "__main__":
    main()