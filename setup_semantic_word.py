#!/usr/bin/env python3
# SETUP SCRIPT: WORD-LEVEL SEMANTIC AE
# Uses whole words as tokens to prevent "Posterior Collapse".

import os, argparse, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- 1. REALITY GENERATOR ---
SUBJECTS = ["i", "you", "we", "system"]
VERBS = ["create", "observe", "analyze", "destroy", "ignore"]
OBJECTS = ["data", "entropy", "void", "paradox", "pattern"]

def generate_corpus(size=20000):
    sentences = []
    metadata = []
    
    # Guaranteed Test Cases
    guaranteed = [
        ("i", "create", "data"),
        ("system", "observe", "entropy"),
        ("you", "ignore", "void")
    ]
    
    for _ in range(size):
        if _ < len(guaranteed):
            s, v, o = guaranteed[_]
        else:
            s = random.choice(SUBJECTS)
            v = random.choice(VERBS)
            o = random.choice(OBJECTS)
            
        mode = random.choice(["declarative", "future", "negation", "past"])
        
        # Word List Construction
        words = []
        if mode == "declarative": words = [s, v, o]
        elif mode == "future":    words = [s, "will", v, o]
        elif mode == "negation":  words = [s, "not", v, o]
        elif mode == "past":      words = [s, v, "past", o] # Marker for simplicity
            
        sentences.append(words)
        metadata.append({"root": f"{s} {v} {o}", "type": mode})
        
    return sentences, metadata

# --- 2. ARCHITECTURE ---
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

    def forward(self, x):
        z = self.encode(x)
        
        # Decoder init
        h_0 = self.decoder_input(z).unsqueeze(0)
        c_0 = torch.zeros_like(h_0)
        
        embed = self.embedding(x)
        output, _ = self.decoder_rnn(embed, (h_0, c_0))
        logits = self.fc_out(output)
        return logits, z

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="./data/semantic_word_bbit")
    ap.add_argument("--epochs", type=int, default=25)
    args = ap.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data
    print("Generating Corpus...")
    raw_sents, meta = generate_corpus()
    
    # Build Vocab
    all_words = set(w for s in raw_sents for w in s)
    vocab = {w: i+1 for i, w in enumerate(sorted(all_words))}
    vocab["<PAD>"] = 0
    print(f"Vocab Size: {len(vocab)}")
    
    # Tensorize
    max_len = max(len(s) for s in raw_sents)
    data = torch.zeros((len(raw_sents), max_len), dtype=torch.long)
    for i, s in enumerate(raw_sents):
        for j, w in enumerate(s):
            data[i, j] = vocab[w]
            
    loader = DataLoader(torch.utils.data.TensorDataset(data), batch_size=64, shuffle=True)
    
    # Train
    print("Training Word AE...")
    model = WordAE(len(vocab), latent_dim=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.003)
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

    # Save
    print("Saving Atlas...")
    model.eval()
    all_z = []
    full_x = data.to(device)
    
    with torch.no_grad():
        for i in range(0, len(full_x), 512):
            batch = full_x[i:i+512]
            z = model.encode(batch)
            all_z.append(z.cpu().numpy())
            
    Z = np.vstack(all_z)
    
    # Re-stitch sentences for manifest
    joined_sents = [" ".join(s) for s in raw_sents]
    
    export = {"vocab": vocab, "sentences": joined_sents, "metadata": meta}
    np.save(os.path.join(args.outdir, "latents.npy"), Z)
    with open(os.path.join(args.outdir, "manifest.json"), "w") as f:
        json.dump(export, f)
    torch.save(model.state_dict(), os.path.join(args.outdir, "ae_model.pth"))
    print("[ok] Done.")

if __name__ == "__main__":
    main()