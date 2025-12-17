#!/usr/bin/env python3
# SETUP SCRIPT v2: PHONETIC PREDICTION
# Fixes the "Identity Trap" by training on Next-Token Prediction.

import os, argparse, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

WORDS = [
    "system", "nexus", "quantum", "cyber", "neural", "logic", "vector", "entropy",
    "void", "abyss", "hollow", "echo", "shadow", "whisper", "glimmer", "shimmer",
    "chaos", "havoc", "ruin", "storm", "thunder", "lightning", "crash", "boom",
    "flow", "river", "stream", "liquid", "drift", "surge", "pulse", "wave",
    "light", "bright", "shine", "glow", "radiant", "solar", "lunar", "star",
    "dark", "night", "gloom", "shade", "obsidian", "midnight", "dusk", "dawn",
    "fire", "flame", "burn", "inferno", "ember", "ash", "smoke", "blaze",
    "ice", "frost", "frozen", "cold", "crystal", "glass", "mirror", "prism",
    "mind", "thought", "dream", "memory", "reason", "idea", "concept", "theory",
    "love", "hate", "joy", "fear", "anger", "peace", "hope", "doom",
    "alpha", "omega", "zero", "one", "prime", "helix", "vortex", "matrix"
]

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

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # x shape: [batch, seq_len] containing [SOS, c1, c2, PAD]
        
        # 1. ENCODE
        embed = self.embedding(x)
        _, h_n = self.encoder_rnn(embed)
        h_n = h_n.squeeze(0)
        mu = self.fc_mu(h_n)
        logvar = self.fc_logvar(h_n)
        z = self.reparameterize(mu, logvar)
        
        # 2. DECODE
        # We use the latent Z to initialize the decoder hidden state
        h_0 = self.fc_z(z).unsqueeze(0)
        
        # Teacher Forcing: Feed the input sequence to generate the next tokens
        out, _ = self.decoder_rnn(embed, h_0)
        logits = self.fc_out(out)
        
        return logits, mu, logvar

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="./data/phonetic_bbit")
    ap.add_argument("--epochs", type=int, default=400) # Training longer for robust grammar
    args = ap.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Vocab
    chars = sorted(list(set("".join(WORDS))))
    vocab = {c: i+1 for i, c in enumerate(chars)}
    vocab["<PAD>"] = 0
    vocab["<SOS>"] = len(vocab)
    
    # Vectorize
    # Max len + 1 for SOS
    max_len = max(len(w) for w in WORDS) + 1 
    data = torch.zeros((len(WORDS), max_len), dtype=torch.long)
    
    for i, w in enumerate(WORDS):
        data[i, 0] = vocab["<SOS>"]
        for j, c in enumerate(w):
            data[i, j+1] = vocab[c]
            
    loader = DataLoader(torch.utils.data.TensorDataset(data), batch_size=32, shuffle=True)
    
    print("Training Phonetic Cortex (Next-Token)...")
    model = PhoneticVAE(len(vocab), latent_dim=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    
    model.train()
    for epoch in range(args.epochs):
        kl_weight = min(1.0, epoch / 100.0) * 0.05 
        total_loss = 0
        
        for batch in loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            
            # Forward
            logits, mu, logvar = model(x)
            
            # SHIFTED TARGETS
            # Input:  [SOS, A, B, C, PAD]
            # Output Logits matches Input length
            # Target: [A, B, C, PAD, PAD]
            
            # We slice logits to remove the last step (which predicts past the end)
            # We slice x to remove the first step (SOS) to create targets
            
            # Actually, standard way:
            # logits covers [SOS...LastChar]. 
            # prediction at step t (input SOS) should be x[t+1] (FirstChar).
            
            # Logits shape: [Batch, SeqLen, Vocab]
            logits_seq = logits[:, :-1, :] # Drop last prediction
            targets = x[:, 1:]             # Drop SOS
            
            recon_loss = nn.functional.cross_entropy(
                logits_seq.reshape(-1, len(vocab)), 
                targets.reshape(-1), 
                ignore_index=0
            )
            
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
            
            loss = recon_loss + kl_weight * kl_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss {total_loss:.4f}")

    print("Extracting Phonetic Manifold...")
    model.eval()
    export = {"vocab": vocab, "words": WORDS}
    
    full_x = data.to(device)
    with torch.no_grad():
        # For encoding, we just use the full sequence
        _, mu, _ = model(full_x)
        Z = mu.cpu().numpy()
        
    np.save(os.path.join(args.outdir, "latents.npy"), Z)
    with open(os.path.join(args.outdir, "manifest.json"), "w") as f:
        json.dump(export, f)
    torch.save(model.state_dict(), os.path.join(args.outdir, "model.pth"))
    print("[ok] Phonetic Field Built.")

if __name__ == "__main__":
    main()