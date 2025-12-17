#!/usr/bin/env python3
# SETUP SCRIPT: PHONETIC MANIFOLD
# Trains a Char-VAE on evocative words to enable "Neologism Generation".

import os, argparse, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- THE LEXICON ---
# A mix of high-tech, abstract, and natural words.
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
    "love", "hate", "joy", "fear", "anger", "peace", "hope", "doom"
]

# --- ARCHITECTURE: CHARACTER VAE ---
class PhoneticVAE(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64, latent_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Encoder
        self.encoder_rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.fc_z = nn.Linear(latent_dim, hidden_dim)
        self.decoder_rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        embed = self.embedding(x)
        _, h_n = self.encoder_rnn(embed)
        h_n = h_n.squeeze(0)
        
        mu = self.fc_mu(h_n)
        logvar = self.fc_logvar(h_n)
        z = self.reparameterize(mu, logvar)
        
        # Decode (Teacher Forcing)
        h_0 = self.fc_z(z).unsqueeze(0)
        out, _ = self.decoder_rnn(embed, h_0)
        logits = self.fc_out(out)
        
        return logits, mu, logvar

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="./data/phonetic_bbit")
    ap.add_argument("--epochs", type=int, default=300) # Needs many epochs to learn spelling
    args = ap.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Prepare Data
    chars = sorted(list(set("".join(WORDS))))
    vocab = {c: i+1 for i, c in enumerate(chars)}
    vocab["<PAD>"] = 0
    vocab["<SOS>"] = len(vocab) # Start token
    
    # Vectorize
    max_len = max(len(w) for w in WORDS) + 1 # +1 for SOS
    data = torch.zeros((len(WORDS), max_len), dtype=torch.long)
    
    for i, w in enumerate(WORDS):
        data[i, 0] = vocab["<SOS>"]
        for j, c in enumerate(w):
            data[i, j+1] = vocab[c]
            
    loader = DataLoader(torch.utils.data.TensorDataset(data), batch_size=32, shuffle=True)
    
    # 2. Train with KL Annealing
    print("Training Phonetic Cortex...")
    model = PhoneticVAE(len(vocab), latent_dim=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    model.train()
    for epoch in range(args.epochs):
        # KL Annealing: Start at 0, ramp up to 1
        kl_weight = min(1.0, epoch / 100.0) * 0.1 
        
        total_loss = 0
        for batch in loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            
            logits, mu, logvar = model(x)
            
            # Reconstruction Loss
            recon_loss = nn.functional.cross_entropy(logits.reshape(-1, len(vocab)), x.reshape(-1), ignore_index=0)
            
            # KL Divergence
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
            
            loss = recon_loss + kl_weight * kl_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss {total_loss:.4f} (KL-W: {kl_weight:.4f})")

    # 3. Save
    print("Extracting Phonetic Manifold...")
    model.eval()
    export = {"vocab": vocab, "words": WORDS}
    
    # Save Latents
    full_x = data.to(device)
    with torch.no_grad():
        _, mu, _ = model(full_x)
        Z = mu.cpu().numpy()
        
    np.save(os.path.join(args.outdir, "latents.npy"), Z)
    with open(os.path.join(args.outdir, "manifest.json"), "w") as f:
        json.dump(export, f)
    torch.save(model.state_dict(), os.path.join(args.outdir, "model.pth"))
    print("[ok] Phonetic Field Built.")

if __name__ == "__main__":
    main()