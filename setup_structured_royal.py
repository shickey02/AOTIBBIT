#!/usr/bin/env python3
# SETUP SCRIPT: STRUCTURED MANIFOLD
# Forces the latent space to align with fundamental concepts (Gender, Power, Action).

import os, argparse, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- 1. THE ONTOLOGY (The Fundamental Parts) ---
# We define the "Physics" of this world explicitly.
# Format: Word -> [Gender, Power]
# Gender: 1=Male, -1=Female
# Power:  1=Royal, -1=Common, 0=Neutral
ENTITIES = {
    "king":    [1.0, 1.0],  "queen":    [-1.0, 1.0],
    "prince":  [1.0, 0.5],  "princess": [-1.0, 0.5],
    "man":     [1.0, -0.5], "woman":    [-1.0, -0.5],
    "father":  [1.0, 0.0],  "mother":   [-1.0, 0.0],
    "actor":   [1.0, -1.0], "actress":  [-1.0, -1.0]
}

ACTIONS = ["rules", "commands", "sees", "ignores", "loves"]
OBJECTS = ["the kingdom", "the people", "the gold", "the throne"]

def generate_data(size=10000):
    sentences = []
    labels = [] # We will train the Z vector to match these labels
    
    for _ in range(size):
        subj = random.choice(list(ENTITIES.keys()))
        act = random.choice(ACTIONS)
        obj = random.choice(OBJECTS)
        
        # Construct the "True Vector" (The Physics)
        # Dim 0: Gender
        # Dim 1: Power
        # Dim 2-6: Action (One-hotish or random projection? Let's keep it simple: Gender/Power focus)
        physics = ENTITIES[subj] # [Gender, Power]
        
        # Add noise to other dimensions so the model has to learn them too
        # but the first 2 dimensions are STRICT.
        
        sentences.append(["the", subj, act, obj])
        labels.append(physics)
        
    return sentences, np.array(labels, dtype=np.float32)

# --- 2. THE DISENTANGLED BRAIN ---
class StructuredAE(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, latent_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder_rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        
        # The Critical BottleNeck
        self.fc_z = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder_rnn = nn.LSTM(embed_dim + latent_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x):
        embed = self.embedding(x)
        _, (h_n, _) = self.encoder_rnn(embed)
        z = self.fc_z(h_n.squeeze(0))
        return z

    def forward(self, x):
        z = self.encode(x)
        
        # Standard Decode
        seq_len = x.size(1)
        z_expand = z.unsqueeze(1).repeat(1, seq_len, 1)
        embed = self.embedding(x)
        decode_in = torch.cat([embed, z_expand], dim=2)
        out, _ = self.decoder_rnn(decode_in)
        logits = self.fc_out(out)
        
        return logits, z

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="./data/semantic_structured")
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data
    print("Constructing Structured Reality...")
    raw_sents, physics_labels = generate_data()
    
    all_words = set(w for s in raw_sents for w in s)
    vocab = {w: i+1 for i, w in enumerate(sorted(all_words))}
    vocab["<PAD>"] = 0
    
    # Tensorize
    max_len = max(len(s) for s in raw_sents)
    data_x = torch.zeros((len(raw_sents), max_len), dtype=torch.long)
    for i, s in enumerate(raw_sents):
        for j, w in enumerate(s):
            data_x[i, j] = vocab[w]
            
    data_y = torch.tensor(physics_labels) # (N, 2)
    
    dataset = torch.utils.data.TensorDataset(data_x, data_y)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # Train
    print("Training Disentangled Cortex...")
    model = StructuredAE(len(vocab), latent_dim=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    
    criterion_recon = nn.CrossEntropyLoss(ignore_index=0)
    criterion_physics = nn.MSELoss() # Force Z to match Physics
    
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for bx, by in pbar:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            
            logits, z = model(bx)
            
            # 1. Reconstruction Loss (Standard AE stuff)
            loss_r = criterion_recon(logits.view(-1, len(vocab)), bx.view(-1))
            
            # 2. STRUCTURE LOSS (The Magic)
            # We force the first 2 dimensions of Z to match the Physics Labels
            z_core = z[:, :2] 
            loss_p = criterion_physics(z_core, by)
            
            # We weight Physics heavily so it prioritizes structure over perfect grammar
            loss = loss_r + 5.0 * loss_p 
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Phys": f"{loss_p.item():.4f}"})

    # Save
    print("Extracting Crystalized Manifold...")
    model.eval()
    all_z = []
    with torch.no_grad():
        # Encode strictly based on text input
        # Note: We do NOT use the labels here. The model must have internalized the physics.
        for i in range(0, len(data_x), 512):
            batch = data_x[i:i+512].to(device)
            z = model.encode(batch)
            all_z.append(z.cpu().numpy())
            
    Z = np.vstack(all_z)
    flat_sents = [" ".join(s) for s in raw_sents]
    
    export = {"vocab": vocab, "sentences": flat_sents}
    np.save(os.path.join(args.outdir, "latents.npy"), Z)
    with open(os.path.join(args.outdir, "manifest.json"), "w") as f:
        json.dump(export, f)
    torch.save(model.state_dict(), os.path.join(args.outdir, "model.pth"))
    print("[ok] Structured Field Built.")

if __name__ == "__main__":
    main()