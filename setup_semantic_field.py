#!/usr/bin/env python3
# SETUP SCRIPT: SEMANTIC FIELD GENERATOR (Tokenless Language)
# 
# 1. Generates a "Synthetic Language" of fundamental concepts (Subject-Verb-Object).
# 2. Trains an LSTM-VAE to compress these sentences into a Semantic Manifold.
# 3. Saves the 'Brain' (Latents) and 'Dictionary' for BBIT navigation.

import os, argparse, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- 1. THE SYNTHETIC REALITY (Data Generator) ---
# We build concepts from fundamental parts so we can track them perfectly.
SUBJECTS = ["i", "you", "he", "she", "we", "they", "the system", "the user"]
VERBS = ["create", "destroy", "observe", "ignore", "love", "hate", "analyze", "delete"]
OBJECTS = ["data", "logic", "entropy", "the void", "the pattern", "self", "memory", "paradox"]
MODIFIERS = ["quickly", "slowly", "forever", "never", "with joy", "with fear"]

def generate_corpus(size=10000):
    sentences = []
    metadata = [] # Tracks the "Physics" of the sentence (Tense, Sentiment)
    
    for _ in range(size):
        s = random.choice(SUBJECTS)
        v = random.choice(VERBS)
        o = random.choice(OBJECTS)
        
        # Fundamental Structures
        structure = random.choice(["simple", "negation", "future", "past"])
        
        text = ""
        meta = {"root": f"{s} {v} {o}"}
        
        if structure == "simple":
            text = f"{s} {v}s {o}"
            meta["type"] = "declarative"
        elif structure == "negation":
            text = f"{s} does not {v} {o}"
            meta["type"] = "negation"
        elif structure == "future":
            text = f"{s} will {v} {o}"
            meta["type"] = "future"
        elif structure == "past":
            text = f"{s} {v}d {o}" # Simple heuristic for synthetic past
            meta["type"] = "past"
            
        sentences.append(text)
        metadata.append(meta)
        
    return sentences, metadata

# --- 2. THE TOKENLESS BRAIN (LSTM-VAE Architecture) ---
class SentenceVAE(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, latent_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Encoder (Text -> Thought)
        self.encoder_rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder (Thought -> Text)
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder_rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # x: (Batch, SeqLen)
        embed = self.embedding(x)
        
        # Encode
        _, (h_n, _) = self.encoder_rnn(embed) # h_n: (1, Batch, Hidden)
        h_n = h_n.squeeze(0)
        
        mu = self.fc_mu(h_n)
        logvar = self.fc_logvar(h_n)
        z = self.reparameterize(mu, logvar)
        
        # Decode (Teacher Forcing for training)
        # In a real tokenless system, we wouldn't use tokens here, but we need them to train the manifold.
        decoder_h = self.decoder_input(z).unsqueeze(0) # Init hidden state from thought
        c_0 = torch.zeros_like(decoder_h)
        
        output, _ = self.decoder_rnn(embed, (decoder_h, c_0))
        logits = self.fc_out(output)
        
        return logits, mu, logvar

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="./data/semantic_bbit")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Gen Data
    print("Generating Synthetic Reality...")
    sentences, meta = generate_corpus(size=20000)
    
    # Tokenizer (Character level for simplicity & robustness)
    chars = set("".join(sentences))
    vocab = {c: i+1 for i, c in enumerate(sorted(chars))}
    vocab["<PAD>"] = 0
    vocab_inv = {i: c for c, i in vocab.items()}
    
    print(f"Vocab Size: {len(vocab)}")
    
    # Vectorize
    max_len = max(len(s) for s in sentences)
    data_tensor = torch.zeros((len(sentences), max_len), dtype=torch.long)
    
    for i, s in enumerate(sentences):
        for j, c in enumerate(s):
            data_tensor[i, j] = vocab[c]
            
    dataset = torch.utils.data.TensorDataset(data_tensor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 2. Train VAE
    print("Training Semantic Brain...")
    model = SentenceVAE(len(vocab), latent_dim=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            x = batch[0].to(device)
            optimizer.zero_grad()
            
            recon, mu, logvar = model(x)
            
            # Loss: Reconstruction + KL
            # Flatten for CrossEntropy: (Batch*Seq, Vocab)
            recon_flat = recon.view(-1, len(vocab))
            x_flat = x.view(-1)
            
            BCE = criterion(recon_flat, x_flat)
            KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss = BCE + 0.01 * KLD # Weight KL gently
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    # 3. Extract Manifold & Save
    print("Extracting Semantic Manifold...")
    model.eval()
    all_z = []
    
    # We encode the entire corpus to get the "Atlas" of meaning
    full_tensor = data_tensor.to(device)
    batch_size = 512
    with torch.no_grad():
        for i in tqdm(range(0, len(full_tensor), batch_size)):
            batch = full_tensor[i:i+batch_size]
            _, mu, _ = model(batch) # We use Mu as the "Concept Location"
            all_z.append(mu.cpu().numpy())
            
    Z = np.vstack(all_z)
    
    # Save everything needed for the "English+" Decoder
    export = {
        "vocab": vocab,
        "sentences": sentences,
        "metadata": meta,
        "latents_path": "latents.npy"
    }
    
    np.save(os.path.join(args.outdir, "latents.npy"), Z)
    with open(os.path.join(args.outdir, "manifest.json"), "w") as f:
        json.dump(export, f)
        
    # Save model for the Decoder (to verify results)
    torch.save(model.state_dict(), os.path.join(args.outdir, "vae_model.pth"))
    
    print(f"[ok] Semantic Field built in {args.outdir}")

if __name__ == "__main__":
    main()