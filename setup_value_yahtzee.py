#!/usr/bin/env python3
# SETUP SCRIPT: VALUE-AWARE MANIFOLD
# Trains a VAE that organizes the latent space by both PATTERN and SCORE.

import os, argparse, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

# --- YAHTZEE LOGIC (Reused) ---
def analyze_hand_value(dice):
    # Returns the MAX potential score of this hand (best category)
    counts = Counter(dice)
    s = sum(dice)
    
    scores = []
    # Upper
    for i in range(1, 7): scores.append(dice.count(i) * i)
    # Lower
    scores.append(s if any(c >= 3 for c in counts.values()) else 0) # 3k
    scores.append(s if any(c >= 4 for c in counts.values()) else 0) # 4k
    
    has_3 = any(c == 3 for c in counts.values())
    has_2 = any(c == 2 for c in counts.values())
    is_Y = any(c == 5 for c in counts.values())
    scores.append(25 if (has_3 and has_2) or is_Y else 0) # FH
    
    u = sorted(list(set(dice)))
    seq = 0
    max_seq = 0
    for i in range(len(u)-1):
        if u[i+1] == u[i]+1: seq+=1
        else: seq=0
        max_seq = max(max_seq, seq)
    
    scores.append(30 if max_seq >= 3 else 0) # SmStr
    scores.append(40 if max_seq >= 4 else 0) # LgStr
    scores.append(50 if is_Y else 0) # Yahtzee
    scores.append(s) # Chance
    
    return max(scores)

def analyze_hand_label(dice):
    # Same label logic as before for coloring the map
    counts = Counter(dice)
    if any(c == 5 for c in counts.values()): return 8
    if any(c == 4 for c in counts.values()): return 7
    
    u = sorted(list(set(dice)))
    seq=0
    max_seq=0
    for i in range(len(u)-1):
        if u[i+1] == u[i]+1: seq+=1
        else: seq=0
        max_seq = max(max_seq, seq)
        
    if max_seq>=4: return 6
    if max_seq>=3: return 5
    
    has_3 = any(c == 3 for c in counts.values())
    has_2 = any(c == 2 for c in counts.values())
    if has_3 and has_2: return 4
    if has_3: return 3
    if any(c==2 for c in counts.values()): return 1
    return 0

def generate_value_data(size=50000):
    print(f"Rolling {size} hands (Value Annotated)...")
    X = []
    Y_val = [] # The score
    Y_lbl = [] # The label (for centroids)
    
    for _ in range(size):
        if random.random() < 0.25:
            # Force structure
            base = random.randint(1, 6)
            hand = [base]*random.randint(2,5)
            while len(hand)<5: hand.append(random.randint(1,6))
            random.shuffle(hand)
        else:
            hand = [random.randint(1, 6) for _ in range(5)]
            
        # Input: Counts
        counts = [0]*6
        for d in hand: counts[d-1] += 1
        
        score = analyze_hand_value(hand)
        label = analyze_hand_label(hand)
        
        X.append(counts)
        Y_val.append(score)
        Y_lbl.append(label)
        
    return np.array(X, dtype=np.float32), np.array(Y_val, dtype=np.float32), np.array(Y_lbl, dtype=np.int64)

# --- DUAL-HEAD BRAIN ---
class ValueBrain(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(6, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16) # Latent Z
        )
        
        # Head 1: Reconstruction (Pattern)
        self.decoder_pattern = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 6) # Output counts
        )
        
        # Head 2: Valuation (Score)
        self.decoder_value = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 1) # Scalar score prediction
        )

    def forward(self, x):
        z = self.encoder(x)
        pat = self.decoder_pattern(z)
        val = self.decoder_value(z)
        return pat, val, z

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="./data/yahtzee_value")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cpu")
    
    X, Y_val, Y_lbl = generate_value_data()
    
    # Normalize Value (0-50 range -> 0-1) for easier training
    Y_val_norm = Y_val / 50.0
    
    dataset = TensorDataset(torch.tensor(X), torch.tensor(Y_val_norm).unsqueeze(1))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = ValueBrain().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    
    crit_pat = nn.MSELoss()
    crit_val = nn.MSELoss()
    
    print("Training Value-Aware Cortex...")
    model.train()
    for epoch in range(25):
        total_loss = 0
        for bx, by in loader:
            optimizer.zero_grad()
            rec_pat, rec_val, z = model(bx)
            
            # Multi-Task Loss
            # We care about pattern match AND value prediction
            l_p = crit_pat(rec_pat, bx)
            l_v = crit_val(rec_val, by)
            
            # Weight value heavily so the latent space organizes by worth
            loss = l_p + (2.0 * l_v)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    # Save Manifold
    model.eval()
    with torch.no_grad():
        _, _, Z = model(torch.tensor(X))
        
    np.save(os.path.join(args.outdir, "latents.npy"), Z.numpy())
    np.save(os.path.join(args.outdir, "labels.npy"), Y_lbl)
    torch.save(model.state_dict(), os.path.join(args.outdir, "model.pth"))
    print("[ok] Value Manifold Built.")

if __name__ == "__main__":
    main()