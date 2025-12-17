#!/usr/bin/env python3
# SETUP TEMPORAL MANIFOLD (FIXED)
# Adds 'Rolls Remaining' to the input vector so the agent knows when to panic.

import os, argparse, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

# --- LOGIC ---
def analyze_hand_value(dice):
    # Standard max score calc
    counts = Counter(dice); s = sum(dice)
    scores = [dice.count(i)*i for i in range(1,7)]
    scores.append(s if any(c>=3 for c in counts.values()) else 0)
    scores.append(s if any(c>=4 for c in counts.values()) else 0)
    has_3 = any(c==3 for c in counts.values())
    has_2 = any(c==2 for c in counts.values())
    scores.append(25 if (has_3 and has_2) or any(c==5 for c in counts.values()) else 0)
    u = sorted(list(set(dice)))
    seq=0; m_seq=0
    for i in range(len(u)-1):
        if u[i+1]==u[i]+1: seq+=1
        else: seq=0
        m_seq=max(m_seq,seq)
    scores.append(30 if m_seq>=3 else 0)
    scores.append(40 if m_seq>=4 else 0)
    scores.append(50 if any(c==5 for c in counts.values()) else 0)
    scores.append(s)
    return max(scores)

def generate_data(size=100000):
    print(f"Generating {size} temporal states...")
    X = []
    Y_val = []
    
    for _ in range(size):
        dice = [random.randint(1,6) for _ in range(5)]
        rolls_left = random.randint(0, 2) # 0, 1, or 2 rerolls left
        
        # Input: [Count_1...Count_6, Rolls_Norm]
        counts = [0]*6
        for d in dice: counts[d-1]+=1
        
        # Normalize rolls (0.0, 0.5, 1.0)
        x_vec = counts + [rolls_left / 2.0]
        
        base_val = analyze_hand_value(dice)
        
        # Calculate sequence locally for the heuristic
        u = sorted(list(set(dice)))
        seq=0; m_seq=0
        for i in range(len(u)-1):
            if u[i+1]==u[i]+1: seq+=1
            else: seq=0
            m_seq=max(m_seq,seq)
        
        # Future Potential Heuristic
        potential = base_val
        if rolls_left > 0:
            # If we have a small straight (seq>=3), extra time is valuable
            if m_seq >= 3: potential += 10 * rolls_left 
            # If we have pairs, extra time is valuable
            if any(c>=2 for c in counts): potential += 5 * rolls_left 
            
        X.append(x_vec)
        Y_val.append(potential)
        
    return np.array(X, dtype=np.float32), np.array(Y_val, dtype=np.float32)

class TemporalBrain(nn.Module):
    def __init__(self):
        super().__init__()
        # Input 7: 6 Dice Counts + 1 Time
        self.encoder = nn.Sequential(
            nn.Linear(7, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.decoder_val = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        z = self.encoder(x)
        v = self.decoder_val(z)
        return v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="./data/yahtzee_temporal")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    
    X, Y = generate_data()
    dataset = TensorDataset(torch.tensor(X), torch.tensor(Y/50.0).unsqueeze(1))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = TemporalBrain()
    opt = optim.Adam(model.parameters(), lr=0.001)
    crit = nn.MSELoss()
    
    print("Training Temporal Brain...")
    for ep in range(15):
        loss_sum = 0
        for bx, by in loader:
            opt.zero_grad()
            pred = model(bx)
            loss = crit(pred, by)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
        print(f"Epoch {ep}: Loss {loss_sum:.4f}")
        
    torch.save(model.state_dict(), os.path.join(args.outdir, "model.pth"))
    
    # Save dummy Latents for compatibility
    np.save(os.path.join(args.outdir, "latents.npy"), np.zeros((1, 16)))
    np.save(os.path.join(args.outdir, "labels.npy"), np.zeros((1,)))
    print("Done.")

if __name__ == "__main__":
    main()