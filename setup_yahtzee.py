#!/usr/bin/env python3
# SETUP SCRIPT: YAHTZEE MANIFOLD
# Maps dice rolls into a semantic space of "Poker Hands".

import os, argparse, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

# --- YAHTZEE LOGIC ---
LABELS = {
    0: "Garbage",
    1: "Pair",
    2: "Two Pair",
    3: "Three of a Kind",
    4: "Full House",
    5: "Small Straight", # 4 in a row
    6: "Large Straight", # 5 in a row
    7: "Four of a Kind",
    8: "Yahtzee"         # 5 same
}

def analyze_hand(dice):
    counts = Counter(dice)
    vals = sorted(list(set(dice)))
    
    # Counts
    is_yahtzee = any(c == 5 for c in counts.values())
    is_four = any(c >= 4 for c in counts.values())
    is_three = any(c >= 3 for c in counts.values())
    is_pair = any(c >= 2 for c in counts.values())
    pairs = len([c for c in counts.values() if c >= 2])
    is_full = is_three and is_pair
    
    # Straights
    straight_len = 0
    max_straight = 0
    for i in range(len(vals)-1):
        if vals[i+1] == vals[i] + 1:
            straight_len += 1
        else:
            straight_len = 0
        max_straight = max(max_straight, straight_len)
    
    if is_yahtzee: return 8
    if is_four: return 7
    if max_straight >= 4: return 6 # Actual Large Straight length is 4 steps (1-2-3-4-5)
    if is_full: return 4 # Priority over straights/3oak usually
    if max_straight >= 3: return 5 # Small straight
    if is_three: return 3
    if pairs == 2: return 2
    if is_pair: return 1
    return 0

def generate_data(size=50000):
    print(f"Rolling {size} hands...")
    X = []
    Y = []
    
    for _ in range(size):
        # Weighted generation to ensure we see Rare hands
        if random.random() < 0.2:
            # Force a good hand structure
            base = random.randint(1, 6)
            hand = [base] * random.randint(2, 5)
            while len(hand) < 5: hand.append(random.randint(1, 6))
            random.shuffle(hand)
        else:
            # Random roll
            hand = [random.randint(1, 6) for _ in range(5)]
            
        hand = sorted(hand) # Sort to help the AI see patterns
        label = analyze_hand(hand)
        
        # Input Format: One-Hot Encoding of Dice?
        # Or just normalized values?
        # Let's use Count Vectors: [Count1, Count2, ... Count6]
        # This makes geometry easier (Full House = [0, 2, 0, 3, 0, 0])
        counts = [0]*6
        for d in hand: counts[d-1] += 1
        
        X.append(counts)
        Y.append(label)
        
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.longlong)

# --- BRAIN ---
class YahtzeeBrain(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 6 counts
        self.encoder = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 16), # 16D Latent Space
        )
        # We classify the hand type
        self.classifier = nn.Linear(16, 9) 

    def forward(self, x):
        z = self.encoder(x)
        logits = self.classifier(z)
        return logits, z

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="./data/yahtzee_bbit")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cpu") # Small model, CPU is fine
    
    X, Y = generate_data()
    
    # Train
    dataset = TensorDataset(torch.tensor(X), torch.tensor(Y))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = YahtzeeBrain().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    print("Training Dice Cortex...")
    model.train()
    for epoch in range(20):
        total_loss = 0
        for bx, by in loader:
            optimizer.zero_grad()
            logits, z = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    # Save Manifold
    model.eval()
    with torch.no_grad():
        _, Z = model(torch.tensor(X))
        
    np.save(os.path.join(args.outdir, "latents.npy"), Z.numpy())
    np.save(os.path.join(args.outdir, "labels.npy"), Y)
    torch.save(model.state_dict(), os.path.join(args.outdir, "model.pth"))
    print("[ok] Yahtzee Manifold Built.")

if __name__ == "__main__":
    main()