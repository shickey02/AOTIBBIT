#!/usr/bin/env python3
# AOTI LEVEL 7: THE ENTROPIC MANIFOLD
# Prevents "Zero Collapse" by enforcing Geometric Distinctions (Entropy).
# Learns complex 2D vector math from scratch.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# --- CONFIGURATION ---
RANGE = 20 # Smaller range for sharper visualization (-20 to +20)
VOCAB_SIZE = (RANGE * 2) + 1
OFFSET = RANGE

class EntropicBrain(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # 2D Vectors [Real, Imaginary]
        self.manifold = nn.Embedding(vocab_size, 2)
        
        # Initialize with distinct random noise to help breakout
        nn.init.normal_(self.manifold.weight, mean=0.0, std=0.5)

    def forward_geometry(self, idx_a, idx_b, op):
        vec_a = self.manifold(idx_a)
        vec_b = self.manifold(idx_b)
        
        if op == 'add':
            return vec_a + vec_b
        elif op == 'mult':
            # Complex Multiplication: (a+bi)(c+di)
            real = (vec_a[:,0] * vec_b[:,0]) - (vec_a[:,1] * vec_b[:,1])
            imag = (vec_a[:,0] * vec_b[:,1]) + (vec_a[:,1] * vec_b[:,0])
            return torch.stack([real, imag], dim=1)

    def entropy_loss(self):
        # FORCE DISPERSION: Punish vectors for being too close to each other
        # Sample a random subset to save compute
        indices = torch.randperm(VOCAB_SIZE)[:30]
        vectors = self.manifold(indices)
        
        # Calculate pairwise distances
        # (batch, 1, 2) - (1, batch, 2) broadcasts to (batch, batch, 2)
        diff = vectors.unsqueeze(1) - vectors.unsqueeze(0)
        dist_sq = torch.sum(diff**2, dim=2)
        
        # We want to maximize distance, so we minimize exp(-distance)
        # Add epsilon to diagonal to ignore self-distance
        eye = torch.eye(len(indices)).to(vectors.device)
        repulsion = torch.exp(-dist_sq) * (1 - eye)
        
        return torch.mean(repulsion)

# --- DATA GENERATION ---
def get_data(batch_size=32):
    add_batch, mult_batch = [], []
    for _ in range(batch_size):
        a = random.randint(-10, 10)
        b = random.randint(-10, 10)
        
        # ADD
        if abs(a+b) <= RANGE:
            add_batch.append([a+OFFSET, b+OFFSET, a+b+OFFSET])
        
        # MULT (Keep numbers small to stay in vocab range)
        ma, mb = random.randint(-4, 4), random.randint(-4, 4)
        if abs(ma*mb) <= RANGE:
            mult_batch.append([ma+OFFSET, mb+OFFSET, ma*mb+OFFSET])
            
    return torch.tensor(add_batch), torch.tensor(mult_batch)

# --- TRAINING ---
def train_entropy():
    print("--- AOTI LEVEL 7: ENTROPIC MATH ---")
    brain = EntropicBrain(VOCAB_SIZE)
    # Lower LR for stability
    optimizer = optim.Adam(brain.parameters(), lr=0.005) 
    
    for epoch in range(5001):
        optimizer.zero_grad()
        
        add_set, mult_set = get_data(64)
        
        # 1. Geometric Accuracy Loss
        pred_add = brain.forward_geometry(add_set[:,0], add_set[:,1], 'add')
        target_add = brain.manifold(add_set[:,2])
        loss_acc = torch.mean((pred_add - target_add)**2)
        
        pred_mult = brain.forward_geometry(mult_set[:,0], mult_set[:,1], 'mult')
        target_mult = brain.manifold(mult_set[:,2])
        loss_acc += torch.mean((pred_mult - target_mult)**2)
        
        # 2. Entropy Loss (The Repulsion Force)
        loss_entropy = brain.entropy_loss()
        
        # 3. Anchor Loss (The Unit Fix)
        # Force Vector(1) to be at [1.0, 0.0] approximately
        # This gives the universe a "Standard Unit" to build around.
        vec_one = brain.manifold(torch.tensor(1 + OFFSET))
        loss_anchor = torch.sum((vec_one - torch.tensor([1.0, 0.0]))**2)
        
        # Total Loss
        loss = loss_acc + (loss_entropy * 0.5) + (loss_anchor * 1.0)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f"Ep {epoch}: Acc {loss_acc:.4f} | Entropy {loss_entropy:.4f} | Anchor {loss_anchor:.4f}")
            
    return brain

# --- ALGEBRA SOLVER ---
def solve_x(brain, op, target, a):
    # Setup the Ghost Vector 'x'
    x = torch.zeros(2, requires_grad=True)
    # Optimizer for x
    opt = optim.Adam([x], lr=0.1)
    
    vec_a = brain.manifold(torch.tensor(a + OFFSET)).detach()
    vec_target = brain.manifold(torch.tensor(target + OFFSET)).detach()
    
    # "Thinking" Loop
    for _ in range(200):
        opt.zero_grad()
        
        if op == 'add':
            pred = x + vec_a
        elif op == 'mult':
            real = (x[0] * vec_a[0]) - (x[1] * vec_a[1])
            imag = (x[0] * vec_a[1]) + (x[1] * vec_a[0])
            pred = torch.stack([real, imag])
            
        loss = torch.sum((pred - vec_target)**2)
        loss.backward()
        opt.step()
        
    # Decode
    all_vecs = brain.manifold.weight.detach()
    dists = torch.norm(all_vecs - x, dim=1)
    result = torch.argmin(dists).item() - OFFSET
    return result

def verify(brain):
    print("\n--- GEOMETRIC DIAGNOSTICS ---")
    
    # Check 1 vs -1 Rotation
    v1 = brain.manifold(torch.tensor(1+OFFSET)).detach().numpy()
    v_neg1 = brain.manifold(torch.tensor(-1+OFFSET)).detach().numpy()
    
    ang1 = np.degrees(np.arctan2(v1[1], v1[0]))
    ang_neg1 = np.degrees(np.arctan2(v_neg1[1], v_neg1[0]))
    diff = abs(ang1 - ang_neg1)
    if diff > 180: diff = 360 - diff
    
    print(f"Angle(1): {ang1:.1f}° | Angle(-1): {ang_neg1:.1f}°")
    print(f"Difference: {diff:.1f}° (Ideal 180°)")
    
    print("\n--- ALGEBRA TEST ---")
    
    # Test 1: x + 5 = 8
    ans = solve_x(brain, 'add', 8, 5)
    print(f"x + 5 = 8  -> Solved: {ans} (Expect 3)")
    
    # Test 2: x + 10 = 5 (Negative Result)
    ans = solve_x(brain, 'add', 5, 10)
    print(f"x + 10 = 5 -> Solved: {ans} (Expect -5)")
    
    # Test 3: x * 2 = 6
    ans = solve_x(brain, 'mult', 6, 2)
    print(f"x * 2 = 6  -> Solved: {ans} (Expect 3)")
    
    # Test 4: x * -2 = 6 (Negative Slope)
    ans = solve_x(brain, 'mult', 6, -2)
    print(f"x * -2 = 6 -> Solved: {ans} (Expect -3)")

if __name__ == "__main__":
    ai = train_entropy()
    verify(ai)