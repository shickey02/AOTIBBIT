#!/usr/bin/env python3
# AOTI LEVEL 8: THE TOPOLOGICAL MANIFOLD
# Uses Neighbor Loss to force the numbers into a coherent Integer Lattice.
# Includes Visualization to see the Geometry of Math.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
RANGE = 20 
VOCAB_SIZE = (RANGE * 2) + 1
OFFSET = RANGE

class TopologicalBrain(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # 2D Vectors
        self.manifold = nn.Embedding(vocab_size, 2)
        # Init small random noise
        nn.init.uniform_(self.manifold.weight, -0.1, 0.1)

    def forward_geometry(self, idx_a, idx_b, op):
        vec_a = self.manifold(idx_a)
        vec_b = self.manifold(idx_b)
        
        if op == 'add':
            return vec_a + vec_b
        elif op == 'mult':
            # Complex Mult
            real = (vec_a[:,0] * vec_b[:,0]) - (vec_a[:,1] * vec_b[:,1])
            imag = (vec_a[:,0] * vec_b[:,1]) + (vec_a[:,1] * vec_b[:,0])
            return torch.stack([real, imag], dim=1)

    def neighbor_loss(self):
        # TOPOLOGY CONSTRAINT
        # V(n) should be distance 1.0 away from V(n-1)
        # We grab the whole sequence -RANGE to +RANGE
        indices = torch.arange(VOCAB_SIZE - 1)
        next_indices = torch.arange(1, VOCAB_SIZE)
        
        vecs = self.manifold(indices)
        next_vecs = self.manifold(next_indices)
        
        # Calculate distance between n and n+1
        dists = torch.norm(vecs - next_vecs, dim=1)
        
        # We want that distance to be exactly 1.0 (The Unit Step)
        return torch.mean((dists - 1.0)**2)

# --- DATA GENERATION (Same as before) ---
def get_data(batch_size=64):
    add_batch, mult_batch = [], []
    for _ in range(batch_size):
        a = random.randint(-10, 10)
        b = random.randint(-10, 10)
        if abs(a+b) <= RANGE:
            add_batch.append([a+OFFSET, b+OFFSET, a+b+OFFSET])
        ma, mb = random.randint(-4, 4), random.randint(-4, 4)
        if abs(ma*mb) <= RANGE:
            mult_batch.append([ma+OFFSET, mb+OFFSET, ma*mb+OFFSET])
    return torch.tensor(add_batch), torch.tensor(mult_batch)

# --- TRAINING ---
def train_topology():
    print("--- AOTI LEVEL 8: TOPOLOGICAL MATH ---")
    brain = TopologicalBrain(VOCAB_SIZE)
    optimizer = optim.Adam(brain.parameters(), lr=0.01)
    
    for epoch in range(3001):
        optimizer.zero_grad()
        
        add_set, mult_set = get_data(128)
        
        # 1. Calculation Accuracy
        pred_add = brain.forward_geometry(add_set[:,0], add_set[:,1], 'add')
        loss_acc = torch.mean((pred_add - brain.manifold(add_set[:,2]))**2)
        
        pred_mult = brain.forward_geometry(mult_set[:,0], mult_set[:,1], 'mult')
        loss_acc += torch.mean((pred_mult - brain.manifold(mult_set[:,2]))**2)
        
        # 2. Neighbor Loss (The Chain Link)
        loss_topo = brain.neighbor_loss()
        
        # 3. Anchor (Lock 0 to Origin)
        # Note: We lock 0 to [0,0] and 1 to [1,0] to define the axis
        vec_zero = brain.manifold(torch.tensor(0 + OFFSET))
        vec_one = brain.manifold(torch.tensor(1 + OFFSET))
        loss_anchor = torch.sum(vec_zero**2) + torch.sum((vec_one - torch.tensor([1.0, 0.0]))**2)
        
        loss = loss_acc + loss_topo + loss_anchor
        
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Ep {epoch}: Acc {loss_acc:.4f} | Topo {loss_topo:.4f}")
            
    return brain

# --- ALGEBRA SOLVER ---
def solve_x(brain, op, target, a):
    x = torch.zeros(2, requires_grad=True)
    opt = optim.Adam([x], lr=0.1)
    vec_a = brain.manifold(torch.tensor(a + OFFSET)).detach()
    vec_target = brain.manifold(torch.tensor(target + OFFSET)).detach()
    
    for _ in range(100):
        opt.zero_grad()
        if op == 'add': pred = x + vec_a
        elif op == 'mult':
            real = (x[0]*vec_a[0]) - (x[1]*vec_a[1])
            imag = (x[0]*vec_a[1]) + (x[1]*vec_a[0])
            pred = torch.stack([real, imag])
        loss = torch.sum((pred - vec_target)**2)
        loss.backward()
        opt.step()
        
    dists = torch.norm(brain.manifold.weight.detach() - x, dim=1)
    return torch.argmin(dists).item() - OFFSET

def visualize_manifold(brain):
    # Dumps a scatter plot of the number line
    vecs = brain.manifold.weight.detach().numpy()
    labels = range(-RANGE, RANGE+1)
    
    print("\n[VISUALIZATION] Numbers should form a straight line.")
    print(f"Vec(0): {vecs[OFFSET]}")
    print(f"Vec(1): {vecs[OFFSET+1]}")
    print(f"Vec(2): {vecs[OFFSET+2]}")
    print(f"Vec(10): {vecs[OFFSET+10]}")
    
    # Simple ASCII Plot since we are in terminal
    print("\nASCII Map (X-Axis projection):")
    line = [" "] * 50
    for i in range(-5, 6): # Plot -5 to 5
        idx = i + OFFSET
        x_val = vecs[idx][0]
        # Map x_val (approx -5 to 5) to screen 0-50
        screen_x = int((x_val + 5) * 5)
        if 0 <= screen_x < 50:
            line[screen_x] = str(i) if i != 0 else "0"
    print("".join(line))

if __name__ == "__main__":
    ai = train_topology()
    visualize_manifold(ai)
    
    print("\n--- FINAL ALGEBRA TEST ---")
    print(f"x + 5 = 8   -> {solve_x(ai, 'add', 8, 5)} (Exp 3)")
    print(f"x + 10 = 5  -> {solve_x(ai, 'add', 5, 10)} (Exp -5)")
    print(f"x * 2 = 6   -> {solve_x(ai, 'mult', 6, 2)} (Exp 3)")
    print(f"x * -2 = 6  -> {solve_x(ai, 'mult', 6, -2)} (Exp -3)")