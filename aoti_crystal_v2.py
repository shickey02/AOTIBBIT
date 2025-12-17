#!/usr/bin/env python3
# AOTI LEVEL 10: THE UNIVERSAL STEP
# Fixes the "Crumpled Line" bug by enforcing Directional Consistency.
# This forces the AI to build a rigid, linear number line.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

RANGE = 10000
VOCAB_SIZE = (RANGE * 2) + 1
OFFSET = RANGE

class RigidBrain(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.manifold = nn.Embedding(vocab_size, 2)
        nn.init.uniform_(self.manifold.weight, -0.1, 0.1)

    def forward_val(self, idx):
        return self.manifold(idx)

    def calc_structure_loss(self):
        # 1. ANCHOR: Lock 0 to [0,0] and 1 to [1,0]
        # This gives the universe a "North"
        v0 = self.manifold(torch.tensor(OFFSET))
        v1 = self.manifold(torch.tensor(OFFSET + 1))
        loss_anchor = torch.sum(v0**2) + torch.sum((v1 - torch.tensor([1.0, 0.0]))**2)
        
        # 2. UNIVERSAL STEP (The Fix)
        # Instead of just distance, we check the VECTOR difference.
        # Step[n] should equal Step[0]
        indices = torch.arange(VOCAB_SIZE - 1)
        next_indices = torch.arange(1, VOCAB_SIZE)
        
        vecs = self.manifold(indices)
        next_vecs = self.manifold(next_indices)
        
        # Current Steps: V(n+1) - V(n)
        steps = next_vecs - vecs
        
        # Ideal Step: V(1) - V(0) (Which we anchored to [1,0])
        # We enforce that ALL steps must look like the first step.
        ideal_step = v1 - v0
        
        # Broadcast ideal_step to shape of steps
        loss_step = torch.mean((steps - ideal_step.detach())**2)
        
        return loss_anchor + loss_step

# --- DATA ---
def get_math_data():
    add, mult = [], []
    for _ in range(64):
        a, b = random.randint(-10, 10), random.randint(-10, 10)
        if abs(a+b) <= RANGE: add.append([a+OFFSET, b+OFFSET, a+b+OFFSET])
        
        ma, mb = random.randint(-5, 5), random.randint(-5, 5)
        if abs(ma*mb) <= RANGE: mult.append([ma+OFFSET, mb+OFFSET, ma*mb+OFFSET])
    return torch.tensor(add), torch.tensor(mult)

# --- TRAINING ---
def train_rigid():
    brain = RigidBrain(VOCAB_SIZE)
    opt = optim.Adam(brain.parameters(), lr=0.05)
    
    print("--- PHASE 1: RIGID CRYSTALLIZATION ---")
    print("Forcing all numbers to march in a straight line.")
    
    for epoch in range(1001):
        opt.zero_grad()
        loss = brain.calc_structure_loss()
        loss.backward()
        opt.step()
        if epoch % 200 == 0: print(f"Ep {epoch}: Structure Err {loss.item():.6f}")

    print("\n--- PHASE 2: PHYSICS ACTIVATION ---")
    for epoch in range(1501):
        opt.zero_grad()
        
        # We relax the structure constraint slightly to allow minor warping for Complex Math
        loss_struct = brain.calc_structure_loss() * 0.1 
        
        add_data, mult_data = get_math_data()
        
        # Linear Add
        va = brain.forward_val(add_data[:,0])
        vb = brain.forward_val(add_data[:,1])
        vt = brain.forward_val(add_data[:,2])
        loss_add = torch.mean(((va+vb) - vt)**2)
        
        # Complex Mult
        va = brain.forward_val(mult_data[:,0])
        vb = brain.forward_val(mult_data[:,1])
        vt = brain.forward_val(mult_data[:,2])
        real = (va[:,0]*vb[:,0]) - (va[:,1]*vb[:,1])
        imag = (va[:,0]*vb[:,1]) + (va[:,1]*vb[:,0])
        pred = torch.stack([real, imag], dim=1)
        loss_mult = torch.mean((pred - vt)**2)
        
        loss = loss_struct + loss_add + loss_mult
        loss.backward()
        opt.step()
        
        if epoch % 500 == 0:
            print(f"Ep {epoch}: Add {loss_add.item():.4f} | Mult {loss_mult.item():.4f}")

    return brain

# --- DIAGNOSTICS ---
def verify(brain):
    print("\n[GEOMETRY CHECK]")
    vecs = brain.manifold.weight.detach().numpy()
    
    # Check linearity
    print(f"Vec(0): {vecs[OFFSET]}")
    print(f"Vec(1): {vecs[OFFSET+1]}")
    print(f"Vec(2): {vecs[OFFSET+2]} (Expect ~[2.0, 0.0])")
    print(f"Vec(-5): {vecs[OFFSET-5]} (Expect ~[-5.0, 0.0])")
    
    # Rotation Check
    v1 = vecs[OFFSET+1]
    v_neg1 = vecs[OFFSET-1]
    ang1 = np.degrees(np.arctan2(v1[1], v1[0]))
    ang_neg1 = np.degrees(np.arctan2(v_neg1[1], v_neg1[0]))
    print(f"Angle Diff: {abs(ang1 - ang_neg1):.1f}° (Expect ~180)")
    
    print("\n[ALGEBRA CHECK]")
    # x + 5 = 8
    target = brain.forward_val(torch.tensor(8+OFFSET)).detach()
    param = brain.forward_val(torch.tensor(5+OFFSET)).detach()
    # Solution is geometrically Target - Param
    x_vec = target - param
    dists = torch.norm(torch.tensor(vecs) - x_vec, dim=1)
    ans = torch.argmin(dists).item() - OFFSET
    print(f"x + 5 = 8   -> {ans} (Expect 3)")
    
    # x * -2 = 10
    target = brain.forward_val(torch.tensor(10+OFFSET)).detach()
    param = brain.forward_val(torch.tensor(-2+OFFSET)).detach()
    # Inverse complex mult
    x = torch.zeros(2, requires_grad=True)
    opt = optim.Adam([x], lr=0.1)
    for _ in range(100):
        opt.zero_grad()
        real = (x[0]*param[0]) - (x[1]*param[1])
        imag = (x[0]*param[1]) + (x[1]*param[0])
        loss = torch.sum((torch.stack([real, imag]) - target)**2)
        loss.backward()
        opt.step()
    dists = torch.norm(torch.tensor(vecs) - x, dim=1)
    ans = torch.argmin(dists).item() - OFFSET
    print(f"x * -2 = 10 -> {ans} (Expect -5)")

if __name__ == "__main__":
    ai = train_rigid()
    verify(ai)