#!/usr/bin/env python3
# AOTI LEVEL 9: THE CRYSTAL LATTICE
# Enforces a strict order of operations:
# 1. Build the Geometry (Topology)
# 2. Learn the Physics (Math)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# --- CONFIGURATION ---
RANGE = 20
VOCAB_SIZE = (RANGE * 2) + 1
OFFSET = RANGE

class CrystalBrain(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.manifold = nn.Embedding(vocab_size, 2)
        # Init small random noise
        nn.init.uniform_(self.manifold.weight, -0.1, 0.1)

    def forward_val(self, idx):
        return self.manifold(idx)

    def calc_neighbor_loss(self):
        # Force distance between n and n+1 to be exactly 1.0
        indices = torch.arange(VOCAB_SIZE - 1)
        next_indices = torch.arange(1, VOCAB_SIZE)
        
        vecs = self.manifold(indices)
        next_vecs = self.manifold(next_indices)
        
        dists = torch.norm(vecs - next_vecs, dim=1)
        return torch.mean((dists - 1.0)**2)

    def calc_anchor_loss(self):
        # Lock 0 -> [0,0] and 1 -> [1,0]
        # This defines the "Prime Meridian" of our number universe
        v0 = self.manifold(torch.tensor(OFFSET))
        v1 = self.manifold(torch.tensor(OFFSET + 1))
        
        loss = torch.sum(v0**2) # Minimize distance to origin
        loss += torch.sum((v1 - torch.tensor([1.0, 0.0]))**2)
        return loss

    def calc_linearity_loss(self):
        # Force the line to be straight, not a circle.
        # Vector(2) should be 2*Vector(1)
        v1 = self.manifold(torch.tensor(OFFSET + 1))
        v2 = self.manifold(torch.tensor(OFFSET + 2))
        target = v1 * 2.0
        return torch.sum((v2 - target)**2)

# --- DATA ---
def get_math_data():
    add, mult = [], []
    for _ in range(100):
        a, b = random.randint(-10, 10), random.randint(-10, 10)
        if abs(a+b) <= RANGE: add.append([a+OFFSET, b+OFFSET, a+b+OFFSET])
        
        ma, mb = random.randint(-5, 5), random.randint(-5, 5)
        if abs(ma*mb) <= RANGE: mult.append([ma+OFFSET, mb+OFFSET, ma*mb+OFFSET])
    return torch.tensor(add), torch.tensor(mult)

# --- CURRICULUM ---
def train_crystal():
    brain = CrystalBrain(VOCAB_SIZE)
    # High LR for structure, lower for fine tuning
    opt = optim.Adam(brain.parameters(), lr=0.02)
    
    print("--- PHASE 1: CRYSTALLIZATION (Building the Number Line) ---")
    print("Goal: Arrange atoms -20 to 20 into a perfect grid.")
    
    for epoch in range(1001):
        opt.zero_grad()
        loss_topo = brain.calc_neighbor_loss() * 10.0 # High Priority
        loss_anchor = brain.calc_anchor_loss() * 10.0
        loss_straight = brain.calc_linearity_loss() * 5.0
        
        loss = loss_topo + loss_anchor + loss_straight
        loss.backward()
        opt.step()
        
        if epoch % 200 == 0:
            print(f"Ep {epoch}: Topology Error {loss_topo.item():.4f}")

    print("\n--- PHASE 2: ACTIVATING PHYSICS (Math) ---")
    print("Goal: Verify that this structure supports Math.")
    
    for epoch in range(2001):
        opt.zero_grad()
        
        # 1. Maintain Structure (Don't let math break the grid!)
        loss_struct = (brain.calc_neighbor_loss() + brain.calc_anchor_loss()) * 1.0
        
        # 2. Add Math Constraints
        add_data, mult_data = get_math_data()
        
        # Addition (Linear)
        va = brain.forward_val(add_data[:,0])
        vb = brain.forward_val(add_data[:,1])
        v_res = brain.forward_val(add_data[:,2])
        loss_add = torch.mean(((va + vb) - v_res)**2)
        
        # Multiplication (Complex/Rotation)
        va = brain.forward_val(mult_data[:,0])
        vb = brain.forward_val(mult_data[:,1])
        v_res = brain.forward_val(mult_data[:,2])
        
        # (a+bi)(c+di)
        real = (va[:,0]*vb[:,0]) - (va[:,1]*vb[:,1])
        imag = (va[:,0]*vb[:,1]) + (va[:,1]*vb[:,0])
        pred_mult = torch.stack([real, imag], dim=1)
        
        loss_mult = torch.mean((pred_mult - v_res)**2)
        
        loss = loss_struct + loss_add + loss_mult
        loss.backward()
        opt.step()
        
        if epoch % 500 == 0:
            print(f"Ep {epoch}: Add Err {loss_add.item():.4f} | Mult Err {loss_mult.item():.4f}")

    return brain

# --- DIAGNOSTICS ---
def verify(brain):
    print("\n[VISUALIZATION]")
    vecs = brain.manifold.weight.detach().numpy()
    
    # Check 0, 1, 2 alignment
    v0 = vecs[OFFSET]; v1 = vecs[OFFSET+1]; v2 = vecs[OFFSET+2]
    print(f"Vec(0): [{v0[0]:.2f}, {v0[1]:.2f}]")
    print(f"Vec(1): [{v1[0]:.2f}, {v1[1]:.2f}]")
    print(f"Vec(2): [{v2[0]:.2f}, {v2[1]:.2f}]")
    
    # Check Negativity
    v_neg1 = vecs[OFFSET-1]
    print(f"Vec(-1):[{v_neg1[0]:.2f}, {v_neg1[1]:.2f}] (Expect approx [-1.0, 0.0])")

    print("\n[ALGEBRA CHECK]")
    # Solve x + 5 = 8
    target = brain.forward_val(torch.tensor(8+OFFSET)).detach()
    param = brain.forward_val(torch.tensor(5+OFFSET)).detach()
    
    # Exact calculation using vector subtraction
    x_vec = target - param 
    
    # Find nearest
    dists = torch.norm(torch.tensor(vecs) - x_vec, dim=1)
    ans = torch.argmin(dists).item() - OFFSET
    print(f"x + 5 = 8  -> Solved: {ans} (Expect 3)")

    # Solve x * -2 = 10
    target = brain.forward_val(torch.tensor(10+OFFSET)).detach()
    param = brain.forward_val(torch.tensor(-2+OFFSET)).detach()
    
    # Complex Division: (a+bi)/(c+di) = [(ac+bd) + (bc-ad)i] / (c^2+d^2)
    # Or just optimize it quickly
    x = torch.zeros(2, requires_grad=True)
    opt = optim.Adam([x], lr=0.1)
    for _ in range(100):
        opt.zero_grad()
        real = (x[0]*param[0]) - (x[1]*param[1])
        imag = (x[0]*param[1]) + (x[1]*param[0])
        pred = torch.stack([real, imag])
        loss = torch.sum((pred - target)**2)
        loss.backward()
        opt.step()
        
    dists = torch.norm(torch.tensor(vecs) - x, dim=1)
    ans = torch.argmin(dists).item() - OFFSET
    print(f"x * -2 = 10 -> Solved: {ans} (Expect -5)")

if __name__ == "__main__":
    ai = train_crystal()
    verify(ai)