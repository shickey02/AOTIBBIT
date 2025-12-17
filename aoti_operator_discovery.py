#!/usr/bin/env python3
# AOTI: OPERATOR DISCOVERY (The Genesis of Logarithms)
# The AI must learn Multiplication by warping space itself.
# Constraint: The AI can ONLY add vectors.
# To solve "2 * 3 = 6", it must map '2', '3', '6' to a Logarithmic Space.

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# --- CONFIGURATION ---
VOCAB_SIZE = 100
TRAIN_SIZE = 10000

class DualManifold(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Manifold A: The "Additive" Universe
        self.space_linear = nn.Embedding(vocab_size, 1)
        
        # Manifold B: The "Multiplicative" Universe
        self.space_log = nn.Embedding(vocab_size, 1)
        
        # Initialize with small random noise
        nn.init.uniform_(self.space_linear.weight, -0.1, 0.1)
        nn.init.uniform_(self.space_log.weight, -0.1, 0.1)
        
    def forward(self, a_idx, b_idx, mode):
        if mode == 'add':
            # In Linear Space: Vec(A) + Vec(B) should match Vec(C)
            va = self.space_linear(a_idx)
            vb = self.space_linear(b_idx)
            return va + vb
        elif mode == 'mult':
            # In Log Space: Vec(A) + Vec(B) should match Vec(C)
            # (Because log(a) + log(b) = log(a*b))
            va = self.space_log(a_idx)
            vb = self.space_log(b_idx)
            return va + vb

    def get_pos(self, idx, mode):
        if mode == 'add': return self.space_linear(torch.tensor(idx)).item()
        if mode == 'mult': return self.space_log(torch.tensor(idx)).item()

def generate_data():
    add_data = []
    mult_data = []
    
    # Generate "Laws of Physics" observations
    for _ in range(TRAIN_SIZE):
        # Addition Facts
        a = random.randint(0, 49)
        b = random.randint(0, 49)
        c = a + b
        if c < VOCAB_SIZE:
            add_data.append([a, b, c])
            
        # Multiplication Facts
        # Avoid 0 for log space stability (Log(0) is undefined)
        ma = random.randint(1, 9)
        mb = random.randint(1, 9)
        mc = ma * mb
        if mc < VOCAB_SIZE:
            mult_data.append([ma, mb, mc])
            
    return torch.tensor(add_data), torch.tensor(mult_data)

def train_operators():
    print("--- AOTI LEVEL 2: OPERATOR DISCOVERY ---")
    print("Constraint: The Agent can ONLY perform Vector Addition.")
    print("Goal: Invent Logarithms to solve Multiplication.\n")
    
    model = DualManifold(VOCAB_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    add_set, mult_set = generate_data()
    
    for epoch in range(2001):
        optimizer.zero_grad()
        loss = 0
        
        # 1. Train Linear Space (a + b = c)
        pred_add = model(add_set[:,0], add_set[:,1], 'add')
        target_add = model.space_linear(add_set[:,2])
        loss += torch.mean((pred_add - target_add)**2)
        
        # 2. Train Log Space (a * b = c)
        pred_mult = model(add_set[:,0], add_set[:,1], 'mult') # Reuse indices just for batching? No.
        # Need correct mult data
        pred_mult = model(mult_set[:,0], mult_set[:,1], 'mult')
        target_mult = model.space_log(mult_set[:,2])
        loss += torch.mean((pred_mult - target_mult)**2)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Geometric Error {loss.item():.6f}")
            
    return model

def analyze_geometry(model):
    print("\n--- GEOMETRY ANALYSIS ---")
    print(f"{'SYM':<4} | {'LINEAR POS':<10} | {'LOG POS':<10} | {'LOG(SYM)':<10}")
    print("-" * 50)
    
    # Check symbols 1, 2, 4, 8, 16 (Powers of 2)
    # In Log Space, these should be evenly spaced (Linearly progressive)
    check_nums = [1, 2, 3, 4, 5, 8, 10, 16, 20, 32, 64]
    
    for n in check_nums:
        if n >= VOCAB_SIZE: continue
        lin_pos = model.get_pos(n, 'add')
        log_pos = model.get_pos(n, 'mult')
        true_log = np.log(n) if n > 0 else 0
        
        print(f"{n:<4} | {lin_pos:<10.4f} | {log_pos:<10.4f} | {true_log:<10.4f}")

    print("\n[INTERPRETATION]")
    print("If AOTI works:")
    print("1. Linear Pos should match N (scaled).")
    print("2. Log Pos should match Log(N).")
    print("   (e.g., Distance 1->2 should equal Distance 2->4, 4->8)")

if __name__ == "__main__":
    brain = train_operators()
    analyze_geometry(brain)