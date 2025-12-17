#!/usr/bin/env python3
# AOTI LEVEL 5: CURRICULUM REASONING
# Corrects the "Lazy Router" problem by pre-training the geometric manifolds
# before teaching the router how to switch.

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# --- CONFIGURATION ---
VOCAB_SIZE = 100
TEMPERATURE = 5.0

class ModularBrain(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # The Distinct Cortices
        self.cortex_linear = nn.Embedding(vocab_size, 1) # For Add/Sub
        self.cortex_log = nn.Embedding(vocab_size, 1)    # For Mult/Div
        
        # The Executive Router
        self.router = nn.Embedding(4, 2) 
        
        # Init
        nn.init.uniform_(self.cortex_linear.weight, -0.1, 0.1)
        nn.init.uniform_(self.cortex_log.weight, -0.1, 0.1)
        nn.init.zeros_(self.router.weight)

    def forward(self, a_idx, b_idx, op_idx):
        # 1. Calculate outcomes in both cortices
        lin_res = self.cortex_linear(a_idx) + self.cortex_linear(b_idx)
        log_res = self.cortex_log(a_idx) + self.cortex_log(b_idx)
        
        # 2. Route
        logits = self.router(op_idx) * TEMPERATURE
        weights = torch.softmax(logits, dim=1)
        
        return lin_res, log_res, weights

    def get_vec(self, idx, mode):
        if mode == 'linear': return self.cortex_linear(torch.tensor(idx))
        return self.cortex_log(torch.tensor(idx))

# --- DATA GENERATION ---
def get_add_data():
    data = []
    for _ in range(2000):
        a, b = random.randint(0, 40), random.randint(0, 40)
        if a+b < VOCAB_SIZE: data.append([a, b, a+b])
    return torch.tensor(data)

def get_mult_data():
    data = []
    for _ in range(2000):
        a, b = random.randint(1, 9), random.randint(1, 9)
        if a*b < VOCAB_SIZE: data.append([a, b, a*b])
    return torch.tensor(data)

# --- TRAINING PHASES ---
def train_curriculum():
    brain = ModularBrain(VOCAB_SIZE)
    
    # --- PHASE 1: PRE-TRAINING LINEAR CORTEX ---
    print("--- PHASE 1: Training Linear Cortex (Addition) ---")
    opt_lin = optim.Adam(brain.cortex_linear.parameters(), lr=0.05)
    add_data = get_add_data()
    
    for epoch in range(501):
        opt_lin.zero_grad()
        # Only use Linear Cortex
        pred = brain.cortex_linear(add_data[:,0]) + brain.cortex_linear(add_data[:,1])
        target = brain.cortex_linear(add_data[:,2])
        loss = torch.mean((pred - target)**2)
        loss.backward()
        opt_lin.step()
        if epoch % 250 == 0: print(f"  Ep {epoch}: Add Error {loss.item():.6f}")

    # --- PHASE 2: PRE-TRAINING LOG CORTEX ---
    print("\n--- PHASE 2: Training Log Cortex (Multiplication) ---")
    opt_log = optim.Adam(brain.cortex_log.parameters(), lr=0.05)
    mult_data = get_mult_data()
    
    for epoch in range(1001): # Needs more time to find Log rhythm
        opt_log.zero_grad()
        # Only use Log Cortex
        pred = brain.cortex_log(mult_data[:,0]) + brain.cortex_log(mult_data[:,1])
        target = brain.cortex_log(mult_data[:,2])
        loss = torch.mean((pred - target)**2)
        loss.backward()
        opt_log.step()
        if epoch % 500 == 0: print(f"  Ep {epoch}: Mult Error {loss.item():.6f}")

    # --- PHASE 3: TRAINING THE ROUTER ---
    print("\n--- PHASE 3: Training Executive Router ---")
    # Freeze the Cortices!
    brain.cortex_linear.weight.requires_grad = False
    brain.cortex_log.weight.requires_grad = False
    
    opt_router = optim.Adam(brain.router.parameters(), lr=0.1)
    
    # Mixed Data
    d_add = [{'a':d[0], 'b':d[1], 'op':0, 't':d[2]} for d in add_data]
    d_mult = [{'a':d[0], 'b':d[1], 'op':2, 't':d[2]} for d in mult_data]
    dataset = d_add + d_mult
    random.shuffle(dataset)
    
    for epoch in range(201):
        opt_router.zero_grad()
        
        # Batch prep
        a_s = torch.tensor([d['a'] for d in dataset])
        b_s = torch.tensor([d['b'] for d in dataset])
        ops = torch.tensor([d['op'] for d in dataset])
        targets = torch.tensor([d['t'] for d in dataset])
        
        # Forward (Manifolds are frozen, only Router updates)
        res_lin, res_log, weights = brain(a_s, b_s, ops)
        
        target_lin = brain.cortex_linear(targets)
        target_log = brain.cortex_log(targets)
        
        # Loss: Which manifold gets closer to the answer?
        err_lin = (res_lin - target_lin)**2
        err_log = (res_log - target_log)**2
        
        # Router must minimize the *chosen* error
        losses = weights[:,0].unsqueeze(1)*err_lin + weights[:,1].unsqueeze(1)*err_log
        loss = torch.mean(losses)
        
        loss.backward()
        opt_router.step()
        
        if epoch % 100 == 0: print(f"  Ep {epoch}: Routing Error {loss.item():.6f}")

    return brain

def verify(brain):
    print("\n--- DIAGNOSTICS ---")
    
    # CHECK ROUTER
    logits_add = brain.router(torch.tensor([0])) * TEMPERATURE
    w_add = torch.softmax(logits_add, dim=1).detach().numpy()[0]
    print(f"ADDITION ROUTE       -> Linear: {w_add[0]:.4f} | Log: {w_add[1]:.4f}")
    
    logits_mult = brain.router(torch.tensor([2])) * TEMPERATURE
    w_mult = torch.softmax(logits_mult, dim=1).detach().numpy()[0]
    print(f"MULTIPLICATION ROUTE -> Linear: {w_mult[0]:.4f} | Log: {w_mult[1]:.4f}")
    
    # SOLVE F=ma
    m=5; a=4
    print(f"\nSolving F = {m} * {a}...")
    
    # 1. Select Manifold
    if w_mult[1] > 0.5:
        print(">> Router selected LOG Cortex.")
        # 2. Perform Geometric Addition in Log Space
        vec_res = brain.get_vec(m, 'log') + brain.get_vec(a, 'log')
        # 3. Decode
        all_vecs = brain.cortex_log.weight.detach()
        dists = torch.norm(all_vecs - vec_res, dim=1)
        pred = torch.argmin(dists).item()
        print(f">> Result: {pred}")
        if pred == 20: print(">> SUCCESS.")
    else:
        print(">> Router failed (Selected Linear).")

if __name__ == "__main__":
    ai = train_curriculum()
    verify(ai)