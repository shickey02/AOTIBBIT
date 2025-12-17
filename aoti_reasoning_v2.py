#!/usr/bin/env python3
# AOTI LEVEL 4: THE HARD-GATED REASONER
# Uses Temperature Scaling to force the AI to choose a SINGLE geometric manifold
# for each problem type, preventing "Concept Bleed."

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# --- CONFIGURATION ---
VOCAB_SIZE = 100
TEMPERATURE = 5.0 # High value forces sharp decisions (0.99 vs 0.01)

class GatedMultiverse(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Manifold A: Linear (Additive)
        self.linear = nn.Embedding(vocab_size, 1)
        # Manifold B: Logarithmic (Multiplicative)
        self.log = nn.Embedding(vocab_size, 1)
        
        # The Gatekeeper
        self.router = nn.Embedding(4, 2) 
        
        # Init small random weights
        nn.init.uniform_(self.linear.weight, -0.1, 0.1)
        nn.init.uniform_(self.log.weight, -0.1, 0.1)
        nn.init.zeros_(self.router.weight) # Start perfectly agnostic

    def forward(self, a_idx, b_idx, op_idx):
        # 1. Retrieve Coordinates
        lin_a, lin_b = self.linear(a_idx), self.linear(b_idx)
        log_a, log_b = self.log(a_idx), self.log(b_idx)
        
        # 2. Geometric Operations (Vector Sum in both spaces)
        res_lin = lin_a + lin_b
        res_log = log_a + log_b
        
        # 3. THE HARD GATE
        # We multiply logits by TEMPERATURE before softmax.
        # If logits are [0.6, 0.4] -> [3.0, 2.0] -> Softmax becomes sharper.
        logits = self.router(op_idx) * TEMPERATURE
        weights = torch.softmax(logits, dim=1) # [Batch, 2]
        
        return res_lin, res_log, weights

    def get_embedding(self, idx, manifold):
        if manifold == 'linear': return self.linear(torch.tensor(idx))
        return self.log(torch.tensor(idx))

def generate_physics_data(size=5000):
    data = []
    for _ in range(size):
        # ADD TASK (F_net = F1 + F2)
        f1, f2 = random.randint(0, 40), random.randint(0, 40)
        res = f1 + f2
        if res < VOCAB_SIZE:
            data.append({'a': f1, 'b': f2, 'op': 0, 'target': res})
            
        # MULT TASK (F = m * a)
        m, a = random.randint(1, 9), random.randint(1, 9)
        res = m * a
        if res < VOCAB_SIZE:
            data.append({'a': m, 'b': a, 'op': 2, 'target': res})
            
    return data

def train_reasoner():
    print("--- AOTI LEVEL 4: HARD-GATED PHYSICS ---")
    print(f"Temperature: {TEMPERATURE} (Forcing Binary Logic)\n")
    
    brain = GatedMultiverse(VOCAB_SIZE)
    optimizer = optim.Adam(brain.parameters(), lr=0.05) # Higher LR for crisp convergence
    
    dataset = generate_physics_data()
    
    for epoch in range(2001):
        optimizer.zero_grad()
        
        # Prepare Batch
        a_s = torch.tensor([d['a'] for d in dataset])
        b_s = torch.tensor([d['b'] for d in dataset])
        ops = torch.tensor([d['op'] for d in dataset])
        targets = torch.tensor([d['target'] for d in dataset])
        
        # Forward
        res_lin, res_log, weights = brain(a_s, b_s, ops)
        
        # Targets in both spaces
        target_lin = brain.linear(targets)
        target_log = brain.log(targets)
        
        # Competitive Loss
        # We assume the model wants to minimize the error of the *chosen* manifold
        error_lin = (res_lin - target_lin)**2
        error_log = (res_log - target_log)**2
        
        # Weighting
        losses = weights[:, 0].unsqueeze(1) * error_lin + weights[:, 1].unsqueeze(1) * error_log
        loss = torch.mean(losses)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Error {loss.item():.6f}")
            
    return brain

def verify_physics(brain):
    print("\n--- PHYSICS ENGINE DIAGNOSTICS ---")
    
    # Check ADDITION Gate
    logits_add = brain.router(torch.tensor([0])) * TEMPERATURE
    w_add = torch.softmax(logits_add, dim=1).detach().numpy()[0]
    print(f"OP: ADDITION       -> Linear Conf: {w_add[0]:.4f} | Log Conf: {w_add[1]:.4f}")
    
    # Check MULTIPLICATION Gate
    logits_mult = brain.router(torch.tensor([2])) * TEMPERATURE
    w_mult = torch.softmax(logits_mult, dim=1).detach().numpy()[0]
    print(f"OP: MULTIPLICATION -> Linear Conf: {w_mult[0]:.4f} | Log Conf: {w_mult[1]:.4f}")
    
    if w_add[0] > 0.9 and w_mult[1] > 0.9:
        print("\n>> SUCCESS: Brain has correctly segregated Physics Laws.")
    else:
        print("\n>> FAIL: Brain is confused.")
        return

    # Solve the Problem
    m = 5; a = 4
    print(f"\nSolving F = {m}kg * {a}m/s^2...")
    
    # 1. Manifold Hop: Router selects Log Space (Index 1)
    # 2. Geometric Addition in Log Space
    v_sum = brain.get_embedding(m, 'log') + brain.get_embedding(a, 'log')
    
    # 3. Decode
    all_logs = brain.log.weight.detach()
    dists = torch.norm(all_logs - v_sum, dim=1)
    pred = torch.argmin(dists).item()
    
    print(f"Geometric Calculation Result: {pred} Newtons")
    if pred == 20: 
        print(">> EXACT MATCH. First Principles Reasoning Achieved.")
    else:
        print(f">> Mismatch (Expected 20).")

if __name__ == "__main__":
    ai = train_reasoner()
    verify_physics(ai)