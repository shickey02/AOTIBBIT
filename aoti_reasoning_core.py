#!/usr/bin/env python3
# AOTI LEVEL 3: THE REASONING CORE
# Solves multi-step physics/math problems by hopping between geometric manifolds.
# Problem: "Calculate Force (F=ma) given mass and acceleration."
# The AI must learn to switch to Log-Space to solve this, without being told F=ma is multiplication.

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# --- 1. THE MULTIVERSE BRAIN ---
class MultiverseBrain(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Manifold A: Linear (for Addition/Subtraction)
        self.linear = nn.Embedding(vocab_size, 1)
        # Manifold B: Logarithmic (for Mult/Div/Power)
        self.log = nn.Embedding(vocab_size, 1)
        
        # The Router: Determines which manifold to use based on the "Operation Token"
        # 0: ADD, 1: SUB, 2: MULT, 3: DIV
        self.router = nn.Embedding(4, 2) # Weights for [Linear, Log]
        
        # Init small weights
        nn.init.uniform_(self.linear.weight, -0.1, 0.1)
        nn.init.uniform_(self.log.weight, -0.1, 0.1)
        nn.init.uniform_(self.router.weight, 0.4, 0.6) # Start unsure

    def forward(self, a_idx, b_idx, op_idx):
        # 1. Retrieve Geometric Coordinates from BOTH universes
        lin_a, lin_b = self.linear(a_idx), self.linear(b_idx)
        log_a, log_b = self.log(a_idx), self.log(b_idx)
        
        # 2. Perform the operation in BOTH universes
        # Addition/Subtraction applies to Linear Space
        # Multiplication/Division applies to Log Space (as add/sub)
        
        # We calculate candidates for all possible geometric moves
        # Candidate 1: Linear Add
        res_lin = lin_a + lin_b
        # Candidate 2: Log Add (Multiplication)
        res_log = log_a + log_b
        
        # 3. The Router Decision (The "Attention" Mechanism)
        # The router learns: "When I see op_idx=2 (MULT), I should listen to the Log Manifold."
        weights = torch.softmax(self.router(op_idx), dim=1) # [Batch, 2]
        
        # We predict the result vector by mixing the manifolds based on router weights
        # Note: We can't just mix the scalar results directly because 
        # linear(6) != log(6). We need to map the result back to symbol space.
        
        # SIMPLIFIED AOTI: 
        # The network outputs a "Truth Value" distance for the target.
        return res_lin, res_log, weights

    def get_embedding(self, idx, manifold='linear'):
        if manifold=='linear': return self.linear(torch.tensor(idx))
        return self.log(torch.tensor(idx))

# --- 2. THE CURRICULUM ---
def generate_physics_data(size=5000):
    # We teach it Newton's Second Law: F = m * a
    # But we treat it as generic operations.
    data = []
    
    for _ in range(size):
        # Type 0: Net Force (Addition) -> F_net = F1 + F2
        f1 = random.randint(0, 40)
        f2 = random.randint(0, 40)
        f_net = f1 + f2
        data.append({'a': f1, 'b': f2, 'op': 0, 'target': f_net, 'type': 'ADD'})
        
        # Type 2: Force Law (Multiplication) -> F = m * a
        m = random.randint(1, 9)
        a = random.randint(1, 9)
        f = m * a
        data.append({'a': m, 'b': a, 'op': 2, 'target': f, 'type': 'MULT'})
        
    return data

# --- 3. TRAINING THE REASONER ---
def train_reasoner():
    print("--- AOTI LEVEL 3: PHYSICS REASONER ---")
    print("Goal: Learn to switch geometric spaces to solve F=ma vs F_net=F1+F2.")
    
    brain = MultiverseBrain(100) # Vocab 0-99
    optimizer = optim.Adam(brain.parameters(), lr=0.02)
    
    dataset = generate_physics_data()
    
    for epoch in range(1501):
        total_loss = 0
        optimizer.zero_grad()
        
        # Batching (Manual for clarity)
        a_s = torch.tensor([d['a'] for d in dataset])
        b_s = torch.tensor([d['b'] for d in dataset])
        ops = torch.tensor([d['op'] for d in dataset])
        targets = torch.tensor([d['target'] for d in dataset])
        
        # Forward Pass
        res_lin, res_log, weights = brain(a_s, b_s, ops)
        
        # Target Embeddings
        # Crucial: The target '6' exists in both manifolds.
        # We need the router to pick the manifold where '6' is correctly constructed.
        target_lin = brain.linear(targets)
        target_log = brain.log(targets)
        
        # Loss Calculation (Competition)
        # If op is ADD (0), we want res_lin to match target_lin
        # If op is MULT (2), we want res_log to match target_log
        
        # We let the router decide which error matters!
        # Weighted MSE Error
        error_lin = (res_lin - target_lin)**2
        error_log = (res_log - target_log)**2
        
        # Mix errors based on router confidence
        # combined_error = weights[0] * error_lin + weights[1] * error_log
        # This allows the router to learn "Ignore Linear Error when doing Mult"
        
        losses = weights[:, 0].unsqueeze(1) * error_lin + weights[:, 1].unsqueeze(1) * error_log
        loss = torch.mean(losses)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Error {loss.item():.6f}")
            
    return brain

# --- 4. VERIFICATION ---
def verify_physics(brain):
    print("\n--- PHYSICS ENGINE TEST ---")
    
    # Test ADDITION (Net Force)
    op_add = torch.tensor([0]) # 0 = ADD
    w_add = torch.softmax(brain.router(op_add), dim=1).detach().numpy()[0]
    print(f"Operation 'ADD' Router Weights -> Linear: {w_add[0]:.2f} | Log: {w_add[1]:.2f}")
    if w_add[0] > 0.9: print(">> CORRECT: Uses Linear Space for Sums.")
    else: print(">> FAILED.")

    # Test MULTIPLICATION (F=ma)
    op_mult = torch.tensor([2]) # 2 = MULT
    w_mult = torch.softmax(brain.router(op_mult), dim=1).detach().numpy()[0]
    print(f"Operation 'MULT' Router Weights -> Linear: {w_mult[0]:.2f} | Log: {w_mult[1]:.2f}")
    if w_mult[1] > 0.9: print(">> CORRECT: Uses Log Space for Physics.")
    else: print(">> FAILED.")
    
    # Solve a specific problem
    m = 5; a = 4
    # We simulate the full path
    # 1. Router checks Op
    # 2. Router selects Log Space
    # 3. Brain retrieves Log(5) + Log(4)
    # 4. Result is Log(20) -> Decodes to '20'
    print(f"\nSolving F = {m}kg * {a}m/s^2...")
    
    # Get vector sum in log space
    v_sum = brain.get_embedding(m, 'log') + brain.get_embedding(a, 'log')
    
    # Decode: Find closest symbol in Log Manifold
    all_logs = brain.log.weight.detach()
    dists = torch.norm(all_logs - v_sum, dim=1)
    pred = torch.argmin(dists).item()
    
    print(f"Brain predicted: {pred} Newtons")
    if pred == 20: print(">> PHYSICS SIMULATION ACCURATE.")

if __name__ == "__main__":
    ai = train_reasoner()
    verify_physics(ai)  