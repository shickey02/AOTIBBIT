#!/usr/bin/env python3
# AOTI LEVEL 6: THE COMPLEX MANIFOLD
# Learns Signed Math (-5 * -5 = 25) and Basic Algebra (2x = 10)
# by modeling numbers as 2D Vectors (Magnitude + Phase).

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# --- CONFIGURATION ---
# We represent integers -50 to +50
RANGE = 50 
VOCAB_SIZE = (RANGE * 2) + 1  # 101 tokens
OFFSET = RANGE # Shift so -50 is index 0, 0 is index 50, +50 is index 100

class ComplexBrain(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Instead of 1D, we use 2D vectors [Real, Imaginary]
        # This naturally handles Magnitude and Direction (Sign)
        self.manifold = nn.Embedding(vocab_size, 2)
        
        # Initialize with small random vectors
        nn.init.uniform_(self.manifold.weight, -0.1, 0.1)

    def get_vector(self, val):
        idx = val + OFFSET
        if idx < 0 or idx >= VOCAB_SIZE: return None # Out of bounds
        return self.manifold(torch.tensor(idx))

    def geometry_add(self, vec_a, vec_b):
        # Addition in Linear Space is just Vector Addition
        return vec_a + vec_b

    def geometry_mult(self, vec_a, vec_b):
        # Multiplication in Complex Space:
        # 1. Multiply Magnitudes (Add Logs)
        # 2. Add Angles (Rotation)
        
        # To make this differentiable and discoverable, we simulate 
        # complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        real_a, imag_a = vec_a[..., 0], vec_a[..., 1]
        real_b, imag_b = vec_b[..., 0], vec_b[..., 1]
        
        real_res = (real_a * real_b) - (imag_a * imag_b)
        imag_res = (real_a * imag_b) + (imag_a * real_b)
        
        return torch.stack([real_res, imag_res], dim=-1)

    def forward(self, a_idx, b_idx, op_mode):
        vec_a = self.manifold(a_idx)
        vec_b = self.manifold(b_idx)
        
        if op_mode == 'add':
            return self.geometry_add(vec_a, vec_b)
        elif op_mode == 'mult':
            return self.geometry_mult(vec_a, vec_b)

# --- DATA GENERATION ---
def generate_signed_data(count=10000):
    add_data = []
    mult_data = []
    
    for _ in range(count):
        # Random integers between -10 and 10 (keep it simple for fast convergence)
        a = random.randint(-10, 10)
        b = random.randint(-10, 10)
        
        # ADDITION
        res = a + b
        if -RANGE <= res <= RANGE:
            add_data.append([a+OFFSET, b+OFFSET, res+OFFSET])
            
        # MULTIPLICATION
        res = a * b
        if -RANGE <= res <= RANGE:
            mult_data.append([a+OFFSET, b+OFFSET, res+OFFSET])
            
    return torch.tensor(add_data), torch.tensor(mult_data)

# --- TRAINING ---
def train_complex_math():
    print("--- AOTI LEVEL 6: THE COMPLEX MANIFOLD ---")
    print(f"Learning Signed Arithmetic in 2D Space (-{RANGE} to +{RANGE})")
    
    brain = ComplexBrain(VOCAB_SIZE)
    optimizer = optim.Adam(brain.parameters(), lr=0.01)
    
    add_set, mult_set = generate_signed_data()
    
    for epoch in range(3001):
        optimizer.zero_grad()
        loss = 0
        
        # 1. Train Addition (Linear Geometry)
        pred_add = brain(add_set[:,0], add_set[:,1], 'add')
        target_add = brain.manifold(add_set[:,2])
        loss += torch.mean((pred_add - target_add)**2)
        
        # 2. Train Multiplication (Rotational Geometry)
        pred_mult = brain(mult_set[:,0], mult_set[:,1], 'mult')
        target_mult = brain.manifold(mult_set[:,2])
        loss += torch.mean((pred_mult - target_mult)**2)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Geometric Error {loss.item():.6f}")
            
    return brain

# --- ALGEBRA SOLVER (Solving for X) ---
def solve_algebra(brain, equation_type, target_val, param_a):
    """
    Solves equations like "x + a = target" or "x * a = target"
    by creating a 'Ghost Vector' (x) and optimizing it.
    """
    print(f"\n--- ALGEBRAIC SOLVER: Solving {equation_type} ---")
    
    # 1. Create 'x' as a learnable vector (The Unknown)
    # Start it at the origin (0,0)
    x_vec = torch.tensor([0.0, 0.0], requires_grad=True)
    
    # 2. Get fixed vectors (Constants)
    # Note: We detach them so we don't accidentally retrain our brain
    a_vec = brain.get_vector(param_a).detach()
    target_vec = brain.get_vector(target_val).detach()
    
    # 3. Optimize 'x' until the gap closes
    # This is "System 2 Reasoning" at runtime
    solver_optim = optim.Adam([x_vec], lr=0.1)
    
    for step in range(100):
        solver_optim.zero_grad()
        
        # Reconstruct the equation geometrically
        if equation_type == 'add': # x + a = target
            prediction = brain.geometry_add(x_vec, a_vec)
        elif equation_type == 'mult': # x * a = target
            prediction = brain.geometry_mult(x_vec, a_vec)
            
        # The Gap
        loss = torch.sum((prediction - target_vec)**2)
        
        loss.backward()
        solver_optim.step()
        
        if loss < 0.001:
            break
            
    # 4. Decode 'x'
    # Find which symbol lies closest to our optimised x_vec
    all_vecs = brain.manifold.weight.detach()
    dists = torch.norm(all_vecs - x_vec, dim=1)
    best_idx = torch.argmin(dists).item()
    val = best_idx - OFFSET
    
    print(f"Equation: x {'+' if equation_type=='add' else '*'} {param_a} = {target_val}")
    print(f"Solved x: {val}")
    
    # Verify correctness
    expected = (target_val - param_a) if equation_type == 'add' else (target_val // param_a)
    if val == expected: print(">> CORRECT.")
    else: print(f">> FAILED (Expected {expected}).")

def verify_geometry(brain):
    print("\n--- GEOMETRIC ANALYSIS ---")
    # Check if the brain discovered that Negative = 180 degree rotation
    
    pos_one = brain.get_vector(1).detach().numpy()
    neg_one = brain.get_vector(-1).detach().numpy()
    
    # Angle calculation
    angle_pos = np.arctan2(pos_one[1], pos_one[0])
    angle_neg = np.arctan2(neg_one[1], neg_one[0])
    
    diff_deg = np.degrees(abs(angle_pos - angle_neg))
    # Normalize to 0-360
    if diff_deg > 180: diff_deg = 360 - diff_deg
        
    print(f"Vector(+1): {pos_one}")
    print(f"Vector(-1): {neg_one}")
    print(f"Rotational Difference: {diff_deg:.2f} degrees")
    print("(Ideal is ~180.00 degrees, proving it learned Negativity is Rotation)")

if __name__ == "__main__":
    brain = train_complex_math()
    verify_geometry(brain)
    
    # Test 1: Simple Algebra (x + 5 = 8)
    solve_algebra(brain, 'add', 8, 5)
    
    # Test 2: Negative Algebra (x + 10 = 5) -> x should be -5
    solve_algebra(brain, 'add', 5, 10)
    
    # Test 3: Division via Multiplication (x * 4 = 12) -> x should be 3
    solve_algebra(brain, 'mult', 12, 4)
    
    # Test 4: Signed Multiplication (x * -2 = 10) -> x should be -5
    solve_algebra(brain, 'mult', 10, -2)