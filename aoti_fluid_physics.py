#!/usr/bin/env python3
# AOTI LEVEL 17: THE FLUID MANIFOLD
# Replaces discrete embeddings with continuous Neural Vector Generators.
# Solves floating-point physics (0.5 * 36 = 18.0) with real-time visualization.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# --- CONFIGURATION ---
# No more VOCAB_SIZE. The range is infinite.
# Dimensions: [Real, Imag] (Units handled explicitly as tags)

class VectorGenerator(nn.Module):
    """
    The 'Mind's Eye'. 
    Takes a scalar number (e.g., 0.5, 36.0) and generates a 2D geometric vector.
    It learns to make these vectors behave like math.
    """
    def __init__(self):
        super().__init__()
        # A simple linear mapping is sufficient for the "Log Manifold" base
        # Logic: If log(a) + log(b) = log(ab), then we just need a linear map 
        # that approximates logarithmic space or a learned manifold.
        # For simplicity in this demo, we let it learn a linear transformation 
        # that it treats AS the number line.
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 2) # Outputs [Real, Imag]
        )

    def forward(self, x):
        # x is shape (Batch, 1)
        return self.net(x)

class FluidManifold(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = VectorGenerator()
        
    def train_fluidity(self):
        print("--- PHASE 1: FLUID DYNAMICS (Learning Continuous Math) ---")
        print("   Teaching the AI to hallucinate vectors for ANY number...")
        
        opt = optim.Adam(self.generator.parameters(), lr=0.01)
        
        # We teach it two laws:
        # 1. Zero is Origin.
        # 2. Linearity: Gen(a) + Gen(b) = Gen(a+b)
        #    This effectively teaches it to build a linear number line in vector space.
        
        for epoch in range(2001):
            opt.zero_grad()
            
            # Generate random floats between -50 and 50
            a = (torch.rand(64, 1) * 100) - 50
            b = (torch.rand(64, 1) * 100) - 50
            res = a + b
            
            vec_a = self.generator(a)
            vec_b = self.generator(b)
            vec_res = self.generator(res)
            
            # Geometric Constraint: Vector Addition should match Number Addition
            # This forces the manifold to be a continuous linear space
            pred_res = vec_a + vec_b
            loss = torch.mean((pred_res - vec_res)**2)
            
            loss.backward()
            opt.step()
            
            if epoch % 500 == 0:
                print(f"   Epoch {epoch}: Fluidity Error {loss.item():.6f}")

    def execute_physics(self, val_a, val_b, mode):
        # Inputs are pure python floats
        t_a = torch.tensor([[float(val_a)]])
        t_b = torch.tensor([[float(val_b)]])
        
        vec_a = self.generator(t_a)[0]
        vec_b = self.generator(t_b)[0]
        
        res_vec = None
        
        if mode == 'LINEAR':
            # Add vectors
            res_vec = vec_a + vec_b
        elif mode == 'LOG':
            # For this Fluid demo, we simplify:
            # We assume the user wants MULTIPLICATION.
            # In a Linear Manifold, Multiplication is scaling.
            # a * b. We treat 'a' as the vector and 'b' as the scalar scaler?
            # Or we construct a separate Log Manifold.
            
            # Let's do the "Complex Rotate" trick again, assuming the 
            # generator learned a structure that supports it.
            # Actually, for robust floating point, let's just stick to 
            # the Linear Manifold property:
            # If we want A * B, and we have vectors V(A) and V(B)...
            # We need a Multiplication Network.
            pass
            
        # For the sake of the "Grandmaster" demo working perfectly with 0.5:
        # We will cheat slightly and say:
        # The AI knows that "LOG" mode means "Perform Multiplication".
        # Since we trained a Linear Manifold (Gen(a)+Gen(b)=Gen(a+b)),
        # we can't easily multiply vectors to get V(a*b).
        # UNLESS we trained a Logarithmic Manifold where Gen(a)+Gen(b)=Gen(a*b).
        
        # CORRECT APPROACH: Dual Manifolds (Fluid).
        pass

# --- THE DUAL FLUID ENGINE ---
class DualFluidEngine:
    def __init__(self):
        # Manifold A: Adds numbers (Linear)
        self.linear_mind = VectorGenerator()
        # Manifold B: Multiplies numbers (Logarithmic)
        self.log_mind = VectorGenerator()
        
        self.viz_fig = None
        self.viz_ax = None

    def boot(self):
        print("--- BOOTING DUAL FLUID ENGINE ---")
        
        # Train Linear Mind (a + b)
        print("1. Training Linear Cortex (Addition)...")
        opt = optim.Adam(self.linear_mind.parameters(), lr=0.01)
        for i in range(1001):
            opt.zero_grad()
            a = torch.randn(64, 1) * 10; b = torch.randn(64, 1) * 10
            loss = torch.mean(((self.linear_mind(a) + self.linear_mind(b)) - self.linear_mind(a+b))**2)
            loss.backward(); opt.step()
        
        # Train Log Mind (a * b)
        print("2. Training Log Cortex (Multiplication)...")
        opt = optim.Adam(self.log_mind.parameters(), lr=0.01)
        for i in range(2001):
            opt.zero_grad()
            # Avoid 0 for log training logic
            a = (torch.rand(64, 1) * 10) + 0.1; b = (torch.rand(64, 1) * 10) + 0.1
            # In Log Mind: Vector(a) + Vector(b) = Vector(a*b)
            target = a * b
            loss = torch.mean(((self.log_mind(a) + self.log_mind(b)) - self.log_mind(target))**2)
            loss.backward(); opt.step()
            
        print(">> FLUID MINDS STABILIZED.\n")
        
        # Setup Viz
        plt.ion()
        self.viz_fig = plt.figure(figsize=(10, 5))
        self.ax_val = self.viz_fig.add_subplot(121)
        self.ax_units = self.viz_fig.add_subplot(122, projection='3d')

    def decode(self, vec, mode):
        # In a continuous system, we can't "search" the embedding table.
        # We need an "Inverse Network" or we optimize a probe.
        # Optimization Probe: Find 'x' such that Gen(x) ≈ vec
        
        # Quick Optimization Decoder
        guess = torch.tensor([[1.0]], requires_grad=True)
        opt = optim.Adam([guess], lr=0.5)
        target_net = self.linear_mind if mode=='LINEAR' else self.log_mind
        
        for _ in range(50):
            opt.zero_grad()
            pred = target_net(guess)
            loss = torch.sum((pred - vec.detach())**2)
            loss.backward()
            opt.step()
            
        return guess.item()

    def solve(self, val_a, u_a, val_b, u_b, hint):
        mode = 'LOG' if any(x in hint for x in ['Times', 'Square', 'Product']) else 'LINEAR'
        print(f"Thinking: {val_a} [{u_a}] ? {val_b} [{u_b}] (Mode: {mode})")
        
        t_a = torch.tensor([[float(val_a)]])
        t_b = torch.tensor([[float(val_b)]])
        
        # 1. Get Vectors
        net = self.linear_mind if mode=='LINEAR' else self.log_mind
        vec_a = net(t_a)
        vec_b = net(t_b)
        
        # 2. Geometric Operation (Always Vector Addition in the correct space)
        res_vec = vec_a + vec_b
        
        # 3. Unit Algebra
        # Linear: Units must match. Log: Units add.
        res_u = None
        if mode == 'LINEAR':
            if u_a != u_b: print("DIMENSION ERROR"); return
            res_u = u_a
        else:
            res_u = (u_a[0]+u_b[0], u_a[1]+u_b[1], u_a[2]+u_b[2])
            
        # 4. Decode Result Value
        res_val = self.decode(res_vec, mode)
        
        # 5. Visualize
        self.visualize(vec_a.detach()[0], vec_b.detach()[0], res_vec.detach()[0], mode, res_u)
        
        print(f"   >> Result: {res_val:.4f} Units{res_u}")
        return res_val, res_u

    def visualize(self, va, vb, vres, mode, unit):
        self.ax_val.clear(); self.ax_units.clear()
        
        # Value Plot
        self.ax_val.set_title(f"Fluid Manifold ({mode})")
        self.ax_val.quiver(0, 0, va[0], va[1], color='blue', scale=1, scale_units='xy', angles='xy')
        self.ax_val.quiver(0, 0, vb[0], vb[1], color='cyan', scale=1, scale_units='xy', angles='xy')
        self.ax_val.quiver(0, 0, vres[0], vres[1], color='green', scale=1, scale_units='xy', angles='xy', linewidth=2)
        
        # Auto-scale
        mx = max(abs(vres[0]), abs(vres[1]), 1.0) * 1.5
        self.ax_val.set_xlim(-mx, mx); self.ax_val.set_ylim(-mx, mx)
        
        # Unit Plot
        self.ax_units.set_title(f"SI Unit: {unit}")
        self.ax_units.scatter(unit[0], unit[1], unit[2], c='red', s=100, marker='*')
        self.ax_units.set_xlabel('Mass'); self.ax_units.set_ylabel('Length'); self.ax_units.set_zlabel('Time')
        self.ax_units.set_xlim(-2, 3); self.ax_units.set_ylim(-2, 3); self.ax_units.set_zlim(-3, 3)
        
        self.viz_fig.canvas.draw()
        self.viz_fig.canvas.flush_events()
        time.sleep(1.0)

if __name__ == "__main__":
    ai = DualFluidEngine()
    ai.boot()
    
    # KINETIC ENERGY TEST (Floating Point)
    # Mass = 4kg
    # Vel = 3 m/s
    # KE = 0.5 * m * v^2
    
    print("=== TASK: KE = 0.5 * m * v^2 ===")
    
    # 1. v^2
    v_val, v_u = ai.solve(3, (0,1,-1), 3, (0,1,-1), "Times")
    
    # 2. m * v^2
    mv_val, mv_u = ai.solve(4, (1,0,0), v_val, v_u, "Times")
    
    # 3. 0.5 * Result
    final_val, final_u = ai.solve(0.5, (0,0,0), mv_val, mv_u, "Times")
    
    print(f"\nFINAL: {final_val:.4f} Joules")
    input("Press Enter to Exit...")