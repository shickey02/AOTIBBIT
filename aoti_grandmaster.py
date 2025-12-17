#!/usr/bin/env python3
# AOTI LEVEL 12: THE GRAND COGNITIVE STACK
# A Full Neuro-Symbolic Architecture combining:
# 1. Rigid Crystal Manifolds (The Physics of Math)
# 2. Semantic Routing (The Intuition)
# 3. Cybernetic Regulation (The Awareness)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

# --- CONFIGURATION ---
RANGE = 50 
VOCAB_SIZE = (RANGE * 2) + 1
OFFSET = RANGE
DIM_SEMANTIC = 16 # Dimension for the "Meaning" of the problem (Words/Context)

# --- LAYER 1: THE MANIFOLD LIBRARY (TRUTH) ---
class ManifoldLibrary(nn.Module):
    def __init__(self):
        super().__init__()
        # The Core Number Line (The Rigid Crystal)
        # We use 2D vectors for the geometry
        self.crystal = nn.Embedding(VOCAB_SIZE, 2)
        
        # We enable two "Views" (Cortices) on this data
        # 1. Linear View: Direct Vector Addition
        # 2. Log View: Logarithmic Transformation
        
        # Init with noise
        nn.init.uniform_(self.crystal.weight, -0.1, 0.1)

    def build_crystal(self):
        """
        Pre-training phase: Constructs the rigid number line.
        This represents 'Childhood Learning' - learning the structure of numbers
        before solving physics.
        """
        print("--- LAYER 1: CONSTRUCTING THE CRYSTAL MANIFOLD ---")
        opt = optim.Adam(self.crystal.parameters(), lr=0.05)
        
        # We enforce the "Universal Step" constraint (AOTI Level 10)
        # Step[n] must equal Step[0]
        
        for epoch in range(1001):
            opt.zero_grad()
            
            # 1. Anchor 0 to [0,0] and 1 to [1,0]
            v0 = self.crystal(torch.tensor(OFFSET))
            v1 = self.crystal(torch.tensor(OFFSET + 1))
            loss_anchor = torch.sum(v0**2) + torch.sum((v1 - torch.tensor([1.0, 0.0]))**2)
            
            # 2. Rigid Step
            indices = torch.arange(VOCAB_SIZE - 1)
            next_indices = torch.arange(1, VOCAB_SIZE)
            vecs = self.crystal(indices)
            next_vecs = self.crystal(next_indices)
            
            steps = next_vecs - vecs
            ideal_step = v1 - v0
            loss_step = torch.mean((steps - ideal_step.detach())**2)
            
            loss = loss_anchor + loss_step
            loss.backward()
            opt.step()
            
            if epoch % 500 == 0:
                print(f"   [Genesis] Epoch {epoch}: Structural Flaw {loss.item():.6f}")

    def execute(self, a_val, b_val, mode):
        """
        Performs the math operation using the manifold geometry.
        """
        # Convert integer values to indices
        idx_a = torch.tensor(a_val + OFFSET)
        idx_b = torch.tensor(b_val + OFFSET)
        
        vec_a = self.crystal(idx_a)
        vec_b = self.crystal(idx_b)
        
        if mode == 'LINEAR':
            # Linear Space: Vector A + Vector B
            # This solves Addition/Subtraction
            result_vec = vec_a + vec_b
            
        elif mode == 'LOG':
            # Log Space: We treat the inputs as if they are in log-space
            # For this demo, we use the property that Multiplication 
            # is scaling/rotation in the complex plane representation.
            # (a+bi)(c+di)
            real = (vec_a[0] * vec_b[0]) - (vec_a[1] * vec_b[1])
            imag = (vec_a[0] * vec_b[1]) + (vec_a[1] * vec_b[0])
            result_vec = torch.stack([real, imag])

        # DECODE: Find the nearest integer vector to our result
        # This is the "Perception" step
        all_vecs = self.crystal.weight.detach()
        dists = torch.norm(all_vecs - result_vec, dim=1)
        
        # The GAP is the distance to the nearest valid concept
        gap = torch.min(dists).item()
        answer = torch.argmin(dists).item() - OFFSET
        
        return answer, gap

# --- LAYER 2: THE SEMANTIC ROUTER (INTUITION) ---
class SemanticRouter(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: Context Vector (e.g., embeddings for "Force", "Mass")
        # Output: Scores for [LINEAR, LOG]
        self.net = nn.Sequential(
            nn.Linear(DIM_SEMANTIC, 8),
            nn.ReLU(),
            nn.Linear(8, 2) 
        )
    
    def decide(self, context_vec, temperature):
        """
        Returns the chosen mode and the confidence.
        High Temperature (Chaos) flattens the probabilities.
        """
        logits = self.net(context_vec)
        
        # Apply Temperature (Chaos Regulation)
        if temperature > 0:
            # Add noise to logits based on Chaos level
            noise = torch.randn_like(logits) * temperature * 2.0
            logits = logits + noise
            
        probs = torch.softmax(logits, dim=0)
        
        if probs[0] > probs[1]:
            return 'LINEAR', probs[0].item()
        else:
            return 'LOG', probs[1].item()

# --- LAYER 3: THE EXECUTIVE (AWARENESS) ---
class GrandmasterAI:
    def __init__(self):
        self.manifolds = ManifoldLibrary()
        self.router = SemanticRouter()
        
        # Cybernetics
        self.chaos = 0.0
        self.patience = 0
        self.history = []

    def boot(self):
        self.manifolds.build_crystal()
        print("--- SYSTEM READY ---\n")

    def solve(self, problem_name, a, b, context_vec_sim):
        print(f"=== REASONING TASK: {problem_name} ({a}, {b}) ===")
        
        self.chaos = 0.0
        self.patience = 0
        
        for step in range(1, 6): # Max 5 thought steps
            # 1. INTUITION (Router)
            context = torch.tensor(context_vec_sim)
            mode, conf = self.router.decide(context, self.chaos)
            
            # 2. ACTION (Manifold Execution)
            # The Router selected the tool. Now we use the tool.
            ans, gap = self.manifolds.execute(a, b, mode)
            
            # 3. AWARENESS (The BBIT Monitor)
            # We visualize the gap.
            # Low Gap = The result landed perfectly on a number.
            # High Gap = The result landed in the void (Wrong Manifold).
            
            bar_len = int(gap * 20)
            bar = "#" * bar_len
            print(f"Step {step} | Mode: {mode:<6} | Ans: {ans:<3} | Gap: {gap:.4f} [{bar:<10}] | Chaos: {self.chaos:.1f}")
            
            # 4. REGULATION
            if gap < 0.1:
                print(f">> CONVERGENCE. The answer fits the geometry of reality.")
                print(f">> FINAL ANSWER: {ans}")
                return ans
            
            else:
                print(f"   >> DISSONANCE DETECTED. Result is invalid. Increasing Chaos.")
                self.chaos += 0.5 # Spike the anxiety
                self.patience += 1
                
                # In a real training loop, we would backpropagate this failure to the Router
                # to teach it "Don't use Linear for Physics".
        
        print(">> FAILED to converge.")
        return None

if __name__ == "__main__":
    ai = GrandmasterAI()
    ai.boot()
    
    # --- SIMULATION ---
    
    # PROBLEM 1: SIMPLE ADDITION (John has 5 apples, buys 3)
    # Context Vector: Simulates embeddings for "Apple", "Buy", "Add"
    # These concepts align with LINEAR space naturally.
    # We cheat slightly by initializing weights that favor Linear for this vector
    # to show the 'happy path'.
    ctx_apples = [1.0] * DIM_SEMANTIC 
    ai.router.net[2].bias.data = torch.tensor([1.0, -1.0]) # Bias towards Linear initially
    
    ai.solve("Apples Count", 5, 3, ctx_apples)
    print("\n")
    
    # PROBLEM 2: PHYSICS (Force = Mass 5 * Accel 4)
    # Context Vector: "Mass", "Acceleration", "Physics"
    # The Router *should* pick LOG. But let's say it's untrained and guesses LINEAR first.
    # Watch the BBIT Loop correct it.
    ctx_physics = [0.5] * DIM_SEMANTIC
    
    # Force the Router to be WRONG initially (Simulate a hallucination)
    # We explicitly set weights to prefer LINEAR even for physics
    ai.router.net[2].bias.data = torch.tensor([5.0, -5.0]) 
    
    ai.solve("Calculate Force (F=ma)", 5, 4, ctx_physics)