#!/usr/bin/env python3
# BBIT INFINITE: The Color Navigator
# Demonstrates Vector Navigation in a continuous, open-ended space.

import numpy as np
import random

# --- 1. THE MANIFOLD (Concept Space) ---
# We define abstract concepts as centroids in RGB space (normalized 0-1)
CONCEPTS = {
    "WARMTH": np.array([0.9, 0.4, 0.1]), # Orange-ish
    "COLD":   np.array([0.1, 0.3, 0.9]), # Blue-ish
    "NATURE": np.array([0.2, 0.8, 0.2]), # Green
    "VOID":   np.array([0.0, 0.0, 0.0]), # Black
    "LIGHT":  np.array([1.0, 1.0, 1.0])  # White
}

class ColorAgent:
    def __init__(self, target_concept):
        self.target_vec = CONCEPTS[target_concept]
        self.current_pos = np.random.rand(3) # Start at random color
        
        # APEX PERSONALITY (Evolved from Yahtzee)
        self.risk = -0.5  # Negative Risk (Chaos allowed)
        self.obsession = 5.0 
        self.precision = 0.1 # Step size (Learning Rate)

    def sense(self):
        # In a real infinite domain, this would be an Encoder (ResNet/Transformer)
        # Here, we just see our current RGB values
        return self.current_pos

    def act(self):
        # 1. Calculate Vector Gap
        # "Where am I relative to the goal?"
        gap = self.target_vec - self.current_pos
        
        # 2. Apex Modulation
        # "How desperate am I?"
        dist = np.linalg.norm(gap)
        
        # Dynamic Step Size:
        # If far away, move fast (Aggressive).
        # If close, slow down (Precision).
        step_magnitude = self.precision * (1.0 + dist*2.0)
        
        # 3. The Move (Vector Addition)
        # We don't check "Is moving Red good?" We just MOVE towards the vector.
        # movement = direction * magnitude
        move = (gap / (dist + 0.001)) * step_magnitude
        
        # 4. Chaos Injection (Negative Risk)
        # If we are stuck (gap is small but not zero), shake the system
        if dist < 0.1 and self.risk < 0:
            noise = np.random.normal(0, 0.05, 3)
            move += noise
            
        # Apply Physics (Environment constraints)
        new_pos = self.current_pos + move
        self.current_pos = np.clip(new_pos, 0.0, 1.0)
        
        return dist

def run_simulation():
    target = "WARMTH"
    print(f"--- BBIT INFINITE NAVIGATOR ---")
    print(f"Target Concept: {target} {CONCEPTS[target]}")
    
    agent = ColorAgent(target)
    print(f"Start Pos: {agent.current_pos}")
    
    for t in range(1, 21):
        dist = agent.act()
        rgb = [int(x*255) for x in agent.current_pos]
        
        # Status Bar
        bar = "=" * int((1.0 - dist) * 20)
        print(f"Step {t:02d}: RGB{rgb} | Dist: {dist:.4f} | {bar}>")
        
        if dist < 0.01:
            print(">> CONVERGENCE ACHIEVED.")
            break

if __name__ == "__main__":
    run_simulation()