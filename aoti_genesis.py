#!/usr/bin/env python3
# AOTI LEVEL 29: THE GENESIS ENGINE (Reaction-Diffusion)
# Simulates the Gray-Scott Model of Morphogenesis.
# This math describes how biological patterns (spots, stripes) emerge.
# Solves: du/dt = Du*Laplacian(u) - uv^2 + F(1-u)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- CONFIGURATION (The DNA of the System) ---
GRID_SIZE = 256
DT = 1.0

# "Mitosis" Settings (The sweet spot for cell division patterns)
# Try changing these slightly to get "Spots" vs "Fingerprints"
FEED_RATE = 0.055      # "Food" (F)
KILL_RATE = 0.062      # "Death" (k)

DIFFUSION_U = 1.0      # Rate A spreads
DIFFUSION_V = 0.5      # Rate B spreads

class GenesisManifold:
    def __init__(self):
        # Two Chemical Fields: U (Prey/Food), V (Predator/Cell)
        self.U = np.ones((GRID_SIZE, GRID_SIZE))
        self.V = np.zeros((GRID_SIZE, GRID_SIZE))
        
        # Seed the chaos: Place a small square of "V" in the center
        # This is the "Spark of Life"
        r = 10
        cx, cy = GRID_SIZE//2, GRID_SIZE//2
        self.V[cx-r:cx+r, cy-r:cy+r] = 1.0
        
        # Add random noise to break symmetry
        self.V += np.random.rand(GRID_SIZE, GRID_SIZE) * 0.05

    def laplacian(self, grid):
        """
        Geometric Curvature (2D Discrete Laplacian).
        How much does the center differ from its neighbors?
        Convolution Kernel:
        [[0.05, 0.2, 0.05],
         [0.2, -1.0, 0.2],
         [0.05, 0.2, 0.05]]
        """
        # Using np.roll for periodic boundaries (Torus topology)
        top = np.roll(grid, 1, axis=0)
        bottom = np.roll(grid, -1, axis=0)
        left = np.roll(grid, 1, axis=1)
        right = np.roll(grid, -1, axis=1)
        
        # Diagonal neighbors (optional, but makes it smoother)
        # For speed, we stick to the 5-point stencil or a weighted 9-point
        # Let's use simple 5-point for speed:
        # lap = (top + bottom + left + right - 4*center)
        
        lap = (top + bottom + left + right - 4*grid)
        return lap

    def step(self):
        # 1. Calculate Spatial Spread (Diffusion)
        # How chemicals move through the manifold
        Lu = self.laplacian(self.U)
        Lv = self.laplacian(self.V)
        
        # 2. Calculate Reaction (The Biology)
        # uv^2 is the interaction term (Predator eating Prey)
        reaction = self.U * (self.V ** 2)
        
        # Gray-Scott Equations:
        # dU = Diff_U - Reaction + Feed
        du = (DIFFUSION_U * Lu) - reaction + (FEED_RATE * (1 - self.U))
        
        # dV = Diff_V + Reaction - Kill
        dv = (DIFFUSION_V * Lv) + reaction - ((FEED_RATE + KILL_RATE) * self.V)
        
        # 3. Update State
        self.U += du * DT
        self.V += dv * DT
        
        # Clip to valid range [0, 1] for stability
        self.U = np.clip(self.U, 0, 1)
        self.V = np.clip(self.V, 0, 1)

def run_genesis():
    sim = GenesisManifold()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    print("--- AOTI GENESIS ENGINE ---")
    print("Simulating Reaction-Diffusion (Turing Patterns).")
    print("Yellow = The Living Pattern (Chemical V).")
    print("Purple = Empty Space (Chemical U).")
    print("Watch 'Biology' grow from a single square.")
    
    # We visualize Chemical V (The "Cells")
    img = ax.imshow(sim.V, cmap='inferno', interpolation='bicubic')
    ax.axis('off') # No axes needed for organic life
    
    def update(frame):
        # Speed up: 20 physics steps per render frame
        # Life is slow; we fast-forward it.
        for _ in range(20):
            sim.step()
        
        img.set_array(sim.V)
        ax.set_title(f"Genesis Cycle {frame*20}")
        return img,

    ani = animation.FuncAnimation(fig, update, frames=2000, interval=1, blit=False)
    plt.show()

if __name__ == "__main__":
    run_genesis()