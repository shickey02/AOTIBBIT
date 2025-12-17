#!/usr/bin/env python3
# AOTI LEVEL 30: THE GENESIS ENGINE (TUNED)
# Parameters tuned for aggressive "Mitosis" (Cell Division).
# Removed background noise for a clean, sharp visual.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- THE GOLDILOCKS ZONE ---
GRID_SIZE = 200 # Smaller grid = Faster growth visualization
DT = 1.0

# TUNED PARAMETERS (The "Worms" Regime)
# This specific pair causes spots to elongate and split.
FEED_RATE = 0.037
KILL_RATE = 0.060

DIFFUSION_U = 1.0
DIFFUSION_V = 0.5

class GenesisManifold:
    def __init__(self):
        # 1. PURE BACKGROUND
        # U = 1.0 (Food is everywhere)
        # V = 0.0 (No life yet)
        self.U = np.ones((GRID_SIZE, GRID_SIZE))
        self.V = np.zeros((GRID_SIZE, GRID_SIZE))
        
        # 2. THE SEED (Single Square)
        # We plant a 20x20 block of "Life" in the center.
        r = 10
        cx, cy = GRID_SIZE//2, GRID_SIZE//2
        
        # Set V=1 (Life) in the center
        self.V[cx-r:cx+r, cy-r:cy+r] = 1.0
        
        # Add tiny noise ONLY to the seed (to break symmetry so it grows organically)
        noise = np.random.rand(2*r, 2*r) * 0.1
        self.V[cx-r:cx+r, cy-r:cy+r] += noise

    def laplacian(self, grid):
        # Convolution with 3x3 kernel for diffusion
        # Center weight -4, Neighbors +1
        top = np.roll(grid, 1, axis=0)
        bottom = np.roll(grid, -1, axis=0)
        left = np.roll(grid, 1, axis=1)
        right = np.roll(grid, -1, axis=1)
        return (top + bottom + left + right - 4*grid)

    def step(self):
        Lu = self.laplacian(self.U)
        Lv = self.laplacian(self.V)
        
        # Reaction: 2V + U -> 3V (V eats U and reproduces)
        reaction = self.U * (self.V ** 2)
        
        du = (DIFFUSION_U * Lu) - reaction + (FEED_RATE * (1 - self.U))
        dv = (DIFFUSION_V * Lv) + reaction - ((FEED_RATE + KILL_RATE) * self.V)
        
        self.U += du * DT
        self.V += dv * DT
        
        # Constrain
        self.U = np.clip(self.U, 0, 1)
        self.V = np.clip(self.V, 0, 1)

def run_genesis():
    sim = GenesisManifold()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    print("--- AOTI GENESIS V2 ---")
    print("Regime: Mitosis (Worm/Coral Growth)")
    print("Initial State: Pure Purple with one Yellow Seed.")
    
    # VISUALIZATION FIX:
    # V is usually small (0.2 - 0.4).
    # We cap vmax at 0.5 to make the "Life" glow bright Yellow.
    img = ax.imshow(sim.V, cmap='magma', interpolation='bicubic', vmin=0, vmax=0.4)
    ax.axis('off')
    
    def update(frame):
        # Fast Forward: 40 steps per frame
        for _ in range(40):
            sim.step()
        
        img.set_array(sim.V)
        ax.set_title(f"Genesis Cycle {frame*40}")
        return img,

    ani = animation.FuncAnimation(fig, update, frames=2000, interval=1, blit=False)
    plt.show()

if __name__ == "__main__":
    run_genesis()