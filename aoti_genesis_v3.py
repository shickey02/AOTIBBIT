#!/usr/bin/env python3
# AOTI LEVEL 31: THE CORAL REEF (STABILIZED)
# Parameters tuned for slow, branching growth (Brain Coral).
# Includes Play/Pause controls to watch the genesis.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- CONFIGURATION ---
GRID_SIZE = 200
DT = 1.0

# "CORAL" PARAMETERS
# These are famous values for creating "Brain Coral" structures.
FEED_RATE = 0.0545
KILL_RATE = 0.0620

DIFFUSION_U = 1.0
DIFFUSION_V = 0.5

class GenesisManifold:
    def __init__(self):
        # U = Food (1.0), V = Life (0.0)
        self.U = np.ones((GRID_SIZE, GRID_SIZE))
        self.V = np.zeros((GRID_SIZE, GRID_SIZE))
        
        # Plant the seed in the center
        r = 5 # Smaller seed
        cx, cy = GRID_SIZE//2, GRID_SIZE//2
        self.V[cx-r:cx+r, cy-r:cy+r] = 1.0
        
        # Add noise to the seed to encourage branching
        self.V[cx-r:cx+r, cy-r:cy+r] += np.random.rand(2*r, 2*r) * 0.1

    def step(self):
        # 5-point Laplacian Stencil
        # (Faster calculation)
        top = np.roll(self.U, 1, axis=0)
        bottom = np.roll(self.U, -1, axis=0)
        left = np.roll(self.U, 1, axis=1)
        right = np.roll(self.U, -1, axis=1)
        Lu = top + bottom + left + right - 4*self.U
        
        top = np.roll(self.V, 1, axis=0)
        bottom = np.roll(self.V, -1, axis=0)
        left = np.roll(self.V, 1, axis=1)
        right = np.roll(self.V, -1, axis=1)
        Lv = top + bottom + left + right - 4*self.V
        
        # Reaction: 2V + U -> 3V
        uvv = self.U * (self.V ** 2)
        
        du = (DIFFUSION_U * Lu) - uvv + (FEED_RATE * (1 - self.U))
        dv = (DIFFUSION_V * Lv) + uvv - ((FEED_RATE + KILL_RATE) * self.V)
        
        self.U += du * DT
        self.V += dv * DT
        
        self.U = np.clip(self.U, 0, 1)
        self.V = np.clip(self.V, 0, 1)

def run_genesis():
    sim = GenesisManifold()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    print("--- GENESIS: CORAL REEF ---")
    print("Yellow = Living Coral")
    print("Purple = Empty Ocean")
    print("The seed will grow slowly. Give it 30 seconds.")
    
    # Visualization: V usually stays around 0.2 - 0.4 in this mode
    img = ax.imshow(sim.V, cmap='magma', vmin=0, vmax=0.4, interpolation='bicubic')
    ax.axis('off')
    
    # State for Play/Pause
    is_paused = [False]

    def on_key(event):
        if event.key == ' ':
            is_paused[0] = not is_paused[0]

    fig.canvas.mpl_connect('key_press_event', on_key)
    
    def update(frame):
        if is_paused[0]:
            return img,
            
        # Run 8 steps per frame (Fast enough to see growth, slow enough to watch)
        for _ in range(8):
            sim.step()
        
        img.set_array(sim.V)
        ax.set_title(f"Growth Cycle {frame*8} (Space to Pause)")
        return img,

    ani = animation.FuncAnimation(fig, update, frames=5000, interval=1, blit=False)
    plt.show()

if __name__ == "__main__":
    run_genesis()