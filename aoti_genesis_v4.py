#!/usr/bin/env python3
# AOTI LEVEL 32: STABLE GENESIS ENGINE
# Fixes numerical explosion by reducing Time Step (DT).
# Parameters set for "Mitosis" (Cell Division) -> "Coral".

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- PHYSICS CONFIGURATION ---
GRID_SIZE = 200
# THE FIX: Lower DT prevents numerical explosions
DT = 0.1  
DIFFUSION_U = 1.0
DIFFUSION_V = 0.5

# PARAMETERS: "Mitosis" moving into "Coral"
# F=0.0545, k=0.0620 is the standard "Brain Coral" spot
FEED_RATE = 0.0545
KILL_RATE = 0.0620

class GenesisManifold:
    def __init__(self):
        # 1. Background: Pure Food (U=1), No Life (V=0)
        self.U = np.ones((GRID_SIZE, GRID_SIZE))
        self.V = np.zeros((GRID_SIZE, GRID_SIZE))
        
        # 2. Seed: A jagged 20x20 block in the center
        # We make it asymmetrical to force chaotic growth
        r = 10
        cx, cy = GRID_SIZE//2, GRID_SIZE//2
        
        # Create a random blob
        seed_mask = np.random.rand(2*r, 2*r) > 0.5
        self.V[cx-r:cx+r, cy-r:cy+r][seed_mask] = 1.0

    def step(self):
        # Laplacian (5-point stencil)
        # Using numpy roll is slow but readable. 
        # For high-performance, scipy.ndimage.convolve is better, but we keep it pure numpy.
        
        # Pre-calculate rolls to avoid re-allocating inside the math line
        u = self.U
        v = self.V
        
        Lu = (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) + 
              np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 4*u)
              
        Lv = (np.roll(v, 1, axis=0) + np.roll(v, -1, axis=0) + 
              np.roll(v, 1, axis=1) + np.roll(v, -1, axis=1) - 4*v)
        
        # Reaction: 2V + U -> 3V
        uvv = u * (v ** 2)
        
        # Update
        # Note: We apply diffusion * DT. Since DT is 0.1, the change is gradual.
        du = (DIFFUSION_U * Lu) - uvv + (FEED_RATE * (1 - u))
        dv = (DIFFUSION_V * Lv) + uvv - ((FEED_RATE + KILL_RATE) * v)
        
        self.U += du * DT
        self.V += dv * DT
        
        # Clip ensures we never drift into invalid negative numbers
        self.U = np.clip(self.U, 0, 1)
        self.V = np.clip(self.V, 0, 1)

    def perturb(self):
        # Poke the petri dish (Inject noise)
        cx, cy = np.random.randint(0, GRID_SIZE, 2)
        r = 10
        # Clamp bounds
        x1, x2 = max(0, cx-r), min(GRID_SIZE, cx+r)
        y1, y2 = max(0, cy-r), min(GRID_SIZE, cy+r)
        self.V[x1:x2, y1:y2] = 1.0

def run_genesis():
    sim = GenesisManifold()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    print("--- STABLE GENESIS ENGINE ---")
    print("Numerical Instability Fixed (DT=0.1).")
    print("Controls:")
    print(" [SPACE] : Pause/Play")
    print(" [P]     : Perturb (Add new seed)")
    
    # Visualization
    img = ax.imshow(sim.V, cmap='magma', vmin=0, vmax=0.4, interpolation='bicubic')
    ax.axis('off')
    
    is_paused = [False]

    def on_key(event):
        if event.key == ' ':
            is_paused[0] = not is_paused[0]
        elif event.key == 'p':
            sim.perturb()
            print(">> Seed Injected")

    fig.canvas.mpl_connect('key_press_event', on_key)
    
    def update(frame):
        if is_paused[0]: return img,
            
        # Run 50 physics steps per render frame
        # (Since DT is 0.1, we need more steps to see visible change)
        for _ in range(50):
            sim.step()
        
        img.set_array(sim.V)
        ax.set_title(f"Cycle {frame*50}")
        return img,

    ani = animation.FuncAnimation(fig, update, frames=5000, interval=1, blit=False)
    plt.show()

if __name__ == "__main__":
    run_genesis()