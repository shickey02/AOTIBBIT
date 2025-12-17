#!/usr/bin/env python3
# AOTI LEVEL 28: THE NAVIER-STOKES WIND TUNNEL (LBM)
# Simulates fluid dynamics using pure geometric rules (Stream & Collide).
# Visualizes Vortex Shedding behind an obstacle.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- CONFIGURATION ---
HEIGHT = 80
WIDTH = 300
VISCOSITY = 0.02
OMEGA = 1 / (3*VISCOSITY + 0.5) # Relaxation parameter
U0 = 0.1 # Inflow velocity

# --- GEOMETRY: D2Q9 LATTICE ---
# 9 Velocities: Center, N, S, W, E, NW, NE, SW, SE
N_DISCRETE_VELOCITIES = 9
LATTICE_VELOCITIES = np.array([
    [0, 0],  [1, 0],  [0, 1], 
    [-1, 0], [0, -1], [1, 1], 
    [-1, 1], [-1, -1], [1, -1]
])
LATTICE_INDICES = np.arange(N_DISCRETE_VELOCITIES)
OPPOSITE_INDICES = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

# Geometric Weights for each direction
LATTICE_WEIGHTS = np.array([
    4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36
])

class WindTunnel:
    def __init__(self):
        # Initialize grid with uniform flow
        self.F = np.ones((HEIGHT, WIDTH, N_DISCRETE_VELOCITIES)) 
        # Add slight noise to trigger asymmetry (Chaos)
        self.F += np.random.randn(HEIGHT, WIDTH, N_DISCRETE_VELOCITIES) * 0.01
        
        # Define Obstacle (Cylinder in the middle)
        self.cylinder = np.full((HEIGHT, WIDTH), False)
        y, x = np.ogrid[:HEIGHT, :WIDTH]
        center_y, center_x = HEIGHT//2, WIDTH//4
        radius = 12
        dist_sq = (x - center_x)**2 + (y - center_y)**2
        self.cylinder[dist_sq < radius**2] = True

    def get_density(self):
        return np.sum(self.F, axis=2)

    def get_velocity(self, rho):
        # Momentum = Sum(F * c)
        ux = np.sum(self.F * LATTICE_VELOCITIES[:, 0], axis=2) / rho
        uy = np.sum(self.F * LATTICE_VELOCITIES[:, 1], axis=2) / rho
        return ux, uy

    def get_equilibrium(self, rho, ux, uy):
        # AOTI Logic: What is the "Perfect" geometric distribution for this velocity?
        # We project the macroscopic velocity onto the 9 discrete directions.
        eq = np.zeros_like(self.F)
        u_sq = ux**2 + uy**2
        
        for i, w in enumerate(LATTICE_WEIGHTS):
            # Dot product: c_i * u
            cu = LATTICE_VELOCITIES[i, 0] * ux + LATTICE_VELOCITIES[i, 1] * uy
            eq[:, :, i] = rho * w * (1 + 3*cu + 4.5*cu**2 - 1.5*u_sq)
        return eq

    def step(self):
        # 1. STREAMING (Geometric Shift)
        # Move particles to their neighbors
        for i, (cx, cy) in enumerate(LATTICE_VELOCITIES):
            # np.roll shifts the array elements cyclically
            self.F[:, :, i] = np.roll(self.F[:, :, i], cx, axis=1)
            self.F[:, :, i] = np.roll(self.F[:, :, i], cy, axis=0)

        # 2. BOUNDARIES (The Walls)
        # Reflect particles hitting the cylinder (Bounce-Back)
        boundary_F = self.F[self.cylinder, :]
        self.F[self.cylinder, :] = boundary_F[:, OPPOSITE_INDICES]

        # 3. COLLISION (The Relaxation)
        # Calculate Macroscopic variables
        rho = self.get_density()
        ux, uy = self.get_velocity(rho)
        
        # Force Inflow (Left side) to constant velocity U0
        ux[:, 0] = U0
        uy[:, 0] = 0
        rho[:, 0] = 1 # Approximation
        
        # Calculate Equilibrium
        F_eq = self.get_equilibrium(rho, ux, uy)
        
        # Relax towards equilibrium
        self.F += -(1.0 / OMEGA) * (self.F - F_eq) # No explicit collision calc, just geometric decay!

        # 4. CURL (Vorticity) for visualization
        # curl = d(uy)/dx - d(ux)/dy
        duy_dx = np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)
        dux_dy = np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)
        curl = duy_dx - dux_dy
        
        # Mask cylinder in visual
        curl[self.cylinder] = np.nan
        return curl

def run_wind_tunnel():
    tunnel = WindTunnel()
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    print("--- AOTI WIND TUNNEL ---")
    print("Simulating Navier-Stokes equations via Lattice Boltzmann.")
    print("Watch the 'Vortex Street' emerge behind the cylinder.")
    print("Red/Blue = Spinning Air (Clockwise/Counter-Clockwise).")
    
    # Initial Plot
    curl = tunnel.step()
    img = ax.imshow(curl, cmap='bwr', vmin=-0.05, vmax=0.05, origin='lower')
    
    # Draw Cylinder
    circle = plt.Circle((WIDTH//4, HEIGHT//2), 12, color='black')
    ax.add_artist(circle)
    ax.set_title("Initializing Flow...")
    
    def update(frame):
        # Speed up: Run 3 physics steps per animation frame
        for _ in range(3):
            curl = tunnel.step()
        
        img.set_array(curl)
        ax.set_title(f"AOTI Fluid Dynamics (Frame {frame*3})")
        return img,

    ani = animation.FuncAnimation(fig, update, frames=2000, interval=1, blit=False)
    plt.show()

if __name__ == "__main__":
    run_wind_tunnel()