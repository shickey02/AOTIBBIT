#!/usr/bin/env python3
# AOTI LEVEL 23: THE RELATIVISTIC ENGINE
# Simulates General Relativity (Schwarzschild Metric).
# Light rays follow Geodesics (Straight lines in curved space).
# Demonstrates Gravitational Lensing and the Event Horizon.

import torch
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
SCHWARZSCHILD_RADIUS = 2.0  # The Event Horizon size (Rs)
GRID_RANGE = 10.0
STEP_SIZE = 0.05

class SpacetimeManifold:
    """
    A Manifold where distance is defined by the Schwarzschild Metric.
    In this space, a 'straight line' is a curve.
    """
    def __init__(self, rs):
        self.Rs = rs

    def get_metric(self, pos):
        """
        Returns the curvature at a point (x, y).
        Metric g describes how to measure distance.
        """
        r = torch.norm(pos)
        
        # Avoid singularity
        if r < 0.1: r = torch.tensor(0.1)
        
        # Schwarzschild curvature factor (Simplified for 2D visual)
        # Space is stretched radially near the mass.
        # factor = 1 / (1 - Rs/r)
        
        # We model the "Effective Potential" for light.
        # In GR, light bends because time moves slower near mass.
        # Refractive Index n(r) approx 1 + 2*GM/rc^2
        # n(r) = 1 + Rs/r
        
        n = 1.0 + (self.Rs / r)
        return n

    def geodesic_step(self, pos, vel):
        """
        Moves the photon forward along a Geodesic.
        Snell's Law / Fermat's Principle: Light takes the path of "most time" (locally).
        Equation of Motion: d(n*v)/ds = grad(n)
        """
        # 1. Current position and refractive index
        n = self.get_metric(pos)
        
        # 2. Calculate Gradient of Curvature (Which way is 'Down'?)
        # We probe slightly to find the "slope" of spacetime
        epsilon = 0.01
        
        pos_x = pos + torch.tensor([epsilon, 0.0])
        pos_y = pos + torch.tensor([0.0, epsilon])
        
        nx = self.get_metric(pos_x)
        ny = self.get_metric(pos_y)
        
        grad_n = torch.tensor([
            (nx - n) / epsilon,
            (ny - n) / epsilon
        ])
        
        # 3. Update Velocity (The bending of light)
        # Acceleration = Gradient(n) - (Velocity * (Velocity dot Gradient(n)))
        # This keeps speed constant (c=1) but changes direction.
        
        # Normalize velocity to c=1
        vel = vel / torch.norm(vel)
        
        # Accel points towards higher curvature (the Black Hole)
        accel = grad_n
        
        # Update
        new_vel = vel + (accel * STEP_SIZE)
        new_vel = new_vel / torch.norm(new_vel) # Re-normalize (Light speed is constant)
        
        new_pos = pos + (new_vel * STEP_SIZE)
        
        return new_pos, new_vel

def run_relativity_sim():
    space = SpacetimeManifold(SCHWARZSCHILD_RADIUS)
    
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    
    # 1. Draw the Black Hole
    event_horizon = plt.Circle((0, 0), SCHWARZSCHILD_RADIUS, color='black', zorder=10, label='Event Horizon')
    photon_sphere = plt.Circle((0, 0), SCHWARZSCHILD_RADIUS * 1.5, color='orange', fill=False, linestyle='--', label='Photon Sphere')
    ax.add_artist(event_horizon)
    ax.add_artist(photon_sphere)
    
    print("--- RELATIVISTIC RAY TRACING ---")
    print(f"Black Hole Radius: {SCHWARZSCHILD_RADIUS}")
    print("Simulating light beams...")
    
    # 2. Fire Light Beams
    # We fire them from the left (-10) at different heights (y)
    start_x = -GRID_RANGE
    
    y_levels = np.linspace(-GRID_RANGE, GRID_RANGE, 25)
    
    for y in y_levels:
        # Initial State
        pos = torch.tensor([start_x, float(y)])
        vel = torch.tensor([1.0, 0.0]) # Moving Right
        
        path_x = []
        path_y = []
        
        captured = False
        
        for t in range(400): # Steps
            path_x.append(pos[0].item())
            path_y.append(pos[1].item())
            
            # Move
            pos, vel = space.geodesic_step(pos, vel)
            
            # Check Event Horizon Collision
            dist = torch.norm(pos)
            if dist < SCHWARZSCHILD_RADIUS:
                captured = True
                break
                
            # Stop if out of bounds
            if pos[0] > GRID_RANGE or abs(pos[1]) > GRID_RANGE:
                break
        
        # Plot
        color = 'red' if captured else 'cyan'
        alpha = 0.3 if captured else 0.8
        lw = 1 if captured else 1.5
        
        ax.plot(path_x, path_y, color=color, alpha=alpha, linewidth=lw)

    # Styling
    ax.set_facecolor('midnightblue')
    ax.set_xlim(-GRID_RANGE, GRID_RANGE)
    ax.set_ylim(-GRID_RANGE, GRID_RANGE)
    ax.set_aspect('equal')
    ax.set_title("AOTI General Relativity: Gravitational Lensing")
    plt.legend(loc='upper right')
    
    print("Render Complete.")
    plt.show()

if __name__ == "__main__":
    run_relativity_sim()