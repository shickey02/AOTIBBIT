#!/usr/bin/env python3
# AOTI LEVEL 26: CALIBRATED QUANTUM TUNNELING
# Parameters mathematically tuned for ~20% visible transmission.
# The "Ghost" will be a substantial chunk of the wave.

import torch
import numpy as np
import matplotlib.pyplot as plt

# --- CALIBRATED PHYSICS ---
GRID_SIZE = 400
DX = 0.1
DT = 0.005 # High precision time step
BARRIER_X = 200

# PHYSICS TUNING
# We want Energy (E) approx equal to Barrier Height (V)
# Kinetic E = 0.5 * k^2
k0 = 3.0          # Momentum
Energy = 0.5 * (k0**2) 
# We set barrier height slightly above Particle Energy to force tunneling
BARRIER_HEIGHT = Energy * 1.05 
BARRIER_WIDTH = 4 # Very thin wall (4 units = 0.4 distance)

class QuantumManifold:
    def __init__(self, size):
        self.size = size
        self.psi_real = torch.zeros(size)
        self.psi_imag = torch.zeros(size)
        self.V = torch.zeros(size)
        
    def build_barrier(self):
        # Create the Wall
        self.V[BARRIER_X : BARRIER_X + BARRIER_WIDTH] = BARRIER_HEIGHT

    def initialize_wavepacket(self, x0, k0, sigma):
        x = torch.arange(self.size).float() * DX # Physical distance
        # Normalize carefully
        norm = (1.0 / (np.pi * sigma**2))**0.25
        arg = -0.5 * ((x - x0*DX) / sigma)**2
        envelope = torch.exp(arg)
        # Phase = k * x (Physical)
        phase = k0 * x
        
        self.psi_real = envelope * torch.cos(phase)
        self.psi_imag = envelope * torch.sin(phase)
        
        # Normalize sum to 1
        prob = self.psi_real**2 + self.psi_imag**2
        scale = 1.0 / torch.sqrt(torch.sum(prob)*DX)
        self.psi_real *= scale
        self.psi_imag *= scale

    def time_step(self):
        # Discrete Laplacian (Standard 3-point stencil)
        # d2f/dx2 = (f(x+1) - 2f(x) + f(x-1)) / dx^2
        # Kinetic = -0.5 * Laplacian
        
        psi_r = self.psi_real
        psi_i = self.psi_imag
        
        # Curvature Real
        curv_r = (torch.roll(psi_r, 1) - 2*psi_r + torch.roll(psi_r, -1)) / (DX**2)
        # Curvature Imag
        curv_i = (torch.roll(psi_i, 1) - 2*psi_i + torch.roll(psi_i, -1)) / (DX**2)
        
        # Hamiltonians
        H_r = (-0.5 * curv_r) + (self.V * psi_r)
        H_i = (-0.5 * curv_i) + (self.V * psi_i)
        
        # Update (Euler)
        self.psi_real += H_i * DT
        self.psi_imag -= H_r * DT
        
        # Soft Sponge Boundaries
        sponge = torch.ones(self.size)
        sponge[:20] = 0.0; sponge[-20:] = 0.0
        self.psi_real *= sponge; self.psi_imag *= sponge

    def get_probability(self):
        return self.psi_real**2 + self.psi_imag**2

def run_simulation():
    space = QuantumManifold(GRID_SIZE)
    space.build_barrier()
    
    # Initialize well to the left
    space.initialize_wavepacket(x0=150, k0=k0, sigma=2.0)
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    print(f"--- CALIBRATED QUANTUM TUNNELING ---")
    print(f"Particle Energy: {Energy:.2f}")
    print(f"Barrier Height:  {BARRIER_HEIGHT:.2f}")
    print(f"Barrier Width:   {BARRIER_WIDTH * DX:.2f}")
    print("Regime: Resonant Tunneling (Ghost should be clearly visible)")
    
    for t in range(2000):
        # Perform multiple physics steps per render frame for speed
        for _ in range(5):
            space.time_step()
        
        if t % 10 == 0:
            prob = space.get_probability().numpy()
            barrier = space.V.numpy()
            
            ax.clear()
            
            # --- THE SPLIT VIEW ---
            # We want to see both the bounce and the ghost
            ax.set_xlim(120, 300) 
            ax.set_ylim(0, 0.25)
            
            # Draw Barrier
            ax.fill_between(range(GRID_SIZE), 0, barrier * 0.05, color='orange', alpha=0.5, label='The Wall')
            
            # Draw Wave
            ax.plot(prob, color='blue', linewidth=2, label='Particle |Ψ|²')
            ax.fill_between(range(GRID_SIZE), 0, prob, color='blue', alpha=0.2)
            
            # Calculate Ghost Mass (Probability on right side of wall)
            right_side_prob = np.sum(prob[BARRIER_X + BARRIER_WIDTH + 5:]) * DX
            
            ax.set_title(f"Time {t*5} | Ghost Probability: {right_side_prob*100:.1f}%")
            ax.legend(loc='upper right')
            
            # Visual Marker
            if right_side_prob > 0.05:
                ax.text(240, 0.05, "GHOST\nPACKET", color='green', fontsize=14, fontweight='bold', ha='center')
                ax.arrow(240, 0.04, 10, 0, head_width=0.01, color='green')
            
            plt.pause(0.001)

    plt.show()

if __name__ == "__main__":
    run_simulation()