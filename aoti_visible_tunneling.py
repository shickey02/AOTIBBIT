#!/usr/bin/env python3
# AOTI LEVEL 25: THE THIN WALL EXPERIMENT
# Tuned parameters to make Quantum Tunneling unmistakably visible.
# Barrier Width reduced to allow massive leakage.

import torch
import numpy as np
import matplotlib.pyplot as plt
import time

# --- TUNED CONFIGURATION ---
GRID_SIZE = 300
DX = 0.1
DT = 0.015  # Slower time step for precision
# THE CRITICAL CHANGE: A Thinner Wall
BARRIER_X = 150
BARRIER_WIDTH = 5   # Was 15. This is 3x thinner.
BARRIER_HEIGHT = 1.5 # Lowered slightly from 2.0

class QuantumManifold:
    def __init__(self, size):
        self.size = size
        self.psi_real = torch.zeros(size)
        self.psi_imag = torch.zeros(size)
        self.V = torch.zeros(size)
        
    def build_barrier(self):
        self.V[BARRIER_X : BARRIER_X + BARRIER_WIDTH] = BARRIER_HEIGHT

    def initialize_wavepacket(self, x0, k0, sigma):
        x = torch.arange(self.size).float()
        envelope = torch.exp(-0.5 * ((x - x0) / sigma)**2)
        phase = k0 * x
        self.psi_real = envelope * torch.cos(phase)
        self.psi_imag = envelope * torch.sin(phase)

    def compute_curvature(self, field):
        left = torch.roll(field, 1)
        right = torch.roll(field, -1)
        return (left - 2*field + right) / (DX**2)

    def time_step(self):
        # 1. Kinetic
        curv_real = self.compute_curvature(self.psi_real)
        curv_imag = self.compute_curvature(self.psi_imag)
        kin_real = -0.5 * curv_real
        kin_imag = -0.5 * curv_imag
        
        # 2. Hamiltonian
        H_real = kin_real + (self.V * self.psi_real)
        H_imag = kin_imag + (self.V * self.psi_imag)
        
        # 3. Evolution
        self.psi_real += H_imag * DT
        self.psi_imag -= H_real * DT
        
        # 4. Sponge (Boundaries)
        sponge = torch.ones(self.size)
        sponge[:10] = 0.0; sponge[-10:] = 0.0 
        self.psi_real *= sponge; self.psi_imag *= sponge
        
        # Normalize
        prob = self.psi_real**2 + self.psi_imag**2
        total_prob = torch.sum(prob) * DX + 1e-9
        scale = 1.0 / torch.sqrt(total_prob)
        self.psi_real *= scale
        self.psi_imag *= scale

    def get_probability(self):
        return self.psi_real**2 + self.psi_imag**2

def run_quantum_sim():
    space = QuantumManifold(GRID_SIZE)
    space.build_barrier()
    
    # Fast, sharp particle
    space.initialize_wavepacket(x0=130, k0=5.0, sigma=5.0)
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    print("--- QUANTUM TUNNELING: THIN WALL EDITION ---")
    print("Wall Thickness: 5 units (previously 15)")
    print("Particle Energy: High")
    
    for t in range(800):
        space.time_step()
        
        if t % 5 == 0:
            prob = space.get_probability().numpy()
            barrier = space.V.numpy()
            
            ax.clear()
            
            # --- HYBRID VIEW ---
            # We use a Linear Scale now because the Ghost will be big enough to see!
            ax.plot(prob, color='blue', linewidth=2, label='Particle |Ψ|²')
            ax.fill_between(range(GRID_SIZE), 0, prob, color='blue', alpha=0.2)
            
            # Draw Barrier
            ax.fill_between(range(GRID_SIZE), 0, barrier * 0.05, color='orange', alpha=0.5, label='The Wall')
            
            # Zoom in on the exit zone
            ax.set_xlim(120, 220)
            ax.set_ylim(0, 0.1)
            
            ax.set_title(f"Quantum Tunneling (Time {t})")
            ax.legend(loc='upper right')
            
            # Label the Ghost if it exists
            if t > 100:
                # Check for mass on the right side
                right_mass = np.sum(prob[BARRIER_X + BARRIER_WIDTH + 5 :])
                if right_mass > 0.01:
                    ax.text(180, 0.02, "!!! GHOST ESCAPING !!!", color='red', fontsize=12, fontweight='bold')
            
            plt.pause(0.001)

    plt.show()

if __name__ == "__main__":
    run_quantum_sim()