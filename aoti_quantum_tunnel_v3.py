#!/usr/bin/env python3
# AOTI LEVEL 24: LOGARITHMIC QUANTUM ENGINE
# Uses Log-Scale visualization to reveal the "Ghost Particle"
# that was previously invisible to the naked eye.

import torch
import numpy as np
import matplotlib.pyplot as plt
import time

# --- CONFIGURATION ---
GRID_SIZE = 300
DX = 0.1
DT = 0.02
BARRIER_X = 150
BARRIER_WIDTH = 15
BARRIER_HEIGHT = 2.0 

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
        norm = 1.0 / (sigma * np.sqrt(2 * np.pi))
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
        
        # 4. Sponge
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
    
    # Standard High Energy Init
    space.initialize_wavepacket(x0=130, k0=4.0, sigma=6.0)
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    print("--- LOGARITHMIC QUANTUM VIEWER ---")
    print("Y-Axis is now Log Scale (Powers of 10).")
    print("Watch the 'Ghost' appear at 10^-4 level.")
    
    for t in range(800):
        space.time_step()
        
        if t % 5 == 0:
            prob = space.get_probability().numpy()
            barrier = space.V.numpy()
            
            ax.clear()
            
            # --- THE FIX: SEMI-LOG PLOT ---
            # We plot Probability on Log Scale
            # Add small epsilon to avoid log(0)
            ax.semilogy(prob + 1e-9, color='blue', linewidth=2, label='Log Probability')
            
            # Draw Barrier (Arbitrary height for visual context)
            ax.fill_between(range(GRID_SIZE), 1e-9, (barrier > 0) * 0.1, color='orange', alpha=0.3, label='Barrier')
            
            # Set Limits to see the "Ants"
            ax.set_ylim(1e-7, 1.0) 
            ax.set_xlim(100, 250)
            
            ax.grid(True, which="both", ls="-", alpha=0.5)
            ax.set_title(f"Logarithmic Tunneling View (Time {t})")
            ax.set_ylabel("Probability (Log Scale)")
            ax.legend(loc='upper right')
            
            # Highlight the Tunneling Zone
            if t > 200:
                ax.text(190, 1e-5, "THE GHOST ->", color='green', fontsize=12, fontweight='bold')
            
            plt.pause(0.001)

    plt.show()

if __name__ == "__main__":
    run_quantum_sim()