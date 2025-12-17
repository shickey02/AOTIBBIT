#!/usr/bin/env python3
# AOTI LEVEL 22: HIGH-VELOCITY QUANTUM ENGINE
# Features:
# 1. Absorbing Boundary Conditions (The Sponge) to prevent edge artifacts.
# 2. High-Momentum Initialization for visible Tunneling.
# 3. Zoomed-In Visualization.

import torch
import numpy as np
import matplotlib.pyplot as plt
import time

# --- CONFIGURATION ---
GRID_SIZE = 300
DX = 0.1
DT = 0.02 # Smaller time step for high-energy stability
BARRIER_X = 150
BARRIER_WIDTH = 15
BARRIER_HEIGHT = 2.0 # High wall

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
        # Normalization factor
        norm = 1.0 / (sigma * np.sqrt(2 * np.pi))
        envelope = torch.exp(-0.5 * ((x - x0) / sigma)**2)
        
        # Rotation (Momentum)
        phase = k0 * x
        
        self.psi_real = envelope * torch.cos(phase)
        self.psi_imag = envelope * torch.sin(phase)

    def compute_curvature(self, field):
        # Discrete Laplacian (2nd Derivative)
        left = torch.roll(field, 1)
        right = torch.roll(field, -1)
        return (left - 2*field + right) / (DX**2)

    def time_step(self):
        # 1. Kinetic (Curvature)
        curv_real = self.compute_curvature(self.psi_real)
        curv_imag = self.compute_curvature(self.psi_imag)
        
        kin_real = -0.5 * curv_real
        kin_imag = -0.5 * curv_imag
        
        # 2. Hamiltonian (Total Energy)
        H_real = kin_real + (self.V * self.psi_real)
        H_imag = kin_imag + (self.V * self.psi_imag)
        
        # 3. Evolution (Rotation)
        self.psi_real += H_imag * DT
        self.psi_imag -= H_real * DT
        
        # 4. THE SPONGE (Absorbing Boundaries)
        # Prevents waves from wrapping around the universe
        sponge = torch.ones(self.size)
        sponge[:10] = 0.0  
        sponge[-10:] = 0.0 
        self.psi_real *= sponge
        self.psi_imag *= sponge
        
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
    
    # --- HIGH ENERGY INITIALIZATION ---
    # Start closer to the wall (130)
    # Move much faster (Momentum 4.0)
    space.initialize_wavepacket(x0=130, k0=4.0, sigma=6.0)
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    print("--- QUANTUM TUNNELING EXPERIMENT ---")
    print("Blue = Particle Wave")
    print("Orange = The Wall")
    print("Look to the RIGHT of the wall for the 'Ghost Particle'.")
    
    for t in range(800):
        space.time_step()
        
        # Speed up rendering (draw every 4th frame)
        if t % 4 == 0:
            prob = space.get_probability().numpy()
            barrier = space.V.numpy()
            
            ax.clear()
            # Zoom in on the action
            ax.set_xlim(100, 250)
            ax.set_ylim(0, 0.15) 
            
            # Plot Wall
            # Scale it down visually so it doesn't block the view
            ax.fill_between(range(GRID_SIZE), 0, barrier * 0.04, color='orange', alpha=0.5, label='Potential Barrier')
            
            # Plot Particle
            ax.plot(prob, color='blue', linewidth=2, label='Probability |Ψ|²')
            ax.fill_between(range(GRID_SIZE), 0, prob, color='blue', alpha=0.2)
            
            ax.set_title(f"Quantum Simulation (Time {t})")
            ax.legend(loc='upper right')
            
            plt.pause(0.001)

    plt.show()

if __name__ == "__main__":
    run_quantum_sim()