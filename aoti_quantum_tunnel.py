#!/usr/bin/env python3
# AOTI LEVEL 21: THE QUANTUM ENGINE
# Simulates the Time-Dependent Schrödinger Equation using Geometric Rotation.
# Demonstrates Quantum Tunneling: A particle passing through a solid wall.

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import cmath

# --- CONFIGURATION ---
GRID_SIZE = 300
DX = 0.1
DT = 0.05
BARRIER_X = 150
BARRIER_WIDTH = 20
BARRIER_HEIGHT = 1.5 # Potential Energy V

class QuantumManifold:
    """
    A 1D Manifold of Complex Vectors.
    Each point in space is a spinning clock.
    """
    def __init__(self, size):
        self.size = size
        # Wavefunction psi (Real, Imag)
        self.psi_real = torch.zeros(size)
        self.psi_imag = torch.zeros(size)
        
        # Potential Energy Landscape (The Wall)
        self.V = torch.zeros(size)
        
    def build_barrier(self):
        # Create a solid wall
        self.V[BARRIER_X : BARRIER_X + BARRIER_WIDTH] = BARRIER_HEIGHT

    def initialize_wavepacket(self, x0, k0, sigma):
        """
        Creates a Gaussian particle.
        x0: Start Position
        k0: Momentum (Velocity)
        sigma: Width
        """
        x = torch.arange(self.size).float()
        # Gaussian Envelope
        norm = 1.0 / (sigma * np.sqrt(2 * np.pi))
        envelope = torch.exp(-0.5 * ((x - x0) / sigma)**2)
        
        # Complex Rotation (Momentum)
        # psi = envelope * exp(i * k * x)
        # Euler: exp(ix) = cos(x) + i*sin(x)
        phase = k0 * x
        
        self.psi_real = envelope * torch.cos(phase)
        self.psi_imag = envelope * torch.sin(phase)

    def compute_curvature(self, field):
        """
        Geometric Curvature (Laplacian).
        How much does the vector differ from the average of its neighbors?
        Discrete 2nd Derivative: (f(x+1) - 2f(x) + f(x-1)) / dx^2
        """
        # We use torch.roll to access neighbors efficiently
        left = torch.roll(field, 1)
        right = torch.roll(field, -1)
        
        # Apply Dirichlet boundary conditions (zero at edges) by masking roll wrap-around
        # (Simplified: Just ignore edges in visual)
        curvature = (left - 2*field + right) / (DX**2)
        return curvature

    def time_step(self):
        """
        The Schrödinger Update:
        d(Psi)/dt = -i * Hamiltonian * Psi
        
        Hamiltonian = Kinetic (Curvature) + Potential (V)
        """
        # 1. Calculate Kinetic Energy (Curvature)
        # Kinetic = -0.5 * Laplacian
        curv_real = self.compute_curvature(self.psi_real)
        curv_imag = self.compute_curvature(self.psi_imag)
        
        kin_real = -0.5 * curv_real
        kin_imag = -0.5 * curv_imag
        
        # 2. Calculate Total Energy Operation (H * psi)
        # H_real = kin_real + V * psi_real
        H_real = kin_real + (self.V * self.psi_real)
        H_imag = kin_imag + (self.V * self.psi_imag)
        
        # 3. Time Evolution (Rotation)
        # d(Real)/dt = H_imag
        # d(Imag)/dt = -H_real
        # (This is the geometric rotation of the complex vector)
        
        self.psi_real += H_imag * DT
        self.psi_imag -= H_real * DT
        
        # Normalize (Conservation of Probability)
        # In a perfect symplectic integrator this isn't needed, but explicit Euler drifts.
        prob = self.psi_real**2 + self.psi_imag**2
        total_prob = torch.sum(prob) * DX
        scale = 1.0 / torch.sqrt(total_prob)
        self.psi_real *= scale
        self.psi_imag *= scale

    def get_probability(self):
        # Prob = |Psi|^2
        return self.psi_real**2 + self.psi_imag**2

# --- THE SIMULATOR ---
def run_quantum_sim():
    space = QuantumManifold(GRID_SIZE)
    space.build_barrier()
    
    # Initialize Particle
    # Start at 50, Moving Right (momentum 1.5), Width 10
    space.initialize_wavepacket(x0=50, k0=1.5, sigma=10.0)
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    print("--- QUANTUM REALITY ENGINE ---")
    print("Blue Wave  = The Particle (Probability)")
    print("Orange Box = The Impossible Wall")
    print("Observation: Watch the particle disappear and reappear on the other side.")
    
    for t in range(500):
        # 1. Physics Step
        space.time_step()
        
        if t % 2 == 0: # Render every 2nd frame
            prob = space.get_probability().numpy()
            barrier = space.V.numpy()
            
            ax.clear()
            ax.set_ylim(0, 0.05)
            ax.set_xlim(0, GRID_SIZE)
            
            # Plot Barrier
            # Scale barrier for visibility
            ax.fill_between(range(GRID_SIZE), 0, barrier * 0.02, color='orange', alpha=0.3, label='Potential Barrier (The Wall)')
            
            # Plot Wavefunction
            ax.plot(prob, color='blue', linewidth=2, label='Particle Probability |Ψ|²')
            ax.fill_between(range(GRID_SIZE), 0, prob, color='blue', alpha=0.1)
            
            # Plot Real/Imag components (The "Spin")
            # We scale them down just to show they exist
            # ax.plot(space.psi_real.numpy() * 0.01, 'g--', alpha=0.3, label='Real')
            # ax.plot(space.psi_imag.numpy() * 0.01, 'r--', alpha=0.3, label='Imag')

            ax.set_title(f"AOTI Quantum Engine (Time {t})")
            ax.text(10, 0.045, "AOTI Logic: Energy is Rotation, not Speed.", fontsize=10)
            ax.legend(loc='upper right')
            
            plt.pause(0.001)

    plt.show()

if __name__ == "__main__":
    run_quantum_sim()