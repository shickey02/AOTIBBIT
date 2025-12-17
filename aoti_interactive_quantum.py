#!/usr/bin/env python3
# AOTI LEVEL 27: INTERACTIVE QUANTUM LAB
# Features:
# 1. Manual Frame Stepping (Right Arrow)
# 2. Play/Pause Toggle (Space)
# 3. Uses the "Calibrated" physics where tunneling is visible (~20% transmission)

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# --- PHYSICS CONFIGURATION (The Goldilocks Zone) ---
GRID_SIZE = 400
DX = 0.1
DT = 0.005 
BARRIER_X = 200
BARRIER_WIDTH = 4 

# Energy Calibration
k0 = 3.0          
Energy = 0.5 * (k0**2) 
BARRIER_HEIGHT = Energy * 1.05 

class QuantumManifold:
    def __init__(self, size):
        self.size = size
        self.psi_real = torch.zeros(size)
        self.psi_imag = torch.zeros(size)
        self.V = torch.zeros(size)
        self.build_barrier()
        
    def build_barrier(self):
        self.V[BARRIER_X : BARRIER_X + BARRIER_WIDTH] = BARRIER_HEIGHT

    def initialize_wavepacket(self):
        # Always start at same spot
        x0 = 150
        sigma = 2.0
        
        x = torch.arange(self.size).float() * DX 
        norm = (1.0 / (np.pi * sigma**2))**0.25
        arg = -0.5 * ((x - x0*DX) / sigma)**2
        envelope = torch.exp(arg)
        phase = k0 * x
        
        self.psi_real = envelope * torch.cos(phase)
        self.psi_imag = envelope * torch.sin(phase)
        
        # Normalize
        prob = self.psi_real**2 + self.psi_imag**2
        scale = 1.0 / torch.sqrt(torch.sum(prob)*DX)
        self.psi_real *= scale
        self.psi_imag *= scale

    def time_step(self):
        # Physics Update Loop
        psi_r = self.psi_real
        psi_i = self.psi_imag
        
        curv_r = (torch.roll(psi_r, 1) - 2*psi_r + torch.roll(psi_r, -1)) / (DX**2)
        curv_i = (torch.roll(psi_i, 1) - 2*psi_i + torch.roll(psi_i, -1)) / (DX**2)
        
        H_r = (-0.5 * curv_r) + (self.V * psi_r)
        H_i = (-0.5 * curv_i) + (self.V * psi_i)
        
        self.psi_real += H_i * DT
        self.psi_imag -= H_r * DT
        
        # Sponge
        sponge = torch.ones(self.size)
        sponge[:20] = 0.0; sponge[-20:] = 0.0
        self.psi_real *= sponge; self.psi_imag *= sponge

    def get_probability(self):
        return self.psi_real**2 + self.psi_imag**2

class InteractiveSim:
    def __init__(self):
        self.space = QuantumManifold(GRID_SIZE)
        self.space.initialize_wavepacket()
        
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.running = False # Start Paused
        self.frame = 0
        
        # Connect Events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        self.update_plot()
        print("--- INTERACTIVE CONTROLS ---")
        print(" [SPACE] : Toggle Play/Pause")
        print(" [RIGHT] : Step Forward 1 Frame")
        print(" [R]     : Reset Simulation")
        
        plt.show()

    def on_key(self, event):
        if event.key == ' ':
            self.running = not self.running
            if self.running:
                self.run_loop()
        elif event.key == 'right':
            self.step_physics()
            self.update_plot()
        elif event.key == 'r':
            self.reset()

    def reset(self):
        self.space = QuantumManifold(GRID_SIZE)
        self.space.initialize_wavepacket()
        self.frame = 0
        self.running = False
        self.update_plot()

    def run_loop(self):
        # While running is true, keep stepping
        while self.running:
            self.step_physics()
            self.update_plot()
            plt.pause(0.001)

    def step_physics(self):
        # Calculate 10 physics steps per render frame for smoothness
        for _ in range(10):
            self.space.time_step()
        self.frame += 10

    def update_plot(self):
        self.ax.clear()
        
        prob = self.space.get_probability().numpy()
        barrier = self.space.V.numpy()
        
        # Split View Limits
        self.ax.set_xlim(120, 300) 
        self.ax.set_ylim(0, 0.25)
        
        # Draw Wall
        self.ax.fill_between(range(GRID_SIZE), 0, barrier * 0.05, color='orange', alpha=0.5, label='The Wall')
        
        # Draw Particle
        self.ax.plot(prob, color='blue', linewidth=2, label='Wavefunction')
        self.ax.fill_between(range(GRID_SIZE), 0, prob, color='blue', alpha=0.2)
        
        # Stats
        right_mass = np.sum(prob[BARRIER_X + BARRIER_WIDTH + 5:]) * DX
        
        self.ax.set_title(f"Time {self.frame} | Ghost Mass: {right_mass*100:.1f}%")
        self.ax.text(125, 0.23, "CONTROLS: Space=Play/Pause, Right=Step, R=Reset", fontsize=9)
        
        # Detect Ghost
        if right_mass > 0.05:
            self.ax.text(240, 0.05, "GHOST DETECTED", color='green', fontsize=14, fontweight='bold', ha='center')
            self.ax.arrow(240, 0.04, 10, 0, head_width=0.01, color='green')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

if __name__ == "__main__":
    sim = InteractiveSim()