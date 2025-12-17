#!/usr/bin/env python3
# AOTI LEVEL 16: THE SI TESSERACT ENGINE & VISUALIZER
# Solves physics using 5D geometry [Value, Phase, Mass, Length, Time].
# Includes a real-time rendering engine to see the AI's geometric reasoning.

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# --- CONFIGURATION ---
RANGE = 200 # Larger range for calculations like v^2
VOCAB_SIZE = (RANGE * 2) + 1
OFFSET = RANGE
# Dimensions: [Real, Imag, MASS, LENGTH, TIME]
DIMENSIONS = 5 

class ReasoningVisualizer:
    """
    Handles real-time plotting of the AI's geometric state.
    """
    def __init__(self):
        plt.ion() # Interactive mode
        self.fig = plt.figure(figsize=(10, 5))
        
        # Subplot 1: 2D Complex Plane (Value & Phase)
        self.ax_val = self.fig.add_subplot(121)
        self.ax_val.set_title("Manifold Layer 1: Value Space (Complex Plane)")
        self.ax_val.set_xlabel("Real Magnitude")
        self.ax_val.set_ylabel("Imaginary (Phase)")
        self.ax_val.grid(True)
        self.ax_val.set_xlim(-RANGE*1.2, RANGE*1.2)
        self.ax_val.set_ylim(-RANGE*1.2, RANGE*1.2)
        
        # Subplot 2: 3D Unit Space (Mass, Length, Time)
        self.ax_units = self.fig.add_subplot(122, projection='3d')
        self.ax_units.set_title("Manifold Layer 2: SI Unit Space")
        self.ax_units.set_xlabel("Mass [M]")
        self.ax_units.set_ylabel("Length [L]")
        self.ax_units.set_zlabel("Time [T]")
        self.ax_units.set_xlim(-3, 3); self.ax_units.set_ylim(-3, 3); self.ax_units.set_zlim(-3, 3)

    def visualize_step(self, vec_a, vec_b, res_vec, target_vec, mode, gap, status):
        self.ax_val.clear(); self.ax_units.clear()
        self.ax_val.grid(True)
        self.ax_val.set_xlim(-RANGE/2, RANGE); self.ax_val.set_ylim(-RANGE/2, RANGE/2)
        self.ax_units.set_xlim(-3, 3); self.ax_units.set_ylim(-3, 3); self.ax_units.set_zlim(-3, 3)
        self.ax_units.set_xlabel("Mass"); self.ax_units.set_ylabel("Length"); self.ax_units.set_zlabel("Time")

        # --- PLOT 1: VALUE SPACE ---
        # Plot Integer Grid dots for reference
        grid_x = np.arange(-RANGE, RANGE, 5)
        self.ax_val.scatter(grid_x, np.zeros_like(grid_x), color='gray', alpha=0.2, s=10)

        # Plot Inputs
        self.ax_val.quiver(0, 0, vec_a[0], vec_a[1], angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.5, label="Input A")
        self.ax_val.quiver(0, 0, vec_b[0], vec_b[1], angles='xy', scale_units='xy', scale=1, color='cyan', alpha=0.5, label="Input B")
        
        # Plot Result Vector
        color = 'green' if status == "OK" else 'red'
        self.ax_val.quiver(0, 0, res_vec[0], res_vec[1], angles='xy', scale_units='xy', scale=1, color=color, linewidth=2, label=f"Result ({mode})")
        
        # Plot Gap (Line from Result to nearest Valid Integer)
        if target_vec is not None:
            self.ax_val.plot([res_vec[0], target_vec[0]], [res_vec[1], target_vec[1]], 'r--', linewidth=1, label=f"Gap: {gap:.2f}")
            self.ax_val.scatter(target_vec[0], target_vec[1], color='black', marker='x', s=50, label="Nearest Valid Concept")

        self.ax_val.legend(loc='upper left')
        self.ax_val.set_title(f"Reasoning Mode: {mode} | Status: {status}")

        # --- PLOT 2: UNIT SPACE ---
        # Plot the Unit coordinates as points in 3D space
        self.ax_units.scatter(vec_a[2], vec_a[3], vec_a[4], c='blue', marker='o', s=50, label="Unit A")
        self.ax_units.scatter(vec_b[2], vec_b[3], vec_b[4], c='cyan', marker='^', s=50, label="Unit B")
        self.ax_units.scatter(res_vec[2], res_vec[3], res_vec[4], c=color, marker='*', s=100, label="Result Unit")
        
        # Draw lines to origin to show them as vectors
        self.ax_units.plot([0, vec_a[2]], [0, vec_a[3]], [0, vec_a[4]], 'b-', alpha=0.3)
        self.ax_units.plot([0, vec_b[2]], [0, vec_b[3]], [0, vec_b[4]], 'c-', alpha=0.3)
        self.ax_units.plot([0, res_vec[2]], [0, res_vec[3]], [0, res_vec[4]], color=color, linewidth=2)

        if status == "DIMENSIONAL MISMATCH":
             self.ax_units.text(0, 0, 4, "MISMATCH!", color='red', fontsize=12, ha='center')

        self.ax_units.legend()
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        time.sleep(1.5) # Pause to let human see it

class SIManifold(nn.Module):
    def __init__(self):
        super().__init__()
        # The Scalar Crystal (Values only)
        self.scalar_crystal = nn.Embedding(VOCAB_SIZE, 2)
        # Initialize perfectly linear on real axis
        with torch.no_grad():
            for i in range(VOCAB_SIZE):
                self.scalar_crystal.weight[i] = torch.tensor([float(i - OFFSET), 0.0])
        self.scalar_crystal.weight.requires_grad = False

    def get_5d_vector(self, val, units):
        # units is tuple (M, L, T) e.g., (1, 0, 0) for kg
        if val + OFFSET < 0 or val + OFFSET >= VOCAB_SIZE:
             base = torch.tensor([9999.0, 9999.0]) # Void
        else:
             base = self.scalar_crystal(torch.tensor(int(val) + OFFSET))
        
        # Combine Scalar part with Unit part
        return torch.cat([base, torch.tensor([float(u) for u in units])])

    def execute_si(self, val_a, units_a, val_b, units_b, mode):
        vec_a = self.get_5d_vector(val_a, units_a)
        vec_b = self.get_5d_vector(val_b, units_b)
        
        res_vec = None
        status = "OK"
        gap = 0.0
        target_vec_2d = None
        
        if mode == 'LINEAR':
            # LINEAR PHYSICS: Units must match exactly.
            if units_a != units_b:
                res_vec = vec_a + vec_b # Show the bad addition anyway
                # The resulting unit vector will lie halfway between the valid units
                status = "DIMENSIONAL MISMATCH"
                gap = 999.0
            else:
                # Units match, add values.
                val_part = vec_a[:2] + vec_b[:2]
                unit_part = vec_a[2:] # Preserve unit
                res_vec = torch.cat([val_part, unit_part])

        elif mode == 'LOG':
            # LOG PHYSICS: Values Multiply, Units Add.
            # Complex Mult (Value)
            real = (vec_a[0]*vec_b[0]) - (vec_a[1]*vec_b[1])
            imag = (vec_a[0]*vec_b[1]) + (vec_a[1]*vec_b[0])
            # Unit Add (Vector Add in MLT space)
            new_unit = vec_a[2:] + vec_b[2:]
            res_vec = torch.cat([torch.stack([real, imag]), new_unit])

        # DECODE & GAP CHECK
        # 1. Identify resulting unit type
        res_units_tuple = tuple([int(round(x.item())) for x in res_vec[2:]])
        
        # 2. Find nearest integer value on the Real Axis
        target_val = round(res_vec[0].item())
        target_vec_2d = torch.tensor([float(target_val), 0.0])
        
        # 3. Calculate Gap (Distance from result value to nearest integer value)
        gap = torch.norm(res_vec[:2] - target_vec_2d).item()

        if gap > 0.01 and status == "OK": status = "GEOMETRIC GAP"
        
        return target_val, res_units_tuple, res_vec, target_vec_2d, gap, status

class ExecutiveMindSI:
    def __init__(self):
        self.manifold = SIManifold()
        self.viz = ReasoningVisualizer()
        
    def solve_step(self, v1, u1, v2, u2, hint):
        print(f"   Thinking: {v1}{u1} ? {v2}{u2} (Hint: {hint})")
        
        # Simple Router Bias based on hint
        current_mode = 'LOG' if any(x in hint for x in ['Product', 'Times', 'Square']) else 'LINEAR'
        
        for attempt in range(2):
            # Execute & Visualize
            val, units, res_vec, target_vec, gap, status = self.manifold.execute_si(v1, u1, v2, u2, current_mode)
            
            # Get input vectors just for visualization
            va_5d = self.manifold.get_5d_vector(v1, u1)
            vb_5d = self.manifold.get_5d_vector(v2, u2)
            self.viz.visualize_step(va_5d, vb_5d, res_vec, target_vec, current_mode, gap, status)
            
            if gap < 0.01 and status == "OK":
                print(f"      >> Convergence via {current_mode}: {val} Units{units}")
                return val, units
            
            print(f"      >> Dissonance ({status}). Switching Strategy...")
            current_mode = 'LOG' if current_mode == 'LINEAR' else 'LINEAR'
            
        print("      >> FAILURE. Reality Collapse.")
        return None, None

if __name__ == "__main__":
    ai = ExecutiveMindSI()
    
    # === HARDER PHYSICS SCENARIO: Kinetic Energy ===
    # Problem: An object with mass 4kg moves at velocity 3 m/s.
    # Calculate KE = 1/2 * m * v^2
    # Units: Mass=(1,0,0), Velocity=(0,1,-1) [L/T]
    # Target Unit: Energy = Mass * (L/T)^2 = M L^2 T^-2 = (1, 2, -2)
    
    print("=== TASK: Calculate Kinetic Energy (KE = 0.5 * m * v^2) ===")
    mass_val = 4; mass_unit = (1, 0, 0) # kg
    vel_val = 3;  vel_unit = (0, 1, -1) # m/s
    scalar_val = 0.5; scalar_unit = (0, 0, 0) # Pure number

    # Step 1: Calculate v^2 (Velocity times Velocity)
    print("\n[Step 1] Calculating v^2...")
    v2_val, v2_unit = ai.solve_step(vel_val, vel_unit, vel_val, vel_unit, "Square/Times")
    
    # Step 2: Calculate m * v^2
    print("\n[Step 2] Calculating m * v^2...")
    mv2_val, mv2_unit = ai.solve_step(mass_val, mass_unit, v2_val, v2_unit, "Times")

    # Step 3: Calculate 0.5 * (mv^2)
    # NOTE: Our current crystal only handles integers. 0.5 will cause a "Geometric Gap".
    # This is a perfect test to see the visualizer show the gap.
    print("\n[Step 3] Calculating 0.5 * Result...")
    ke_val, ke_unit = ai.solve_step(scalar_val, scalar_unit, mv2_val, mv2_unit, "Times")
    
    print(f"\n=== FINAL ANSWER: {ke_val} Joules (Units {ke_unit}) ===")
    print("(Note: The final answer rounded to integer because our crystal is currently integer-only.)")
    input("Press Enter to close visualization...")