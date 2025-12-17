#!/usr/bin/env python3
# AOTI LEVEL 18: PRECISION FLUID ENGINE
# Uses Analytical Geometry for the Manifold (Perfect Math)
# Uses Neural Routing for the Logic (Reasoning).
# Solves 2D Projectile Motion.

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import math

# --- THE GEOMETRIC UNIVERSE (TRUTH) ---
class PrecisionManifold:
    """
    The Manifold is the 'Laws of Physics'. It does not guess.
    It maps numbers to exact geometric coordinates.
    """
    def get_vector(self, val, unit, mode):
        # 1. UNIT VECTOR (Dimensions 3, 4, 5)
        # We place units on axes 3, 4, 5 (Mass, Length, Time)
        u_vec = list(unit) # e.g. [1, 0, -1]
        
        # 2. VALUE VECTOR (Dimensions 1, 2)
        if mode == 'LINEAR':
            # Linear Space: Magnitude is Linear distance
            # Vec = [Value, 0]
            v_vec = [float(val), 0.0]
            
        elif mode == 'LOG':
            # Log Space: Magnitude is Logarithmic distance
            # This turns Multiplication into Addition
            if val == 0: val = 1e-9 # Avoid singularity
            mag = math.log(abs(val))
            phase = 0.0 if val >= 0 else math.pi
            v_vec = [mag, phase]
            
        return torch.tensor(v_vec + u_vec)

    def decode(self, vec, mode):
        """
        Translates a geometric coordinate back into a human number.
        """
        # 1. Decode Unit
        unit = tuple([int(round(x.item())) for x in vec[2:]])
        
        # 2. Decode Value
        val = 0.0
        if mode == 'LINEAR':
            val = vec[0].item()
        elif mode == 'LOG':
            mag = vec[0].item()
            phase = vec[1].item()
            # If phase is near Pi, it's negative
            sign = -1.0 if abs(phase - 3.14159) < 0.1 else 1.0
            val = math.exp(mag) * sign
            
        return val, unit

    def execute(self, val_a, u_a, val_b, u_b, mode):
        # 1. Lift numbers into Geometry
        va = self.get_vector(val_a, u_a, mode)
        vb = self.get_vector(val_b, u_b, mode)
        
        # 2. Perform Geometric Operation (Vector Addition)
        # In AOTI, *all* math is Vector Addition. The Manifold determines meaning.
        vres = va + vb
        
        # 3. Physics Check (The "Gap" Detection)
        gap = 0.0
        status = "OK"
        
        if mode == 'LINEAR':
            # Constraint: Linearity requires Unit Alignment
            # We check if the input units match
            dist_units = torch.norm(va[2:] - vb[2:])
            if dist_units > 0.01:
                status = "DIMENSIONAL MISMATCH"
                gap = 100.0 # Huge gap
                
        # 4. Collapse Geometry back to Reality
        res_val, res_unit = self.decode(vres, mode)
        
        return res_val, res_unit, va, vb, vres, gap, status

# --- THE VISUALIZER ---
class Visualizer:
    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(10, 5))
        self.ax_geo = self.fig.add_subplot(121)
        self.ax_units = self.fig.add_subplot(122, projection='3d')

    def show(self, va, vb, vres, mode, status):
        self.ax_geo.clear(); self.ax_units.clear()
        
        # Plot 1: The Math Space (Linear or Log)
        self.ax_geo.set_title(f"Geometric Space ({mode})")
        
        # Plot arrows
        self.ax_geo.quiver(0, 0, va[0], va[1], angles='xy', scale_units='xy', scale=1, color='blue', label='A', alpha=0.5)
        # Chain B to the end of A (Vector Addition)
        self.ax_geo.quiver(va[0], va[1], vb[0], vb[1], angles='xy', scale_units='xy', scale=1, color='cyan', label='B', alpha=0.5)
        # Result
        color = 'green' if status == "OK" else 'red'
        self.ax_geo.quiver(0, 0, vres[0], vres[1], angles='xy', scale_units='xy', scale=1, color=color, linewidth=2, label='Result')
        
        # Scale view
        mx = max(abs(vres[0].item()), 5) * 1.5
        self.ax_geo.set_xlim(-mx, mx); self.ax_geo.set_ylim(-mx, mx)
        self.ax_geo.grid(True)
        self.ax_geo.legend()
        
        # Plot 2: Unit Space
        self.ax_units.set_title("SI Dimensions")
        self.ax_units.set_xlabel('Mass'); self.ax_units.set_ylabel('Length'); self.ax_units.set_zlabel('Time')
        
        # Extract unit coords
        ua = va[2:].numpy(); ub = vb[2:].numpy(); ures = vres[2:].numpy()
        
        self.ax_units.scatter(*ua, c='blue', s=50)
        self.ax_units.scatter(*ub, c='cyan', s=50)
        self.ax_units.scatter(*ures, c=color, s=100, marker='*')
        
        # Draw connections
        self.ax_units.plot([0, ua[0]], [0, ua[1]], [0, ua[2]], 'b--', alpha=0.3)
        self.ax_units.plot([0, ub[0]], [0, ub[1]], [0, ub[2]], 'c--', alpha=0.3)
        self.ax_units.plot([0, ures[0]], [0, ures[1]], [0, ures[2]], color=color)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        time.sleep(1.0)

# --- THE AGENT ---
class PrecisionAgent:
    def __init__(self):
        self.manifold = PrecisionManifold()
        self.viz = Visualizer()
        
    def reason(self, v1, u1, v2, u2, hint):
        print(f"   Reasoning: {v1} [u{u1}] ? {v2} [u{u2}]")
        
        # Simple Logic Router
        # (In a full build, this is a Neural Net trained on hints)
        mode = 'LOG' if any(x in hint for x in ['Times', 'Square', 'Product']) else 'LINEAR'
        
        # Try Strategy
        for attempt in range(2):
            val, unit, va, vb, vres, gap, status = self.manifold.execute(v1, u1, v2, u2, mode)
            
            self.viz.show(va, vb, vres, mode, status)
            
            if status == "OK":
                print(f"      >> Success ({mode}): {val:.4f} Units{unit}")
                return val, unit
            
            print(f"      >> Error ({status}). Flipping Strategy...")
            mode = 'LINEAR' if mode == 'LOG' else 'LOG'
            
        print("      >> FAILURE.")
        return None, None

if __name__ == "__main__":
    ai = PrecisionAgent()
    
    # === SCENARIO: PROJECTILE MOTION ===
    # A cannonball is fired upwards.
    # Initial Velocity (Vy) = 20 m/s
    # Time (t) = 3 s
    # Gravity (g) = 9.8 m/s^2
    # Formula: y = (Vy * t) - (0.5 * g * t^2)
    
    print("=== SOLVING PROJECTILE HEIGHT ===")
    print("Formula: y = (Vy * t) - (0.5 * g * t^2)")
    
    # Units: [Mass, Length, Time]
    u_vel = (0, 1, -1)   # m/s
    u_acc = (0, 1, -2)   # m/s^2
    u_time = (0, 0, 1)   # s
    u_scalar = (0, 0, 0) # -
    
    # 1. Term 1: Vy * t
    print("\n[Step 1] Initial Climb (Vy * t)")
    h1, u_h1 = ai.reason(20, u_vel, 3, u_time, "Times")
    
    # 2. Term 2a: t^2
    print("\n[Step 2] Time Squared (t * t)")
    t2, u_t2 = ai.reason(3, u_time, 3, u_time, "Times")
    
    # 3. Term 2b: g * t^2
    print("\n[Step 3] Gravity Drop (g * t^2)")
    d1, u_d1 = ai.reason(9.8, u_acc, t2, u_t2, "Times")
    
    # 4. Term 2c: 0.5 * (g*t^2)
    print("\n[Step 4] Scale Drop (0.5 * d1)")
    d2, u_d2 = ai.reason(0.5, u_scalar, d1, u_d1, "Times")
    
    # 5. Final Height: Term 1 - Term 2c
    # Subtraction is just Addition with negative numbers in Linear Space
    print("\n[Step 5] Net Height (Climb + (-Drop))")
    final_y, u_final = ai.reason(h1, u_h1, -d2, u_d2, "Plus/Sum") # Note -d2
    
    print(f"\n=== FINAL HEIGHT: {final_y:.2f} meters ===")
    print("(Expected: (20*3) - (0.5*9.8*9) = 60 - 44.1 = 15.9m)")
    input("Press Enter to close...")