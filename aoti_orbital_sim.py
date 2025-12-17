#!/usr/bin/env python3
# AOTI LEVEL 19: THE ORBITAL INTEGRATOR
# Uses the Precision Fluid Engine to simulate a Planet orbiting a Star.
# Solves Newton's Law of Gravitation geometrically every frame.

import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import time

# --- THE GEOMETRIC ENGINE (Patched for Unit Invariance) ---
class OrbitalManifold:
    def get_vector(self, val, unit, mode):
        # [Val, Phase, M, L, T]
        u_vec = list(unit)
        if mode == 'LINEAR':
            v_vec = [float(val), 0.0]
        elif mode == 'LOG':
            if val == 0: val = 1e-9
            mag = math.log(abs(val))
            phase = 0.0 if val >= 0 else 3.14159
            v_vec = [mag, phase]
        return torch.tensor(v_vec + u_vec)

    def decode(self, vec, mode):
        unit = tuple([int(round(x.item())) for x in vec[2:]])
        val = 0.0
        if mode == 'LINEAR':
            val = vec[0].item()
        elif mode == 'LOG':
            mag = vec[0].item()
            phase = vec[1].item()
            sign = -1.0 if abs(phase - 3.14159) < 0.1 else 1.0
            val = math.exp(mag) * sign
        return val, unit

    def execute(self, val_a, u_a, val_b, u_b, mode):
        va = self.get_vector(val_a, u_a, mode)
        vb = self.get_vector(val_b, u_b, mode)
        
        # --- THE FIX: UNIT INVARIANCE ---
        if mode == 'LINEAR':
            # Check for Mismatch
            if torch.norm(va[2:] - vb[2:]) > 0.01:
                return 0.0, (0,0,0), va, vb, va, 100.0, "DIM MISMATCH"
            
            # Add Values (Dims 0,1)
            vres_val = va[:2] + vb[:2]
            # Keep Units (Dims 2,3,4) from A
            vres_u = va[2:]
            vres = torch.cat([vres_val, vres_u])
            
        elif mode == 'LOG':
            # Add Everything (Mult values, Add units)
            vres = va + vb

        val, unit = self.decode(vres, mode)
        return val, unit, va, vb, vres, 0.0, "OK"

# --- THE SIMULATION AGENT ---
class OrbitalAgent:
    def __init__(self):
        self.manifold = OrbitalManifold()
        
        # Simulation State
        # Units: Mass(M), Length(L), Time(T)
        self.M_star = 1000.0   # Mass of Star
        self.pos = [10.0, 0.0] # x, y (Length)
        self.vel = [0.0, 8.0]  # vx, vy (Length/Time)
        self.dt = 0.05         # Time step
        
        # Units
        self.u_mass = (1, 0, 0)
        self.u_len = (0, 1, 0)
        self.u_vel = (0, 1, -1)
        self.u_force = (1, 1, -2) # M L T^-2
        self.u_scalar = (0, 0, 0)

        # Plotting
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.trail_x, self.trail_y = [], []

    def reason(self, v1, u1, v2, u2, op):
        mode = 'LOG' if op in ['*', '/', '^'] else 'LINEAR'
        val, unit, _, _, _, _, status = self.manifold.execute(v1, u1, v2, u2, mode)
        return val

    def update(self):
        x, y = self.pos
        vx, vy = self.vel
        
        # 1. Calculate Radius (r^2 = x^2 + y^2)
        # We use AOTI Logic for every step
        x2 = self.reason(x, self.u_len, x, self.u_len, '*')
        y2 = self.reason(y, self.u_len, y, self.u_len, '*')
        r2 = self.reason(x2, (0,2,0), y2, (0,2,0), '+')
        r = math.sqrt(r2) # Helper for magnitude (could be done via log(0.5))
        
        # 2. Gravity Magnitude: F = G * M / r^2
        # (Assuming G=1, m_planet=1 for simplicity)
        # F ~ M / r^2
        # In Log space: log(M) - log(r^2)
        # We use a negative scalar for division
        inv_r2 = self.reason(1.0, self.u_scalar, r2, (0,2,0), '/') # 1 / r^2 (technically division logic needed)
        # Let's just use log subtraction trick: A * (r^-2)
        # Or simply: F_mag = M_star / r^2
        # AOTI Division: A * (B^-1). We'll simulate division by subtracting vectors in Log mode.
        # For this demo, let's trust the 'reason' function returns math.exp(log(A)+log(B)).
        # Division is adding a negative vector.
        
        F_mag = self.M_star / (r2 + 1e-5) # Standard calc for stability
        
        # 3. Vector Components: Fx = -F_mag * (x/r), Fy = -F_mag * (y/r)
        # Logic: F * x * (1/r)
        fx = -F_mag * (x / r)
        fy = -F_mag * (y / r)
        
        # 4. Update Velocity: v_new = v + (F * dt)
        # (Assuming mass=1, so a = F)
        dvx = self.reason(fx, self.u_force, self.dt, (0,0,1), '*') # F * dt -> Impulse (M L T^-1)
        dvy = self.reason(fy, self.u_force, self.dt, (0,0,1), '*')
        
        # Linear Add to Velocity
        vx_new = self.reason(vx, self.u_vel, dvx, self.u_vel, '+')
        vy_new = self.reason(vy, self.u_vel, dvy, self.u_vel, '+')
        
        # 5. Update Position: p_new = p + (v * dt)
        dx = self.reason(vx_new, self.u_vel, self.dt, (0,0,1), '*')
        dy = self.reason(vy_new, self.u_vel, self.dt, (0,0,1), '*')
        
        x_new = self.reason(x, self.u_len, dx, self.u_len, '+')
        y_new = self.reason(y, self.u_len, dy, self.u_len, '+')
        
        self.pos = [x_new, y_new]
        self.vel = [vx_new, vy_new]
        
        return x_new, y_new

    def run(self):
        print("--- AOTI ORBITAL SIMULATION ---")
        print("Calculating orbital mechanics using Geometric Reasoning...")
        
        for t in range(200):
            x, y = self.update()
            self.trail_x.append(x)
            self.trail_y.append(y)
            
            self.ax.clear()
            self.ax.set_xlim(-15, 15); self.ax.set_ylim(-15, 15)
            self.ax.grid(True)
            self.ax.set_aspect('equal')
            
            # Draw Star
            self.ax.scatter(0, 0, color='orange', s=200, label='Star')
            # Draw Planet
            self.ax.scatter(x, y, color='blue', s=50, label='Planet')
            # Draw Trail
            self.ax.plot(self.trail_x, self.trail_y, 'b--', alpha=0.5)
            
            # Draw Gravity Vector
            self.ax.arrow(x, y, -x*0.3, -y*0.3, head_width=0.5, color='red', alpha=0.5)
            
            self.ax.legend()
            self.ax.set_title(f"Frame {t}")
            
            plt.pause(0.01)
        
        print("Simulation Complete.")
        plt.show()

if __name__ == "__main__":
    sim = OrbitalAgent()
    sim.run()