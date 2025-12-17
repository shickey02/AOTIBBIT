#!/usr/bin/env python3
# AOTI LEVEL 20: THE PREDICTIVE LANDER (Inverse Physics)
# The AI "hallucinates" future trajectories to control a rocket.
# Solves for Thrust geometrically to minimize Impact Velocity.

import torch
import numpy as np
import matplotlib.pyplot as plt
import time

# --- THE GEOMETRIC MIND (Recycled from Orbital Sim) ---
class GeometricMind:
    def add_vectors(self, v1, v2):
        # Linear addition in our precision manifold
        return [v1[0]+v2[0], v1[1]+v2[1]]
    
    def scale_vector(self, v, s):
        return [v[0]*s, v[1]*s]

    def predict_state(self, pos, vel, thrust, gravity, dt, steps=20):
        """
        Hallucinates the future.
        Returns the trajectory of the 'Ghost Rocket'.
        """
        ghost_pos = list(pos)
        ghost_vel = list(vel)
        trajectory = []
        
        # Total Acceleration = Gravity + (Thrust / Mass)
        # We assume Mass = 1 for simplicity
        # Accel Vector = [0, -9.8] + [0, thrust]
        total_accel = [0, -gravity + thrust]
        
        for _ in range(steps):
            # v_new = v + a*dt
            dv = self.scale_vector(total_accel, dt)
            ghost_vel = self.add_vectors(ghost_vel, dv)
            
            # p_new = p + v*dt
            dp = self.scale_vector(ghost_vel, dt)
            ghost_pos = self.add_vectors(ghost_pos, dp)
            
            trajectory.append(ghost_pos)
            
            # Stop hallucinating if we hit the ground
            if ghost_pos[1] <= 0:
                break
                
        return trajectory, ghost_pos, ghost_vel

# --- THE ROCKET AGENT ---
class RocketAI:
    def __init__(self):
        self.mind = GeometricMind()
        self.gravity = 9.8
        self.dt = 0.1
        self.max_thrust = 20.0
        
    def decide_thrust(self, pos, vel):
        """
        The Executive Loop.
        Instead of 'solving' an equation, it probes the geometry.
        """
        best_thrust = 0.0
        min_crash_speed = float('inf')
        best_ghost_traj = []
        
        # Geometric Sweep: Try different thrust vectors
        # In a real neural net, this would be Gradient Descent on the 'Thrust Manifold'
        options = np.linspace(0, self.max_thrust, 20) # Test 20 distinct thrust levels
        
        for thrust in options:
            # 1. Hallucinate Future
            traj, end_pos, end_vel = self.mind.predict_state(pos, vel, thrust, self.gravity, self.dt, steps=30)
            
            # 2. Analyze Outcome
            # We want End Height approx 0
            # We want End Velocity approx 0
            
            # If the ghost didn't touch ground in prediction window, it's hovering too high?
            # Or if it's going up, that's bad (unless we are too low).
            
            impact_vel = end_vel[1]
            final_h = end_pos[1]
            
            # Heuristic Cost Function (The "Gap")
            # We want impact_vel to be close to 0 (but negative is okay if small)
            # We penalize high impact speed OR flying away
            
            cost = 0
            if final_h > 0.5: 
                # Hovering too high, not landing. Cost is distance from ground.
                cost = final_h * 10
            elif final_h < 0:
                # Crashed. Cost is impact velocity squared.
                cost = (impact_vel ** 2)
            else:
                # Perfect height. Cost is velocity.
                cost = abs(impact_vel)
            
            if cost < min_crash_speed:
                min_crash_speed = cost
                best_thrust = thrust
                best_ghost_traj = traj
                
        return best_thrust, best_ghost_traj

# --- SIMULATION VISUALIZER ---
def run_simulation():
    ai = RocketAI()
    
    # Initial State
    pos = [0.0, 100.0] # 100 meters up
    vel = [0.0, 0.0]   # Stationary
    fuel = 100.0
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 8))
    
    print("--- INITIATING LANDING SEQUENCE ---")
    print("Green Line = AI Prediction (The Ghost)")
    print("Blue Dot   = Real Rocket")
    
    frame = 0
    landed = False
    
    while not landed and frame < 300:
        # 1. AI THINKS
        thrust, ghost_traj = ai.decide_thrust(pos, vel)
        
        # Burn fuel
        if fuel <= 0: thrust = 0
        fuel -= thrust * 0.05
        
        # 2. PHYSICS UPDATE (Reality)
        # Accel = Gravity + Thrust
        accel_y = -ai.gravity + thrust
        vel[1] += accel_y * ai.dt
        pos[1] += vel[1] * ai.dt
        
        # 3. VISUALIZE
        ax.clear()
        ax.set_xlim(-10, 10); ax.set_ylim(-5, 120)
        ax.set_aspect('equal')
        
        # Draw Ground
        ax.axhline(0, color='black', linewidth=3)
        
        # Draw Ghost Trajectory (What the AI is thinking)
        if len(ghost_traj) > 0:
            gx, gy = zip(*ghost_traj)
            ax.plot(gx, gy, 'g--', alpha=0.6, label='AI Prediction')
        
        # Draw Real Rocket
        ax.scatter(pos[0], pos[1], s=100, c='blue', label='Rocket')
        
        # Draw Thrust Vector
        ax.arrow(pos[0], pos[1], 0, -thrust, color='orange', width=0.5, label='Thrust')
        
        # Info
        info = f"Alt: {pos[1]:.1f}m | Vel: {vel[1]:.1f}m/s | Fuel: {int(fuel)}%"
        ax.text(-9, 110, info, fontsize=12)
        ax.set_title(f"AOTI Lander (Frame {frame})")
        ax.legend(loc='upper right')
        
        plt.pause(0.01)
        frame += 1
        
        # Check Landing
        if pos[1] <= 0:
            landed = True
            final_v = vel[1]
            print(f"\nTOUCHDOWN.")
            print(f"Impact Velocity: {final_v:.2f} m/s")
            
            if final_v > -5.0:
                print(">> SUCCESS: Soft Landing.")
                ax.text(-5, 50, "SUCCESS", color='green', fontsize=20, fontweight='bold')
            else:
                print(">> FAILURE: Crash.")
                ax.text(-5, 50, "CRASH", color='red', fontsize=20, fontweight='bold')
            
            plt.pause(2.0)

    plt.show()

if __name__ == "__main__":
    run_simulation()