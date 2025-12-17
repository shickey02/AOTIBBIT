#!/usr/bin/env python3
# BBIT UNIVERSAL AGENT FRAMEWORK
# A template for solving any problem by treating it as Vector Navigation.

import numpy as np

class UniversalAgent:
    def __init__(self, target_vector):
        self.target = target_vector
        self.state = None
        
        # THE APEX PERSONALITY (The "Ghost in the Machine")
        # These are the params that matter across ALL domains.
        self.params = {
            'precision': 0.1,   # Base Learning Rate
            'obsession': 5.0,   # How much we care about the gap
            'risk': -0.5,       # Negative = Allow Chaos/Noise
            'patience': 10      # How long we tolerate stagnation
        }
        self.stagnation_counter = 0
        self.last_dist = float('inf')

    def sense(self, environment_data):
        """
        Input: Raw Data (Image, Text, Stock Price)
        Output: Latent Vector (The "Meaning")
        """
        # TODO: Plug in your Encoder here (ResNet, BERT, Autoencoder)
        # return model.encode(environment_data)
        return np.array(environment_data) # Placeholder

    def think(self, current_vector):
        """
        Calculates the Vector Gap and modulates personality.
        """
        gap = self.target - current_vector
        dist = np.linalg.norm(gap)
        
        # Cybernetic Regulation (Stafford Beer Style)
        # If we are improving, stabilize. If we are stuck, destabilize.
        if dist >= self.last_dist:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
            
        self.last_dist = dist
        
        # Dynamic Chaos Injection
        current_chaos = 0.0
        if self.stagnation_counter > self.params['patience']:
            # We are stuck. Inject massive noise (The "Wobble")
            current_chaos = abs(self.params['risk']) * (self.stagnation_counter / 5.0)
            
        return gap, dist, current_chaos

    def act(self, gap, chaos_level):
        """
        Converts the mental 'Gap' into a physical 'Action'.
        """
        # 1. Calculate ideal move
        move_vector = gap * self.params['precision']
        
        # 2. Add Chaos (The Spark)
        if chaos_level > 0:
            noise = np.random.normal(0, chaos_level, size=gap.shape)
            move_vector += noise
            
        return move_vector

# --- USAGE EXAMPLE ---
def solve_problem(start_state, target_state):
    agent = UniversalAgent(target_state)
    curr = start_state
    
    print(f"Goal: {target_state}")
    
    for t in range(50):
        # 1. Sense
        vec = agent.sense(curr)
        
        # 2. Think
        gap, dist, chaos = agent.think(vec)
        
        # 3. Act
        action = agent.act(gap, chaos)
        
        # 4. Environment Feedback (Simulation)
        curr = curr + action
        
        print(f"T{t}: Dist {dist:.4f} | Chaos {chaos:.2f}")
        if dist < 0.01:
            print(">> SOLVED.")
            break

if __name__ == "__main__":
    # Example: Navigating a 10-dimensional hypercube
    solve_problem(np.zeros(10), np.ones(10))