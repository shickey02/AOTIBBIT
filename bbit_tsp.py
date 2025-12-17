#!/usr/bin/env python3
# BBIT OPTIMIZER: The Traveling Salesman
# Solves TSP using Cybernetic Regulation (Precision vs. Chaos).

import random, math, time, copy
import numpy as np

# --- 1. THE ENVIRONMENT (The World) ---
class CityMap:
    def __init__(self, num_cities=20):
        self.cities = []
        # Generate random cities on a 100x100 grid
        for i in range(num_cities):
            self.cities.append((random.uniform(0, 100), random.uniform(0, 100)))
            
    def get_distance(self, route):
        """Calculates total distance of the path."""
        dist = 0
        for i in range(len(route)):
            c1 = self.cities[route[i]]
            c2 = self.cities[route[(i+1) % len(route)]] # Loop back to start
            dist += math.hypot(c1[0]-c2[0], c1[1]-c2[1])
        return dist

# --- 2. THE APEX AGENT (The Mind) ---
class ApexOptimizer:
    def __init__(self, num_cities):
        self.route = list(range(num_cities)) # Initial random order
        random.shuffle(self.route)
        
        self.best_route = list(self.route)
        self.best_score = float('inf')
        
        # CYBERNETIC STATE
        self.stagnation = 0
        self.patience = 50      # How long before we panic?
        self.chaos_level = 0.0  # Current "Negative Risk"
        self.learning_rate = 1  # How many swaps we do

    def think(self, current_score):
        """
        Feedback Loop: Monitors progress and regulates Chaos.
        """
        # 1. Check for improvement (Closing the Gap)
        if current_score < self.best_score:
            self.best_score = current_score
            self.best_route = list(self.route)
            self.stagnation = 0 # Relief! Reset anxiety.
            self.chaos_level = 0.0
            print(f"   >>> NEW RECORD: {current_score:.2f}")
        else:
            self.stagnation += 1
            
        # 2. Regulate Anxiety (The Algedonic Loop)
        if self.stagnation > self.patience:
            # We are stuck. RAMP UP CHAOS.
            # Chaos scales with how long we've been stuck.
            self.chaos_level = min(0.8, (self.stagnation - self.patience) / 100.0)
        else:
            self.chaos_level = 0.0
            
        return self.chaos_level

    def act(self):
        """
        Executes a move based on current Chaos Level.
        """
        new_route = list(self.route)
        
        # MODE A: PRECISION (Chaos = 0)
        # Gentle optimization. Swap two cities.
        if self.chaos_level < 0.1:
            idx1, idx2 = random.sample(range(len(new_route)), 2)
            new_route[idx1], new_route[idx2] = new_route[idx2], new_route[idx1]
            return new_route, "Precision"

        # MODE B: PANIC (Chaos > 0.1)
        # Massive structural change. Reverse or Scramble sub-sections.
        # "I am failing! Break the pattern!"
        else:
            # Scramble a chunk of the route proportional to Chaos
            chunk_size = int(len(new_route) * self.chaos_level)
            chunk_size = max(2, chunk_size)
            
            start = random.randint(0, len(new_route) - chunk_size)
            sub = new_route[start : start+chunk_size]
            random.shuffle(sub) # Total chaos in this sector
            new_route[start : start+chunk_size] = sub
            
            return new_route, f"CHAOS ({self.chaos_level:.2f})"

    def run_cycle(self, world_map):
        # 1. Sense (Current Distance)
        current_dist = world_map.get_distance(self.route)
        
        # 2. Think (Adjust Anxiety)
        chaos = self.think(current_dist)
        
        # 3. Act (Generate Candidate)
        candidate_route, mode = self.act()
        
        # 4. Evaluate (Did the action help?)
        cand_dist = world_map.get_distance(candidate_route)
        
        # DECISION LOGIC:
        # If High Chaos (Panic), we might accept a WORSE state just to move.
        # This is "Simulated Annealing" emergent from Anxiety.
        
        gap = cand_dist - current_dist
        
        if gap < 0:
            # Improvement! Always accept.
            self.route = candidate_route
            return current_dist, mode
            
        elif chaos > 0:
            # We are panicking. We accept bad moves to escape local optima.
            # Probability of accepting bad move is proportional to Chaos.
            acceptance_prob = chaos * 0.1 # 10% chance at max chaos
            if random.random() < acceptance_prob:
                self.route = candidate_route
                return cand_dist, f"{mode} [BAD MOVE ACCEPTED]"
                
        return current_dist, mode

# --- 3. MAIN LOOP ---
def main():
    print("--- BBIT TRAVELING SALESMAN ---")
    print("Optimization via Cybernetic Regulation")
    
    # Setup
    world = CityMap(num_cities=30)
    agent = ApexOptimizer(num_cities=30)
    
    initial_dist = world.get_distance(agent.route)
    print(f"Start Distance: {initial_dist:.2f}")
    print("-" * 40)
    
    try:
        for t in range(5000):
            dist, mode = agent.run_cycle(world)
            
            # Visualization
            if t % 50 == 0:
                bar = "#" * int(agent.chaos_level * 20)
                status = "STABLE" if agent.chaos_level == 0 else "PANIC "
                print(f"Step {t:04d} | Dist: {dist:.1f} | State: {status} [{bar:<20}] | Mode: {mode}")
                
            if dist < 300: # Arbitrary goal
                print(f"\n>> CONVERGENCE. Final Distance: {dist:.2f}")
                break
                
    except KeyboardInterrupt:
        pass
        
    print("-" * 40)
    print(f"Best Distance Achieved: {agent.best_score:.2f}")
    improvement = initial_dist - agent.best_score
    print(f"Total Improvement: {improvement:.2f} units")

if __name__ == "__main__":
    main()