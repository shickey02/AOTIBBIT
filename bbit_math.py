#!/usr/bin/env python3
# BBIT SCIENTIST: Symbolic Regression
# Discovers mathematical formulas from raw data using Cybernetic Evolution.

import random, math, copy, time
import numpy as np

# --- 1. THE MYSTERY (The Environment) ---
class PhysicsWorld:
    def __init__(self):
        # The Secret Law: y = 2x^2 - 5x + 3
        self.secret_formula = lambda x: 2*(x**2) - 5*x + 3
        
        # Generate Data Points (Observation)
        self.X = np.linspace(-10, 10, 20)
        self.Y = self.secret_formula(self.X)
        
    def evaluate(self, formula_func):
        try:
            preds = formula_func(self.X)
            # Mean Squared Error
            loss = np.mean((self.Y - preds)**2)
            if np.isnan(loss) or np.isinf(loss): return float('inf')
            return loss
        except:
            return float('inf')

# --- 2. THE GENOME (Expression Tree) ---
# We represent math as a list of tokens: ['x', '2', '^']
OPS = ['+', '-', '*', 'x'] # Simplified op set for speed

class FormulaAgent:
    def __init__(self):
        # Start with random guess: "x + x"
        self.genome = ['x', '+', 'x']
        
        self.best_genome = list(self.genome)
        self.best_loss = float('inf')
        
        # Cybernetics
        self.stagnation = 0
        self.chaos_level = 0.0

    def compile(self, genome):
        # Convert list ['x', '+', '2'] into a lambda function
        expr = "".join(str(g) for g in genome)
        try:
            # Safety wrapper for eval
            return lambda x: eval(expr, {"x": x, "np": np, "sin": np.sin})
        except:
            return lambda x: x * float('inf') # Bad syntax penalty

    def act(self):
        candidate = list(self.genome)
        
        # MODE A: PRECISION (Tune Constants)
        if self.chaos_level < 0.2:
            # Find a number and tweak it
            for i in range(len(candidate)):
                if isinstance(candidate[i], (int, float)):
                    candidate[i] += random.gauss(0, 0.5) # Nudge
                    return candidate, "Precision"
            # If no numbers, switch to structure mode
        
        # MODE B: STRUCTURE (Change Ops)
        if self.chaos_level < 0.5:
            idx = random.randint(0, len(candidate)-1)
            if candidate[idx] in OPS:
                candidate[idx] = random.choice(OPS)
            elif isinstance(candidate[idx], (int, float)):
                candidate[idx] = random.randint(1, 9)
            return candidate, "Structure"

        # MODE C: CHAOS (Grow/Shrink Tree)
        # "I am failing! Add complexity!"
        action = random.choice(['grow', 'shrink', 'scramble'])
        if action == 'grow':
            op = random.choice(['+', '-', '*'])
            val = random.choice(['x', random.randint(1,5)])
            candidate.extend([op, val])
        elif action == 'shrink' and len(candidate) > 3:
            candidate = candidate[:-2]
        elif action == 'scramble':
            random.shuffle(candidate) # Dangerous!
            
        return candidate, f"CHAOS ({self.chaos_level:.1f})"

    def run_cycle(self, world):
        # Evaluate Current
        func = self.compile(self.genome)
        loss = world.evaluate(func)
        
        # Think (Regulate)
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_genome = list(self.genome)
            self.stagnation = 0
            self.chaos_level = 0.0
            print(f"   >>> NEW THEORY: {' '.join(str(x) for x in self.genome)} (Error: {loss:.4f})")
        else:
            self.stagnation += 1
            if self.stagnation > 50:
                self.chaos_level = min(0.9, (self.stagnation - 50)/100.0)
                
        # Act
        new_genome, mode = self.act()
        
        # Evaluate Candidate
        new_func = self.compile(new_genome)
        new_loss = world.evaluate(new_func)
        
        # Selection Logic
        if new_loss < loss:
            self.genome = new_genome # Accept improvement
        elif self.chaos_level > 0 and random.random() < self.chaos_level * 0.1:
            self.genome = new_genome # Accept bad theory to explore
            
        return loss, self.chaos_level

def main():
    print("--- BBIT SCIENTIST ---")
    world = PhysicsWorld()
    agent = FormulaAgent()
    
    # Cheat start structure to ensure valid syntax for demo speed
    # Start: x * x (x^2)
    agent.genome = ['x', '*', 'x'] 
    
    for t in range(5000):
        loss, chaos = agent.run_cycle(world)
        if t % 500 == 0:
            print(f"Step {t} | Error: {loss:.2f} | Chaos: {chaos:.2f}")
        if loss < 0.1:
            print(f"\n>> EUREKA! Law Discovered: {' '.join(str(x) for x in agent.genome)}")
            break

if __name__ == "__main__":
    main()