#!/usr/bin/env python3
# BBIT SCIENTIST v2 (Robust Symbolic Regression)
# Discovers mathematical formulas using Structured Grammatical Evolution.

import random, math, copy, time
import numpy as np
import warnings

# Suppress runtime warnings (overflows, etc.)
warnings.filterwarnings('ignore')

# --- 1. THE PHYSICS WORLD ---
class PhysicsWorld:
    def __init__(self):
        # The Secret Law: y = x^2 - 3x + 5
        # Simple enough to learn quickly, complex enough to require structure.
        self.secret_formula = lambda x: (x**2) - (3*x) + 5
        
        # Generate Data Points
        self.X = np.linspace(-5, 5, 20)
        self.Y = self.secret_formula(self.X)
        
    def evaluate(self, func_str):
        try:
            # Safe eval wrapper
            f = lambda x: eval(func_str, {"x": x, "np": np})
            preds = f(self.X)
            
            # Check for validity
            if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                return float('inf')
                
            loss = np.mean((self.Y - preds)**2)
            return loss
        except:
            return float('inf')

# --- 2. THE GRAMMAR AGENT ---
class ScientistAgent:
    def __init__(self):
        # Genome is alternating [Term, Op, Term, Op...]
        # Start simple: "x * x"
        self.terms = ['x', 'x']
        self.ops = ['*']
        
        self.best_loss = float('inf')
        self.best_structure = (list(self.terms), list(self.ops))
        
        self.stagnation = 0
        self.chaos = 0.0

    def get_formula(self):
        # Interleave terms and ops: term op term op term...
        expr = str(self.terms[0])
        for i in range(len(self.ops)):
            expr += f" {self.ops[i]} {self.terms[i+1]}"
        return expr

    def think(self, current_loss):
        # Regulation Loop
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_structure = (list(self.terms), list(self.ops))
            self.stagnation = 0
            self.chaos = 0.0
            print(f"   >>> NEW THEORY: y = {self.get_formula()} (Error: {current_loss:.4f})")
            return True
        
        self.stagnation += 1
        if self.stagnation > 20:
            self.chaos = min(0.5, (self.stagnation - 20) / 100.0)
            
        return False

    def act(self):
        # Create candidate copies
        new_terms = list(self.terms)
        new_ops = list(self.ops)
        
        # MODE A: TUNING (Adjust Constants)
        if self.chaos < 0.1:
            # Pick a numeric term and nudge it
            numeric_indices = [i for i, t in enumerate(new_terms) if isinstance(t, (int, float))]
            if numeric_indices:
                idx = random.choice(numeric_indices)
                new_terms[idx] += random.gauss(0, 0.5)
            else:
                # No numbers? Switch an 'x' to a number
                idx = random.randint(0, len(new_terms)-1)
                new_terms[idx] = random.randint(1, 5)
                
        # MODE B: STRUCTURE (Mutate Operators)
        elif self.chaos < 0.3:
            idx = random.randint(0, len(new_ops)-1)
            new_ops[idx] = random.choice(['+', '-', '*'])
            
        # MODE C: COMPLEXITY (Grow/Shrink)
        else:
            if random.random() < 0.5:
                # Grow: Add [Op, Term]
                new_ops.append(random.choice(['+', '-', '*']))
                new_terms.append(random.choice(['x', random.randint(1, 5)]))
            elif len(new_ops) > 0:
                # Shrink: Remove last [Op, Term]
                new_ops.pop()
                new_terms.pop()
                
        return new_terms, new_ops

    def run_cycle(self, world):
        # 1. Current State
        curr_loss = world.evaluate(self.get_formula())
        
        # 2. Panic Check (Did we break physics?)
        if curr_loss == float('inf'):
            # Revert to last known good theory
            self.terms, self.ops = copy.deepcopy(self.best_structure)
            return float('inf'), self.chaos
            
        # 3. Think
        self.think(curr_loss)
        
        # 4. Act
        cand_terms, cand_ops = self.act()
        
        # 5. Tentative Evaluate
        # Build candidate string manually to check
        expr = str(cand_terms[0])
        for i in range(len(cand_ops)):
            expr += f" {cand_ops[i]} {cand_terms[i+1]}"
            
        cand_loss = world.evaluate(expr)
        
        # 6. Decision
        if cand_loss < curr_loss:
            # Improvement: Accept
            self.terms, self.ops = cand_terms, cand_ops
        elif self.chaos > 0 and cand_loss != float('inf'):
            # Chaos: Accept worse (but valid) theory to explore
            if random.random() < self.chaos * 0.2:
                self.terms, self.ops = cand_terms, cand_ops
                
        return curr_loss, self.chaos

def main():
    print("--- BBIT SCIENTIST v2 (Structured) ---")
    print("Target: y = x^2 - 3x + 5")
    
    world = PhysicsWorld()
    agent = ScientistAgent()
    
    # Seed with basic structure
    agent.terms = ['x', 'x'] 
    agent.ops = ['*']
    
    start_time = time.time()
    
    for t in range(1, 5001):
        loss, chaos = agent.run_cycle(world)
        
        if t % 500 == 0:
            print(f"Step {t} | Error: {loss:.4f} | Chaos: {chaos:.2f}")
            
        if loss < 0.1:
            print(f"\n>> EUREKA! Law Discovered: y = {agent.get_formula()}")
            print(f"Final Error: {loss:.6f}")
            print(f"Time: {time.time()-start_time:.2f}s")
            break

if __name__ == "__main__":
    main()