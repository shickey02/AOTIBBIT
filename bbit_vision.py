#!/usr/bin/env python3
# BBIT ARTIST: Evolutionary Image Generation
# Reconstructs an image using semi-transparent polygons (Vector Graphics).

import numpy as np
import random, copy, time

# --- 1. THE CANVAS ---
class Canvas:
    def __init__(self, size=32):
        self.size = size
        # Target: A Simple Diagonal Gradient
        self.target = np.zeros((size, size))
        for y in range(size):
            for x in range(size):
                self.target[y, x] = (x+y) / (size*2) # 0.0 to 1.0
                
    def render(self, shapes):
        # Render list of [x, y, radius, intensity] blobs
        img = np.zeros((self.size, self.size))
        for s in shapes:
            x, y, r, val = s
            # Simple box drawing for speed (no math libs)
            y_min = max(0, int(y-r)); y_max = min(self.size, int(y+r))
            x_min = max(0, int(x-r)); x_max = min(self.size, int(x+r))
            img[y_min:y_max, x_min:x_max] += val * 0.5 # Additive blending
        return np.clip(img, 0.0, 1.0)

    def diff(self, img):
        return np.mean((self.target - img)**2)

# --- 2. THE ARTIST AGENT ---
class ArtistAgent:
    def __init__(self, size):
        # Gene: [x, y, radius, intensity]
        self.dna = [[size/2, size/2, size/4, 0.5] for _ in range(10)] # 10 shapes
        self.best_dna = copy.deepcopy(self.dna)
        self.best_score = float('inf')
        self.stagnation = 0
        self.chaos = 0.0
        self.size = size

    def act(self):
        cand = copy.deepcopy(self.dna)
        
        # PRECISION: Tweak one shape
        if self.chaos < 0.2:
            idx = random.randint(0, len(cand)-1)
            param = random.randint(0, 3)
            cand[idx][param] += random.gauss(0, 1.0 if param<3 else 0.1)
        
        # CHAOS: Randomize shape
        else:
            idx = random.randint(0, len(cand)-1)
            cand[idx] = [random.uniform(0, self.size), random.uniform(0, self.size), 
                         random.uniform(1, 10), random.uniform(0, 1)]
            
        return cand

    def cycle(self, env):
        # Render Current
        curr_img = env.render(self.dna)
        loss = env.diff(curr_img)
        
        # Think
        if loss < self.best_score:
            self.best_score = loss
            self.best_dna = copy.deepcopy(self.dna)
            self.stagnation = 0; self.chaos = 0.0
            # print(f"   >>> IMPROVEMENT: {loss:.5f}")
        else:
            self.stagnation += 1
            if self.stagnation > 50: self.chaos = min(0.8, (self.stagnation-50)/100)
            
        # Act
        cand_dna = self.act()
        cand_img = env.render(cand_dna)
        cand_loss = env.diff(cand_img)
        
        if cand_loss < loss:
            self.dna = cand_dna
        elif self.chaos > 0 and random.random() < self.chaos * 0.1:
            self.dna = cand_dna # WOBBLE
            
        return loss

def main():
    print("--- BBIT VISUAL ARTIST ---")
    env = Canvas()
    agent = ArtistAgent(32)
    
    for t in range(5000):
        loss = agent.cycle(env)
        if t % 500 == 0:
            print(f"Step {t} | MSE: {loss:.5f} | Chaos: {agent.chaos:.2f}")
            
    print(f"Final Score: {loss:.5f} (Lower is better)")

if __name__ == "__main__":
    main()