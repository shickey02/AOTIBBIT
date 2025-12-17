#!/usr/bin/env python3
# MASSIVE SCALE APEX EVOLUTION
# Uses Multiprocessing to scale population size from 50 -> 1000+.
# Implements persistence (Save/Load) to allow indefinite training.

import os, argparse, copy, random, itertools, time, pickle
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from multiprocessing import Pool, cpu_count

# ==============================================================================
# 1. CORE DEFINITIONS (Must be top-level for Pickling)
# ==============================================================================

class YahtzeeBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 16))
    def forward(self, x): return self.encoder(x)

class YahtzeeGame:
    def __init__(self):
        self.scorecard = {i: None for i in range(1, 14)}
        self.turns_left = 13
        self.total_score = 0
        self.upper_score = 0
        self.upper_counts = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
    def get_open_slots(self): return [k for k, v in self.scorecard.items() if v is None]
    def calculate_score(self, dice, category):
        counts = Counter(dice); s = sum(dice)
        if 1 <= category <= 6: return dice.count(category) * category
        if category == 7: return s if any(c >= 3 for c in counts.values()) else 0
        if category == 8: return s if any(c >= 4 for c in counts.values()) else 0
        if category == 9: 
            has_3 = any(c == 3 for c in counts.values())
            has_2 = any(c == 2 for c in counts.values())
            is_Y = any(c == 5 for c in counts.values())
            return 25 if (has_3 and has_2) or is_Y else 0
        if category == 10:
            u = sorted(list(set(dice))); seq=0; max_s=0
            for i in range(len(u)-1):
                if u[i+1] == u[i]+1: seq+=1
                else: seq=0
                max_s = max(max_s, seq)
            return 30 if max_s>=3 else 0
        if category == 11:
            u = sorted(list(set(dice))); seq=0; max_s=0
            for i in range(len(u)-1):
                if u[i+1] == u[i]+1: seq+=1
                else: seq=0
                max_s = max(max_s, seq)
            return 40 if max_s>=4 else 0
        if category == 12: return 50 if any(c==5 for c in counts.values()) else 0
        if category == 13: return s
        return 0
    def commit_score(self, category, score):
        self.scorecard[category] = score
        self.total_score += score
        if category <= 6:
            self.upper_score += score
            self.upper_counts[category] = score // category
        self.turns_left -= 1
    def check_bonus(self): return 35 if self.upper_score >= 63 else 0

# --- THE APEX AGENT (Stateless Logic for Multiprocessing) ---
# We strip the class down to a function for the worker to execute lightly
def play_game_with_genome(args):
    genome, brain_state, centroids, seed = args
    
    # Reconstruct Brain locally (prevents shared memory locks)
    # This is fast because the model is tiny
    brain = YahtzeeBrain()
    brain.load_state_dict(brain_state)
    
    # Local Helper to get Z
    def get_z(dice):
        c = [0]*6
        for d in dice: c[d-1] += 1
        with torch.no_grad(): return brain(torch.tensor([c], dtype=torch.float32)).numpy()[0]

    random.seed(seed)
    game = YahtzeeGame()
    
    while game.turns_left > 0:
        dice = [random.randint(1, 6) for _ in range(5)]
        
        # --- MODULATE PERSONALITY ---
        tp = 13 - game.turns_left
        exp = tp * 16; def_ = max(0, exp - game.total_score)
        c_risk = genome[2] + (tp/13 * genome[5]) + (min(1, def_/50) * genome[7])
        
        surplus = 0
        for num in range(1, 7):
            if game.scorecard[num] is not None: surplus += (game.upper_counts[num]-3)*num
        
        obs = genome[9]
        if surplus > 0: obs -= surplus * genome[10] * 0.1
        else: obs += abs(surplus) * genome[10] * 0.2
        obs = max(0, obs)
        
        best_cat = -1
        
        # --- TURNS ---
        for roll in range(3):
            open_cats = game.get_open_slots()
            max_u = -999
            z = get_z(dice)
            
            # Target Selection
            for cat in open_cats:
                ideal = {1:3, 2:6, 3:9, 4:12, 5:15, 6:18, 7:20, 8:25, 9:25, 10:30, 11:40, 12:50, 13:20}
                pot = ideal.get(cat,0)
                map_ = {9:4, 10:5, 11:6, 12:8}
                if cat in map_: dist = np.linalg.norm(z - centroids[map_[cat]])
                else: dist = (5 - dice.count(cat))*2.0 if cat<=6 else 4.0
                
                u = (genome[0]*pot) - (genome[1]*dist)
                if cat <= 6:
                    u += genome[8]*10
                    if dice.count(cat)>=3: u += obs*20
                if cat == 12 and dice.count(dice[0]) == 5: u += 1000
                if u > max_u: max_u = u; best_cat = cat
            
            if roll == 2: break
            
            # Hold Selection
            if best_cat <= 6: holds = [d for d in dice if d == best_cat]
            elif best_cat in map_:
                t_vec = centroids[map_[best_cat]]
                best_h = []; min_c = 999
                for r in range(1, 6):
                    for sub in itertools.combinations(range(5), r):
                        sd = [dice[i] for i in sub]
                        d = np.linalg.norm(get_z(sd) - t_vec)
                        r_val = c_risk if abs(c_risk) > 0.01 else 0.01
                        pen = (5-len(sd)) * (1.0/r_val)
                        if d+pen < min_c: min_c = d+pen; best_h = sd
                holds = best_h
            else:
                c = Counter(dice); holds = [d for d in dice if c[d]>=2] or [max(dice)]
            
            if len(holds) == 5: break
            dice = holds + [random.randint(1, 6) for _ in range(5-len(holds))]
            
        # Score
        s = game.calculate_score(dice, best_cat)
        if s == 0:
            for cat in game.get_open_slots():
                if game.calculate_score(dice, cat)>0: best_cat=cat; s=game.calculate_score(dice, cat); break
            if s == 0:
                opts = game.get_open_slots()
                prio = [1, 2, 12, 13, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                for p in prio:
                    if p in opts: best_cat = p; break
                if best_cat == -1: best_cat = opts[0]
        game.commit_score(best_cat, s)
        
    return game.total_score + game.check_bonus()

# ==============================================================================
# 2. EVOLUTIONARY ENGINE
# ==============================================================================

def load_data(data_dir):
    # Load Brain State
    brain = YahtzeeBrain()
    try:
        full = torch.load(os.path.join(data_dir, "model.pth"))
        enc = {k:v for k,v in full.items() if 'encoder' in k}
        brain.load_state_dict(enc, strict=False)
        state = brain.state_dict()
    except:
        print("Model not found. Using random init.")
        state = brain.state_dict()
        
    # Load Manifold
    try:
        Z = np.load(os.path.join(data_dir, "latents.npy"))
        Y = np.load(os.path.join(data_dir, "labels.npy"))
        centroids = {}
        for lid in [4, 5, 6, 8]:
            mask = (Y == lid)
            if np.sum(mask) > 0: centroids[lid] = np.mean(Z[mask], axis=0)
    except:
        print("Manifold not found.")
        centroids = {}
        
    return state, centroids

def save_checkpoint(pop, hof, gen, filename="checkpoint.pkl"):
    with open(filename, "wb") as f:
        pickle.dump({"pop": pop, "hof": hof, "gen": gen}, f)
    # print("   [Saved Checkpoint]")

def run_evolution(data_dir):
    brain_state, centroids = load_data(data_dir)
    
    POP_SIZE = 500 # SCALE UP
    ELITE_SIZE = 20
    
    # Load or Init
    if os.path.exists("checkpoint.pkl"):
        print("--- RESUMING FROM CHECKPOINT ---")
        with open("checkpoint.pkl", "rb") as f:
            data = pickle.load(f)
            population = data["pop"]
            hall_of_fame = data["hof"]
            start_gen = data["gen"]
    else:
        print("--- STARTING NEW MASSIVE EVOLUTION ---")
        # Init with Apex Genome + Noise
        base = np.array([1.5, 0.4, -0.6, 0.5, 0.0, 1.5, 1.0, 2.0, 3.0, 5.0, 2.0])
        population = []
        for _ in range(POP_SIZE):
            population.append(list(base + np.random.normal(0, 0.3, 11)))
        hall_of_fame = []
        start_gen = 1

    # Multiprocessing Pool
    cores = cpu_count()
    print(f"   Using {cores} CPU Cores for Simulation.")
    
    try:
        with Pool(cores) as pool:
            for gen in range(start_gen, 10001): # Run forever essentially
                t0 = time.time()
                
                # Prepare Tasks
                tasks = []
                for genome in population:
                    # Each agent plays 1 game. 
                    # To reduce variance, ideally play 3, but let's do 1 for speed & population size
                    seed = random.randint(0, 1000000)
                    tasks.append((genome, brain_state, centroids, seed))
                
                # Run Batch
                scores = pool.map(play_game_with_genome, tasks)
                
                # Stats
                avg = np.mean(scores)
                mx = np.max(scores)
                
                # Update HoF
                # Store (Score, Genome)
                for s, g in zip(scores, population):
                    hall_of_fame.append((s, g))
                
                # Sort HoF and Trim
                hall_of_fame.sort(key=lambda x: x[0], reverse=True)
                hall_of_fame = hall_of_fame[:100] # Keep Top 100 Ever
                
                hof_avg = np.mean([x[0] for x in hall_of_fame])
                
                print(f"GEN {gen}: Avg {avg:.1f} | Max {mx} | HoF Top {hall_of_fame[0][0]} (Avg {hof_avg:.1f}) | {time.time()-t0:.1f}s")
                
                # Save Periodically
                if gen % 10 == 0:
                    save_checkpoint(population, hall_of_fame, gen)
                
                # Breeding
                new_pop = []
                
                # Elitism: Current Best
                ranks = np.argsort(scores)[::-1]
                for i in range(ELITE_SIZE):
                    new_pop.append(population[ranks[i]])
                    
                # HoF Injection
                # 20% of new population comes directly from HoF
                hof_genomes = [x[1] for x in hall_of_fame]
                for _ in range(int(POP_SIZE * 0.2)):
                    new_pop.append(random.choice(hof_genomes))
                    
                # Mutation / Crossover
                while len(new_pop) < POP_SIZE:
                    # Tournament Selection
                    p1 = population[random.choice(ranks[:100])] # Top 20%
                    
                    child = np.array(p1) + np.random.normal(0, 0.15, 11)
                    # Occasional "Large Mutation" to break local optima
                    if random.random() < 0.05:
                        child += np.random.normal(0, 1.0, 11)
                        
                    new_pop.append(list(child))
                    
                population = new_pop
                
    except KeyboardInterrupt:
        print("\nStopping... Saving progress.")
        save_checkpoint(population, hall_of_fame, gen)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/yahtzee_BBIT")
    args = ap.parse_args()
    if not os.path.exists(args.data):
        if os.path.exists("./data/yahtzee_value"): args.data = "./data/yahtzee_value"
    run_evolution(args.data)