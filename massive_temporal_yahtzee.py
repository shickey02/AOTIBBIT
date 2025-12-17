#!/usr/bin/env python3
# MASSIVE TEMPORAL EVOLUTION (FIXED v2)
# Adds 'Time Awareness' to the Apex Agent.
# Fixed integer type error in seed generation.

import os, argparse, copy, random, itertools, time, pickle
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from multiprocessing import Pool, cpu_count

# --- BRAIN (7 Inputs) ---
class TemporalBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(7, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16))
    def forward(self, x): return self.encoder(x)

# --- GAME ---
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

# --- WORKER ---
def play_game_temporal(args):
    genome, brain_state, seed = args
    
    # Rebuild brain locally
    brain = TemporalBrain()
    brain.load_state_dict(brain_state, strict=False) 
    
    # Helper: Get Z with Time
    def get_z(dice, rolls_left):
        c = [0]*6
        for d in dice: c[d-1] += 1
        # Input 7: [Counts... , Rolls_Normalized]
        vec = c + [rolls_left / 2.0]
        with torch.no_grad(): return brain(torch.tensor([vec], dtype=torch.float32)).numpy()[0]

    random.seed(seed)
    game = YahtzeeGame()
    
    while game.turns_left > 0:
        dice = [random.randint(1, 6) for _ in range(5)]
        
        # Personality
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
        
        for roll in range(3):
            rolls_remain = 2 - roll
            open_cats = game.get_open_slots()
            max_u = -999
            
            # Use 0 rolls left for evaluation (Current Value)
            z_curr = get_z(dice, 0) 
            
            # Use Future rolls for potential
            z_pot = get_z(dice, rolls_remain)
            
            for cat in open_cats:
                ideal = {1:3, 2:6, 3:9, 4:12, 5:15, 6:18, 7:20, 8:25, 9:25, 10:30, 11:40, 12:50, 13:20}
                pot = ideal.get(cat,0)
                
                dist = 0
                if cat <= 6: dist = (5 - dice.count(cat)) * 2.0
                elif cat == 12: dist = (5 - max(Counter(dice).values())) * 3.0
                else: dist = 4.0 
                
                u = (genome[0]*pot) - (genome[1]*dist)
                
                # Temporal Brain Injection
                # If Brain says this hand has high potential given time left, boost
                brain_val = np.linalg.norm(z_pot) 
                u += brain_val * 2.0 
                
                if cat <= 6:
                    u += genome[8]*10
                    if dice.count(cat)>=3: u += obs*20
                if cat == 12 and dice.count(dice[0]) == 5: u += 1000
                if u > max_u: max_u = u; best_cat = cat
            
            if roll == 2: break
            
            # Holds
            if best_cat <= 6: holds = [d for d in dice if d == best_cat]
            elif best_cat == 12: 
                c=Counter(dice); m=c.most_common(1)[0][0]
                holds = [d for d in dice if d==m]
            else:
                c = Counter(dice); holds = [d for d in dice if c[d]>=2] or [max(dice)]
            
            if len(holds) == 5: break
            dice = holds + [random.randint(1, 6) for _ in range(5-len(holds))]
            
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

# --- EVOLUTION ---
def run_evolution(data_dir):
    print("--- TEMPORAL APEX EVOLUTION ---")
    
    # Load
    brain = TemporalBrain()
    try:
        # LOAD FIX: Extract encoder weights or use strict=False
        full = torch.load(os.path.join(data_dir, "model.pth"))
        try:
            enc = {k:v for k,v in full.items() if 'encoder' in k}
            brain.load_state_dict(enc, strict=False)
        except:
            brain.load_state_dict(full, strict=False)
            
        brain_state = brain.state_dict()
        print("   [Brain Loaded Successfully]")
    except Exception as e:
        print(f"Model error: {e}"); return

    POP_SIZE = 500
    base = np.array([1.5, 0.4, -0.6, 0.5, 0.0, 1.5, 1.0, 2.0, 3.0, 5.0, 2.0])
    population = [list(base + np.random.normal(0, 0.3, 11)) for _ in range(POP_SIZE)]
    hof = []
    
    cores = cpu_count()
    print(f"   [Running on {cores} Cores]")
    
    with Pool(cores) as pool:
        for gen in range(1, 10001):
            t0 = time.time()
            # FIX: Use integer 1000000 instead of float 1e6
            tasks = [(g, brain_state, random.randint(0, 1000000)) for g in population]
            scores = pool.map(play_game_temporal, tasks)
            
            avg = np.mean(scores); mx = np.max(scores)
            
            for s, g in zip(scores, population): hof.append((s, g))
            hof.sort(key=lambda x: x[0], reverse=True)
            hof = hof[:100]
            
            print(f"GEN {gen}: Avg {avg:.1f} | Max {mx} | HoF {hof[0][0]} | {time.time()-t0:.1f}s")
            
            # Breed
            ranks = np.argsort(scores)[::-1]
            elites = [population[i] for i in ranks[:20]]
            legends = [x[1] for x in hof]
            
            new_pop = list(elites)
            while len(new_pop) < POP_SIZE:
                p = random.choice(elites if random.random()<0.7 else legends)
                child = np.array(p) + np.random.normal(0, 0.15, 11)
                new_pop.append(list(child))
            population = new_pop

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/yahtzee_temporal")
    args = ap.parse_args()
    if not os.path.exists(args.data): print("Run setup_temporal_yahtzee.py first."); exit()
    run_evolution(args.data)