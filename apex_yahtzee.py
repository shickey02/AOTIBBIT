#!/usr/bin/env python3
# APEX YAHTZEE
# Combines Precision Logic, Hall of Fame Memory, and Critical Moment Monte Carlo.

import os, argparse, copy, random, itertools
import numpy as np
import torch
import torch.nn as nn
from collections import Counter

# --- BRAIN ---
class YahtzeeBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 16))
    def forward(self, x): return self.encoder(x)

# --- GAME ENGINE ---
class YahtzeeGame:
    def __init__(self):
        self.scorecard = {i: None for i in range(1, 14)}
        self.turns_left = 13
        self.total_score = 0
        self.upper_score = 0
        self.upper_counts = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0}
    def get_open_slots(self): return [k for k, v in self.scorecard.items() if v is None]
    
    def calculate_score(self, dice, category):
        counts = Counter(dice)
        s = sum(dice)
        if 1 <= category <= 6: return dice.count(category) * category
        if category == 7: return s if any(c >= 3 for c in counts.values()) else 0
        if category == 8: return s if any(c >= 4 for c in counts.values()) else 0
        if category == 9: 
            has_3 = any(c == 3 for c in counts.values())
            has_2 = any(c == 2 for c in counts.values())
            is_Y = any(c == 5 for c in counts.values())
            return 25 if (has_3 and has_2) or is_Y else 0
        if category == 10:
            u = sorted(list(set(dice)))
            seq=0; max_s=0
            for i in range(len(u)-1):
                if u[i+1] == u[i]+1: seq+=1
                else: seq=0
                max_s = max(max_s, seq)
            if max_s>=3: return 30
            return 0
        if category == 11:
            u = sorted(list(set(dice)))
            seq=0; max_s=0
            for i in range(len(u)-1):
                if u[i+1] == u[i]+1: seq+=1
                else: seq=0
                max_s = max(max_s, seq)
            if max_s>=4: return 40
            return 0
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

# --- APEX AGENT ---
class ApexAgent:
    def __init__(self, brain, centroids, genome=None):
        self.brain = brain
        self.centroids = centroids 
        
        # GENOME (11 Params) - Precision Genome
        if genome is None:
            # Starting with a known high-performance set
            self.genome = np.array([1.5, 0.4, -0.6, 0.5, 0.0, 1.5, 1.0, 2.0, 3.0, 5.0, 2.0])
            self.genome += np.random.normal(0, 0.2, 11)
        else:
            self.genome = genome

    def get_z(self, dice):
        c = [0]*6
        for d in dice: c[d-1] += 1
        t = torch.tensor([c], dtype=torch.float32)
        with torch.no_grad(): return self.brain(t).numpy()[0]

    def modulate_personality(self, game):
        turns_passed = 13 - game.turns_left
        progress = turns_passed / 13.0
        expected_score = turns_passed * 16 
        deficit = max(0, expected_score - game.total_score)
        deficit_factor = min(1.0, deficit / 50.0)
        
        curr_greed = self.genome[0] + (progress * self.genome[3]) + (deficit_factor * self.genome[6])
        curr_focus = self.genome[1] + (progress * self.genome[4])
        curr_risk  = self.genome[2] + (progress * self.genome[5]) + (deficit_factor * self.genome[7])
        
        surplus = 0
        for num in range(1, 7):
            if game.scorecard[num] is not None:
                diff = game.upper_counts[num] - 3
                surplus += diff * num
        
        obsession_mod = self.genome[9]
        if surplus > 0: obsession_mod -= (surplus * self.genome[10] * 0.1)
        else: obsession_mod += (abs(surplus) * self.genome[10] * 0.2)
            
        return {
            'greed': max(0.1, curr_greed),
            'focus': max(0.1, curr_focus),
            'risk':  curr_risk,
            'bonus': self.genome[8],
            'obsession': max(0.0, obsession_mod)
        }

    def evaluate_target(self, dice, category, game, personality):
        ideal = {1:3, 2:6, 3:9, 4:12, 5:15, 6:18, 7:20, 8:25, 9:25, 10:30, 11:40, 12:50, 13:20}
        pot = ideal.get(category, 0)
        
        z_curr = self.get_z(dice)
        manifold_map = {9:4, 10:5, 11:6, 12:8}
        
        dist = 0
        if category in manifold_map:
            target = self.centroids[manifold_map[category]]
            dist = np.linalg.norm(z_curr - target)
        else:
            count = dice.count(category) if category <= 6 else 0
            dist = (5 - count) * 2.0
            
        utility = (personality['greed'] * pot) - (personality['focus'] * dist)
        
        if category <= 6:
            utility += personality['bonus'] * 10
            count = dice.count(category)
            if count >= 3: utility += personality['obsession'] * 20
            elif count >= 4: utility += personality['obsession'] * 30
            
        if category == 12 and dice.count(dice[0]) == 5: utility += 1000 
            
        return utility

    # --- CRITICAL MOMENT SIMULATION ---
    def simulate_holds(self, dice, target, game):
        # Only run this if we are in the "Danger Zone"
        # Danger Zone = Turn > 8 AND Upper Score between 40 and 63
        # Or trying for Yahtzee
        is_critical = (game.turns_left < 6 and 40 <= game.upper_score < 63) or (target == 12)
        
        if not is_critical: return None # Use heuristic
        
        # Monte Carlo Logic for critical turns
        best_holds = []
        best_avg = -1
        
        idxs = range(len(dice))
        candidates = []
        for r in range(1, 6):
            for subset in itertools.combinations(idxs, r):
                candidates.append([dice[i] for i in subset])
        
        # Limit candidates for speed (heuristic prune)
        candidates = candidates[:15] 
        
        for holds in candidates:
            total_score = 0
            sims = 20 # Fast sims
            for _ in range(sims):
                needed = 5 - len(holds)
                new_d = [random.randint(1,6) for _ in range(needed)]
                hand = list(holds) + new_d
                total_score += game.calculate_score(hand, target)
            avg = total_score / sims
            if avg > best_avg:
                best_avg = avg
                best_holds = list(holds)
                
        return best_holds

    def choose_holds(self, dice, target, personality, game):
        if target <= 6: return [d for d in dice if d == target]
        
        # Check Simulation First
        sim_result = self.simulate_holds(dice, target, game)
        if sim_result is not None:
            return sim_result
            
        # Fallback to Heuristic (Fast)
        manifold_map = {9:4, 10:5, 11:6, 12:8}
        if target not in manifold_map:
             c = Counter(dice)
             return [d for d in dice if c[d] >= 2] or [max(dice)]
             
        target_vec = self.centroids[manifold_map[target]]
        best_holds = []
        min_cost = float('inf')
        
        idxs = range(len(dice))
        for r in range(1, 6):
            for subset in itertools.combinations(idxs, r):
                sub_dice = [dice[i] for i in subset]
                z_sub = self.get_z(sub_dice)
                d = np.linalg.norm(z_sub - target_vec)
                
                r_val = personality['risk']
                if abs(r_val) < 0.01: r_val = 0.01
                penalty = (5 - len(sub_dice)) * (1.0 / r_val)
                
                cost = d + penalty
                if cost < min_cost:
                    min_cost = cost
                    best_holds = sub_dice
        return best_holds

    def play_turn(self, game):
        dice = [random.randint(1, 6) for _ in range(5)]
        persona = self.modulate_personality(game)
        best_cat = -1
        
        for roll in range(3):
            open_cats = game.get_open_slots()
            max_util = -float('inf')
            for cat in open_cats:
                u = self.evaluate_target(dice, cat, game, persona)
                if u > max_util: max_util = u; best_cat = cat
            
            if roll == 2: break
            holds = self.choose_holds(dice, best_cat, persona, game)
            if len(holds) == 5: break
            new_dice = [random.randint(1, 6) for _ in range(5-len(holds))]
            dice = holds + new_dice
            
        score = game.calculate_score(dice, best_cat)
        if score == 0:
            open_cats = game.get_open_slots()
            for cat in open_cats:
                if game.calculate_score(dice, cat) > 0:
                    best_cat=cat; score=game.calculate_score(dice, cat); break
            if score == 0:
                opts = game.get_open_slots()
                prio = [1, 2, 12, 13, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                for p in prio:
                    if p in opts: best_cat = p; break
                if best_cat == -1: best_cat = opts[0]
        game.commit_score(best_cat, score)

# --- HALL OF FAME ---
class HallOfFame:
    def __init__(self, capacity=40):
        self.capacity = capacity
        self.legends = [] 
    def add(self, score, genome):
        self.legends.append((score, genome))
        self.legends.sort(key=lambda x: x[0], reverse=True)
        self.legends = self.legends[:self.capacity]
    def get_parents(self): return [x[1] for x in self.legends]
    def best_score(self): return self.legends[0][0] if self.legends else 0

def run_sim(data_dir):
    print("--- APEX PREDATOR (HYBRID INTELLIGENCE) ---")
    
    # Load Brain
    brain = YahtzeeBrain()
    try:
        full = torch.load(os.path.join(data_dir, "model.pth"))
        enc = {k:v for k,v in full.items() if 'encoder' in k}
        brain.load_state_dict(enc, strict=False)
        brain.eval()
    except: return
    
    # Load Manifold
    try:
        Z = np.load(os.path.join(data_dir, "latents.npy"))
        Y = np.load(os.path.join(data_dir, "labels.npy"))
    except: return 
    centroids = {}
    for lid in [4, 5, 6, 8]:
        mask = (Y == lid)
        if np.sum(mask) > 0: centroids[lid] = np.mean(Z[mask], axis=0)
        
    pop_size = 50
    population = [ApexAgent(brain, centroids) for _ in range(pop_size)]
    hof = HallOfFame()
    goal = 185
    
    for gen in range(1, 101):
        scores = []
        for agent in population:
            g = YahtzeeGame()
            while g.turns_left > 0: agent.play_turn(g)
            s = g.total_score + g.check_bonus()
            scores.append(s)
            hof.add(s, agent.genome)
            
        avg = np.mean(scores)
        top = np.max(scores)
        hof_max = hof.best_score()
        
        print(f"GEN {gen}: Avg {avg:.1f} | Max {top} | HoF {hof_max} | Goal {goal}")
        
        if avg > goal: goal += 5; print(f">> NEW GOAL: {goal}")
            
        # BREEDING (Hybrid)
        ranks = np.argsort(scores)[::-1]
        winners = [population[i].genome for i in ranks[:8]]
        legends = hof.get_parents()
        
        new_pop = []
        while len(new_pop) < pop_size:
            if legends and random.random() < 0.4: parent = random.choice(legends)
            else: parent = random.choice(winners)
            
            child = np.array(parent) + np.random.normal(0, 0.15, 11)
            new_pop.append(ApexAgent(brain, centroids, list(child)))
            
        population = new_pop

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/yahtzee_BBIT")
    args = ap.parse_args()
    if not os.path.exists(args.data):
        if os.path.exists("./data/yahtzee_value"): args.data = "./data/yahtzee_value"
    run_sim(args.data)