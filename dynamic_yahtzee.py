#!/usr/bin/env python3
# DYNAMIC YAHTZEE (Modulated Vector Agent)
# Evolving a 'Meta-Genome' that shifts strategy based on Turn and Score Deficit.

import os, argparse, copy, random, itertools
import numpy as np
import torch
import torch.nn as nn
from collections import Counter

# --- 1. BRAIN (Stateless) ---
class YahtzeeBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 16))
    def forward(self, x): return self.encoder(x)

# --- 2. GAME ENGINE ---
class YahtzeeGame:
    def __init__(self):
        self.scorecard = {i: None for i in range(1, 14)}
        self.turns_left = 13
        self.total_score = 0
        
    def get_open_slots(self):
        return [k for k, v in self.scorecard.items() if v is None]

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
        if category == 10: # SmStr
            u = sorted(list(set(dice)))
            seq = 0
            for i in range(len(u)-1):
                if u[i+1] == u[i]+1: seq+=1
                else: seq=0
                if seq>=3: return 30
            return 0
        if category == 11: # LgStr
            u = sorted(list(set(dice)))
            seq = 0
            for i in range(len(u)-1):
                if u[i+1] == u[i]+1: seq+=1
                else: seq=0
                if seq>=4: return 40
            return 0
        if category == 12: return 50 if any(c==5 for c in counts.values()) else 0
        if category == 13: return s
        return 0

    def commit_score(self, category, score):
        self.scorecard[category] = score
        self.total_score += score
        self.turns_left -= 1

# --- 3. DYNAMIC VECTOR AGENT ---
class DynamicAgent:
    def __init__(self, brain, centroids, genome=None):
        self.brain = brain
        self.centroids = centroids 
        
        # META-GENOME (9 Params):
        # 0: Base Greed
        # 1: Base Focus
        # 2: Base Risk
        # 3: Turn -> Greed Modifier (Does greed change as game ends?)
        # 4: Turn -> Focus Modifier (Do we get sloppy late game?)
        # 5: Turn -> Risk Modifier (Do we play safer late game?)
        # 6: Deficit -> Greed Boost (Do we get greedy when losing?)
        # 7: Deficit -> Risk Boost (Do we take risks when losing?)
        # 8: Bonus Weight (Obsession with Upper Section)
        
        if genome is None:
            self.genome = np.random.uniform(-0.5, 1.5, 9)
            # Force bases to be positive initially
            self.genome[0:3] = np.abs(self.genome[0:3]) + 0.5
        else:
            self.genome = genome

    def get_z(self, dice):
        c = [0]*6
        for d in dice: c[d-1] += 1
        t = torch.tensor([c], dtype=torch.float32)
        with torch.no_grad(): return self.brain(t).numpy()[0]

    def modulate_personality(self, game):
        # 1. Calculate Game State
        turns_passed = 13 - game.turns_left
        progress = turns_passed / 13.0
        
        # Expected Score Curve (Target ~200 pts)
        expected_score = turns_passed * 15 
        deficit = max(0, expected_score - game.total_score)
        deficit_factor = min(1.0, deficit / 50.0) # Cap at 50pt deficit
        
        # 2. Modulate
        # P = Base + (Time * Mod) + (Deficit * Mod)
        
        curr_greed = self.genome[0] + (progress * self.genome[3]) + (deficit_factor * self.genome[6])
        curr_focus = self.genome[1] + (progress * self.genome[4]) # Deficit usually shouldn't change focus much?
        curr_risk  = self.genome[2] + (progress * self.genome[5]) + (deficit_factor * self.genome[7])
        
        # Clamp to sane values
        return {
            'greed': max(0.1, curr_greed),
            'focus': max(0.1, curr_focus),
            'risk':  max(0.1, curr_risk),
            'bonus': self.genome[8]
        }

    def evaluate_target(self, dice, category, game, personality):
        # Standard ideal scores
        ideal = {1:3, 2:6, 3:9, 4:12, 5:15, 6:18, 7:20, 8:25, 9:25, 10:30, 11:40, 12:50, 13:20}
        pot = ideal.get(category, 0)
        
        # Geometric Distance
        z_curr = self.get_z(dice)
        manifold_map = {9:4, 10:5, 11:6, 12:8}
        
        dist = 0
        if category in manifold_map:
            target = self.centroids[manifold_map[category]]
            dist = np.linalg.norm(z_curr - target)
        else:
            count = dice.count(category) if category <= 6 else 0
            dist = (5 - count) * 2.0
            
        # Utility Calculation using CURRENT Personality
        utility = (personality['greed'] * pot) - (personality['focus'] * dist)
        
        # Bonus Logic
        if category <= 6 and game.turns_left > 5:
            utility += personality['bonus'] * 10
            
        return utility

    def choose_holds(self, dice, target, personality):
        if target <= 6: return [d for d in dice if d == target]
        
        # Vector Logic for complex hands
        manifold_map = {9:4, 10:5, 11:6, 12:8}
        
        # Fallback for non-manifold targets
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
                
                # Risk Penalty is now DYNAMIC
                # High Risk Tolerance = Low Penalty for rerolling
                # Cost = Dist + (Rerolls / Risk_Tolerance)
                rerolls = 5 - len(sub_dice)
                penalty = rerolls * (1.0 / personality['risk']) 
                
                cost = d + penalty
                if cost < min_cost:
                    min_cost = cost
                    best_holds = sub_dice
        return best_holds

    def play_turn(self, game):
        dice = [random.randint(1, 6) for _ in range(5)]
        
        # Calculate Personality for THIS turn
        persona = self.modulate_personality(game)
        
        best_cat = -1
        
        for roll in range(3):
            # 1. Pick Target
            open_cats = game.get_open_slots()
            max_util = -float('inf')
            
            for cat in open_cats:
                u = self.evaluate_target(dice, cat, game, persona)
                if u > max_util:
                    max_util = u
                    best_cat = cat
            
            if roll == 2: break
            
            # 2. Pick Holds
            holds = self.choose_holds(dice, best_cat, persona)
            if len(holds) == 5: break
            
            new_dice = [random.randint(1, 6) for _ in range(5-len(holds))]
            dice = holds + new_dice
            
        score = game.calculate_score(dice, best_cat)
        
        # Panic Dump (Simplified)
        if score == 0:
            open_cats = game.get_open_slots()
            # Try to find points
            for cat in open_cats:
                if game.calculate_score(dice, cat) > 0:
                    best_cat = cat; score = game.calculate_score(dice, cat); break
            # Dump logic
            if score == 0:
                if 1 in open_cats: best_cat = 1
                elif 12 in open_cats: best_cat = 12
                elif open_cats: best_cat = open_cats[0]
                
        game.commit_score(best_cat, score)

def run_sim(data_dir):
    print("--- DYNAMIC EVOLUTION (ADRENALINE ENGINE) ---")
    
    # Load Brain
    brain = YahtzeeBrain()
    full = torch.load(os.path.join(data_dir, "model.pth"))
    enc = {k:v for k,v in full.items() if 'encoder' in k}
    brain.load_state_dict(enc, strict=False)
    brain.eval()
    
    # Load Manifold
    Z = np.load(os.path.join(data_dir, "latents.npy"))
    Y = np.load(os.path.join(data_dir, "labels.npy"))
    centroids = {}
    for lid in [4, 5, 6, 8]:
        mask = (Y == lid)
        if np.sum(mask) > 0: centroids[lid] = np.mean(Z[mask], axis=0)
        
    # Init Population
    pop_size = 60
    population = [DynamicAgent(brain, centroids) for _ in range(pop_size)]
    
    goal = 150 # Start higher
    
    for gen in range(1, 101):
        scores = []
        for agent in population:
            g = YahtzeeGame()
            while g.turns_left > 0:
                agent.play_turn(g)
            scores.append(g.total_score)
            
        avg = np.mean(scores)
        top = np.max(scores)
        
        print(f"GEN {gen}: Avg {avg:.1f} | Max {top} | Goal {goal}")
        
        if avg > goal:
            goal += 10
            print(f">> UPGRADE: New Goal {goal}")
            
        # Selection
        ranks = np.argsort(scores)[::-1]
        parents = [population[i] for i in ranks[:12]]
        
        best = parents[0]
        # Print a snapshot of the meta-genome
        # Just printing base stats + deficit boost to see if it learns "Clutch Factor"
        print(f"   Top Meta: BaseRisk={best.genome[2]:.2f} DeficitBoost={best.genome[7]:.2f}")
        
        new_pop = []
        while len(new_pop) < pop_size:
            p = random.choice(parents)
            child_genome = p.genome + np.random.normal(0, 0.15, 9) # Higher mutation rate
            new_pop.append(DynamicAgent(brain, centroids, child_genome))
            
        population = new_pop

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/yahtzee_bbit")
    args = ap.parse_args()
    if not os.path.exists(args.data):
        print("Error: Data not found. Run setup_yahtzee.py first.")
    else:
        run_sim(args.data)