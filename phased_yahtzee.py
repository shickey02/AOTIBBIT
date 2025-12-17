#!/usr/bin/env python3
# PHASED YAHTZEE
# Splits the game into Opening, Mid, and End games, each with evolved personalities.

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
            seq = 0
            for i in range(len(u)-1):
                if u[i+1] == u[i]+1: seq+=1
                else: seq=0
                if seq>=3: return 30
            return 0
        if category == 11:
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
        if category <= 6: self.upper_score += score
        self.turns_left -= 1
        
    def check_bonus(self):
        return 35 if self.upper_score >= 63 else 0

# --- PHASED AGENT ---
class PhasedAgent:
    def __init__(self, brain, centroids, genome=None):
        self.brain = brain
        self.centroids = centroids 
        
        # GENOME (12 Params) - 3 Phases x 4 Params
        # Phase 1: Opening (Turns 13-9) -> [Risk, BonusWt, Greed, Focus]
        # Phase 2: Midgame (Turns 8-5)  -> [Risk, BonusWt, Greed, Focus]
        # Phase 3: Endgame (Turns 4-1)  -> [Risk, BonusWt, Greed, Focus]
        
        if genome is None:
            self.genome = np.array([
                # Opening: High Chaos, High Bonus
                -0.5, 3.0, 1.5, 0.2,
                # Mid: Moderate Risk, Moderate Bonus
                1.0, 2.0, 1.2, 0.8,
                # End: Low Risk, Low Bonus
                2.0, 1.0, 1.0, 1.0
            ])
            self.genome += np.random.normal(0, 0.2, 12)
        else:
            self.genome = genome

    def get_z(self, dice):
        c = [0]*6
        for d in dice: c[d-1] += 1
        t = torch.tensor([c], dtype=torch.float32)
        with torch.no_grad(): return self.brain(t).numpy()[0]

    def get_current_personality(self, game):
        # Determine Phase
        tl = game.turns_left
        
        if tl >= 9: # OPENING (13-9)
            base_idx = 0
        elif tl >= 5: # MID (8-5)
            base_idx = 4
        else: # END (4-1)
            base_idx = 8
            
        p = self.genome[base_idx : base_idx+4]
        
        # Add Dynamic Injection (Panic)
        # If we are in Endgame and losing badly, override Risk
        if tl < 5:
            expected = (13-tl) * 16
            if game.total_score < expected - 20:
                p[0] = -0.5 # Inject Chaos
        
        return {
            'risk': p[0],
            'bonus': p[1],
            'greed': max(0.1, p[2]),
            'focus': max(0.1, p[3])
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
        
        # Phase-Specific Logic
        if category <= 6:
            utility += personality['bonus'] * 10
            # Super Bonus for high numbers (4,5,6) in early phases
            if category >= 4 and game.turns_left > 8:
                utility += personality['bonus'] * 5
                
        # Yahtzee Hunt
        if category == 12:
            utility += 20 # Always good
            
        return utility

    def choose_holds(self, dice, target, personality):
        if target <= 6: return [d for d in dice if d == target]
        
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
                
                rerolls = 5 - len(sub_dice)
                
                # RISK LOGIC (Negative allowed)
                r_val = personality['risk']
                if abs(r_val) < 0.01: r_val = 0.01
                
                penalty = rerolls * (1.0 / r_val)
                
                cost = d + penalty
                if cost < min_cost:
                    min_cost = cost
                    best_holds = sub_dice
        return best_holds

    def play_turn(self, game):
        dice = [random.randint(1, 6) for _ in range(5)]
        persona = self.get_current_personality(game)
        
        best_cat = -1
        
        for roll in range(3):
            open_cats = game.get_open_slots()
            max_util = -float('inf')
            
            for cat in open_cats:
                u = self.evaluate_target(dice, cat, game, persona)
                if u > max_util: max_util = u; best_cat = cat
            
            if roll == 2: break
            
            holds = self.choose_holds(dice, best_cat, persona)
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
                if 1 in opts: best_cat=1
                elif 12 in opts: best_cat=12
                elif opts: best_cat=opts[0]
                
        game.commit_score(best_cat, score)

def run_sim(data_dir):
    print("--- PHASED EVOLUTION (CHUNKING) ---")
    
    # Load Brain
    brain = YahtzeeBrain()
    try:
        full = torch.load(os.path.join(data_dir, "model.pth"))
        enc = {k:v for k,v in full.items() if 'encoder' in k}
        brain.load_state_dict(enc, strict=False)
        brain.eval()
    except: return
    
    # Load Manifold
    Z = np.load(os.path.join(data_dir, "latents.npy"))
    Y = np.load(os.path.join(data_dir, "labels.npy"))
    centroids = {}
    for lid in [4, 5, 6, 8]:
        mask = (Y == lid)
        if np.sum(mask) > 0: centroids[lid] = np.mean(Z[mask], axis=0)
        
    pop_size = 60
    population = [PhasedAgent(brain, centroids) for _ in range(pop_size)]
    
    goal = 180
    
    for gen in range(1, 101):
        scores = []
        for agent in population:
            g = YahtzeeGame()
            while g.turns_left > 0: agent.play_turn(g)
            scores.append(g.total_score + g.check_bonus())
            
        avg = np.mean(scores)
        top = np.max(scores)
        
        print(f"GEN {gen}: Avg {avg:.1f} | Max {top} | Goal {goal}")
        
        if avg > goal:
            goal += 5
            print(f">> UPGRADE: New Goal {goal}")
            
        ranks = np.argsort(scores)[::-1]
        parents = [population[i] for i in ranks[:12]]
        
        best = parents[0]
        # Monitor Phases
        print(f"   Top: OpenRisk={best.genome[0]:.2f} MidRisk={best.genome[4]:.2f} EndRisk={best.genome[8]:.2f}")
        
        new_pop = []
        while len(new_pop) < pop_size:
            p = random.choice(parents)
            child_genome = np.array(p.genome) + np.random.normal(0, 0.15, 12)
            new_pop.append(PhasedAgent(brain, centroids, list(child_genome)))
            
        population = new_pop

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/yahtzee_BBIT")
    args = ap.parse_args()
    if not os.path.exists(args.data):
        if os.path.exists("./data/yahtzee_value"): args.data = "./data/yahtzee_value"
    run_sim(args.data)