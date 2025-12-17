#!/usr/bin/env python3
# PRECISION HUNTER
# The "Bonus Hunter" logic, refined with exact "Gap-to-Bonus" tracking per category.

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
        # Track exactly how many of each Upper number we have banked
        # (This helps the agent know if it "over-scored" on Fives and can "under-score" on Twos)
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
            # Reverse calculate count from score
            self.upper_counts[category] = score // category
        self.turns_left -= 1
        
    def check_bonus(self):
        return 35 if self.upper_score >= 63 else 0

# --- PRECISION AGENT ---
class PrecisionAgent:
    def __init__(self, brain, centroids, genome=None):
        self.brain = brain
        self.centroids = centroids 
        
        # GENOME (11 Params) - Refined for Precision
        # 0: Greed
        # 1: Focus
        # 2: Risk (Negative allowed)
        # 3: T->Greed
        # 4: T->Focus
        # 5: T->Risk
        # 6: D->Greed
        # 7: D->Risk
        # 8: Bonus Weight (General)
        # 9: GAP OBSESSION (How much we panic over specific missing numbers)
        # 10: SURPLUS RELAXATION (How much we relax if we have extra upper points)
        
        if genome is None:
            self.genome = np.array([
                1.4, 0.4, -0.5, # Start with Negative Risk (Chaos)
                0.5, 0.0, 1.5,
                1.0, 2.0,
                2.0, # Bonus Wt
                5.0, # Gap Obsession (HUGE)
                2.0  # Surplus Relax
            ])
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
        
        # Standard Deficit Logic
        expected_score = turns_passed * 16 
        deficit = max(0, expected_score - game.total_score)
        deficit_factor = min(1.0, deficit / 50.0)
        
        curr_greed = self.genome[0] + (progress * self.genome[3]) + (deficit_factor * self.genome[6])
        curr_focus = self.genome[1] + (progress * self.genome[4])
        curr_risk  = self.genome[2] + (progress * self.genome[5]) + (deficit_factor * self.genome[7])
        
        # Calculate UPPER HEALTH
        # We need 3 of each number (roughly).
        # Surplus = (My_Fives - 3) * 5 + (My_Sixes - 3) * 6 ...
        # If Surplus > 0, we are safe. If < 0, we are in danger.
        
        surplus = 0
        for num in range(1, 7):
            if game.scorecard[num] is not None:
                count = game.upper_counts[num]
                diff = count - 3
                surplus += diff * num
        
        # If surplus is high, Obsession goes down.
        # If surplus is negative, Obsession goes UP.
        obsession_mod = self.genome[9]
        if surplus > 0:
            obsession_mod -= (surplus * self.genome[10] * 0.1)
        else:
            obsession_mod += (abs(surplus) * self.genome[10] * 0.2)
            
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
        
        # PRECISION LOGIC
        if category <= 6:
            # Base Bonus Desire
            utility += personality['bonus'] * 10
            
            # Count Logic
            count = dice.count(category)
            if count >= 3:
                # We hit the target! Massive reward.
                utility += personality['obsession'] * 20
            elif count >= 4:
                # Surplus! Even better.
                utility += personality['obsession'] * 30
            else:
                # Less than 3? The utility is lower because we aren't helping the bonus much.
                # BUT, if we are desperate (high obsession), maybe we take 2 Fives?
                # No, we want to force REROLLS to get 3.
                pass 
        
        # Yahtzee Logic (Always Infinite Utility if real)
        if category == 12 and dice.count(dice[0]) == 5:
            utility += 1000 
            
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
                
                # Handle Negative Risk (Reward for Chaos)
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
        persona = self.modulate_personality(game)
        
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
        
        # SMART DUMP (Safety Net)
        if score == 0:
            open_cats = game.get_open_slots()
            for cat in open_cats:
                if game.calculate_score(dice, cat) > 0:
                    best_cat=cat; score=game.calculate_score(dice, cat); break
            if score == 0:
                # Dump order
                opts = game.get_open_slots()
                prio = [1, 2, 12, 13, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                for p in prio:
                    if p in opts: best_cat = p; break
                if best_cat == -1: best_cat = opts[0]
                
        game.commit_score(best_cat, score)

def run_sim(data_dir):
    print("--- PRECISION HUNTER EVOLUTION ---")
    
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
        
    pop_size = 60
    population = [PrecisionAgent(brain, centroids) for _ in range(pop_size)]
    
    goal = 185 # We aim high
    
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
        print(f"   Top: GapObsession={best.genome[9]:.2f} SurplusRelax={best.genome[10]:.2f}")
        
        new_pop = []
        while len(new_pop) < pop_size:
            p = random.choice(parents)
            child_genome = np.array(p.genome) + np.random.normal(0, 0.15, 11)
            new_pop.append(PrecisionAgent(brain, centroids, list(child_genome)))
            
        population = new_pop

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/yahtzee_BBIT")
    args = ap.parse_args()
    if not os.path.exists(args.data):
        if os.path.exists("./data/yahtzee_value"): args.data = "./data/yahtzee_value"
    run_sim(args.data)