#!/usr/bin/env python3
# SMART YAHTZEE (ADAPTIVE STABILIZER)
# Uses "Pace Logic" to switch between Conservative and Chaotic modes.

import os, argparse, copy, random, itertools
import numpy as np
import torch
import torch.nn as nn
from collections import Counter

# --- BRAIN (Standard) ---
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

# --- STABILIZER AGENT ---
class StabilizerAgent:
    def __init__(self, brain, centroids, genome=None):
        self.brain = brain
        self.centroids = centroids 
        
        # META-GENOME (11 Params):
        # 0: Base Greed
        # 1: Base Focus
        # 2: Base Risk
        # 3: Bonus Sensitivity (How much we care about the Upper Section)
        # 4: Pace Sensitivity (How much we react to being ahead/behind)
        # 5: Panic Threshold (At what deficit do we go crazy?)
        # 6: Safety Threshold (At what surplus do we play safe?)
        # 7: Late Game Panic (Turn > 10 modifier)
        # 8: Yahtzee Hunt (Bias towards Cat 12)
        # 9: Upper Obsession (Bias towards Cat 1-6)
        # 10: Inversion Factor (Negative Risk Multiplier)
        
        if genome is None:
            self.genome = np.array([
                1.5,  # Greed
                0.5,  # Focus
                1.0,  # Risk
                2.0,  # Bonus Sens
                1.5,  # Pace Sens
                -10.0, # Panic Threshold (Behind by 10)
                5.0,   # Safety Threshold (Ahead by 5)
                2.0,  # Late Game Panic
                1.0,  # Yahtzee Hunt
                3.0,  # Upper Obsession
                1.0   # Inversion
            ])
            self.genome += np.random.normal(0, 0.2, 11)
        else:
            self.genome = genome

    def get_z(self, dice):
        c = [0]*6
        for d in dice: c[d-1] += 1
        t = torch.tensor([c], dtype=torch.float32)
        with torch.no_grad(): return self.brain(t).numpy()[0]

    def analyze_pace(self, game):
        # Calculate "Par" for this turn
        turns_played = 13 - game.turns_left
        # We need 63 points total. ~4.85 pts per turn.
        # However, we usually score Upper points in bursts (e.g., 4 Fives = 20).
        # Linear approximation:
        target_pace = turns_played * 5.0
        delta = game.upper_score - target_pace
        return delta

    def modulate_personality(self, game):
        delta = self.analyze_pace(game)
        
        curr_greed = self.genome[0]
        curr_focus = self.genome[1]
        curr_risk = self.genome[2]
        
        # PHASE LOGIC
        if delta < self.genome[5]: 
            # BEHIND PACE (PANIC MODE)
            # We need big scores. Increase Risk (make it negative/small), Increase Greed.
            curr_risk = -0.5 * self.genome[10] # Invert risk to encourage chaos
            curr_greed += 2.0
            
        elif delta > self.genome[6]:
            # AHEAD OF PACE (SAFETY MODE)
            # We are safe. Don't blow it.
            # Increase Risk (make it large positive -> High Penalty for reroll)
            curr_risk = 3.0 
            curr_greed -= 0.5 # Take sure things
            
        else:
            # NEUTRAL
            pass
            
        # Late Game Desperation
        if game.turns_left < 4 and delta < 0:
             curr_risk = -1.0 * self.genome[10] # Maximum Chaos
             
        return {
            'greed': max(0.1, curr_greed),
            'focus': max(0.1, curr_focus),
            'risk':  curr_risk, # Can be negative!
            'upper_bias': self.genome[9],
            'yahtzee_bias': self.genome[8]
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
        
        # Contextual Biases
        if category <= 6:
            utility += personality['upper_bias'] * 10
            # Super bonus for completing a set (3+)
            if dice.count(category) >= 3:
                utility += 50 # Massive spike to lock it in
                
        if category == 12: # Yahtzee
            utility += personality['yahtzee_bias'] * 5
            
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
                
                # RISK LOGIC
                # If Risk is Negative: Penalty becomes Reward.
                # If Risk is Positive: Penalty is Cost.
                if personality['risk'] == 0: personality['risk'] = 0.01
                
                penalty = rerolls * (1.0 / personality['risk'])
                
                cost = d + penalty
                if cost < min_cost:
                    min_cost = cost
                    best_holds = sub_dice
        return best_holds

    def play_turn(self, game):
        dice = [random.randint(1, 6) for _ in range(5)]
        
        # Calculate Personality for THIS specific moment
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
            
            # Optimization: If holding 5 dice, stop.
            if len(holds) == 5: break
            
            # Optimization: If we have a "Natural" good hand and we are in Safety Mode, Stop.
            # E.g. We have Full House naturally, and Risk is High (Safety). Take it.
            if persona['risk'] > 2.0:
                current_score = game.calculate_score(dice, best_cat)
                if current_score >= 25: break # Satisficed
            
            new_dice = [random.randint(1, 6) for _ in range(5-len(holds))]
            dice = holds + new_dice
            
        score = game.calculate_score(dice, best_cat)
        
        if score == 0:
            open_cats = game.get_open_slots()
            for cat in open_cats:
                if game.calculate_score(dice, cat) > 0:
                    best_cat = cat; score = game.calculate_score(dice, cat); break
            if score == 0:
                opts = game.get_open_slots()
                if 1 in opts: best_cat = 1
                elif 12 in opts: best_cat = 12
                elif opts: best_cat = opts[0]
                
        game.commit_score(best_cat, score)

def run_sim(data_dir):
    print("--- ADAPTIVE STABILIZER EVOLUTION ---")
    
    # Load Brain
    brain = YahtzeeBrain()
    try:
        full = torch.load(os.path.join(data_dir, "model.pth"))
        enc = {k:v for k,v in full.items() if 'encoder' in k}
        brain.load_state_dict(enc, strict=False)
        brain.eval()
    except:
        print("Data error.")
        return
    
    # Load Manifold
    Z = np.load(os.path.join(data_dir, "latents.npy"))
    Y = np.load(os.path.join(data_dir, "labels.npy"))
    centroids = {}
    for lid in [4, 5, 6, 8]:
        mask = (Y == lid)
        if np.sum(mask) > 0: centroids[lid] = np.mean(Z[mask], axis=0)
        
    pop_size = 60
    population = [StabilizerAgent(brain, centroids) for _ in range(pop_size)]
    
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
        # Monitor the Stabilizer Params
        # Inversion Factor tells us how crazy it gets when behind
        print(f"   Top: Inversion={best.genome[10]:.2f} PanicThresh={best.genome[5]:.2f}")
        
        new_pop = []
        while len(new_pop) < pop_size:
            p = random.choice(parents)
            child_genome = np.array(p.genome) + np.random.normal(0, 0.15, 11)
            new_pop.append(StabilizerAgent(brain, centroids, list(child_genome)))
            
        population = new_pop

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/yahtzee_BBIT")
    args = ap.parse_args()
    if not os.path.exists(args.data):
        if os.path.exists("./data/yahtzee_value"): args.data = "./data/yahtzee_value"
    run_sim(args.data)