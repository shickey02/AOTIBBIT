#!/usr/bin/env python3
# HEDGE FUND YAHTZEE
# Replaces rigid phases with a continuous "Return on Investment" (ROI) calculation.
# ROI = (Potential Reward) / (Vector Distance + Risk Penalty)

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

# --- HEDGE FUND AGENT ---
class HedgeAgent:
    def __init__(self, brain, centroids, genome=None):
        self.brain = brain
        self.centroids = centroids 
        
        # META-GENOME (7 Params):
        # 0: ROI Sensitivity (Greed)
        # 1: Distance Penalty (Caution)
        # 2: Risk Tolerance (Base)
        # 3: Bonus Weight (Upper Section ROI Multiplier)
        # 4: Yahtzee Weight (Jackpot ROI Multiplier)
        # 5: Desperation Multiplier (How much Risk increases when losing)
        # 6: Safety Brake (How much Risk decreases when winning)
        
        if genome is None:
            self.genome = np.array([
                1.0,  # ROI Sens
                1.5,  # Distance Penalty (Distance hurts)
                1.0,  # Risk Tolerance
                3.0,  # Bonus Weight (High)
                5.0,  # Yahtzee Weight (Very High)
                2.0,  # Desperation
                2.0   # Safety Brake
            ])
            self.genome += np.random.normal(0, 0.2, 7)
        else:
            self.genome = genome

    def get_z(self, dice):
        c = [0]*6
        for d in dice: c[d-1] += 1
        t = torch.tensor([c], dtype=torch.float32)
        with torch.no_grad(): return self.brain(t).numpy()[0]

    def get_current_risk(self, game):
        # Calculate Pnl (Profit and Loss) vs Pace
        turns_passed = 13 - game.turns_left
        expected_score = turns_passed * 16
        diff = game.total_score - expected_score
        
        base_risk = self.genome[2]
        
        if diff < 0:
            # Losing: Increase Risk Tolerance
            # (Risk goes UP, so penalty goes DOWN)
            risk = base_risk + (abs(diff) / 20.0 * self.genome[5])
        else:
            # Winning: Decrease Risk Tolerance
            # (Risk goes DOWN, penalty goes UP)
            risk = base_risk / (1.0 + (diff / 20.0 * self.genome[6]))
            
        return max(0.1, risk) # Never 0

    def calculate_roi(self, dice, category, game, risk_tolerance):
        # 1. Potential Reward
        ideal = {1:3, 2:6, 3:9, 4:12, 5:15, 6:18, 7:20, 8:25, 9:25, 10:30, 11:40, 12:50, 13:20}
        reward = ideal.get(category, 0)
        
        # Multipliers
        if category <= 6: reward *= self.genome[3]
        if category == 12: reward *= self.genome[4]
        
        # 2. Cost (Distance)
        z_curr = self.get_z(dice)
        manifold_map = {9:4, 10:5, 11:6, 12:8}
        
        dist = 0
        if category in manifold_map:
            target = self.centroids[manifold_map[category]]
            dist = np.linalg.norm(z_curr - target)
        else:
            count = dice.count(category) if category <= 6 else 0
            dist = (5 - count) * 1.5
            
        # 3. ROI Calculation
        # ROI = Reward / (Distance^Penalty)
        # We assume 0 distance = Infinite ROI, so add epsilon
        denominator = (dist ** self.genome[1]) + 0.1
        
        roi = (self.genome[0] * reward) / denominator
        
        return roi

    def choose_holds(self, dice, target, risk_tolerance):
        # If target is Upper Section, always hold target dice
        if target <= 6: return [d for d in dice if d == target]
        
        # For complex hands, use ROI logic on subsets
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
                dist = np.linalg.norm(z_sub - target_vec)
                
                # Risk Penalty
                rerolls = 5 - len(sub_dice)
                penalty = rerolls * (1.0 / risk_tolerance)
                
                cost = dist + penalty
                if cost < min_cost:
                    min_cost = cost
                    best_holds = sub_dice
        return best_holds

    def play_turn(self, game):
        dice = [random.randint(1, 6) for _ in range(5)]
        risk = self.get_current_risk(game)
        
        best_cat = -1
        
        for roll in range(3):
            # 1. Pick Target by ROI
            open_cats = game.get_open_slots()
            max_roi = -float('inf')
            
            for cat in open_cats:
                roi = self.calculate_roi(dice, cat, game, risk)
                if roi > max_roi:
                    max_roi = roi
                    best_cat = cat
            
            if roll == 2: break
            
            # 2. Pick Holds
            holds = self.choose_holds(dice, best_cat, risk)
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
                # Smart Dump: Dump 1s, then 2s, then Yahtzee, then Chance
                dump_prio = [1, 2, 12, 13, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                for d in dump_prio:
                    if d in opts: best_cat = d; break
                if best_cat == -1: best_cat = opts[0]
                
        game.commit_score(best_cat, score)

def run_sim(data_dir):
    print("--- HEDGE FUND EVOLUTION (ROI MAXIMIZATION) ---")
    
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
    population = [HedgeAgent(brain, centroids) for _ in range(pop_size)]
    
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
        print(f"   Top: DistPenalty={best.genome[1]:.2f} Desperation={best.genome[5]:.2f}")
        
        new_pop = []
        while len(new_pop) < pop_size:
            p = random.choice(parents)
            child_genome = np.array(p.genome) + np.random.normal(0, 0.15, 7)
            new_pop.append(HedgeAgent(brain, centroids, list(child_genome)))
            
        population = new_pop

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/yahtzee_BBIT")
    args = ap.parse_args()
    if not os.path.exists(args.data):
        if os.path.exists("./data/yahtzee_value"): args.data = "./data/yahtzee_value"
    run_sim(args.data)