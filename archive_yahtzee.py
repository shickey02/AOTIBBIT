#!/usr/bin/env python3
# ARCHIVE YAHTZEE (HALL OF FAME EVOLUTION)
# Maintains a persistent library of the best genomes ever seen to prevent regression.

import os, argparse, copy, random, itertools, heapq
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

# --- HEDGE FUND AGENT (Reused) ---
class HedgeAgent:
    def __init__(self, brain, centroids, genome=None):
        self.brain = brain
        self.centroids = centroids 
        
        # 0:ROI, 1:Dist, 2:Risk, 3:Bonus, 4:Yahtzee, 5:Desperation, 6:Safety
        if genome is None:
            self.genome = np.array([1.0, 1.5, 1.0, 3.0, 5.0, 2.0, 2.0])
            self.genome += np.random.normal(0, 0.2, 7)
        else:
            self.genome = genome

    def get_z(self, dice):
        c = [0]*6
        for d in dice: c[d-1] += 1
        t = torch.tensor([c], dtype=torch.float32)
        with torch.no_grad(): return self.brain(t).numpy()[0]

    def get_current_risk(self, game):
        turns_passed = 13 - game.turns_left
        expected_score = turns_passed * 16
        diff = game.total_score - expected_score
        base_risk = self.genome[2]
        if diff < 0:
            risk = base_risk + (abs(diff) / 20.0 * self.genome[5])
        else:
            risk = base_risk / (1.0 + (diff / 20.0 * self.genome[6]))
        return max(0.1, risk)

    def calculate_roi(self, dice, category, game, risk_tolerance):
        ideal = {1:3, 2:6, 3:9, 4:12, 5:15, 6:18, 7:20, 8:25, 9:25, 10:30, 11:40, 12:50, 13:20}
        reward = ideal.get(category, 0)
        if category <= 6: reward *= self.genome[3]
        if category == 12: reward *= self.genome[4]
        
        z_curr = self.get_z(dice)
        manifold_map = {9:4, 10:5, 11:6, 12:8}
        
        dist = 0
        if category in manifold_map:
            target = self.centroids[manifold_map[category]]
            dist = np.linalg.norm(z_curr - target)
        else:
            count = dice.count(category) if category <= 6 else 0
            dist = (5 - count) * 1.5
            
        denominator = (dist ** self.genome[1]) + 0.1
        roi = (self.genome[0] * reward) / denominator
        return roi

    def choose_holds(self, dice, target, risk_tolerance):
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
                dist = np.linalg.norm(z_sub - target_vec)
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
            open_cats = game.get_open_slots()
            max_roi = -float('inf')
            for cat in open_cats:
                roi = self.calculate_roi(dice, cat, game, risk)
                if roi > max_roi: max_roi = roi; best_cat = cat
            if roll == 2: break
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
                dump_prio = [1, 2, 12, 13, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                for d in dump_prio:
                    if d in opts: best_cat = d; break
                if best_cat == -1: best_cat = opts[0]
        game.commit_score(best_cat, score)

# --- HALL OF FAME SYSTEM ---
class HallOfFame:
    def __init__(self, capacity=50):
        self.capacity = capacity
        # Heap of (score, genome)
        self.legends = [] 
        
    def add(self, score, genome):
        # We add negative score because heapq is a min-heap
        # We want to keep the HIGHEST scores.
        # Actually, simpler: store (score, genome), sort, keep top 50
        self.legends.append((score, genome))
        self.legends.sort(key=lambda x: x[0], reverse=True)
        self.legends = self.legends[:self.capacity]
        
    def get_parents(self, n=1):
        # Return random high-performers
        return [random.choice(self.legends)[1] for _ in range(n)]
        
    def best_score(self):
        return self.legends[0][0] if self.legends else 0

def run_sim(data_dir):
    print("--- ARCHIVE EVOLUTION (HALL OF FAME) ---")
    
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
    # Start with the best Hedge Genome found so far
    base_genome = [1.0, 3.0, 1.0, 3.0, 5.0, 2.0, 2.0]
    population = [HedgeAgent(brain, centroids, base_genome) for _ in range(pop_size)]
    
    hof = HallOfFame(capacity=50)
    
    goal = 180
    
    for gen in range(1, 101):
        scores = []
        gen_best_genome = None
        gen_best_score = 0
        
        for agent in population:
            g = YahtzeeGame()
            while g.turns_left > 0: agent.play_turn(g)
            final_score = g.total_score + g.check_bonus()
            scores.append(final_score)
            
            # Check for HoF entry
            hof.add(final_score, agent.genome)
            
            if final_score > gen_best_score:
                gen_best_score = final_score
                gen_best_genome = agent.genome
            
        avg = np.mean(scores)
        top = np.max(scores)
        hof_top = hof.best_score()
        
        print(f"GEN {gen}: Avg {avg:.1f} | Max {top} | HoF Top {hof_top} | Goal {goal}")
        
        if avg > goal:
            goal += 5
            print(f">> UPGRADE: New Goal {goal}")
            
        # BREEDING STRATEGY
        # 1. Current Winners (Top 10 of this generation)
        ranks = np.argsort(scores)[::-1]
        current_winners = [population[i].genome for i in ranks[:10]]
        
        # 2. Ancient Legends (Random selection from HoF)
        legends = [x[1] for x in hof.legends]
        
        new_pop = []
        
        # Elitism: Always keep the absolute best form current gen unchanged
        new_pop.append(HedgeAgent(brain, centroids, gen_best_genome))
        
        while len(new_pop) < pop_size:
            # 70% chance to breed from current winners (Adaptation)
            # 30% chance to breed from Legends (Memory)
            if random.random() < 0.7:
                parent = random.choice(current_winners)
            else:
                parent = random.choice(legends)
                
            # Mutation
            child_genome = np.array(parent) + np.random.normal(0, 0.15, 7)
            child_genome = np.clip(child_genome, 0.1, 10.0)
            
            new_pop.append(HedgeAgent(brain, centroids, list(child_genome)))
            
        population = new_pop

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/yahtzee_BBIT")
    args = ap.parse_args()
    if not os.path.exists(args.data):
        if os.path.exists("./data/yahtzee_value"): args.data = "./data/yahtzee_value"
    run_sim(args.data)