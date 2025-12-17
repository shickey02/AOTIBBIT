#!/usr/bin/env python3
# GRADIENT YAHTZEE (FIXED)
# Agent maximizes the predicted value of the hand in latent space.

import os, argparse, copy, random, itertools
import numpy as np
import torch
import torch.nn as nn
from collections import Counter

# --- RE-DEFINE BRAIN (MUST MATCH TRAINING EXACTLY) ---
class ValueBrain(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(6, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # Head 1: Reconstruction (Pattern) - Included for compatibility
        self.decoder_pattern = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 6) 
        )
        
        # Head 2: Valuation (Score)
        self.decoder_value = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        z = self.encoder(x)
        # We only need value for inference, but defining forward for completeness
        pat = self.decoder_pattern(z)
        val = self.decoder_value(z)
        return val, z

# --- GAME ENGINE (Standard) ---
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
        self.turns_left -= 1

# --- GRADIENT AGENT ---
class GradientAgent:
    def __init__(self, brain, centroids, genome=None):
        self.brain = brain
        self.centroids = centroids
        # Genome: [Greed, Focus, Risk, DeficitBoost]
        if genome is None:
            self.genome = [1.4, 0.4, 1.0, 1.5] 
        else:
            self.genome = genome

    def predict_value(self, dice):
        # Maps dice -> Predicted Score (0-1)
        c = [0]*6
        for d in dice: c[d-1] += 1
        t = torch.tensor([c], dtype=torch.float32)
        with torch.no_grad():
            # Model returns (pattern, value, z) or (val, z) depending on forward?
            # Let's call components directly to be safe or use unpacked forward
            val, z = self.brain(t)
        return val.item(), z.numpy()[0]

    def choose_holds(self, dice, target_cat, game):
        # Dynamic Risk calculation
        deficit = max(0, ((13-game.turns_left)*15) - game.total_score)
        deficit_factor = min(1.0, deficit/50.0)
        curr_risk = self.genome[2] + (deficit_factor * self.genome[3])
        
        best_holds = []
        max_score = -float('inf')
        
        # If target is Upper Section (1-6), simple hold logic is best
        if target_cat <= 6:
            return [d for d in dice if d == target_cat]
        
        # For Complex Hands, use Gradient Ascent
        idxs = range(len(dice))
        for r in range(1, 6):
            for subset in itertools.combinations(idxs, r):
                sub_dice = [dice[i] for i in subset]
                
                # PREDICTION:
                # We feed the partial hand to the brain.
                val, z = self.predict_value(sub_dice)
                
                # Alignment with Target Centroid
                manifold_map = {9:4, 10:5, 11:6, 12:8}
                align_score = 0
                if target_cat in manifold_map:
                    target = self.centroids[manifold_map[target_cat]]
                    dist = np.linalg.norm(z - target)
                    align_score = 1.0 / (dist + 0.1)
                
                # Risk Penalty
                rerolls = 5 - len(sub_dice)
                penalty = rerolls * (1.0 / (curr_risk + 0.01)) * 0.05
                
                # Combined Score
                final_score = (val * 2.0) + (align_score * self.genome[1]) - penalty
                
                if final_score > max_score:
                    max_score = final_score
                    best_holds = sub_dice
                    
        return best_holds or [max(dice)] # Fallback if empty

    def play_turn(self, game):
        dice = [random.randint(1, 6) for _ in range(5)]
        
        best_cat = -1
        
        for roll in range(3):
            # 1. Pick Target
            open_cats = game.get_open_slots()
            max_util = -float('inf')
            
            val, z = self.predict_value(dice)
            
            for cat in open_cats:
                # Potential points
                ideal = {1:3, 2:6, 3:9, 4:12, 5:15, 6:18, 7:20, 8:25, 9:25, 10:30, 11:40, 12:50, 13:20}
                pot = ideal.get(cat, 0)
                
                # Distance
                dist = 0
                manifold_map = {9:4, 10:5, 11:6, 12:8}
                if cat in manifold_map:
                    target = self.centroids[manifold_map[cat]]
                    dist = np.linalg.norm(z - target)
                elif cat <= 6:
                    dist = (5 - dice.count(cat)) * 2.0
                else: dist = 4.0 
                
                # Utility
                u = (self.genome[0] * pot) - (self.genome[1] * dist)
                
                if u > max_util:
                    max_util = u
                    best_cat = cat
            
            if roll == 2: break
            
            holds = self.choose_holds(dice, best_cat, game)
            
            # Optimization: If holding 5 dice, stop early
            if len(holds) == 5: break
            
            new_dice = [random.randint(1, 6) for _ in range(5-len(holds))]
            dice = holds + new_dice
            
        score = game.calculate_score(dice, best_cat)
        
        if score == 0:
            # Rescue
            for cat in game.get_open_slots():
                if game.calculate_score(dice, cat) > 0:
                    best_cat=cat; score=game.calculate_score(dice, cat); break
            if score == 0:
                # Dump
                opts = game.get_open_slots()
                if 1 in opts: best_cat=1
                elif 12 in opts: best_cat=12
                elif opts: best_cat=opts[0]
                
        game.commit_score(best_cat, score)

def run_sim(data_dir):
    print("--- GRADIENT ASCENT YAHTZEE ---")
    
    # Init Brain
    brain = ValueBrain()
    try:
        brain.load_state_dict(torch.load(os.path.join(data_dir, "model.pth")))
    except RuntimeError as e:
        print(f"Error loading model: {e}")
        return
        
    brain.eval()
    
    # Load Centroids
    Z = np.load(os.path.join(data_dir, "latents.npy"))
    Y = np.load(os.path.join(data_dir, "labels.npy"))
    centroids = {}
    for lid in [4, 5, 6, 8]:
        mask = (Y == lid)
        if np.sum(mask) > 0: centroids[lid] = np.mean(Z[mask], axis=0)
        
    # Population
    pop_size = 50
    base_genome = [1.4, 0.4, 1.0, 1.5]
    population = [GradientAgent(brain, centroids, base_genome) for _ in range(pop_size)]
    
    goal = 170
    
    for gen in range(1, 101):
        scores = []
        for agent in population:
            g = YahtzeeGame()
            while g.turns_left > 0: agent.play_turn(g)
            scores.append(g.total_score)
            
        avg = np.mean(scores)
        top = np.max(scores)
        
        print(f"GEN {gen}: Avg {avg:.1f} | Max {top} | Goal {goal}")
        
        if avg > goal:
            goal += 5
            print(f">> UPGRADE: New Goal {goal}")
            
        # Selection
        ranks = np.argsort(scores)[::-1]
        parents = [population[i] for i in ranks[:10]]
        
        best = parents[0]
        # print(f"   Top: G={best.genome[0]:.2f} F={best.genome[1]:.2f} R={best.genome[2]:.2f}")

        new_pop = []
        while len(new_pop) < pop_size:
            p = random.choice(parents)
            child_genome = np.array(p.genome) + np.random.normal(0, 0.1, 4)
            child_genome = np.clip(child_genome, 0.1, 5.0)
            new_pop.append(GradientAgent(brain, centroids, list(child_genome)))
            
        population = new_pop

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/yahtzee_value")
    args = ap.parse_args()
    if not os.path.exists(args.data):
        print("Run setup_value_yahtzee.py first.")
    else:
        run_sim(args.data)