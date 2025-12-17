#!/usr/bin/env python3
# FULL YAHTZEE SIMULATION (EVOLUTIONARY GEOMETRY)
# Trains a Vector Agent to master the full 13-turn game via Genetic Algorithms.

import os, argparse, copy, random, itertools
import numpy as np
import torch
import torch.nn as nn
from collections import Counter

# --- 1. THE GEOMETRIC BRAIN ---
class YahtzeeBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(6, 32), nn.ReLU(),
            nn.Linear(32, 16)
        )
    def forward(self, x):
        return self.encoder(x)

# --- 2. THE GAME ENGINE ---
class YahtzeeGame:
    def __init__(self):
        # 13 Categories
        # Upper Section: Ones(1)..Sixes(6)
        # Lower Section: 3Kind(7), 4Kind(8), FH(9), SmStr(10), LgStr(11), Yahtzee(12), Chance(13)
        self.scorecard = {i: None for i in range(1, 14)}
        self.turns_left = 13
        self.total_score = 0
        
    def get_open_slots(self):
        return [k for k, v in self.scorecard.items() if v is None]

    def calculate_score(self, dice, category):
        counts = Counter(dice)
        s = sum(dice)
        
        # Upper (1-6)
        if 1 <= category <= 6:
            return dice.count(category) * category
            
        # Lower
        if category == 7: # 3 of a Kind
            return s if any(c >= 3 for c in counts.values()) else 0
        if category == 8: # 4 of a Kind
            return s if any(c >= 4 for c in counts.values()) else 0
        if category == 9: # Full House (25)
            has_3 = any(c == 3 for c in counts.values())
            has_2 = any(c == 2 for c in counts.values())
            is_yahtzee = any(c == 5 for c in counts.values())
            return 25 if (has_3 and has_2) or is_yahtzee else 0
        if category == 10: # Sm Straight (30)
            # Flatten unique sorted
            u = sorted(list(set(dice)))
            # Check for sequence of 4
            seq = 0
            for i in range(len(u)-1):
                if u[i+1] == u[i]+1: seq+=1
                else: seq = 0
                if seq >= 3: return 30
            return 0
        if category == 11: # Lg Straight (40)
            u = sorted(list(set(dice)))
            seq = 0
            for i in range(len(u)-1):
                if u[i+1] == u[i]+1: seq+=1
                else: seq = 0
                if seq >= 4: return 40
            return 0
        if category == 12: # Yahtzee (50)
            return 50 if any(c == 5 for c in counts.values()) else 0
        if category == 13: # Chance
            return s
        return 0

    def commit_score(self, category, score):
        self.scorecard[category] = score
        self.total_score += score
        self.turns_left -= 1

# --- 3. THE GENOME AGENT ---
class VectorAgent:
    def __init__(self, brain, centroids, genome=None):
        self.brain = brain
        self.centroids = centroids # Dict of latent vectors for ideal hands
        
        # GENOME: [Greed, Focus, Risk, Bonus_Weight]
        # Greed: Importance of potential score
        # Focus: Importance of geometric closeness (Distance)
        # Risk: Penalty for needing to reroll many dice
        # Bonus: Importance of Upper Section Bonus (aiming for 63+)
        if genome is None:
            self.genome = np.random.uniform(0.5, 2.0, 4)
        else:
            self.genome = genome
            
    def get_z(self, dice):
        c = [0]*6
        for d in dice: c[d-1] += 1
        t = torch.tensor([c], dtype=torch.float32)
        with torch.no_grad():
            return self.brain(t).numpy()[0]

    def evaluate_target(self, dice, category, turns_remaining):
        # 1. Potential Score (Greed)
        # We estimate "Ideal Score" for this category
        ideal_scores = {
            1:3, 2:6, 3:9, 4:12, 5:15, 6:18, # Upper averages
            7:20, 8:25, 9:25, 10:30, 11:40, 12:50, 13:20
        }
        potential = ideal_scores[category]
        
        # 2. Geometric Distance (Focus)
        # Map current dice to Z
        z_curr = self.get_z(dice)
        
        # Map Category to a Target Vector
        # Map Yahtzee categories to our Manifold IDs
        # 1-6 map to "Pair/3kind" clusters vaguely? No, let's use the explicit clusters we found.
        # 9 (FH) -> Manifold ID 4
        # 10 (Sm) -> Manifold ID 5
        # 11 (Lg) -> Manifold ID 6
        # 12 (Y)  -> Manifold ID 8
        # For Upper section, we don't have explicit clusters, so we use "Chance" or nearest neighbor.
        
        manifold_map = {9:4, 10:5, 11:6, 12:8}
        
        dist = 0
        if category in manifold_map:
            target_vec = self.centroids[manifold_map[category]]
            dist = np.linalg.norm(z_curr - target_vec)
        else:
            # Fallback for simple categories (Ones, Twos...)
            # Distance is inversely proportional to "How many of X do I have?"
            # This is a 'soft' vector heuristic
            count = dice.count(category) if category <= 6 else 0
            dist = (5 - count) * 2.0 # Crude distance
            
        # 3. Calculate Utility
        # U = (Greed * Potential) - (Focus * Distance)
        greed, focus, risk, bonus = self.genome
        
        utility = (greed * potential) - (focus * dist)
        
        # Bonus bias for Upper Section early in game
        if category <= 6 and turns_remaining > 6:
            utility += bonus * 10
            
        return utility, dist

    def choose_holds(self, dice, target_category):
        # Brute force all hold combos to minimize distance to Target
        # If target is Upper Section (e.g. Fives), just hold the Fives.
        if target_category <= 6:
            return [d for d in dice if d == target_category]
        
        # For Lower Section, use Vector Geometry
        if target_category == 13: # Chance
            return [d for d in dice if d >= 4] # Keep high numbers
            
        # For Complex (FH, Str, Yahtzee), use Manifold
        manifold_map = {9:4, 10:5, 11:6, 12:8}
        if target_category not in manifold_map: # 3kind/4kind
             # Fallback: keep duplicates
             counts = Counter(dice)
             return [d for d in dice if counts[d] >= 2] or [max(dice)]

        target_vec = self.centroids[manifold_map[target_category]]
        
        best_holds = []
        min_dist = float('inf')
        
        # Try all subsets
        idxs = range(len(dice))
        for r in range(1, 6):
            for subset in itertools.combinations(idxs, r):
                subset_dice = [dice[i] for i in subset]
                
                # Check Vector Distance of this subset
                z_sub = self.get_z(subset_dice)
                d = np.linalg.norm(z_sub - target_vec)
                
                # Add Risk Penalty (Genome[2])
                # More dice to reroll = Higher Risk
                rerolls = 5 - len(subset_dice)
                penalty = rerolls * self.genome[2]
                
                score = d + penalty
                
                if score < min_dist:
                    min_dist = score
                    best_holds = subset_dice
                    
        return best_holds

    def play_turn(self, game):
        # 1. Roll 1
        dice = [random.randint(1, 6) for _ in range(5)]
        
        # 2. Strategy Selection (After Roll 1)
        # Scan open slots, pick best utility
        best_cat = -1
        max_util = -float('inf')
        
        open_cats = game.get_open_slots()
        
        for cat in open_cats:
            util, _ = self.evaluate_target(dice, cat, game.turns_left)
            if util > max_util:
                max_util = util
                best_cat = cat
        
        # 3. Rerolls (Roll 2 and 3)
        for _ in range(2):
            holds = self.choose_holds(dice, best_cat)
            
            # If we have 5 dice held and it fits, stop?
            # Naive: always use 3 rolls unless perfect.
            
            num_reroll = 5 - len(holds)
            if num_reroll == 0: break
            
            new_dice = [random.randint(1, 6) for _ in range(num_reroll)]
            dice = holds + new_dice
            
            # Re-evaluate target? (Advanced agent would, we stick to plan)
            
        # 4. Score
        score = game.calculate_score(dice, best_cat)
        
        # Emergency: If score is 0, check if we can score elsewhere?
        if score == 0:
            # Panic mode: Find ANY category that gives points
            for cat in open_cats:
                s = game.calculate_score(dice, cat)
                if s > 0:
                    best_cat = cat
                    score = s
                    break
            # If still 0, we must take a 0 somewhere. 
            # Strategy: Dump 0 in Yahtzee or Ones (low loss)
            if score == 0:
                # Naive dump
                if 12 in open_cats: best_cat = 12
                elif 1 in open_cats: best_cat = 1
        
        game.commit_score(best_cat, score)
        return best_cat, score, dice

# --- 4. EVOLUTION LOOP ---
def run_simulation(data_dir):
    print("--- LOADING GEOMETRY ---")
    
    # Load Brain
    brain = YahtzeeBrain()
    # Partial load logic needed because we saved full model previously? 
    # Let's assume the previous YahtzeeBrain architecture matches.
    # Actually, previous script saved 'model.pth' with encoder+classifier.
    # We just want the encoder weights.
    full_state = torch.load(os.path.join(data_dir, "model.pth"))
    # Filter for 'encoder' keys
    enc_state = {k:v for k,v in full_state.items() if 'encoder' in k}
    brain.load_state_dict(enc_state, strict=False)
    brain.eval()
    
    # Load Centroids
    Z = np.load(os.path.join(data_dir, "latents.npy"))
    Y = np.load(os.path.join(data_dir, "labels.npy"))
    centroids = {}
    for lid in [4, 5, 6, 8]: # FH, Sm, Lg, Y
        mask = (Y == lid)
        if np.sum(mask) > 0:
            centroids[lid] = np.mean(Z[mask], axis=0)
            
    print("--- INITIALIZING POPULATION ---")
    pop_size = 50
    population = [VectorAgent(brain, centroids) for _ in range(pop_size)]
    
    goal_score = 100
    generation = 1
    
    while True:
        scores = []
        
        # Run Games
        for agent in population:
            g = YahtzeeGame()
            while g.turns_left > 0:
                agent.play_turn(g)
            scores.append(g.total_score)
            
        avg_score = np.mean(scores)
        max_score = np.max(scores)
        
        print(f"GEN {generation}: Avg {avg_score:.1f} | Max {max_score} | Goal {goal_score}")
        
        # Check Success
        if avg_score > goal_score:
            print(f">> SUCCESS! Goal {goal_score} passed.")
            goal_score += 20
            if goal_score > 250:
                print(">> GRANDMASTER LEVEL ACHIEVED.")
                break
        
        # Selection (Top 20%)
        ranked_indices = np.argsort(scores)[::-1]
        top_indices = ranked_indices[:10]
        parents = [population[i] for i in top_indices]
        
        # Log Best Genome
        best_agent = parents[0]
        print(f"   Best Genome: G={best_agent.genome[0]:.2f} F={best_agent.genome[1]:.2f} R={best_agent.genome[2]:.2f}")
        
        # Reproduction
        new_pop = []
        while len(new_pop) < pop_size:
            # Crossover
            p1, p2 = random.sample(parents, 2)
            # Simple average crossover
            child_genome = (p1.genome + p2.genome) / 2
            
            # Mutation (10% noise)
            mutation = np.random.normal(0, 0.1, 4)
            child_genome += mutation
            child_genome = np.clip(child_genome, 0.0, 5.0) # Keep positive
            
            new_pop.append(VectorAgent(brain, centroids, child_genome))
            
        population = new_pop
        generation += 1
        
        # Safety break
        if generation > 120: 
            print(">> SIMULATION ENDED (Time Limit).")
            break

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/yahtzee_bbit")
    args = ap.parse_args()
    run_simulation(args.data)