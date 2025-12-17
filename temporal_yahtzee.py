#!/usr/bin/env python3
# TEMPORAL YAHTZEE (Recurrent Vector Agent)
# Adds an LSTM to the agent so it can plan across the 13-turn sequence.

import os, argparse, copy, random, itertools
import numpy as np
import torch
import torch.nn as nn
from collections import Counter

# --- 1. THE RECURRENT BRAIN ---
class TemporalBrain(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 6 (Dice) + 13 (Scorecard Status) + 1 (Time) = 20
        self.input_dim = 20
        self.hidden_dim = 32
        
        # Spatial Encoder (Process the Dice)
        self.dice_encoder = nn.Linear(6, 16)
        
        # Context Encoder (Process the Board)
        self.context_encoder = nn.Linear(14, 16)
        
        # Memory Core (LSTM)
        self.lstm = nn.LSTMCell(32, self.hidden_dim) # 16+16 input
        
        # Policy Head (Output: Latent Vector Strategy)
        self.policy = nn.Linear(self.hidden_dim, 16) # Output a 16D 'Desire Vector'

    def forward(self, dice_vec, context_vec, h, c):
        # 1. Perception
        e_dice = torch.relu(self.dice_encoder(dice_vec))
        e_ctx = torch.relu(self.context_encoder(context_vec))
        
        # 2. Integration
        fusion = torch.cat([e_dice, e_ctx], dim=1)
        
        # 3. Memory Update
        h_new, c_new = self.lstm(fusion, (h, c))
        
        # 4. Intention
        intention_vec = self.policy(h_new)
        
        return intention_vec, (h_new, c_new)

# --- 2. GAME ENGINE (Standard) ---
class YahtzeeGame:
    def __init__(self):
        self.scorecard = {i: None for i in range(1, 14)}
        self.turns_left = 13
        self.total_score = 0
        
    def get_open_slots(self):
        return [k for k, v in self.scorecard.items() if v is None]

    def get_context_vector(self):
        # Returns [Scorecard_Flags (13), Turns_Norm (1)]
        flags = [1.0 if self.scorecard[i] is not None else 0.0 for i in range(1, 14)]
        time = [self.turns_left / 13.0]
        return torch.tensor([flags + time], dtype=torch.float32)

    def calculate_score(self, dice, category):
        # (Same logic as before, abbreviated for brevity)
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

# --- 3. THE TEMPORAL AGENT ---
class TemporalAgent:
    def __init__(self, brain, centroids, genome=None):
        self.brain = brain
        self.centroids = centroids 
        # Memory State (Hidden, Cell)
        self.h = torch.zeros(1, 32)
        self.c = torch.zeros(1, 32)
        
        # GENOME: [Greed, Focus, Risk, Patience]
        if genome is None: self.genome = np.random.uniform(0.5, 2.0, 4)
        else: self.genome = genome
            
    def reset_memory(self):
        self.h = torch.zeros(1, 32)
        self.c = torch.zeros(1, 32)

    def get_intention(self, dice, game):
        # Input Construction
        c = [0]*6
        for d in dice: c[d-1] += 1
        dice_vec = torch.tensor([c], dtype=torch.float32)
        ctx_vec = game.get_context_vector()
        
        # Run LSTM
        # We don't backprop through time here (Evolutionary Strategy), 
        # so we just detach to keep it fast.
        with torch.no_grad():
            intention, (hn, cn) = self.brain(dice_vec, ctx_vec, self.h, self.c)
        
        self.h, self.c = hn, cn
        return intention.numpy()[0]

    def choose_target(self, dice, game):
        # 1. Get Intention Vector from LSTM
        # This vector represents "What I WANT the dice to look like"
        # based on my memory of what I need.
        intention = self.get_intention(dice, game)
        
        # 2. Match Intention to available Categories
        open_cats = game.get_open_slots()
        best_cat = -1
        max_util = -float('inf')
        
        manifold_map = {9:4, 10:5, 11:6, 12:8}
        
        for cat in open_cats:
            # Score Potential
            pot = 25 # Placeholder average
            if cat <= 6: pot = cat*3
            elif cat==12: pot=50
            elif cat==11: pot=40
            
            # Geometric Alignment
            # How close is the category centroid to my Intention?
            dist = 0
            if cat in manifold_map:
                target = self.centroids[manifold_map[cat]]
                dist = np.linalg.norm(intention - target)
            else:
                # For non-manifold cats (Upper section), LSTM must learn to 
                # output a vector that "looks like" Three Fives if it wants Fives.
                # Since we don't have centroids for Fives, we use the Dice Count distance.
                # This is a hybrid heuristic.
                count = dice.count(cat) if cat <= 6 else 0
                dist = (5-count)*1.5
            
            # Patience Factor (Gene[3])
            # If turns_left is high, penalize taking low-score Upper Section?
            # Or penalize taking Chance early?
            time_factor = 0
            if cat == 13 and game.turns_left > 10: time_factor = -10 * self.genome[3]
            
            # Utility
            u = (self.genome[0] * pot) - (self.genome[1] * dist) + time_factor
            if u > max_util:
                max_util = u
                best_cat = cat
                
        return best_cat

    def choose_holds(self, dice, target_cat):
        # (Same logic as before, simplified)
        if target_cat <= 6: return [d for d in dice if d == target_cat]
        if target_cat == 9: # FH
             c = Counter(dice)
             return [d for d in dice if c[d] >= 2]
        if target_cat in [10,11]: # Str
             u = sorted(list(set(dice)))
             # Try to keep sequence
             # Naive sequence finder
             best_seq = []
             for i in range(len(u)):
                 seq = [u[i]]
                 for j in range(i+1, len(u)):
                     if u[j] == seq[-1]+1: seq.append(u[j])
                 if len(seq) > len(best_seq): best_seq = seq
             return [d for d in dice if d in best_seq]
             
        # Fallback (3kind/4kind/Yahtzee/Chance)
        c = Counter(dice)
        return [d for d in dice if c[d] >= 2] or [max(dice)]

    def play_turn(self, game):
        dice = [random.randint(1, 6) for _ in range(5)]
        
        # Strategy locked after Roll 1? Or fluid?
        # Let's let it be fluid.
        
        for roll in range(3):
            best_cat = self.choose_target(dice, game)
            
            if roll == 2: break
            
            holds = self.choose_holds(dice, best_cat)
            if len(holds) == 5: break
            
            new_dice = [random.randint(1, 6) for _ in range(5-len(holds))]
            dice = holds + new_dice
            
        score = game.calculate_score(dice, best_cat)
        
        # Panic Logic
        if score == 0:
            # Try to save it
            open_cats = game.get_open_slots()
            for cat in open_cats:
                s = game.calculate_score(dice, cat)
                if s > 0: 
                    best_cat = cat; score = s; break
            if score == 0:
                # Dump
                if 1 in open_cats: best_cat=1
                elif 2 in open_cats: best_cat=2
                elif 12 in open_cats: best_cat=12 # Dump Yahtzee if hopeless
                elif open_cats: best_cat=open_cats[0]
                
        game.commit_score(best_cat, score)

def run_sim(data_dir):
    print("--- TEMPORAL EVOLUTION ---")
    
    # Init Brain (Shared Weights)
    brain = TemporalBrain()
    
    # Load Centroids
    Z = np.load(os.path.join(data_dir, "latents.npy"))
    Y = np.load(os.path.join(data_dir, "labels.npy"))
    centroids = {}
    for lid in [4, 5, 6, 8]:
        mask = (Y == lid)
        if np.sum(mask) > 0: centroids[lid] = np.mean(Z[mask], axis=0)
        
    pop_size = 50
    population = [TemporalAgent(brain, centroids) for _ in range(pop_size)]
    
    goal = 120
    
    for gen in range(1, 51):
        scores = []
        for agent in population:
            g = YahtzeeGame()
            agent.reset_memory() # Clear LSTM
            while g.turns_left > 0:
                agent.play_turn(g)
            scores.append(g.total_score)
            
        avg = np.mean(scores)
        top = np.max(scores)
        
        print(f"GEN {gen}: Avg {avg:.1f} | Max {top} | Goal {goal}")
        
        if avg > goal:
            goal += 15
            print(f">> UPGRADE: New Goal {goal}")
            
        # Select & Mutate
        ranks = np.argsort(scores)[::-1]
        parents = [population[i] for i in ranks[:10]]
        
        best = parents[0]
        print(f"   Top Genome: G={best.genome[0]:.2f} F={best.genome[1]:.2f} R={best.genome[2]:.2f} P={best.genome[3]:.2f}")
        
        new_pop = []
        while len(new_pop) < pop_size:
            p = random.choice(parents)
            # Clone brain? No, brain is shared/static in this version (Policy Evolution).
            # To do full Deep RL, we'd need backprop. 
            # Here we just evolve the *Utility Function Parameters* (Genome).
            
            child_genome = p.genome + np.random.normal(0, 0.1, 4)
            child_genome = np.clip(child_genome, 0.1, 5.0)
            new_pop.append(TemporalAgent(brain, centroids, child_genome))
            
        population = new_pop

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/yahtzee_bbit")
    args = ap.parse_args()
    if not os.path.exists(args.data):
        print(f"Error: Data not found at {args.data}. Run setup_yahtzee.py first.")
    else:
        run_sim(args.data)