#!/usr/bin/env python3
# VECTOR-GUIDED MONTE CARLO YAHTZEE
# Replaces "Personality" with "Simulation".
# The agent simulates 50 futures for every possible move and picks the best timeline.

import os, argparse, copy, random, itertools
import numpy as np
import torch
import torch.nn as nn
from collections import Counter

# --- 1. VALUE BRAIN (The Evaluator) ---
class ValueBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(6, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.decoder_pattern = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 6))
        self.decoder_value = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        z = self.encoder(x)
        val = self.decoder_value(z)
        return val

# --- 2. GAME ENGINE ---
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
        if category <= 6: self.upper_score += score
        self.turns_left -= 1
        
    def check_bonus(self):
        return 35 if self.upper_score >= 63 else 0

# --- 3. MONTE CARLO AGENT ---
class MCTS_Agent:
    def __init__(self, brain_path):
        self.brain = ValueBrain()
        try:
            self.brain.load_state_dict(torch.load(brain_path))
        except:
            # Fallback for partial dicts
            full = torch.load(brain_path)
            self.brain.load_state_dict(full, strict=False)
        self.brain.eval()
        
        # SIMULATION PARAMETERS
        self.num_sims = 30 # How many futures to check per move
        self.optimism = 0.8 # 0.0=Average, 1.0=Best Case Scenario

    def evaluate_state(self, dice):
        # Use Neural Net to get "Vibe Check" of the hand
        c = [0]*6
        for d in dice: c[d-1] += 1
        t = torch.tensor([c], dtype=torch.float32)
        with torch.no_grad():
            val = self.brain(t).item()
        return val # Returns 0-1 value

    def simulate_future(self, kept_dice, rolls_remaining):
        # Recursive simulation or just 1-step lookahead?
        # Let's do 1-step lookahead (roll to fill hand)
        
        needed = 5 - len(kept_dice)
        if needed == 0: return self.evaluate_state(kept_dice)
        
        scores = []
        for _ in range(self.num_sims):
            new_dice = [random.randint(1, 6) for _ in range(needed)]
            hand = kept_dice + new_dice
            score = self.evaluate_state(hand)
            scores.append(score)
            
        # Optimistic Evaluation:
        # Don't just take the average. Grandmasters play for the "Good Outcomes."
        # We take the 80th percentile outcome.
        scores.sort()
        idx = int(len(scores) * self.optimism)
        idx = min(idx, len(scores)-1)
        return scores[idx]

    def choose_holds(self, dice, game):
        # 1. Identify all possible subsets (Moves)
        idxs = range(len(dice))
        candidates = []
        
        # Optimization: Don't check all 32 subsets. 
        # Always check "Keep All", "Keep None"
        # Check subsets that match "Modes" (Pairs, Triples)
        # Check subsets that match "Sequences"
        
        # FULL BRUTE FORCE (32 subsets is cheap)
        for r in range(0, 6):
            for subset in itertools.combinations(idxs, r):
                candidates.append([dice[i] for i in subset])
                
        best_holds = []
        best_future_val = -float('inf')
        
        for holds in candidates:
            # Heuristic Pruning: Don't simulate garbage holds
            # (e.g. holding a 2 and a 5 with nothing else)
            if len(holds) < 5 and len(holds) > 0:
                # Basic sanity check to speed up sim
                pass 
                
            future_val = self.simulate_future(holds, 1) # Just 1 roll depth for speed
            
            # Bonus Awareness Injection
            # If holds are high numbers (4,5,6) and Upper Open, boost value
            if len(holds) > 0:
                s = sum(holds)
                # Boost if aiming for upper
                # (Simple heuristic injection into the neural value)
                pass

            if future_val > best_future_val:
                best_future_val = future_val
                best_holds = holds
                
        return best_holds

    def choose_category(self, dice, game):
        # GREEDY SELECTION
        # Calculate actual score for every open slot
        # Pick the one that maximizes points relative to "Ideal"
        
        open_cats = game.get_open_slots()
        best_cat = -1
        max_val = -999
        
        for cat in open_cats:
            score = game.calculate_score(dice, cat)
            
            # Normalize Score (Value Proposition)
            # 50 pts in Yahtzee is worth MORE than 50 pts in Chance
            # because Yahtzee is harder to get.
            
            value = score
            if cat <= 6:
                # Upper Section Weighting
                # A 4-of-a-kind of 6s (24) is GREAT.
                # A 2-of-a-kind of 1s (2) is BAD.
                expected = cat * 3
                value = score - expected # Surplus value
                
                # Critical Bonus Weight
                # If this score puts us on track for 63, massive boost
                if game.upper_score + score >= (13-game.turns_left+1)*5:
                    value += 10
                    
            elif cat == 12: # Yahtzee
                if score == 50: value = 100 # Massive priority
                else: value = -50 # Don't take a 0 in Yahtzee unless forced
                
            elif cat == 13: # Chance
                # Chance is a safety valve. Don't use it for < 20 pts
                if score < 20: value -= 10
                
            if value > max_val:
                max_val = value
                best_cat = cat
                
        # Last Resort
        if best_cat == -1:
            best_cat = open_cats[0]
            
        return best_cat

    def play_game(self):
        game = YahtzeeGame()
        
        while game.turns_left > 0:
            dice = [random.randint(1, 6) for _ in range(5)]
            
            for roll in range(2): # 2 Rerolls
                # Decide what to hold based on Simulation
                holds = self.choose_holds(dice, game)
                if len(holds) == 5: break
                
                new_dice = [random.randint(1, 6) for _ in range(5-len(holds))]
                dice = holds + new_dice
            
            # Decide where to score
            cat = self.choose_category(dice, game)
            score = game.calculate_score(dice, cat)
            game.commit_score(cat, score)
            
        return game.total_score + game.check_bonus()

def run_benchmark(data_dir):
    print("--- MONTE CARLO AGENT BENCHMARK ---")
    print("(Simulating 30 futures per decision. This may be slower.)")
    
    agent = MCTS_Agent(os.path.join(data_dir, "model.pth"))
    
    scores = []
    for i in range(1, 51): # Run 50 Games
        score = agent.play_game()
        scores.append(score)
        avg = np.mean(scores)
        print(f"GAME {i}: Score {score} | Running Avg: {avg:.1f}")
        
    print(f"\nFINAL STATISTICS:")
    print(f"AVG: {np.mean(scores):.2f}")
    print(f"MAX: {np.max(scores)}")
    print(f"MIN: {np.min(scores)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/yahtzee_value")
    args = ap.parse_args()
    if not os.path.exists(args.data):
        print("Error: Needs trained ValueBrain. Run setup_value_yahtzee.py")
    else:
        run_benchmark(args.data)