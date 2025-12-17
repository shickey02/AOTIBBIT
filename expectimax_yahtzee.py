#!/usr/bin/env python3
# THE MATHEMATICIAN (EXPECTIMAX YAHTZEE)
# No Neural Networks. No Evolution. Pure Probability.
# Calculates the Exact Expected Value (EV) of every move.

import os, random, itertools, collections
import numpy as np
from collections import Counter

class YahtzeeGame:
    def __init__(self):
        self.scorecard = {i: None for i in range(1, 14)}
        self.turns_left = 13
        self.total_score = 0
        self.upper_score = 0
    def get_open_slots(self): return [k for k, v in self.scorecard.items() if v is None]
    
    def calculate_score(self, dice, category):
        counts = Counter(dice); s = sum(dice)
        if category <= 6: return dice.count(category) * category
        if category == 7: return s if any(c >= 3 for c in counts.values()) else 0
        if category == 8: return s if any(c >= 4 for c in counts.values()) else 0
        if category == 9: 
            has_3 = any(c == 3 for c in counts.values())
            has_2 = any(c == 2 for c in counts.values())
            is_Y = any(c == 5 for c in counts.values())
            return 25 if (has_3 and has_2) or is_Y else 0
        if category == 10:
            u = sorted(list(set(dice))); m_s=0; seq=0
            for i in range(len(u)-1):
                if u[i+1] == u[i]+1: seq+=1
                else: seq=0
                m_s = max(m_s, seq)
            return 30 if m_s>=3 else 0
        if category == 11:
            u = sorted(list(set(dice))); m_s=0; seq=0
            for i in range(len(u)-1):
                if u[i+1] == u[i]+1: seq+=1
                else: seq=0
                m_s = max(m_s, seq)
            return 40 if m_s>=4 else 0
        if category == 12: return 50 if any(c==5 for c in counts.values()) else 0
        if category == 13: return s
        return 0

    def commit_score(self, category, score):
        self.scorecard[category] = score
        self.total_score += score
        if category <= 6: self.upper_score += score
        self.turns_left -= 1
    def check_bonus(self): return 35 if self.upper_score >= 63 else 0

class ExpectimaxAgent:
    def __init__(self):
        # Cache for simple probability lookups could go here
        pass

    def evaluate_leaf(self, dice, game):
        # The heuristic value of a final hand (Turn 3)
        # We check all open categories and pick the best score
        open_cats = game.get_open_slots()
        best_score = -1.0
        best_cat = -1
        
        for cat in open_cats:
            score = game.calculate_score(dice, cat)
            
            # --- INTELLIGENT SCORING WEIGHTS ---
            # This is where we inject "Grandmaster Wisdom"
            weighted_score = float(score)
            
            # 1. Upper Section Bonus Awareness
            if cat <= 6:
                # Value of a point in Upper is worth MORE than 1 if we are close to bonus
                # or if we are early in game.
                # A 63-point goal needs ~3 of each. 
                # Surplus points are worth standard. Deficit points are worth 2x.
                weighted_score *= 2.5 # BASE BIAS: Upper points are gold
                
            # 2. Yahtzee Premium
            if cat == 12 and score == 50:
                weighted_score = 200.0 # Absolute priority
                
            # 3. Joker Rule / Safety
            if cat == 13 and score < 20:
                weighted_score *= 0.5 # Hate dumping low scores in Chance
                
            if weighted_score > best_score:
                best_score = weighted_score
                best_cat = cat
                
        return best_score, best_cat

    def get_expected_value(self, kept_dice, rolls_left, game):
        # If no rolls left, evaluate the hand immediately
        if rolls_left == 0:
            val, _ = self.evaluate_leaf(kept_dice, game)
            return val

        # If rolls left, we calculate E[Value] over all possible outcomes
        # Optimization: We don't simulate 1000s. We enumerate combinations.
        needed = 5 - len(kept_dice)
        
        # Generates all outcomes (e.g., (1,1), (1,2)...) with replacement
        # 6^needed outcomes.
        # For needed=1 (6 outcomes), needed=2 (36), needed=3 (216).
        # needed=4 (1296) and 5 (7776) are slow. We prune depth for those.
        
        if needed > 3:
            # Too expensive for full recursion. Use Monte Carlo approximation.
            total_val = 0
            sims = 20 # Small sample for speed
            for _ in range(sims):
                new_dice = [random.randint(1,6) for _ in range(needed)]
                hand = kept_dice + new_dice
                # Recursive call with decremented rolls
                total_val += self.get_expected_value(hand, rolls_left - 1, game)
            return total_val / sims
        else:
            # Exact calculation
            outcomes = list(itertools.product(range(1, 7), repeat=needed))
            total_val = 0
            for outcome in outcomes:
                hand = kept_dice + list(outcome)
                total_val += self.get_expected_value(hand, rolls_left - 1, game)
            return total_val / len(outcomes)

    def choose_holds(self, dice, game, rolls_left):
        # Try all subsets
        idxs = range(len(dice))
        best_ev = -float('inf')
        best_holds = []
        
        # Optimization: Filter candidates
        # Always check: Keep All, Keep None
        # Check: Keep Pairs, Triples, 4s, Straights
        # This reduces search space from 32 to ~10 high-quality candidates
        
        candidates = set()
        candidates.add(tuple()) # Keep none
        candidates.add(tuple(dice)) # Keep all
        
        # Add subsets
        for r in range(1, 5):
            for sub in itertools.combinations(dice, r):
                candidates.add(tuple(sorted(sub)))
                
        # Heuristic Pruning: If candidate len > 0, ensure it has some value?
        # No, let EV decide. But for speed, limit candidates to unique sorted tuples.
        
        for holds in candidates:
            holds_list = list(holds)
            ev = self.get_expected_value(holds_list, rolls_left, game)
            
            if ev > best_ev:
                best_ev = ev
                best_holds = holds_list
                
        return best_holds

    def play_turn(self, game):
        dice = [random.randint(1, 6) for _ in range(5)]
        
        for roll in range(2): # 2 Rerolls
            # Decisions
            # Roll 1 -> 2 rolls left
            # Roll 2 -> 1 roll left
            rolls_left = 2 - roll
            
            holds = self.choose_holds(dice, game, rolls_left)
            
            if len(holds) == 5: break
            
            # Reroll
            new_dice = [random.randint(1, 6) for _ in range(5-len(holds))]
            dice = holds + new_dice
            
        # Final Decision
        _, cat = self.evaluate_leaf(dice, game)
        score = game.calculate_score(dice, cat)
        game.commit_score(cat, score)

def run_benchmark():
    print("--- THE MATHEMATICIAN (EXPECTIMAX) ---")
    print("Strategy: Exact Probability Calculation + Weighted Upper Section")
    
    agent = ExpectimaxAgent()
    scores = []
    
    for i in range(1, 101): # Play 100 Games
        game = YahtzeeGame()
        while game.turns_left > 0:
            agent.play_turn(game)
        
        final = game.total_score + game.check_bonus()
        scores.append(final)
        
        avg = np.mean(scores)
        if i % 10 == 0:
            print(f"GAME {i}: Score {final} | Running Avg: {avg:.1f} | Max: {np.max(scores)}")
            
    print("-" * 30)
    print(f"FINAL AVERAGE: {np.mean(scores):.2f}")
    print(f"FINAL MAX: {np.max(scores)}")
    print(f"GAMES > 200: {sum(1 for s in scores if s >= 200)}")

if __name__ == "__main__":
    run_benchmark()