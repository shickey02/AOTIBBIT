#!/usr/bin/env python3
# EXPERT SYSTEM EVOLUTION (MASSIVE SCALE)
# Removes the Neural Net. Focuses on evolving "Cliff Logic" parameters.
# Population: 1,000 | Speed: ~500,000 games/hour.

import os, random, itertools, time, pickle
import numpy as np
from collections import Counter
from multiprocessing import Pool, cpu_count

# --- GAME ENGINE (Optimized) ---
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

# --- THE EXPERT AGENT (Stateless Worker) ---
def play_expert_game(args):
    genome, seed = args
    # GENOME MAP:
    # 0: Upper Bias (Base)
    # 1: Yahtzee Bias
    # 2: Risk Tolerance (Base)
    # 3: Panic Threshold (Score Deficit)
    # 4: Bonus Cliff Multiplier (Impact of being close to 63)
    # 5: Straight Focus (Weight for 10/11)
    # 6: Full House Focus
    # 7: Chance Dump Threshold (Only dump if score < X)
    
    random.seed(seed)
    game = YahtzeeGame()
    
    # Precompute static targets for distance calc
    # (Simplified vector logic: Just counts)
    
    while game.turns_left > 0:
        dice = [random.randint(1, 6) for _ in range(5)]
        
        # --- STATE ANALYSIS ---
        tp = 13 - game.turns_left
        exp = tp * 15.5
        deficit = max(0, exp - game.total_score)
        
        # Risk Modulation
        risk = genome[2]
        if deficit > genome[3]: risk *= 2.0 # Panic Mode
        
        # Bonus Proximity (The Cliff)
        dist_to_63 = 63 - game.upper_score
        bonus_urgency = 1.0
        if 0 < dist_to_63 <= 15: # In the "Kill Zone"
            bonus_urgency = genome[4]
            
        best_cat = -1
        
        for roll in range(3):
            open_cats = game.get_open_slots()
            max_util = -float('inf')
            
            # --- DECISION LOGIC ---
            for cat in open_cats:
                # 1. Base Score Potential
                ideal = {1:3, 2:6, 3:9, 4:12, 5:15, 6:18, 7:20, 8:25, 9:25, 10:30, 11:40, 12:50, 13:20}
                pot = ideal.get(cat, 0)
                
                # 2. Distance Cost
                dist = 0
                if cat <= 6: dist = (5 - dice.count(cat)) * 2.0
                elif cat == 12: dist = (5 - max(Counter(dice).values())) * 3.0
                elif cat == 11: 
                    u = len(set(dice))
                    dist = (5 - u) * 1.5 # Rough approximation for Straight
                else: dist = 4.0
                
                # 3. Utility Calculation
                u = pot - (dist * 5.0) # Base formula
                
                # 4. Modifiers (The Genome at work)
                if cat <= 6:
                    u += genome[0] * 10
                    u *= bonus_urgency # The Cliff Multiplier
                    
                    # Exact Match Bonus (Filling the Gap)
                    # If we need 10 points to get bonus, and this category gives 10+
                    if dist_to_63 > 0:
                        potential_score = dice.count(cat) * cat
                        if potential_score >= dist_to_63:
                            u += 1000 # CRITICAL: LOCK IT IN
                            
                if cat == 12: u += genome[1] * 10
                if cat == 10 or cat == 11: u += genome[5] * 5
                if cat == 9: u += genome[6] * 5
                
                if u > max_util: max_util = u; best_cat = cat
            
            if roll == 2: break
            
            # --- HOLD LOGIC ---
            holds = []
            if best_cat <= 6: 
                holds = [d for d in dice if d == best_cat]
            elif best_cat == 12:
                # Keep most common
                c = Counter(dice); mode = c.most_common(1)[0][0]
                holds = [d for d in dice if d == mode]
            elif best_cat == 11 or best_cat == 10:
                # Keep sequence
                u = sorted(list(set(dice)))
                best_seq = []
                for i in range(len(u)):
                    seq = [u[i]]
                    for j in range(i+1, len(u)):
                        if u[j] == seq[-1]+1: seq.append(u[j])
                    if len(seq) > len(best_seq): best_seq = seq
                holds = [d for d in dice if d in best_seq]
            else:
                # Default: Keep pairs/triples
                c = Counter(dice)
                holds = [d for d in dice if c[d] >= 2]
                
            # Risk Injection (Randomize holds if losing)
            if random.random() > risk:
                # If we are "Safe", we hold. 
                # If we are "Risky", we might drop a dice to fish?
                # For this simplified model, we trust the holds.
                pass
                
            if len(holds) == 5: break
            dice = holds + [random.randint(1, 6) for _ in range(5-len(holds))]
            
        s = game.calculate_score(dice, best_cat)
        
        # --- PANIC DUMP (The Safety Net) ---
        if s == 0:
            # Can we score > 0 elsewhere?
            for cat in game.get_open_slots():
                if game.calculate_score(dice, cat) > 0:
                    best_cat = cat; s = game.calculate_score(dice, cat); break
            
            if s == 0:
                # Must take a zero.
                # Dump Priority: 1s -> 2s -> Yahtzee -> Chance(if < threshold)
                opts = game.get_open_slots()
                prio = [1, 2, 12, 3, 4, 5, 6] 
                best_cat = opts[0]
                for p in prio:
                    if p in opts: best_cat = p; break
                    
        game.commit_score(best_cat, s)
        
    return game.total_score + game.check_bonus()

# --- EVOLUTION CORE ---
def run_evolution():
    print("--- EXPERT SYSTEM MASSIVE EVOLUTION ---")
    
    # 0:Up, 1:Yhtz, 2:Risk, 3:Panic, 4:Cliff, 5:Str, 6:FH, 7:Chnc
    base_genome = [2.0, 5.0, 0.8, 15.0, 3.0, 1.0, 1.0, 18.0]
    
    POP_SIZE = 1000
    population = [list(np.array(base_genome) + np.random.normal(0, 0.5, 8)) for _ in range(POP_SIZE)]
    hof = []
    
    cores = cpu_count()
    print(f"   [Running on {cores} Cores]")
    
    try:
        with Pool(cores) as pool:
            for gen in range(1, 10001):
                t0 = time.time()
                # Task: (Genome, RandomSeed)
                tasks = [(g, random.randint(0, 1000000)) for g in population]
                scores = pool.map(play_expert_game, tasks)
                
                avg = np.mean(scores); mx = np.max(scores)
                
                # Update HoF
                for s, g in zip(scores, population): hof.append((s, g))
                hof.sort(key=lambda x: x[0], reverse=True)
                hof = hof[:100]
                
                print(f"GEN {gen}: Avg {avg:.1f} | Max {mx} | HoF {hof[0][0]} | {time.time()-t0:.1f}s")
                
                # Breeding (Elitism + HoF Injection)
                ranks = np.argsort(scores)[::-1]
                elites = [population[i] for i in ranks[:50]]
                legends = [x[1] for x in hof]
                
                new_pop = list(elites)
                while len(new_pop) < POP_SIZE:
                    if random.random() < 0.2: parent = random.choice(legends) # 20% Memory
                    else: parent = random.choice(elites) # 80% Adaptation
                    
                    child = np.array(parent) + np.random.normal(0, 0.2, 8)
                    new_pop.append(list(child))
                population = new_pop
                
    except KeyboardInterrupt:
        print("Stopping.")

if __name__ == "__main__":
    run_evolution()