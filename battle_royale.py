#!/usr/bin/env python3
# BBIT BATTLE ROYALE
# Pits the top 3 evolutionary architectures against each other in a 1,000-game match.

import os, argparse, copy, random, itertools, time
import numpy as np
import torch
import torch.nn as nn
from collections import Counter

# --- SHARED BRAIN ---
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
        self.upper_counts = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0} # Needed for Precision Agent
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
            self.upper_counts[category] = score // category
        self.turns_left -= 1
    def check_bonus(self): return 35 if self.upper_score >= 63 else 0

# ==========================================
# 1. THE BONUS HUNTER (Dynamic v2)
# ==========================================
class BonusHunterAgent:
    def __init__(self, brain, centroids):
        self.brain = brain
        self.centroids = centroids
        self.name = "Bonus Hunter (Dyn v2)"
        # The Evolved Genome (Gen 85 Peak)
        self.genome = [1.4, 0.4, -0.6, 0.5, 0.0, 1.5, 1.0, 2.0, 2.0, 5.0]

    def get_z(self, dice):
        c = [0]*6
        for d in dice: c[d-1] += 1
        with torch.no_grad(): return self.brain(torch.tensor([c], dtype=torch.float32)).numpy()[0]

    def play_turn(self, game):
        dice = [random.randint(1,6) for _ in range(5)]
        
        # Modulate
        tp = 13 - game.turns_left
        exp = tp * 16; def_ = max(0, exp - game.total_score)
        exp_up = (tp/13)*63; up_def = max(0, exp_up - game.upper_score)
        
        c_risk = self.genome[2] + (tp/13 * self.genome[5]) + (min(1, def_/50) * self.genome[7])
        c_obsess = self.genome[9] * (1.0 + (up_def/10))
        
        best_cat = -1
        for roll in range(3):
            # Target
            open_cats = game.get_open_slots()
            max_u = -999
            
            z = self.get_z(dice)
            for cat in open_cats:
                ideal = {1:3, 2:6, 3:9, 4:12, 5:15, 6:18, 7:20, 8:25, 9:25, 10:30, 11:40, 12:50, 13:20}
                pot = ideal.get(cat,0)
                
                # Dist
                map_ = {9:4, 10:5, 11:6, 12:8}
                if cat in map_: dist = np.linalg.norm(z - self.centroids[map_[cat]])
                else: dist = (5 - dice.count(cat))*2.0 if cat<=6 else 4.0
                
                u = (self.genome[0]*pot) - (self.genome[1]*dist)
                if cat<=6:
                    u += self.genome[8]*10
                    if dice.count(cat)>=3: u += c_obsess*20
                
                if u > max_u: max_u=u; best_cat=cat
            
            if roll == 2: break
            
            # Hold
            if best_cat <= 6: holds = [d for d in dice if d == best_cat]
            elif best_cat in map_:
                # Vector hold
                best_h = []
                min_c = 999
                t_vec = self.centroids[map_[best_cat]]
                for r in range(1,6):
                    for sub in itertools.combinations(range(5), r):
                        sd = [dice[i] for i in sub]
                        d = np.linalg.norm(self.get_z(sd) - t_vec)
                        pen = (5-len(sd)) * (1.0/(c_risk+0.01))
                        if d+pen < min_c: min_c=d+pen; best_h=sd
                holds = best_h
            else:
                c = Counter(dice)
                holds = [d for d in dice if c[d]>=2] or [max(dice)]
            
            if len(holds)==5: break
            dice = holds + [random.randint(1,6) for _ in range(5-len(holds))]
            
        # Score
        s = game.calculate_score(dice, best_cat)
        if s == 0:
            for cat in game.get_open_slots():
                if game.calculate_score(dice, cat)>0: best_cat=cat; s=game.calculate_score(dice, cat); break
            if s==0:
                opts=game.get_open_slots()
                if 1 in opts: best_cat=1
                elif 12 in opts: best_cat=12
                else: best_cat=opts[0]
        game.commit_score(best_cat, s)

# ==========================================
# 2. THE HEDGE FUND AGENT
# ==========================================
class HedgeAgent:
    def __init__(self, brain, centroids):
        self.brain = brain
        self.centroids = centroids
        self.name = "Hedge Fund"
        # Best Genome
        self.genome = [1.0, 1.5, 1.0, 3.0, 5.0, 2.0, 2.0]

    def get_z(self, dice):
        c = [0]*6
        for d in dice: c[d-1] += 1
        with torch.no_grad(): return self.brain(torch.tensor([c], dtype=torch.float32)).numpy()[0]

    def play_turn(self, game):
        dice = [random.randint(1,6) for _ in range(5)]
        
        # Risk Calc
        tp = 13 - game.turns_left
        diff = game.total_score - (tp*16)
        base = self.genome[2]
        if diff < 0: risk = base + (abs(diff)/20 * self.genome[5])
        else: risk = base / (1.0 + (diff/20 * self.genome[6]))
        risk = max(0.1, risk)
        
        best_cat = -1
        for roll in range(3):
            open_cats = game.get_open_slots()
            max_roi = -999
            
            z = self.get_z(dice)
            for cat in open_cats:
                ideal = {1:3, 2:6, 3:9, 4:12, 5:15, 6:18, 7:20, 8:25, 9:25, 10:30, 11:40, 12:50, 13:20}
                rew = ideal.get(cat,0)
                if cat<=6: rew *= self.genome[3]
                if cat==12: rew *= self.genome[4]
                
                map_ = {9:4, 10:5, 11:6, 12:8}
                if cat in map_: dist = np.linalg.norm(z - self.centroids[map_[cat]])
                else: dist = (5 - dice.count(cat))*1.5 if cat<=6 else 4.0
                
                roi = (self.genome[0]*rew) / ((dist**self.genome[1])+0.1)
                if roi > max_roi: max_roi=roi; best_cat=cat
            
            if roll == 2: break
            
            # Holds
            if best_cat<=6: holds = [d for d in dice if d==best_cat]
            elif best_cat in map_:
                t_vec = self.centroids[map_[best_cat]]
                best_h = []
                min_c = 999
                for r in range(1,6):
                    for sub in itertools.combinations(range(5), r):
                        sd = [dice[i] for i in sub]
                        d = np.linalg.norm(self.get_z(sd)-t_vec)
                        pen = (5-len(sd))*(1.0/risk)
                        if d+pen < min_c: min_c=d+pen; best_h=sd
                holds = best_h
            else:
                c = Counter(dice)
                holds = [d for d in dice if c[d]>=2] or [max(dice)]
                
            if len(holds)==5: break
            dice = holds + [random.randint(1,6) for _ in range(5-len(holds))]
            
        s = game.calculate_score(dice, best_cat)
        if s==0:
            for cat in game.get_open_slots():
                if game.calculate_score(dice, cat)>0: best_cat=cat; s=game.calculate_score(dice, cat); break
            if s==0:
                opts=game.get_open_slots()
                if 1 in opts: best_cat=1
                elif 12 in opts: best_cat=12
                else: best_cat=opts[0]
        game.commit_score(best_cat, s)

# ==========================================
# 3. APEX PREDATOR (Hybrid)
# ==========================================
class ApexAgent:
    def __init__(self, brain, centroids):
        self.brain = brain
        self.centroids = centroids
        self.name = "Apex Predator"
        self.genome = [1.5, 0.4, -0.6, 0.5, 0.0, 1.5, 1.0, 2.0, 3.0, 5.0, 2.0]

    def get_z(self, dice):
        c = [0]*6
        for d in dice: c[d-1] += 1
        with torch.no_grad(): return self.brain(torch.tensor([c], dtype=torch.float32)).numpy()[0]

    def play_turn(self, game):
        # (Simplified Logic for speed - keeps vector but skips slow Monte Carlo for this battle)
        # Apex is mostly Precision Hunter + Chaos
        dice = [random.randint(1,6) for _ in range(5)]
        
        # Modulate
        tp=13-game.turns_left
        exp=tp*16; def_=max(0, exp-game.total_score)
        c_risk = self.genome[2] + (tp/13*self.genome[5]) + (min(1, def_/50)*self.genome[7])
        
        surplus=0
        for n in range(1,7):
            if game.scorecard[n] is not None: surplus += (game.upper_counts[n]-3)*n
        
        obs = self.genome[9]
        if surplus>0: obs -= surplus*self.genome[10]*0.1
        else: obs += abs(surplus)*self.genome[10]*0.2
        obs = max(0, obs)
        
        best_cat=-1
        for roll in range(3):
            open_cats=game.get_open_slots()
            max_u=-999
            z=self.get_z(dice)
            for cat in open_cats:
                ideal = {1:3, 2:6, 3:9, 4:12, 5:15, 6:18, 7:20, 8:25, 9:25, 10:30, 11:40, 12:50, 13:20}
                pot=ideal.get(cat,0)
                map_={9:4, 10:5, 11:6, 12:8}
                if cat in map_: dist=np.linalg.norm(z-self.centroids[map_[cat]])
                else: dist=(5-dice.count(cat))*2.0 if cat<=6 else 4.0
                
                u=(self.genome[0]*pot)-(self.genome[1]*dist)
                if cat<=6:
                    u+=self.genome[8]*10
                    if dice.count(cat)>=3: u+=obs*20
                if cat==12 and dice.count(dice[0])==5: u+=1000
                if u>max_u: max_u=u; best_cat=cat
            
            if roll==2: break
            
            if best_cat<=6: holds=[d for d in dice if d==best_cat]
            elif best_cat in map_:
                t_vec=self.centroids[map_[best_cat]]
                best_h=[]; min_c=999
                for r in range(1,6):
                    for sub in itertools.combinations(range(5),r):
                        sd=[dice[i] for i in sub]
                        d=np.linalg.norm(self.get_z(sd)-t_vec)
                        r_val = c_risk if abs(c_risk)>0.01 else 0.01
                        pen=(5-len(sd))*(1.0/r_val)
                        if d+pen<min_c: min_c=d+pen; best_h=sd
                holds=best_h
            else:
                c=Counter(dice); holds=[d for d in dice if c[d]>=2] or [max(dice)]
            
            if len(holds)==5: break
            dice=holds+[random.randint(1,6) for _ in range(5-len(holds))]
            
        s=game.calculate_score(dice, best_cat)
        if s==0:
            for cat in game.get_open_slots():
                if game.calculate_score(dice, cat)>0: best_cat=cat; s=game.calculate_score(dice, cat); break
            if s==0:
                opts=game.get_open_slots()
                if 1 in opts: best_cat=1
                elif 12 in opts: best_cat=12
                else: best_cat=opts[0]
        game.commit_score(best_cat, s)

# --- TOURNAMENT ---
def run_tournament(data_dir):
    print("--- BBIT BATTLE ROYALE ---")
    print("Pitting Champions against each other in 1,000 games...")
    
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
        
    agents = [
        BonusHunterAgent(brain, centroids),
        HedgeAgent(brain, centroids),
        ApexAgent(brain, centroids)
    ]
    
    results = {a.name: [] for a in agents}
    
    start_time = time.time()
    
    for i in range(1000):
        if i % 100 == 0: print(f"Playing Match {i}...")
        
        # Each agent plays a game with independent dice (fair comparison of skill over N)
        for agent in agents:
            g = YahtzeeGame()
            while g.turns_left > 0: agent.play_turn(g)
            score = g.total_score + g.check_bonus()
            results[agent.name].append(score)
            
    print("\n--- FINAL RESULTS (1,000 Games) ---")
    print(f"{'AGENT':<25} | {'AVG':<8} | {'MAX':<8} | {'MIN':<8} | {'>200':<8} | {'>300':<8}")
    print("-" * 75)
    
    for name, scores in results.items():
        avg = np.mean(scores)
        mx = np.max(scores)
        mn = np.min(scores)
        over200 = sum(1 for s in scores if s >= 200)
        over300 = sum(1 for s in scores if s >= 300)
        
        print(f"{name:<25} | {avg:<8.1f} | {mx:<8} | {mn:<8} | {over200:<8} | {over300:<8}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/yahtzee_BBIT")
    args = ap.parse_args()
    if not os.path.exists(args.data):
        if os.path.exists("./data/yahtzee_value"): args.data = "./data/yahtzee_value"
    run_tournament(args.data)