#!/usr/bin/env python3
# VECTOR YAHTZEE
# Decides which dice to keep by minimizing geometric distance to 'Winning Hands'.

import os, argparse, itertools, random
import numpy as np
import torch
import torch.nn as nn

class YahtzeeBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
        self.classifier = nn.Linear(16, 9)
    def forward(self, x):
        z = self.encoder(x)
        return self.classifier(z), z
    def get_z(self, x):
        return self.encoder(x)

LABELS = {
    4: "Full House", 5: "Sm. Straight", 6: "Lg. Straight", 7: "4-Kind", 8: "YAHTZEE"
}

class VectorRoller:
    def __init__(self, data_dir):
        self.device = torch.device("cpu")
        self.Z = np.load(os.path.join(data_dir, "latents.npy"))
        self.Y = np.load(os.path.join(data_dir, "labels.npy"))
        
        self.model = YahtzeeBrain()
        self.model.load_state_dict(torch.load(os.path.join(data_dir, "model.pth"), map_location=self.device))
        self.model.eval()
        
        # Calculate Target Centroids (The "Gravity Wells" of good hands)
        self.targets = {}
        for label_id in [4, 5, 6, 7, 8]: # Only care about big hands
            mask = (self.Y == label_id)
            if np.sum(mask) > 0:
                self.targets[label_id] = np.mean(self.Z[mask], axis=0)

    def to_counts(self, dice):
        c = [0]*6
        for d in dice: c[d-1] += 1
        return torch.tensor([c], dtype=torch.float32)

    def get_z(self, dice):
        with torch.no_grad():
            return self.model.get_z(self.to_counts(dice)).numpy()[0]

    def decide_keep(self, dice):
        # 1. Current State
        z_curr = self.get_z(dice)
        
        # 2. Find Closest Dream (Which target is nearest?)
        best_target_id = None
        min_dist_to_target = float('inf')
        
        for tid, tvec in self.targets.items():
            dist = np.linalg.norm(z_curr - tvec)
            # Weight targets: Yahtzee is better than Full House
            # Bias distance: subtract (ID * 0.5) to incentivize greed
            adj_dist = dist - (tid * 0.1) 
            
            if adj_dist < min_dist_to_target:
                min_dist_to_target = adj_dist
                best_target_id = tid
                
        target_vec = self.targets[best_target_id]
        target_name = LABELS[best_target_id]
        
        print(f">> TARGET ACQUIRED: {target_name} (Dist: {min_dist_to_target:.2f})")
        
        # 3. Simulate Keeps
        # We can keep any subset of dice.
        # indices = [0, 1, 2, 3, 4]
        best_keep_indices = []
        best_proj_dist = float('inf')
        
        # Try all 31 subsets (len 1 to 5)
        indices = range(len(dice))
        for r in range(1, 6):
            for subset in itertools.combinations(indices, r):
                kept_dice = [dice[i] for i in subset]
                
                # To estimate value, we assume "Perfect Luck" for the missing dice?
                # Or we just check if the Kept Dice *align* with the target vector?
                # Vector approach: map the 'partial' hand.
                # Since input is Counts, a partial hand [2, 2] is just [0, 2, 0, 0, 0, 0].
                
                z_proj = self.get_z(kept_dice)
                dist = np.linalg.norm(z_proj - target_vec)
                
                # Penalize dropping too many dice (Drift penalty)
                penalty = (5 - len(kept_dice)) * 0.2
                final_dist = dist + penalty
                
                if final_dist < best_proj_dist:
                    best_proj_dist = final_dist
                    best_keep_indices = list(subset)
        
        kept_values = [dice[i] for i in best_keep_indices]
        return kept_values, target_name

def play_round(bot):
    print("\n--- NEW ROUND ---")
    dice = [random.randint(1, 6) for _ in range(5)]
    rolls_left = 2
    
    while rolls_left >= 0:
        print(f"ROLL: {dice}")
        if rolls_left == 0: break
        
        # Bot Decides
        kept, target = bot.decide_keep(dice)
        print(f"AI Strategy: Aiming for {target}. Keeping {kept}")
        
        if len(kept) == 5:
            print("AI STANDS.")
            break
            
        # Reroll rest
        num_reroll = 5 - len(kept)
        new_dice = [random.randint(1, 6) for _ in range(num_reroll)]
        dice = kept + new_dice
        rolls_left -= 1
        
    print(f"FINAL HAND: {dice}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/yahtzee_bbit")
    args = ap.parse_args()
    bot = VectorRoller(args.data)
    
    for _ in range(3):
        play_round(bot)
        input("[Enter]...")

if __name__ == "__main__":
    main()