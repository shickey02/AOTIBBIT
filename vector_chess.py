#!/usr/bin/env python3
# VECTOR CHESS: PLAYING BY GRAVITY
# The AI plays Hexapawn by calculating the vector to the "Winning Cluster"
# and snapping to the nearest legal move.

import os, argparse, copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors

# --- RE-DEFINE MODEL ---
class GameAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def encode(self, x):
        return self.encoder(x)

# --- ENGINE ---
class VectorEngine:
    def __init__(self, data_dir):
        self.device = torch.device("cpu")
        self.states = np.load(os.path.join(data_dir, "states.npy"))
        self.outcomes = np.load(os.path.join(data_dir, "outcomes.npy"))
        self.latents = np.load(os.path.join(data_dir, "latents.npy"))
        
        self.model = GameAE()
        self.model.load_state_dict(torch.load(os.path.join(data_dir, "model.pth"), map_location=self.device))
        self.model.eval()
        
        # Define Goals
        # Target: States where I WIN (outcome = my_color)
        self.white_wins_z = self.latents[self.outcomes == 1]
        self.black_wins_z = self.latents[self.outcomes == -1]
        
        # Calculate Gravitational Centers
        self.white_centroid = np.mean(self.white_wins_z, axis=0)
        self.black_centroid = np.mean(self.black_wins_z, axis=0)

    def get_z(self, board, turn):
        vec = np.array(board.flatten().tolist() + [turn], dtype=np.float32)
        t = torch.tensor([vec])
        with torch.no_grad():
            return self.model.encode(t).numpy()[0]

    def choose_move(self, game, gain=1.0):
        # 1. Current Position
        z_curr = self.get_z(game.board, game.turn)
        
        # 2. Determine Goal
        target_z = self.white_centroid if game.turn == 1 else self.black_centroid
        
        # 3. Calculate "Victory Vector"
        v_win = target_z - z_curr
        
        # 4. Ideal Next State (in Latent Space)
        # We assume one move gets us fractionally closer, but with high gain we pull hard.
        z_ideal = z_curr + (v_win * gain)
        
        # 5. Evaluate Legal Moves (The "Snap")
        moves = game.get_moves()
        if not moves: return None
        
        best_move = None
        best_dist = float('inf')
        
        print(f"\n// THINKING (Turn: {'White' if game.turn==1 else 'Black'})...")
        print(f">> VECTOR: [Mag: {np.linalg.norm(v_win):.2f}] [Gain: {gain}]")
        
        for m in moves:
            # Simulate
            next_game = game.apply_move(m)
            z_next = self.get_z(next_game.board, next_game.turn) # Note: Next turn is opponent's
            
            # Wait! The Z of the next state represents the OPPONENT'S turn.
            # We want the board state that leads to OUR win.
            # The latent space encodes "Board + Turn".
            # So we compare z_next to z_ideal.
            
            dist = np.linalg.norm(z_next - z_ideal)
            
            # Neologism Check: Is this a "Known Strategy"?
            # Check distance to nearest training point
            # (Simplified for output flavor)
            
            print(f"   :: OPTION: {m} -> Dist {dist:.4f}")
            
            if dist < best_dist:
                best_dist = dist
                best_move = m
                
        return best_move

# --- GAME LOOP ---
class Hexapawn:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.board[0, :] = -1
        self.board[2, :] = 1
        self.turn = 1
    def get_moves(self):
        moves = []
        d = -1 if self.turn == 1 else 1
        for r in range(3):
            for c in range(3):
                if self.board[r, c] == self.turn:
                    # Forward
                    rn = r + d
                    if 0 <= rn < 3 and self.board[rn, c] == 0:
                        moves.append(((r,c), (rn, c)))
                    # Capture
                    for cn in [c-1, c+1]:
                        if 0 <= rn < 3 and 0 <= cn < 3:
                            if self.board[rn, cn] == -self.turn:
                                moves.append(((r,c), (rn, cn)))
        return moves
    def apply_move(self, move):
        g = copy.deepcopy(self)
        (r1, c1), (r2, c2) = move
        g.board[r2, c2] = g.board[r1, c1]
        g.board[r1, c1] = 0
        g.turn *= -1
        return g
    def print_board(self):
        chars = {0: '.', 1: 'W', -1: 'B'}
        print("  0 1 2")
        for r in range(3):
            row = [chars[self.board[r, c]] for c in range(3)]
            print(f"{r} {' '.join(row)}")
    def is_over(self):
        if 1 in self.board[0, :]: return 1
        if -1 in self.board[2, :]: return -1
        if not self.get_moves(): return -self.turn
        return 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/hexapawn_bbit")
    args = ap.parse_args()
    
    brain = VectorEngine(args.data)
    game = Hexapawn()
    
    print("=== VECTOR CHESS: HEXAPAWN ===")
    game.print_board()
    
    while True:
        winner = game.is_over()
        if winner != 0:
            print(f"\nGAME OVER. Winner: {'White' if winner==1 else 'Black'}")
            break
            
        move = brain.choose_move(game, gain=2.0)
        
        # English+ Output
        (r1,c1), (r2,c2) = move
        print(f"<< SNAP: Move ({r1},{c1}) -> ({r2},{c2})")
        
        game = game.apply_move(move)
        game.print_board()
        input("[Press Enter for Next Turn]")

if __name__ == "__main__":
    main()