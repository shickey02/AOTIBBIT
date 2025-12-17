#!/usr/bin/env python3
# SETUP SCRIPT: HEXAPAWN MANIFOLD
# Generates the complete state space of 3x3 Chess (Hexapawn) and trains a Game Brain.

import os, argparse, json, copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- GAME ENGINE ---
class Hexapawn:
    def __init__(self):
        # 3x3 Board. 1=White, -1=Black, 0=Empty
        self.board = np.zeros((3, 3), dtype=int)
        self.board[0, :] = -1 # Black
        self.board[2, :] = 1  # White
        self.turn = 1         # White moves first

    def get_moves(self):
        moves = [] # (r1, c1, r2, c2)
        direction = -1 if self.turn == 1 else 1
        
        for r in range(3):
            for c in range(3):
                if self.board[r, c] == self.turn:
                    # 1. Move Forward
                    r_next = r + direction
                    if 0 <= r_next < 3:
                        if self.board[r_next, c] == 0:
                            moves.append(((r,c), (r_next, c)))
                        
                        # 2. Capture Diagonal
                        for c_next in [c-1, c+1]:
                            if 0 <= c_next < 3:
                                target = self.board[r_next, c_next]
                                if target == -self.turn:
                                    moves.append(((r,c), (r_next, c_next)))
        return moves

    def apply_move(self, move):
        new_game = copy.deepcopy(self)
        (r1, c1), (r2, c2) = move
        new_game.board[r2, c2] = new_game.board[r1, c1]
        new_game.board[r1, c1] = 0
        new_game.turn *= -1
        return new_game

    def is_win(self):
        # Win if reached other side
        if 1 in self.board[0, :]: return 1  # White Wins
        if -1 in self.board[2, :]: return -1 # Black Wins
        # Win if opponent has no moves
        if not self.get_moves():
            return -self.turn # Current player stuck -> Other wins
        return 0

    def serialize(self):
        # Flatten board + turn
        return self.board.flatten().tolist() + [self.turn]

# --- DATA GENERATOR (Exhaustive Search) ---
def generate_universe():
    print("Generating Hexapawn Universe...")
    # BFS to find all states
    queue = [Hexapawn()]
    seen = set()
    states = []
    outcomes = [] # 1=WhiteWin, -1=BlackWin, 0=Ongoing
    
    # Simple encoding for 'seen': string representation
    seen.add(str(queue[0].board) + str(queue[0].turn))
    
    # We just want a rich dataset of states
    processed = 0
    while queue:
        game = queue.pop(0)
        processed += 1
        
        flat = game.serialize()
        status = game.is_win()
        
        states.append(flat)
        outcomes.append(status)
        
        if status == 0: # If game not over, expand
            moves = game.get_moves()
            for m in moves:
                next_game = game.apply_move(m)
                key = str(next_game.board) + str(next_game.turn)
                if key not in seen:
                    seen.add(key)
                    queue.append(next_game)
                    
    print(f"Total Unique Game States: {len(states)}")
    return np.array(states, dtype=np.float32), np.array(outcomes, dtype=np.float32)

# --- THE GAME BRAIN ---
class GameAE(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 9 squares + 1 turn = 10 inputs
        self.encoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8) # 8D Latent Space for Chess
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        z = self.encoder(x)
        # Normalize z to sphere? Let's try standard first, maybe sphere later.
        # Games have distinct states, maybe clusters are better.
        return self.decoder(z), z

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="./data/hexapawn_bbit")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Generate Data
    X_raw, Y_raw = generate_universe()
    
    # 2. Train Brain
    dataset = torch.utils.data.TensorDataset(torch.tensor(X_raw), torch.tensor(Y_raw))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = GameAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print("Training Game Brain...")
    model.train()
    for epoch in range(50):
        total_loss = 0
        for bx, by in loader:
            bx = bx.to(device)
            optimizer.zero_grad()
            recon, z = model(bx)
            loss = criterion(recon, bx)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {total_loss:.4f}")

    # 3. Save Manifold
    print("Extracting Game Manifold...")
    model.eval()
    with torch.no_grad():
        _, Z = model(torch.tensor(X_raw).to(device))
    
    Z = Z.cpu().numpy()
    
    np.save(os.path.join(args.outdir, "states.npy"), X_raw)
    np.save(os.path.join(args.outdir, "outcomes.npy"), Y_raw)
    np.save(os.path.join(args.outdir, "latents.npy"), Z)
    torch.save(model.state_dict(), os.path.join(args.outdir, "model.pth"))
    print("[ok] Hexapawn Manifold Built.")

if __name__ == "__main__":
    main()