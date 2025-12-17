#!/usr/bin/env python3
# VECTOR ZERO: Connect 4 Agent (FIXED)
# Plays by navigating the latent manifold toward "Victory".

import os, argparse, copy, sys, random
import numpy as np
import torch
import torch.nn as nn

# --- RE-DEFINE ARCHITECTURE (MUST MATCH TRAINING EXACTLY) ---
class BoardVAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        # Encoder
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_z = nn.Linear(32 * 6 * 7, latent_dim)
        
        # Decoder (Included only for weight loading compatibility)
        self.fc_dec = nn.Linear(latent_dim, 32 * 6 * 7)
        self.dec_conv = nn.Sequential(
            nn.Unflatten(1, (32, 6, 7)),
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, padding=1),
            nn.Tanh() 
        )

    def encode(self, x):
        h = self.enc_conv(x)
        return self.fc_z(h)

# --- GAME ENGINE ---
class Connect4:
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.turn = 1 # 1 = AI, -1 = Human
    def valid_moves(self):
        return [c for c in range(self.cols) if self.board[0, c] == 0]
    def drop(self, col):
        g = copy.deepcopy(self)
        for r in range(self.rows-1, -1, -1):
            if g.board[r, col] == 0:
                g.board[r, col] = g.turn
                break
        g.turn *= -1
        return g
    def check_win(self):
        # Simplified check for 4 in a row
        b = self.board
        # Horizontal
        for r in range(self.rows):
            for c in range(self.cols-3):
                if abs(sum(b[r,c:c+4]))==4: return np.sign(b[r,c])
        # Vertical
        for r in range(self.rows-3):
            for c in range(self.cols):
                if abs(sum(b[r:r+4,c]))==4: return np.sign(b[r,c])
        # Diagonal 1
        for r in range(self.rows-3):
            for c in range(self.cols-3):
                if abs(sum(b[r+i,c+i] for i in range(4)))==4: return np.sign(b[r,c])
        # Diagonal 2
        for r in range(3, self.rows):
            for c in range(self.cols-3):
                if abs(sum(b[r-i,c+i] for i in range(4)))==4: return np.sign(b[r,c])
        return 0

class VectorBot:
    def __init__(self, data_dir):
        self.device = torch.device("cpu")
        self.Z = np.load(os.path.join(data_dir, "latents.npy"))
        self.Y = np.load(os.path.join(data_dir, "outcomes.npy"))
        
        self.model = BoardVAE(latent_dim=64)
        self.model.load_state_dict(torch.load(os.path.join(data_dir, "model.pth"), map_location=self.device))
        self.model.eval()
        
        # Calculate The Goal
        # AI plays as "1" in its own mind. So we look for Outcome=1.
        self.win_centroid = np.mean(self.Z[self.Y == 1], axis=0)
        self.lose_centroid = np.mean(self.Z[self.Y == -1], axis=0)
        
        # The Strategic Vector: Away from Loss, Toward Win
        self.strategy_vec = self.win_centroid - self.lose_centroid

    def get_z(self, board):
        # AI always views board relative to itself.
        # If the bot is playing (1), board is board.
        # If the bot is simulating opponent (-1), it should flip perspective?
        # Actually, self-play data was saved relative to "Current Turn".
        # So we just feed the board as-is, assuming '1' is the active player.
        
        view = board * 1 
        t = torch.tensor(view, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            return self.model.encode(t).numpy()[0]

    def choose_move(self, game):
        moves = game.valid_moves()
        z_curr = self.get_z(game.board)
        
        best_score = -float('inf')
        best_move = random.choice(moves)
        
        print("\n// VECTOR_ZERO THINKING...")
        for col in moves:
            # Simulate move
            next_game = game.drop(col)
            
            # Get Z of the resulting board.
            # CRITICAL: The resulting board has turn = -1 (Opponent's turn).
            # The VAE was trained on "Current Player View".
            # So to evaluate "How good is this for ME?", we need to flip the board perspective
            # so the VAE thinks it's looking at MY board again?
            # Or simpler: We just look at the raw board state.
            # Let's trust the "Win Centroid" captures board states that favor Player 1.
            
            z_next = self.get_z(next_game.board) 
            
            # Project onto Strategy Vector
            # High Score = Moving in the direction of Wins
            alignment = np.dot(z_next - z_curr, self.strategy_vec)
            
            # Distance check: Are we physically closer to the Winning Cluster?
            dist_win = np.linalg.norm(z_next - self.win_centroid)
            
            # Score = Alignment - Distance (Maximize alignment, minimize distance)
            final_score = alignment - (dist_win * 0.1)
            
            print(f"   :: Col {col} -> Score: {final_score:.4f} (Align: {alignment:.2f})")
            
            if final_score > best_score:
                best_score = final_score
                best_move = col
                
        return best_move

def print_board(b):
    print("\n  0 1 2 3 4 5 6")
    for r in range(6):
        row = "|"
        for c in range(7):
            if b[r,c]==1: row+="O "
            elif b[r,c]==-1: row+="X "
            else: row+=". "
        print(row + "|")
    print("  -------------\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/connect4_bbit")
    args = ap.parse_args()
    
    bot = VectorBot(args.data)
    game = Connect4()
    
    print("=== VECTOR-ZERO CONNECT 4 ===")
    print("You are X (Player -1). AI is O (Player 1).")
    
    while True:
        print_board(game.board)
        res = game.check_win()
        if res == 1: print("AI WINS!"); break
        if res == -1: print("YOU WIN!"); break
        if not game.valid_moves(): print("DRAW"); break
        
        if game.turn == -1:
            # Human Turn
            try:
                inp = input("Your Move (0-6): ")
                if inp == "q": break
                col = int(inp)
                if col not in game.valid_moves(): continue
                game = game.drop(col)
            except: continue
        else:
            # AI Turn
            move = bot.choose_move(game)
            print(f">> AI DROPS: {move}")
            game = game.drop(move)

if __name__ == "__main__":
    main()