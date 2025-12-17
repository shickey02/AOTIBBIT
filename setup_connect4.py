#!/usr/bin/env python3
# SETUP SCRIPT: CONNECT 4 MANIFOLD
# Generates self-play data to map the geometry of "Winning" in a complex game.

import os, argparse, copy, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# --- GAME ENGINE ---
class Connect4:
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.turn = 1 # 1 = Player 1, -1 = Player 2
        self.history = []

    def get_valid_moves(self):
        return [c for c in range(self.cols) if self.board[0, c] == 0]

    def drop_piece(self, col):
        if self.board[0, col] != 0: return None
        new_game = copy.deepcopy(self)
        for r in range(self.rows-1, -1, -1):
            if new_game.board[r, col] == 0:
                new_game.board[r, col] = new_game.turn
                break
        new_game.turn *= -1
        return new_game

    def check_win(self):
        # Horizontal, Vertical, Diagonal check
        b = self.board
        # Horizontal
        for r in range(self.rows):
            for c in range(self.cols-3):
                if abs(np.sum(b[r, c:c+4])) == 4: return np.sign(b[r, c])
        # Vertical
        for r in range(self.rows-3):
            for c in range(self.cols):
                if abs(np.sum(b[r:r+4, c])) == 4: return np.sign(b[r, c])
        # Diag 1
        for r in range(self.rows-3):
            for c in range(self.cols-3):
                if abs(sum(b[r+i, c+i] for i in range(4))) == 4: return np.sign(b[r, c])
        # Diag 2
        for r in range(3, self.rows):
            for c in range(self.cols-3):
                if abs(sum(b[r-i, c+i] for i in range(4))) == 4: return np.sign(b[r, c])
        
        if len(self.get_valid_moves()) == 0: return 0 # Draw
        return None # Ongoing

# --- BRAIN ARCHITECTURE (CNN-VAE) ---
class BoardVAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        # Encoder (Spatial Awareness)
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_z = nn.Linear(32 * 6 * 7, latent_dim)
        
        # Decoder (Reconstruction)
        self.fc_dec = nn.Linear(latent_dim, 32 * 6 * 7)
        self.dec_conv = nn.Sequential(
            nn.Unflatten(1, (32, 6, 7)),
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, padding=1),
            nn.Tanh() # Output -1 to 1
        )

    def encode(self, x):
        h = self.enc_conv(x)
        return self.fc_z(h)

    def forward(self, x):
        z = self.encode(x)
        dec = self.fc_dec(z)
        out = self.dec_conv(dec)
        return out, z

def generate_self_play(games=5000):
    print(f"Simulating {games} games (Self-Play)...")
    boards = []
    outcomes = [] # Who eventually won from this state?
    
    for _ in tqdm(range(games)):
        game = Connect4()
        history = []
        winner = None
        
        while winner is None:
            moves = game.get_valid_moves()
            if not moves: break
            move = random.choice(moves)
            game = game.drop_piece(move)
            
            # Store state (Board relative to current turn)
            # We flip board so AI always sees "1" as itself
            view = game.board * game.turn 
            history.append(view)
            winner = game.check_win()
            
        # Backpropagate outcome
        # If Player 1 won, every board where turn=1 gets +1, turn=-1 gets -1
        if winner is not None:
            final_res = winner # 1 or -1 relative to P1
            for i, board in enumerate(history):
                # The board was stored relative to the player whose turn it was
                # So if that player eventually won, label is 1.
                # Actually, simple heuristic: history stores "current player view".
                # If current player won, Outcome=1.
                # winner is absolute (1 or -1).
                # turn at step i was (1 if i%2==0 else -1)
                
                turn_at_step = 1 if i % 2 == 0 else -1
                label = 1 if winner == turn_at_step else -1
                
                boards.append(board)
                outcomes.append(label)
                
    return np.array(boards), np.array(outcomes)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="./data/connect4_bbit")
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Generate Data
    X_raw, Y_raw = generate_self_play()
    print(f"Dataset Size: {len(X_raw)} positions")
    
    # 2. Train VAE
    print("Training Strategic Cortex...")
    # Reshape for CNN: (N, 1, 6, 7)
    X_tensor = torch.tensor(X_raw, dtype=torch.float32).unsqueeze(1)
    Y_tensor = torch.tensor(Y_raw, dtype=torch.float32)
    
    dataset = TensorDataset(X_tensor, Y_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = BoardVAE(latent_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for bx, _ in loader:
            bx = bx.to(device)
            optimizer.zero_grad()
            recon, z = model(bx)
            loss = criterion(recon, bx)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    # 3. Save Manifold
    print("Extracting Strategic Map...")
    model.eval()
    all_z = []
    
    # Encode all data to find the "Winning Cluster"
    with torch.no_grad():
        # Process in chunks
        for i in range(0, len(X_tensor), 1000):
            chunk = X_tensor[i:i+1000].to(device)
            z = model.encode(chunk)
            all_z.append(z.cpu().numpy())
            
    Z = np.vstack(all_z)
    
    np.save(os.path.join(args.outdir, "latents.npy"), Z)
    np.save(os.path.join(args.outdir, "outcomes.npy"), Y_raw)
    torch.save(model.state_dict(), os.path.join(args.outdir, "model.pth"))
    print("[ok] Connect 4 Brain Built.")

if __name__ == "__main__":
    main()