#!/usr/bin/env python3
# SETUP SCRIPT: MNIST MANIFOLD GENERATOR
# Trains a Convolutional Autoencoder to create a latent space for BBIT testing.

import os, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------
# The Architecture (The "Eyes")
# ---------------------------------------------------------
class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        # Encoder (Compress 28x28 -> Latent)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # -> 14x14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # -> 7x7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, latent_dim)
        )
        
        # Decoder (Reconstruct Latent -> 28x28)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # -> 28x28
            nn.Sigmoid() # Pixels 0-1
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="./data/mnist_bbit", help="Where to save .npy files")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--latent_dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load MNIST
    print("Loading MNIST...")
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    # We want ALL data for the latent dump, so we make a big loader
    full_data = torch.utils.data.ConcatDataset([train_data, test_data])
    full_loader = DataLoader(full_data, batch_size=args.batch_size, shuffle=False)

    # 2. Train Autoencoder
    print(f"Training Autoencoder (Dim={args.latent_dim})...")
    model = ConvAutoencoder(latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for imgs, _ in pbar:
            imgs = imgs.to(device)
            
            # Forward
            recon, z = model(imgs)
            loss = criterion(recon, imgs)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    # 3. Extract & Save Latents
    print("Extracting Latent Manifold...")
    model.eval()
    all_z = []
    all_y = []

    with torch.no_grad():
        for imgs, labels in tqdm(full_loader, desc="Encoding"):
            imgs = imgs.to(device)
            _, z = model(imgs)
            all_z.append(z.cpu().numpy())
            all_y.append(labels.numpy())

    Z = np.vstack(all_z)
    Y = np.concatenate(all_y)

    print(f"Manifold Shape: {Z.shape}")
    
    z_path = os.path.join(args.outdir, "latents_mnist.npy")
    y_path = os.path.join(args.outdir, "labels_mnist.npy")
    
    np.save(z_path, Z)
    np.save(y_path, Y)
    print(f"[ok] Saved to {args.outdir}")

if __name__ == "__main__":
    main()