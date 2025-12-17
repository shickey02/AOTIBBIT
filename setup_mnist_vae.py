#!/usr/bin/env python3
# SETUP SCRIPT: MNIST VAE (Variational Autoencoder)
# Creates a continuous, smooth manifold for geometric reasoning tests.

import os, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------
# The VAE Architecture
# ---------------------------------------------------------
class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 7x7
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Latent Distribution Heads (Mu and LogVar)
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)
        
        # Decoder input
        self.fc_decode = nn.Linear(latent_dim, 64 * 7 * 7)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),  # 28x28
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        
        # --- FIX IS HERE ---
        # Project latent z (64) back up to feature map size (3136)
        z_projected = self.fc_decode(z)
        recon = self.decoder(z_projected)
        
        return recon, mu, logvar

# Loss Function (Reconstruction + KL Divergence)
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    # KLD = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="./data/mnist_vae_bbit", help="Where to save .npy files")
    ap.add_argument("--epochs", type=int, default=15) 
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
    full_data = torch.utils.data.ConcatDataset([train_data, test_data])
    full_loader = DataLoader(full_data, batch_size=args.batch_size, shuffle=False)

    # 2. Train VAE
    print(f"Training VAE (Dim={args.latent_dim})...")
    model = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(args.epochs):
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for imgs, _ in pbar:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            
            recon, mu, logvar = model(imgs)
            loss = loss_function(recon, imgs, mu, logvar)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item() / len(imgs):.1f}"})

    # 3. Extract Manifold (Using Mu - the deterministic center)
    print("Extracting Smooth Manifold...")
    model.eval()
    all_z = []
    all_y = []

    with torch.no_grad():
        for imgs, labels in tqdm(full_loader, desc="Encoding"):
            imgs = imgs.to(device)
            # For geometric operations, we use the MEAN (mu) of the distribution
            _, mu, _ = model(imgs)
            all_z.append(mu.cpu().numpy())
            all_y.append(labels.numpy())

    Z = np.vstack(all_z)
    Y = np.concatenate(all_y)

    print(f"Manifold Shape: {Z.shape}")
    
    z_path = os.path.join(args.outdir, "latents_vae.npy")
    y_path = os.path.join(args.outdir, "labels_vae.npy")
    
    np.save(z_path, Z)
    np.save(y_path, Y)
    print(f"[ok] Saved to {args.outdir}")

if __name__ == "__main__":
    main()