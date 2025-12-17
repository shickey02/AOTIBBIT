# bbit_geomlang/geomlang_multichannel.py

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ----------------------------
# Config
# ----------------------------
GRID = 32
NUM_SAMPLES = 256      # you can bump this later
BATCH_SIZE = 16
LATENT_DIM = 32
EPOCHS = 800           # feels similar to previous runs
LR = 1e-3
OUT_DIR = "outputs_multichannel"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------------------
# Data generation
# ----------------------------

def draw_shape(grid, channel, shape_type="cube"):
    """
    grid: (2, G, G, G)
    channel: 0 or 1
    shape_type: "cube" or "sphere"
    """
    G = grid.shape[1]

    # Size and center
    size = np.random.randint(G // 6, G // 3)
    cx = np.random.randint(size, G - size)
    cy = np.random.randint(size, G - size)
    cz = np.random.randint(size, G - size)

    if shape_type == "cube":
        xs = slice(cx - size // 2, cx + size // 2)
        ys = slice(cy - size // 2, cy + size // 2)
        zs = slice(cz - size // 2, cz + size // 2)
        grid[channel, xs, ys, zs] = 1.0
    else:  # "sphere" (rough)
        x = np.arange(G)[:, None, None]
        y = np.arange(G)[None, :, None]
        z = np.arange(G)[None, None, :]
        dist2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
        mask = dist2 <= (size / 2) ** 2
        grid[channel][mask] = 1.0


def make_scene():
    """
    Returns a volume with shape (2, GRID, GRID, GRID)
    channel 0: object A
    channel 1: object B
    """
    grid = np.zeros((2, GRID, GRID, GRID), dtype=np.float32)

    # Object A
    draw_shape(grid, 0, "cube" if np.random.rand() < 0.5 else "sphere")

    # Object B (independent; allowed to overlap)
    draw_shape(grid, 1, "cube" if np.random.rand() < 0.5 else "sphere")

    return grid


def generate_dataset(n=NUM_SAMPLES):
    arr = np.zeros((n, 2, GRID, GRID, GRID), dtype=np.float32)
    for i in range(n):
        arr[i] = make_scene()
    return arr


# ----------------------------
# Model: 3D Conv Autoencoder
# ----------------------------

class SceneAutoencoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(inplace=True),
        )
        self.enc_fc = nn.Linear(64 * 8 * 8 * 8, latent_dim)

        self.dec_fc = nn.Linear(latent_dim, 64 * 8 * 8 * 8)
        self.dec = nn.Sequential(
            nn.ConvTranspose3d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # 8 -> 16
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # 16 -> 32
            nn.ReLU(inplace=True),

            nn.Conv3d(16, 2, kernel_size=1),  # logits for 2 channels
        )

    def encode(self, x):
        h = self.enc(x)
        h = h.view(h.size(0), -1)
        z = self.enc_fc(h)
        return z

    def decode(self, z):
        h = self.dec_fc(z)
        h = h.view(h.size(0), 64, 8, 8, 8)
        x_logit = self.dec(h)
        return x_logit

    def forward(self, x):
        z = self.encode(x)
        x_logit = self.decode(z)
        return x_logit, z


# ----------------------------
# Visualization helpers
# ----------------------------

def volume_to_rgb(vol):
    """
    vol: (2, GRID, GRID, GRID)
    Returns (GRID, GRID, 3) RGB image:
      channel 0 -> red
      channel 1 -> blue
      overlap -> magenta
    """
    # Max projection along z
    ch0 = vol[0].max(axis=2)  # (GRID, GRID)
    ch1 = vol[1].max(axis=2)

    ch0 = np.clip(ch0, 0, 1)
    ch1 = np.clip(ch1, 0, 1)

    img = np.zeros((GRID, GRID, 3), dtype=np.float32)
    img[..., 0] = ch0  # R
    img[..., 1] = 0.0  # G
    img[..., 2] = ch1  # B

    return img


def save_rgb(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.imsave(path, img)


# ----------------------------
# Occlusion + latent recovery
# ----------------------------

def occlude_volume(vol, frac=0.4):
    """
    Zeroes a random block in both channels of vol (2,G,G,G).
    frac is the fraction of the grid along each axis.
    """
    G = vol.shape[1]
    block_size = int(G * frac)

    # Random start indices
    sx = np.random.randint(0, G - block_size)
    sy = np.random.randint(0, G - block_size)
    sz = np.random.randint(0, G - block_size)

    occluded = vol.copy()
    occluded[:, sx:sx + block_size, sy:sy + block_size, sz:sz + block_size] = 0.0
    return occluded, (sx, sy, sz, block_size)


def latent_recovery(model, original, occluded, steps=400, lr=1e-1):
    """
    Given original and occluded volume (1,2,G,G,G), optimize a latent code
    such that decode(z) matches the original (not the occluded).
    This is the BBIT-ish "fill in the blocked manifold" step.
    """
    model.eval()
    with torch.no_grad():
        z_init = model.encode(occluded)  # starting guess from corrupted input

    z = z_init.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=lr)
    mse = nn.MSELoss()

    for i in range(steps):
        optimizer.zero_grad()
        logits = model.decode(z)
        recon = torch.sigmoid(logits)
        loss = mse(recon, original)
        loss.backward()
        optimizer.step()

        if i % 100 == 0 or i == steps - 1:
            print(f"  Recover step {i}/{steps} | Loss: {loss.item():.6f}")

    with torch.no_grad():
        final_logits = model.decode(z)
        final_recon = torch.sigmoid(final_logits)
    return final_recon


# ----------------------------
# Main
# ----------------------------

def main():
    # 1) Data
    print("Generating dataset...")
    data_np = generate_dataset()
    data_t = torch.from_numpy(data_np)
    dataset = TensorDataset(data_t)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2) Model + training
    model = SceneAutoencoder().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("Training autoencoder on multichannel scenes...")
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for (x,) in loader:
            x = x.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, x)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        mean_loss = running_loss / len(dataset)
        if epoch % 100 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch}/{EPOCHS} | Loss: {mean_loss:.6f}")

    # 3) Pick an example scene
    idx = 0
    scene = data_t[idx:idx + 1].to(device)  # (1,2,G,G,G)

    model.eval()
    with torch.no_grad():
        logits, _ = model(scene)
        recon = torch.sigmoid(logits)

    scene_np = scene.cpu().numpy()[0]
    recon_np = recon.cpu().numpy()[0]

    img_orig = volume_to_rgb(scene_np)
    img_recon = volume_to_rgb(recon_np)

    save_rgb(img_orig, os.path.join(OUT_DIR, "original.png"))
    save_rgb(img_recon, os.path.join(OUT_DIR, "recon.png"))
    print(f"Saved original & recon to {OUT_DIR}/original.png and recon.png")

    # 4) Occlusion + recovery
    print("Running occlusion + latent recovery...")
    scene_np_occ, block_info = occlude_volume(scene_np, frac=0.4)
    print("Occluded block:", block_info)

    occluded_t = torch.from_numpy(scene_np_occ[None, ...]).to(device)  # (1,2,G,G,G)

    recovered = latent_recovery(model, scene, occluded_t, steps=400, lr=1e-1)
    recovered_np = recovered.cpu().numpy()[0]

    img_occ = volume_to_rgb(scene_np_occ)
    img_recov = volume_to_rgb(recovered_np)

    save_rgb(img_occ, os.path.join(OUT_DIR, "occluded.png"))
    save_rgb(img_recov, os.path.join(OUT_DIR, "recovered.png"))
    print(f"Saved occlusion test to {OUT_DIR}/occluded.png and recovered.png")


if __name__ == "__main__":
    main()
