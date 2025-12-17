#!/usr/bin/env python3
"""
geomlang_edges_relscale.py

Clean BBIT "edges" model:

- 64x64 images
- 3 channels:
    0: red object fill
    1: blue object fill
    2: white edges (outlines of both shapes)
- Two objects (always squares here, to keep geometry simple)
- Spatial relations (red relative to blue):
    0: left_of
    1: right_of
    2: above
    3: below
    4: inside      (red inside blue)
    5: overlapping (generic overlap)
- Relative scale labels:
    0: red_larger
    1: red_smaller
    2: similar

We train a conv autoencoder + linear heads for:
    - relation label
    - scale label

Then we:
    - dump all latents + labels to outputs_edges/latents_dump.npz
    - save the AE checkpoint to outputs_edges/conv_autoencoder_edges.pt

Other scripts (geomlang_dynamics.py, geomlang_dynamics_viz_frames.py)
assume:
    - IMG_SIZE = 64
    - LATENT_DIM = 128
    - 3-channel input as described above
"""

import os
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

IMG_SIZE      = 64
NUM_CHANNELS  = 3   # [red_fill, blue_fill, edges]
LATENT_DIM    = 128

RELATION_NAMES = ["left_of", "right_of", "above", "below", "inside", "overlapping"]
SCALE_NAMES    = ["red_larger", "red_smaller", "similar"]

NUM_REL   = len(RELATION_NAMES)
NUM_SCALE = len(SCALE_NAMES)

# dataset / training
NUM_SAMPLES = 12000
BATCH_SIZE  = 128
EPOCHS      = 200
LR          = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR = "outputs_edges"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------

def draw_filled_square(grid, cx, cy, half_size, channel_idx):
    """Draw a filled axis-aligned square."""
    h, w = grid.shape[1], grid.shape[2]
    x0 = max(0, cx - half_size)
    x1 = min(w, cx + half_size + 1)
    y0 = max(0, cy - half_size)
    y1 = min(h, cy + half_size + 1)
    grid[channel_idx, y0:y1, x0:x1] = 1.0


def draw_square_edges(grid, cx, cy, half_size, channel_idx):
    """Draw a 1-pixel wide border of a square."""
    h, w = grid.shape[1], grid.shape[2]
    x0 = max(0, cx - half_size)
    x1 = min(w - 1, cx + half_size)
    y0 = max(0, cy - half_size)
    y1 = min(h - 1, cy + half_size)

    # top & bottom edges
    grid[channel_idx, y0, x0:x1 + 1] = 1.0
    grid[channel_idx, y1, x0:x1 + 1] = 1.0
    # left & right edges
    grid[channel_idx, y0:y1 + 1, x0] = 1.0
    grid[channel_idx, y0:y1 + 1, x1] = 1.0


# ---------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------

def rand_band(low, high, alt_low, alt_high):
    """
    Safe randint helper:
    - Uses [low, high) if it's a valid interval (high > low + 1).
    - Otherwise falls back to [alt_low, alt_high).
    """
    if high <= low + 1:
        low, high = alt_low, alt_high
    return np.random.randint(low, high)


def generate_scene(rel_label, scale_label):
    """
    Generate a single scene:

    rel_label: 0..5   (red relative to blue)
    scale_label: 0..2 (red_larger, red_smaller, similar)
    """
    img = np.zeros((NUM_CHANNELS, IMG_SIZE, IMG_SIZE), dtype=np.float32)

    margin = max(4, IMG_SIZE // 10)   # keep shapes away from border a bit

    # ---------- choose base sizes ----------
    base = np.random.randint(5, 10)  # half-sizes 5..9

    # ---------- sizes (with special handling for 'inside') ----------
    if rel_label == 4:  # inside: red strictly smaller than blue
        scale_label = 1  # force red_smaller
        r_blue = np.random.randint(base + 2, base + 5)   # larger
        r_red  = max(3, r_blue - np.random.randint(2, 4))
    else:
        if scale_label == 0:      # red_larger
            r_red  = base + 2
            r_blue = base
        elif scale_label == 1:    # red_smaller
            r_red  = base
            r_blue = base + 2
        else:                     # similar
            delta = np.random.randint(-1, 2)  # -1,0,1
            r_red  = base
            r_blue = base + delta

    full_low  = margin
    full_high = IMG_SIZE - margin
    mid       = IMG_SIZE // 2

    # ---------- positions conditioned on relation ----------
    if rel_label == 0:  # left_of
        cy = rand_band(full_low, full_high, full_low, full_high)

        cx_red  = rand_band(margin, mid - margin, full_low, mid)
        cx_blue = rand_band(mid + margin // 2, IMG_SIZE - margin, mid, full_high)

        cy_red = cy_blue = cy

    elif rel_label == 1:  # right_of
        cy = rand_band(full_low, full_high, full_low, full_high)

        cx_blue = rand_band(margin, mid - margin, full_low, mid)
        cx_red  = rand_band(mid + margin // 2, IMG_SIZE - margin, mid, full_high)

        cy_red = cy_blue = cy

    elif rel_label == 2:  # above
        cx = rand_band(full_low, full_high, full_low, full_high)

        cy_red  = rand_band(margin, mid - margin, full_low, mid)
        cy_blue = rand_band(mid + margin // 2, IMG_SIZE - margin, mid, full_high)

        cx_red = cx_blue = cx

    elif rel_label == 3:  # below
        cx = rand_band(full_low, full_high, full_low, full_high)

        cy_blue = rand_band(margin, mid - margin, full_low, mid)
        cy_red  = rand_band(mid + margin // 2, IMG_SIZE - margin, mid, full_high)

        cx_red = cx_blue = cx

    elif rel_label == 4:  # inside (red inside blue)
        # pick center that keeps BLUE fully inside frame
        low  = margin + r_blue
        high = IMG_SIZE - margin - r_blue
        if high <= low + 1:
            low, high = full_low, full_high

        cx = np.random.randint(low, high)
        cy = np.random.randint(low, high)

        cx_red = cx_blue = cx
        cy_red = cy_blue = cy

    else:  # overlapping
        cx_red = np.random.randint(full_low, full_high)
        cy_red = np.random.randint(full_low, full_high)

        # small offset for blue so they overlap but aren't identical
        offset_x = np.random.randint(-8, 9)
        offset_y = np.random.randint(-8, 9)
        cx_blue = int(np.clip(cx_red + offset_x, full_low, full_high - 1))
        cy_blue = int(np.clip(cy_red + offset_y, full_low, full_high - 1))

    # ---------- draw shapes ----------
    # red fill
    draw_filled_square(img, cx_red,  cy_red,  r_red,  channel_idx=0)
    # blue fill
    draw_filled_square(img, cx_blue, cy_blue, r_blue, channel_idx=1)
    # edges (white outline of both)
    draw_square_edges(img, cx_red,  cy_red,  r_red,  channel_idx=2)
    draw_square_edges(img, cx_blue, cy_blue, r_blue, channel_idx=2)

    img = np.clip(img, 0.0, 1.0)
    return img, scale_label


def generate_dataset(num_samples=NUM_SAMPLES, seed=0):
    np.random.seed(seed)
    random.seed(seed)

    scenes       = []
    rel_labels   = []
    scale_labels = []

    for _ in range(num_samples):
        rel   = np.random.randint(0, NUM_REL)
        scale = np.random.randint(0, NUM_SCALE)

        img, scale = generate_scene(rel, scale)
        scenes.append(img)
        rel_labels.append(rel)
        scale_labels.append(scale)

    scenes       = torch.tensor(np.stack(scenes),       dtype=torch.float32)
    rel_labels   = torch.tensor(rel_labels,             dtype=torch.long)
    scale_labels = torch.tensor(scale_labels,           dtype=torch.long)
    return scenes, rel_labels, scale_labels


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

class ConvEncoderEdges(nn.Module):
    """
    Simple conv encoder for 64x64x3 → latent_dim.
    """
    def __init__(self, in_channels=NUM_CHANNELS, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1)  # 32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)          # 16x16
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)         # 8x8
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)        # 4x4
        self.fc    = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z


class ConvDecoderEdges(nn.Module):
    """
    Decoder for latent_dim → 64x64x3.
    """
    def __init__(self, out_channels=NUM_CHANNELS, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc    = nn.Linear(latent_dim, 256 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 8x8
        self.deconv2 = nn.ConvTranspose2d(128, 64,  kernel_size=4, stride=2, padding=1)  # 16x16
        self.deconv3 = nn.ConvTranspose2d(64,  32,  kernel_size=4, stride=2, padding=1)  # 32x32
        self.deconv4 = nn.ConvTranspose2d(32,  out_channels, kernel_size=4, stride=2, padding=1)  # 64x64

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, 4, 4)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))
        return x


class ConvAutoencoderEdges(nn.Module):
    """
    Autoencoder wrapper so other scripts can import this.
    """
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = ConvEncoderEdges(latent_dim=latent_dim)
        self.decoder = ConvDecoderEdges(latent_dim=latent_dim)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z   = self.encode(x)
        rec = self.decode(z)
        return rec, z


class SceneModelEdges(nn.Module):
    """
    Full scene model: AE + relation & scale heads.
    """
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.ae = ConvAutoencoderEdges(latent_dim=latent_dim)
        self.rel_head   = nn.Linear(latent_dim, NUM_REL)
        self.scale_head = nn.Linear(latent_dim, NUM_SCALE)

    def encode(self, x):
        return self.ae.encode(x)

    def decode(self, z):
        return self.ae.decode(z)

    def forward(self, x):
        rec, z = self.ae(x)
        rel_logits   = self.rel_head(z)
        scale_logits = self.scale_head(z)
        return rec, z, rel_logits, scale_logits


# ---------------------------------------------------------------------
# Training + latent dump
# ---------------------------------------------------------------------

def train_and_dump():
    print(f"[edges] Using device: {DEVICE}")

    scenes, rel_labels, scale_labels = generate_dataset()
    dataset = TensorDataset(scenes, rel_labels, scale_labels)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SceneModelEdges().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)

    N = len(dataset)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_recon = total_rel_ce = total_scale_ce = 0.0
        correct_rel = correct_scale = 0
        total = 0

        for x, rel, scale in loader:
            x     = x.to(DEVICE)
            rel   = rel.to(DEVICE)
            scale = scale.to(DEVICE)

            opt.zero_grad()
            rec, z, rel_logits, scale_logits = model(x)

            bce      = F.binary_cross_entropy(rec, x)
            rel_ce   = F.cross_entropy(rel_logits, rel)
            scale_ce = F.cross_entropy(scale_logits, scale)

            loss = bce + rel_ce + scale_ce
            loss.backward()
            opt.step()

            bs = x.size(0)
            total += bs

            total_recon    += bce.item() * bs
            total_rel_ce   += rel_ce.item() * bs
            total_scale_ce += scale_ce.item() * bs

            correct_rel   += (rel_logits.argmax(dim=1) == rel).sum().item()
            correct_scale += (scale_logits.argmax(dim=1) == scale).sum().item()

        if epoch % 20 == 0 or epoch == 1 or epoch == EPOCHS:
            print(
                f"[edges] Epoch {epoch:3d}/{EPOCHS} | "
                f"Recon {total_recon/total:.4f} | "
                f"RelCE {total_rel_ce/total:.4f} | "
                f"ScaleCE {total_scale_ce/total:.4f} | "
                f"Acc_rel {100*correct_rel/total:5.2f}% | "
                f"Acc_scale {100*correct_scale/total:5.2f}%"
            )

    # ---- dump latents ----
    model.eval()
    with torch.no_grad():
        all_z    = []
        all_rel  = []
        all_scale = []

        for x, rel, scale in DataLoader(dataset, batch_size=256, shuffle=False):
            x = x.to(DEVICE)
            z = model.encode(x)
            all_z.append(z.cpu())
            all_rel.append(rel)
            all_scale.append(scale)

        all_z     = torch.cat(all_z, dim=0).numpy()
        all_rel   = torch.cat(all_rel, dim=0).numpy()
        all_scale = torch.cat(all_scale, dim=0).numpy()

    latents_path = os.path.join(OUT_DIR, "latents_dump.npz")
    np.savez_compressed(latents_path, z=all_z, rel=all_rel, scale=all_scale)
    print(f"[edges] Saved latents -> {latents_path}")

    # ---- save AE checkpoint ----
    ckpt_path = os.path.join(OUT_DIR, "conv_autoencoder_edges.pt")
    torch.save(
        {
            "ae_state_dict": model.ae.state_dict(),
            "config": {
                "img_size": IMG_SIZE,
                "latent_dim": LATENT_DIM,
                "num_channels": NUM_CHANNELS,
            },
        },
        ckpt_path,
    )
    print(f"[edges] Saved AE checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    train_and_dump()
