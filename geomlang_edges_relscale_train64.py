#!/usr/bin/env python3
# geomlang_edges_relscale_train64.py
#
# 64x64 geomlang with:
#   channel 0: red fill
#   channel 1: blue fill
#   channel 2: edges (outlines of both shapes)
#
# Trains a conv autoencoder + relation / scale / shape heads and
# saves to: outputs_edges_relscale/scene_model_edges_relscale.pt
#
# Run:
#   python bbit_geomlang/geomlang_edges_relscale_train64.py

import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------
# Config
# -----------------------
IMG_SIZE    = 64
NUM_CH      = 3  # red, blue, edges
LATENT_DIM  = 64

N_SAMPLES   = 12000
BATCH_SIZE  = 128
EPOCHS      = 40
LR          = 1e-3

OUT_DIR         = "outputs_edges_relscale"
CKPT_SCENEMODEL = os.path.join(OUT_DIR, "scene_model_edges_relscale.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUT_DIR, exist_ok=True)

REL_LEFT, REL_RIGHT, REL_ABOVE, REL_BELOW, REL_OVERLAP = range(5)
REL_NAMES = ["left_of", "right_of", "above", "below", "overlap"]

# -----------------------
# Shape drawing
# -----------------------

def draw_circle(mask, cx, cy, radius):
    """Draw filled circle into 2D mask (in-place, float32 0/1)."""
    H, W = mask.shape
    yy, xx = np.ogrid[:H, :W]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    mask[dist2 <= radius ** 2] = 1.0


def draw_square(mask, cx, cy, half_size):
    """Draw filled square into 2D mask (in-place, float32 0/1)."""
    H, W = mask.shape
    x0 = max(0, cx - half_size)
    x1 = min(W, cx + half_size + 1)
    y0 = max(0, cy - half_size)
    y1 = min(H, cy + half_size + 1)
    mask[y0:y1, x0:x1] = 1.0


def make_edges(red_mask, blue_mask):
    """
    Compute edge channel from union of red/blue masks using a cheap
    binary 'inner vs border' trick (no scipy dependency).
    """
    union = (red_mask > 0.5) | (blue_mask > 0.5)
    union = union.astype(np.float32)

    H, W = union.shape
    interior = np.zeros_like(union)
    # pixels that have all 4-neighbours "on" are interior
    interior[1:-1, 1:-1] = (
        union[1:-1, 1:-1] *
        union[:-2, 1:-1] *
        union[2:, 1:-1] *
        union[1:-1, :-2] *
        union[1:-1, 2:]
    )
    edges = union - interior
    edges[edges < 0] = 0.0
    return edges


def relation_from_centers(cx_r, cy_r, cx_b, cy_b, tol=2.0):
    """
    Geometric relation between red and blue centers.
    Returns an int in {0..4} for REL_*.
    """
    dx = cx_b - cx_r
    dy = cy_b - cy_r

    if abs(dx) > abs(dy) + tol:
        # mostly horizontal
        if dx > 0:
            return REL_LEFT
        else:
            return REL_RIGHT
    elif abs(dy) > abs(dx) + tol:
        # mostly vertical
        if dy > 0:
            return REL_ABOVE
        else:
            return REL_BELOW
    else:
        return REL_OVERLAP


def scale_class_from_size(s):
    """3 scale classes from approximate radius/half-size."""
    if s <= 7:
        return 0  # small
    elif s <= 11:
        return 1  # medium
    else:
        return 2  # large


# -----------------------
# Dataset
# -----------------------

class GeomEdges64Dataset(Dataset):
    """
    Static scenes, 64x64, with:
      - red blob in channel 0
      - blue blob in channel 1
      - edges of both blobs in channel 2
    """

    def __init__(self, n_samples):
        super().__init__()
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        H = W = IMG_SIZE
        margin = 6

        # shapes: 0=circle, 1=square
        shape_red = np.random.randint(0, 2)
        shape_blue = np.random.randint(0, 2)

        # sizes ~ roughly "radius" or half side
        base = np.random.randint(5, 13)  # 5..12
        size_red = int(np.clip(base + np.random.randint(-2, 3), 4, 14))
        size_blue = int(np.clip(base + np.random.randint(-2, 3), 4, 14))

        def sample_center(s):
            cx = np.random.randint(margin + s, W - margin - s)
            cy = np.random.randint(margin + s, H - margin - s)
            return cx, cy

        cx_r, cy_r = sample_center(size_red)
        cx_b, cy_b = sample_center(size_blue)

        red = np.zeros((H, W), dtype=np.float32)
        blue = np.zeros((H, W), dtype=np.float32)

        if shape_red == 0:
            draw_circle(red, cx_r, cy_r, size_red)
        else:
            draw_square(red, cx_r, cy_r, size_red)

        if shape_blue == 0:
            draw_circle(blue, cx_b, cy_b, size_blue)
        else:
            draw_square(blue, cx_b, cy_b, size_blue)

        edges = make_edges(red, blue)

        img = np.stack([red, blue, edges], axis=0)  # [3,64,64]

        rel = relation_from_centers(cx_r, cy_r, cx_b, cy_b)
        scale = scale_class_from_size((size_red + size_blue) * 0.5)
        shape_r_lbl = shape_red      # 0 circle, 1 square
        shape_b_lbl = shape_blue

        return (
            torch.from_numpy(img),
            torch.tensor(rel, dtype=torch.long),
            torch.tensor(scale, dtype=torch.long),
            torch.tensor(shape_r_lbl, dtype=torch.long),
            torch.tensor(shape_b_lbl, dtype=torch.long),
        )


# -----------------------
# Model
# -----------------------

class Encoder(nn.Module):
    def __init__(self, in_channels=NUM_CH, latent_dim=LATENT_DIM):
        super().__init__()
        # 64 -> 32 -> 16 -> 8
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # 4x4
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z


class Decoder(nn.Module):
    def __init__(self, out_channels=NUM_CH, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 8
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # 16
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)    # 32
        self.deconv4 = nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1)  # 64

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, 4, 4)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))
        return x


class SceneModelEdges64(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

        # Heads on latent code
        self.rel_head = nn.Linear(LATENT_DIM, 5)      # left/right/above/below/overlap
        self.scale_head = nn.Linear(LATENT_DIM, 3)    # small/medium/large
        self.shape_r_head = nn.Linear(LATENT_DIM, 2)  # circle/square
        self.shape_b_head = nn.Linear(LATENT_DIM, 2)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        rel_logits = self.rel_head(z)
        scale_logits = self.scale_head(z)
        s_r_logits = self.shape_r_head(z)
        s_b_logits = self.shape_b_head(z)
        return recon, rel_logits, scale_logits, s_r_logits, s_b_logits, z


# -----------------------
# Training loop
# -----------------------

def train():
    print(f"[train64] Using device: {DEVICE}")
    ds = GeomEdges64Dataset(N_SAMPLES)
    n_train = int(0.9 * N_SAMPLES)
    n_val = N_SAMPLES - n_train
    ds_train, ds_val = torch.utils.data.random_split(ds, [n_train, n_val])

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    dl_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SceneModelEdges64().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    recon_loss_fn = nn.BCELoss()
    ce_loss = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for imgs, rel, scale, s_r, s_b in dl_train:
            imgs = imgs.to(DEVICE)
            rel = rel.to(DEVICE)
            scale = scale.to(DEVICE)
            s_r = s_r.to(DEVICE)
            s_b = s_b.to(DEVICE)

            opt.zero_grad()
            recon, rel_logits, scale_logits, s_r_logits, s_b_logits, z = model(imgs)

            loss_recon = recon_loss_fn(recon, imgs)
            loss_rel   = ce_loss(rel_logits, rel)
            loss_scale = ce_loss(scale_logits, scale)
            loss_sr    = ce_loss(s_r_logits, s_r)
            loss_sb    = ce_loss(s_b_logits, s_b)

            loss = loss_recon + loss_rel + loss_scale + loss_sr + loss_sb
            loss.backward()
            opt.step()

            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(dl_train.dataset)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, rel, scale, s_r, s_b in dl_val:
                imgs = imgs.to(DEVICE)
                rel = rel.to(DEVICE)
                scale = scale.to(DEVICE)
                s_r = s_r.to(DEVICE)
                s_b = s_b.to(DEVICE)

                recon, rel_logits, scale_logits, s_r_logits, s_b_logits, z = model(imgs)

                loss_recon = recon_loss_fn(recon, imgs)
                loss_rel   = ce_loss(rel_logits, rel)
                loss_scale = ce_loss(scale_logits, scale)
                loss_sr    = ce_loss(s_r_logits, s_r)
                loss_sb    = ce_loss(s_b_logits, s_b)

                loss = loss_recon + loss_rel + loss_scale + loss_sr + loss_sb
                val_loss += loss.item() * imgs.size(0)

        val_loss /= len(dl_val.dataset)
        print(f"[train64] Epoch {epoch:3d}/{EPOCHS} | train loss={train_loss:.4f} | val loss={val_loss:.4f}")

    # save checkpoint
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": {
            "img_size": IMG_SIZE,
            "latent_dim": LATENT_DIM,
            "num_channels": NUM_CH,
            "rel_names": REL_NAMES,
        },
    }
    torch.save(ckpt, CKPT_SCENEMODEL)
    print(f"[train64] Saved SceneModel -> {CKPT_SCENEMODEL}")


def main():
    train()


if __name__ == "__main__":
    main()
