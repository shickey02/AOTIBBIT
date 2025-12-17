#!/usr/bin/env python3
# geomlang_edges_relscale_train64_latent256.py
#
# Train a 64x64 edges+relscale SceneModel with LATENT_DIM=256.
# Saves checkpoint to:
#   outputs_edges_relscale256/scene_model_edges_relscale256.pt

import os
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------
# Config
# -----------------------
IMG_SIZE   = 64
NUM_CH     = 3          # red, blue, edges
LATENT_DIM = 256

N_TRAIN    = 24000
N_VAL      = 6000
BATCH_SIZE = 128
N_EPOCHS   = 40
LR         = 1e-3

OUT_DIR         = "outputs_edges_relscale256"
CKPT_SCENEMODEL = os.path.join(OUT_DIR, "scene_model_edges_relscale256.pt")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REL_LEFT, REL_RIGHT, REL_ABOVE, REL_BELOW, REL_OVERLAP = range(5)
REL_NAMES = ["left_of", "right_of", "above", "below", "overlap"]

# -----------------------
# Shape drawing / scene generation
# -----------------------

def draw_circle(mask, cx, cy, radius):
    H, W = mask.shape
    yy, xx = np.ogrid[:H, :W]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    mask[dist2 <= radius ** 2] = 1.0


def draw_square(mask, cx, cy, half_size):
    H, W = mask.shape
    x0 = max(0, cx - half_size)
    x1 = min(W, cx + half_size + 1)
    y0 = max(0, cy - half_size)
    y1 = min(H, cy + half_size + 1)
    mask[y0:y1, x0:x1] = 1.0


def make_edges(red_mask, blue_mask):
    union = (red_mask > 0.5) | (blue_mask > 0.5)
    union = union.astype(np.float32)

    H, W = union.shape
    interior = np.zeros_like(union)
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
    dx = cx_b - cx_r
    dy = cy_b - cy_r

    if abs(dx) > abs(dy) + tol:
        return REL_LEFT if dx > 0 else REL_RIGHT
    elif abs(dy) > abs(dx) + tol:
        return REL_ABOVE if dy > 0 else REL_BELOW
    else:
        return REL_OVERLAP


def scale_class_from_size(s):
    if s <= 7:
        return 0
    elif s <= 11:
        return 1
    else:
        return 2


class GeomEdges64Dataset(Dataset):
    def __init__(self, n_samples, seed=None):
        super().__init__()
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        H = W = IMG_SIZE
        margin = 6

        rng = self.rng  # dataset-local RNG (seedable)

        # 0 circle, 1 square
        shape_red  = int(rng.integers(0, 2))
        shape_blue = int(rng.integers(0, 2))

        base = int(rng.integers(5, 13))  # 5..12 inclusive
        size_red  = int(np.clip(base + int(rng.integers(-2, 3)), 4, 14))
        size_blue = int(np.clip(base + int(rng.integers(-2, 3)), 4, 14))

        def sample_center(s):
            cx = int(rng.integers(margin + s, W - margin - s))
            cy = int(rng.integers(margin + s, H - margin - s))
            return cx, cy

        cx_r, cy_r = sample_center(size_red)
        cx_b, cy_b = sample_center(size_blue)

        red  = np.zeros((H, W), dtype=np.float32)
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
        shape_r_lbl = shape_red
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
        # 64 -> 32 -> 16 -> 8 -> 4
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


class SceneModelEdges64_256(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        # heads (kept for compatibility with analysis scripts)
        self.rel_head      = nn.Linear(LATENT_DIM, 5)
        self.scale_head    = nn.Linear(LATENT_DIM, 3)
        self.shape_r_head  = nn.Linear(LATENT_DIM, 2)
        self.shape_b_head  = nn.Linear(LATENT_DIM, 2)

    def forward(self, x):
        z = self.encoder(x)
        rec = self.decoder(z)
        rel_logits   = self.rel_head(z)
        scale_logits = self.scale_head(z)
        shape_r_log  = self.shape_r_head(z)
        shape_b_log  = self.shape_b_head(z)
        return rec, rel_logits, scale_logits, shape_r_log, shape_b_log

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

# -----------------------
# Training
# -----------------------

def main():
    print(f"[train64-256] Using device: {DEVICE}")

    train_ds = GeomEdges64Dataset(N_TRAIN)
    val_ds   = GeomEdges64Dataset(N_VAL)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SceneModelEdges64_256().to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    bce = nn.BCELoss()
    ce  = nn.CrossEntropyLoss()

    def run_epoch(loader, train=True):
        if train:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        n_batches = 0

        with torch.set_grad_enabled(train):
            for imgs, rel, scale, s_r, s_b in loader:
                imgs  = imgs.to(DEVICE)
                rel   = rel.to(DEVICE)
                scale = scale.to(DEVICE)
                s_r   = s_r.to(DEVICE)
                s_b   = s_b.to(DEVICE)

                if train:
                    opt.zero_grad()

                rec, rel_log, scale_log, s_r_log, s_b_log = model(imgs)

                rec_loss   = bce(rec, imgs)
                rel_loss   = ce(rel_log, rel)
                scale_loss = ce(scale_log, scale)
                s_r_loss   = ce(s_r_log, s_r)
                s_b_loss   = ce(s_b_log, s_b)

                # Small weight on classification heads – mostly autoencoder
                cls_loss = rel_loss + scale_loss + s_r_loss + s_b_loss
                loss = rec_loss + 0.25 * cls_loss

                if train:
                    loss.backward()
                    opt.step()

                total_loss += loss.item()
                n_batches  += 1

        return total_loss / max(1, n_batches)

    for epoch in range(1, N_EPOCHS + 1):
        train_loss = run_epoch(train_dl, train=True)
        val_loss   = run_epoch(val_dl,   train=False)

        print(f"[train64-256] Epoch {epoch:3d}/{N_EPOCHS} "
              f"| train loss={train_loss:.4f} | val loss={val_loss:.4f}")

    # Save checkpoint
    ckpt = {"model_state_dict": model.state_dict()}
    torch.save(ckpt, CKPT_SCENEMODEL)
    print(f"[train64-256] Saved SceneModel -> {CKPT_SCENEMODEL}")


if __name__ == "__main__":
    main()
