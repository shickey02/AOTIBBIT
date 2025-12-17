#!/usr/bin/env python3
# geomlang_edges_relscale_train64_128.py
#
# 64x64 geomlang with:
#   - channel 0: red fill
#   - channel 1: blue fill
#   - channel 2: edges (outlines of both shapes)
# and multi-head labels for:
#   - relation: left_of / right_of / above / below / overlap   (5 classes)
#   - relative scale: red smaller / similar / red larger       (3 classes)
#   - shape_red:  circle / square                              (2 classes)
#   - shape_blue: circle / square                              (2 classes)
#
# This version uses LATENT_DIM = 128.
# Checkpoint is saved to outputs_edges_relscale_128/.

import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# --------------------
# Config
# --------------------
IMG_SIZE    = 64
LATENT_DIM  = 128
N_SAMPLES   = 12000

BATCH_SIZE  = 128
EPOCHS      = 40
LR          = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR = "outputs_edges_relscale_128"
os.makedirs(OUT_DIR, exist_ok=True)
CKPT_PATH = os.path.join(OUT_DIR, "scene_model_edges_relscale_128.pt")

# loss weights
LAMBDA_RECON  = 1.0
LAMBDA_REL    = 0.5
LAMBDA_SCALE  = 0.2
LAMBDA_SHAPE  = 0.2  # applies to both red+blue (each 0.1 effectively)


# --------------------
# Drawing helpers
# --------------------
def draw_filled_circle(mask, cx, cy, r):
    h, w = mask.shape
    for y in range(max(0, cy - r), min(h, cy + r + 1)):
        dy2 = (y - cy) * (y - cy)
        for x in range(max(0, cx - r), min(w, cx + r + 1)):
            if (x - cx) * (x - cx) + dy2 <= r * r:
                mask[y, x] = 1.0


def draw_filled_square(mask, cx, cy, half):
    h, w = mask.shape
    x0 = max(0, cx - half)
    x1 = min(w, cx + half + 1)
    y0 = max(0, cy - half)
    y1 = min(h, cy + half + 1)
    mask[y0:y1, x0:x1] = 1.0


def draw_filled_and_edge(grid, shape_id, cx, cy, size, fill_ch, edge_ch):
    """
    grid: [3, H, W]
    shape_id: 0 circle, 1 square
    size: radius or half-size
    """
    H, W = grid.shape[1], grid.shape[2]
    fill = np.zeros((H, W), dtype=np.float32)

    if shape_id == 0:
        draw_filled_circle(fill, cx, cy, size)
    else:
        draw_filled_square(fill, cx, cy, size)

    # write fill
    grid[fill_ch] = np.maximum(grid[fill_ch], fill)

    # edges = pixels that are 1 and have at least one 4-neighbour = 0
    edge = np.zeros_like(fill)
    for y in range(H):
        for x in range(W):
            if fill[y, x] > 0.5:
                # 4-connected neighbours
                neigh = []
                if x > 0:       neigh.append(fill[y, x - 1])
                if x < W - 1:   neigh.append(fill[y, x + 1])
                if y > 0:       neigh.append(fill[y - 1, x])
                if y < H - 1:   neigh.append(fill[y + 1, x])
                if any(v < 0.5 for v in neigh):
                    edge[y, x] = 1.0

    grid[edge_ch] = np.maximum(grid[edge_ch], edge)


# --------------------
# Dataset generation
# --------------------
REL_LEFT, REL_RIGHT, REL_ABOVE, REL_BELOW, REL_OVERLAP = range(5)

def compute_relation(cx_r, cy_r, cx_b, cy_b, tol=2.0):
    dx = cx_r - cx_b
    dy = cy_r - cy_b

    if abs(dx) <= tol and abs(dy) <= tol:
        return REL_OVERLAP

    if cx_r + tol < cx_b - tol:
        return REL_LEFT
    if cx_r - tol > cx_b + tol:
        return REL_RIGHT
    if cy_r + tol < cy_b - tol:
        return REL_ABOVE
    if cy_r - tol > cy_b + tol:
        return REL_BELOW

    # "ambiguous" -> treat as overlap / near
    return REL_OVERLAP


def compute_scale_label(size_r, size_b, eps=0.1):
    """
    scale classes:
      0: red smaller
      1: similar
      2: red larger
    """
    ratio = size_r / float(size_b + 1e-6)
    if ratio < 1.0 - eps:
        return 0
    elif ratio > 1.0 + eps:
        return 2
    else:
        return 1


def generate_sample():
    """
    Returns:
      img: [3, H, W] float32 in [0,1]
      rel: int in [0..4]
      scale: int in [0..2]
      shape_red: int in [0..1]
      shape_blue: int in [0..1]
    """
    H = W = IMG_SIZE
    margin = 10

    # shapes
    shape_red  = np.random.randint(0, 2)   # 0 circle, 1 square
    shape_blue = np.random.randint(0, 2)

    # sizes (a bit random but coarsely quantised)
    base = np.random.randint(6, 13)  # 6..12
    jitter_r = np.random.randint(-2, 3)
    jitter_b = np.random.randint(-2, 3)
    size_r = int(np.clip(base + jitter_r, 5, 14))
    size_b = int(np.clip(base + jitter_b, 5, 14))

    # sample centers
    def sample_center(size):
        cx = np.random.randint(margin + size, W - margin - size)
        cy = np.random.randint(margin + size, H - margin - size)
        return cx, cy

    # keep sampling until we get a non-degenerate relation
    for _ in range(50):
        cx_r, cy_r = sample_center(size_r)
        cx_b, cy_b = sample_center(size_b)
        rel = compute_relation(cx_r, cy_r, cx_b, cy_b)
        # accept any relation (even overlap); just avoid exact identical centers
        if not (cx_r == cx_b and cy_r == cy_b):
            break

    scale_lbl = compute_scale_label(size_r, size_b)

    grid = np.zeros((3, H, W), dtype=np.float32)
    # red: fill ch0, edges ch2
    draw_filled_and_edge(grid, shape_red,  cx_r, cy_r, size_r, fill_ch=0, edge_ch=2)
    # blue: fill ch1, edges ch2
    draw_filled_and_edge(grid, shape_blue, cx_b, cy_b, size_b, fill_ch=1, edge_ch=2)

    return grid, rel, scale_lbl, shape_red, shape_blue


def generate_dataset(n_samples=N_SAMPLES):
    imgs      = np.zeros((n_samples, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    rel_lbl   = np.zeros((n_samples,), dtype=np.int64)
    scale_lbl = np.zeros((n_samples,), dtype=np.int64)
    shape_r   = np.zeros((n_samples,), dtype=np.int64)
    shape_b   = np.zeros((n_samples,), dtype=np.int64)

    for i in range(n_samples):
        img, r, s, sr, sb = generate_sample()
        imgs[i]    = img
        rel_lbl[i] = r
        scale_lbl[i] = s
        shape_r[i] = sr
        shape_b[i] = sb

    return (
        torch.from_numpy(imgs),
        torch.from_numpy(rel_lbl),
        torch.from_numpy(scale_lbl),
        torch.from_numpy(shape_r),
        torch.from_numpy(shape_b),
    )


# --------------------
# Model
# --------------------
class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1)   # 32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)           # 16x16
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)          # 8x8
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)         # 4x4
        self.fc    = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z


class Decoder(nn.Module):
    def __init__(self, out_channels=3, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
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


class SceneModelEdgesRelscale(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        # heads
        self.rel_head        = nn.Linear(LATENT_DIM, 5)  # 5 relations
        self.scale_head      = nn.Linear(LATENT_DIM, 3)  # 3 scale classes
        self.shape_red_head  = nn.Linear(LATENT_DIM, 2)  # circle/square
        self.shape_blue_head = nn.Linear(LATENT_DIM, 2)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        rel_logits   = self.rel_head(z)
        scale_logits = self.scale_head(z)
        shape_r_logits = self.shape_red_head(z)
        shape_b_logits = self.shape_blue_head(z)
        return recon, rel_logits, scale_logits, shape_r_logits, shape_b_logits


# --------------------
# Training
# --------------------
def main():
    print(f"[train64-128] Using device: {DEVICE}")

    print("[train64-128] Generating dataset...")
    imgs, rel_lbl, scale_lbl, shape_r_lbl, shape_b_lbl = generate_dataset(N_SAMPLES)

    # normalise to [0,1] tensor
    imgs = imgs.float()

    # train/val split (90/10)
    n_train = int(0.9 * N_SAMPLES)
    idx = torch.randperm(N_SAMPLES)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:]

    def split(t):
        return t[train_idx], t[val_idx]

    imgs_tr, imgs_val = split(imgs)
    rel_tr,  rel_val  = split(rel_lbl)
    scale_tr, scale_val = split(scale_lbl)
    sr_tr, sr_val = split(shape_r_lbl)
    sb_tr, sb_val = split(shape_b_lbl)

    train_ds = TensorDataset(imgs_tr, rel_tr, scale_tr, sr_tr, sb_tr)
    val_ds   = TensorDataset(imgs_val, rel_val, scale_val, sr_val, sb_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = SceneModelEdgesRelscale().to(DEVICE)

    recon_loss_fn = nn.BCELoss()
    ce_loss_fn    = nn.CrossEntropyLoss()

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for x, rel, scale, sr, sb in train_loader:
            x = x.to(DEVICE)
            rel = rel.to(DEVICE)
            scale = scale.to(DEVICE)
            sr = sr.to(DEVICE)
            sb = sb.to(DEVICE)

            opt.zero_grad()

            recon, rel_logits, scale_logits, sr_logits, sb_logits = model(x)

            loss_recon  = recon_loss_fn(recon, x)
            loss_rel    = ce_loss_fn(rel_logits, rel)
            loss_scale  = ce_loss_fn(scale_logits, scale)
            loss_shape_r = ce_loss_fn(sr_logits, sr)
            loss_shape_b = ce_loss_fn(sb_logits, sb)

            loss = (
                LAMBDA_RECON * loss_recon
                + LAMBDA_REL * loss_rel
                + LAMBDA_SCALE * loss_scale
                + LAMBDA_SHAPE * (loss_shape_r + loss_shape_b) / 2.0
            )

            loss.backward()
            opt.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)

        # ---- validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, rel, scale, sr, sb in val_loader:
                x = x.to(DEVICE)
                rel = rel.to(DEVICE)
                scale = scale.to(DEVICE)
                sr = sr.to(DEVICE)
                sb = sb.to(DEVICE)

                recon, rel_logits, scale_logits, sr_logits, sb_logits = model(x)

                loss_recon  = recon_loss_fn(recon, x)
                loss_rel    = ce_loss_fn(rel_logits, rel)
                loss_scale  = ce_loss_fn(scale_logits, scale)
                loss_shape_r = ce_loss_fn(sr_logits, sr)
                loss_shape_b = ce_loss_fn(sb_logits, sb)

                loss = (
                    LAMBDA_RECON * loss_recon
                    + LAMBDA_REL * loss_rel
                    + LAMBDA_SCALE * loss_scale
                    + LAMBDA_SHAPE * (loss_shape_r + loss_shape_b) / 2.0
                )

                val_loss += loss.item() * x.size(0)

        val_loss /= len(val_loader.dataset)

        print(
            f"[train64-128] Epoch {epoch:3d}/{EPOCHS} | "
            f"train loss={train_loss:.4f} | val loss={val_loss:.4f}"
        )

    # save checkpoint
    ckpt = {
        "model_state_dict": model.state_dict(),
        "latent_dim": LATENT_DIM,
        "img_size": IMG_SIZE,
        "config": {
            "latent_dim": LATENT_DIM,
            "img_size": IMG_SIZE,
            "n_samples": N_SAMPLES,
            "epochs": EPOCHS,
        },
    }
    torch.save(ckpt, CKPT_PATH)
    print(f"[train64-128] Saved SceneModel -> {CKPT_PATH}")


if __name__ == "__main__":
    main()
