#!/usr/bin/env python3
# geomlang_multishape_relscale.py
#
# Multichannel BBIT demo:
# - 2 objects: red + blue
# - Shapes: circle or square (per object)
# - Spatial relations (red relative to blue):
#     left_of, right_of, above, below, inside, overlapping
# - Relative scale labels:
#     red_larger, red_smaller, similar
#
# A shared latent z encodes the full scene; simple linear heads read out:
#   - relation
#   - scale
#   - red shape
#   - blue shape

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

IMG_SIZE = 32
NUM_CHANNELS = 3  # RGB
LATENT_DIM = 48

RELATION_NAMES = ["left_of", "right_of", "above", "below", "inside", "overlapping"]
SCALE_NAMES = ["red_larger", "red_smaller", "similar"]
SHAPE_NAMES = ["circle", "square"]  # 0 = circle, 1 = square

NUM_REL = len(RELATION_NAMES)
NUM_SCALE = len(SCALE_NAMES)
NUM_SHAPE = len(SHAPE_NAMES)

NUM_SAMPLES = 4000
BATCH_SIZE = 64
EPOCHS = 600
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# Utility: drawing primitives
# ---------------------------------------------------------------------

def draw_circle(grid, cx, cy, radius, channel_idx):
    h, w = grid.shape[1], grid.shape[2]
    for y in range(h):
        for x in range(w):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                grid[channel_idx, y, x] = 1.0


def draw_square(grid, cx, cy, half_size, channel_idx):
    h, w = grid.shape[1], grid.shape[2]
    x0 = max(0, cx - half_size)
    x1 = min(w, cx + half_size + 1)
    y0 = max(0, cy - half_size)
    y1 = min(h, cy + half_size + 1)
    grid[channel_idx, y0:y1, x0:x1] = 1.0


def draw_shape(grid, shape_id, cx, cy, size, channel_idx):
    if shape_id == 0:  # circle
        draw_circle(grid, cx, cy, size, channel_idx)
    else:  # square
        draw_square(grid, cx, cy, size, channel_idx)


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

def generate_scene(rel_label, scale_label, shape_red, shape_blue):
    """
    rel_label: 0..5   (red relative to blue)
    scale_label: 0..2 (red_larger, red_smaller, similar)
    shape_red/shape_blue: 0=circle,1=square

    Uses robust randint ranges that won't collapse on small IMG_SIZE.
    """
    img = np.zeros((NUM_CHANNELS, IMG_SIZE, IMG_SIZE), dtype=np.float32)

    # margin scales with image size so we don't choke on small grids
    margin = max(2, IMG_SIZE // 8)   # e.g. 32 → 4, 64 → 8

    # ---------- choose base sizes ----------
    base = np.random.randint(4, 7)  # 4..6

    # ---------- sizes (with special handling for 'inside') ----------
    if rel_label == 4:  # inside: red strictly smaller than blue
        scale_label = 1  # force red_smaller
        # largest blue radius that fits within margins
        r_blue = min(base + 3, (IMG_SIZE - 2 * margin) // 2)
        r_blue = max(r_blue, 3)
        r_red = max(2, r_blue - 2)
    else:
        if scale_label == 0:  # red_larger
            r_red = base + 2
            r_blue = base
        elif scale_label == 1:  # red_smaller
            r_red = base
            r_blue = base + 2
        else:  # similar
            delta = np.random.randint(-1, 2)  # -1,0,1
            r_red = base
            r_blue = base + delta

    # helpful global fallback range (whole interior of image)
    full_low = margin
    full_high = IMG_SIZE - margin
    mid = IMG_SIZE // 2

    # ---------- positions conditioned on relation ----------
    if rel_label == 0:  # left_of (red left of blue)
        cy = rand_band(full_low, full_high, full_low, full_high)

        # try to bias red to left half, blue to right half
        cx_red = rand_band(margin, mid - margin, full_low, mid)
        cx_blue = rand_band(mid + margin // 2, IMG_SIZE - margin,
                            mid, full_high)

        cy_red = cy_blue = cy

    elif rel_label == 1:  # right_of (red right of blue)
        cy = rand_band(full_low, full_high, full_low, full_high)

        cx_blue = rand_band(margin, mid - margin, full_low, mid)
        cx_red = rand_band(mid + margin // 2, IMG_SIZE - margin,
                           mid, full_high)

        cy_red = cy_blue = cy

    elif rel_label == 2:  # above (red above blue)
        cx = rand_band(full_low, full_high, full_low, full_high)

        cy_red = rand_band(margin, mid - margin, full_low, mid)
        cy_blue = rand_band(mid + margin // 2, IMG_SIZE - margin,
                            mid, full_high)

        cx_red = cx_blue = cx

    elif rel_label == 3:  # below (red below blue)
        cx = rand_band(full_low, full_high, full_low, full_high)

        cy_blue = rand_band(margin, mid - margin, full_low, mid)
        cy_red = rand_band(mid + margin // 2, IMG_SIZE - margin,
                           mid, full_high)

        cx_red = cx_blue = cx

    elif rel_label == 4:  # inside (red inside blue)
        # pick a center that keeps BLUE inside the frame
        low = margin + r_blue
        high = IMG_SIZE - margin - r_blue

        if high <= low + 1:
            low = full_low
            high = full_high

        cx = np.random.randint(low, high)
        cy = np.random.randint(low, high)
        cx_red = cx_blue = cx
        cy_red = cy_blue = cy

    else:  # overlapping
        # pick red anywhere in the interior
        cx_red = np.random.randint(full_low, full_high)
        cy_red = np.random.randint(full_low, full_high)

        # small offset for blue so shapes overlap but aren't identical
        offset_x = np.random.randint(-5, 6)
        offset_y = np.random.randint(-5, 6)
        cx_blue = int(np.clip(cx_red + offset_x, full_low, full_high - 1))
        cy_blue = int(np.clip(cy_red + offset_y, full_low, full_high - 1))

    # ---------- draw shapes ----------
    draw_shape(img, shape_red, cx_red, cy_red, r_red, channel_idx=0)   # red
    draw_shape(img, shape_blue, cx_blue, cy_blue, r_blue, channel_idx=2)  # blue

    img = np.clip(img, 0.0, 1.0)
    return img, scale_label


def generate_dataset(num_samples=NUM_SAMPLES, seed=0):
    np.random.seed(seed)
    random.seed(seed)

    scenes = []
    rel_labels = []
    scale_labels = []
    shape_red_labels = []
    shape_blue_labels = []

    for _ in range(num_samples):
        rel = np.random.randint(0, NUM_REL)
        scale = np.random.randint(0, NUM_SCALE)
        shape_r = np.random.randint(0, NUM_SHAPE)
        shape_b = np.random.randint(0, NUM_SHAPE)

        img, scale = generate_scene(rel, scale, shape_r, shape_b)

        scenes.append(img)
        rel_labels.append(rel)
        scale_labels.append(scale)
        shape_red_labels.append(shape_r)
        shape_blue_labels.append(shape_b)

    scenes = torch.tensor(np.stack(scenes), dtype=torch.float32)
    rel_labels = torch.tensor(rel_labels, dtype=torch.long)
    scale_labels = torch.tensor(scale_labels, dtype=torch.long)
    shape_red_labels = torch.tensor(shape_red_labels, dtype=torch.long)
    shape_blue_labels = torch.tensor(shape_blue_labels, dtype=torch.long)

    return scenes, rel_labels, scale_labels, shape_red_labels, shape_blue_labels


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, in_channels=NUM_CHANNELS, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1)  # 16x16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)           # 8x8
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)           # 4x4
        self.fc = nn.Linear(64 * 4 * 4, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z


class Decoder(nn.Module):
    def __init__(self, out_channels=NUM_CHANNELS, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 8x8
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)  # 16x16
        self.deconv3 = nn.ConvTranspose2d(16, out_channels, kernel_size=4, stride=2, padding=1)  # 32x32

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 64, 4, 4)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        return x


class SceneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.rel_head = nn.Linear(LATENT_DIM, NUM_REL)
        self.scale_head = nn.Linear(LATENT_DIM, NUM_SCALE)
        self.shape_red_head = nn.Linear(LATENT_DIM, NUM_SHAPE)
        self.shape_blue_head = nn.Linear(LATENT_DIM, NUM_SHAPE)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        rel_logits = self.rel_head(z)
        scale_logits = self.scale_head(z)
        shape_r_logits = self.shape_red_head(z)
        shape_b_logits = self.shape_blue_head(z)
        return recon, z, rel_logits, scale_logits, shape_r_logits, shape_b_logits


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train():
    print(f"Using device: {DEVICE}")
    scenes, rel_labels, scale_labels, shape_r, shape_b = generate_dataset()
    dataset = TensorDataset(scenes, rel_labels, scale_labels, shape_r, shape_b)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SceneModel().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_recon = total_rel = total_scale = total_sr = total_sb = 0.0
        correct_rel = correct_scale = correct_sr = correct_sb = 0
        total_samples = 0

        for x, rel, scale, sr, sb in loader:
            x = x.to(DEVICE)
            rel = rel.to(DEVICE)
            scale = scale.to(DEVICE)
            sr = sr.to(DEVICE)
            sb = sb.to(DEVICE)

            opt.zero_grad()
            recon, z, rel_logits, scale_logits, sr_logits, sb_logits = model(x)

            bce = F.binary_cross_entropy(recon, x)
            rel_ce = F.cross_entropy(rel_logits, rel)
            scale_ce = F.cross_entropy(scale_logits, scale)
            sr_ce = F.cross_entropy(sr_logits, sr)
            sb_ce = F.cross_entropy(sb_logits, sb)

            loss = bce + rel_ce + scale_ce + sr_ce + sb_ce
            loss.backward()
            opt.step()

            batch_size = x.size(0)
            total_samples += batch_size

            total_recon += bce.item() * batch_size
            total_rel += rel_ce.item() * batch_size
            total_scale += scale_ce.item() * batch_size
            total_sr += sr_ce.item() * batch_size
            total_sb += sb_ce.item() * batch_size

            correct_rel += (rel_logits.argmax(dim=1) == rel).sum().item()
            correct_scale += (scale_logits.argmax(dim=1) == scale).sum().item()
            correct_sr += (sr_logits.argmax(dim=1) == sr).sum().item()
            correct_sb += (sb_logits.argmax(dim=1) == sb).sum().item()

        if epoch % 50 == 0 or epoch == EPOCHS - 1:
            print(
                f"Epoch {epoch}/{EPOCHS-1} | "
                f"Recon: {total_recon/total_samples:.5f} | "
                f"RelCE: {total_rel/total_samples:.5f} | "
                f"ScaleCE: {total_scale/total_samples:.5f} | "
                f"ShapeRCE: {total_sr/total_samples:.5f} | "
                f"ShapeBCE: {total_sb/total_samples:.5f} | "
                f"Acc_rel: {100*correct_rel/total_samples:.2f}% | "
                f"Acc_scale: {100*correct_scale/total_samples:.2f}% | "
                f"Acc_shapeR: {100*correct_sr/total_samples:.2f}% | "
                f"Acc_shapeB: {100*correct_sb/total_samples:.2f}%"
            )

    # -----------------------------------------------------------------
    # Latent analysis (relations, scale, shape)
    # -----------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        all_z = []
        all_rel = []
        all_scale = []
        all_sr = []
        all_sb = []

        for x, rel, scale, sr, sb in loader:
            x = x.to(DEVICE)
            z = model.encode(x)
            all_z.append(z.cpu())
            all_rel.append(rel)
            all_scale.append(scale)
            all_sr.append(sr)
            all_sb.append(sb)

        all_z = torch.cat(all_z, dim=0)
        all_rel = torch.cat(all_rel, dim=0)
        all_scale = torch.cat(all_scale, dim=0)
        all_sr = torch.cat(all_sr, dim=0)
        all_sb = torch.cat(all_sb, dim=0)

    def mean_by_label(labels, num_classes, name_list):
        means = []
        for i in range(num_classes):
            mask = (labels == i)
            z_mean = all_z[mask].mean(dim=0)
            means.append(z_mean)
            print(f"  {name_list[i]}:\n    {z_mean.numpy()}\n")
        return means

    print("\nFinal metrics:")
    print(f"  Recon BCE:   {total_recon/total_samples:.6f}")
    print(f"  Rel CE:      {total_rel/total_samples:.6f}")
    print(f"  Scale CE:    {total_scale/total_samples:.6f}")
    print(f"  ShapeR CE:   {total_sr/total_samples:.6f}")
    print(f"  ShapeB CE:   {total_sb/total_samples:.6f}")

    print("\nMean latent per relation:")
    rel_means = mean_by_label(all_rel, NUM_REL, RELATION_NAMES)

    print("Pairwise L2 distances between relation means:")
    for i in range(NUM_REL):
        for j in range(i + 1, NUM_REL):
            d = torch.norm(rel_means[i] - rel_means[j]).item()
            print(f"  {RELATION_NAMES[i]} <-> {RELATION_NAMES[j]}: {d:.4f}")
    print()

    print("Mean latent per scale label:")
    scale_means = mean_by_label(all_scale, NUM_SCALE, SCALE_NAMES)

    print("Pairwise L2 distances between scale means:")
    for i in range(NUM_SCALE):
        for j in range(i + 1, NUM_SCALE):
            d = torch.norm(scale_means[i] - scale_means[j]).item()
            print(f"  {SCALE_NAMES[i]} <-> {SCALE_NAMES[j]}: {d:.4f}")
    print()

    print("Mean latent per RED shape:")
    red_shape_means = mean_by_label(all_sr, NUM_SHAPE, [f"red_{n}" for n in SHAPE_NAMES])

    print("Pairwise L2 distances between RED shape means:")
    for i in range(NUM_SHAPE):
        for j in range(i + 1, NUM_SHAPE):
            d = torch.norm(red_shape_means[i] - red_shape_means[j]).item()
            print(f"  red_{SHAPE_NAMES[i]} <-> red_{SHAPE_NAMES[j]}: {d:.4f}")
    print()

    print("Mean latent per BLUE shape:")
    blue_shape_means = mean_by_label(all_sb, NUM_SHAPE, [f"blue_{n}" for n in SHAPE_NAMES])

    print("Pairwise L2 distances between BLUE shape means:")
    for i in range(NUM_SHAPE):
        for j in range(i + 1, NUM_SHAPE):
            d = torch.norm(blue_shape_means[i] - blue_shape_means[j]).item()
            print(f"  blue_{SHAPE_NAMES[i]} <-> blue_{SHAPE_NAMES[j]}: {d:.4f}")
    print()


if __name__ == "__main__":
    train()
