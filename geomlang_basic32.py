#!/usr/bin/env python3
# geomlang_basic32.py
#
# Clean rebuild:
#   - 32x32 images
#   - 3 channels: red object (ch 0), blue object (ch 2), empty green
#   - relations: left_of, right_of, above, below, inside, overlapping
#   - relative scale: red_larger, red_smaller, similar
#
# Trains a *pure* conv autoencoder, then dumps:
#   outputs_clean/
#       - ae_basic32.pt          (AE checkpoint)
#       - recon_grid_basic32.png (orig vs recon)
#       - latents_basic32.npz    (z, rel, scale, shapes)
#       - tsne_relations_basic32.png  (optional sanity check)
#
# This is intentionally minimal and independent of Codex changes.

import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

OUT_DIR       = "outputs_clean"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE      = 32
NUM_CHANNELS  = 3   # [0]=red, [2]=blue
LATENT_DIM    = 32

RELATION_NAMES = ["left_of", "right_of", "above", "below", "inside", "overlapping"]
SCALE_NAMES    = ["red_larger", "red_smaller", "similar"]
SHAPE_NAMES    = ["circle", "square"]  # 0 = circle, 1 = square

NUM_REL   = len(RELATION_NAMES)
NUM_SCALE = len(SCALE_NAMES)
NUM_SHAPE = len(SHAPE_NAMES)

NUM_SAMPLES = 8000
BATCH_SIZE  = 128
EPOCHS      = 200
LR          = 1e-3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# Utility: drawing primitives
# ---------------------------------------------------------------------

def draw_circle(grid, cx, cy, radius, channel_idx):
    h, w = grid.shape[1], grid.shape[2]
    yy, xx = np.ogrid[:h, :w]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    grid[channel_idx][mask] = 1.0

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
    else:              # square
        draw_square(grid, cx, cy, size, channel_idx)

# ---------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------

def rand_band(low, high, alt_low, alt_high):
    """Safe randint helper for potentially narrow bands."""
    if high <= low + 1:
        low, high = alt_low, alt_high
    return np.random.randint(low, high)

def generate_scene(rel_label, scale_label, shape_red, shape_blue):
    """
    rel_label: 0..5   (red relative to blue)
    scale_label: 0..2 (red_larger, red_smaller, similar)
    shape_red/shape_blue: 0=circle,1=square
    """
    img = np.zeros((NUM_CHANNELS, IMG_SIZE, IMG_SIZE), dtype=np.float32)

    # margin scales with image size
    margin = max(2, IMG_SIZE // 8)   # e.g. 32→4

    # ----- sizes -----
    base = np.random.randint(4, 7)  # 4..6 pixels

    if rel_label == 4:  # inside: force red smaller
        scale_label = 1  # red_smaller
        r_blue = min(base + 3, (IMG_SIZE - 2 * margin) // 2)
        r_blue = max(r_blue, 3)
        r_red  = max(2, r_blue - 2)
    else:
        if scale_label == 0:    # red_larger
            r_red, r_blue = base + 2, base
        elif scale_label == 1:  # red_smaller
            r_red, r_blue = base, base + 2
        else:                   # similar
            delta = np.random.randint(-1, 2)  # -1,0,1
            r_red, r_blue = base, base + delta

    full_low  = margin
    full_high = IMG_SIZE - margin
    mid       = IMG_SIZE // 2

    # ----- positions conditioned on relation -----
    if rel_label == 0:  # left_of
        cy = rand_band(full_low, full_high, full_low, full_high)
        cx_red  = rand_band(margin, mid - margin, full_low, mid)
        cx_blue = rand_band(mid + margin // 2, IMG_SIZE - margin,
                            mid, full_high)
        cy_red = cy_blue = cy

    elif rel_label == 1:  # right_of
        cy = rand_band(full_low, full_high, full_low, full_high)
        cx_blue = rand_band(margin, mid - margin, full_low, mid)
        cx_red  = rand_band(mid + margin // 2, IMG_SIZE - margin,
                            mid, full_high)
        cy_red = cy_blue = cy

    elif rel_label == 2:  # above
        cx = rand_band(full_low, full_high, full_low, full_high)
        cy_red  = rand_band(margin, mid - margin, full_low, mid)
        cy_blue = rand_band(mid + margin // 2, IMG_SIZE - margin,
                            mid, full_high)
        cx_red = cx_blue = cx

    elif rel_label == 3:  # below
        cx = rand_band(full_low, full_high, full_low, full_high)
        cy_blue = rand_band(margin, mid - margin, full_low, mid)
        cy_red  = rand_band(mid + margin // 2, IMG_SIZE - margin,
                            mid, full_high)
        cx_red = cx_blue = cx

    elif rel_label == 4:  # inside
        low = margin + r_blue
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
        offset_x = np.random.randint(-5, 6)
        offset_y = np.random.randint(-5, 6)
        cx_blue = int(np.clip(cx_red + offset_x, full_low,  full_high - 1))
        cy_blue = int(np.clip(cy_red + offset_y, full_low,  full_high - 1))

    # ----- draw shapes -----
    draw_shape(img, shape_red,  cx_red,  cy_red,  r_red,  channel_idx=0)
    draw_shape(img, shape_blue, cx_blue, cy_blue, r_blue, channel_idx=2)

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
        rel    = np.random.randint(0, NUM_REL)
        scale  = np.random.randint(0, NUM_SCALE)
        shape_r = np.random.randint(0, NUM_SHAPE)
        shape_b = np.random.randint(0, NUM_SHAPE)

        img, scale = generate_scene(rel, scale, shape_r, shape_b)

        scenes.append(img)
        rel_labels.append(rel)
        scale_labels.append(scale)
        shape_red_labels.append(shape_r)
        shape_blue_labels.append(shape_b)

    scenes            = torch.tensor(np.stack(scenes), dtype=torch.float32)
    rel_labels        = torch.tensor(rel_labels, dtype=torch.long)
    scale_labels      = torch.tensor(scale_labels, dtype=torch.long)
    shape_red_labels  = torch.tensor(shape_red_labels, dtype=torch.long)
    shape_blue_labels = torch.tensor(shape_blue_labels, dtype=torch.long)

    return scenes, rel_labels, scale_labels, shape_red_labels, shape_blue_labels

# ---------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, in_channels=NUM_CHANNELS, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)  # 16x16
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)           # 8x8
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)          # 4x4
        self.fc    = nn.Linear(128 * 4 * 4, latent_dim)

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
        self.fc      = nn.Linear(latent_dim, 128 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(128, 64,  kernel_size=4, stride=2, padding=1)  # 8x8
        self.deconv2 = nn.ConvTranspose2d(64,  32,  kernel_size=4, stride=2, padding=1)  # 16x16
        self.deconv3 = nn.ConvTranspose2d(32,  out_channels, kernel_size=4, stride=2, padding=1)  # 32x32

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 128, 4, 4)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        return x

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z    = self.encode(x)
        recon = self.decode(z)
        return recon, z

# ---------------------------------------------------------------------
# Viz helpers
# ---------------------------------------------------------------------

def channels_to_rgb(t):
    """
    t: [3,H,W] with channel 0=red object, 2=blue object.
    Returns [H,W,3] in [0,1].
    """
    x = t.detach().cpu().numpy()
    r = x[0]
    b = x[2]
    h, w = r.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[..., 0] = np.clip(r, 0.0, 1.0)  # red
    rgb[..., 2] = np.clip(b, 0.0, 1.0)  # blue
    return rgb

# ---------------------------------------------------------------------
# Training + latent dump
# ---------------------------------------------------------------------

def main():
    print(f"[basic32] Using device: {DEVICE}")

    scenes, rel_labels, scale_labels, shape_r, shape_b = generate_dataset()
    dataset = TensorDataset(scenes, rel_labels, scale_labels, shape_r, shape_b)

    # simple train/test split
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_test  = n_total - n_train
    train_ds, test_ds = torch.utils.data.random_split(dataset, [n_train, n_test])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    model = Autoencoder().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)

    # ---- training ----
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train = 0.0
        n_train_px  = 0

        for batch in train_loader:
            x = batch[0].to(DEVICE)

            opt.zero_grad()
            recon, _ = model(x)
            loss = F.binary_cross_entropy(recon, x)
            loss.backward()
            opt.step()

            total_train += loss.item() * x.size(0)
            n_train_px  += x.size(0)

        # test recon
        model.eval()
        total_test = 0.0
        n_test_px = 0
        with torch.no_grad():
            for batch in test_loader:
                x = batch[0].to(DEVICE)
                recon, _ = model(x)
                loss = F.binary_cross_entropy(recon, x)
                total_test += loss.item() * x.size(0)
                n_test_px  += x.size(0)

        if epoch % 20 == 0 or epoch == 1 or epoch == EPOCHS:
            print(f"[basic32] Epoch {epoch:3d}/{EPOCHS} | "
                  f"train BCE={total_train/n_train_px:.5f} | "
                  f"test BCE={total_test/n_test_px:.5f}")

    # ---- save checkpoint ----
    ckpt_path = os.path.join(OUT_DIR, "ae_basic32.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "latent_dim": LATENT_DIM,
            "img_size": IMG_SIZE,
            "num_channels": NUM_CHANNELS,
        }
    }, ckpt_path)
    print(f"[basic32] Saved AE -> {ckpt_path}")

    # ---- save recon grid ----
    model.eval()
    with torch.no_grad():
        x = scenes[:16].to(DEVICE)  # first 16 examples
        recon, _ = model(x)

    x = x.cpu()
    recon = recon.cpu()

    n = 16
    fig, axes = plt.subplots(2, n, figsize=(n * 1.2, 2.4), dpi=120)
    for i in range(n):
        axes[0, i].axis("off")
        axes[1, i].axis("off")

        rgb_orig = channels_to_rgb(x[i])
        rgb_rec  = channels_to_rgb(recon[i])

        axes[0, i].imshow(rgb_orig, interpolation="nearest")
        axes[1, i].imshow(rgb_rec,  interpolation="nearest")

    axes[0, 0].set_ylabel("orig", fontsize=10)
    axes[1, 0].set_ylabel("recon", fontsize=10)
    plt.tight_layout()
    recon_path = os.path.join(OUT_DIR, "recon_grid_basic32.png")
    plt.savefig(recon_path)
    plt.close(fig)
    print(f"[basic32] Saved recon grid -> {recon_path}")

    # ---- encode full dataset & dump latents ----
    model.eval()
    all_z = []
    with torch.no_grad():
        for i in range(0, len(scenes), BATCH_SIZE):
            batch = scenes[i:i + BATCH_SIZE].to(DEVICE)
            _, z = model(batch)
            all_z.append(z.cpu())

    all_z = torch.cat(all_z, dim=0).numpy()

    npz_path = os.path.join(OUT_DIR, "latents_basic32.npz")
    np.savez(
        npz_path,
        z=all_z,
        rel=rel_labels.numpy(),
        scale=scale_labels.numpy(),
        shape_red=shape_r.numpy(),
        shape_blue=shape_b.numpy(),
    )
    print(f"[basic32] Saved latents -> {npz_path} (z.shape={all_z.shape})")

    # ---- optional: t-SNE by relation for sanity ----
    try:
        print("[basic32] Running t-SNE on 3000 points for sanity plot...")

        idx = np.random.choice(all_z.shape[0],
                               size=min(3000, all_z.shape[0]),
                               replace=False)
        z_sample = all_z[idx]
        rel_sample = rel_labels.numpy()[idx]

        # Use only universally supported constructor args
        tsne = TSNE(
            n_components=2,
            perplexity=40,
            learning_rate=200,
            verbose=1,
        )

        # n_iter is passed here for compatibility
        z2d = tsne.fit_transform(z_sample)

        plt.figure(figsize=(6, 6), dpi=120)
        for r in range(NUM_REL):
            mask = rel_sample == r
            plt.scatter(z2d[mask, 0], z2d[mask, 1],
                        s=6, alpha=0.7, label=f"rel={r}:{RELATION_NAMES[r]}")
        plt.legend(fontsize=8, markerscale=2)
        plt.title("t-SNE of latents by relation (basic32)")
        tsne_path = os.path.join(OUT_DIR, "tsne_relations_basic32.png")
        plt.tight_layout()
        plt.savefig(tsne_path)
        plt.close()
        print(f"[basic32] Saved t-SNE plot -> {tsne_path}")

    except Exception as e:
        print(f"[basic32] t-SNE failed (ok to ignore): {e}")

if __name__ == "__main__":
    main()
