#!/usr/bin/env python3
# geomlang_lie_commutator_stats_generic.py
#
# Measure "curvature" of the latent relation manifold by
#   - constructing LR and AB generators from mean difference vectors
#   - running a small commutator loop:
#         z_xy = (z0 + eps*g_x) + eps*g_y
#         z_yx = (z0 + eps*g_y) + eps*g_x
#   - measuring ||z_xy - z_yx|| and image MSE
#
# Run once per checkpoint / latent size by editing the CONFIG block.

import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

# --------------------------------------------------
# CONFIG – EDIT THESE PER MODEL
# --------------------------------------------------
IMG_SIZE   = 64          # all your geomlang edges models are 64x64
NUM_CH     = 3           # red, blue, edges
LATENT_DIM = 128          # <-- set to 32 / 64 / 128 / 256 as needed

OUT_DIR          = "outputs_edges_relscale"       # 64-dim model dir
CKPT_SCENEMODEL  = os.path.join(OUT_DIR, "scene_model_edges_relscale.pt")

# For the 256-dim run, change to:
# LATENT_DIM = 256
# OUT_DIR          = "outputs_edges_relscale256"
# CKPT_SCENEMODEL  = os.path.join(OUT_DIR, "scene_model_edges_relscale256.pt")

os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REL_LEFT, REL_RIGHT, REL_ABOVE, REL_BELOW, REL_OVERLAP = range(5)
REL_NAMES = ["left_of", "right_of", "above", "below", "overlap"]

N_SAMPLES   = 6000   # dataset size for stats
BATCH_SIZE  = 256
N_BASE_COMM = 1000   # how many base points for commutator sampling
EPS         = 0.6    # step size along each generator

# --------------------------------------------------
# Scene generation (same as edges64)
# --------------------------------------------------

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
    def __init__(self, n_samples):
        super().__init__()
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        H = W = IMG_SIZE
        margin = 6

        shape_red  = np.random.randint(0, 2)   # 0 circle, 1 square
        shape_blue = np.random.randint(0, 2)

        base = np.random.randint(5, 13)
        size_red  = int(np.clip(base + np.random.randint(-2, 3), 4, 14))
        size_blue = int(np.clip(base + np.random.randint(-2, 3), 4, 14))

        def sample_center(s):
            cx = np.random.randint(margin + s, W - margin - s)
            cy = np.random.randint(margin + s, H - margin - s)
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
        img   = np.stack([red, blue, edges], axis=0)

        rel   = relation_from_centers(cx_r, cy_r, cx_b, cy_b)
        scale = scale_class_from_size(0.5 * (size_red + size_blue))
        shape_r_lbl = shape_red
        shape_b_lbl = shape_blue

        return (
            torch.from_numpy(img),
            torch.tensor(rel, dtype=torch.long),
            torch.tensor(scale, dtype=torch.long),
            torch.tensor(shape_r_lbl, dtype=torch.long),
            torch.tensor(shape_b_lbl, dtype=torch.long),
        )

# --------------------------------------------------
# Model – same conv stack for all latent sizes
# --------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, in_channels=NUM_CH, latent_dim=LATENT_DIM):
        super().__init__()
        # 64 -> 32 -> 16 -> 8 -> 4
        self.conv1 = nn.Conv2d(in_channels, 32,  kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32,          64,  kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64,          128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128,         256, kernel_size=4, stride=2, padding=1)  # 4x4
        self.fc    = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

class Decoder(nn.Module):
    def __init__(self, out_channels=NUM_CH, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc     = nn.Linear(latent_dim, 256 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64,  kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64,  32,  kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32,  out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, 4, 4)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))
        return x

class SceneModelEdges(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        # heads for checkpoint compatibility (not used in analysis)
        self.rel_head      = nn.Linear(LATENT_DIM, 5)
        self.scale_head    = nn.Linear(LATENT_DIM, 3)
        self.shape_r_head  = nn.Linear(LATENT_DIM, 2)
        self.shape_b_head  = nn.Linear(LATENT_DIM, 2)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

def load_scene_model():
    print(f"[comm-generic] Loading SceneModel from {CKPT_SCENEMODEL}")
    ckpt = torch.load(CKPT_SCENEMODEL, map_location=DEVICE)
    model = SceneModelEdges()
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt, strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.to(DEVICE)
    model.eval()
    return model

# --------------------------------------------------
# Helper: build LR / AB difference clouds
# --------------------------------------------------

def build_difference_clouds(z_np, rel_np):
    rng = np.random.default_rng(0)

    idx_left   = np.where(rel_np == REL_LEFT)[0]
    idx_right  = np.where(rel_np == REL_RIGHT)[0]
    idx_above  = np.where(rel_np == REL_ABOVE)[0]
    idx_below  = np.where(rel_np == REL_BELOW)[0]

    def make_pairs(idxs_a, idxs_b):
        n = min(len(idxs_a), len(idxs_b))
        if n == 0:
            return np.array([], dtype=int), np.array([], dtype=int)
        idxs_a = rng.permutation(idxs_a)[:n]
        idxs_b = rng.permutation(idxs_b)[:n]
        return idxs_a, idxs_b

    left_idx, right_idx   = make_pairs(idx_left, idx_right)
    above_idx, below_idx  = make_pairs(idx_above, idx_below)

    v_lr = z_np[right_idx] - z_np[left_idx]
    v_ab = z_np[below_idx] - z_np[above_idx]

    return v_lr, v_ab

def generator_from_cloud(v, name):
    if v.shape[0] == 0:
        raise RuntimeError(f"No pairs for {name}")
    norms = np.linalg.norm(v, axis=1)
    print(f"\n[comm-generic] {name} norms:")
    print(f"  mean={norms.mean():.4f}, std={norms.std():.4f}, "
          f"min={norms.min():.4f}, max={norms.max():.4f}")
    v_mean = v.mean(axis=0)
    mean_norm = np.linalg.norm(v_mean)
    print(f"[comm-generic] ||mean_{name}|| = {mean_norm:.4f}")
    return v_mean, mean_norm

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    print(f"[comm-generic] Using device: {DEVICE}, LATENT_DIM={LATENT_DIM}")
    model = load_scene_model()

    # Build dataset
    print(f"[comm-generic] Generating dataset: N={N_SAMPLES}")
    ds = GeomEdges64Dataset(N_SAMPLES)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_z   = []
    all_rel = []

    with torch.no_grad():
        for imgs, rel, scale, s_r, s_b in dl:
            imgs = imgs.to(DEVICE)
            z    = model.encode(imgs)
            all_z.append(z.cpu())
            all_rel.append(rel)

    all_z   = torch.cat(all_z, dim=0)        # [N,D]
    all_rel = torch.cat(all_rel, dim=0)      # [N]
    z_np    = all_z.numpy()
    rel_np  = all_rel.numpy()
    N = z_np.shape[0]

    # Relation counts (just for sanity)
    for r in range(5):
        n = int((rel_np == r).sum())
        print(f"[comm-generic] relation {r}: {REL_NAMES[r]:>7}, count={n}")

    # Build LR/AB clouds
    v_lr, v_ab = build_difference_clouds(z_np, rel_np)
    v_lr_mean, norm_lr = generator_from_cloud(v_lr, "left→right")
    v_ab_mean, norm_ab = generator_from_cloud(v_ab, "above→below")

    # Unit generators
    g_x = v_lr_mean / (norm_lr + 1e-8)
    g_y = v_ab_mean / (norm_ab + 1e-8)

    print(f"\n[comm-generic] step EPS = {EPS:.3f}, EPS^2 = {EPS*EPS:.4f}")

    # --------------------------------------------------
    # Commutator stats over random base points
    # --------------------------------------------------
    rng = np.random.default_rng(1)
    base_idx = rng.choice(N, size=min(N_BASE_COMM, N), replace=False)

    latent_norms = []
    image_mses   = []

    with torch.no_grad():
        g_x_t = torch.from_numpy(g_x).to(DEVICE).view(1, -1)
        g_y_t = torch.from_numpy(g_y).to(DEVICE).view(1, -1)

        for i in base_idx:
            z0 = all_z[i:i+1].to(DEVICE)  # [1,D]

            z_x  = z0 + EPS * g_x_t
            z_xy = z_x + EPS * g_y_t

            z_y  = z0 + EPS * g_y_t
            z_yx = z_y + EPS * g_x_t

            # latent norm
            diff_latent = (z_xy - z_yx).cpu().numpy().reshape(-1)
            latent_norms.append(np.linalg.norm(diff_latent))

            # image MSE
            img_xy = model.decode(z_xy).cpu()
            img_yx = model.decode(z_yx).cpu()
            mse = torch.mean((img_xy - img_yx) ** 2).item()
            image_mses.append(mse)

    latent_norms = np.array(latent_norms)
    image_mses   = np.array(image_mses)

    print("\n[comm-generic] latent commutator norm ||z_xy - z_yx||:")
    print(f"  mean={latent_norms.mean():.6e}, std={latent_norms.std():.6e}, "
          f"min={latent_norms.min():.6e}, max={latent_norms.max():.6e}")

    print("\n[comm-generic] image commutator MSE:")
    print(f"  mean={image_mses.mean():.6e}, std={image_mses.std():.6e}, "
          f"min={image_mses.min():.6e}, max={image_mses.max():.6e}")

    ratio = latent_norms.mean() / (EPS * EPS)
    print(f"\n[comm-generic] mean latent commutator / EPS^2 = {ratio:.6e}")

    # --------------------------------------------------
    # Hist plots
    # --------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=120)
    axes[0].hist(latent_norms, bins=40)
    axes[0].set_title("Latent commutator norms\n||z_xy - z_yx||")
    axes[0].set_xlabel("norm")
    axes[0].set_ylabel("count")

    axes[1].hist(image_mses, bins=40)
    axes[1].set_title("Image commutator MSE")
    axes[1].set_xlabel("MSE")
    axes[1].set_ylabel("count")

    plt.tight_layout()

    tag = f"latent{LATENT_DIM}"
    out_path = os.path.join(OUT_DIR, f"lie_commutator_stats_{tag}.png")
    plt.savefig(out_path)
    print(f"[comm-generic] Saved histograms -> {out_path}")

if __name__ == "__main__":
    main()
