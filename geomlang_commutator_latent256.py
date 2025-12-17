#!/usr/bin/env python3
# geomlang_commutator_latent256.py
#
# Estimate a Lie-style commutator for the global relation generators g_x, g_y
# in the 64x64 edges+relscale model with LATENT_DIM=256.
#
# We treat g_x, g_y as *global* tangent directions in latent space, but define
# non-linear flows via the autoencoder:
#   F_x(z) = encode(decode(z + eps * g_x))
#   F_y(z) = encode(decode(z + eps * g_y))
#
# Then for seeds z_0 we estimate the commutator
#   c(z_0) ≈ (F_y(F_x(z_0)) - F_x(F_y(z_0))) / eps^2
#
# and decompose c into components parallel and perpendicular to span{g_x, g_y}.

import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

# -----------------------
# Config
# -----------------------
IMG_SIZE   = 64     # matches training
NUM_CH     = 3      # red, blue, edges
LATENT_DIM = 256

N_SAMPLES   = 6000  # dataset used to estimate means & pick seeds
BATCH_SIZE  = 128
N_SEEDS     = 400   # how many seeds for commutator estimates
EPS         = 0.4   # step size along g_x / g_y for flows

OUT_DIR         = "outputs_edges_relscale256"
CKPT_SCENEMODEL = os.path.join(OUT_DIR, "scene_model_edges_relscale256.pt")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REL_LEFT, REL_RIGHT, REL_ABOVE, REL_BELOW, REL_OVERLAP = range(5)
REL_NAMES = ["left_of", "right_of", "above", "below", "overlap"]

# -----------------------
# Scene generation (same as other 64x64 edges scripts)
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
    """Static 64x64 scenes with red/blue fill + edge channel."""

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
        scale = scale_class_from_size((size_red + size_blue) * 0.5)
        shape_r_lbl = shape_red
        shape_b_lbl = shape_blue

        return (
            torch.from_numpy(img),
            torch.tensor(rel,         dtype=torch.long),
            torch.tensor(scale,       dtype=torch.long),
            torch.tensor(shape_r_lbl, dtype=torch.long),
            torch.tensor(shape_b_lbl, dtype=torch.long),
        )

# -----------------------
# Model – same 64x64 edges architecture as training
# -----------------------

class Encoder(nn.Module):
    # 64 -> 32 -> 16 -> 8 -> 4
    def __init__(self, in_channels=NUM_CH, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32,  kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32,          64,  kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64,          128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128,         256, kernel_size=4, stride=2, padding=1)  # 4x4
        self.fc    = nn.Linear(256 * 4 * 4, latent_dim)  # 4096 -> 256

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


class SceneModelEdges256(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        # heads for checkpoint compatibility (unused here)
        self.rel_head      = nn.Linear(LATENT_DIM, 5)
        self.scale_head    = nn.Linear(LATENT_DIM, 3)
        self.shape_r_head  = nn.Linear(LATENT_DIM, 2)
        self.shape_b_head  = nn.Linear(LATENT_DIM, 2)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


def load_scene_model():
    print(f"[comm256] Loading SceneModel from {CKPT_SCENEMODEL}")
    ckpt = torch.load(CKPT_SCENEMODEL, map_location=DEVICE)
    model = SceneModelEdges256()
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.to(DEVICE)
    model.eval()
    return model

# -----------------------
# Helpers
# -----------------------

def orthonormalize_generators(mu_left, mu_right, mu_above, mu_below):
    """Return unit, approximately orthogonal g_x, g_y using Gram-Schmidt."""
    g_x_raw = mu_right - mu_left
    g_y_raw = mu_below - mu_above

    gx = g_x_raw / np.linalg.norm(g_x_raw)

    # remove gx component from gy_raw
    gy = g_y_raw - np.dot(g_y_raw, gx) * gx
    gy = gy / np.linalg.norm(gy)

    angle = math.degrees(math.acos(np.clip(np.dot(gx, gy), -1.0, 1.0)))
    print(f"[comm256] ||mu_right - mu_left|| = {np.linalg.norm(g_x_raw):.4f}")
    print(f"[comm256] ||mu_below - mu_above|| = {np.linalg.norm(g_y_raw):.4f}")
    print(f"[comm256] angle(g_x, g_y) = {angle:.2f}°")

    return gx, gy


def flow(model, z, dir_vec_torch, eps):
    """
    Non-linear flow: move along dir in latent, decode, re-encode.
    z: torch tensor (latent_dim,)
    dir_vec_torch: torch tensor (latent_dim,)
    """
    with torch.no_grad():
        z_step = z + eps * dir_vec_torch
        x_step = model.decode(z_step.unsqueeze(0))
        z_back = model.encode(x_step).squeeze(0)
    return z_back


# -----------------------
# Main
# -----------------------

def main():
    print(f"[comm256] Using device: {DEVICE}")
    model = load_scene_model()

    # ---- Generate dataset and encode ----
    print(f"[comm256] Generating dataset: N={N_SAMPLES}")
    ds = GeomEdges64Dataset(N_SAMPLES)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_rel = []
    all_z   = []

    with torch.no_grad():
        for imgs, rel, scale, s_r, s_b in dl:
            imgs = imgs.to(DEVICE)
            rel  = rel.to(DEVICE)
            z    = model.encode(imgs)

            all_rel.append(rel.cpu())
            all_z.append(z.cpu())

    rel_t = torch.cat(all_rel, dim=0)          # (N,)
    z_t   = torch.cat(all_z,   dim=0)          # (N, D)
    z_np  = z_t.numpy()
    rel_np = rel_t.numpy()
    N, D = z_np.shape
    print(f"[comm256] Latent shape: {z_np.shape}")

    # ---- Relation means and global generators ----
    mu_r = []
    for r in range(5):
        mask = (rel_np == r)
        count = int(mask.sum())
        mu = z_np[mask].mean(axis=0)
        mu_r.append(mu)
        print(f"[comm256] relation {r}={REL_NAMES[r]:>7}, count={count}, ||mu_r||={np.linalg.norm(mu):.3f}")

    mu_left   = mu_r[REL_LEFT]
    mu_right  = mu_r[REL_RIGHT]
    mu_above  = mu_r[REL_ABOVE]
    mu_below  = mu_r[REL_BELOW]

    g_x_np, g_y_np = orthonormalize_generators(mu_left, mu_right, mu_above, mu_below)

    gx_t = torch.from_numpy(g_x_np).to(DEVICE).float()
    gy_t = torch.from_numpy(g_y_np).to(DEVICE).float()

    # ---- Pick random seeds for commutator estimates ----
    rng = np.random.default_rng(0)
    idx_all = np.arange(N)
    idx_seeds = rng.choice(idx_all, size=min(N_SEEDS, N), replace=False)
    print(f"[comm256] Using {len(idx_seeds)} seeds with EPS={EPS}")

    comms = []
    comms_par = []
    comms_perp = []
    rel_seeds = []

    with torch.no_grad():
        for i in idx_seeds:
            z0_np = z_np[i]
            r0 = int(rel_np[i])
            rel_seeds.append(r0)

            z0_t = torch.from_numpy(z0_np).to(DEVICE).float()

            # F_y(F_x(z0))
            zx = flow(model, z0_t, gx_t, EPS)
            zxy = flow(model, zx, gy_t, EPS)

            # F_x(F_y(z0))
            zy = flow(model, z0_t, gy_t, EPS)
            zyx = flow(model, zy, gx_t, EPS)

            c_t = (zxy - zyx) / (EPS * EPS)
            c_np = c_t.cpu().numpy()

            # Decompose onto span{g_x, g_y}
            cx = np.dot(c_np, g_x_np)
            cy = np.dot(c_np, g_y_np)
            c_par = cx * g_x_np + cy * g_y_np
            c_perp = c_np - c_par

            comms.append(np.linalg.norm(c_np))
            comms_par.append(np.linalg.norm(c_par))
            comms_perp.append(np.linalg.norm(c_perp))

    comms = np.array(comms)
    comms_par = np.array(comms_par)
    comms_perp = np.array(comms_perp)
    frac_perp = comms_perp / np.maximum(comms, 1e-8)

    print("\n[comm256] Global commutator norm stats (||c||):")
    print(f"  mean = {comms.mean():.4f}")
    print(f"  std  = {comms.std():.4f}")
    print(f"  min  = {comms.min():.4f}")
    print(f"  max  = {comms.max():.4f}")

    print("\n[comm256] Parallel component ||c_parallel|| stats:")
    print(f"  mean = {comms_par.mean():.4f}")
    print(f"  std  = {comms_par.std():.4f}")
    print(f"  min  = {comms_par.min():.4f}")
    print(f"  max  = {comms_par.max():.4f}")

    print("\n[comm256] Perpendicular component ||c_perp|| stats:")
    print(f"  mean = {comms_perp.mean():.4f}")
    print(f"  std  = {comms_perp.std():.4f}")
    print(f"  min  = {comms_perp.min():.4f}")
    print(f"  max  = {comms_perp.max():.4f}")

    print("\n[comm256] Fraction of commutator in residual subspace ||c_perp|| / ||c||:")
    print(f"  mean = {frac_perp.mean():.4f}")
    print(f"  std  = {frac_perp.std():.4f}")
    print(f"  min  = {frac_perp.min():.4f}")
    print(f"  max  = {frac_perp.max():.4f}")

    # ---- Simple histograms ----
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
    ax[0].hist(comms, bins=30)
    ax[0].set_title("Comm. norm ||c|| (latent256)")
    ax[0].set_xlabel("||c||")
    ax[0].set_ylabel("count")

    ax[1].hist(frac_perp, bins=30)
    ax[1].set_title("Fraction in residual subspace")
    ax[1].set_xlabel("||c_perp|| / ||c||")
    ax[1].set_ylabel("count")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "commutator_hist_latent256.png")
    plt.savefig(out_path)
    print(f"[comm256] Saved commutator histograms -> {out_path}")

    print("\n[comm256] Done.")


if __name__ == "__main__":
    main()
