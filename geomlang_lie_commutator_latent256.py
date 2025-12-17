#!/usr/bin/env python3
# geomlang_lie_commutator_latent256.py
#
# Empirical Lie bracket / commutator test for the 256-dim latent model.
#
# - Loads SceneModelEdges256 from:
#       outputs_edges_relscale256/scene_model_edges_relscale256.pt
# - Regenerates a 6000-sample dataset (64x64, 3 channels: red, blue, edges)
# - Encodes latents, builds left→right and above→below difference clouds
# - Computes mean relation vectors v_LR, v_AB and normalizes them
# - For a few base scenes, constructs a "commutator loop" in latent space:
#
#       z0
#       |          (g_y)
#      z_x  ---->  z_xy
#   (g_x)          ^
#       v          |
#      z_y  ---->  z_yx
#               (g_x)
#
# - Decodes all points and renders grids to see if z_xy and z_yx coincide
#   (perfect commutativity) or differ (non-zero Lie bracket).
#
# Run:
#   python bbit_geomlang/geomlang_lie_commutator_latent256.py

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
IMG_SIZE   = 64           # image resolution used by this model
NUM_CH     = 3            # red, blue, edges
LATENT_DIM = 256

N_SAMPLES  = 6000
BATCH_SIZE = 256

OUT_DIR          = "outputs_edges_relscale256"
CKPT_SCENEMODEL  = os.path.join(OUT_DIR, "scene_model_edges_relscale256.pt")
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


class GeomEdgesDataset(Dataset):
    def __init__(self, n_samples):
        super().__init__()
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        H = W = IMG_SIZE
        margin = 6

        shape_red = np.random.randint(0, 2)   # 0 circle, 1 square
        shape_blue = np.random.randint(0, 2)

        base = np.random.randint(5, 13)
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
        img = np.stack([red, blue, edges], axis=0)

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
# Model definition (matches training)
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
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1)

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
        # heads kept for checkpoint compatibility (unused here)
        self.rel_head   = nn.Linear(LATENT_DIM, 5)
        self.scale_head = nn.Linear(LATENT_DIM, 3)
        self.shape_r_head = nn.Linear(LATENT_DIM, 2)
        self.shape_b_head = nn.Linear(LATENT_DIM, 2)

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
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt, strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.to(DEVICE)
    model.eval()
    return model

# -----------------------
# Utility
# -----------------------

def chw_to_rgb(img_chw):
    c, h, w = img_chw.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[..., 0] = img_chw[0]          # red
    rgb[..., 1] = img_chw[2]          # edges
    rgb[..., 2] = img_chw[1]          # blue
    return rgb

# -----------------------
# Main
# -----------------------

def main():
    print(f"[comm256] Using device: {DEVICE}")
    model = load_scene_model()

    # Generate dataset and encode
    print(f"[comm256] Generating dataset: N={N_SAMPLES}")
    ds = GeomEdgesDataset(N_SAMPLES)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_imgs, all_rel, all_z = [], [], []
    with torch.no_grad():
        for imgs, rel, scale, s_r, s_b in dl:
            imgs = imgs.to(DEVICE)
            rel  = rel.to(DEVICE)
            z = model.encode(imgs)

            all_imgs.append(imgs.cpu())
            all_rel.append(rel.cpu())
            all_z.append(z.cpu())

    all_imgs = torch.cat(all_imgs, dim=0)
    all_rel  = torch.cat(all_rel, dim=0)
    all_z    = torch.cat(all_z, dim=0)

    rel_np = all_rel.numpy()
    z_np   = all_z.numpy()

    # Indices by relation
    idx_left  = np.where(rel_np == REL_LEFT)[0]
    idx_right = np.where(rel_np == REL_RIGHT)[0]
    idx_above = np.where(rel_np == REL_ABOVE)[0]
    idx_below = np.where(rel_np == REL_BELOW)[0]

    rng = np.random.default_rng(0)

    def make_pairs(idxs_a, idxs_b):
        n = min(len(idxs_a), len(idxs_b))
        idxs_a = rng.permutation(idxs_a)[:n]
        idxs_b = rng.permutation(idxs_b)[:n]
        return idxs_a, idxs_b

    left_idx, right_idx = make_pairs(idx_left, idx_right)
    above_idx, below_idx = make_pairs(idx_above, idx_below)

    z_left  = z_np[left_idx]
    z_right = z_np[right_idx]
    z_above = z_np[above_idx]
    z_below = z_np[below_idx]

    v_lr = z_right - z_left
    v_ab = z_below - z_above

    # Mean relation vectors and normalization
    v_lr_mean = v_lr.mean(axis=0)
    v_ab_mean = v_ab.mean(axis=0)

    v_lr_norm = v_lr_mean / (np.linalg.norm(v_lr_mean) + 1e-8)
    v_ab_norm = v_ab_mean / (np.linalg.norm(v_ab_mean) + 1e-8)

    print(f"[comm256] ||v_LR_mean|| = {np.linalg.norm(v_lr_mean):.4f}")
    print(f"[comm256] ||v_AB_mean|| = {np.linalg.norm(v_ab_mean):.4f}")

    # Choose a few base indices (mix of relations)
    base_indices = []
    # 2 left_of
    if len(idx_left) > 0:
        base_indices.extend(list(rng.choice(idx_left, size=min(2, len(idx_left)), replace=False)))
    # 2 above
    if len(idx_above) > 0:
        base_indices.extend(list(rng.choice(idx_above, size=min(2, len(idx_above)), replace=False)))
    base_indices = np.unique(np.array(base_indices, dtype=int))
    print(f"[comm256] Base indices: {base_indices}")

    eps = 0.6  # small square step size

    for b_i, base_idx in enumerate(base_indices):
        z0 = all_z[base_idx:base_idx+1].to(DEVICE)      # [1,D]
        img0 = all_imgs[base_idx]                       # [3,64,64]
        rel0 = REL_NAMES[int(all_rel[base_idx].item())]

        # directions as torch
        gx = torch.from_numpy(v_lr_norm).to(DEVICE).view(1, -1).float()
        gy = torch.from_numpy(v_ab_norm).to(DEVICE).view(1, -1).float()

        # Square loop:
        # z_x   = z0 + eps * gx
        # z_xy  = z0 + eps * gx + eps * gy
        # z_y   = z0 + eps * gy
        # z_yx  = z0 + eps * gy + eps * gx  (same algebraically, but decoded diff shows nonlinearity)
        z_x  = z0 + eps * gx
        z_xy = z0 + eps * gx + eps * gy
        z_y  = z0 + eps * gy
        z_yx = z0 + eps * gy + eps * gx

        with torch.no_grad():
            imgs_dec = model.decode(torch.cat([z0, z_x, z_xy, z_y, z_yx], dim=0)).cpu()

        # Build grid: 2 rows × 3 cols for readability
        # Row 0: z0, z_x, z_xy
        # Row 1: z0, z_y, z_yx
        fig, axes = plt.subplots(2, 3, figsize=(6, 4), dpi=120)
        fig.suptitle(
            f"Commutator loop around base idx={base_idx} "
            f"(rel={rel0}, eps={eps:.2f})",
            fontsize=10
        )

        def show(ax, img_chw, title):
            ax.axis("off")
            ax.set_title(title, fontsize=8)
            ax.imshow(chw_to_rgb(img_chw.detach().numpy()))

        show(axes[0, 0], img0, "z0 (data)")
        show(axes[0, 1], imgs_dec[1], "z_x = z0+eps*g_x")
        show(axes[0, 2], imgs_dec[2], "z_xy (x then y)")

        show(axes[1, 0], imgs_dec[0], "z0 (recon)")
        show(axes[1, 1], imgs_dec[3], "z_y = z0+eps*g_y")
        show(axes[1, 2], imgs_dec[4], "z_yx (y then x)")

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        out_path = os.path.join(
            OUT_DIR,
            f"lie_commutator_base_idx{base_idx}_latent256.png"
        )
        plt.savefig(out_path)
        plt.close(fig)
        print(f"[comm256] Saved commutator grid -> {out_path}")

    print("[comm256] Done.")

if __name__ == "__main__":
    main()
