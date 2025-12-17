#!/usr/bin/env python3
# geomlang_residual_pca_curvature_latent256.py
#
# Analyse how commutator curvature is distributed inside the residual
# subspace orthogonal to the global relation generators g_x, g_y.
#
# - Train config: 64x64, 3 channels (red, blue, edges), LATENT_DIM=256
# - Checkpoint: outputs_edges_relscale256/scene_model_edges_relscale256.pt

import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -----------------------
# Config
# -----------------------
IMG_SIZE   = 64
NUM_CH     = 3
LATENT_DIM = 256

N_SAMPLES  = 6000
BATCH_SIZE = 128

OUT_DIR         = "outputs_edges_relscale256"
CKPT_SCENEMODEL = os.path.join(OUT_DIR, "scene_model_edges_relscale256.pt")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REL_LEFT, REL_RIGHT, REL_ABOVE, REL_BELOW, REL_OVERLAP = range(5)
REL_NAMES = ["left_of", "right_of", "above", "below", "overlap"]

# -----------------------
# Scene generation (64x64) – same as other edges_relscale256 scripts
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
# Model – 64x64 edges architecture with LATENT_DIM=256
# -----------------------

class Encoder(nn.Module):
    # 64 -> 32 -> 16 -> 8 -> 4
    def __init__(self, in_channels=NUM_CH, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32,  kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32,          64,  kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64,          128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128,         256, kernel_size=4, stride=2, padding=1)
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
        self.rel_head      = nn.Linear(LATENT_DIM, 5)
        self.scale_head    = nn.Linear(LATENT_DIM, 3)
        self.shape_r_head  = nn.Linear(LATENT_DIM, 2)
        self.shape_b_head  = nn.Linear(LATENT_DIM, 2)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


def load_scene_model():
    print(f"[resPCA256] Loading SceneModel from {CKPT_SCENEMODEL}")
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

def orthonormalize_pair(a, b):
    """Gram-Schmidt: take two nonzero vectors a, b -> unit g_x, g_y."""
    ax = a / np.linalg.norm(a)
    b_proj = b - np.dot(b, ax) * ax
    by_norm = np.linalg.norm(b_proj)
    if by_norm < 1e-8:
        raise RuntimeError("b is almost parallel to a in orthonormalize_pair.")
    ay = b_proj / by_norm
    return ax, ay

# -----------------------
# Main
# -----------------------

def main():
    print(f"[resPCA256] Using device: {DEVICE}")
    model = load_scene_model()

    print(f"[resPCA256] Generating dataset: N={N_SAMPLES}")
    ds = GeomEdges64Dataset(N_SAMPLES)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_z_list = []
    all_rel_list = []

    with torch.no_grad():
        for imgs, rel, scale, s_r, s_b in dl:
            imgs = imgs.to(DEVICE)
            z = model.encode(imgs)
            all_z_list.append(z.cpu())
            all_rel_list.append(rel)

    all_z = torch.cat(all_z_list, dim=0).numpy()   # (N, D)
    all_rel = torch.cat(all_rel_list, dim=0).numpy()
    N, D = all_z.shape
    print(f"[resPCA256] Latent shape: {all_z.shape}")

    # --- relation means & generators ---
    idx_left   = np.where(all_rel == REL_LEFT)[0]
    idx_right  = np.where(all_rel == REL_RIGHT)[0]
    idx_above  = np.where(all_rel == REL_ABOVE)[0]
    idx_below  = np.where(all_rel == REL_BELOW)[0]

    mu_left  = all_z[idx_left].mean(axis=0)
    mu_right = all_z[idx_right].mean(axis=0)
    mu_above = all_z[idx_above].mean(axis=0)
    mu_below = all_z[idx_below].mean(axis=0)
    mu0      = all_z.mean(axis=0)

    gx_raw = mu_right - mu_left
    gy_raw = mu_below - mu_above
    gx, gy = orthonormalize_pair(gx_raw, gy_raw)

    print(f"[resPCA256] ||mu_right - mu_left|| = {np.linalg.norm(gx_raw):.4f}")
    print(f"[resPCA256] ||mu_below - mu_above|| = {np.linalg.norm(gy_raw):.4f}")
    angle = math.degrees(math.acos(np.clip(np.dot(gx, gy), -1.0, 1.0)))
    print(f"[resPCA256] angle(g_x, g_y) = {angle:.2f}°")

    # --- residual PCA: subtract plane and run PCA on r_perp ---
    z_center = all_z - mu0[None, :]
    tx = z_center @ gx
    ty = z_center @ gy
    plane = np.outer(tx, gx) + np.outer(ty, gy)
    R = z_center - plane   # residuals, (N, D)

    # sanity stats
    r_norm = np.linalg.norm(R, axis=1)
    rel_norm = r_norm / np.linalg.norm(z_center, axis=1)
    print("[resPCA256] Residual norms ||r_perp|| stats:")
    print(f"   mean = {r_norm.mean():.4f}, std = {r_norm.std():.4f}, "
          f"min = {r_norm.min():.4f}, max = {r_norm.max():.4f}")
    print("[resPCA256] Relative residual ||r_perp||/||z-mu0|| stats:")
    print(f"   mean = {rel_norm.mean():.4f}, std = {rel_norm.std():.4f}, "
          f"min = {rel_norm.min():.4f}, max = {rel_norm.max():.4f}")

    # PCA on residuals
    max_components = 32
    n_comp = min(max_components, D, N)
    pca = PCA(n_components=n_comp, svd_solver="full", random_state=0)
    pca.fit(R)
    evr = pca.explained_variance_ratio_

    print("\n[resPCA256] PCA on residual subspace:")
    for i, r in enumerate(evr, 1):
        print(f"  PC{i:02d}: {100*r:5.2f}% variance")
    print(f"  cumulative PC1..PC{n_comp}: {100*evr.cumsum()[-1]:.2f}%")

    eff_rank = int((evr > 0.01).sum())
    print(f"  effective rank (>=1% each): {eff_rank}")

    # ------------------------------
    # Commutators and projection into residual PCs
    # ------------------------------
    EPS = 0.4
    N_SEEDS = 400
    rng = np.random.default_rng(0)
    seed_idx = rng.choice(N, size=N_SEEDS, replace=False)
    print(f"\n[resPCA256] Using {N_SEEDS} seeds for commutator analysis, EPS={EPS}")

    gx_t = torch.from_numpy(gx).to(DEVICE).view(1, -1)  # (1, D)
    gy_t = torch.from_numpy(gy).to(DEVICE).view(1, -1)

    all_z_tensor = torch.from_numpy(all_z).to(DEVICE)
    seeds = all_z_tensor[seed_idx]  # (B, D)

    def flow(z_batch, dir_vec):
        """One generator step with encode-decode re-projection."""
        with torch.no_grad():
            z_step = z_batch + EPS * dir_vec
            imgs = model.decode(z_step)
            z_next = model.encode(imgs)
        return z_next

    with torch.no_grad():
        # z_xy = Fy(Fx(z0)), z_yx = Fx(Fy(z0))
        z0 = seeds
        z_fx = flow(z0, gx_t)
        z_fy = flow(z0, gy_t)
        z_xy = flow(z_fx, gy_t)
        z_yx = flow(z_fy, gx_t)

    c = (z_xy - z_yx).cpu().numpy() / (EPS ** 2)   # (B, D)

    # Decompose commutators into plane + residual
    c_tx = c @ gx
    c_ty = c @ gy
    c_plane = np.outer(c_tx, gx) + np.outer(c_ty, gy)
    c_perp = c - c_plane

    c_norm = np.linalg.norm(c, axis=1)
    c_plane_norm = np.linalg.norm(c_plane, axis=1)
    c_perp_norm = np.linalg.norm(c_perp, axis=1)

    print("\n[resPCA256] Global commutator stats (for reference):")
    print(f"  ||c|| mean={c_norm.mean():.4f}, std={c_norm.std():.4f}, "
          f"min={c_norm.min():.4f}, max={c_norm.max():.4f}")
    print(f"  ||c_plane|| mean={c_plane_norm.mean():.4f}, std={c_plane_norm.std():.4f}")
    print(f"  ||c_perp|| mean={c_perp_norm.mean():.4f}, std={c_perp_norm.std():.4f}")
    frac_perp = c_perp_norm / c_norm
    print(f"  frac_perp mean={frac_perp.mean():.4f}, std={frac_perp.std():.4f}, "
          f"min={frac_perp.min():.4f}, max={frac_perp.max():.4f}")

    # Project c_perp onto residual PCs
    comps = pca.components_           # (K, D)
    proj = c_perp @ comps.T          # (B, K)
    energy_pc = (proj ** 2).sum(axis=0)   # energy in each PC
    total_energy = (c_perp ** 2).sum()

    frac_pc = energy_pc / total_energy
    cum_frac_pc = np.cumsum(frac_pc)

    print("\n[resPCA256] Fraction of commutator residual energy per PC:")
    for i, f in enumerate(frac_pc, 1):
        print(f"  PC{i:02d}: {100*f:5.2f}%   (cumulative {100*cum_frac_pc[i-1]:5.2f}%)")

    for K in [1, 2, 3, 5, 10, 20]:
        K = min(K, n_comp)
        fK = cum_frac_pc[K-1]
        print(f"[resPCA256] Cumulative commutator energy in top {K} residual PCs: "
              f"{100*fK:5.2f}%")

    # Per-seed fraction of norm captured by top-K residual PCs
    for K in [1, 2, 3, 5, 10]:
        K = min(K, n_comp)
        proj_K = proj[:, :K]
        norms_K = np.linalg.norm(proj_K, axis=1)
        frac_K = norms_K / c_perp_norm
        print(f"[resPCA256] Per-seed ||P_top{K} c_perp|| / ||c_perp||: "
              f"mean={frac_K.mean():.4f}, std={frac_K.std():.4f}, "
              f"min={frac_K.min():.4f}, max={frac_K.max():.4f}")

    # ------------------------------
    # Plot energy per PC + cumulative
    # ------------------------------
    x = np.arange(1, n_comp + 1)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=120)

    ax[0].bar(x, frac_pc)
    ax[0].set_title("Comm. residual energy per PC (latent256)")
    ax[0].set_xlabel("Residual PC index")
    ax[0].set_ylabel("fraction of total energy")

    ax[1].plot(x, cum_frac_pc, marker="o")
    ax[1].set_ylim(0.0, 1.01)
    ax[1].set_title("Cumulative comm. residual energy")
    ax[1].set_xlabel("Residual PC index")
    ax[1].set_ylabel("cumulative fraction")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "commutator_residual_pcs_latent256.png")
    plt.savefig(out_path)
    print(f"[resPCA256] Saved commutator residual PC plot -> {out_path}")

    print("\n[resPCA256] Done.")


if __name__ == "__main__":
    main()
