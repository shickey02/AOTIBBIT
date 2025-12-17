#!/usr/bin/env python3
# geomlang_lie_commutator_stats_latent256.py
#
# Quantitative commutator / curvature stats for the 64x64 edges+relscale model
# with LATENT_DIM = 256.
#
# - Loads SceneModelEdges256 from:
#       outputs_edges_relscale256/scene_model_edges_relscale256.pt
# - Generates a fresh static dataset of scenes
# - Estimates LR and AB generators g_x, g_y (from mean difference vectors)
# - For many base points z0, computes:
#       z_xy = (z0 + eps g_x) + eps g_y
#       z_yx = (z0 + eps g_y) + eps g_x
#   and measures:
#       ||z_xy - z_yx||   (latent commutator norm)
#       MSE(xy, yx)       (image space commutator effect)
# - Prints summary stats and saves histograms:
#       outputs_edges_relscale256/lie_commutator_stats_latent256.png
#
# Run:
#   python bbit_geomlang/geomlang_lie_commutator_stats_latent256.py

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
IMG_SIZE   = 64
NUM_CH     = 3      # red, blue, edges
LATENT_DIM = 256

N_SAMPLES  = 6000
BATCH_SIZE = 256

EPS_LIST   = [0.15, 0.30, 0.60, 1.00]

OUT_DIR         = "outputs_edges_relscale256"
CKPT_SCENEMODEL = os.path.join(OUT_DIR, "scene_model_edges_relscale256.pt")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REL_LEFT, REL_RIGHT, REL_ABOVE, REL_BELOW, REL_OVERLAP = range(5)
REL_NAMES = ["left_of", "right_of", "above", "below", "overlap"]

# How many base points for commutator stats
N_BASE_POINTS = 1000
# Small step size epsilon along each generator
EPS = 0.60

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
    """
    Edge channel from union of red/blue masks, using a cheap
    binary erosion trick (no scipy).
    """
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
    if s <= 7:
        return 0
    elif s <= 11:
        return 1
    else:
        return 2


class GeomEdgesDataset(Dataset):
    """
    Static scenes, 64x64, with:
      - red blob in channel 0
      - blue blob in channel 1
      - edges in channel 2
    """

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

        base = np.random.randint(5, 13)       # 5..12
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
# Model – same as other 64x64/latent256 scripts
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


class SceneModelEdges256(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

        # heads (kept for checkpoint compatibility; not used here)
        self.rel_head      = nn.Linear(LATENT_DIM, 5)
        self.scale_head    = nn.Linear(LATENT_DIM, 3)
        self.shape_r_head  = nn.Linear(LATENT_DIM, 2)
        self.shape_b_head  = nn.Linear(LATENT_DIM, 2)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


def load_scene_model():
    print(f"[comm256-stats] Loading SceneModel from {CKPT_SCENEMODEL}")
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
# Utility
# -----------------------

def mse_np(a, b):
    return float(np.mean((a - b) ** 2))

def chw_to_rgb(img_chw):
    """
    img_chw: [3,H,W], with:
      ch0 = red,
      ch1 = blue,
      ch2 = edges
    R <- red, G <- edges, B <- blue
    """
    c, h, w = img_chw.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[..., 0] = img_chw[0]
    rgb[..., 1] = img_chw[2]
    rgb[..., 2] = img_chw[1]
    return rgb

# -----------------------
# Main
# -----------------------

def main():
    print(f"[comm256-stats] Using device: {DEVICE}")
    model = load_scene_model()

    # Generate dataset
    print(f"[comm256-stats] Generating dataset: N={N_SAMPLES}")
    ds = GeomEdgesDataset(N_SAMPLES)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_imgs = []
    all_rel  = []
    all_z    = []

    with torch.no_grad():
        for imgs, rel, scale, s_r, s_b in dl:
            imgs = imgs.to(DEVICE)
            rel  = rel.to(DEVICE)

            z = model.encode(imgs)

            all_imgs.append(imgs.cpu())
            all_rel.append(rel.cpu())
            all_z.append(z.cpu())

    all_imgs = torch.cat(all_imgs, dim=0)   # [N,3,64,64]
    all_rel  = torch.cat(all_rel,  dim=0)   # [N]
    all_z    = torch.cat(all_z,   dim=0)    # [N,D]

    rel_np = all_rel.numpy()
    z_np   = all_z.numpy()
    N, D   = z_np.shape

    # -----------------------
    # Build LR and AB difference clouds
    # -----------------------
    idx_left  = np.where(rel_np == REL_LEFT)[0]
    idx_right = np.where(rel_np == REL_RIGHT)[0]
    idx_above = np.where(rel_np == REL_ABOVE)[0]
    idx_below = np.where(rel_np == REL_BELOW)[0]

    rng = np.random.default_rng(0)

    def make_pairs(idxs_a, idxs_b, max_pairs=None):
        n = min(len(idxs_a), len(idxs_b))
        if max_pairs is not None:
            n = min(n, max_pairs)
        if n == 0:
            return np.array([], dtype=int), np.array([], dtype=int)
        idxs_a = rng.permutation(idxs_a)[:n]
        idxs_b = rng.permutation(idxs_b)[:n]
        return idxs_a, idxs_b

    left_idx, right_idx   = make_pairs(idx_left, idx_right)
    above_idx, below_idx  = make_pairs(idx_above, idx_below)

    z_left   = z_np[left_idx]
    z_right  = z_np[right_idx]
    z_above  = z_np[above_idx]
    z_below  = z_np[below_idx]

    v_lr = z_right - z_left
    v_ab = z_below - z_above

    # Mean difference vectors and their norms
    v_lr_mean = v_lr.mean(axis=0)
    v_ab_mean = v_ab.mean(axis=0)
    norm_lr   = np.linalg.norm(v_lr_mean)
    norm_ab   = np.linalg.norm(v_ab_mean)

    print(f"[comm256-stats] ||v_LR_mean|| = {norm_lr:.4f}")
    print(f"[comm256-stats] ||v_AB_mean|| = {norm_ab:.4f}")

    # Generators: normalized mean vectors
    g_x = v_lr_mean / (norm_lr + 1e-8)
    g_y = v_ab_mean / (norm_ab + 1e-8)

    # Convert to torch
    g_x_t = torch.from_numpy(g_x).to(DEVICE).float()
    g_y_t = torch.from_numpy(g_y).to(DEVICE).float()

    # -----------------------
    # Commutator stats over many base points
    # -----------------------
    n_bases = min(N_BASE_POINTS, N)
    base_indices = rng.choice(N, size=n_bases, replace=False)
    print(f"[comm256-stats] Sampling {n_bases} base points for commutator stats")

    latent_comm_norms = []
    image_comm_mse    = []

    with torch.no_grad():
        for i, idx in enumerate(base_indices):
            z0 = all_z[idx:idx+1].to(DEVICE)           # [1,D]
            img0 = all_imgs[idx:idx+1].to(DEVICE)      # [1,3,64,64]

            # x then y
            z_x  = z0 + EPS * g_x_t.unsqueeze(0)
            z_xy = z_x + EPS * g_y_t.unsqueeze(0)

            # y then x
            z_y  = z0 + EPS * g_y_t.unsqueeze(0)
            z_yx = z_y + EPS * g_x_t.unsqueeze(0)

            # latent commutator vector
            comm_vec = (z_xy - z_yx).cpu().numpy().reshape(-1)
            latent_comm_norms.append(np.linalg.norm(comm_vec))

            # decode to images
            img_xy = model.decode(z_xy).cpu().numpy()[0]
            img_yx = model.decode(z_yx).cpu().numpy()[0]

            # image-space MSE (over all 3 channels)
            image_comm_mse.append(mse_np(img_xy, img_yx))

    latent_comm_norms = np.array(latent_comm_norms)
    image_comm_mse    = np.array(image_comm_mse)

    # -----------------------
    # Summary stats
    # -----------------------
    def print_stats(name, arr):
        print(f"\n[comm256-stats] {name}:")
        print(f"  mean={arr.mean():.6f}, std={arr.std():.6f}, "
              f"min={arr.min():.6f}, max={arr.max():.6f}")

    print_stats("latent commutator norm ||z_xy - z_yx||", latent_comm_norms)
    print_stats("image commutator MSE", image_comm_mse)

    # Relative to step size (epsilon^2 ~ area of loop)
    step_len = EPS  # because g_x, g_y are unit norm
    area_scale = step_len ** 2
    print(f"\n[comm256-stats] step length = {step_len:.3f}, epsilon^2 = {area_scale:.4f}")
    print(f"[comm256-stats] mean latent commutator / epsilon^2 = "
          f"{(latent_comm_norms.mean() / (area_scale + 1e-8)):.6f}")

    # -----------------------
    # Plots: histograms
    # -----------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=120)

    axes[0].hist(latent_comm_norms, bins=40, alpha=0.8)
    axes[0].set_title("Latent commutator norms\n||z_xy - z_yx||")
    axes[0].set_xlabel("norm")
    axes[0].set_ylabel("count")

    axes[1].hist(image_comm_mse, bins=40, alpha=0.8)
    axes[1].set_title("Image commutator MSE")
    axes[1].set_xlabel("MSE")
    axes[1].set_ylabel("count")

    plt.tight_layout()
    out_hist = os.path.join(OUT_DIR, "lie_commutator_stats_latent256.png")
    plt.savefig(out_hist)
    print(f"[comm256-stats] Saved histograms -> {out_hist}")

    print(f"[comm-eps] Using device: {DEVICE}, LATENT_DIM={LATENT_DIM}")
    model = load_scene_model()

    # Generate dataset + encode everything once
    print(f"[comm-eps] Generating dataset: N={N_SAMPLES}")
    ds = GeomEdgesDataset(N_SAMPLES)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_imgs = []
    all_rel  = []
    all_z    = []

    with torch.no_grad():
        for imgs, rel, scale, s_r, s_b in dl:
            imgs = imgs.to(DEVICE)
            rel  = rel.to(DEVICE)
            z    = model.encode(imgs)

            all_imgs.append(imgs.cpu())
            all_rel.append(rel.cpu())
            all_z.append(z.cpu())

    all_imgs = torch.cat(all_imgs, dim=0)   # [N,3,64,64]
    all_rel  = torch.cat(all_rel,  dim=0)   # [N]
    all_z    = torch.cat(all_z,   dim=0)    # [N,D]

    rel_np = all_rel.numpy()
    z_np   = all_z.numpy()

    # Index sets
    idx_left  = np.where(rel_np == REL_LEFT)[0]
    idx_right = np.where(rel_np == REL_RIGHT)[0]
    idx_above = np.where(rel_np == REL_ABOVE)[0]
    idx_below = np.where(rel_np == REL_BELOW)[0]

    print("[comm-eps] relation counts:")
    for r in range(5):
        n = int((rel_np == r).sum())
        print(f"   {REL_NAMES[r]:>7}: {n}")

    rng = np.random.default_rng(0)

    def make_pairs(idxs_a, idxs_b, max_pairs=None):
        n = min(len(idxs_a), len(idxs_b))
        if max_pairs is not None:
            n = min(n, max_pairs)
        if n == 0:
            return np.array([], dtype=int), np.array([], dtype=int)
        idxs_a = rng.permutation(idxs_a)[:n]
        idxs_b = rng.permutation(idxs_b)[:n]
        return idxs_a, idxs_b

    left_idx, right_idx   = make_pairs(idx_left,  idx_right)
    above_idx, below_idx  = make_pairs(idx_above, idx_below)

    z_left   = z_np[left_idx]
    z_right  = z_np[right_idx]
    z_above  = z_np[above_idx]
    z_below  = z_np[below_idx]

    v_lr = z_right - z_left   # left→right
    v_ab = z_below - z_above  # above→below

    v_lr_mean = v_lr.mean(axis=0)
    v_ab_mean = v_ab.mean(axis=0)

    norm_lr = float(np.linalg.norm(v_lr_mean))
    norm_ab = float(np.linalg.norm(v_ab_mean))
    print(f"\n[comm-eps] ||mean_left→right|| = {norm_lr:.4f}")
    print(f"[comm-eps] ||mean_above→below|| = {norm_ab:.4f}")

    g_x = torch.from_numpy(v_lr_mean / (norm_lr + 1e-8)).float().to(DEVICE)
    g_y = torch.from_numpy(v_ab_mean / (norm_ab + 1e-8)).float().to(DEVICE)

    # Sample base points for commutator stats
    n_base = 1000
    base_idx = rng.choice(z_np.shape[0], size=min(n_base, z_np.shape[0]), replace=False)
    print(f"\n[comm-eps] Sampling {len(base_idx)} base points for commutator ε-sweep")

    results = []

    for eps in EPS_LIST:
        eps2 = eps * eps
        print(f"\n[comm-eps] ==== EPS = {eps:.3f} (eps^2 = {eps2:.4f}) ====")
        latent_norms = []
        img_mses     = []

        with torch.no_grad():
            for start in range(0, len(base_idx), BATCH_SIZE):
                end = min(len(base_idx), start + BATCH_SIZE)
                idx_batch = base_idx[start:end]
                z0 = all_z[idx_batch].to(DEVICE)       # [B,D]

                # z_x = z0 + eps * g_x, etc.
                z_x  = z0 + eps * g_x
                z_y  = z0 + eps * g_y
                z_xy = z_x + eps * g_y
                z_yx = z_y + eps * g_x

                # latent commutator
                dz = z_xy - z_yx
                latent_norms.append(dz.norm(dim=1).cpu().numpy())

                # image commutator
                img_xy = model.decode(z_xy)
                img_yx = model.decode(z_yx)
                mse = ((img_xy - img_yx) ** 2).mean(dim=[1, 2, 3])  # [B]
                img_mses.append(mse.cpu().numpy())

        latent_norms = np.concatenate(latent_norms, axis=0)
        img_mses     = np.concatenate(img_mses,     axis=0)

        lat_mean = float(latent_norms.mean())
        lat_std  = float(latent_norms.std())
        lat_min  = float(latent_norms.min())
        lat_max  = float(latent_norms.max())

        mse_mean = float(img_mses.mean())
        mse_std  = float(img_mses.std())
        mse_min  = float(img_mses.min())
        mse_max  = float(img_mses.max())

        scaled = lat_mean / (eps2 + 1e-12)

        print(f"[comm-eps] latent ||z_xy - z_yx||: "
              f"mean={lat_mean:.6e}, std={lat_std:.6e}, "
              f"min={lat_min:.6e}, max={lat_max:.6e}")
        print(f"[comm-eps] image  MSE: "
              f"mean={mse_mean:.6e}, std={mse_std:.6e}, "
              f"min={mse_min:.6e}, max={mse_max:.6e}")
        print(f"[comm-eps] mean latent / eps^2 = {scaled:.6e}")

        results.append((eps, eps2, lat_mean, scaled, mse_mean))

    # Simple plot: mean latent commutator vs eps^2 (log-log-ish)
    eps_arr   = np.array([r[0] for r in results])
    eps2_arr  = np.array([r[1] for r in results])
    lat_arr   = np.array([r[2] for r in results])

    plt.figure(figsize=(5,4), dpi=120)
    plt.plot(eps2_arr, lat_arr, marker="o")
    plt.xlabel(r"$\epsilon^2$")
    plt.ylabel(r"mean $\|[g_x,g_y]\|$ (latent)")
    plt.title("Commutator norm vs $\\,\\epsilon^2$ (latent256, 64x64)")
    plt.tight_layout()
    out_plot = os.path.join(OUT_DIR, "lie_commutator_eps_sweep_latent256.png")
    plt.savefig(out_plot)
    print(f"\n[comm-eps] Saved ε-sweep plot -> {out_plot}")

    print("\n[comm-eps] Summary:")
    for eps, eps2, lat_mean, scaled, mse_mean in results:
        print(f"  eps={eps:4.2f}, eps^2={eps2:5.3f}, "
              f"mean||comm||={lat_mean:.3e}, "
              f"mean||comm||/eps^2={scaled:.3e}, "
              f"mean MSE={mse_mean:.3e}")


if __name__ == "__main__":
    main()
