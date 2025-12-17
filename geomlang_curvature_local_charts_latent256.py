#!/usr/bin/env python3
# geomlang_curvature_local_charts_latent256.py
#
# Local curvature charts around representative points in (t_x, t_y) generator
# space for the 64x64 edges+relscale model with LATENT_DIM = 256.
#
#  - First we compute global generator coords (t_x, t_y) using g_x, g_y as
#    in the other Lie scripts.
#  - Then we project residuals r_perp into the dominant residual PCs and
#    take h1, h2 as the top two curvature directions.
#  - For several seed locations (center overlap, far left, far above,
#    diagonal left+above), we build a small grid in (Δt_x, Δt_y) around the
#    seed and, for each cell, average u1, u2, ||u|| over points whose
#    (t_x, t_y) fall inside that cell.
#  - We save one figure per seed with heatmaps of u1, u2, ||u||.
#
# Run from your project root like:
#   python bbit_geomlang/geomlang_curvature_local_charts_latent256.py

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
# Model – same 64x64 edges architecture with LATENT_DIM=256
# -----------------------

class Encoder(nn.Module):
    # 64 -> 32 -> 16 -> 8 -> 4
    def __init__(self, in_channels=NUM_CH, latent_dim=LATENT_DIM):
        super().__init__()
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
        # Heads are unused here but kept for checkpoint compatibility
        self.rel_head      = nn.Linear(LATENT_DIM, 5)
        self.scale_head    = nn.Linear(LATENT_DIM, 3)
        self.shape_r_head  = nn.Linear(LATENT_DIM, 2)
        self.shape_b_head  = nn.Linear(LATENT_DIM, 2)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


def load_scene_model():
    print(f"[curvLocal256] Loading SceneModel from {CKPT_SCENEMODEL}")
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
# Helper: build local curvature grid
# -----------------------

def build_local_grid(tx, ty, u1, u2, center_tx, center_ty,
                     dx_min=-2.0, dx_max=2.0, dy_min=-2.0, dy_max=2.0,
                     n_steps=21, min_points=5):
    """
    Given arrays tx, ty, u1, u2 (shape [N]), and a center (center_tx, center_ty),
    build a (Δt_x, Δt_y) grid and average u1,u2,||u|| inside each cell.

    Returns:
      dx_vals, dy_vals, U1, U2, UMAG, COUNT
      where arrays have shape [ny, nx] with (row, col) = (dy, dx).
    """
    dx_vals = np.linspace(dx_min, dx_max, n_steps)
    dy_vals = np.linspace(dy_min, dy_max, n_steps)

    step_x = dx_vals[1] - dx_vals[0]
    step_y = dy_vals[1] - dy_vals[0]
    half_x = 0.6 * step_x
    half_y = 0.6 * step_y

    nx = len(dx_vals)
    ny = len(dy_vals)

    U1   = np.full((ny, nx), np.nan, dtype=np.float32)
    U2   = np.full((ny, nx), np.nan, dtype=np.float32)
    UMAG = np.full((ny, nx), np.nan, dtype=np.float32)
    COUNT = np.zeros((ny, nx), dtype=np.int32)

    for iy, dty in enumerate(dy_vals):
        ty_c = center_ty + dty
        for ix, dtx in enumerate(dx_vals):
            tx_c = center_tx + dtx
            mask = (
                (np.abs(tx - tx_c) <= half_x) &
                (np.abs(ty - ty_c) <= half_y)
            )
            idx = np.where(mask)[0]
            n = idx.size
            COUNT[iy, ix] = n
            if n >= min_points:
                uu1 = u1[idx]
                uu2 = u2[idx]
                U1[iy, ix]   = uu1.mean()
                U2[iy, ix]   = uu2.mean()
                UMAG[iy, ix] = np.sqrt(uu1**2 + uu2**2).mean()
    return dx_vals, dy_vals, U1, U2, UMAG, COUNT


def plot_local_curvature(dx_vals, dy_vals, U1, U2, UMAG, seed_name, fname_out):
    """
    Make a 1×3 figure: heatmaps for u1, u2, ||u|| vs (Δt_x, Δt_y).
    """
    extent = [dx_vals[0], dx_vals[-1], dy_vals[0], dy_vals[-1]]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=120)
    fig.suptitle(f"Local curvature around {seed_name}\n"
                 "axes: Δt_x (LR), Δt_y (AB)")

    # u1
    ax = axes[0]
    vmax1 = np.nanmax(np.abs(U1))
    im1 = ax.imshow(U1, origin="lower", extent=extent, aspect="equal",
                    cmap="coolwarm", vmin=-vmax1, vmax=vmax1)
    ax.set_title("u1 (projection on h1)")
    ax.set_xlabel("Δt_x")
    ax.set_ylabel("Δt_y")
    fig.colorbar(im1, ax=ax, shrink=0.8)

    # u2
    ax = axes[1]
    vmax2 = np.nanmax(np.abs(U2))
    im2 = ax.imshow(U2, origin="lower", extent=extent, aspect="equal",
                    cmap="coolwarm", vmin=-vmax2, vmax=vmax2)
    ax.set_title("u2 (projection on h2)")
    ax.set_xlabel("Δt_x")
    ax.set_ylabel("Δt_y")
    fig.colorbar(im2, ax=ax, shrink=0.8)

    # |u|
    ax = axes[2]
    im3 = ax.imshow(UMAG, origin="lower", extent=extent, aspect="equal",
                    cmap="viridis")
    ax.set_title("||u|| = sqrt(u1^2 + u2^2)")
    ax.set_xlabel("Δt_x")
    ax.set_ylabel("Δt_y")
    fig.colorbar(im3, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(fname_out)
    plt.close(fig)
    print(f"[curvLocal256] Saved local curvature chart -> {fname_out}")


# -----------------------
# Main
# -----------------------

def main():
    print(f"[curvLocal256] Using device: {DEVICE}")
    model = load_scene_model()

    print(f"[curvLocal256] Generating dataset: N={N_SAMPLES}")
    ds = GeomEdges64Dataset(N_SAMPLES)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_z   = []
    all_rel = []

    with torch.no_grad():
        for imgs, rel, scale, s_r, s_b in dl:
            imgs = imgs.to(DEVICE)
            z    = model.encode(imgs)
            all_z.append(z.cpu())
            all_rel.append(rel.cpu())

    z   = torch.cat(all_z,   dim=0).numpy()  # (N, 256)
    rel = torch.cat(all_rel, dim=0).numpy()  # (N,)

    N, D = z.shape
    print(f"[curvLocal256] Latent shape: {z.shape}")
    for r in range(5):
        cnt = int((rel == r).sum())
        print(f"[curvLocal256] relation {r}={REL_NAMES[r]:>7}, count={cnt}")

    # --- Global generator plane
    mu0 = z.mean(axis=0)

    def mean_rel(r_id):
        mask = (rel == r_id)
        return z[mask].mean(axis=0)

    mu_left  = mean_rel(REL_LEFT)
    mu_right = mean_rel(REL_RIGHT)
    mu_above = mean_rel(REL_ABOVE)
    mu_below = mean_rel(REL_BELOW)

    g_x = mu_right - mu_left
    g_y = mu_below - mu_above

    gx_norm2 = np.dot(g_x, g_x)
    gy_norm2 = np.dot(g_y, g_y)
    angle = math.degrees(
        math.acos(np.clip(np.dot(g_x, g_y) / math.sqrt(gx_norm2 * gy_norm2), -1.0, 1.0))
    )
    print(f"[curvLocal256] ||mu_right - mu_left|| = {math.sqrt(gx_norm2):.4f}")
    print(f"[curvLocal256] ||mu_below - mu_above|| = {math.sqrt(gy_norm2):.4f}")
    print(f"[curvLocal256] angle(g_x, g_y) = {angle:.2f}°")

    z_centered = z - mu0[None, :]

    t_x = (z_centered @ g_x) / gx_norm2
    t_y = (z_centered @ g_y) / gy_norm2

    plane_part = np.outer(t_x, g_x) + np.outer(t_y, g_y)
    r_perp = z_centered - plane_part

    # --- Residual PCA, curvature basis h1,h2
    print("[curvLocal256] PCA on residual subspace...")
    pca = PCA(n_components=min(32, D), svd_solver="full", random_state=0)
    pca.fit(r_perp)
    evr = pca.explained_variance_ratio_
    for i, r in enumerate(evr, 1):
        print(f"  PC{i:02d}: {100*r:5.2f}% variance")
    print(f"  cumulative PC1..PC{len(evr)}: {100*evr.cumsum()[-1]:.2f}%")

    h1 = pca.components_[0]  # dominant residual axis
    h2 = pca.components_[1]  # second residual axis

    u1 = r_perp @ h1
    u2 = r_perp @ h2

    # --- Choose seeds in (t_x, t_y) plane

    def pick_seed(mask, description, objective):
        idx = np.where(mask)[0]
        if idx.size == 0:
            raise RuntimeError(f"No candidates for seed '{description}'")
        scores = objective(idx)
        best_local = int(idx[np.argmin(scores)])
        print(f"[curvLocal256] Seed '{description}': idx={best_local}, "
              f"rel={REL_NAMES[rel[best_local]]}, "
              f"t_x={t_x[best_local]:.2f}, t_y={t_y[best_local]:.2f}")
        return best_local

    # 1) Overlap center: closest to (0,0) among overlap points
    seed_overlap = pick_seed(
        (rel == REL_OVERLAP),
        "overlap_center",
        lambda idx: t_x[idx]**2 + t_y[idx]**2
    )

    # 2) Far left: most negative t_x among left_of, lightly penalize big |t_y|
    seed_left = pick_seed(
        (rel == REL_LEFT),
        "far_left",
        lambda idx: -t_x[idx] + 0.1 * np.abs(t_y[idx])
    )

    # 3) Far above: most negative t_y among above, lightly penalize big |t_x|
    seed_above = pick_seed(
        (rel == REL_ABOVE),
        "far_above",
        lambda idx: -t_y[idx] + 0.1 * np.abs(t_x[idx])
    )

    # 4) Diagonal left+above: points with t_x<0,t_y<0, maximize radius
    mask_diag = (t_x < 0) & (t_y < 0)
    seed_diag = pick_seed(
        mask_diag,
        "diag_left_above",
        lambda idx: -(t_x[idx]**2 + t_y[idx]**2)
    )

    seeds = [
        ("overlap_center", seed_overlap),
        ("far_left",       seed_left),
        ("far_above",      seed_above),
        ("diag_left_above", seed_diag),
    ]

    # --- Build and plot local grids
    for name, idx_seed in seeds:
        tx0 = float(t_x[idx_seed])
        ty0 = float(t_y[idx_seed])
        print(f"[curvLocal256] Building local grid around {name} at "
              f"(t_x={tx0:.2f}, t_y={ty0:.2f})")

        dx_vals, dy_vals, U1, U2, UMAG, COUNT = build_local_grid(
            t_x, t_y, u1, u2,
            center_tx=tx0,
            center_ty=ty0,
            dx_min=-2.0, dx_max=2.0,
            dy_min=-2.0, dy_max=2.0,
            n_steps=21,
            min_points=5
        )

        print(f"[curvLocal256] Non-empty cells for {name}: {(COUNT > 0).sum()} "
              f"/ {COUNT.size}")

        fname_out = os.path.join(
            OUT_DIR,
            f"curvature_local_{name}_latent256.png"
        )
        plot_local_curvature(dx_vals, dy_vals, U1, U2, UMAG, name, fname_out)

    print("[curvLocal256] Done.")


if __name__ == "__main__":
    main()
