#!/usr/bin/env python3
# geomlang_relation_interpolate_latent256.py
#
# Visualize latent geometry for 64x64 edges+relscale model with LATENT_DIM=256.
# - Walk along g_x (left/right) and g_y (above/below) axes
# - 2D grid in (g_x, g_y) plane (GLOBAL ranges) + label heatmap
# - NEW:
#   * confidence heatmap (max softmax prob) to reveal boundaries
#   * option to center 2D plane on overlap seed vs dataset mean/median
#   * auto-run BOTH gx directions (+gx and -gx) so "left_of" isn't accidentally off-frame
#   * orthogonality report (Δt_x vs Δt_y for each walk)
#   * pairwise latent morphs

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

# -----------------------
# Config / paths
# -----------------------
IMG_SIZE    = 64
LATENT_DIM  = 256
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR = "outputs_edges_relscale256"
SCENE_MODEL_PATH  = os.path.join(OUT_DIR, "scene_model_edges_relscale256.pt")
DATA_LATENTS_PATH = os.path.join(OUT_DIR, "encoded_dataset_latents.npy")
DATA_LABELS_PATH  = os.path.join(OUT_DIR, "encoded_dataset_labels.npy")

REL_LEFT, REL_RIGHT, REL_ABOVE, REL_BELOW, REL_OVERLAP = range(5)
REL_NAMES = ["left_of", "right_of", "above", "below", "overlap"]

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------
# 2D GRID OPTIONS (EDIT ME)
# -----------------------

# Where to anchor the (g_x,g_y) plane:
#   "overlap_seed" : use an overlap example latent
#   "mean"         : center plane on dataset mean latent
#   "median"       : center plane on dataset median latent (robust)
GRID_CENTER_MODE = "overlap_seed"   # <- try "mean" or "median" if you want cleaner symmetry

# Heatmap resolution (labels/confidence only) – boundaries get sharper as this increases
GRID_LABEL_N = 96   # 64, 96, 128 are good; 96 is a sweet spot

# Decoded image atlas resolution – keep small (decoded atlas size explodes fast)
GRID_IMG_N = 11
SAVE_LARGE_IMAGE_ATLAS = False  # True makes decoded atlas = GRID_LABEL_N (very large file)

# Auto-run both gx directions so we don't "miss" left_of territory
RUN_BOTH_GX_DIRECTIONS = True

# -----------------------
# Model (same as training)
# -----------------------
NUM_CH = 3  # red, blue, edges


class Encoder(nn.Module):
    def __init__(self, in_channels=NUM_CH, latent_dim=LATENT_DIM):
        super().__init__()
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
        self.deconv2 = nn.ConvTranspose2d(128, 64,  kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64,  32,  kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1)

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
        self.rel_head     = nn.Linear(LATENT_DIM, 5)
        self.scale_head   = nn.Linear(LATENT_DIM, 3)
        self.shape_r_head = nn.Linear(LATENT_DIM, 2)
        self.shape_b_head = nn.Linear(LATENT_DIM, 2)

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
# Helpers
# -----------------------

def load_dataset_latents():
    Z = np.load(DATA_LATENTS_PATH)
    y = np.load(DATA_LABELS_PATH)
    return Z, y


def compute_relation_axes(Z, y):
    """Returns unit vectors g_x, g_y."""
    Z_t = torch.from_numpy(Z).float().to(DEVICE)
    mu = {}
    for r in range(5):
        mu[r] = Z_t[y == r].mean(dim=0)

    g_left  = mu[REL_LEFT]
    g_right = mu[REL_RIGHT]
    g_above = mu[REL_ABOVE]
    g_below = mu[REL_BELOW]

    g_x = g_right - g_left
    g_y = g_below - g_above

    g_x = g_x / g_x.norm()
    g_y = g_y / g_y.norm()
    return g_x.cpu().numpy(), g_y.cpu().numpy()


def load_scene_model():
    ckpt = torch.load(SCENE_MODEL_PATH, map_location=DEVICE)
    if "model_state_dict" not in ckpt:
        raise RuntimeError(
            "[interp256] Checkpoint must contain 'model_state_dict' "
            "as produced by geomlang_edges_relscale_train64_latent256.py"
        )
    model = SceneModelEdges64_256().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("[interp256] Loaded SceneModelEdges64_256 from checkpoint.")
    return model


@torch.no_grad()
def decode_latents(model, Z_np):
    z = torch.from_numpy(Z_np).float().to(DEVICE)
    imgs = model.decode(z)
    return imgs.cpu()


@torch.no_grad()
def predict_rel_labels_and_conf(model, Z_np):
    """
    Returns:
        labels_np: (N,) int labels 0..4
        conf_np  : (N,) max softmax prob
    """
    z = torch.from_numpy(Z_np).float().to(DEVICE)
    logits = model.rel_head(z)
    probs = torch.softmax(logits, dim=1)
    conf, labels = probs.max(dim=1)
    return labels.cpu().numpy(), conf.cpu().numpy()


def save_strip(images, out_path, nrow=None):
    if nrow is None:
        nrow = len(images)
    grid = make_grid(images, nrow=nrow, padding=2)
    save_image(grid, out_path)
    print(f"[interp256] Saved grid -> {out_path}")


def axis_stats_and_range(Z, axis_vec, seed_idx, axis_name, k=3.0):
    axis = axis_vec / np.linalg.norm(axis_vec)
    all_t = Z @ axis
    mu = float(all_t.mean())
    sigma = float(all_t.std())
    t_min = mu - k * sigma
    t_max = mu + k * sigma
    t0 = float(all_t[seed_idx])

    print(
        f"[interp256] Axis {axis_name}: mean t={mu:.4f}, std t={sigma:.4f}\n"
        f"[interp256]   seed idx={seed_idx}, t0={t0:.4f}\n"
        f"[interp256]   t range: [{t_min:.4f}, {t_max:.4f}]"
    )
    return t_min, t_max, t0


def make_1d_walk(model, Z, seed_idx, axis_vec, axis_name, g_x, g_y, n_steps=12, rel_names=REL_NAMES):
    z0 = Z[seed_idx]
    t_min, t_max, t0 = axis_stats_and_range(Z, axis_vec, seed_idx, axis_name, k=3.0)

    axis = axis_vec / np.linalg.norm(axis_vec)
    ts = np.linspace(t_min, t_max, n_steps)
    deltas = ts - t0
    z_path = z0[None, :] + deltas[:, None] * axis[None, :]
    z_path_np = z_path.astype(np.float32)

    imgs = decode_latents(model, z_path_np)

    labels, conf = predict_rel_labels_and_conf(model, z_path_np)
    label_strs = [rel_names[i] for i in labels]
    print(f"[interp256]   labels along {axis_name}: " + " | ".join(label_strs))
    print(f"[interp256]   confidences: " + " ".join(f"{p:.2f}" for p in conf))

    gx = g_x / np.linalg.norm(g_x)
    gy = g_y / np.linalg.norm(g_y)
    t_x = z_path_np @ gx
    t_y = z_path_np @ gy
    dx = t_x - t_x.mean()
    dy = t_y - t_y.mean()
    print(
        f"[interp256]   orthogonality ({axis_name} walk): "
        f"Δt_x std={dx.std():.4f}, Δt_y std={dy.std():.4f}"
    )
    return imgs


def make_pairwise_morph(model, z0, z1, n_steps=12):
    alphas = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)
    z_path = (1.0 - alphas)[:, None] * z0[None, :] + alphas[:, None] * z1[None, :]
    imgs = decode_latents(model, z_path)
    return imgs


def choose_grid_center(Z, y, idx_overlap):
    if GRID_CENTER_MODE == "overlap_seed":
        print("[interp256] 2D grid center: overlap seed")
        return Z[idx_overlap]
    if GRID_CENTER_MODE == "mean":
        print("[interp256] 2D grid center: dataset mean")
        return Z.mean(axis=0)
    if GRID_CENTER_MODE == "median":
        print("[interp256] 2D grid center: dataset median")
        return np.median(Z, axis=0)
    raise ValueError(f"Unknown GRID_CENTER_MODE={GRID_CENTER_MODE}")


def make_global_grid(Z, gx, gy, z0, labels_N, tx_min, tx_max, ty_min, ty_max):
    # anchor point projections for this plane center
    t_x0 = float((z0 @ gx))
    t_y0 = float((z0 @ gy))

    tx_vals = np.linspace(tx_min, tx_max, labels_N)
    ty_vals = np.linspace(ty_min, ty_max, labels_N)

    z_grid = []
    for ty in ty_vals:
        for tx in tx_vals:
            z = z0 + (tx - t_x0) * gx + (ty - t_y0) * gy
            z_grid.append(z)
    z_grid = np.stack(z_grid, axis=0).astype(np.float32)
    return z_grid, tx_vals, ty_vals


def save_label_and_conf_heatmaps(labels_grid, conf_grid, out_prefix, title_suffix):
    # Labels (discrete)
    cmap = plt.get_cmap("tab10", 5)
    plt.figure(figsize=(6, 6))
    plt.imshow(labels_grid, origin="lower", cmap=cmap, vmin=-0.5, vmax=4.5)
    plt.colorbar(ticks=range(5), label="relation")
    plt.xticks([]); plt.yticks([])
    plt.title(f"Relation labels over (g_x, g_y) grid ({title_suffix})")
    out_labels = out_prefix + "_labels.png"
    plt.tight_layout()
    plt.savefig(out_labels, dpi=200)
    plt.close()
    print(f"[interp256] Saved label heatmap -> {out_labels}")

    # Confidence (continuous)
    plt.figure(figsize=(6, 6))
    plt.imshow(conf_grid, origin="lower")
    plt.colorbar(label="max softmax prob")
    plt.xticks([]); plt.yticks([])
    plt.title(f"Confidence over (g_x, g_y) grid ({title_suffix})")
    out_conf = out_prefix + "_conf.png"
    plt.tight_layout()
    plt.savefig(out_conf, dpi=200)
    plt.close()
    print(f"[interp256] Saved confidence heatmap -> {out_conf}")


# -----------------------
# Main
# -----------------------

def main():
    print(f"[interp256] Using device: {DEVICE}")

    Z, y = load_dataset_latents()
    print(f"[interp256] Z shape: {Z.shape}, y shape: {y.shape}")

    g_x, g_y = compute_relation_axes(Z, y)

    rng = np.random.default_rng(0)
    idx_left    = int(rng.choice(np.where(y == REL_LEFT)[0]))
    idx_right   = int(rng.choice(np.where(y == REL_RIGHT)[0]))
    idx_above   = int(rng.choice(np.where(y == REL_ABOVE)[0]))
    idx_below   = int(rng.choice(np.where(y == REL_BELOW)[0]))
    idx_overlap = int(rng.choice(np.where(y == REL_OVERLAP)[0]))

    print(f"[interp256] Seed for left_of: idx={idx_left}")
    print(f"[interp256] Seed for right_of: idx={idx_right}")
    print(f"[interp256] Seed for above: idx={idx_above}")
    print(f"[interp256] Seed for below: idx={idx_below}")
    print(f"[interp256] Seed for overlap: idx={idx_overlap}")

    model = load_scene_model()

    # ----- 1D walks along g_x / g_y for each relation -----
    def run_walks(gx_vec, gy_vec, tag):
        print(f"[interp256] === 1D WALKS ({tag}) ===")
        def run_walk(seed_idx, rel_name):
            imgs_x = make_1d_walk(model, Z, seed_idx, gx_vec, "g_x", gx_vec, gy_vec, n_steps=12)
            out_x = os.path.join(OUT_DIR, f"interp256_{rel_name}_along_gx_{tag}_idx{seed_idx}.png")
            save_strip(imgs_x, out_x)

            imgs_y = make_1d_walk(model, Z, seed_idx, gy_vec, "g_y", gx_vec, gy_vec, n_steps=12)
            out_y = os.path.join(OUT_DIR, f"interp256_{rel_name}_along_gy_{tag}_idx{seed_idx}.png")
            save_strip(imgs_y, out_y)

        run_walk(idx_left, "left_of")
        run_walk(idx_right, "right_of")
        run_walk(idx_above, "above")
        run_walk(idx_below, "below")
        run_walk(idx_overlap, "overlap")

    gx_pos = g_x.copy()
    gy = g_y.copy()

    run_walks(gx_pos, gy, "gx_pos")

    if RUN_BOTH_GX_DIRECTIONS:
        gx_neg = -g_x.copy()
        run_walks(gx_neg, gy, "gx_neg")

    # ----- 2D (g_x, g_y) manifold grid (GLOBAL ranges) -----
    print("[interp256] === 2D GLOBAL GRID + HEATMAPS ===")

    # We will also run BOTH gx directions for the 2D plot if enabled
    gx_variants = [("gx_pos", gx_pos)]
    if RUN_BOTH_GX_DIRECTIONS:
        gx_variants.append(("gx_neg", -gx_pos))

    # For global ranges: project dataset onto gx/gy for each gx choice.
    for tag, gx_vec in gx_variants:
        gx_u = gx_vec / np.linalg.norm(gx_vec)
        gy_u = gy / np.linalg.norm(gy)

        t_x_all = Z @ gx_u
        t_y_all = Z @ gy_u

        tx_min, tx_max = float(t_x_all.min()), float(t_x_all.max())
        ty_min, ty_max = float(t_y_all.min()), float(t_y_all.max())

        print(f"[interp256] 2D grid ({tag}) using GLOBAL ranges")
        print(f"[interp256]   GLOBAL t_x range: [{tx_min:.4f}, {tx_max:.4f}]")
        print(f"[interp256]   GLOBAL t_y range: [{ty_min:.4f}, {ty_max:.4f}]")

        z0 = choose_grid_center(Z, y, idx_overlap)

        # Build label/conf grid
        z_grid_L, _, _ = make_global_grid(Z, gx_u, gy_u, z0, GRID_LABEL_N, tx_min, tx_max, ty_min, ty_max)
        labels_flat, conf_flat = predict_rel_labels_and_conf(model, z_grid_L)
        labels_grid = labels_flat.reshape(GRID_LABEL_N, GRID_LABEL_N)
        conf_grid   = conf_flat.reshape(GRID_LABEL_N, GRID_LABEL_N)

        out_prefix = os.path.join(
            OUT_DIR,
            f"interp256_global_{tag}_center_{GRID_CENTER_MODE}_N{GRID_LABEL_N}"
        )
        save_label_and_conf_heatmaps(labels_grid, conf_grid, out_prefix, f"{tag}, center={GRID_CENTER_MODE}, GLOBAL")

        # Optional decoded atlas
        imgN = GRID_IMG_N
        if SAVE_LARGE_IMAGE_ATLAS:
            imgN = GRID_LABEL_N

        z_grid_I, _, _ = make_global_grid(Z, gx_u, gy_u, z0, imgN, tx_min, tx_max, ty_min, ty_max)
        imgs_grid = decode_latents(model, z_grid_I)
        grid_img = make_grid(imgs_grid, nrow=imgN, padding=1)
        out_grid = os.path.join(OUT_DIR, f"interp256_global_{tag}_decoded_center_{GRID_CENTER_MODE}_N{imgN}.png")
        save_image(grid_img, out_grid)
        print(f"[interp256] Saved decoded grid -> {out_grid}")

    # ----- Pairwise morphs -----
    print("[interp256] === Pairwise morphs ===")

    print(f"[interp256]   left_of (idx={idx_left}) -> right_of (idx={idx_right})")
    imgs_lr = make_pairwise_morph(model, Z[idx_left], Z[idx_right])
    out_lr = os.path.join(OUT_DIR, f"interp256_morph_left_to_right_idx{idx_left}_{idx_right}.png")
    save_strip(imgs_lr, out_lr)

    print(f"[interp256]   above (idx={idx_above}) -> below (idx={idx_below})")
    imgs_ab = make_pairwise_morph(model, Z[idx_above], Z[idx_below])
    out_ab = os.path.join(OUT_DIR, f"interp256_morph_above_to_below_idx{idx_above}_{idx_below}.png")
    save_strip(imgs_ab, out_ab)

    print(f"[interp256]   overlap (idx={idx_overlap}) -> left_of (idx={idx_left})")
    imgs_ol = make_pairwise_morph(model, Z[idx_overlap], Z[idx_left])
    out_ol = os.path.join(OUT_DIR, f"interp256_morph_overlap_to_left_idx{idx_overlap}_{idx_left}.png")
    save_strip(imgs_ol, out_ol)


if __name__ == "__main__":
    main()
