#!/usr/bin/env python3
# geomlang_interpolate_basic32.py
#
# Cross-manifold latent interpolation for the basic 32×32 geomlang model.
#
# - Loads SceneModel (autoencoder) from outputs_basic32/scene_model_basic32.pt
# - Generates a static dataset of random red/blue scenes
# - Labels each scene with a *geometric* relation: left_of, right_of, above, below, overlap
# - Picks pairs:
#       left_of  -> right_of
#       above   -> below
# - Interpolates linearly in latent space between each pair
# - Decodes all intermediate latents to images and saves an interpolation grid
# - Runs t-SNE on the latent manifold + interpolation points and plots paths
#
# Run:
#   python bbit_geomlang/geomlang_interpolate_basic32.py

import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# -----------------------
# Config
# -----------------------
IMG_SIZE      = 32
NUM_CHANNELS  = 3
LATENT_DIM    = 48

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CKPT_SCENEMODEL = os.path.join("outputs_basic32", "scene_model_basic32.pt")
OUT_DIR         = "outputs_basic32"
os.makedirs(OUT_DIR, exist_ok=True)

N_STATIC_SCENES     = 2000      # how many random scenes to sample
N_PAIRS_PER_RELTYPE = 4         # pairs for left<->right and above<->below
N_STEPS_INTERP      = 9         # number of interpolation steps (incl. endpoints)

REL_NAMES = ["left_of", "right_of", "above", "below", "overlap"]  # geometric labels 0..4

# -----------------------
# Drawing primitives
# -----------------------
def draw_circle(grid, cx, cy, radius, channel_idx):
    """Draw a filled circle onto grid[channel_idx]."""
    h, w = grid.shape[1], grid.shape[2]
    for y in range(h):
        for x in range(w):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                grid[channel_idx, y, x] = 1.0


def draw_square(grid, cx, cy, half_size, channel_idx):
    """Draw a filled square onto grid[channel_idx]."""
    h, w = grid.shape[1], grid.shape[2]
    x0 = max(0, cx - half_size)
    x1 = min(w, cx + half_size + 1)
    y0 = max(0, cy - half_size)
    y1 = min(h, cy + half_size + 1)
    grid[channel_idx, y0:y1, x0:x1] = 1.0


def draw_shape(grid, shape_id, cx, cy, size, channel_idx):
    if shape_id == 0:
        draw_circle(grid, cx, cy, size, channel_idx)
    else:
        draw_square(grid, cx, cy, size, channel_idx)


# -----------------------
# Geometry / relations
# -----------------------
def compute_centroid(img_chw, channel_idx):
    """
    img_chw: [C,H,W] tensor or np.array in [0,1]
    Returns (cx, cy) in pixel coords.
    """
    if isinstance(img_chw, torch.Tensor):
        x = img_chw[channel_idx].detach().cpu().numpy()
    else:
        x = img_chw[channel_idx]
    H, W = x.shape
    yy, xx = np.mgrid[0:H, 0:W]
    weights = x.astype(np.float32)
    total = weights.sum()
    if total <= 1e-6:
        return W / 2.0, H / 2.0
    cx = float((weights * xx).sum() / total)
    cy = float((weights * yy).sum() / total)
    return cx, cy


def relation_from_centroids(cx_r, cy_r, cx_b, cy_b, tol=1.0):
    """
    Simple geometric relation between red and blue based on centroids.

    Returns int in {0..4} for REL_NAMES.
    """
    # horizontal first
    if cx_r + tol < cx_b - tol:
        return 0  # left_of
    if cx_r - tol > cx_b + tol:
        return 1  # right_of

    # vertical
    if cy_r + tol < cy_b - tol:
        return 2  # above
    if cy_r - tol > cy_b + tol:
        return 3  # below

    return 4      # overlap / near


# -----------------------
# Static dataset generation
# -----------------------
def generate_static_scene():
    """
    Generate a single random static scene (no labels).
    Returns: img [C,H,W] in [0,1], plus internal metadata if needed.
    """
    H = W = IMG_SIZE
    margin = 4

    grid = np.zeros((NUM_CHANNELS, H, W), dtype=np.float32)

    # sample shapes
    shape_red = np.random.randint(0, 2)   # 0 circle, 1 square
    shape_blue = np.random.randint(0, 2)

    base = np.random.randint(4, 7)        # size ~4..6
    r_red = int(np.clip(base + np.random.randint(-1, 2), 3, 7))
    r_blue = int(np.clip(base + np.random.randint(-1, 2), 3, 7))

    def sample_center(r):
        cx = np.random.randint(margin + r, W - margin - r)
        cy = np.random.randint(margin + r, H - margin - r)
        return cx, cy

    cx_r, cy_r = sample_center(r_red)
    cx_b, cy_b = sample_center(r_blue)

    draw_shape(grid, shape_red, cx_r, cy_r, r_red, 0)  # red
    draw_shape(grid, shape_blue, cx_b, cy_b, r_blue, 2)  # blue

    return grid, (cx_r, cy_r, r_red, shape_red, cx_b, cy_b, r_blue, shape_blue)


def generate_labeled_static_dataset(n_scenes):
    """
    Returns:
        imgs:    np.array [N,C,H,W]
        labels:  np.array [N] (relation id 0..4)
    """
    imgs = []
    labels = []

    while len(imgs) < n_scenes:
        img, meta = generate_static_scene()
        cx_r, cy_r, _, _, cx_b, cy_b, _, _ = meta
        rel = relation_from_centroids(cx_r, cy_r, cx_b, cy_b)
        imgs.append(img)
        labels.append(rel)

    imgs = np.stack(imgs, axis=0)
    labels = np.array(labels, dtype=np.int64)
    return imgs, labels


# -----------------------
# SceneModel (same as training)
# -----------------------
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
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self, out_channels=NUM_CHANNELS, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(16, out_channels, kernel_size=4, stride=2, padding=1)

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
        # classifier heads exist in checkpoint but aren't needed here
        self.rel_head = nn.Linear(LATENT_DIM, 6)
        self.scale_head = nn.Linear(LATENT_DIM, 3)
        self.shape_red_head = nn.Linear(LATENT_DIM, 2)
        self.shape_blue_head = nn.Linear(LATENT_DIM, 2)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


def load_scene_model(path=CKPT_SCENEMODEL):
    print(f"[interp] Loading SceneModel from {path}")
    ckpt = torch.load(path, map_location=DEVICE)
    if isinstance(ckpt, SceneModel):
        model = ckpt
    else:
        model = SceneModel()
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
# Helper: pick pairs between relation types
# -----------------------
def pick_pairs(latents, labels, rel_a, rel_b, n_pairs, rng):
    """
    latents: [N,D]
    labels:  [N]
    Returns: list of (idx_a, idx_b)
    """
    idx_a = np.where(labels == rel_a)[0]
    idx_b = np.where(labels == rel_b)[0]

    if len(idx_a) == 0 or len(idx_b) == 0:
        print(f"[interp] Warning: not enough examples for {REL_NAMES[rel_a]} or {REL_NAMES[rel_b]}")
        return []

    pairs = []
    for _ in range(n_pairs):
        i = int(rng.choice(idx_a))
        z0 = latents[i]
        # nearest neighbor in rel_b
        z_b = latents[idx_b]
        dists = np.linalg.norm(z_b - z0[None, :], axis=1)
        j = int(idx_b[np.argmin(dists)])
        pairs.append((i, j))
    return pairs


# -----------------------
# Interpolation + grids
# -----------------------
def to_rgb(img_chw):
    """Convert [C,H,W] (red channel 0, blue channel 2) to RGB for plotting."""
    if isinstance(img_chw, torch.Tensor):
        img = img_chw.detach().cpu().numpy()
    else:
        img = img_chw
    r = img[0]
    b = img[2]
    H, W = r.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    rgb[..., 0] = r
    rgb[..., 2] = b
    return rgb


def main():
    print(f"[interp] Using device: {DEVICE}")

    # 1) Load AE
    ae = load_scene_model()

    # 2) Generate static dataset
    print(f"[interp] Generating {N_STATIC_SCENES} static scenes...")
    imgs_np, labels = generate_labeled_static_dataset(N_STATIC_SCENES)  # [N,C,H,W], [N]
    N = imgs_np.shape[0]

    # 3) Encode to latents
    print("[interp] Encoding latents...")
    with torch.no_grad():
        x = torch.from_numpy(imgs_np).float().to(DEVICE)
        z = ae.encode(x).cpu().numpy()  # [N,D]

    # 4) Pick pairs: left<->right and above<->below
    rng = np.random.default_rng(0)

    pairs = []
    pair_types = []  # string labels per row

    lr_pairs = pick_pairs(z, labels, 0, 1, N_PAIRS_PER_RELTYPE, rng)  # left_of -> right_of
    for (i, j) in lr_pairs:
        pairs.append((i, j))
        pair_types.append("left→right")

    ab_pairs = pick_pairs(z, labels, 2, 3, N_PAIRS_PER_RELTYPE, rng)  # above -> below
    for (i, j) in ab_pairs:
        pairs.append((i, j))
        pair_types.append("above→below")

    if len(pairs) == 0:
        print("[interp] No relation pairs found; aborting.")
        return

    R = len(pairs)
    S = N_STEPS_INTERP
    print(f"[interp] Using {R} pairs, {S} interpolation steps each.")

    # 5) Build all interpolation latents
    alphas = np.linspace(0.0, 1.0, S, dtype=np.float32)

    z_interp_all = np.zeros((R, S, LATENT_DIM), dtype=np.float32)
    for r, (i, j) in enumerate(pairs):
        z0 = z[i]
        z1 = z[j]
        for s, alpha in enumerate(alphas):
            z_interp_all[r, s] = (1.0 - alpha) * z0 + alpha * z1

    # 6) Decode all interpolations in one go
    z_interp_flat = torch.from_numpy(z_interp_all.reshape(-1, LATENT_DIM)).float().to(DEVICE)
    with torch.no_grad():
        imgs_interp_flat = ae.decode(z_interp_flat).cpu().numpy()
    imgs_interp = imgs_interp_flat.reshape(R, S, NUM_CHANNELS, IMG_SIZE, IMG_SIZE)

    # 7) Make interpolation grid figure
    fig, axes = plt.subplots(
        nrows=R,
        ncols=S,
        figsize=(S * 1.2, R * 1.2),
        dpi=120,
    )
    fig.suptitle("Latent interpolations (basic32): each row is one relation transition", fontsize=12)

    for r in range(R):
        for s in range(S):
            ax = axes[r, s] if R > 1 else axes[s]
            ax.axis("off")
            if r == 0:
                ax.set_title(f"α={alphas[s]:.2f}", fontsize=8)
            img = imgs_interp[r, s]
            ax.imshow(to_rgb(img))

        # label row on the left
        axes[r, 0].set_ylabel(pair_types[r], fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    grid_path = os.path.join(OUT_DIR, "interpolate_grid_basic32.png")
    plt.savefig(grid_path)
    print(f"[interp] Saved interpolation grid -> {grid_path}")

    # ---------------- t-SNE manifold with interpolation paths ----------------
    print("[interp] Running t-SNE...")
    z_static = z  # [N,D]
    z_interp_flat = z_interp_all.reshape(-1, LATENT_DIM)  # [R*S,D]

    z_all = np.concatenate([z_static, z_interp_flat], axis=0)
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="random",
        random_state=0,
    )
    Y_all = tsne.fit_transform(z_all)

    Y_static = Y_all[:N]
    Y_interp = Y_all[N:].reshape(R, S, 2)

    fig2, ax2 = plt.subplots(figsize=(8, 8), dpi=120)
    ax2.set_title("t-SNE manifold with latent interpolations (basic32)")

    # base manifold
    ax2.scatter(
        Y_static[:, 0],
        Y_static[:, 1],
        s=5,
        alpha=0.25,
        color="lightsteelblue",
        label="static latents",
    )

    # interpolation paths
    colors = {
        "left→right": "darkorange",
        "above→below": "seagreen",
    }

    for r in range(R):
        pts = Y_interp[r]  # [S,2]
        c = colors.get(pair_types[r], "black")
        ax2.plot(pts[:, 0], pts[:, 1], "-", linewidth=1.0, color=c, alpha=0.9)
        ax2.scatter(pts[0, 0], pts[0, 1], marker="o", color=c, s=25)  # start
        ax2.scatter(pts[-1, 0], pts[-1, 1], marker="x", color=c, s=30)  # end

    # simple legend
    handles = []
    labels_legend = []

    handles.append(plt.Line2D([0], [0], marker="o", linestyle="none",
                              color="lightsteelblue", markersize=5))
    labels_legend.append("static latents")

    for name, c in colors.items():
        handles.append(plt.Line2D([0], [0], color=c))
        labels_legend.append(name)

    ax2.legend(handles, labels_legend, loc="best")

    ax2.set_xticks([])
    ax2.set_yticks([])

    tsne_path = os.path.join(OUT_DIR, "interpolate_tsne_basic32.png")
    plt.tight_layout()
    plt.savefig(tsne_path)
    print(f"[interp] Saved t-SNE interpolation plot -> {tsne_path}")


if __name__ == "__main__":
    main()
