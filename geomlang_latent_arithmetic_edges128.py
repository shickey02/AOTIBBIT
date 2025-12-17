#!/usr/bin/env python3
# geomlang_latent_arithmetic_edges64.py
#
# Latent relation arithmetic for the 64x64 edges+relscale model.
#
# - Loads SceneModelEdges64 from:
#       outputs_edges_relscale/scene_model_edges_relscale.pt
# - Generates a fresh static dataset of scenes with:
#       ch0: red fill
#       ch1: blue fill
#       ch2: edges (both shapes)
# - Encodes all scenes into latents
# - Computes left→right and above→below latent "relation vectors"
# - Measures cosine consistency of these vectors
# - Renders:
#       * A grid: left, left+mean_vec, right (for a few examples)
#       * A t-SNE plot with left/right and above/below arrows
#
# Run:
#   python bbit_geomlang/geomlang_latent_arithmetic_edges64.py

import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# -----------------------
# Config
# -----------------------
IMG_SIZE   = 128
NUM_CH     = 3    # red, blue, edges
LATENT_DIM = 256

N_SAMPLES  = 6000
BATCH_SIZE = 128

OUT_DIR          = "outputs_edges_relscale_128"
CKPT_SCENEMODEL  = os.path.join(OUT_DIR, "scene_model_edges_relscale_128.pt")
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


class GeomEdges64Dataset(Dataset):
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
# Model – must match training architecture
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


class SceneModelEdges64(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

        # heads (for checkpoint compatibility)
        self.rel_head   = nn.Linear(LATENT_DIM, 5)
        self.scale_head = nn.Linear(LATENT_DIM, 3)
        self.shape_r_head = nn.Linear(LATENT_DIM, 2)
        self.shape_b_head = nn.Linear(LATENT_DIM, 2)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


def load_scene_model():
    print(f"[arith64] Loading SceneModel from {CKPT_SCENEMODEL}")
    ckpt = torch.load(CKPT_SCENEMODEL, map_location=DEVICE)
    model = SceneModelEdges64()
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
# Utility: visualization helper
# -----------------------

def chw_to_rgb(img_chw):
    """
    img_chw: [3,H,W], with:
      ch0 = red,
      ch1 = blue,
      ch2 = edges
    We'll map:
      R <- red
      G <- edges
      B <- blue
    """
    c, h, w = img_chw.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[..., 0] = img_chw[0]          # red
    rgb[..., 1] = img_chw[2]          # edges
    rgb[..., 2] = img_chw[1]          # blue
    return rgb

# -----------------------
# Main arithmetic / viz
# -----------------------

def main():
    print(f"[arith64] Using device: {DEVICE}")
    model = load_scene_model()

    # Generate dataset
    print(f"[arith64] Generating dataset: N={N_SAMPLES}")
    ds = GeomEdges64Dataset(N_SAMPLES)
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

    # Relation counts
    counts = []
    for r in range(5):
        n = int((rel_np == r).sum())
        counts.append(n)
        print(f"[arith64] relation {r}: {REL_NAMES[r]:>7}, count={n}")

    # Build LR and AB difference clouds using paired samples
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

    left_idx, right_idx = make_pairs(idx_left, idx_right)
    above_idx, below_idx = make_pairs(idx_above, idx_below)

    z_left  = z_np[left_idx]
    z_right = z_np[right_idx]
    z_above = z_np[above_idx]
    z_below = z_np[below_idx]

    v_lr = z_right - z_left    # left→right
    v_ab = z_below - z_above   # above→below

    def vector_stats(v, name):
        if v.shape[0] == 0:
            print(f"[arith64] WARNING: no pairs for {name}")
            return np.zeros((LATENT_DIM,), dtype=np.float32)
        norms = np.linalg.norm(v, axis=1)
        print(f"\n[arith64] {name} vector norms:")
        print(f"  mean={norms.mean():.4f}, std={norms.std():.4f}, "
              f"min={norms.min():.4f}, max={norms.max():.4f}")
        v_mean = v.mean(axis=0)
        mean_norm = np.linalg.norm(v_mean) + 1e-8
        cos = (v @ v_mean) / (norms * mean_norm + 1e-8)
        print(f"[arith64] {name} vector cosine similarity to mean:")
        print(f"  mean={cos.mean():.3f}, std={cos.std():.3f}")
        return v_mean

    v_lr_mean = vector_stats(v_lr, "left→right")
    v_ab_mean = vector_stats(v_ab, "above→below")

    # -----------------------
    # Grid visualization: left, left+v_lr_mean, right
    # -----------------------
    n_rows = 8
    n_rows = min(n_rows, len(left_idx))
    if n_rows > 0:
        fig, axes = plt.subplots(n_rows, 3, figsize=(6, 2*n_rows), dpi=120)
        fig.suptitle("64x64 edges: latent arithmetic (left→right)", fontsize=12)

        for row in range(n_rows):
            li = left_idx[row]
            ri = right_idx[row]

            img_left = all_imgs[li]           # [3,64,64]
            z_left_t = all_z[li:li+1].to(DEVICE)
            z_left_offset = (z_left_t + torch.from_numpy(v_lr_mean).to(DEVICE)).float()
            img_pred = model.decode(z_left_offset).cpu()[0]

            img_right = all_imgs[ri]

            for col, img_chw in enumerate([img_left, img_pred, img_right]):
                ax = axes[row, col] if n_rows > 1 else axes[col]
                ax.axis("off")
                if row == 0:
                    if col == 0:
                        ax.set_title("left", fontsize=8)
                    elif col == 1:
                        ax.set_title("left + μ(LR)", fontsize=8)
                    else:
                        ax.set_title("right (true)", fontsize=8)
                ax.imshow(chw_to_rgb(img_chw.detach().cpu().numpy()))


        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out_grid = os.path.join(OUT_DIR, "latent_arith_grid_edges64.png")
        plt.savefig(out_grid)
        print(f"[arith64] Saved grid -> {out_grid}")

    # -----------------------
    # t-SNE with arrows
    # -----------------------
    print("[arith64] Computing t-SNE embedding (this may take a bit)...")
    # Subsample for t-SNE if needed
    max_tsne = 2000
    N = z_np.shape[0]
    if N > max_tsne:
        tsne_idx = rng.choice(N, size=max_tsne, replace=False)
        z_tsne_in = z_np[tsne_idx]
        rel_tsne = rel_np[tsne_idx]
    else:
        tsne_idx = np.arange(N)
        z_tsne_in = z_np
        rel_tsne = rel_np

    tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="random", random_state=0)
    z_2d = tsne.fit_transform(z_tsne_in)

    # map original indices -> tsne indices (for arrow drawing)
    idx_map = {orig_i: k for k, orig_i in enumerate(tsne_idx)}

    # base scatter
    colors = {
        REL_LEFT:   "tab:red",
        REL_RIGHT:  "tab:blue",
        REL_ABOVE:  "tab:green",
        REL_BELOW:  "tab:purple",
        REL_OVERLAP:"tab:gray",
    }

    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    for r in range(5):
        m = (rel_tsne == r)
        if not np.any(m):
            continue
        ax.scatter(z_2d[m, 0], z_2d[m, 1], s=8, alpha=0.4, label=REL_NAMES[r], color=colors.get(r, "black"))

    # draw some LR and AB arrows
    def draw_arrows(pair_idx_a, pair_idx_b, color, label_prefix, max_arrows=80):
        n = min(len(pair_idx_a), len(pair_idx_b), max_arrows)
        if n == 0:
            return
        sel = rng.choice(n, size=n, replace=False)
        for i in sel:
            ia = pair_idx_a[i]
            ib = pair_idx_b[i]
            if ia not in idx_map or ib not in idx_map:
                continue
            pa = z_2d[idx_map[ia]]
            pb = z_2d[idx_map[ib]]
            ax.arrow(pa[0], pa[1], (pb[0] - pa[0]), (pb[1] - pa[1]),
                     head_width=0.4, head_length=0.6, length_includes_head=True,
                     alpha=0.5, color=color)

    draw_arrows(left_idx, right_idx, "black", "LR")
    draw_arrows(above_idx, below_idx, "orange", "AB")

    ax.set_title("t-SNE of latents (64x64 edges) with LR / AB arrows", fontsize=10)
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    out_tsne = os.path.join(OUT_DIR, "latent_arith_tsne_edges64.png")
    plt.savefig(out_tsne)
    print(f"[arith64] Saved t-SNE arrows -> {out_tsne}")


if __name__ == "__main__":
    main()
