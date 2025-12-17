#!/usr/bin/env python3
# geomlang_latent_arithmetic_basic32.py
#
# Latent relation vector arithmetic for basic32:
# - regenerate a synthetic dataset with 5 relations (left/right/above/below/overlap)
# - encode with SceneModel autoencoder
# - compute relation class means and relation vectors (e.g., left→right)
# - evaluate consistency via cosine similarity
# - visualize original vs transformed images
# - visualize t-SNE manifold with arithmetic arrows

import os
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE

# --------------------
# Config
# --------------------
IMG_SIZE     = 32
NUM_CHANNELS = 3
LATENT_DIM   = 48

N_SAMPLES    = 6000   # scenes to generate
N_VIS_ROWS   = 8      # rows per block in the image grid
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR          = "outputs_basic32"
CKPT_SCENEMODEL  = os.path.join(OUT_DIR, "scene_model_basic32.pt")

os.makedirs(OUT_DIR, exist_ok=True)

REL_NAMES = ["left_of", "right_of", "above", "below", "overlap"]
N_REL = len(REL_NAMES)

# --------------------
# Drawing primitives
# --------------------
def draw_circle(grid, cx, cy, radius, channel_idx):
    h, w = grid.shape[1], grid.shape[2]
    for y in range(h):
        for x in range(w):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                grid[channel_idx, y, x] = 1.0


def draw_square(grid, cx, cy, half_size, channel_idx):
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

# --------------------
# Synthetic scene generator
# --------------------
def generate_scene():
    """
    Generate one basic32 scene:
      - 2 shapes (red in channel 0, blue in channel 2)
      - 5 possible relations: left_of, right_of, above, below, overlap
    Returns:
      img: [3, H, W] float32 in [0,1]
      rel_idx: 0..4
    """
    H = W = IMG_SIZE
    margin = 4

    # Shapes & sizes
    shape_red  = np.random.randint(0, 2)   # 0 circle, 1 square
    shape_blue = np.random.randint(0, 2)
    base = np.random.randint(4, 7)         # 4..6
    r_red  = int(np.clip(base + np.random.randint(-1, 2), 3, 7))
    r_blue = int(np.clip(base + np.random.randint(-1, 2), 3, 7))

    img = np.zeros((NUM_CHANNELS, H, W), dtype=np.float32)

    # Choose relation
    rel_idx = np.random.randint(0, N_REL)
    # we want x,y at least (margin+size) away from borders
    def rand_x(size):
        return np.random.randint(margin + size, W - margin - size)
    def rand_y(size):
        return np.random.randint(margin + size, H - margin - size)

    if rel_idx == 0:  # left_of
        cy = rand_y(max(r_red, r_blue))
        cx_red  = np.random.randint(margin + r_red,  W//2 - margin)
        cx_blue = np.random.randint(W//2 + margin,  W - margin - r_blue)
        cy_red  = cy_blue = cy
    elif rel_idx == 1:  # right_of
        cy = rand_y(max(r_red, r_blue))
        cx_blue = np.random.randint(margin + r_blue, W//2 - margin)
        cx_red  = np.random.randint(W//2 + margin, W - margin - r_red)
        cy_red  = cy_blue = cy
    elif rel_idx == 2:  # above
        cx = rand_x(max(r_red, r_blue))
        cy_red  = np.random.randint(margin + r_red,  H//2 - margin)
        cy_blue = np.random.randint(H//2 + margin,  H - margin - r_blue)
        cx_red  = cx_blue = cx
    elif rel_idx == 3:  # below
        cx = rand_x(max(r_red, r_blue))
        cy_blue = np.random.randint(margin + r_blue, H//2 - margin)
        cy_red  = np.random.randint(H//2 + margin, H - margin - r_red)
        cx_red  = cx_blue = cx
    else:  # overlap/near
        cx = rand_x(max(r_red, r_blue))
        cy = rand_y(max(r_red, r_blue))
        jitter = np.random.randint(-2, 3, size=2)
        cx_red, cy_red = cx, cy
        cx_blue = int(np.clip(cx + jitter[0], margin + r_blue, W - margin - r_blue))
        cy_blue = int(np.clip(cy + jitter[1], margin + r_blue, H - margin - r_blue))

    draw_shape(img, shape_red,  cx_red,  cy_red,  r_red,  channel_idx=0)
    draw_shape(img, shape_blue, cx_blue, cy_blue, r_blue, channel_idx=2)

    return img, rel_idx


def generate_dataset(n_samples):
    imgs  = []
    rels  = []
    for _ in range(n_samples):
        img, r = generate_scene()
        imgs.append(img)
        rels.append(r)
    imgs = np.stack(imgs, axis=0)  # [N, C, H, W]
    rels = np.array(rels, dtype=np.int64)
    return imgs, rels

# --------------------
# SceneModel (AE) – same as in other basic32 scripts
# --------------------
class Encoder(nn.Module):
    def __init__(self, in_channels=NUM_CHANNELS, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, 2, 1)  # 16x16
        self.conv2 = nn.Conv2d(16, 32, 3, 2, 1)           # 8x8
        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)           # 4x4
        self.fc    = nn.Linear(64 * 4 * 4, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self, out_channels=NUM_CHANNELS, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc      = nn.Linear(latent_dim, 64 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(16, out_channels, 4, 2, 1)

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
        # classifier heads exist in the checkpoint; we include them so load_state_dict works
        self.rel_head        = nn.Linear(LATENT_DIM, 6)
        self.scale_head      = nn.Linear(LATENT_DIM, 3)
        self.shape_red_head  = nn.Linear(LATENT_DIM, 2)
        self.shape_blue_head = nn.Linear(LATENT_DIM, 2)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


def load_scene_model():
    print(f"[arith] Loading SceneModel from {CKPT_SCENEMODEL}")
    ckpt = torch.load(CKPT_SCENEMODEL, map_location=DEVICE)

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

# --------------------
# Helpers
# --------------------
def to_rgb(img_chw):
    # Only use red & blue channels for visualization
    r = img_chw[0]
    b = img_chw[2]
    h, w = r.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[..., 0] = r
    rgb[..., 2] = b
    return rgb


def cosine_similarity(u, v, eps=1e-8):
    u = u / (np.linalg.norm(u) + eps)
    v = v / (np.linalg.norm(v) + eps)
    return float(np.dot(u, v))

# --------------------
# Main analysis
# --------------------
def main():
    print(f"[arith] Using device: {DEVICE}")

    # Load AE
    ae = load_scene_model()

    # Generate dataset
    print(f"[arith] Generating dataset: N={N_SAMPLES}")
    imgs_np, rels = generate_dataset(N_SAMPLES)

    # Encode
    x = torch.from_numpy(imgs_np).float().to(DEVICE)
    with torch.no_grad():
        z = ae.encode(x).cpu().numpy()   # [N, D]

    # Per-relation clusters & means
    zs_by_rel = []
    mu = []
    for r in range(N_REL):
        mask = (rels == r)
        cluster = z[mask]
        zs_by_rel.append(cluster)
        mean_r = cluster.mean(axis=0)
        mu.append(mean_r)
        print(f"[arith] relation {r}: {REL_NAMES[r]:>8s}, count={cluster.shape[0]}")

    mu = np.stack(mu, axis=0)  # [5, D]

    # Relation vectors (we focus on bidirectional pairs)
    v_left_to_right   = mu[1] - mu[0]
    v_right_to_left   = -v_left_to_right
    v_above_to_below  = mu[3] - mu[2]
    v_below_to_above  = -v_above_to_below

    print("\n[arith] Relation vector norms:")
    print(f"  ||left→right||   = {np.linalg.norm(v_left_to_right):.4f}")
    print(f"  ||above→below||  = {np.linalg.norm(v_above_to_below):.4f}")

    # Consistency: cosine similarity of instance offsets vs mean vector
    rng = np.random.default_rng(0)
    def consistency(rel_a, rel_b, v_mean, n_pairs=300):
        za = zs_by_rel[rel_a]
        zb = zs_by_rel[rel_b]
        n = min(len(za), len(zb), n_pairs)
        idx_a = rng.choice(len(za), size=n, replace=False)
        idx_b = rng.choice(len(zb), size=n, replace=False)
        cos = []
        for i, j in zip(idx_a, idx_b):
            d = zb[j] - za[i]
            cos.append(cosine_similarity(d, v_mean))
        cos = np.array(cos)
        return cos.mean(), cos.std()

    mean_lr, std_lr = consistency(0, 1, v_left_to_right)
    mean_ab, std_ab = consistency(2, 3, v_above_to_below)

    print("\n[arith] Vector consistency (cosine similarity to mean vector):")
    print(f"  left→right:  mean={mean_lr:.3f}, std={std_lr:.3f}")
    print(f"  above→below: mean={mean_ab:.3f}, std={std_ab:.3f}")

    # --------------------
    # Visualization grid: original vs transformed
    # --------------------
    ae.eval()
    with torch.no_grad():
        # select some examples from each relation
        def sample_indices(rel, k):
            idx = np.where(rels == rel)[0]
            if len(idx) == 0:
                return np.array([], dtype=int)
            if len(idx) < k:
                return rng.choice(idx, size=len(idx), replace=False)
            return rng.choice(idx, size=k, replace=False)

        rows = N_VIS_ROWS
        idx_left  = sample_indices(0, rows)
        idx_right = sample_indices(1, rows)
        idx_above = sample_indices(2, rows)
        idx_below = sample_indices(3, rows)

        def transform_and_decode(idx_list, vec):
            if len(idx_list) == 0:
                return []
            z_src = torch.from_numpy(z[idx_list]).float().to(DEVICE)
            z_tgt = z_src + torch.from_numpy(vec).float().to(DEVICE)
            imgs_src = ae.decode(z_src).cpu().numpy()
            imgs_tgt = ae.decode(z_tgt).cpu().numpy()
            return imgs_src, imgs_tgt

        left_src,  left2right = transform_and_decode(idx_left,  v_left_to_right)
        right_src, right2left = transform_and_decode(idx_right, v_right_to_left)
        above_src, above2below = transform_and_decode(idx_above, v_above_to_below)
        below_src, below2above = transform_and_decode(idx_below, v_below_to_above)

    # grid: 4 blocks, each block: [original | transformed]
    blocks = [
        ("left → right",   left_src,  left2right),
        ("right → left",   right_src, right2left),
        ("above → below",  above_src, above2below),
        ("below → above",  below_src, below2above),
    ]

    n_blocks = len(blocks)
    ncols = 2
    nrows = rows * n_blocks

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * 2.0, nrows * 2.0 / 3.0),
        dpi=120,
    )
    fig.suptitle("Latent relation arithmetic (basic32): original vs transformed", fontsize=14)

    row_offset = 0
    for b_idx, (title, src, tgt) in enumerate(blocks):
        if src is None or len(src) == 0:
            continue
        for r in range(src.shape[0]):
            r_global = row_offset + r
            for c in range(ncols):
                ax = axes[r_global, c]
                ax.axis("off")
                if r == 0:
                    if c == 0:
                        ax.set_title(title + "\nsource", fontsize=8)
                    else:
                        ax.set_title("transformed", fontsize=8)
                img = src[r] if c == 0 else tgt[r]
                ax.imshow(to_rgb(img))
        row_offset += src.shape[0]

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_grid = os.path.join(OUT_DIR, "latent_arith_grid_basic32.png")
    plt.savefig(out_grid)
    print(f"[arith] Saved grid -> {out_grid}")

    # --------------------
    # t-SNE manifold with arrows
    # --------------------
    print("[arith] Computing t-SNE embedding (this may take a bit)...")
    # subsample for TSNE to keep it quick
    max_tsne = 2500
    if N_SAMPLES > max_tsne:
        idx_tsne = rng.choice(N_SAMPLES, size=max_tsne, replace=False)
    else:
        idx_tsne = np.arange(N_SAMPLES)
    z_tsne = z[idx_tsne]

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        init="random",
        random_state=0,
    )
    z_emb = tsne.fit_transform(z_tsne)  # [M,2]

    fig2, ax2 = plt.subplots(figsize=(7, 7), dpi=120)
    ax2.scatter(z_emb[:, 0], z_emb[:, 1], s=4, alpha=0.15, color="steelblue", label="static latents")

    # draw a few arrows for left→right and above→below
    def arrows_for_pair(rel_a, rel_b, v_mean, color, label, n_arrows=15):
        idx_a_all = np.where(rels == rel_a)[0]
        idx_b_all = np.where(rels == rel_b)[0]
        n = min(len(idx_a_all), n_arrows)
        if n == 0:
            return
        idx_a = rng.choice(idx_a_all, size=n, replace=False)
        # map those indices into tsne subset if present
        for i in idx_a:
            if i not in idx_tsne:
                continue
            pos = np.where(idx_tsne == i)[0]
            if len(pos) == 0:
                continue
            j = pos[0]
            x0, y0 = z_emb[j]
            # approximate target as z_i + v_mean, then project via local linearity:
            # we don't have embedding for target, so we just draw arrow in latent TSNE space
            # by adding the projected delta between z_i and (z_i + v_mean)
            # A crude proxy: nearest neighbor difference
            ax2.arrow(
                x0, y0,
                0.0, 0.0,  # no actual delta (we just mark origin); but we mark head with a dot
            )

    # Instead of actually projecting transformed latents (expensive), we
    # re-embed a small set explicitly.
    def embed_arrows(rel_a, rel_b, v_mean, color, label, n_arrows=20):
        idx_a_all = np.where(rels == rel_a)[0]
        n = min(len(idx_a_all), n_arrows)
        if n == 0:
            return
        idx_a = rng.choice(idx_a_all, size=n, replace=False)
        z_start = z[idx_a]
        z_target = z_start + v_mean
        z_pair = np.concatenate([z_start, z_target], axis=0)
        emb_pair = TSNE(
            n_components=2,
            perplexity=5,
            learning_rate=100,
            init="random",
            random_state=42,
        ).fit_transform(z_pair)
        emb_start = emb_pair[:n]
        emb_end   = emb_pair[n:]
        for k in range(n):
            x0, y0 = emb_start[k]
            x1, y1 = emb_end[k]
            ax2.plot([x0, x1], [y0, y1], color=color, linewidth=1)
            ax2.scatter([x0], [y0], color=color, s=12)
            ax2.scatter([x1], [y1], color=color, s=12, marker="x")
        ax2.plot([], [], color=color, label=label)  # legend handle

    embed_arrows(0, 1, v_left_to_right,  "orange", "left→right")
    embed_arrows(2, 3, v_above_to_below, "green",  "above→below")

    ax2.set_title("t-SNE manifold with latent relation vectors (basic32)")
    ax2.legend(loc="upper right")
    ax2.set_xticks([])
    ax2.set_yticks([])

    out_tsne = os.path.join(OUT_DIR, "latent_arith_tsne_basic32.png")
    plt.tight_layout()
    plt.savefig(out_tsne)
    print(f"[arith] Saved t-SNE arrows -> {out_tsne}")


if __name__ == "__main__":
    main()
