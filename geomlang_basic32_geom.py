#!/usr/bin/env python3
# geomlang_basic32_geom.py
#
# Basic 32x32 two-object scenes:
#   - Channels: 0=red object, 1=blue object, 2=unused (can be edges later)
#   - Relations: left_of, right_of, above, below, inside, overlapping
#   - Scale: red_larger, red_smaller, similar
#   - Shapes: circle or square
#
# Model:
#   - Conv encoder/decoder autoencoder
#   - Heads for relation, scale, red_shape, blue_shape
#   - NEW: geometry head predicting
#       (cx_red, cy_red, r_red, cx_blue, cy_blue, r_blue)
#
# Outputs (in outputs_basic32/):
#   - scene_model_basic32_geom.pt        (checkpoint)
#   - latents_basic32_geom.npz           (z, labels, geom)
#   - recon_grid_basic32_geom.png        (samples + recons)
#   - tsne_relations_basic32_geom.png    (latent clusters by relation)

import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Optional viz imports (t-SNE, matplotlib)
try:
    from sklearn.manifold import TSNE
    HAVE_TSNE = True
except Exception:
    HAVE_TSNE = False

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

IMG_SIZE = 32
NUM_CHANNELS = 3  # RGB-ish: 0=red, 1=blue, 2=unused
LATENT_DIM = 48

RELATION_NAMES = ["left_of", "right_of", "above", "below", "inside", "overlapping"]
SCALE_NAMES    = ["red_larger", "red_smaller", "similar"]
SHAPE_NAMES    = ["circle", "square"]  # 0 = circle, 1 = square

NUM_REL   = len(RELATION_NAMES)
NUM_SCALE = len(SCALE_NAMES)
NUM_SHAPE = len(SHAPE_NAMES)

NUM_SAMPLES = 4000
BATCH_SIZE  = 64
EPOCHS      = 300
LR          = 1e-3
GEOM_LAMBDA = 0.1   # weight on geometry MSE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR             = "outputs_basic32"
CKPT_PATH           = os.path.join(OUT_DIR, "scene_model_basic32_geom.pt")
LATENTS_PATH        = os.path.join(OUT_DIR, "latents_basic32_geom.npz")
RECON_GRID_PATH     = os.path.join(OUT_DIR, "recon_grid_basic32_geom.png")
TSNE_REL_PATH       = os.path.join(OUT_DIR, "tsne_relations_basic32_geom.png")

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Utility: drawing primitives
# ---------------------------------------------------------------------

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
    if shape_id == 0:  # circle
        draw_circle(grid, cx, cy, size, channel_idx)
    else:              # square
        draw_square(grid, cx, cy, size, channel_idx)


# ---------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------

def rand_band(low, high, alt_low, alt_high):
    """
    Safe randint helper:
    - Uses [low, high) if it's a valid interval (high > low + 1).
    - Otherwise falls back to [alt_low, alt_high).
    """
    if high <= low + 1:
        low, high = alt_low, alt_high
    return np.random.randint(low, high)


def generate_scene(rel_label, scale_label, shape_red, shape_blue):
    """
    rel_label:   0..5 (red relative to blue)
    scale_label: 0..2 (red_larger, red_smaller, similar)
    shape_red/shape_blue: 0=circle, 1=square

    Returns:
        img:   [3,H,W] float32
        scale_label (possibly adjusted)
        geom: np.array([cx_r, cy_r, r_r, cx_b, cy_b, r_b], float32)
    """
    img = np.zeros((NUM_CHANNELS, IMG_SIZE, IMG_SIZE), dtype=np.float32)

    margin = max(2, IMG_SIZE // 8)   # e.g. 32 → 4

    # ---------- choose base sizes ----------
    base = np.random.randint(4, 7)  # 4..6

    # ---------- sizes (with special handling for 'inside') ----------
    if rel_label == 4:  # inside: red strictly smaller than blue
        scale_label = 1  # force red_smaller
        r_blue = min(base + 3, (IMG_SIZE - 2 * margin) // 2)
        r_blue = max(r_blue, 3)
        r_red = max(2, r_blue - 2)
    else:
        if scale_label == 0:  # red_larger
            r_red = base + 2
            r_blue = base
        elif scale_label == 1:  # red_smaller
            r_red = base
            r_blue = base + 2
        else:  # similar
            delta = np.random.randint(-1, 2)  # -1,0,1
            r_red = base
            r_blue = base + delta

    full_low  = margin
    full_high = IMG_SIZE - margin
    mid       = IMG_SIZE // 2

    # ---------- positions conditioned on relation ----------
    if rel_label == 0:  # left_of
        cy = rand_band(full_low, full_high, full_low, full_high)
        cx_red  = rand_band(margin, mid - margin, full_low, mid)
        cx_blue = rand_band(mid + margin // 2, IMG_SIZE - margin,
                            mid, full_high)
        cy_red = cy_blue = cy

    elif rel_label == 1:  # right_of
        cy = rand_band(full_low, full_high, full_low, full_high)
        cx_blue = rand_band(margin, mid - margin, full_low, mid)
        cx_red  = rand_band(mid + margin // 2, IMG_SIZE - margin,
                            mid, full_high)
        cy_red = cy_blue = cy

    elif rel_label == 2:  # above
        cx = rand_band(full_low, full_high, full_low, full_high)
        cy_red  = rand_band(margin, mid - margin, full_low, mid)
        cy_blue = rand_band(mid + margin // 2, IMG_SIZE - margin,
                            mid, full_high)
        cx_red = cx_blue = cx

    elif rel_label == 3:  # below
        cx = rand_band(full_low, full_high, full_low, full_high)
        cy_blue = rand_band(margin, mid - margin, full_low, mid)
        cy_red  = rand_band(mid + margin // 2, IMG_SIZE - margin,
                            mid, full_high)
        cx_red = cx_blue = cx

    elif rel_label == 4:  # inside
        low  = margin + r_blue
        high = IMG_SIZE - margin - r_blue
        if high <= low + 1:
            low, high = full_low, full_high
        cx = np.random.randint(low, high)
        cy = np.random.randint(low, high)
        cx_red = cx_blue = cx
        cy_red = cy_blue = cy

    else:  # overlapping
        cx_red = np.random.randint(full_low, full_high)
        cy_red = np.random.randint(full_low, full_high)
        offset_x = np.random.randint(-5, 6)
        offset_y = np.random.randint(-5, 6)
        cx_blue = int(np.clip(cx_red + offset_x, full_low, full_high - 1))
        cy_blue = int(np.clip(cy_red + offset_y, full_low, full_high - 1))

    # ---------- draw shapes ----------
    draw_shape(img, shape_red,  cx_red,  cy_red,  r_red,  channel_idx=0)
    draw_shape(img, shape_blue, cx_blue, cy_blue, r_blue, channel_idx=1)

    img = np.clip(img, 0.0, 1.0)

    geom = np.array(
        [cx_red, cy_red, r_red, cx_blue, cy_blue, r_blue],
        dtype=np.float32,
    )
    return img, scale_label, geom


def generate_dataset(num_samples=NUM_SAMPLES, seed=0):
    np.random.seed(seed)
    random.seed(seed)

    scenes            = []
    rel_labels        = []
    scale_labels      = []
    shape_red_labels  = []
    shape_blue_labels = []
    geom_list         = []

    for _ in range(num_samples):
        rel     = np.random.randint(0, NUM_REL)
        scale   = np.random.randint(0, NUM_SCALE)
        shape_r = np.random.randint(0, NUM_SHAPE)
        shape_b = np.random.randint(0, NUM_SHAPE)

        img, scale, geom = generate_scene(rel, scale, shape_r, shape_b)

        scenes.append(img)
        rel_labels.append(rel)
        scale_labels.append(scale)
        shape_red_labels.append(shape_r)
        shape_blue_labels.append(shape_b)
        geom_list.append(geom)

    scenes            = torch.tensor(np.stack(scenes),            dtype=torch.float32)
    rel_labels        = torch.tensor(rel_labels,                  dtype=torch.long)
    scale_labels      = torch.tensor(scale_labels,                dtype=torch.long)
    shape_red_labels  = torch.tensor(shape_red_labels,            dtype=torch.long)
    shape_blue_labels = torch.tensor(shape_blue_labels,           dtype=torch.long)
    geom_targets      = torch.tensor(np.stack(geom_list),         dtype=torch.float32)

    return scenes, rel_labels, scale_labels, shape_red_labels, shape_blue_labels, geom_targets


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, in_channels=NUM_CHANNELS, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1)  # 16x16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)           # 8x8
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)           # 4x4
        self.fc    = nn.Linear(64 * 4 * 4, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z


class Decoder(nn.Module):
    def __init__(self, out_channels=NUM_CHANNELS, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc      = nn.Linear(latent_dim, 64 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 8x8
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)  # 16x16
        self.deconv3 = nn.ConvTranspose2d(16, out_channels, kernel_size=4, stride=2, padding=1)  # 32x32

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
        self.rel_head        = nn.Linear(LATENT_DIM, NUM_REL)
        self.scale_head      = nn.Linear(LATENT_DIM, NUM_SCALE)
        self.shape_red_head  = nn.Linear(LATENT_DIM, NUM_SHAPE)
        self.shape_blue_head = nn.Linear(LATENT_DIM, NUM_SHAPE)
        # NEW: geometry head (cx_r, cy_r, r_r, cx_b, cy_b, r_b)
        self.geom_head       = nn.Linear(LATENT_DIM, 6)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z           = self.encode(x)
        recon       = self.decode(z)
        rel_logits  = self.rel_head(z)
        scale_logits= self.scale_head(z)
        shape_r_log = self.shape_red_head(z)
        shape_b_log = self.shape_blue_head(z)
        geom_pred   = self.geom_head(z)
        return recon, z, rel_logits, scale_logits, shape_r_log, shape_b_log, geom_pred


# ---------------------------------------------------------------------
# Color helper for viz
# ---------------------------------------------------------------------

def tensor_to_rgb(img_tensor):
    """
    img_tensor: [3,H,W] in [0,1]
    Channel 0 -> red, channel 1 -> blue, channel 2 unused
    """
    x = img_tensor.detach().cpu().numpy()
    r = np.clip(x[0], 0.0, 1.0)
    b = np.clip(x[1], 0.0, 1.0)

    H, W = r.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    rgb[..., 0] = r
    rgb[..., 2] = b
    return rgb


# ---------------------------------------------------------------------
# Training + analysis
# ---------------------------------------------------------------------

def train():
    print(f"[basic32-geom] Using device: {DEVICE}")
    (
        scenes,
        rel_labels,
        scale_labels,
        shape_r,
        shape_b,
        geom_targets,
    ) = generate_dataset()

    dataset = TensorDataset(
        scenes, rel_labels, scale_labels, shape_r, shape_b, geom_targets
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SceneModel().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_recon = total_rel = total_scale = total_sr = total_sb = total_geom = 0.0
        correct_rel = correct_scale = correct_sr = correct_sb = 0
        total_samples = 0

        for x, rel, scale, sr, sb, geom in loader:
            x     = x.to(DEVICE)
            rel   = rel.to(DEVICE)
            scale = scale.to(DEVICE)
            sr    = sr.to(DEVICE)
            sb    = sb.to(DEVICE)
            geom  = geom.to(DEVICE)

            opt.zero_grad()
            recon, z, rel_logits, scale_logits, sr_logits, sb_logits, geom_pred = model(x)

            bce      = F.binary_cross_entropy(recon, x)
            rel_ce   = F.cross_entropy(rel_logits,   rel)
            scale_ce = F.cross_entropy(scale_logits, scale)
            sr_ce    = F.cross_entropy(sr_logits,    sr)
            sb_ce    = F.cross_entropy(sb_logits,    sb)
            geom_mse = F.mse_loss(geom_pred, geom)

            loss = bce + rel_ce + scale_ce + sr_ce + sb_ce + GEOM_LAMBDA * geom_mse
            loss.backward()
            opt.step()

            batch_size = x.size(0)
            total_samples += batch_size

            total_recon  += bce.item()      * batch_size
            total_rel    += rel_ce.item()   * batch_size
            total_scale  += scale_ce.item() * batch_size
            total_sr     += sr_ce.item()    * batch_size
            total_sb     += sb_ce.item()    * batch_size
            total_geom   += geom_mse.item() * batch_size

            correct_rel   += (rel_logits.argmax(dim=1)   == rel).sum().item()
            correct_scale += (scale_logits.argmax(dim=1) == scale).sum().item()
            correct_sr    += (sr_logits.argmax(dim=1)    == sr).sum().item()
            correct_sb    += (sb_logits.argmax(dim=1)    == sb).sum().item()

        if epoch % 50 == 0 or epoch == EPOCHS - 1:
            print(
                f"Epoch {epoch}/{EPOCHS-1} | "
                f"Recon: {total_recon/total_samples:.5f} | "
                f"RelCE: {total_rel/total_samples:.5f} | "
                f"ScaleCE: {total_scale/total_samples:.5f} | "
                f"GeomMSE: {total_geom/total_samples:.5f} | "
                f"Acc_rel: {100*correct_rel/total_samples:.2f}% | "
                f"Acc_scale: {100*correct_scale/total_samples:.2f}% | "
                f"Acc_shapeR: {100*correct_sr/total_samples:.2f}% | "
                f"Acc_shapeB: {100*correct_sb/total_samples:.2f}%"
            )

    # Save checkpoint
    ckpt = {
        "config": {
            "latent_dim": LATENT_DIM,
            "img_size": IMG_SIZE,
        },
        "model_state_dict": model.state_dict(),
    }
    torch.save(ckpt, CKPT_PATH)
    print(f"[basic32-geom] Saved checkpoint -> {CKPT_PATH}")

    # -----------------------------------------------------------------
    # Latent dump + basic analysis
    # -----------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        all_z = []
        all_rel = []
        all_scale = []
        all_sr = []
        all_sb = []
        all_geom = []

        for x, rel, scale, sr, sb, geom in loader:
            x = x.to(DEVICE)
            z = model.encode(x)
            all_z.append(z.cpu())
            all_rel.append(rel)
            all_scale.append(scale)
            all_sr.append(sr)
            all_sb.append(sb)
            all_geom.append(geom)

        all_z    = torch.cat(all_z,    dim=0)
        all_rel  = torch.cat(all_rel,  dim=0)
        all_scale= torch.cat(all_scale,dim=0)
        all_sr   = torch.cat(all_sr,   dim=0)
        all_sb   = torch.cat(all_sb,   dim=0)
        all_geom = torch.cat(all_geom, dim=0)

    np.savez(
        LATENTS_PATH,
        z=all_z.numpy(),
        rel=all_rel.numpy(),
        scale=all_scale.numpy(),
        shape_red=all_sr.numpy(),
        shape_blue=all_sb.numpy(),
        geom=all_geom.numpy(),
    )
    print(f"[basic32-geom] Saved latent dump -> {LATENTS_PATH}")

    # -----------------------------------------------------------------
    # Recon grid
    # -----------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        x, _, _, _, _, _ = next(iter(loader))
        x = x.to(DEVICE)[:16]
        z = model.encode(x)
        recon = model.decode(z)

    x      = x.cpu()
    recon  = recon.cpu()
    n_show = x.size(0)

    fig, axes = plt.subplots(2, n_show, figsize=(n_show * 1.2, 2.4), dpi=120)
    for i in range(n_show):
        axes[0, i].imshow(tensor_to_rgb(x[i]))
        axes[0, i].axis("off")
        axes[1, i].imshow(tensor_to_rgb(recon[i]))
        axes[1, i].axis("off")
    axes[0, 0].set_title("orig", fontsize=8)
    axes[1, 0].set_title("recon", fontsize=8)
    plt.tight_layout()
    plt.savefig(RECON_GRID_PATH)
    plt.close(fig)
    print(f"[basic32-geom] Saved recon grid -> {RECON_GRID_PATH}")

    # -----------------------------------------------------------------
    # t-SNE by relation (optional)
    # -----------------------------------------------------------------
    if HAVE_TSNE:
        print("[basic32-geom] Running t-SNE (this can take a bit)...")
        tsne = TSNE(n_components=2, perplexity=30, init="pca")
        emb = tsne.fit_transform(all_z.numpy())

        fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
        for rel_idx in range(NUM_REL):
            mask = (all_rel.numpy() == rel_idx)
            ax.scatter(
                emb[mask, 0],
                emb[mask, 1],
                s=6,
                label=f"rel={rel_idx}:{RELATION_NAMES[rel_idx]}",
            )
        ax.legend(loc="best", fontsize=6)
        ax.set_title("t-SNE of latents by relation (basic32-geom)")
        plt.tight_layout()
        plt.savefig(TSNE_REL_PATH)
        plt.close(fig)
        print(f"[basic32-geom] Saved t-SNE -> {TSNE_REL_PATH}")
    else:
        print("[basic32-geom] sklearn.manifold.TSNE not available; skipping t-SNE.")


if __name__ == "__main__":
    train()
