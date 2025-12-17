#!/usr/bin/env python3
# geomlang_cross_manifold_latent256.py
#
# Cross–manifold charts for LR / AB generators.
# For the 256-latent, 256x256, 3-channel edges model.

import os, math, random
import numpy as np
from PIL import Image

import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

# -----------------------
# Config
# -----------------------
IMG_SIZE   = 64
NUM_CH     = 3     # red, blue, edges
LATENT_DIM = 256

N_SAMPLES  = 6000
BATCH_SIZE = 256

OUT_DIR         = "outputs_edges_relscale256"
CKPT_SCENEMODEL = os.path.join(OUT_DIR, "scene_model_edges_relscale256.pt")
os.makedirs(OUT_DIR, exist_ok=True)

REL_NAMES = ["left_of", "right_of", "above", "below", "overlap"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[cross256] Using device: {device}")


# -----------------------
# Tiny geometry helper
# -----------------------

def norm3(a):
    return math.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])


# -----------------------
# Synthetic scene generator (same family as training)
# -----------------------

def draw_disk(img, cx, cy, r, chan, val=1.0, edge_val=1.0):
    """Draw a filled disk into 'chan' plus edge into channel 2."""
    H, W = img.shape[1], img.shape[2]
    yy, xx = np.ogrid[:H, :W]
    dist2 = (xx - cx)**2 + (yy - cy)**2
    inside = dist2 <= r*r
    img[chan][inside] = val

    # crude edge: ring between r-1 and r+1
    edge_mask = (dist2 >= (r-1)**2) & (dist2 <= (r+1)**2)
    img[2][edge_mask] = edge_val


def draw_square(img, cx, cy, s, chan, val=1.0, edge_val=1.0):
    """Axis-aligned square, center (cx,cy), half-size s."""
    H, W = img.shape[1], img.shape[2]
    x0 = max(0, int(cx - s))
    x1 = min(W, int(cx + s))
    y0 = max(0, int(cy - s))
    y1 = min(H, int(cy + s))
    img[chan, y0:y1, x0:x1] = val
    # edges
    img[2, y0:y1, x0:x0+1] = edge_val
    img[2, y0:y1, x1-1:x1] = edge_val
    img[2, y0:y0+1, x0:x1] = edge_val
    img[2, y1-1:y1, x0:x1] = edge_val


def sample_scene(img_size=IMG_SIZE):
    """
    Returns:
      img: (3,H,W) float32
      rel_id: int in [0..4]
    Two shapes with relation in {left_of, right_of, above, below, overlap}.
    """
    H = W = img_size
    img = np.zeros((3, H, W), dtype=np.float32)

    # random choose shape types
    shape_r = random.choice(["disk", "square"])
    shape_b = random.choice(["disk", "square"])

    # base sizes
    r1 = random.randint(10, 20)
    r2 = random.randint(10, 20)

    # random relation
    rel_id = random.randint(0, 4)

    # base positions roughly in middle
    cx = W * 0.35 + random.uniform(-5, 5)
    cy = H * 0.5  + random.uniform(-5, 5)
    dx = W * 0.20
    dy = H * 0.20

    if rel_id == 0:     # left_of
        c1 = (cx - dx, cy)   # red left
        c2 = (cx + dx, cy)   # blue right
    elif rel_id == 1:   # right_of
        c1 = (cx + dx, cy)
        c2 = (cx - dx, cy)
    elif rel_id == 2:   # above
        c1 = (cx, cy - dy)
        c2 = (cx, cy + dy)
    elif rel_id == 3:   # below
        c1 = (cx, cy + dy)
        c2 = (cx, cy - dy)
    else:               # overlap
        c1 = (cx, cy)
        c2 = (cx + random.uniform(-10, 10),
              cy + random.uniform(-10, 10))

    # draw shapes
    if shape_r == "disk":
        draw_disk(img, c1[0], c1[1], r1, chan=0)
    else:
        draw_square(img, c1[0], c1[1], r1, chan=0)

    if shape_b == "disk":
        draw_disk(img, c2[0], c2[1], r2, chan=1)
    else:
        draw_square(img, c2[0], c2[1], r2, chan=1)

    return img, rel_id


def generate_dataset(n_samples):
    imgs = []
    rels = []
    for _ in range(n_samples):
        img, rel_id = sample_scene()
        imgs.append(img)
        rels.append(rel_id)
    imgs = torch.from_numpy(np.stack(imgs, axis=0))  # (N,3,H,W)
    rels = torch.tensor(rels, dtype=torch.long)
    return imgs, rels


# -----------------------
# Model (reuse the one that matches the 256×256 checkpoint)
# -----------------------

import sys

# Make sure we can import other scripts from this folder
THIS_DIR = os.path.dirname(__file__)
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)

from geomlang_lie_group_latent256 import SceneModelEdges256



def load_scene_model():
    print(f"[cross256] Loading SceneModel from {CKPT_SCENEMODEL}")
    ckpt = torch.load(CKPT_SCENEMODEL, map_location=device)
    model = SceneModelEdges256().to(device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model


# -----------------------
# Geometry helpers
# -----------------------

def estimate_generators(z, rels):
    """Estimate global LR and AB generators via relation-wise means."""
    z = z.detach().cpu()
    rels = rels.cpu()

    mu = []
    for k in range(5):
        mask = (rels == k)
        mu_k = z[mask].mean(dim=0)
        mu.append(mu_k)
    mu = torch.stack(mu, dim=0)  # (5, D)

    v_LR = mu[1] - mu[0]   # right_of - left_of
    v_AB = mu[3] - mu[2]   # below - above

    print(f"[cross256] ||mu_right - mu_left|| = {v_LR.norm().item():.4f}")
    print(f"[cross256] ||mu_below - mu_above|| = {v_AB.norm().item():.4f}")

    return v_LR.to(device), v_AB.to(device)


def to_rgb(img_tensor):
    """(3,H,W) -> (H,W,3) numpy in [0,1]"""
    x = img_tensor.detach().cpu().numpy()
    x = np.clip(x, 0.0, 1.0)
    x = np.transpose(x, (1, 2, 0))
    return x


def chart_for_base(model, z0, g_x, g_y, rel_name, base_idx, alphas, betas, out_path):
    """
    Make a |alphas| x |betas| grid of z0 + α g_x + β g_y.
    Title: base relation & index.
    """
    n_a = len(alphas)
    n_b = len(betas)

    fig, axes = plt.subplots(n_a, n_b, figsize=(1.8*n_b, 1.8*n_a))
    fig.suptitle(
        f"Cross-manifold chart around base idx={base_idx} (rel={rel_name}, latent256)\n"
        "rows: α·g_x (LR), cols: β·g_y (AB)", fontsize=10
    )

    z0 = z0.to(device)

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            z = z0 + alpha * g_x + beta * g_y
            with torch.no_grad():
                img = model.decode(z.unsqueeze(0))[0]  # (3,H,W)
                rel_logits = model.rel_head(z.unsqueeze(0))
                rel_hat = rel_logits.argmax(dim=1).item()

            ax = axes[i, j]
            ax.imshow(to_rgb(img))
            ax.axis("off")
            ax.set_title(
                f"α={alpha:+.1f}\nβ={beta:+.1f}\n{REL_NAMES[rel_hat]}",
                fontsize=6
            )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[cross256] Saved chart -> {out_path}")


# -----------------------
# Main
# -----------------------

def main():
    # 1) Load the trained 256×256 scene model
    model = load_scene_model()

    # 2) Generate synthetic dataset on CPU
    print(f"[cross256] Generating dataset: N={N_SAMPLES}")
    imgs, rels = generate_dataset(N_SAMPLES)   # imgs: (N,3,H,W) on CPU
    rels = rels.to(device)

    # 3) Encode images in small batches to avoid CUDA OOM
    model.eval()
    zs = []
    BATCH_ENC = 64   # you can try 128 if memory allows

    with torch.no_grad():
        for start in range(0, N_SAMPLES, BATCH_ENC):
            end = min(start + BATCH_ENC, N_SAMPLES)
            batch = imgs[start:end].to(device)      # move only this chunk
            z_batch = model.encode(batch)           # (b, D)
            zs.append(z_batch.cpu())                # keep z on CPU for now

    z = torch.cat(zs, dim=0).to(device)             # (N, D) back on GPU for math

    # 4) Estimate global LR / AB generators from the latents
    v_LR, v_AB = estimate_generators(z, rels)

    # Normalise generators so that |g_x| = |g_y| = 1
    g_x = v_LR / v_LR.norm()
    g_y = v_AB / v_AB.norm()

    # 5) Choose one base index per relation
    base_indices = []
    for k in range(5):
        idxs = (rels == k).nonzero(as_tuple=True)[0]
        if len(idxs) == 0:
            continue
        base_indices.append((k, idxs[0].item()))

    print("[cross256] Base indices per relation:", base_indices)

    # 6) Define chart coordinates (α, β) grid
    alphas = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    betas  = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]

    # 7) Generate cross–manifold charts per base point
    for rel_id, base_idx in base_indices:
        z0 = z[base_idx]
        rel_name = REL_NAMES[rel_id]
        out_path = os.path.join(
            OUT_DIR,
            f"cross_chart_rel{rel_id}_{rel_name}_idx{base_idx}_latent256.png"
        )
        chart_for_base(model, z0, g_x, g_y, rel_name, base_idx, alphas, betas, out_path)

    print("[cross256] Done.")


if __name__ == "__main__":
    main()
