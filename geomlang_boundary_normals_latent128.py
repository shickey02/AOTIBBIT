#!/usr/bin/env python3
# geomlang_boundary_normals_latent128.py
#
# Measure how decision-boundary normals sit relative to the learned
# LR / AB generators for the 64x64, latent=128 edges_relscale model.

import os, math, random, sys
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# -----------------------
# Config
# -----------------------
IMG_SIZE   = 64
NUM_CH     = 3      # red, blue, edges
LATENT_DIM = 128

N_SAMPLES      = 6000     # synthetic scenes to probe geometry
N_BASES        = 300      # how many base points to try for boundaries
MAX_T          = 3.0      # max |t| along ±g_x / ±g_y when searching
N_STEPS_SCAN   = 60       # resolution of the boundary scan

OUT_DIR         = "outputs_edges_relscale"
CKPT_SCENEMODEL = os.path.join(OUT_DIR, "scene_model_edges_relscale.pt")
os.makedirs(OUT_DIR, exist_ok=True)

REL_NAMES = ["left_of", "right_of", "above", "below", "overlap"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[normal128] Using device: {device}")

# -----------------------
# Synthetic scene generator (same family as training)
# -----------------------

def draw_disk(img, cx, cy, r, chan, val=1.0, edge_val=1.0):
    """Draw a filled disk + edge ring into a (3,H,W) numpy image."""
    H, W = img.shape[1], img.shape[2]
    yy, xx = np.ogrid[:H, :W]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    inside = dist2 <= r * r
    img[chan][inside] = val

    # crude 2-px edge
    edge_mask = (dist2 >= (r - 1) ** 2) & (dist2 <= (r + 1) ** 2)
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
    """
    H = W = img_size
    img = np.zeros((3, H, W), dtype=np.float32)

    shape_r = random.choice(["disk", "square"])
    shape_b = random.choice(["disk", "square"])

    # radii chosen for 64x64 canvas
    r1 = random.randint(6, 10)
    r2 = random.randint(6, 10)

    rel_id = random.randint(0, 4)

    cx = W * 0.35 + random.uniform(-2, 2)
    cy = H * 0.50 + random.uniform(-2, 2)
    dx = W * 0.20
    dy = H * 0.20

    if rel_id == 0:      # left_of
        c1 = (cx - dx, cy)
        c2 = (cx + dx, cy)
    elif rel_id == 1:    # right_of
        c1 = (cx + dx, cy)
        c2 = (cx - dx, cy)
    elif rel_id == 2:    # above
        c1 = (cx, cy - dy)
        c2 = (cx, cy + dy)
    elif rel_id == 3:    # below
        c1 = (cx, cy + dy)
        c2 = (cx, cy - dy)
    else:                # overlap
        c1 = (cx, cy)
        c2 = (cx + random.uniform(-4, 4),
              cy + random.uniform(-4, 4))

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
    imgs = torch.from_numpy(np.stack(imgs, axis=0))   # (N,3,H,W)
    rels = torch.tensor(rels, dtype=torch.long)
    return imgs, rels


# -----------------------
# Model
# -----------------------

# Make sure we can import the model from this folder
THIS_DIR = os.path.dirname(__file__)
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)

# IMPORTANT: adjust this import to match your 128-latent model definition.
# Use the same import you used in geomlang_lie_commutator_stats_generic.py.
from geomlang_lie_group_latent128 import SceneModelEdges128 as SceneModelEdges



def load_scene_model():
    print(f"[normal128] Loading SceneModel from {CKPT_SCENEMODEL}")
    ckpt = torch.load(CKPT_SCENEMODEL, map_location=device)
    model = SceneModelEdges().to(device)
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
    """Global LR / AB generators from relation-wise means."""
    z = z.detach().cpu()
    rels = rels.cpu()

    mus = []
    for k in range(5):
        mask = (rels == k)
        mus.append(z[mask].mean(dim=0))
    mu = torch.stack(mus, dim=0)  # (5,D)

    v_LR = mu[1] - mu[0]   # right_of - left_of
    v_AB = mu[3] - mu[2]   # below - above

    print(f"[normal128] ||mu_right - mu_left|| = {v_LR.norm().item():.4f}")
    print(f"[normal128] ||mu_below - mu_above|| = {v_AB.norm().item():.4f}")

    return v_LR.to(device), v_AB.to(device)


def angle_deg(u, v):
    """Angle between two torch vectors in degrees."""
    u = u / (u.norm() + 1e-8)
    v = v / (v.norm() + 1e-8)
    cos = torch.clamp(torch.dot(u, v), -1.0, 1.0)
    return float(torch.rad2deg(torch.acos(cos)))


def search_boundary_along_dir(model, z0, y0, d, t_max=MAX_T, n_steps=N_STEPS_SCAN):
    """
    1D scan along direction d starting at z0 until relation changes.
    Returns (t*, z*, y1) or None if no crossing in [-t_max, t_max].
    We scan both + and - inside this helper.
    """
    ts = torch.linspace(-t_max, t_max, 2 * n_steps + 1, device=device)
    z0 = z0.to(device)

    with torch.no_grad():
        z_grid = z0[None, :] + ts[:, None] * d[None, :]
        logits = model.rel_head(z_grid)
        preds = logits.argmax(dim=1)

    # find first index with different label, closest in |t|
    diff = (preds != y0)
    if not diff.any():
        return None

    idxs = torch.nonzero(diff, as_tuple=False).squeeze(1)
    # choose the t with smallest |t|
    best = idxs[torch.argmin(ts[idxs].abs())].item()
    t_star = ts[best].item()
    z_star = z_grid[best]
    y1 = preds[best].item()
    return t_star, z_star, y1


def estimate_normal(model, z_star, y_a, y_b):
    """
    Gradient-based estimate of decision-boundary normal at z_star,
    for the boundary between classes y_a and y_b.
    """
    z_star = z_star.detach().to(device).requires_grad_(True)
    logits = model.rel_head(z_star[None, :])[0]
    margin = logits[y_a] - logits[y_b]
    model.zero_grad(set_to_none=True)
    margin.backward()
    n = z_star.grad.detach()
    return n / (n.norm() + 1e-8)


# -----------------------
# Main
# -----------------------

def main():
    model = load_scene_model()

    print("[normal128] Generating synthetic dataset...")
    imgs, rels = generate_dataset(N_SAMPLES)
    imgs = imgs.to(device)
    rels = rels.to(device)

    print("[normal128] Encoding dataset to latent...")
    with torch.no_grad():
        z = model.encode(imgs)  # (N,D)
    print(f"[normal128] Latent shape: {z.shape}")

    v_LR, v_AB = estimate_generators(z, rels)
    g_x = v_LR / v_LR.norm()
    g_y = v_AB / v_AB.norm()

    # pick random base indices to probe
    all_indices = torch.randperm(N_SAMPLES)[:N_BASES].tolist()
    angles_x = []
    angles_y = []
    closer_to_x = 0
    closer_to_y = 0

    print(f"[normal128] Searching for boundaries over {N_BASES} bases...")

    for idx in all_indices:
        z0 = z[idx]
        y0 = rels[idx].item()

        for axis_name, d in [("gx", g_x), ("gy", g_y)]:
            out = search_boundary_along_dir(model, z0, y0, d)
            if out is None:
                continue

            t_star, z_star, y1 = out
            # only consider true label flips
            if y1 == y0:
                continue

            n = estimate_normal(model, z_star, y0, y1)
            ax = angle_deg(n, g_x)
            ay = angle_deg(n, g_y)
            angles_x.append(ax)
            angles_y.append(ay)

            if ax < ay:
                closer_to_x += 1
            else:
                closer_to_y += 1

            print(
                f"[normal128] boundary at idx={idx}, axis={axis_name}, "
                f"{REL_NAMES[y0]}↔{REL_NAMES[y1]}, t*={t_star:+.3f}, "
                f"angle_x={ax:.1f}°, angle_y={ay:.1f}°"
            )

    angles_x = np.array(angles_x)
    angles_y = np.array(angles_y)
    print(f"[normal128] Collected {len(angles_x)} boundary normals.")

    if len(angles_x) == 0:
        print("[normal128] No boundaries found in scan range – nothing to plot.")
        return

    print("[normal128] angle to g_x stats:")
    print(f"   mean = {angles_x.mean():.2f}°")
    print(f"   std  = {angles_x.std():.2f}°")
    print(f"   min  = {angles_x.min():.2f}°")
    print(f"   max  = {angles_x.max():.2f}°")

    print("[normal128] angle to g_y stats:")
    print(f"   mean = {angles_y.mean():.2f}°")
    print(f"   std  = {angles_y.std():.2f}°")
    print(f"   min  = {angles_y.min():.2f}°")
    print(f"   max  = {angles_y.max():.2f}°")

    print(f"[normal128] boundaries closer to g_x: {closer_to_x} / {len(angles_x)}")
    print(f"[normal128] boundaries closer to g_y: {closer_to_y} / {len(angles_x)}")

    # histogram plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(angles_x, bins=12)
    axes[0].set_title("Angle normal vs g_x (LR)")
    axes[0].set_xlabel("degrees")
    axes[0].set_ylabel("count")

    axes[1].hist(angles_y, bins=12)
    axes[1].set_title("Angle normal vs g_y (AB)")
    axes[1].set_xlabel("degrees")
    axes[1].set_ylabel("count")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "boundary_normal_angles_latent128.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[normal128] Saved histogram -> {out_path}")


if __name__ == "__main__":
    main()
