#!/usr/bin/env python3
# geomlang_decision_boundary_search_latent256.py
#
# For the 256-dim latent, 64x64, 3-channel edges model:
#   - generate a dataset
#   - encode to latent
#   - estimate global LR / AB generators
#   - for many base points, search along ±g_x and ±g_y until the
#     relation label changes
#   - refine boundary with bisection
#   - report statistics on distances to the nearest decision boundary.

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
IMG_SIZE   = 64        # matches SceneModelEdges256 checkpoint
NUM_CH     = 3         # red, blue, edges
LATENT_DIM = 256

N_SAMPLES  = 6000
BATCH_SIZE = 256

OUT_DIR         = "outputs_edges_relscale256"
CKPT_SCENEMODEL = os.path.join(OUT_DIR, "scene_model_edges_relscale256.pt")
os.makedirs(OUT_DIR, exist_ok=True)

REL_NAMES = ["left_of", "right_of", "above", "below", "overlap"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[bound-search256] Using device: {device}")

# -----------------------
# Synthetic scene generator
# (same family as we used for the 256-latent experiments)
# -----------------------

def draw_disk(img, cx, cy, r, chan, val=1.0, edge_val=1.0):
    H, W = img.shape[1], img.shape[2]
    yy, xx = np.ogrid[:H, :W]
    dist2 = (xx - cx)**2 + (yy - cy)**2
    inside = dist2 <= r*r
    img[chan][inside] = val

    edge_mask = (dist2 >= (r-1)**2) & (dist2 <= (r+1)**2)
    img[2][edge_mask] = edge_val

def draw_square(img, cx, cy, s, chan, val=1.0, edge_val=1.0):
    H, W = img.shape[1], img.shape[2]
    x0 = max(0, int(cx - s))
    x1 = min(W, int(cx + s))
    y0 = max(0, int(cy - s))
    y1 = min(H, int(cy + s))
    img[chan, y0:y1, x0:x1] = val
    img[2, y0:y1, x0:x0+1] = edge_val
    img[2, y0:y1, x1-1:x1] = edge_val
    img[2, y0:y0+1, x0:x1] = edge_val
    img[2, y1-1:y1, x0:x1] = edge_val

def sample_scene(img_size=IMG_SIZE):
    H = W = img_size
    img = np.zeros((3, H, W), dtype=np.float32)

    shape_r = random.choice(["disk", "square"])
    shape_b = random.choice(["disk", "square"])

    r1 = random.randint(5, 10)
    r2 = random.randint(5, 10)

    rel_id = random.randint(0, 4)

    cx = W * 0.35 + random.uniform(-3, 3)
    cy = H * 0.5  + random.uniform(-3, 3)
    dx = W * 0.20
    dy = H * 0.20

    if rel_id == 0:       # left_of
        c1 = (cx - dx, cy)
        c2 = (cx + dx, cy)
    elif rel_id == 1:     # right_of
        c1 = (cx + dx, cy)
        c2 = (cx - dx, cy)
    elif rel_id == 2:     # above
        c1 = (cx, cy - dy)
        c2 = (cx, cy + dy)
    elif rel_id == 3:     # below
        c1 = (cx, cy + dy)
        c2 = (cx, cy - dy)
    else:                 # overlap
        c1 = (cx, cy)
        c2 = (cx + random.uniform(-5, 5),
              cy + random.uniform(-5, 5))

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
# Import model
# -----------------------
THIS_DIR = os.path.dirname(__file__)
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)

from geomlang_lie_group_latent256 import SceneModelEdges256

def load_scene_model():
    print(f"[bound-search256] Loading SceneModel from {CKPT_SCENEMODEL}")
    ckpt = torch.load(CKPT_SCENEMODEL, map_location=device)
    model = SceneModelEdges256().to(device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model

# -----------------------
# Helpers
# -----------------------

def encode_dataset(model, imgs):
    """Encode all images to latent (N,D) and return z, rels."""
    N = imgs.shape[0]
    zs = []
    with torch.no_grad():
        for i in range(0, N, BATCH_SIZE):
            batch = imgs[i:i+BATCH_SIZE].to(device)
            z_batch = model.encode(batch)
            zs.append(z_batch.cpu())
    z = torch.cat(zs, dim=0)  # (N,D)
    return z

def estimate_generators(z, rels):
    """Estimate global LR/AB generators via class means."""
    z = z.detach().cpu()
    rels = rels.cpu()

    mu = []
    for k in range(5):
        mask = (rels == k)
        mu_k = z[mask].mean(dim=0)
        mu.append(mu_k)
    mu = torch.stack(mu, dim=0)  # (5,D)

    v_LR = mu[1] - mu[0]   # right_of - left_of
    v_AB = mu[3] - mu[2]   # below - above

    print(f"[bound-search256] ||mu_right - mu_left|| = {v_LR.norm().item():.4f}")
    print(f"[bound-search256] ||mu_below - mu_above|| = {v_AB.norm().item():.4f}")

    g_x = v_LR / v_LR.norm()
    g_y = v_AB / v_AB.norm()
    return g_x.to(device), g_y.to(device)

def classify_latent(model, z):
    """z: (N,D) -> predicted relation ids."""
    with torch.no_grad():
        logits = model.rel_head(z)
        return logits.argmax(dim=1)

def search_boundary_1d(model, z0, d, rel0, step=0.3, max_t=6.0, n_bisect=10):
    """
    Starting at z0 (class rel0), march along direction d until label changes
    or max_t is reached. Returns (t_boundary or None).
    """
    z0 = z0.to(device)
    d = d.to(device)

    # Ensure d is unit length
    d = d / (d.norm() + 1e-9)

    # march forward
    t = step
    prev_t = 0.0
    prev_label = rel0

    while t <= max_t:
        z = z0.unsqueeze(0) + t * d.unsqueeze(0)
        label = classify_latent(model, z)[0].item()
        if label != rel0:
            # bracket [prev_t, t]
            a, b = prev_t, t
            for _ in range(n_bisect):
                m = 0.5 * (a + b)
                zm = z0.unsqueeze(0) + m * d.unsqueeze(0)
                lab_m = classify_latent(model, zm)[0].item()
                if lab_m == rel0:
                    a = m
                else:
                    b = m
            return abs(b)  # approximate boundary distance
        prev_t = t
        t += step

    # no boundary found
    return None

# -----------------------
# Main
# -----------------------

def main():
    model = load_scene_model()

    print("[bound-search256] Generating dataset...")
    imgs, rels = generate_dataset(N_SAMPLES)
    print("[bound-search256] Encoding dataset to latent...")
    z = encode_dataset(model, imgs)

    print(f"[bound-search256] Latent shape: {z.shape}")
    g_x, g_y = estimate_generators(z, rels)

    # choose random base points from the cloud
    N_BASE = 300
    idxs = torch.randint(low=0, high=z.shape[0], size=(N_BASE,))
    z_bases = z[idxs]
    rel_bases = rels[idxs]

    dists = []  # distances to nearest boundary over all directions

    for i in range(N_BASE):
        z0 = z_bases[i]
        rel0 = rel_bases[i].item()

        # directions: +g_x, -g_x, +g_y, -g_y
        dirs = [g_x, -g_x, g_y, -g_y]
        best = None

        for d in dirs:
            t_boundary = search_boundary_1d(model, z0, d, rel0)
            if t_boundary is not None:
                if best is None or t_boundary < best:
                    best = t_boundary

        if best is not None:
            dists.append(best)

    dists = np.array(dists, dtype=np.float32)
    print(f"[bound-search256] Found {len(dists)} boundaries out of {N_BASE} bases.")

    if len(dists) == 0:
        print("[bound-search256] No boundaries found within search radius.")
        return

    print("[bound-search256] Boundary distance stats (in generator units):")
    print(f"  mean = {dists.mean():.3f}")
    print(f"  std  = {dists.std():.3f}")
    print(f"  min  = {dists.min():.3f}")
    print(f"  max  = {dists.max():.3f}")

    # --- Filtered stats: ignore points extremely close to a boundary ---
    thr = 0.1  # "epsilon" margin; adjust if you like
    mask = dists > thr

    if mask.any():
        d_far = dists[mask]
        print(f"[bound-search256] Filtered stats (dist > {thr:.2f}):")
        print(f"  count = {len(d_far)}")
        print(f"  mean  = {d_far.mean():.3f}")
        print(f"  std   = {d_far.std():.3f}")
        print(f"  min   = {d_far.min():.3f}")
        print(f"  max   = {d_far.max():.3f}")
    else:
        print(f"[bound-search256] No distances greater than {thr:.2f}")


    # histogram
    plt.figure(figsize=(6,4))
    plt.hist(dists, bins=30)
    plt.xlabel("distance to nearest boundary along ±g_x/±g_y")
    plt.ylabel("count")
    plt.title("Decision-boundary distances (latent256)")
    out_path = os.path.join(OUT_DIR, "decision_boundary_distances_latent256.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[bound-search256] Saved histogram -> {out_path}")

if __name__ == "__main__":
    main()
