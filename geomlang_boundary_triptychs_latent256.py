#!/usr/bin/env python3
# geomlang_boundary_triptychs_latent256.py
#
# Find a few genuine decision-boundary crossings along ±g_x / ±g_y
# and save triptychs: base -> boundary -> past-boundary.

import os, math, random
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# -----------------------
# Config
# -----------------------
IMG_SIZE   = 64
NUM_CH     = 3
LATENT_DIM = 256

N_SAMPLES  = 6000
OUT_DIR = "outputs_edges_relscale256"
CKPT_SCENEMODEL = os.path.join(OUT_DIR, "scene_model_edges_relscale256.pt")
os.makedirs(OUT_DIR, exist_ok=True)

REL_NAMES = ["left_of", "right_of", "above", "below", "overlap"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[trip256] Using device: {device}")

# -----------------------
# Synthetic scenes (same family as training)
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

    cx = W * 0.35 + random.uniform(-2, 2)
    cy = H * 0.5  + random.uniform(-2, 2)
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
# Model load
# -----------------------

import sys
THIS_DIR = os.path.dirname(__file__)
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)

from geomlang_lie_group_latent256 import SceneModelEdges256

def load_scene_model():
    print(f"[trip256] Loading SceneModel from {CKPT_SCENEMODEL}")
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
    z = z.detach().cpu()
    rels = rels.cpu()
    mus = []
    for k in range(5):
        mask = (rels == k)
        mus.append(z[mask].mean(dim=0))
    mus = torch.stack(mus, dim=0)
    v_LR = mus[1] - mus[0]   # right_of - left_of
    v_AB = mus[3] - mus[2]   # below - above
    print(f"[trip256] ||mu_right - mu_left|| = {v_LR.norm().item():.4f}")
    print(f"[trip256] ||mu_below - mu_above|| = {v_AB.norm().item():.4f}")
    return v_LR.to(device), v_AB.to(device)

def to_rgb(img_tensor):
    x = img_tensor.detach().cpu().numpy()
    x = np.clip(x, 0.0, 1.0)
    x = np.transpose(x, (1, 2, 0))
    return x

# -----------------------
# Boundary search and triptych
# -----------------------

def find_boundary_along(model, z0, base_rel, g_dir, t_max=6.0, n_steps=60):
    """
    Scan along +g_dir, return first t>0 where relation label changes.
    Coarse scan only; good enough for visualization.
    """
    ts = np.linspace(0.0, t_max, n_steps+1)[1:]
    with torch.no_grad():
        for t in ts:
            z_t = z0 + t * g_dir
            logits = model.rel_head(z_t.unsqueeze(0))
            rel_hat = logits.argmax(dim=1).item()
            if rel_hat != base_rel:
                return t, rel_hat
    return None, base_rel

def save_triptych(model, z0, g_dir, t_boundary, base_rel, new_rel,
                  axis_name, sign_name, base_idx, out_dir):
    """
    Save a 1x3 image: base, at boundary, and slightly past.
    """
    dt = 0.2  # step beyond boundary for visualization
    zs = [
        z0,
        z0 + t_boundary * g_dir,
        z0 + (t_boundary + dt) * g_dir,
    ]
    titles = [
        f"base\n{REL_NAMES[base_rel]}",
        f"boundary t={t_boundary:.2f}\n{REL_NAMES[new_rel]}",
        f"past t={t_boundary+dt:.2f}\n{REL_NAMES[new_rel]}",
    ]

    imgs = []
    with torch.no_grad():
        for z in zs:
            img = model.decode(z.unsqueeze(0))[0]
            imgs.append(to_rgb(img))

    fig, axes = plt.subplots(1, 3, figsize=(6, 2))
    for ax, im, ttl in zip(axes, imgs, titles):
        ax.imshow(im)
        ax.axis("off")
        ax.set_title(ttl, fontsize=8)

    fig.suptitle(
        f"Boundary crossing (axis={axis_name}, sign={sign_name}, base_idx={base_idx})",
        fontsize=10
    )
    fname = os.path.join(
        out_dir,
        f"boundary_triptych_idx{base_idx}_{axis_name}_{sign_name}.png"
    )
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[trip256] Saved triptych -> {fname}")

# -----------------------
# Main
# -----------------------

def main():
    model = load_scene_model()

    print("[trip256] Generating dataset...")
    imgs, rels = generate_dataset(N_SAMPLES)
    imgs = imgs.to(device)
    rels = rels.to(device)

    print("[trip256] Encoding dataset...")
    with torch.no_grad():
        z = model.encode(imgs)

    v_LR, v_AB = estimate_generators(z, rels)
    g_x = v_LR / v_LR.norm()
    g_y = v_AB / v_AB.norm()

    directions = [
        ("gx", "+", g_x),
        ("gx", "-", -g_x),
        ("gy", "+", g_y),
        ("gy", "-", -g_y),
    ]

    # We’ll collect a few nice crossings
    max_triptychs = 5
    made = 0

    N = z.shape[0]
    indices = list(range(N))
    random.shuffle(indices)

    for base_idx in indices:
        if made >= max_triptychs:
            break

        z0 = z[base_idx]
        base_rel = rels[base_idx].item()

        for axis_name, sign_name, g_dir in directions:
            t_boundary, new_rel = find_boundary_along(
                model, z0, base_rel, g_dir,
                t_max=6.0, n_steps=60
            )
            if t_boundary is not None:
                save_triptych(
                    model, z0, g_dir, t_boundary,
                    base_rel, new_rel,
                    axis_name, sign_name, base_idx,
                    OUT_DIR
                )
                made += 1
                if made >= max_triptychs:
                    break

    if made == 0:
        print("[trip256] No boundaries found within scan range.")
    else:
        print(f"[trip256] Done, created {made} triptych image(s).")

if __name__ == "__main__":
    main()
