#!/usr/bin/env python3
# geomlang_boundary_normals_latent256.py
#
# Estimate local decision-boundary normals in latent space and
# compare them to the LR / AB generators for the 256-dim latent,
# 64x64x3 edges model.

import os, math, random, sys
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# -----------------------
# Config
# -----------------------
IMG_SIZE   = 64          # important: matches training / checkpoint (4096 = 64*64)
NUM_CH     = 3           # red, blue, edges
LATENT_DIM = 256

N_SAMPLES       = 6000   # synthetic scenes to generate
BATCH_SIZE      = 256
N_BASE_SAMPLES  = 300    # how many base points to try for boundary search
MAX_BOUNDARIES  = 40     # stop after we have this many normals

OUT_DIR         = "outputs_edges_relscale256"
CKPT_SCENEMODEL = os.path.join(OUT_DIR, "scene_model_edges_relscale256.pt")

os.makedirs(OUT_DIR, exist_ok=True)

REL_NAMES = ["left_of", "right_of", "above", "below", "overlap"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[normal256] Using device: {device}")

# Make sure we can import the model definition
THIS_DIR = os.path.dirname(__file__)
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)

from geomlang_lie_group_latent256 import SceneModelEdges256


# -----------------------
# Simple drawing utilities (same style as other scripts)
# -----------------------

def draw_disk(img, cx, cy, r, chan, val=1.0, edge_val=1.0):
    """Draw a filled disk into 'chan' plus edge into channel 2."""
    H, W = img.shape[1], img.shape[2]
    yy, xx = np.ogrid[:H, :W]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    inside = dist2 <= r * r
    img[chan][inside] = val

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
    Two shapes with relation in {left_of, right_of, above, below, overlap}.
    """
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
    imgs = torch.from_numpy(np.stack(imgs, axis=0))  # (N,3,H,W)
    rels = torch.tensor(rels, dtype=torch.long)
    return imgs, rels


# -----------------------
# Model loading
# -----------------------

def load_scene_model():
    print(f"[normal256] Loading SceneModel from {CKPT_SCENEMODEL}")
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

    print(f"[normal256] ||mu_right - mu_left|| = {v_LR.norm().item():.4f}")
    print(f"[normal256] ||mu_below - mu_above|| = {v_AB.norm().item():.4f}")

    v_LR = v_LR.to(device)
    v_AB = v_AB.to(device)

    g_x = v_LR / v_LR.norm()
    g_y = v_AB / v_AB.norm()
    return g_x, g_y


def rel_logits_from_latent(model, z_batch):
    """Helper: run rel_head starting from latent z."""
    with torch.no_grad():
        logits = model.rel_head(z_batch)
    return logits


def find_boundary_along_direction(model, z0, base_label, direction,
                                  t_min=-3.0, t_max=3.0, n_steps=25):
    """
    Scan along z(t) = z0 + t * direction for a label change.
    If found, refine with binary search.
    Returns (t_star, alt_label) or (None, None) if no crossing in range.
    """
    direction = direction / direction.norm()
    ts = torch.linspace(t_min, t_max, steps=n_steps, device=device)
    zs = z0.unsqueeze(0) + ts.unsqueeze(1) * direction.unsqueeze(0)
    logits = rel_logits_from_latent(model, zs)
    preds = logits.argmax(dim=1)

    # Look for any adjacent pair with label change
    change_idx = None
    for i in range(len(ts) - 1):
        if preds[i].item() != preds[i+1].item():
            change_idx = i
            break

    if change_idx is None:
        return None, None

    # bracket [t_lo, t_hi] where label changes
    t_lo = ts[change_idx].item()
    t_hi = ts[change_idx+1].item()
    label_lo = preds[change_idx].item()
    label_hi = preds[change_idx+1].item()

    # We care about change from base_label to *some* alt label
    # If base_label doesn't appear at either end, still OK; we'll
    # treat it as a boundary between label_lo and label_hi.
    # Use binary search on logits difference between those two labels.
    alt_label = label_hi if label_lo == base_label else label_lo

    for _ in range(20):
        t_mid = 0.5 * (t_lo + t_hi)
        z_mid = z0 + t_mid * direction
        with torch.no_grad():
            log_mid = model.rel_head(z_mid.unsqueeze(0))[0]
        # decision function: f = logit(base) - logit(alt)
        f_lo = (log_mid[base_label] - log_mid[alt_label]).item()
        # Determine which side we're on by also evaluating at t_lo
        z_lo = z0 + t_lo * direction
        with torch.no_grad():
            log_lo = model.rel_head(z_lo.unsqueeze(0))[0]
        f_lo_sign = (log_lo[base_label] - log_lo[alt_label]).item()

        # If signs differ between t_lo and t_mid, root is in [t_lo, t_mid]
        if f_lo_sign == 0:
            t_hi = t_mid
        elif f_lo * f_lo_sign < 0:
            t_hi = t_mid
        else:
            t_lo = t_mid

    t_star = 0.5 * (t_lo + t_hi)
    return t_star, alt_label


def boundary_normal_via_autograd(model, z_star, label_a, label_b):
    """
    At approximate boundary point z_star, compute gradient of
    f(z) = logit(label_a) - logit(label_b).
    This gradient is normal to the decision surface between a and b.
    """
    z_star = z_star.detach().clone().to(device)
    z_star.requires_grad_(True)

    logits = model.rel_head(z_star.unsqueeze(0))[0]
    f = logits[label_a] - logits[label_b]
    model.zero_grad(set_to_none=True)
    if z_star.grad is not None:
        z_star.grad.zero_()
    f.backward()
    n = z_star.grad.detach()
    return n


def angle_between(u, v):
    """Return angle in degrees between vectors u and v."""
    u = u.detach()
    v = v.detach()
    dot = torch.dot(u, v).item()
    nu = u.norm().item()
    nv = v.norm().item()
    if nu == 0 or nv == 0:
        return float("nan")
    cos = max(-1.0, min(1.0, dot / (nu * nv)))
    return math.degrees(math.acos(cos))


# -----------------------
# Main
# -----------------------

def main():
    model = load_scene_model()

    print("[normal256] Generating synthetic dataset...")
    imgs, rels = generate_dataset(N_SAMPLES)
    imgs = imgs.to(device)
    rels = rels.to(device)

    # Encode to latent
    print("[normal256] Encoding dataset to latent...")
    zs = []
    with torch.no_grad():
        for i in range(0, N_SAMPLES, BATCH_SIZE):
            batch = imgs[i:i+BATCH_SIZE]
            z_batch = model.encode(batch)   # (b, D)
            zs.append(z_batch)
    z = torch.cat(zs, dim=0)
    print(f"[normal256] Latent shape: {z.shape}")

    # Estimate global generators
    g_x, g_y = estimate_generators(z, rels)

    # Boundary search over random bases
    indices = torch.randperm(N_SAMPLES)[:N_BASE_SAMPLES].tolist()
    directions = [
        ("+gx", g_x),
        ("-gx", -g_x),
        ("+gy", g_y),
        ("-gy", -g_y),
    ]

    normals = []
    angles_x = []
    angles_y = []
    meta = []

    print(f"[normal256] Searching for boundaries over {len(indices)} bases...")

    for base_idx in indices:
        z0 = z[base_idx]
        base_label = rels[base_idx].item()
        for tag, direction in directions:
            t_star, alt_label = find_boundary_along_direction(
                model, z0, base_label, direction,
                t_min=-3.0, t_max=3.0, n_steps=31
            )
            if t_star is None or alt_label is None:
                continue

            z_star = z0 + t_star * direction
            n = boundary_normal_via_autograd(model, z_star, base_label, alt_label)
            if n.norm().item() == 0.0:
                continue
            n_unit = n / n.norm()

            theta_x = angle_between(n_unit, g_x)
            theta_y = angle_between(n_unit, g_y)

            normals.append(n_unit.cpu().numpy())
            angles_x.append(theta_x)
            angles_y.append(theta_y)
            meta.append((base_idx, base_label, alt_label, tag, t_star))

            print(f"[normal256] boundary at idx={base_idx}, dir={tag}, "
                  f"{REL_NAMES[base_label]}↔{REL_NAMES[alt_label]}, "
                  f"t*={t_star:+.3f}, angle_x={theta_x:.1f}°, angle_y={theta_y:.1f}°")

            if len(normals) >= MAX_BOUNDARIES:
                break
        if len(normals) >= MAX_BOUNDARIES:
            break

    print(f"[normal256] Collected {len(normals)} boundary normals.")

    if len(normals) == 0:
        print("[normal256] No boundaries found in scan range; nothing to plot.")
        return

    angles_x = np.array(angles_x)
    angles_y = np.array(angles_y)

    # ------------------ Stats ------------------

    def summarize(name, arr):
        arr_clean = arr[~np.isnan(arr)]
        print(f"[normal256] {name} angle stats:")
        print(f"   mean = {arr_clean.mean():.2f}°")
        print(f"   std  = {arr_clean.std():.2f}°")
        print(f"   min  = {arr_clean.min():.2f}°")
        print(f"   max  = {arr_clean.max():.2f}°")

    summarize("angle to g_x", angles_x)
    summarize("angle to g_y", angles_y)

    # Which generator is closer?
    closer_x = np.abs(angles_x) < np.abs(angles_y)
    print(f"[normal256] boundaries closer to g_x: {closer_x.sum()} / {len(closer_x)}")
    print(f"[normal256] boundaries closer to g_y: {(~closer_x).sum()} / {len(closer_x)}")

    # ------------------ Plots ------------------

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(angles_x, bins=20)
    axes[0].set_title("Angle normal vs g_x (LR)")
    axes[0].set_xlabel("degrees")
    axes[0].set_ylabel("count")

    axes[1].hist(angles_y, bins=20)
    axes[1].set_title("Angle normal vs g_y (AB)")
    axes[1].set_xlabel("degrees")
    axes[1].set_ylabel("count")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "boundary_normal_angles_latent256.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[normal256] Saved angle histograms -> {out_path}")


if __name__ == "__main__":
    main()
