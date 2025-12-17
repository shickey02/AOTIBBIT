#!/usr/bin/env python3
# geomlang_lie_flows_latent256.py
#
# Global 1-D flows along the learned LR / AB generators in latent256.
#
# For each of a few base scenes z0, we generate:
#   z_LR(t) = z0 + t * g_x
#   z_AB(t) = z0 + t * g_y
#
# and decode them, to see how relations change along these directions.
#
# Run:
#   python bbit_geomlang/geomlang_lie_flows_latent256.py

import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

# -----------------------
# Config
# -----------------------
IMG_SIZE   = 64
NUM_CH     = 3
LATENT_DIM = 256

N_SAMPLES  = 6000
BATCH_SIZE = 256

OUT_DIR         = "outputs_edges_relscale256"
CKPT_SCENEMODEL = os.path.join(OUT_DIR, "scene_model_edges_relscale256.pt")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REL_LEFT, REL_RIGHT, REL_ABOVE, REL_BELOW, REL_OVERLAP = range(5)
REL_NAMES = ["left_of", "right_of", "above", "below", "overlap"]

# -----------------------
# Scene generator (same as before)
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
        return REL_LEFT if dx > 0 else REL_RIGHT
    elif abs(dy) > abs(dx) + tol:
        return REL_ABOVE if dy > 0 else REL_BELOW
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

        base = np.random.randint(5, 13)
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

        img = np.stack([red, blue, edges], axis=0)

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
# Model (must match training)
# -----------------------

class Encoder(nn.Module):
    def __init__(self, in_channels=NUM_CH, latent_dim=LATENT_DIM):
        super().__init__()
        # 64 -> 32 -> 16 -> 8 -> 4
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # 4x4
        self.fc    = nn.Linear(256 * 4 * 4, latent_dim)

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
        self.fc     = nn.Linear(latent_dim, 256 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64,  kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64,  32,  kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32,  out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, 4, 4)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))
        return x

class SceneModelEdges256(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        # heads for compatibility
        self.rel_head     = nn.Linear(LATENT_DIM, 5)
        self.scale_head   = nn.Linear(LATENT_DIM, 3)
        self.shape_r_head = nn.Linear(LATENT_DIM, 2)
        self.shape_b_head = nn.Linear(LATENT_DIM, 2)

    def encode(self, x):  return self.encoder(x)
    def decode(self, z):  return self.decoder(z)

def load_scene_model():
    print(f"[flows256] Loading SceneModel from {CKPT_SCENEMODEL}")
    ckpt = torch.load(CKPT_SCENEMODEL, map_location=DEVICE)
    model = SceneModelEdges256()
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.to(DEVICE)
    model.eval()
    return model

# -----------------------
# Visualization utilities
# -----------------------

def chw_to_rgb(img_chw):
    c, h, w = img_chw.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[..., 0] = img_chw[0]   # red
    rgb[..., 1] = img_chw[2]   # edges → green
    rgb[..., 2] = img_chw[1]   # blue
    return rgb

def predict_rel(model, z):
    with torch.no_grad():
        logits = model.rel_head(z)
        r = torch.argmax(logits, dim=1).item()
    return REL_NAMES[r]

# -----------------------
# Main
# -----------------------

def main():
    print(f"[flows256] Using device: {DEVICE}")
    model = load_scene_model()

    # Generate dataset & encode
    print(f"[flows256] Generating dataset: N={N_SAMPLES}")
    ds = GeomEdges64Dataset(N_SAMPLES)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_imgs, all_rel, all_z = [], [], []
    with torch.no_grad():
        for imgs, rel, scale, s_r, s_b in dl:
            imgs = imgs.to(DEVICE)
            rel  = rel.to(DEVICE)
            z    = model.encode(imgs)

            all_imgs.append(imgs.cpu())
            all_rel.append(rel.cpu())
            all_z.append(z.cpu())

    all_imgs = torch.cat(all_imgs, dim=0)   # [N,3,64,64]
    all_rel  = torch.cat(all_rel,  dim=0)   # [N]
    all_z    = torch.cat(all_z,   dim=0)    # [N,D]

    rel_np = all_rel.numpy()
    z_np   = all_z.numpy()
    N = z_np.shape[0]

    # Relation counts
    for r in range(5):
        n = int((rel_np == r).sum())
        print(f"[flows256] relation {r}: {REL_NAMES[r]:>7}, count={n}")

    # Build LR / AB difference clouds and means
    idx_left  = np.where(rel_np == REL_LEFT)[0]
    idx_right = np.where(rel_np == REL_RIGHT)[0]
    idx_above = np.where(rel_np == REL_ABOVE)[0]
    idx_below = np.where(rel_np == REL_BELOW)[0]

    rng = np.random.default_rng(0)

    def make_pairs(a_idx, b_idx):
        n = min(len(a_idx), len(b_idx))
        a_idx = rng.permutation(a_idx)[:n]
        b_idx = rng.permutation(b_idx)[:n]
        return a_idx, b_idx

    left_idx, right_idx = make_pairs(idx_left, idx_right)
    above_idx, below_idx = make_pairs(idx_above, idx_below)

    v_lr = z_np[right_idx] - z_np[left_idx]
    v_ab = z_np[below_idx] - z_np[above_idx]

    def vector_stats(v, name):
        norms = np.linalg.norm(v, axis=1)
        print(f"\n[flows256] {name} norms:")
        print(f"  mean={norms.mean():.4f}, std={norms.std():.4f}, "
              f"min={norms.min():.4f}, max={norms.max():.4f}")
        v_mean = v.mean(axis=0)
        return v_mean

    g_x = vector_stats(v_lr, "left→right")
    g_y = vector_stats(v_ab, "above→below")

    g_x_t = torch.from_numpy(g_x).float().to(DEVICE)
    g_y_t = torch.from_numpy(g_y).float().to(DEVICE)

    # -----------------------
    # Choose some base indices
    # -----------------------
    base_indices = []

    def pick_one(mask_indices, label):
        if len(mask_indices) == 0:
            return None
        idx = mask_indices[len(mask_indices) // 2]
        print(f"[flows256] base {label}: idx={idx}")
        return idx

    # one of each kind, where possible
    b_left   = pick_one(idx_left,   "left_of")
    b_right  = pick_one(idx_right,  "right_of")
    b_above  = pick_one(idx_above,  "above")
    b_overlap= pick_one(np.where(rel_np == REL_OVERLAP)[0], "overlap")

    for x in [b_left, b_right, b_above, b_overlap]:
        if x is not None:
            base_indices.append(x)

    if not base_indices:
        print("[flows256] No base indices found, aborting.")
        return

    # t grid for the flows (LR & AB)
    t_grid = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5], dtype=np.float32)
    num_t = len(t_grid)

    print(f"[flows256] Using t grid: {t_grid}")

    # -----------------------
    # For each base index, plot two flows
    # -----------------------
    for base_idx in base_indices:
        z0 = all_z[base_idx:base_idx+1].to(DEVICE)  # [1,D]
        img0 = all_imgs[base_idx]                   # [3,64,64]
        base_rel_name = REL_NAMES[int(rel_np[base_idx])]

        print(f"[flows256] Rendering flows for base idx={base_idx} (rel={base_rel_name})")

        fig, axes = plt.subplots(2, num_t, figsize=(2*num_t, 4), dpi=120)
        fig.suptitle(
            f"Latent flows around base idx={base_idx} (latent256)\n"
            f"top: LR (g_x), bottom: AB (g_y)",
            fontsize=10
        )

        with torch.no_grad():
            # LR row
            for j, t in enumerate(t_grid):
                z_t = z0 + float(t) * g_x_t.unsqueeze(0)
                img_t = model.decode(z_t).cpu()[0]
                rel_name = predict_rel(model, z_t)

                ax = axes[0, j]
                ax.axis("off")
                ax.imshow(chw_to_rgb(img_t.numpy()))
                ax.set_title(f"t={t:+.1f}\nrel={rel_name}", fontsize=7)

            # AB row
            for j, t in enumerate(t_grid):
                z_t = z0 + float(t) * g_y_t.unsqueeze(0)
                img_t = model.decode(z_t).cpu()[0]
                rel_name = predict_rel(model, z_t)

                ax = axes[1, j]
                ax.axis("off")
                ax.imshow(chw_to_rgb(img_t.numpy()))
                ax.set_title(f"t={t:+.1f}\nrel={rel_name}", fontsize=7)

        plt.tight_layout(rect=[0, 0, 1, 0.90])
        out_path = os.path.join(OUT_DIR, f"lie_flows_base_idx{base_idx}_latent256.png")
        plt.savefig(out_path)
        plt.close(fig)
        print(f"[flows256] Saved flows grid -> {out_path}")

    print("[flows256] Done.")

if __name__ == "__main__":
    main()
