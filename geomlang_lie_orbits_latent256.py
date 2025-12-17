#!/usr/bin/env python3
# geomlang_lie_orbits_latent256.py
#
# Visualize 2D Lie-like orbits in latent space for the
# 64x64 edges+relscale model with LATENT_DIM=256.
#
# Run:
#   python bbit_geomlang/geomlang_lie_orbits_latent256.py

import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Import model architecture that matches the checkpoint
from geomlang_lie_group_latent256 import SceneModelEdges256

# -----------------------
# Config
# -----------------------
IMG_SIZE   = 64          # IMPORTANT: model was trained on 64x64
NUM_CH     = 3
LATENT_DIM = 256

N_SAMPLES  = 6000
BATCH_SIZE = 128

N_BASE_ORBITS = 4
GRID_ALPHA_BETA = [-1.0, -0.5, 0.0, 0.5, 1.0]

OUT_DIR         = "outputs_edges_relscale256"
CKPT_SCENEMODEL = os.path.join(OUT_DIR, "scene_model_edges_relscale256.pt")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REL_LEFT, REL_RIGHT, REL_ABOVE, REL_BELOW, REL_OVERLAP = range(5)
REL_NAMES = ["left_of", "right_of", "above", "below", "overlap"]

# -----------------------
# Scene generation (64x64, edges+relscale)
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
        union[1:-1, 1:-1]
        * union[:-2, 1:-1]
        * union[2:, 1:-1]
        * union[1:-1, :-2]
        * union[1:-1, 2:]
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
    """
    Static scenes, 64x64, with:
      - red blob (ch0)
      - blue blob (ch1)
      - edges (union, ch2)
    """

    def __init__(self, n_samples):
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
# Utils
# -----------------------

def chw_to_rgb(img_chw):
    c, h, w = img_chw.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[..., 0] = img_chw[0]  # red
    rgb[..., 1] = img_chw[2]  # edges
    rgb[..., 2] = img_chw[1]  # blue
    return rgb


def load_scene_model():
    print(f"[lie256-orbits] Loading SceneModel from {CKPT_SCENEMODEL}")
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
# Main
# -----------------------

def main():
    print(f"[lie256-orbits] Using device: {DEVICE}")
    model = load_scene_model()

    print(f"[lie256-orbits] Generating dataset: N={N_SAMPLES}")
    ds = GeomEdges64Dataset(N_SAMPLES)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_imgs, all_rel, all_z = [], [], []

    with torch.no_grad():
        for imgs, rel, scale, s_r, s_b in dl:
            imgs = imgs.to(DEVICE)
            rel = rel.to(DEVICE)
            z = model.encoder(imgs)
            all_imgs.append(imgs.cpu())
            all_rel.append(rel.cpu())
            all_z.append(z.cpu())

    all_imgs = torch.cat(all_imgs, dim=0)
    all_rel  = torch.cat(all_rel,  dim=0)
    all_z    = torch.cat(all_z,    dim=0)

    rel_np = all_rel.numpy()
    z_np   = all_z.numpy()

    # relation counts (sanity)
    for r in range(5):
        n = int((rel_np == r).sum())
        print(f"[lie256-orbits] relation {r}: {REL_NAMES[r]:>7}, count={n}")

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

    left_idx, right_idx   = make_pairs(idx_left, idx_right)
    above_idx, below_idx  = make_pairs(idx_above, idx_below)

    z_left   = z_np[left_idx]
    z_right  = z_np[right_idx]
    z_above  = z_np[above_idx]
    z_below  = z_np[below_idx]

    v_lr = z_right - z_left
    v_ab = z_below - z_above

    print(f"[lie256-orbits] LR pairs: {v_lr.shape[0]}, AB pairs: {v_ab.shape[0]}")

    # PCA -> first component as generator direction
    pca_lr = PCA(n_components=1, random_state=0)
    pca_lr.fit(v_lr)
    g_x_dir = pca_lr.components_[0]

    pca_ab = PCA(n_components=1, random_state=1)
    pca_ab.fit(v_ab)
    g_y_dir = pca_ab.components_[0]

    proj_lr = v_lr @ g_x_dir
    proj_ab = v_ab @ g_y_dir
    step_lr = proj_lr.mean()
    step_ab = proj_ab.mean()

    g_x = g_x_dir * step_lr
    g_y = g_y_dir * step_ab

    print("[lie256-orbits] Generator stats:")
    print(f"  ||g_x|| (LR) = {np.linalg.norm(g_x):.4f}, mean step={step_lr:.4f}")
    print(f"  ||g_y|| (AB) = {np.linalg.norm(g_y):.4f}, mean step={step_ab:.4f}")

    g_x_t = torch.from_numpy(g_x).float().to(DEVICE)
    g_y_t = torch.from_numpy(g_y).float().to(DEVICE)

    # pick base orbits
    N = z_np.shape[0]
    base_indices = rng.choice(N, size=min(N_BASE_ORBITS, N), replace=False)
    print(f"[lie256-orbits] Base indices: {base_indices}")

    alphas = GRID_ALPHA_BETA
    betas  = GRID_ALPHA_BETA
    nA = len(alphas)
    nB = len(betas)

    for idx_k, base_idx in enumerate(base_indices):
        z0 = all_z[base_idx:base_idx+1].to(DEVICE)  # [1,D]

        zs = []
        for a in alphas:
            for b in betas:
                z_ab = z0 + a * g_x_t.unsqueeze(0) + b * g_y_t.unsqueeze(0)
                zs.append(z_ab)
        zs = torch.cat(zs, dim=0)   # [nA*nB, D]

        with torch.no_grad():
            imgs_dec = model.decoder(zs).cpu()

        with torch.no_grad():
            base_img = model.decoder(z0).cpu()[0]

        fig, axes = plt.subplots(nA, nB, figsize=(2*nB, 2*nA), dpi=120)
        fig.suptitle(
            f"Lie orbit around base index {base_idx} (64x64)\n"
            f"rows: alpha (LR), cols: beta (AB)",
            fontsize=10
        )

        for i, a in enumerate(alphas):
            for j, b in enumerate(betas):
                k = i * nB + j
                ax = axes[i, j]
                ax.axis("off")
                if a == 0.0 and b == 0.0:
                    ax.set_title("α=0, β=0", fontsize=6)
                else:
                    ax.set_title(f"α={a:.1f}, β={b:.1f}", fontsize=6)
                img_chw = imgs_dec[k]
                ax.imshow(chw_to_rgb(img_chw.numpy()), interpolation="nearest")

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        out_path = os.path.join(
            OUT_DIR,
            f"lie_orbit_base{idx_k}_idx{base_idx}_latent256.png"
        )
        plt.savefig(out_path)
        plt.close(fig)
        print(f"[lie256-orbits] Saved orbit grid -> {out_path}")

        fig2, ax2 = plt.subplots(1, 1, figsize=(3, 3), dpi=120)
        ax2.axis("off")
        ax2.set_title(f"Base reconstruction (idx={base_idx})", fontsize=8)
        ax2.imshow(chw_to_rgb(base_img.numpy()), interpolation="nearest")
        base_out = os.path.join(
            OUT_DIR,
            f"lie_orbit_base_only_idx{base_idx}_latent256.png"
        )
        plt.tight_layout()
        plt.savefig(base_out)
        plt.close(fig2)
        print(f"[lie256-orbits] Saved base image -> {base_out}")

    print("[lie256-orbits] Done.")


if __name__ == "__main__":
    main()
