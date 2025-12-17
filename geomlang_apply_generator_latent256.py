#!/usr/bin/env python3
# geomlang_apply_generator_latent256.py
#
# "Action module" for the latent 256D edges+relscale model.
# Given an image x and desired (Δt_x, Δt_y), we:
#   - encode -> z
#   - move:  z' = z + Δt_x * g_x + Δt_y * g_y
#   - decode -> x'
#
# This script:
#   - recomputes g_x, g_y and relation means from a synthetic dataset
#   - applies a few example moves to some random scenes
#   - saves a demo grid PNG.

import os
import math
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from geomlang_global_coords_latent256 import (
    GeomEdges64Dataset,
    SceneModelEdges256,
    load_scene_model,
    REL_LEFT, REL_RIGHT, REL_ABOVE, REL_BELOW, REL_OVERLAP, REL_NAMES,
)

TAG        = "[act256]"
IMG_SIZE   = 64
LATENT_DIM = 256
N_SAMPLES  = 4000
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR    = "outputs_edges_relscale256"
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE = 128

# --------------------------------------------------
# Helpers to encode data + compute generators
# --------------------------------------------------

def encode_dataset(model, n_samples=N_SAMPLES):
    print(f"{TAG} Encoding dataset: N={n_samples}")
    ds = GeomEdges64Dataset(n_samples)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_imgs = []
    all_rel  = []
    all_z    = []

    model.eval()
    with torch.no_grad():
        for imgs, rel, scale, s_r, s_b in dl:
            imgs = imgs.to(DEVICE)
            z    = model.encode(imgs)
            all_imgs.append(imgs.cpu())
            all_rel.append(rel.cpu())
            all_z.append(z.cpu())

    imgs = torch.cat(all_imgs, dim=0)  # (N, C, H, W)
    rel  = torch.cat(all_rel,  dim=0)  # (N,)
    z    = torch.cat(all_z,    dim=0)  # (N, D)

    print(f"{TAG} Latent shape: {tuple(z.shape)}")
    return imgs, rel, z


def compute_generators(z, rel):
    """Return g_x, g_y, relation means mu_r (torch CPU tensors)."""
    mus = []
    rel_np = rel.numpy()
    for rid in range(5):
        mask = (rel_np == rid)
        z_r = z[mask]
        mu_r = z_r.mean(dim=0)
        print(f"{TAG} relation {rid}: {REL_NAMES[rid]:>7}, "
              f"count={mask.sum()}, ||mu_r||={mu_r.norm():.3f}")
        mus.append(mu_r)

    mu_left, mu_right, mu_above, mu_below, mu_overlap = mus

    v_LR = mu_right - mu_left
    v_AB = mu_below - mu_above

    g_x = v_LR / v_LR.norm()
    g_y = v_AB / v_AB.norm()

    cos_xy = torch.dot(g_x, g_y).item()
    angle = math.degrees(math.acos(max(min(cos_xy, 1.0), -1.0)))
    print(f"{TAG} ||mu_right - mu_left|| = {v_LR.norm():.4f}")
    print(f"{TAG} ||mu_below - mu_above|| = {v_AB.norm():.4f}")
    print(f"{TAG} angle(g_x, g_y) = {angle:.2f}°")

    return g_x.to(DEVICE), g_y.to(DEVICE), mus


# --------------------------------------------------
# Action: apply (Δt_x, Δt_y) in latent space
# --------------------------------------------------

def apply_generator(model, img_batch, g_x, g_y, delta_tx, delta_ty):
    """
    img_batch: (B, C, H, W) on DEVICE
    delta_tx, delta_ty: floats (or 1D tensors length B)
    """
    model.eval()
    with torch.no_grad():
        z = model.encode(img_batch)  # (B, D)
        z_shift = z + delta_tx * g_x.unsqueeze(0) + delta_ty * g_y.unsqueeze(0)
        x_rec = model.decode(z_shift)  # (B, C, H, W)
    return x_rec


# --------------------------------------------------
# Demo: grid of actions for a few scenes
# --------------------------------------------------

def demo_grid(model, imgs, rel, g_x, g_y):
    """
    Pick one example for a few relations, apply moves, and save a grid.
    """
    # pick up to one example of each relation
    rel_np = rel.numpy()
    chosen_idx = []
    for rid in [REL_LEFT, REL_RIGHT, REL_ABOVE, REL_BELOW, REL_OVERLAP]:
        where = np.nonzero(rel_np == rid)[0]
        if len(where) > 0:
            chosen_idx.append(int(where[0]))

    if len(chosen_idx) == 0:
        print(f"{TAG} No examples found, aborting demo.")
        return

    base_imgs = imgs[chosen_idx]  # (K, C, H, W)
    K = base_imgs.shape[0]
    base_rel = [REL_NAMES[int(rel[i])] for i in chosen_idx]

    base_imgs_dev = base_imgs.to(DEVICE)

    # moves: (name, Δt_x, Δt_y)
    moves = [
        ("orig",  0.0,  0.0),
        ("+x",   +0.8,  0.0),
        ("-x",   -0.8,  0.0),
        ("+y",    0.0, +0.8),
        ("-y",    0.0, -0.8),
        ("diag", +0.6, +0.6),
    ]

    cols = len(moves)
    rows = K

    fig, axes = plt.subplots(rows, cols, figsize=(2.4*cols, 2.4*rows))
    fig.suptitle("Generator actions on sample scenes (latent256)")

    for c, (label, dx, dy) in enumerate(moves):
        dx_t = torch.tensor(dx, device=DEVICE)
        dy_t = torch.tensor(dy, device=DEVICE)
        x_out = apply_generator(model, base_imgs_dev, g_x, g_y, dx_t, dy_t)
        x_out_np = x_out.cpu().numpy()  # (K, C, H, W)

        for r in range(rows):
            ax = axes[r, c] if rows > 1 else axes[c]
            ax.axis("off")
            img = x_out_np[r]
            if img.shape[0] == 1:
                ax.imshow(img[0], cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(np.transpose(img, (1, 2, 0)))
            if r == 0:
                ax.set_title(label)
            if c == 0:
                ax.text(
                    -0.05, 0.5, base_rel[r],
                    transform=ax.transAxes,
                    va="center", ha="right", fontsize=9,
                )

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "generator_actions_demo_latent256.png")
    print(f"{TAG} Saving {out_path}")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    print(f"{TAG} Using device: {DEVICE}")
    model = load_scene_model()
    model.to(DEVICE)

    imgs, rel, z = encode_dataset(model, N_SAMPLES)
    g_x, g_y, mus = compute_generators(z, rel)

    # Optional: save basis for reuse
    basis_npz = os.path.join(OUT_DIR, "generator_basis_latent256.npz")
    np.savez(
        basis_npz,
        g_x=g_x.cpu().numpy(),
        g_y=g_y.cpu().numpy(),
    )
    print(f"{TAG} Saved generator basis -> {basis_npz}")

    demo_grid(model, imgs, rel, g_x, g_y)
    print(f"{TAG} Done.")

if __name__ == "__main__":
    main()
