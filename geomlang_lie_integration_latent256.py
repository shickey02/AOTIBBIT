#!/usr/bin/env python3
# geomlang_lie_integration_latent256.py
#
# "True Lie-group integration" along the learned relation generators g_x (LR)
# and g_y (AB) for the 256-dim edges+relscale model.
#
# For a few seed latents z0 (one near each relation mean), we generate a
# 2D grid in generator coordinates (t_x, t_y), move:
#   z(t_x, t_y) = z0 + t_x * g_x + t_y * g_y
# decode images and classify them, and save big grid figures.
#
# Requires: geomlang_lie_group_latent256.py in the same directory.

import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from geomlang_lie_group_latent256 import (
    GeomEdges64Dataset,
    SceneModelEdges256,
    REL_LEFT, REL_RIGHT, REL_ABOVE, REL_BELOW, REL_OVERLAP,
    REL_NAMES,
    OUT_DIR, CKPT_SCENEMODEL, DEVICE,
)

IMG_SIZE = 64
N_SAMPLES = 6000
BATCH_SIZE = 128

# Where to save the grids
GRID_OUT_DIR = os.path.join(OUT_DIR, "lie_integration_grids")
os.makedirs(GRID_OUT_DIR, exist_ok=True)


def load_scene_model():
    print(f"[lie-int256] Loading SceneModel from {CKPT_SCENEMODEL}")
    ckpt = torch.load(CKPT_SCENEMODEL, map_location=DEVICE)
    model = SceneModelEdges256()
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.to(DEVICE)
    model.eval()
    return model


def encode_dataset(model):
    print(f"[lie-int256] Generating dataset: N={N_SAMPLES}")
    ds = GeomEdges64Dataset(N_SAMPLES)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_z = []
    all_rel = []

    with torch.no_grad():
        for imgs, rel, scale, s_r, s_b in dl:
            imgs = imgs.to(DEVICE)
            rel = rel.to(DEVICE)
            z = model.encode(imgs)
            all_z.append(z.cpu())
            all_rel.append(rel.cpu())

    all_z = torch.cat(all_z, dim=0)         # (N, 256)
    all_rel = torch.cat(all_rel, dim=0)     # (N,)
    print(f"[lie-int256] Encoded latent shape: {all_z.shape}")
    return all_z.numpy(), all_rel.numpy()


def compute_relation_means(z_np, rel_np):
    mus = {}
    for r in range(5):
        mask = (rel_np == r)
        mu_r = z_np[mask].mean(axis=0)
        mus[r] = mu_r
        print(f"[lie-int256] relation {r}={REL_NAMES[r]:>7}, count={mask.sum()}, "
              f"||mu_r||={np.linalg.norm(mu_r):.3f}")
    return mus


def compute_generators(mus):
    mu_left  = mus[REL_LEFT]
    mu_right = mus[REL_RIGHT]
    mu_above = mus[REL_ABOVE]
    mu_below = mus[REL_BELOW]

    g_x = mu_right - mu_left
    g_y = mu_below - mu_above

    gx_norm = np.linalg.norm(g_x)
    gy_norm = np.linalg.norm(g_y)
    print(f"[lie-int256] ||mu_right - mu_left|| = {gx_norm:.4f}")
    print(f"[lie-int256] ||mu_below - mu_above|| = {gy_norm:.4f}")

    # Normalize generators so t_x, t_y are "distance in units of one mean-separation"
    g_x_unit = g_x / gx_norm
    g_y_unit = g_y / gy_norm

    cos_angle = np.dot(g_x_unit, g_y_unit)
    angle_deg = math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))
    print(f"[lie-int256] angle(g_x, g_y) = {angle_deg:.2f}° (cos={cos_angle:.4f})")

    return g_x_unit, g_y_unit, gx_norm, gy_norm


def pick_seeds(z_np, rel_np, mus, per_relation=1):
    """
    For each relation r, pick `per_relation` seeds that are closest in latent
    space to the relation mean mu_r.
    """
    seeds = []
    for r in range(5):
        mu_r = mus[r]
        mask = (rel_np == r)
        z_r = z_np[mask]
        idx_r = np.where(mask)[0]
        dists = np.linalg.norm(z_r - mu_r[None, :], axis=1)
        order = np.argsort(dists)
        k = min(per_relation, len(order))
        for j in range(k):
            global_idx = int(idx_r[order[j]])
            seeds.append((global_idx, r))
        print(f"[lie-int256] relation {REL_NAMES[r]:>7}: picked {k} seeds")
    return seeds


def classify_relations(model, z_batch):
    """
    Given latent batch z (numpy, shape (B, D)), return predicted relation labels.
    """
    z_t = torch.from_numpy(z_batch).to(DEVICE, dtype=torch.float32)
    with torch.no_grad():
        logits = model.rel_head(z_t)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    return preds


def decode_imgs(model, z_batch):
    z_t = torch.from_numpy(z_batch).to(DEVICE, dtype=torch.float32)
    with torch.no_grad():
        x_rec = model.decode(z_t).cpu().numpy()
    # x_rec: (B, C, H, W), values in [0,1]
    return x_rec


def make_grid_for_seed(model, z_np, seed_idx, seed_rel, g_x_unit, g_y_unit,
                       t_range=(-3.0, 3.0), n_steps=13):
    """
    Build a (n_steps x n_steps) grid in (t_x, t_y) around z0, decode, classify,
    and save as a big figure.
    """
    z0 = z_np[seed_idx]
    rel_name = REL_NAMES[seed_rel]

    ts = np.linspace(t_range[0], t_range[1], n_steps)
    TX, TY = np.meshgrid(ts, ts, indexing="xy")

    # Prepare all latent points in one big batch for efficiency
    z_grid = []
    for i in range(n_steps):
        for j in range(n_steps):
            t_x = TX[i, j]
            t_y = TY[i, j]
            z_ij = z0 + t_x * g_x_unit + t_y * g_y_unit
            z_grid.append(z_ij)
    z_grid = np.stack(z_grid, axis=0)  # (n_steps^2, D)

    preds = classify_relations(model, z_grid)
    imgs = decode_imgs(model, z_grid)

    # Make figure
    fig, axes = plt.subplots(n_steps, n_steps,
                             figsize=(1.2 * n_steps, 1.2 * n_steps),
                             dpi=120)
    fig.suptitle(
        f"Lie integration grid around seed idx={seed_idx} ({rel_name}, latent256)\n"
        f"rows: t_y, cols: t_x"
    )

    # Rescale to RGB for visualization
    def to_rgb(chw):
        # chw: (3, H, W)
        c, h, w = chw.shape
        assert c == 3
        return np.transpose(chw, (1, 2, 0))

    k = 0
    for i in range(n_steps):
        for j in range(n_steps):
            ax = axes[i, j]
            img = to_rgb(imgs[k])
            ax.imshow(img)
            ax.axis("off")

            label = REL_NAMES[int(preds[k])]
            t_x = TX[i, j]
            t_y = TY[i, j]
            ax.set_title(
                f"tx={t_x:+.1f}\n"
                f"ty={t_y:+.1f}\n"
                f"{label}",
                fontsize=6
            )
            k += 1

    plt.tight_layout()
    out_path = os.path.join(
        GRID_OUT_DIR,
        f"lie_grid_seed{seed_idx}_rel{rel_name}_latent256.png"
    )
    plt.savefig(out_path)
    plt.close(fig)
    print(f"[lie-int256] Saved Lie integration grid -> {out_path}")


def main():
    print(f"[lie-int256] Using device: {DEVICE}")
    model = load_scene_model()

    z_np, rel_np = encode_dataset(model)
    mus = compute_relation_means(z_np, rel_np)
    g_x_unit, g_y_unit, gx_norm, gy_norm = compute_generators(mus)

    # Pick one seed per relation (closest to mean)
    seeds = pick_seeds(z_np, rel_np, mus, per_relation=1)

    # Generate grids around each seed
    for seed_idx, seed_rel in seeds:
        make_grid_for_seed(
            model,
            z_np,
            seed_idx=seed_idx,
            seed_rel=seed_rel,
            g_x_unit=g_x_unit,
            g_y_unit=g_y_unit,
            t_range=(-3.0, 3.0),
            n_steps=11,  # 11x11 grid is already pretty big
        )

    print("[lie-int256] Done.")


if __name__ == "__main__":
    main()
