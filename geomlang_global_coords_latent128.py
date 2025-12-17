#!/usr/bin/env python3
# geomlang_global_coords_latent128.py
#
# Global generator coordinates (t_x, t_y) for the 128-dim latent model.
# Mirrors geomlang_global_coords_latent256.py but uses the 128 checkpoint
# in outputs_edges_relscale/scene_model_edges_relscale.pt.

import os
import math
import numpy as np

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Reuse dataset + loader + constants from the 128 Lie-group script
from geomlang_lie_group_latent128 import (
    GeomEdges64Dataset,
    load_scene_model,
    REL_LEFT, REL_RIGHT, REL_ABOVE, REL_BELOW, REL_OVERLAP,
    REL_NAMES,
)

OUT_DIR = "outputs_edges_relscale"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_SAMPLES  = 6000
BATCH_SIZE = 128


def main():
    print(f"[coords128] Using device: {DEVICE}")

    # ---------- load model ----------
    model = load_scene_model()   # comes from geomlang_lie_group_latent128
    model.to(DEVICE)
    model.eval()

    # ---------- generate dataset + encode ----------
    print(f"[coords128] Generating dataset: N={N_SAMPLES}")
    ds = GeomEdges64Dataset(N_SAMPLES)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_z   = []
    all_rel = []

    with torch.no_grad():
        for imgs, rel, scale, s_r, s_b in dl:
            imgs = imgs.to(DEVICE)
            z    = model.encode(imgs)

            all_z.append(z.cpu())
            all_rel.append(rel.cpu())

    z   = torch.cat(all_z, dim=0).numpy()        # (N, D)
    rel = torch.cat(all_rel, dim=0).numpy()      # (N,)

    print(f"[coords128] Latent shape: {z.shape}")

    # ---------- relation counts ----------
    for r in range(5):
        n = int((rel == r).sum())
        print(f"[coords128] relation {r}: {REL_NAMES[r]:>7}, count={n}")

    # ---------- relation means & generators ----------
    mu = {}
    for r in range(5):
        idx = (rel == r)
        mu_r = z[idx].mean(axis=0)
        mu[r] = mu_r

    mu_left   = mu[REL_LEFT]
    mu_right  = mu[REL_RIGHT]
    mu_above  = mu[REL_ABOVE]
    mu_below  = mu[REL_BELOW]

    diff_lr = mu_right - mu_left
    diff_ab = mu_below - mu_above

    norm_lr = np.linalg.norm(diff_lr)
    norm_ab = np.linalg.norm(diff_ab)

    print(f"[coords128] ||mu_right - mu_left|| = {norm_lr:.4f}")
    print(f"[coords128] ||mu_below - mu_above|| = {norm_ab:.4f}")

    g_x = diff_lr / (norm_lr + 1e-8)
    g_y = diff_ab / (norm_ab + 1e-8)

    # angle between generators
    cos_angle = float(np.dot(g_x, g_y))
    cos_angle = max(-1.0, min(1.0, cos_angle))
    angle_deg = math.degrees(math.acos(cos_angle))
    print(f"[coords128] angle(g_x, g_y) = {angle_deg:.2f}° (cos = {cos_angle:.4f})")

    # ---------- global coordinates (t_x, t_y) ----------
    mu0 = z.mean(axis=0)
    v   = z - mu0  # center

    t_x = v @ g_x
    t_y = v @ g_y

    # ---------- scatter in (t_x, t_y) ----------
    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)

    color_map = {
        REL_LEFT:    "tab:blue",
        REL_RIGHT:   "tab:orange",
        REL_ABOVE:   "tab:green",
        REL_BELOW:   "tab:red",
        REL_OVERLAP: "tab:purple",
    }

    for r in range(5):
        idx = (rel == r)
        ax.scatter(
            t_x[idx], t_y[idx],
            s=5, alpha=0.5,
            label=REL_NAMES[r],
            color=color_map.get(r, None),
        )

    ax.axhline(0.0, color="k", linewidth=0.5)
    ax.axvline(0.0, color="k", linewidth=0.5)
    ax.set_xlabel("t_x (LR generator coord)")
    ax.set_ylabel("t_y (AB generator coord)")
    ax.set_title("Global generator coordinates (latent128)")
    ax.legend(loc="upper right", markerscale=3)

    plt.tight_layout()
    out_scatter = os.path.join(OUT_DIR, "global_coords_scatter_latent128.png")
    plt.savefig(out_scatter)
    plt.close(fig)
    print(f"[coords128] Saved global coord scatter -> {out_scatter}")

    # ---------- residual r_perp stats ----------
    proj = np.outer(t_x, g_x) + np.outer(t_y, g_y)
    r_perp = v - proj

    norms_perp = np.linalg.norm(r_perp, axis=1)
    norms_full = np.linalg.norm(v, axis=1) + 1e-8
    rel_ratio  = norms_perp / norms_full

    print("[coords128] Residual norms ||r_perp|| stats:")
    print(f"   mean = {norms_perp.mean():.4f}, "
          f"std = {norms_perp.std():.4f}, "
          f"min = {norms_perp.min():.4f}, "
          f"max = {norms_perp.max():.4f}")
    print("[coords128] Relative residual ||r_perp|| / ||z-mu0|| stats:")
    print(f"   mean = {rel_ratio.mean():.4f}, "
          f"std = {rel_ratio.std():.4f}, "
          f"min = {rel_ratio.min():.4f}, "
          f"max = {rel_ratio.max():.4f}")

    # ---------- residual histogram ----------
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    ax.hist(norms_perp, bins=40)
    ax.set_xlabel("||r_perp||")
    ax.set_ylabel("count")
    ax.set_title("Residual norm after projection onto span{g_x, g_y} (latent128)")
    plt.tight_layout()

    out_hist = os.path.join(OUT_DIR, "global_coords_residual_hist_latent128.png")
    plt.savefig(out_hist)
    plt.close(fig)
    print(f"[coords128] Saved residual histogram -> {out_hist}")

    print("[coords128] Done.")


if __name__ == "__main__":
    main()
