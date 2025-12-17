#!/usr/bin/env python3
# geomlang_noncomm_area_scaling_latent256.py
#
# For several seeds, run commutator-style rectangle loops with
# varying side length eps, and plot:
#   area = eps^2     vs   displacement = ||z_end - z0||
#
# This should be ~linear in area for small eps in a "curvature" regime.

import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
TAG        = "[area256]"
IMG_SIZE   = 64
LATENT_DIM = 256
N_SAMPLES  = 6000
BATCH_SIZE = 128
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR    = "outputs_edges_relscale256"
os.makedirs(OUT_DIR, exist_ok=True)
MODEL_PATH = os.path.join(OUT_DIR, "scene_model_edges_relscale256.pt")

# rectangle side lengths to test (same for x and y)
EPS_LIST = [0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00]

REL_NAMES = ["left_of", "right_of", "above", "below", "overlap"]
REL_LEFT, REL_RIGHT, REL_ABOVE, REL_BELOW, REL_OVERLAP = range(5)

# ---------------------------------------------------------------------
# Import dataset + model definition from your existing script
# (adjust the import path if needed).
# ---------------------------------------------------------------------
from geomlang_global_coords_latent256 import (
    GeomEdges64Dataset,
    SceneModelEdges256,
    load_scene_model,
)

# ---------------------------------------------------------------------
# Encode dataset once
# ---------------------------------------------------------------------
def encode_dataset(model, n_samples=N_SAMPLES):
    print(f"{TAG} Encoding dataset: N={n_samples}")
    ds = GeomEdges64Dataset(n_samples)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_z = []
    all_rel = []
    with torch.no_grad():
        for imgs, rel, scale, s_r, s_b in dl:
            imgs = imgs.to(DEVICE)
            z = model.encode(imgs)
            all_z.append(z.cpu())
            all_rel.append(rel.cpu())

    z_all   = torch.cat(all_z, dim=0)        # (N, D)
    rel_ids = torch.cat(all_rel, dim=0).numpy()  # (N,)
    print(f"{TAG} Latent shape: {tuple(z_all.shape)}")
    return z_all, rel_ids

# ---------------------------------------------------------------------
# Compute global generators from relation means
# ---------------------------------------------------------------------
def compute_global_generators(z_all, rel_ids):
    mus = []
    for rid in range(5):
        mask = (rel_ids == rid)
        rel_z = z_all[mask]
        mu_r = rel_z.mean(0)
        print(f"{TAG} relation {rid}: {REL_NAMES[rid]:>7}, "
              f"count={mask.sum()}, ||mu_r||={mu_r.norm():.3f}")
        mus.append(mu_r)

    mu_left, mu_right, mu_above, mu_below, mu_overlap = mus

    g_x_raw = mu_right - mu_left
    g_y_raw = mu_below - mu_above

    g_x = g_x_raw / g_x_raw.norm()
    g_y = g_y_raw / g_y_raw.norm()

    angle = math.degrees(math.acos(torch.clamp(torch.dot(g_x, g_y), -1.0, 1.0)))
    print(f"{TAG} ||mu_right - mu_left|| = {g_x_raw.norm():.4f}")
    print(f"{TAG} ||mu_below - mu_above|| = {g_y_raw.norm():.4f}")
    print(f"{TAG} angle(g_x, g_y) = {angle:.2f}°")

    return g_x.to(DEVICE), g_y.to(DEVICE), mus

# ---------------------------------------------------------------------
# Flow operators (same pattern as noncomm_loops script)
# ---------------------------------------------------------------------
def flow_step(model, z, direction_vec, eps):
    """
    One latent step + decode+encode projection:
        z' = E(D(z + eps * v))
    """
    z_prop = z + eps * direction_vec
    with torch.no_grad():
        x = model.decode(z_prop.unsqueeze(0))
        z_new = model.encode(x).squeeze(0)
    return z_new

def loop_rectangle(model, z0, g_x, g_y, eps_x, eps_y):
    """
    Commutator-style rectangle:
        +x -> +y -> -x -> -y
    """
    z0 = z0.to(DEVICE)
    zs = [z0]
    z1 = flow_step(model, zs[-1], g_x, +eps_x)
    zs.append(z1)
    z2 = flow_step(model, zs[-1], g_y, +eps_y)
    zs.append(z2)
    z3 = flow_step(model, zs[-1], g_x, -eps_x)
    zs.append(z3)
    z4 = flow_step(model, zs[-1], g_y, -eps_y)
    zs.append(z4)
    return zs  # [z0, z1, z2, z3, z4]

# ---------------------------------------------------------------------
# Choose seeds as in noncomm_loops
# ---------------------------------------------------------------------
def choose_seeds(z_all, rel_ids, mus):
    seeds = {}
    # closest to mu_left, mu_above, mu_overlap
    mu_left, _, mu_above, _, mu_overlap = mus

    def closest_idx(mu, target_rel):
        mask = (rel_ids == target_rel)
        idxs = np.nonzero(mask)[0]
        rel_z = z_all[mask]
        d = torch.norm(rel_z - mu, dim=1)
        local = int(torch.argmin(d))
        return int(idxs[local])

    seeds["left_of"]  = closest_idx(mu_left,   REL_LEFT)
    seeds["above"]    = closest_idx(mu_above,  REL_ABOVE)
    seeds["overlap"]  = closest_idx(mu_overlap, REL_OVERLAP)
    print(f"{TAG} Chosen seeds (global indices): {seeds}")
    return seeds

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print(f"{TAG} Using device: {DEVICE}")
    model = load_scene_model()  # already moves to DEVICE & eval in your helper

    z_all, rel_ids = encode_dataset(model, N_SAMPLES)
    g_x, g_y, mus  = compute_global_generators(z_all, rel_ids)
    seeds = choose_seeds(z_all, rel_ids, mus)

    areas = [eps**2 for eps in EPS_LIST]

    for name, idx in seeds.items():
        print(f"{TAG} Seed '{name}' (idx={idx}, rel={REL_NAMES[rel_ids[idx]]})")
        z0 = z_all[idx].to(DEVICE)

        disp = []
        for eps in EPS_LIST:
            zs = loop_rectangle(model, z0, g_x, g_y, eps, eps)
            z_end = zs[-1]
            d = torch.norm(z_end - z0).item()
            disp.append(d)
            print(f"    eps={eps:.2f}, area={eps*eps:.4f}, ||Δz||={d:.4f}")

        # Plot area vs displacement for this seed
        plt.figure(figsize=(5,4), dpi=140)
        plt.plot(areas, disp, "o-", linewidth=1.5)
        plt.xlabel("rectangle area (eps_x * eps_y) = eps^2")
        plt.ylabel("||z_end - z0||")
        plt.title(f"Loop displacement vs area (seed '{name}')")
        plt.grid(True, alpha=0.3)
        out_path = os.path.join(
            OUT_DIR, f"noncomm_area_scaling_seed_{name}_latent256.png"
        )
        print(f"{TAG} Saving {out_path}")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    print(f"{TAG} Done.")

if __name__ == "__main__":
    main()
