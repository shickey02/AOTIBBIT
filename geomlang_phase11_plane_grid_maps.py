#!/usr/bin/env python3
# geomlang_phase11_plane_grid_maps.py
#
# Phase 11B: 2D plane maps in latent space.
# - Plane 1: (v_between_clean, v_overlap) from a between-clear seed
# - Plane 2: (v_lr, v_tproj) from a non-between seed
#
# Saves decoded grids + head heatmaps.

import os, json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

OUTDIR = "outputs_edges_relternary256_phase11"
os.makedirs(OUTDIR, exist_ok=True)

PHASE7_CKPT = "outputs_edges_relternary256_phase7/scene_model_edges_relternary256_phase7.pt"

PH11_DIRS = os.path.join(OUTDIR, "phase11_directions_ortho.npz")
PH10_SEEDS = "outputs_edges_relternary256_phase10/phase10_walk_seeds.json"
LATENTS = "outputs_edges_relternary256_phase7/encoded_latents_seed123_N6000.npy"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def unit(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + eps)

# Superset model that can decode + run the phase7 heads (strict=False)
class ConvAEHeadsSuperset(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(256*4*4, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 256*4*4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
        )
        def mlp(out_dim):
            return nn.Sequential(
                nn.Linear(latent_dim, 256), nn.ReLU(inplace=True),
                nn.Linear(256, 128), nn.ReLU(inplace=True),
                nn.Linear(128, out_dim),
            )
        self.h_between = mlp(1)
        self.h_tproj   = mlp(1)
        self.h_overlap = mlp(1)
        self.h_lr      = mlp(1)

    def decode(self, z):
        h = self.fc_dec(z).view(z.size(0), 256, 4, 4)
        return self.dec(h)

def load_model():
    ckpt = torch.load(PHASE7_CKPT, map_location=DEVICE)
    latent_dim = int(ckpt.get("latent_dim", 256))
    model = ConvAEHeadsSuperset(latent_dim=latent_dim).to(DEVICE)
    sd = ckpt["model_state"] if "model_state" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("[phase11B] load_state_dict(strict=False) missing:", len(missing), "unexpected:", len(unexpected))
    model.eval()
    return model

def save_heatmap(M, title, path):
    plt.figure(figsize=(6,5))
    plt.imshow(M, origin="lower", aspect="equal")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()

def save_decode_grid(imgs, gridN, title, path):
    # imgs: (gridN*gridN, 3, 64, 64)
    # assemble into big image
    tiles = [imgs[i].transpose(1,2,0) for i in range(imgs.shape[0])]
    rows = []
    for r in range(gridN):
        row = np.concatenate(tiles[r*gridN:(r+1)*gridN], axis=1)
        rows.append(row)
    big = np.concatenate(rows, axis=0)

    plt.figure(figsize=(10,10))
    plt.imshow(big)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()

# -----------------------
# Load resources
# -----------------------
dirs = np.load(PH11_DIRS)
v_lr = unit(dirs["v_lr"])
v_tproj = unit(dirs["v_tproj"])
v_overlap = unit(dirs["v_overlap"])
v_between_clean = unit(dirs["v_between_clean"])

Z = np.load(LATENTS).astype(np.float32)
with open(PH10_SEEDS, "r") as f:
    seeds = json.load(f)

z_between_seed = Z[int(seeds["between_clear"])]
z_left_seed    = Z[int(seeds["left_clean"])]

model = load_model()

# -----------------------
# Plane builder
# -----------------------
def eval_plane(z0, vx, vy, span=10.0, gridN=21, name="plane"):
    xs = np.linspace(-span, span, gridN).astype(np.float32)
    ys = np.linspace(-span, span, gridN).astype(np.float32)

    zs = []
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            zs.append(z0 + x*vx + y*vy)
    zs = np.stack(zs, axis=0).astype(np.float32)

    zt = torch.tensor(zs, device=DEVICE)
    with torch.no_grad():
        xhat = model.decode(zt).cpu().numpy()
        between = torch.sigmoid(model.h_between(zt)).squeeze(1).cpu().numpy()
        overlap = torch.sigmoid(model.h_overlap(zt)).squeeze(1).cpu().numpy()
        lr      = torch.sigmoid(model.h_lr(zt)).squeeze(1).cpu().numpy()
        tproj   = torch.sigmoid(model.h_tproj(zt)).squeeze(1).cpu().numpy()

    # reshape to grids
    def grid(v): return v.reshape(gridN, gridN)

    save_decode_grid(xhat, gridN, f"{name} decoded", os.path.join(OUTDIR, f"{name}_decoded.png"))
    save_heatmap(grid(between), f"{name}: between", os.path.join(OUTDIR, f"{name}_between.png"))
    save_heatmap(grid(overlap), f"{name}: overlap", os.path.join(OUTDIR, f"{name}_overlap.png"))
    save_heatmap(grid(lr),      f"{name}: lr",      os.path.join(OUTDIR, f"{name}_lr.png"))
    save_heatmap(grid(tproj),   f"{name}: tproj",   os.path.join(OUTDIR, f"{name}_tproj.png"))

    # quick summary stats
    return {
        "between_range": [float(between.min()), float(between.max())],
        "overlap_range": [float(overlap.min()), float(overlap.max())],
        "lr_range": [float(lr.min()), float(lr.max())],
        "tproj_range": [float(tproj.min()), float(tproj.max())],
        "span": span,
        "gridN": gridN
    }

summary = {}
print("[phase11B] Plane1: (v_between_clean, v_overlap) from between_clear seed")
summary["plane_between_clean_vs_overlap"] = eval_plane(
    z_between_seed, v_between_clean, v_overlap, span=12.0, gridN=21, name="plane_betweenClean_vs_overlap"
)

print("[phase11B] Plane2: (v_lr, v_tproj) from left_clean seed")
summary["plane_lr_vs_tproj"] = eval_plane(
    z_left_seed, v_lr, v_tproj, span=12.0, gridN=21, name="plane_lr_vs_tproj"
)

with open(os.path.join(OUTDIR, "phase11_plane_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("[phase11B] saved ->", OUTDIR)
print(json.dumps(summary, indent=2))
