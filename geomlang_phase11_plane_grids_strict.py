#!/usr/bin/env python3
# geomlang_phase11_plane_grids_strict.py
#
# Phase 11C: STRICT 2D plane grids with improved boundary seeds
# - Uses cleaned between/overlap directions
# - Saves image grids for:
#   1) betweenClean vs overlapClean (seed=between boundary)
#   2) lr vs tproj (seed=left clean)
#   3) betweenClean vs overlapClean (seed=overlap boundary)

import os, json
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image

# -----------------------
# Paths
# -----------------------
PHASE7_DIR = "outputs_edges_relternary256_phase7"
PHASE8_DIR = "outputs_edges_relternary256_phase8"
OUTDIR     = "outputs_edges_relternary256_phase11"
os.makedirs(OUTDIR, exist_ok=True)

LATENTS = os.path.join(PHASE7_DIR, "encoded_latents_seed123_N6000.npy")
TARGETS = os.path.join(PHASE7_DIR, "encoded_targets_seed123_N6000.npz")
PREDS   = os.path.join(PHASE7_DIR, "encoded_preds_seed123_N6000.npz")
CKPT    = os.path.join(PHASE7_DIR, "scene_model_edges_relternary256_phase7.pt")
THRESH  = os.path.join(PHASE8_DIR, "phase8_thresholds.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE   = 64
LATENT_DIM = 256

class ConvAEHeads(nn.Module):
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
                nn.Linear(128, out_dim)
            )

        self.h_between = mlp(1)
        self.h_tproj   = mlp(1)
        self.h_overlap = mlp(1)
        self.h_lr      = mlp(1)

    def decode(self, z):
        h = self.fc_dec(z).view(z.size(0), 256, 4, 4)
        return self.dec(h)

def sigmoid_np(x):
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-x))

def load_thresholds():
    with open(THRESH, "r") as f:
        d = json.load(f)
    return float(d["Tb"]), float(d["To"])

def unit(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n

def ridge_direction(Z, y, lam=1e-3):
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    Zm = Z.mean(axis=0, keepdims=True)
    Zs = Z.std(axis=0, keepdims=True) + 1e-9
    X  = (Z - Zm) / Zs
    ym = y.mean()
    ys = y.std() + 1e-9
    t  = (y - ym) / ys
    XT = X.T
    A = XT @ X
    A.flat[::A.shape[0] + 1] += lam
    b = XT @ t
    w = np.linalg.solve(A, b)
    w = w / Zs.reshape(-1)
    return unit(w)

def mean_diff_direction(Z, y_binary):
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y_binary).reshape(-1)
    a = Z[y > 0.5]
    b = Z[y <= 0.5]
    if len(a) == 0 or len(b) == 0:
        return unit(np.random.randn(Z.shape[1]))
    v = a.mean(axis=0) - b.mean(axis=0)
    return unit(v)

def gram_schmidt_clean(v_main, v_remove):
    v_remove = unit(v_remove)
    coeff = float(np.dot(v_main, v_remove))
    v_clean = v_main - coeff * v_remove
    return unit(v_clean)

def choose_seed_by_score(mask, score):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None
    j = idx[np.argmin(score[idx])]
    return int(j)

def pick_improved_seeds(between_pred, overlap_prob, lr_prob, Tb, To):
    b = between_pred.reshape(-1)
    o = overlap_prob.reshape(-1)
    lr = lr_prob.reshape(-1)

    FAR = 0.12
    LOWO = 0.35
    HIO  = 0.65

    m_left  = (b < Tb - FAR) & (o < LOWO) & (lr < 0.5)
    m_right = (b < Tb - FAR) & (o < LOWO) & (lr > 0.5)
    left  = choose_seed_by_score(m_left,  (Tb - b) + 0.25*o + 0.10*np.abs(lr-0.0))
    right = choose_seed_by_score(m_right, (Tb - b) + 0.25*o + 0.10*np.abs(lr-1.0))
    if left is None: left = 0
    if right is None: right = 1

    eps_list = [0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10]

    between_boundary = None
    for eps in eps_list:
        m = (np.abs(b - Tb) < eps) & (o < LOWO)
        score = np.abs(b - Tb) + 0.50*o
        between_boundary = choose_seed_by_score(m, score)
        if between_boundary is not None: break
    if between_boundary is None:
        between_boundary = left

    overlap_boundary = None
    for eps in eps_list:
        m = (o > HIO) & (b < Tb - 0.06) & (np.abs(b - (Tb - 0.10)) < eps)
        score = np.abs(b - (Tb - 0.10)) + 0.10*np.abs(o - 0.75)
        overlap_boundary = choose_seed_by_score(m, score)
        if overlap_boundary is not None: break
    if overlap_boundary is None:
        m = (o > HIO) & (b < Tb - 0.06)
        score = (Tb - b) + 0.10*np.abs(o - 0.80)
        overlap_boundary = choose_seed_by_score(m, score)
    if overlap_boundary is None:
        overlap_boundary = left

    return {
        "left_clean": left,
        "right_clean": right,
        "between_boundary_clear": between_boundary,
        "overlap_boundary_only": overlap_boundary,
    }

@torch.no_grad()
def decode_grid(model, Z0, v1, v2, span=12.0, gridN=21, bs=256):
    """
    span is total extent in latent units along each axis.
    """
    Z0 = np.asarray(Z0, dtype=np.float64)
    v1 = unit(v1)
    v2 = unit(v2)

    xs = np.linspace(-span/2, span/2, gridN)
    ys = np.linspace(-span/2, span/2, gridN)

    latents = []
    for yy in ys:
        for xx in xs:
            latents.append(Z0 + xx*v1 + yy*v2)
    latents = np.stack(latents, axis=0).astype(np.float32)

    z = torch.tensor(latents, device=DEVICE)
    imgs = []
    for i in range(0, z.shape[0], bs):
        x = model.decode(z[i:i+bs]).detach().cpu()
        imgs.append(x)
    imgs = torch.cat(imgs, dim=0)  # (gridN*gridN, 3, 64, 64)
    return imgs, xs, ys

def save_plane(model, Z, seed_idx, v1, v2, span, gridN, fname):
    imgs, xs, ys = decode_grid(model, Z[seed_idx], v1, v2, span=span, gridN=gridN, bs=256)
    grid = make_grid(imgs, nrow=gridN, padding=2)
    outpath = os.path.join(OUTDIR, fname)
    save_image(grid, outpath)
    return outpath

def main():
    Tb, To = load_thresholds()
    print(f"[phase11C] Tb={Tb:.6f} To={To:.6f}")

    Z = np.load(LATENTS)
    targets = np.load(TARGETS)
    preds   = np.load(PREDS)

    between_pred = preds["between_pred"].reshape(-1)
    overlap_prob = sigmoid_np(preds["overlap_logit"]).reshape(-1)
    lr_prob      = sigmoid_np(preds["lr_logit"]).reshape(-1)

    between_true = targets["between_score"].reshape(-1)
    overlap_true = targets["overlap_any"].reshape(-1)
    tproj_true   = targets["t_on_BC"].reshape(-1)
    lr_true      = targets["lr_sign"].reshape(-1)

    seeds = pick_improved_seeds(between_pred, overlap_prob, lr_prob, Tb, To)
    print("[phase11C] seeds:", seeds)

    v_lr = mean_diff_direction(Z, lr_true)
    v_between = ridge_direction(Z, between_true, lam=1e-3)
    v_overlap = ridge_direction(Z, overlap_true, lam=1e-3)
    v_tproj   = ridge_direction(Z, tproj_true,   lam=1e-3)

    v_between_clean = gram_schmidt_clean(v_between, v_overlap)
    v_overlap_clean = gram_schmidt_clean(v_overlap, v_between)

    ck = torch.load(CKPT, map_location=DEVICE)
    model = ConvAEHeads(LATENT_DIM).to(DEVICE)
    model.load_state_dict(ck["model_state"], strict=True)
    model.eval()
    print("[phase11C] strict-load OK")

    span = 12.0
    gridN = 21

    p1 = save_plane(
        model, Z, seeds["between_boundary_clear"],
        v_between_clean, v_overlap_clean,
        span, gridN,
        "phase11_plane_betweenClean_vs_overlapClean_seedBetweenBoundary_images.png"
    )
    print("[phase11C] saved:", p1)

    p2 = save_plane(
        model, Z, seeds["left_clean"],
        v_lr, v_tproj,
        span, gridN,
        "phase11_plane_lr_vs_tproj_seedLeftClean_images.png"
    )
    print("[phase11C] saved:", p2)

    p3 = save_plane(
        model, Z, seeds["overlap_boundary_only"],
        v_between_clean, v_overlap_clean,
        span, gridN,
        "phase11_plane_betweenClean_vs_overlapClean_seedOverlapBoundary_images.png"
    )
    print("[phase11C] saved:", p3)

    print("[phase11C] done ->", OUTDIR)

if __name__ == "__main__":
    main()
