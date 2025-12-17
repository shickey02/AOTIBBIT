#!/usr/bin/env python3
# geomlang_phase12_commutativity_test.py
#
# Phase 12 (improved): Commutativity / path dependence test
# - Uses Phase 7 latents/preds/targets
# - Loads Phase 8 thresholds Tb/To
# - Builds directions v_between_clean and v_overlap_clean (Gram–Schmidt)
# - Picks a non-degenerate seed automatically:
#     prefer between_boundary_clear (closest to Tb with low overlap),
#     else overlap_boundary_only, else left_clean.
# - Compares endpoints of two paths:
#     Path1: +a*v_between_clean then +b*v_overlap_clean
#     Path2: +b*v_overlap_clean then +a*v_between_clean
# - Saves a grid image showing the two paths and reports endpoint diffs.
#
# Output -> outputs_edges_relternary256_phase12/

import os, json
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image

PHASE7_DIR = "outputs_edges_relternary256_phase7"
PHASE8_DIR = "outputs_edges_relternary256_phase8"
OUTDIR     = "outputs_edges_relternary256_phase12"
os.makedirs(OUTDIR, exist_ok=True)

LATENTS = os.path.join(PHASE7_DIR, "encoded_latents_seed123_N6000.npy")
TARGETS = os.path.join(PHASE7_DIR, "encoded_targets_seed123_N6000.npz")
PREDS   = os.path.join(PHASE7_DIR, "encoded_preds_seed123_N6000.npz")
CKPT    = os.path.join(PHASE7_DIR, "scene_model_edges_relternary256_phase7.pt")
THRESH  = os.path.join(PHASE8_DIR, "phase8_thresholds.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_DIM = 256

# -----------------------
# Model (must match Phase7)
# -----------------------
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

def gram_schmidt_clean(v_main, v_remove):
    v_remove = unit(v_remove)
    coeff = float(np.dot(v_main, v_remove))
    v_clean = v_main - coeff * v_remove
    return unit(v_clean), coeff

def load_thresholds():
    with open(THRESH, "r") as f:
        d = json.load(f)
    return float(d["Tb"]), float(d["To"])

def choose_seed(mask, score):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None
    return int(idx[np.argmin(score[idx])])

def pick_boundary_seed(between_pred, overlap_prob, lr_prob, Tb):
    """
    Prefer a seed near between boundary with low overlap.
    If none, use overlap boundary (overlap high, between low),
    else use a left_clean style seed.
    """
    b = between_pred.reshape(-1)
    o = overlap_prob.reshape(-1)
    lr = lr_prob.reshape(-1)

    LOWO = 0.35
    HIO  = 0.65

    eps_list = [0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10]

    # between boundary clear
    for eps in eps_list:
        m = (np.abs(b - Tb) < eps) & (o < LOWO)
        s = np.abs(b - Tb) + 0.50*o
        j = choose_seed(m, s)
        if j is not None:
            return j, "between_boundary_clear"

    # overlap boundary only-ish
    m = (o > HIO) & (b < Tb - 0.06)
    s = (Tb - b) + 0.10*np.abs(o - 0.80)
    j = choose_seed(m, s)
    if j is not None:
        return j, "overlap_boundary_only"

    # left_clean fallback
    m = (b < Tb - 0.12) & (o < LOWO) & (lr < 0.5)
    s = (Tb - b) + 0.25*o
    j = choose_seed(m, s)
    if j is not None:
        return j, "left_clean"

    return 0, "seed0_fallback"

@torch.no_grad()
def decode_imgs(model, Zlist, bs=256):
    z = torch.tensor(np.stack(Zlist, axis=0), dtype=torch.float32, device=DEVICE)
    imgs = []
    for i in range(0, z.shape[0], bs):
        imgs.append(model.decode(z[i:i+bs]).detach().cpu())
    return torch.cat(imgs, dim=0)

@torch.no_grad()
def heads(model, z_np):
    z = torch.tensor(z_np[None, :], dtype=torch.float32, device=DEVICE)
    out = {
        "between": torch.sigmoid(model.h_between(z)).item(),
        "overlap": torch.sigmoid(model.h_overlap(z)).item(),
        "lr":      torch.sigmoid(model.h_lr(z)).item(),
        "tproj":   torch.sigmoid(model.h_tproj(z)).item(),
    }
    return out

def l2(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b))

def main():
    Tb, To = load_thresholds()
    Z = np.load(LATENTS)
    targets = np.load(TARGETS)
    preds   = np.load(PREDS)

    between_pred = preds["between_pred"].reshape(-1)
    overlap_prob = sigmoid_np(preds["overlap_logit"]).reshape(-1)
    lr_prob      = sigmoid_np(preds["lr_logit"]).reshape(-1)

    between_true = targets["between_score"].reshape(-1)
    overlap_true = targets["overlap_any"].reshape(-1)

    # directions + cleaning
    v_between = ridge_direction(Z, between_true, lam=1e-3)
    v_overlap = ridge_direction(Z, overlap_true, lam=1e-3)
    v_between_clean, _ = gram_schmidt_clean(v_between, v_overlap)
    v_overlap_clean, _ = gram_schmidt_clean(v_overlap, v_between)

    # pick better seed
    seed, seed_kind = pick_boundary_seed(between_pred, overlap_prob, lr_prob, Tb)
    print(f"[phase12] Tb={Tb:.6f} To={To:.6f} | seed={seed} ({seed_kind})")

    # load model strict
    ck = torch.load(CKPT, map_location=DEVICE)
    model = ConvAEHeads(LATENT_DIM).to(DEVICE)
    model.load_state_dict(ck["model_state"], strict=True)
    model.eval()

    # steps (tune these)
    a = 3.0
    b = 3.0

    z0 = Z[seed].astype(np.float64)

    # path 1: between then overlap
    zA = z0 + a*v_between_clean
    zAB = zA + b*v_overlap_clean

    # path 2: overlap then between
    zB = z0 + b*v_overlap_clean
    zBA = zB + a*v_between_clean

    # decode comparison
    imgs = decode_imgs(model, [z0, zA, zAB, zB, zBA], bs=64)
    # grid: row = 5 images
    grid = make_grid(imgs, nrow=5, padding=2)
    grid_path = os.path.join(OUTDIR, "phase12_path_grids.png")
    save_image(grid, grid_path)

    # endpoint diffs
    endpoint_latent_l2 = l2(zAB, zBA)
    endpoint_img_l2 = float(torch.norm(imgs[2] - imgs[4]).item())

    hAB = heads(model, zAB)
    hBA = heads(model, zBA)

    head_diffs = {k: float(abs(hAB[k] - hBA[k])) for k in hAB}

    report = {
        "Tb": Tb,
        "To": To,
        "seed": int(seed),
        "seed_kind": seed_kind,
        "steps": {"a_between_clean": a, "b_overlap_clean": b},
        "endpoint_latent_l2": endpoint_latent_l2,
        "endpoint_img_l2": endpoint_img_l2,
        "head_endpoint_diffs": head_diffs,
        "grid_image": grid_path,
        "interpretation": [
            "endpoint_latent_l2 should be ~0 (same endpoint by construction).",
            "endpoint_img_l2 and head_endpoint_diffs reveal path dependence / curvature effects.",
            "If endpoint_img_l2 ~0 and head diffs ~0, factors commute (at least locally)."
        ]
    }

    out_json = os.path.join(OUTDIR, "phase12_commutativity_report.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    print("[phase12] saved ->", OUTDIR)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
