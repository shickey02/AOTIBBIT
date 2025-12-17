#!/usr/bin/env python3
# geomlang_phase11_latent_walks_strict.py
#
# Phase 11B: STRICT latent walks with improved boundary seed selection
# - Loads Phase7 latents/targets/preds
# - Loads Phase8 calibrated thresholds (Tb/To)
# - Computes directions: v_lr (mean-diff), v_between/v_overlap/v_tproj (ridge on Z)
# - Cleans entanglement: v_between_clean, v_overlap_clean (Gram-Schmidt)
# - Picks robust seeds near decision boundaries (NOT degenerate)
# - Decodes and saves walk grids + JSON report
#
# Outputs -> outputs_edges_relternary256_phase11/

import os, json, math
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

# -----------------------
# Model definition (match Phase7 training architecture)
# NOTE: This assumes Phase7 checkpoint includes decoder weights (it did in your strict-load runs).
# -----------------------
IMG_SIZE   = 64
LATENT_DIM = 256

class ConvAEHeads(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(inplace=True),   # 32x32
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(inplace=True),  # 16x16
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(inplace=True), # 8x8
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(inplace=True) # 4x4
        )
        self.fc_mu = nn.Linear(256*4*4, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, 256*4*4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(inplace=True), # 8x8
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True),  # 16x16
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(inplace=True),   # 32x32
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid()              # 64x64
        )

        def mlp(out_dim):
            return nn.Sequential(
                nn.Linear(latent_dim, 256), nn.ReLU(inplace=True),
                nn.Linear(256, 128), nn.ReLU(inplace=True),
                nn.Linear(128, out_dim)
            )

        # Phase7 heads
        self.h_between = mlp(1)  # reg -> sigmoid outside
        self.h_tproj   = mlp(1)  # reg -> sigmoid outside
        self.h_overlap = mlp(1)  # logit
        self.h_lr      = mlp(1)  # logit

    def encode(self, x):
        h = self.enc(x)
        h = h.reshape(h.size(0), -1)
        return self.fc_mu(h)

    def decode(self, z):
        h = self.fc_dec(z).view(z.size(0), 256, 4, 4)
        return self.dec(h)

    def forward(self, x):
        z = self.encode(x)
        xhat = self.decode(z)
        return {
            "z": z,
            "xhat": xhat,
            "between": self.h_between(z),
            "tproj":   self.h_tproj(z),
            "overlap": self.h_overlap(z),
            "lr":      self.h_lr(z),
        }

def sigmoid_np(x):
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-x))

def load_thresholds():
    with open(THRESH, "r") as f:
        d = json.load(f)
    Tb = float(d["Tb"])
    To = float(d["To"])
    return Tb, To

def unit(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n

def ridge_direction(Z, y, lam=1e-3):
    """
    Ridge regression direction on standardized Z and y.
    Returns unit vector v in latent space.
    """
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    Zm = Z.mean(axis=0, keepdims=True)
    Zs = Z.std(axis=0, keepdims=True) + 1e-9
    X  = (Z - Zm) / Zs

    ym = y.mean()
    ys = y.std() + 1e-9
    t  = (y - ym) / ys

    # (X^T X + lam I)^{-1} X^T t
    XT = X.T
    A = XT @ X
    A.flat[::A.shape[0] + 1] += lam
    b = XT @ t
    w = np.linalg.solve(A, b)
    # map back to unstandardized latent coords
    w = w / Zs.reshape(-1)
    return unit(w)

def mean_diff_direction(Z, y_binary):
    """
    v = mean(Z[y=1]) - mean(Z[y=0]) -> unit
    """
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y_binary).reshape(-1)
    a = Z[y > 0.5]
    b = Z[y <= 0.5]
    if len(a) == 0 or len(b) == 0:
        return unit(np.random.randn(Z.shape[1]))
    v = a.mean(axis=0) - b.mean(axis=0)
    return unit(v)

def gram_schmidt_clean(v_main, v_remove):
    """
    Remove projection of v_main onto v_remove (assumes v_remove is unit).
    """
    v_main = np.asarray(v_main, dtype=np.float64)
    v_remove = unit(v_remove)
    coeff = float(np.dot(v_main, v_remove))
    v_clean = v_main - coeff * v_remove
    return unit(v_clean), coeff

def choose_seed_by_score(mask, score):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None
    j = idx[np.argmin(score[idx])]
    return int(j)

def pick_improved_seeds(between_pred, overlap_prob, lr_prob, Tb, To):
    """
    Seed logic (robust + boundary-aware):
    - left_clean/right_clean: safely away from between and overlap
    - between_boundary_clear: close to Tb but overlap low
    - overlap_boundary_only: overlap high but between comfortably below Tb
    - between_overlap_boundary: close to Tb with overlap high
    """
    b = between_pred.reshape(-1)
    o = overlap_prob.reshape(-1)
    lr = lr_prob.reshape(-1)

    # "clean" margins
    FAR = 0.12
    LOWO = 0.35
    HIO  = 0.65

    # left/right clean: far below Tb and low overlap
    m_left  = (b < Tb - FAR) & (o < LOWO) & (lr < 0.5)
    m_right = (b < Tb - FAR) & (o < LOWO) & (lr > 0.5)

    left  = choose_seed_by_score(m_left,  (Tb - b) + 0.25*o + 0.10*np.abs(lr-0.0))
    right = choose_seed_by_score(m_right, (Tb - b) + 0.25*o + 0.10*np.abs(lr-1.0))

    if left is None:  left = 0
    if right is None: right = 1

    # boundary search with expanding eps windows
    eps_list = [0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10]

    between_boundary = None
    for eps in eps_list:
        m = (np.abs(b - Tb) < eps) & (o < LOWO)
        score = np.abs(b - Tb) + 0.50*o
        between_boundary = choose_seed_by_score(m, score)
        if between_boundary is not None:
            break
    if between_boundary is None:
        between_boundary = left

    overlap_boundary = None
    for eps in eps_list:
        # overlap high, but not between
        m = (o > HIO) & (b < Tb - 0.06) & (np.abs(b - (Tb - 0.10)) < eps)
        score = np.abs(b - (Tb - 0.10)) + 0.10*np.abs(o - 0.75)
        overlap_boundary = choose_seed_by_score(m, score)
        if overlap_boundary is not None:
            break
    if overlap_boundary is None:
        # fallback: any strong overlap but not between
        m = (o > HIO) & (b < Tb - 0.06)
        score = (Tb - b) + 0.10*np.abs(o - 0.80)
        overlap_boundary = choose_seed_by_score(m, score)
    if overlap_boundary is None:
        overlap_boundary = left

    between_overlap_boundary = None
    for eps in eps_list:
        m = (np.abs(b - Tb) < eps) & (o > HIO)
        score = np.abs(b - Tb) + 0.10*np.abs(o - 0.80)
        between_overlap_boundary = choose_seed_by_score(m, score)
        if between_overlap_boundary is not None:
            break
    if between_overlap_boundary is None:
        between_overlap_boundary = between_boundary

    return {
        "left_clean": left,
        "right_clean": right,
        "between_boundary_clear": between_boundary,
        "overlap_boundary_only": overlap_boundary,
        "between_overlap_boundary": between_overlap_boundary,
    }

@torch.no_grad()
def decode_batch(model, z_np, bs=256):
    z = torch.tensor(z_np, dtype=torch.float32, device=DEVICE)
    outs = []
    for i in range(0, z.shape[0], bs):
        zz = z[i:i+bs]
        x = model.decode(zz).detach().cpu()
        outs.append(x)
    return torch.cat(outs, dim=0)

@torch.no_grad()
def heads_on_batch(model, z_np, bs=256):
    z = torch.tensor(z_np, dtype=torch.float32, device=DEVICE)
    outs = {"between": [], "tproj": [], "overlap": [], "lr": []}
    for i in range(0, z.shape[0], bs):
        zz = z[i:i+bs]
        outs["between"].append(torch.sigmoid(model.h_between(zz)).detach().cpu())
        outs["tproj"].append(torch.sigmoid(model.h_tproj(zz)).detach().cpu())
        outs["overlap"].append(torch.sigmoid(model.h_overlap(zz)).detach().cpu())
        outs["lr"].append(torch.sigmoid(model.h_lr(zz)).detach().cpu())
    for k in outs:
        outs[k] = torch.cat(outs[k], dim=0).numpy().reshape(-1)
    return outs

def walk_and_save(model, Z, seed_idx, v, name, latent_step=0.25):
    alphas = np.linspace(-4.0, 4.0, 33)  # matches your phase10C
    z0 = Z[seed_idx]
    v = unit(v)

    Zwalk = np.stack([z0 + (a * latent_step) * v for a in alphas], axis=0)
    X = decode_batch(model, Zwalk, bs=128)

    grid = make_grid(X, nrow=len(alphas), padding=2)
    outpath = os.path.join(OUTDIR, name)
    save_image(grid, outpath)

    heads = heads_on_batch(model, Zwalk, bs=128)

    return {
        "seed": int(seed_idx),
        "alphas": [float(a) for a in alphas],
        "latent_step": float(latent_step),
        "head_ranges": {k: [float(np.min(vv)), float(np.max(vv))] for k, vv in heads.items()},
        "saved_image": outpath
    }

def main():
    Tb, To = load_thresholds()
    print(f"[phase11B] Tb={Tb:.6f} To={To:.6f}")

    Z = np.load(LATENTS)
    targets = np.load(TARGETS)
    preds   = np.load(PREDS)

    # Phase7 archive keys (confirmed from your logs)
    between_pred = preds["between_pred"].reshape(-1)
    overlap_prob = sigmoid_np(preds["overlap_logit"]).reshape(-1)
    lr_prob      = sigmoid_np(preds["lr_logit"]).reshape(-1)
    tproj_true   = targets["t_on_BC"].reshape(-1)
    between_true = targets["between_score"].reshape(-1)
    overlap_true = targets["overlap_any"].reshape(-1)
    lr_true      = targets["lr_sign"].reshape(-1)

    # Improved seeds (boundary aware)
    seeds = pick_improved_seeds(between_pred, overlap_prob, lr_prob, Tb, To)
    print("[phase11B] seeds:", seeds)

    # Directions
    v_lr = mean_diff_direction(Z, lr_true)                 # right-left in canonical space
    v_between = ridge_direction(Z, between_true, lam=1e-3)
    v_overlap = ridge_direction(Z, overlap_true, lam=1e-3)
    v_tproj   = ridge_direction(Z, tproj_true,   lam=1e-3)

    # Clean between/overlap entanglement (like your phase11 report)
    v_between_clean, coeff_bo = gram_schmidt_clean(v_between, v_overlap)
    v_overlap_clean, coeff_ob = gram_schmidt_clean(v_overlap, v_between)

    dot_before = float(np.dot(v_between, v_overlap))
    dot_after1 = float(np.dot(v_between_clean, v_overlap))
    dot_after2 = float(np.dot(v_overlap_clean, v_between))

    print("[phase11B] dot/coeff:", json.dumps({
        "v_between·v_overlap": dot_before,
        "v_between_clean·v_overlap": dot_after1,
        "v_overlap_clean·v_between": dot_after2,
        "coeff_between_on_overlap": float(coeff_bo),
        "coeff_overlap_on_between": float(coeff_ob),
    }, indent=2))

    # Load model STRICT
    ck = torch.load(CKPT, map_location=DEVICE)
    model = ConvAEHeads(LATENT_DIM).to(DEVICE)
    model.load_state_dict(ck["model_state"], strict=True)
    model.eval()
    print("[phase11B] strict-load OK")

    # Walks
    rep = {
        "Tb": Tb,
        "To": To,
        "seeds": seeds,
        "dot_coeff": {
            "v_between·v_overlap": dot_before,
            "v_between_clean·v_overlap": dot_after1,
            "v_overlap_clean·v_between": dot_after2,
            "coeff_between_on_overlap": float(coeff_bo),
            "coeff_overlap_on_between": float(coeff_ob),
        },
        "walks": {}
    }

    rep["walks"]["v_lr_from_between_boundary"] = walk_and_save(
        model, Z, seeds["between_boundary_clear"], v_lr,
        "phase11_walk_v_lr_from_between_boundary.png"
    )
    rep["walks"]["v_between_clean_from_between_boundary"] = walk_and_save(
        model, Z, seeds["between_boundary_clear"], v_between_clean,
        "phase11_walk_v_between_clean_from_between_boundary.png"
    )
    rep["walks"]["v_overlap_clean_from_overlap_boundary"] = walk_and_save(
        model, Z, seeds["overlap_boundary_only"], v_overlap_clean,
        "phase11_walk_v_overlap_clean_from_overlap_boundary.png"
    )
    rep["walks"]["v_tproj_from_between_boundary"] = walk_and_save(
        model, Z, seeds["between_boundary_clear"], v_tproj,
        "phase11_walk_v_tproj_from_between_boundary.png"
    )

    out_json = os.path.join(OUTDIR, "phase11_walk_report.json")
    with open(out_json, "w") as f:
        json.dump(rep, f, indent=2)

    print("[phase11B] saved report ->", out_json)
    print("[phase11B] done ->", OUTDIR)

if __name__ == "__main__":
    main()
