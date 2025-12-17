#!/usr/bin/env python3
# geomlang_phase10_latent_walks.py
#
# Phase 10B: Walk along discovered latent directions and decode images + head traces.
#
# Outputs:
#   outputs_edges_relternary256_phase10/walk_<name>_grid.png
#   outputs_edges_relternary256_phase10/walk_<name>_traces.png
#   outputs_edges_relternary256_phase10/phase10_walk_seeds.json

import os, json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

OUTDIR = "outputs_edges_relternary256_phase10"
os.makedirs(OUTDIR, exist_ok=True)

PHASE7_CKPT = "outputs_edges_relternary256_phase7/scene_model_edges_relternary256_phase7.pt"
DIRS_NPZ    = os.path.join(OUTDIR, "phase10_directions.npz")

LATENTS = "outputs_edges_relternary256_phase7/encoded_latents_seed123_N6000.npy"
TARGETS = "outputs_edges_relternary256_phase7/encoded_targets_seed123_N6000.npz"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABELS = [
    "A_left_of_BtoC",
    "A_right_of_BtoC",
    "A_between_clear",
    "A_between_overlap",
    "A_overlap_only",
]

def _npz_get(npz, wanted, aliases=()):
    keys = list(npz.files)
    for k in [wanted] + list(aliases):
        if k in npz.files:
            return npz[k]
    raise KeyError(f"Missing '{wanted}'. Tried { [wanted]+list(aliases) }. Available: {keys}")

def _ensure_1d(x, name):
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D (N,) but got {x.shape}")
    return x

def sigmoid_t(x):
    return torch.sigmoid(x)

# -----------------------
# Model (superset) for robust loading with strict=False
# -----------------------
class ConvAEHeadsSuperset(nn.Module):
    """
    Superset head model:
    - encoder/decoder must match checkpoint tensors
    - extra heads are okay (strict=False will ignore missing)
    """
    def __init__(self, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
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

        # Common names used across your phases
        self.h_between = mlp(1)
        self.h_tproj   = mlp(1)
        self.h_overlap = mlp(1)  # overlap_any logit
        self.h_lr      = mlp(1)  # lr_sign logit

        # extra heads (ignored if not in ckpt)
        self.h_csign = mlp(1)
        self.h_cmag  = mlp(1)
        self.h_oB    = mlp(1)
        self.h_oC    = mlp(1)

    def encode(self, x):
        h = self.enc(x).reshape(x.size(0), -1)
        return self.fc_mu(h)

    def decode(self, z):
        h = self.fc_dec(z).view(z.size(0), 256, 4, 4)
        return self.dec(h)

    def forward(self, x):
        z = self.encode(x)
        xhat = self.decode(z)
        return {
            "z": z, "xhat": xhat,
            "between": self.h_between(z),
            "tproj": self.h_tproj(z),
            "overlap": self.h_overlap(z),
            "lr": self.h_lr(z),
        }

def load_model():
    ckpt = torch.load(PHASE7_CKPT, map_location=DEVICE)
    latent_dim = int(ckpt.get("latent_dim", 256))
    model = ConvAEHeadsSuperset(latent_dim=latent_dim).to(DEVICE)
    sd = ckpt["model_state"] if "model_state" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("[phase10B] load_state_dict(strict=False)")
    if missing:
        print("  missing:", len(missing))
    if unexpected:
        print("  unexpected:", len(unexpected))
    model.eval()
    return model

# -----------------------
# Load directions + data
# -----------------------
dirs = np.load(DIRS_NPZ)
v_lr = dirs["v_lr"].astype(np.float32)
v_between = dirs["v_between"].astype(np.float32)
v_overlap = dirs["v_overlap"].astype(np.float32)
v_tproj = dirs["v_tproj"].astype(np.float32)

Z = np.load(LATENTS).astype(np.float32)
targets = np.load(TARGETS)

between = _ensure_1d(_npz_get(targets, "between_score"), "between_score")
tproj   = _ensure_1d(_npz_get(targets, "t_on_BC"), "t_on_BC")
overlap_any = _ensure_1d(_npz_get(targets, "overlap_any"), "overlap_any")
lr_sign = _ensure_1d(_npz_get(targets, "lr_sign"), "lr_sign")

# -----------------------
# Seed selection
# -----------------------
def pick_seed(mask, prefer_mid=True):
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        return int(np.random.randint(0, len(Z)))
    if not prefer_mid:
        return int(idxs[0])
    # choose one closest to median between score (or tproj) to avoid extremes
    b = between[idxs]
    med = np.median(b)
    j = idxs[np.argmin(np.abs(b - med))]
    return int(j)

mask_left_clean  = (between < 0.25) & (overlap_any < 0.25) & (lr_sign < 0.5)
mask_right_clean = (between < 0.25) & (overlap_any < 0.25) & (lr_sign >= 0.5)
mask_between_clear = (between > 0.75) & (overlap_any < 0.25)
mask_between_overlap = (between > 0.75) & (overlap_any >= 0.5)
mask_overlap_only = (between < 0.25) & (overlap_any >= 0.5)

seed_idxs = {
    "left_clean": pick_seed(mask_left_clean),
    "right_clean": pick_seed(mask_right_clean),
    "between_clear": pick_seed(mask_between_clear),
    "between_overlap": pick_seed(mask_between_overlap),
    "overlap_only": pick_seed(mask_overlap_only),
}
with open(os.path.join(OUTDIR, "phase10_walk_seeds.json"), "w") as f:
    json.dump(seed_idxs, f, indent=2)
print("[phase10B] seeds:", seed_idxs)

model = load_model()

# -----------------------
# Walk + decode
# -----------------------
def run_walk(name, z0, v, alphas):
    """
    Decode images and record head traces for z0 + alpha*v.
    """
    z0 = z0.astype(np.float32)
    v  = v.astype(np.float32)
    zs = np.stack([z0 + a*v for a in alphas], axis=0)
    zt = torch.tensor(zs, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        xhat = model.decode(zt).cpu().numpy()  # (T,3,64,64)
        # heads
        dummy = torch.zeros((len(alphas), 3, 64, 64), device=DEVICE)
        out = model(dummy)  # gives heads for encoded dummy; not what we want
        # Instead directly call heads on z:
        b = model.h_between(zt)
        t = model.h_tproj(zt)
        o = model.h_overlap(zt)
        lr = model.h_lr(zt)

        b = sigmoid_t(b).squeeze(1).cpu().numpy()
        t = sigmoid_t(t).squeeze(1).cpu().numpy()
        o = sigmoid_t(o).squeeze(1).cpu().numpy()
        lr = sigmoid_t(lr).squeeze(1).cpu().numpy()

    # Save image grid
    # Make a simple horizontal strip
    strip = np.concatenate([xhat[i].transpose(1,2,0) for i in range(len(alphas))], axis=1)
    plt.figure(figsize=(max(8, len(alphas)*0.8), 3))
    plt.imshow(strip)
    plt.axis("off")
    plt.title(f"{name}: decoded (alpha left→right)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"walk_{name}_grid.png"), dpi=160)
    plt.close()

    # Save traces
    plt.figure(figsize=(8,4))
    plt.plot(alphas, b, label="between_pred")
    plt.plot(alphas, o, label="overlap_prob")
    plt.plot(alphas, lr, label="lr_prob")
    plt.plot(alphas, t, label="tproj_pred")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title(f"{name}: head traces vs alpha")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f"walk_{name}_traces.png"), dpi=160)
    plt.close()

    return {"between": b, "overlap": o, "lr": lr, "tproj": t}

alphas = np.linspace(-4.0, 4.0, 17).astype(np.float32)

# Use sensible seeds per direction
z_left  = Z[seed_idxs["left_clean"]]
z_right = Z[seed_idxs["right_clean"]]
z_bc    = Z[seed_idxs["between_clear"]]
z_bo    = Z[seed_idxs["between_overlap"]]
z_o     = Z[seed_idxs["overlap_only"]]

print("[phase10B] walking v_lr from between_clear seed")
run_walk("v_lr_from_between_clear", z_bc, v_lr, alphas)

print("[phase10B] walking v_between from left_clean seed")
run_walk("v_between_from_left_clean", z_left, v_between, alphas)

print("[phase10B] walking v_overlap from between_clear seed")
run_walk("v_overlap_from_between_clear", z_bc, v_overlap, alphas)

print("[phase10B] walking v_tproj from between_clear seed")
run_walk("v_tproj_from_between_clear", z_bc, v_tproj, alphas)

print("[phase10B] saved walk images ->", OUTDIR)
