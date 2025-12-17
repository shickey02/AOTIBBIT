#!/usr/bin/env python3
# geomlang_phase10_geodesic_tests.py
#
# Phase 10C: Quantify “geodesic-ness” of latent walks:
# - monotonicity of head outputs
# - smoothness / curvature proxy via decoded image changes vs latent step size
#
# Outputs:
#   outputs_edges_relternary256_phase10/phase10_geodesic_report.json

import os, json
import numpy as np
import torch
import torch.nn as nn

OUTDIR = "outputs_edges_relternary256_phase10"
os.makedirs(OUTDIR, exist_ok=True)

PHASE7_CKPT = "outputs_edges_relternary256_phase7/scene_model_edges_relternary256_phase7.pt"
DIRS_NPZ    = os.path.join(OUTDIR, "phase10_directions.npz")
SEEDS_JSON  = os.path.join(OUTDIR, "phase10_walk_seeds.json")
LATENTS     = "outputs_edges_relternary256_phase7/encoded_latents_seed123_N6000.npy"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _unit(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + eps)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def monotonicity_score(y):
    """
    y: (T,) float
    returns fraction of consecutive steps matching global trend sign
    """
    y = np.asarray(y, dtype=np.float32)
    dy = np.diff(y)
    trend = np.sign(y[-1] - y[0])
    if trend == 0:
        return 1.0
    agree = np.mean(np.sign(dy + 1e-9) == trend)
    return float(agree)

def curvature_proxy(lat_step, img_steps):
    """
    Compare image-change per step to latent step magnitude.
    A “straight, uniform geodesic” tends to have relatively uniform img_steps.
    We report coefficient of variation (std/mean) as a curvature proxy.
    """
    img_steps = np.asarray(img_steps, dtype=np.float32)
    m = float(img_steps.mean() + 1e-9)
    s = float(img_steps.std())
    return {"img_step_mean": m, "img_step_std": s, "img_step_cv": float(s / m), "latent_step": float(lat_step)}

# Superset model for strict=False load
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
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model

# -------------------------
# Load data
# -------------------------
dirs = np.load(DIRS_NPZ)
v_lr = _unit(dirs["v_lr"])
v_between = _unit(dirs["v_between"])
v_overlap = _unit(dirs["v_overlap"])
v_tproj = _unit(dirs["v_tproj"])

Z = np.load(LATENTS).astype(np.float32)

if not os.path.exists(SEEDS_JSON):
    raise RuntimeError("Missing phase10_walk_seeds.json. Run geomlang_phase10_latent_walks.py first.")
with open(SEEDS_JSON, "r") as f:
    seed_idxs = json.load(f)

model = load_model()

# -------------------------
# Evaluate walks
# -------------------------
alphas = np.linspace(-4.0, 4.0, 33).astype(np.float32)

def eval_walk(z0, v):
    zs = np.stack([z0 + a*v for a in alphas], axis=0).astype(np.float32)
    zt = torch.tensor(zs, device=DEVICE)

    with torch.no_grad():
        xhat = model.decode(zt).cpu().numpy()  # (T,3,64,64)
        b = torch.sigmoid(model.h_between(zt)).squeeze(1).cpu().numpy()
        t = torch.sigmoid(model.h_tproj(zt)).squeeze(1).cpu().numpy()
        o = torch.sigmoid(model.h_overlap(zt)).squeeze(1).cpu().numpy()
        lr = torch.sigmoid(model.h_lr(zt)).squeeze(1).cpu().numpy()

    # latent step size (constant)
    lat_step = float(np.linalg.norm((alphas[1]-alphas[0]) * v))

    # image steps
    img_steps = []
    for i in range(len(alphas)-1):
        d = xhat[i+1] - xhat[i]
        img_steps.append(float(np.mean(d*d)))

    return {
        "monotonicity": {
            "between": monotonicity_score(b),
            "overlap": monotonicity_score(o),
            "lr": monotonicity_score(lr),
            "tproj": monotonicity_score(t),
        },
        "curvature_proxy": curvature_proxy(lat_step, img_steps),
        "head_ranges": {
            "between": [float(b.min()), float(b.max())],
            "overlap": [float(o.min()), float(o.max())],
            "lr": [float(lr.min()), float(lr.max())],
            "tproj": [float(t.min()), float(t.max())],
        }
    }

z_left  = Z[int(seed_idxs["left_clean"])]
z_bc    = Z[int(seed_idxs["between_clear"])]
z_bo    = Z[int(seed_idxs["between_overlap"])]
z_o     = Z[int(seed_idxs["overlap_only"])]

report = {
    "alphas": [float(a) for a in alphas],
    "walks": {
        "v_lr_from_between_clear": eval_walk(z_bc, v_lr),
        "v_between_from_left_clean": eval_walk(z_left, v_between),
        "v_overlap_from_between_clear": eval_walk(z_bc, v_overlap),
        "v_tproj_from_between_clear": eval_walk(z_bc, v_tproj),
    },
    "interpretation_notes": [
        "Monotonicity near 1.0 means the head output changes cleanly along that direction.",
        "img_step_cv closer to 0 means more uniform decoded change per latent step (straighter walk).",
        "If v_between changes overlap range a lot (or vice versa), those factors are entangled.",
    ]
}

with open(os.path.join(OUTDIR, "phase10_geodesic_report.json"), "w") as f:
    json.dump(report, f, indent=2)

print("[phase10C] saved ->", os.path.join(OUTDIR, "phase10_geodesic_report.json"))
print(json.dumps(report, indent=2))
