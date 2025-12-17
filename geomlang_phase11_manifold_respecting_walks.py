#!/usr/bin/env python3
# geomlang_phase11_manifold_respecting_walks.py
#
# Phase 11C: Manifold-respecting walks (decode -> encode) at each step.
# This prevents latent addition from drifting into "non-representable" areas.

import os, json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

OUTDIR = "outputs_edges_relternary256_phase11"
os.makedirs(OUTDIR, exist_ok=True)

PHASE7_CKPT = "outputs_edges_relternary256_phase7/scene_model_edges_relternary256_phase7.pt"
PH11_DIRS   = os.path.join(OUTDIR, "phase11_directions_ortho.npz")
PH10_SEEDS  = "outputs_edges_relternary256_phase10/phase10_walk_seeds.json"
LATENTS     = "outputs_edges_relternary256_phase7/encoded_latents_seed123_N6000.npy"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def unit(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + eps)

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

    def encode(self, x):
        h = self.enc(x).reshape(x.size(0), -1)
        return self.fc_mu(h)

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

def plot_traces(alphas, traces, title, outpath):
    plt.figure(figsize=(8,4))
    for k, v in traces.items():
        plt.plot(alphas, v, label=k)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=170)
    plt.close()

def save_strip(imgs, title, outpath):
    strip = np.concatenate([imgs[i].transpose(1,2,0) for i in range(len(imgs))], axis=1)
    plt.figure(figsize=(max(8, len(imgs)*0.8), 3))
    plt.imshow(strip)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=170)
    plt.close()

# -----------------------
# Load
# -----------------------
dirs = np.load(PH11_DIRS)
v_lr = unit(dirs["v_lr"])
v_overlap = unit(dirs["v_overlap"])
v_between_clean = unit(dirs["v_between_clean"])
v_tproj = unit(dirs["v_tproj"])

Z = np.load(LATENTS).astype(np.float32)
with open(PH10_SEEDS, "r") as f:
    seeds = json.load(f)

model = load_model()

def manifold_walk(z0, v, alphas, reencode=True):
    """
    If reencode=True:
      z_{k+1} = encode(decode(z_k + step*v))
    else:
      z_{k} = z0 + alpha*v
    """
    z = torch.tensor(z0[None, :], device=DEVICE)
    imgs = []
    trace_between, trace_overlap, trace_lr, trace_tproj = [], [], [], []

    prev_a = alphas[0]
    for a in alphas:
        step = float(a - prev_a)
        prev_a = a
        if reencode:
            z = z + step * torch.tensor(v[None, :], device=DEVICE)
            with torch.no_grad():
                x = model.decode(z)
                z = model.encode(x)  # project back onto manifold
        else:
            z = torch.tensor((z0 + a*v)[None, :], device=DEVICE)
            with torch.no_grad():
                x = model.decode(z)

        with torch.no_grad():
            imgs.append(x.squeeze(0).cpu().numpy())
            trace_between.append(float(torch.sigmoid(model.h_between(z)).item()))
            trace_overlap.append(float(torch.sigmoid(model.h_overlap(z)).item()))
            trace_lr.append(float(torch.sigmoid(model.h_lr(z)).item()))
            trace_tproj.append(float(torch.sigmoid(model.h_tproj(z)).item()))

    traces = {
        "between": trace_between,
        "overlap": trace_overlap,
        "lr": trace_lr,
        "tproj": trace_tproj,
    }
    return np.stack(imgs, axis=0), traces

# Use a wider alpha range than Phase10 (this matters)
alphas = np.linspace(-20.0, 20.0, 41).astype(np.float32)

z_left = Z[int(seeds["left_clean"])]
z_between = Z[int(seeds["between_clear"])]

runs = [
    ("betweenClean_from_left_clean", z_left, v_between_clean),
    ("overlap_from_between_clear", z_between, v_overlap),
    ("lr_from_between_clear", z_between, v_lr),
    ("tproj_from_between_clear", z_between, v_tproj),
]

summary = {}

for name, z0, v in runs:
    print("[phase11C] walk:", name)

    imgs_free, traces_free = manifold_walk(z0, v, alphas, reencode=False)
    imgs_proj, traces_proj = manifold_walk(z0, v, alphas, reencode=True)

    save_strip(imgs_free, f"{name} (free walk)", os.path.join(OUTDIR, f"{name}_free_strip.png"))
    plot_traces(alphas, traces_free, f"{name} (free walk) traces", os.path.join(OUTDIR, f"{name}_free_traces.png"))

    save_strip(imgs_proj, f"{name} (projected walk)", os.path.join(OUTDIR, f"{name}_proj_strip.png"))
    plot_traces(alphas, traces_proj, f"{name} (projected walk) traces", os.path.join(OUTDIR, f"{name}_proj_traces.png"))

    summary[name] = {
        "free_ranges": {k: [float(np.min(vv)), float(np.max(vv))] for k, vv in traces_free.items()},
        "proj_ranges": {k: [float(np.min(vv)), float(np.max(vv))] for k, vv in traces_proj.items()},
        "alpha_range": [float(alphas.min()), float(alphas.max())],
        "N": int(len(alphas))
    }

with open(os.path.join(OUTDIR, "phase11_manifold_walks_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("[phase11C] saved ->", OUTDIR)
print(json.dumps(summary, indent=2))
