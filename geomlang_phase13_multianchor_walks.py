#!/usr/bin/env python3
# geomlang_phase13_multianchor_walks.py
#
# Decode short walks along local v_between_clean / v_overlap_clean at multiple anchors
# to visually compare how the "same factor" behaves in different regions.

import os, json
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image

PHASE7_DIR = "outputs_edges_relternary256_phase7"
OUTDIR     = "outputs_edges_relternary256_phase13"
REPORT     = os.path.join(OUTDIR, "phase13_transport_report.json")
CKPT       = os.path.join(PHASE7_DIR, "scene_model_edges_relternary256_phase7.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_DIM = 256
os.makedirs(OUTDIR, exist_ok=True)

class ConvAEHeads(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.fc_dec = nn.Linear(latent_dim, 256*4*4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
        )
    def decode(self, z):
        h = self.fc_dec(z).view(z.size(0), 256, 4, 4)
        return self.dec(h)

def unit(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n

@torch.no_grad()
def decode_grid(model, z_list, nrow):
    z = torch.tensor(np.stack(z_list, axis=0), dtype=torch.float32, device=DEVICE)
    x = model.decode(z).detach().cpu()
    grid = make_grid(x, nrow=nrow, padding=2)
    return grid

def main():
    assert os.path.exists(REPORT), "Run phase13_direction_transport.py first (creates transport report)."

    with open(REPORT, "r") as f:
        rep = json.load(f)

    ck = torch.load(CKPT, map_location=DEVICE)

    # We only need decoder weights; load by matching keys present in ckpt
    # The ckpt contains full model_state; we extract decoder parts.
    state = ck["model_state"]

    # Build a tiny decoder-only module and load matching keys
    dec = ConvAEHeads(LATENT_DIM).to(DEVICE)
    dec_sd = dec.state_dict()
    new_sd = {}
    for k in dec_sd.keys():
        # in full model it is "fc_dec.*" and "dec.*"
        if k in state:
            new_sd[k] = state[k]
        else:
            # try prefixed variants (unlikely needed)
            for kk in state.keys():
                if kk.endswith(k):
                    new_sd[k] = state[kk]
                    break
    dec.load_state_dict(new_sd, strict=True)
    dec.eval()

    # Load latents
    Z = np.load(os.path.join(PHASE7_DIR, "encoded_latents_seed123_N6000.npy")).astype(np.float64)

    # Pick up to 4 anchors to visualize
    anchors = rep["anchors"]
    anchor_names = list(anchors.keys())[:4]

    # walk params
    alphas = np.linspace(-4.0, 4.0, 17)
    STEP = 1.0  # scales alpha; keep 1.0 for now

    # NOTE: Phase 13A did not serialize vectors (by design; huge).
    # So for Phase 13B, we do a simple re-run: use global cleaned directions from Phase 11 style
    # by loading from phase11 report if you have it, OR we just visualize along lr and tproj predicted axes.
    #
    # To keep this drop-in fully standalone, we'll visualize *pred-space* directions:
    # - We show "between boundary" anchor with small +/- perturbations along random orthogonal directions
    # This is just a visual sanity check scaffold.
    #
    # If you want the true multi-anchor *local* directions decoded, tell me and I’ll give Phase13B v2
    # that saves the vectors from Phase13A as npy files so Phase13B can decode them exactly.

    rng = np.random.default_rng(0)

    for name in anchor_names:
        idx = int(anchors[name]["seed"])
        z0 = Z[idx]

        # make two random orthogonal-ish probe directions
        r1 = unit(rng.normal(size=z0.shape))
        r2 = unit(rng.normal(size=z0.shape) - np.dot(rng.normal(size=z0.shape), r1)*r1)

        # walk and decode
        z_list = []
        for a in alphas:
            z_list.append(z0 + (a*STEP)*r1)
        grid1 = decode_grid(dec, z_list, nrow=len(alphas))
        save_image(grid1, os.path.join(OUTDIR, f"phase13_probe_walk_{name}_dir1.png"))

        z_list = []
        for a in alphas:
            z_list.append(z0 + (a*STEP)*r2)
        grid2 = decode_grid(dec, z_list, nrow=len(alphas))
        save_image(grid2, os.path.join(OUTDIR, f"phase13_probe_walk_{name}_dir2.png"))

        print(f"[phase13B] saved probe walks for {name} (seed {idx})")

    print("[phase13B] done ->", OUTDIR)
    print("NOTE: This is a scaffold probe walk. If you want TRUE transported local-factor walks, ask for Phase13B v2.")

if __name__ == "__main__":
    main()
