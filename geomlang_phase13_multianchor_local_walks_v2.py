#!/usr/bin/env python3
# geomlang_phase13_multianchor_local_walks_v2.py
#
# Decodes walks along the SAVED Phase13A-v2 local vectors
# (v_between_clean / v_overlap_clean / v_lr / v_tproj) at multiple anchors.

import os, json
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image

PHASE7_DIR = "outputs_edges_relternary256_phase7"
OUTDIR     = "outputs_edges_relternary256_phase13"
REPORT     = os.path.join(OUTDIR, "phase13_transport_report_v2.json")
CKPT       = os.path.join(PHASE7_DIR, "scene_model_edges_relternary256_phase7.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATENT_DIM = 256
os.makedirs(OUTDIR, exist_ok=True)

class DecoderOnly(nn.Module):
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

def load_decoder_from_ckpt():
    ck = torch.load(CKPT, map_location=DEVICE)
    state = ck["model_state"]

    dec = DecoderOnly(LATENT_DIM).to(DEVICE)
    dec_sd = dec.state_dict()

    new_sd = {}
    for k in dec_sd.keys():
        if k in state:
            new_sd[k] = state[k]
        else:
            # try suffix match (defensive)
            found = None
            for kk in state.keys():
                if kk.endswith(k):
                    found = kk
                    break
            if found is None:
                raise KeyError(f"Could not find decoder key '{k}' in checkpoint.")
            new_sd[k] = state[found]

    dec.load_state_dict(new_sd, strict=True)
    dec.eval()
    return dec

@torch.no_grad()
def decode_grid(dec, z_list, nrow):
    z = torch.tensor(np.stack(z_list, axis=0), dtype=torch.float32, device=DEVICE)
    x = dec.decode(z).detach().cpu()
    grid = make_grid(x, nrow=nrow, padding=2)
    return grid

def main():
    assert os.path.exists(REPORT), "Run phase13_direction_transport_v2.py first."

    with open(REPORT, "r") as f:
        rep = json.load(f)

    vec_npz_path = rep["vectors_npz"]
    V = np.load(vec_npz_path)

    Z = np.load(os.path.join(PHASE7_DIR, "encoded_latents_seed123_N6000.npy")).astype(np.float64)

    dec = load_decoder_from_ckpt()

    anchors = rep["anchors"]
    anchor_names = list(anchors.keys())

    # walk settings
    alphas = np.linspace(-4.0, 4.0, 17)
    step = 1.0  # keep 1.0; your vectors are unit-ish

    directions = ["v_between_clean", "v_overlap_clean", "v_lr", "v_tproj"]

    for aname in anchor_names:
        idx = int(anchors[aname]["seed"])
        z0 = Z[idx]

        for dname in directions:
            key = f"{aname}__{dname}"
            if key not in V.files:
                continue
            v = V[key].astype(np.float64)

            z_list = [z0 + (a*step)*v for a in alphas]
            grid = decode_grid(dec, z_list, nrow=len(alphas))
            out = os.path.join(OUTDIR, f"phase13_{aname}_walk_{dname}.png")
            save_image(grid, out)
            print("[phase13B-v2] saved ->", out)

    print("[phase13B-v2] done ->", OUTDIR)

if __name__ == "__main__":
    main()
