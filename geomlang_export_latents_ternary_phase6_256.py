#!/usr/bin/env python3
# geomlang_export_latents_ternary_phase6_256.py
#
# Exports:
#   - encoded_latents_seed123_N6000.npy
#   - encoded_targets_seed123_N6000.npz
#   - encoded_preds_seed123_N6000.npz

import os, random, math
import numpy as np
import torch

from geomlang_edges_relternary_train64_latent256_phase6 import (
    ConvAEHeads, TernaryGeomDataset, LATENT_DIM, OUTDIR
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = os.path.join(OUTDIR, "scene_model_edges_relternary256_phase6.pt")

N = 6000
SEED = 123

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"[export-ternary-phase6-256] device = {DEVICE}")
    print(f"[export-ternary-phase6-256] loading ckpt: {CKPT}")

    ck = torch.load(CKPT, map_location=DEVICE)
    model = ConvAEHeads(LATENT_DIM).to(DEVICE)
    model.load_state_dict(ck["model_state"])
    model.eval()

    ds = TernaryGeomDataset(N, seed=SEED)

    Z = np.zeros((N, LATENT_DIM), dtype=np.float32)
    targ = {k: np.zeros((N,), dtype=np.float32) for k in [
        "between_score","t_on_BC","closer_sign","closer_mag","overlap_B","overlap_C"
    ]}
    pred = {k: np.zeros((N,), dtype=np.float32) for k in [
        "between_score","t_on_BC","closer_sign","closer_mag","overlap_B","overlap_C"
    ]}

    with torch.no_grad():
        for i in range(N):
            x, y = ds[i]
            x = x.unsqueeze(0).to(DEVICE)
            out = model(x)
            z = out["z"].squeeze(0).detach().cpu().numpy().astype(np.float32)
            Z[i] = z

            # targets
            for k in targ.keys():
                targ[k][i] = float(y[k].item())

            # preds
            pred["between_score"][i] = float(torch.sigmoid(out["between"]).item())
            pred["t_on_BC"][i]       = float(torch.sigmoid(out["tproj"]).item())
            pred["closer_sign"][i]   = float(torch.sigmoid(out["csign"]).item())
            pred["closer_mag"][i]    = float(torch.sigmoid(out["cmag"]).item())
            pred["overlap_B"][i]     = float(torch.sigmoid(out["oB"]).item())
            pred["overlap_C"][i]     = float(torch.sigmoid(out["oC"]).item())

    print(f"[export-ternary-phase6-256] Z shape = {Z.shape}")
    np.save(os.path.join(OUTDIR, f"encoded_latents_seed{SEED}_N{N}.npy"), Z)
    np.savez(os.path.join(OUTDIR, f"encoded_targets_seed{SEED}_N{N}.npz"), **targ)
    np.savez(os.path.join(OUTDIR, f"encoded_preds_seed{SEED}_N{N}.npz"), **pred)

    print(f"[export-ternary-phase6-256] saved: {OUTDIR}\\encoded_latents_seed{SEED}_N{N}.npy")
    print(f"[export-ternary-phase6-256] saved: {OUTDIR}\\encoded_targets_seed{SEED}_N{N}.npz")
    print(f"[export-ternary-phase6-256] saved: {OUTDIR}\\encoded_preds_seed{SEED}_N{N}.npz")

if __name__ == "__main__":
    main()
