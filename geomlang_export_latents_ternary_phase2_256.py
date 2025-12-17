#!/usr/bin/env python3
# geomlang_export_latents_ternary_phase2_256.py
#
# Deterministically encode Phase 2 ternary dataset into latent Z and labels y.
#
# Saves:
#   outputs_edges_relternary256_phase2/encoded_latents_seed123_N6000.npy
#   outputs_edges_relternary256_phase2/encoded_labels_seed123_N6000.npy

import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from geomlang_edges_relternary_train64_latent256_phase2 import (
    GeomEdgesTernary64Dataset,
    SceneModelTernaryEdges64_256,
    OUT_DIR,
    CKPT_PATH,
    DEVICE,
    REL_NAMES,
)

TAG = "[export-ternary-phase2-256]"
N_SAMPLES = 6000
SEED = 123

LATENT_PATH = os.path.join(OUT_DIR, f"encoded_latents_seed{SEED}_N{N_SAMPLES}.npy")
LABEL_PATH  = os.path.join(OUT_DIR, f"encoded_labels_seed{SEED}_N{N_SAMPLES}.npy")

def main():
    print(f"{TAG} device = {DEVICE}")
    print(f"{TAG} loading ckpt: {CKPT_PATH}")

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    if "model_state_dict" not in ckpt:
        raise RuntimeError(f"{TAG} checkpoint missing model_state_dict")

    model = SceneModelTernaryEdges64_256().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"{TAG} dataset N={N_SAMPLES} seed={SEED}")
    ds = GeomEdgesTernary64Dataset(N_SAMPLES, seed=SEED)
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

    all_z = []
    all_y = []

    with torch.no_grad():
        for imgs, rel, sA, sB, sC in dl:
            imgs = imgs.to(DEVICE)
            z = model.encode(imgs).detach().cpu().numpy()
            all_z.append(z)
            all_y.append(rel.numpy())

    Z = np.concatenate(all_z, axis=0)
    y = np.concatenate(all_y, axis=0)

    print(f"{TAG} Z shape = {Z.shape}, y shape = {y.shape}")
    counts = np.bincount(y, minlength=len(REL_NAMES))
    print(f"{TAG} label counts = {counts}")

    np.save(LATENT_PATH, Z)
    np.save(LABEL_PATH, y)
    print(f"{TAG} saved: {LATENT_PATH}")
    print(f"{TAG} saved: {LABEL_PATH}")

if __name__ == "__main__":
    main()
