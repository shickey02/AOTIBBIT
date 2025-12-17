#!/usr/bin/env python3
# geomlang_export_latents_ternary_phase1_256.py

import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from geomlang_edges_relternary_train64_latent256_phase1 import (
    GeomEdgesTernary64Phase1,
    SceneModelTernaryPhase1_256
)

OUT_DIR = "outputs_edges_relternary256_phase1"
CKPT_PATH = os.path.join(OUT_DIR, "scene_model_edges_relternary256_phase1.pt")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TAG = "[export-ternary-phase1-256]"

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

    model = SceneModelTernaryPhase1_256().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"{TAG} dataset N={N_SAMPLES} seed={SEED}")
    ds = GeomEdgesTernary64Phase1(N_SAMPLES, seed=SEED)
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

    Zs, Ys = [], []
    with torch.no_grad():
        for imgs, rel, sA, sB, sC in dl:
            imgs = imgs.to(DEVICE)
            z = model.encode(imgs).cpu().numpy()
            Zs.append(z)
            Ys.append(rel.numpy())

    Z = np.concatenate(Zs, axis=0)
    y = np.concatenate(Ys, axis=0)

    print(f"{TAG} Z shape = {Z.shape}, y shape = {y.shape}")
    print(f"{TAG} label counts = {np.bincount(y, minlength=int(y.max())+1)}")

    np.save(LATENT_PATH, Z)
    np.save(LABEL_PATH, y)
    print(f"{TAG} saved: {LATENT_PATH}")
    print(f"{TAG} saved: {LABEL_PATH}")

if __name__ == "__main__":
    main()
