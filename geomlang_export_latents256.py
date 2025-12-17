#!/usr/bin/env python3
# geomlang_export_latents256.py
#
# Encode the full GeomEdges64Dataset into latent z-space and save to disk
# so evaluation scripts (relation_eval) can load them quickly.

import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from geomlang_edges_relscale_train64_latent256 import GeomEdges64Dataset, SceneModelEdges64_256

OUT_DIR = "outputs_edges_relscale256"
CKPT_PATH = os.path.join(OUT_DIR, "scene_model_edges_relscale256.pt")
os.makedirs(OUT_DIR, exist_ok=True)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TAG = "[export256]"

N_SAMPLES = 6000
SEED = 123  # deterministic export

LATENT_PATH = os.path.join(OUT_DIR, f"encoded_dataset_latents_seed{SEED}_N{N_SAMPLES}.npy")
LABEL_PATH  = os.path.join(OUT_DIR, f"encoded_dataset_labels_seed{SEED}_N{N_SAMPLES}.npy")

def main():
    print(f"{TAG} Using device = {DEVICE}")

    # ---- Load trained scene model ----
    print(f"{TAG} Loading scene model…")
    model = SceneModelEdges64_256().to(DEVICE)

    ckpt_path = os.path.join(OUT_DIR, "scene_model_edges_relscale256.pt")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    if "model_state_dict" not in ckpt:
        raise RuntimeError(f"{TAG} Checkpoint missing 'model_state_dict': {ckpt_path}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"{TAG} Loaded weights from {ckpt_path}")

    # ---- Load deterministic dataset ----
    print(f"{TAG} Loading dataset (N={N_SAMPLES}, seed={SEED})…")
    ds = GeomEdges64Dataset(N_SAMPLES, seed=SEED)
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

    all_z = []
    all_y = []

    # ---- Encode to latent space ----
    with torch.no_grad():
        for imgs, rel, scale, s_r, s_b in dl:
            imgs = imgs.to(DEVICE)
            z = model.encode(imgs).cpu().numpy()
            all_z.append(z)
            all_y.append(rel.numpy())

    Z = np.concatenate(all_z, axis=0)
    y = np.concatenate(all_y, axis=0)

    print(f"{TAG} Latent array shape = {Z.shape}")
    print(f"{TAG} Labels shape = {y.shape}")

    # ---- Save ----
    print(f"{TAG} Saving → {LATENT_PATH}")
    np.save(LATENT_PATH, Z)

    print(f"{TAG} Saving → {LABEL_PATH}")
    np.save(LABEL_PATH, y)

    print(f"{TAG} Done.")
    print(f"{TAG} Label counts:", np.bincount(y, minlength=5))


if __name__ == "__main__":
    main()
