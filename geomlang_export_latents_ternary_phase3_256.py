#!/usr/bin/env python3
# geomlang_export_latents_ternary_phase3_256.py
#
# Exports latent vectors + labels for Phase 3.

import os
import numpy as np
import torch

from geomlang_edges_relternary_train64_latent256_phase3 import (
    DEVICE, OUT_DIR, CKPT_PATH, REL_NAMES,
    GeomEdgesTernary64Dataset,
    SceneModelTernaryEdges64_256,
)

TAG = "[export-ternary-phase3-256]"

EXPORT_N = 6000
EXPORT_SEED = 123
BATCH_SIZE = 256

def main():
    print(f"{TAG} device = {DEVICE}")
    print(f"{TAG} loading ckpt: {CKPT_PATH}")

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model = SceneModelTernaryEdges64_256().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = GeomEdgesTernary64Dataset(EXPORT_N, seed=EXPORT_SEED)
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    Zs = []
    Ys = []

    with torch.no_grad():
        for batch in dl:
            imgs, rel, sA, sB, sC, u_true, dperp_true = batch
            imgs = imgs.to(DEVICE)
            z = model.encode(imgs)
            Zs.append(z.detach().cpu().numpy())
            Ys.append(rel.numpy())

    Z = np.concatenate(Zs, axis=0)
    y = np.concatenate(Ys, axis=0)

    print(f"{TAG} Z shape = {Z.shape}, y shape = {y.shape}")
    counts = np.bincount(y, minlength=len(REL_NAMES))
    print(f"{TAG} label counts = {counts}")

    outZ = os.path.join(OUT_DIR, f"encoded_latents_seed{EXPORT_SEED}_N{EXPORT_N}.npy")
    outY = os.path.join(OUT_DIR, f"encoded_labels_seed{EXPORT_SEED}_N{EXPORT_N}.npy")

    np.save(outZ, Z)
    np.save(outY, y)

    print(f"{TAG} saved: {outZ}")
    print(f"{TAG} saved: {outY}")

if __name__ == "__main__":
    main()
