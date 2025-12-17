#!/usr/bin/env python3
# geomlang_export_latents_ternary_phase4_256.py
#
# Exports:
#   - Z latents (N,256)
#   - targets dict arrays
#   - preds dict arrays (head outputs)

import os
import numpy as np
import torch

from geomlang_edges_relternary_train64_latent256_phase4 import (
    DEVICE, OUT_DIR, CKPT_PATH,
    LATENT_DIM,
    GeomEdgesTernary64DatasetPhase4,
    SceneModelTernaryEdges64_256_Phase4,
)

TAG = "[export-ternary-phase4-256]"

N = 6000
SEED = 123
BATCH = 256

@torch.no_grad()
def main():
    print(f"{TAG} device = {DEVICE}")
    print(f"{TAG} loading ckpt: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    if "model_state_dict" not in ckpt:
        raise RuntimeError(f"{TAG} checkpoint missing model_state_dict")

    model = SceneModelTernaryEdges64_256_Phase4().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = GeomEdgesTernary64DatasetPhase4(N, seed=SEED)
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0)

    Zs = []
    # targets
    tb, tt, tcs, tcm, toB, toC = [], [], [], [], [], []
    # preds
    pb, pt, pcs, pcm, poB, poC = [], [], [], [], [], []

    for x, b, t, cs, cm, oB, oC in dl:
        x = x.to(DEVICE)
        z = model.encode(x)

        b_pred  = torch.sigmoid(model.between_head(z))
        t_pred  = torch.sigmoid(model.tproj_head(z))
        cs_pred = torch.sigmoid(model.cs_head(z))
        cm_pred = torch.tanh(model.cm_head(z))
        oB_pred = torch.sigmoid(model.oB_head(z))
        oC_pred = torch.sigmoid(model.oC_head(z))

        Zs.append(z.detach().cpu().numpy().astype(np.float32))

        tb.append(b.numpy());  tt.append(t.numpy());  tcs.append(cs.numpy()); tcm.append(cm.numpy()); toB.append(oB.numpy()); toC.append(oC.numpy())
        pb.append(b_pred.cpu().numpy()); pt.append(t_pred.cpu().numpy()); pcs.append(cs_pred.cpu().numpy()); pcm.append(cm_pred.cpu().numpy()); poB.append(oB_pred.cpu().numpy()); poC.append(oC_pred.cpu().numpy())

    Z = np.concatenate(Zs, axis=0)
    targets = {
        "between": np.concatenate(tb, axis=0).squeeze(1),
        "tproj":   np.concatenate(tt, axis=0).squeeze(1),
        "csign":   np.concatenate(tcs,axis=0).squeeze(1),
        "cmag":    np.concatenate(tcm,axis=0).squeeze(1),
        "oB":      np.concatenate(toB, axis=0).squeeze(1),
        "oC":      np.concatenate(toC, axis=0).squeeze(1),
    }
    preds = {
        "between": np.concatenate(pb, axis=0).squeeze(1),
        "tproj":   np.concatenate(pt, axis=0).squeeze(1),
        "csign":   np.concatenate(pcs,axis=0).squeeze(1),
        "cmag":    np.concatenate(pcm,axis=0).squeeze(1),
        "oB":      np.concatenate(poB, axis=0).squeeze(1),
        "oC":      np.concatenate(poC, axis=0).squeeze(1),
    }

    print(f"{TAG} Z shape = {Z.shape}")
    z_path = os.path.join(OUT_DIR, f"encoded_latents_seed{SEED}_N{N}.npy")
    t_path = os.path.join(OUT_DIR, f"encoded_targets_seed{SEED}_N{N}.npz")
    p_path = os.path.join(OUT_DIR, f"encoded_preds_seed{SEED}_N{N}.npz")

    np.save(z_path, Z)
    np.savez(t_path, **targets)
    np.savez(p_path, **preds)

    print(f"{TAG} saved: {z_path}")
    print(f"{TAG} saved: {t_path}")
    print(f"{TAG} saved: {p_path}")

if __name__ == "__main__":
    main()
