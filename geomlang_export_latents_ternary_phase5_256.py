#!/usr/bin/env python3
# geomlang_export_latents_ternary_phase5_256.py
#
# Exports:
#   - encoded_latents_seed123_N6000.npy
#   - encoded_targets_seed123_N6000.npz
#   - encoded_preds_seed123_N6000.npz

import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from geomlang_edges_relternary_train64_latent256_phase5 import (
    TAG, DEVICE,
    OUT_DIR, CKPT_PATH,
    IMG_SIZE, NUM_CH,
    GeomEdgesTernary64DatasetPhase5,
    collate_phase5,
    SceneModelTernaryEdges64_256_Phase5,
)

EVAL_N = 6000
EVAL_SEED = 123
BATCH_SIZE = 256

@torch.no_grad()
def main():
    print(f"[export-ternary-phase5-256] device = {DEVICE}")
    print(f"[export-ternary-phase5-256] loading ckpt: {CKPT_PATH}")

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model = SceneModelTernaryEdges64_256_Phase5().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = GeomEdgesTernary64DatasetPhase5(EVAL_N, seed=EVAL_SEED)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_phase5)

    Z = []
    tgt_between = []
    tgt_tproj = []
    tgt_csign = []
    tgt_cmag = []
    tgt_oB = []
    tgt_oC = []
    tgt_disp = []

    pr_between = []
    pr_tproj = []
    pr_csign = []
    pr_cmag = []
    pr_oB = []
    pr_oC = []

    for x, t in dl:
        x = x.to(DEVICE)
        for k in t: t[k] = t[k].to(DEVICE)

        out = model(x)
        z = out["z"].detach().cpu().numpy()
        Z.append(z)

        # targets
        tgt_between.append(t["between_score"].detach().cpu().numpy())
        tgt_tproj.append(t["t_on_BC"].detach().cpu().numpy())
        tgt_csign.append(t["closer_sign"].detach().cpu().numpy())
        tgt_cmag.append(t["closer_mag"].detach().cpu().numpy())
        tgt_oB.append(t["overlap_B"].detach().cpu().numpy())
        tgt_oC.append(t["overlap_C"].detach().cpu().numpy())
        tgt_disp.append(t["disp_label"].detach().cpu().numpy())

        # preds
        pr_between.append(out["between"].detach().cpu().numpy())
        pr_tproj.append(out["tproj"].detach().cpu().numpy())
        pr_cmag.append(out["cmag"].detach().cpu().numpy())
        pr_csign.append(torch.argmax(out["csign_logits"], dim=1).detach().cpu().numpy())
        pr_oB.append(torch.argmax(out["oB_logits"], dim=1).detach().cpu().numpy())
        pr_oC.append(torch.argmax(out["oC_logits"], dim=1).detach().cpu().numpy())

    Z = np.concatenate(Z, axis=0)
    print(f"[export-ternary-phase5-256] Z shape = {Z.shape}")

    out_lat = os.path.join(OUT_DIR, f"encoded_latents_seed{EVAL_SEED}_N{EVAL_N}.npy")
    np.save(out_lat, Z)
    print(f"[export-ternary-phase5-256] saved: {out_lat}")

    out_tgt = os.path.join(OUT_DIR, f"encoded_targets_seed{EVAL_SEED}_N{EVAL_N}.npz")
    np.savez(
        out_tgt,
        between_score=np.concatenate(tgt_between),
        t_on_BC=np.concatenate(tgt_tproj),
        closer_sign=np.concatenate(tgt_csign),
        closer_mag=np.concatenate(tgt_cmag),
        overlap_B=np.concatenate(tgt_oB),
        overlap_C=np.concatenate(tgt_oC),
        disp_label=np.concatenate(tgt_disp),
    )
    print(f"[export-ternary-phase5-256] saved: {out_tgt}")

    out_pred = os.path.join(OUT_DIR, f"encoded_preds_seed{EVAL_SEED}_N{EVAL_N}.npz")
    np.savez(
        out_pred,
        between_score=np.concatenate(pr_between),
        t_on_BC=np.concatenate(pr_tproj),
        closer_sign=np.concatenate(pr_csign),
        closer_mag=np.concatenate(pr_cmag),
        overlap_B=np.concatenate(pr_oB),
        overlap_C=np.concatenate(pr_oC),
    )
    print(f"[export-ternary-phase5-256] saved: {out_pred}")

if __name__ == "__main__":
    main()
