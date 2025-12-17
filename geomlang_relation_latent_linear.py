#!/usr/bin/env python3
# geomlang_relation_latent_linear.py
#
# Linear relation classifier directly on the 256-D latent z.
# Uses the same SceneModelEdges256 + GeomEdges64Dataset as the other scripts.
#
# For several train fractions (0.10, 0.25, 0.50, 1.00) it:
#   - encodes N_SAMPLES synthetic scenes to z
#   - splits into train / test
#   - standardizes features
#   - trains a multinomial logistic regression on z
#   - reports train / test accuracy and confusion matrix

import os
import math
import numpy as np

import torch
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from geomlang_global_coords_latent256 import (
    GeomEdges64Dataset,
    SceneModelEdges256,
    load_scene_model,
    REL_LEFT, REL_RIGHT, REL_ABOVE, REL_BELOW, REL_OVERLAP, REL_NAMES,
)

TAG        = "[clsZ]"
N_SAMPLES  = 6000
BATCH_SIZE = 128
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR    = "outputs_edges_relscale256"
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_FRACTIONS = [0.10, 0.25, 0.50, 1.00]


# ---------------------------------------------------------------------
# Encode dataset into latent space
# ---------------------------------------------------------------------
def encode_dataset(model, n_samples=N_SAMPLES):
    print(f"{TAG} Encoding dataset: N={n_samples}")
    ds = GeomEdges64Dataset(n_samples)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    zs   = []
    rels = []

    with torch.no_grad():
        for imgs, rel, scale, s_r, s_b in dl:
            imgs = imgs.to(DEVICE)
            z    = model.encode(imgs)    # (B, 256)
            zs.append(z.cpu())
            rels.append(rel.cpu())

    z_all   = torch.cat(zs,   dim=0)     # (N, 256)
    rel_all = torch.cat(rels, dim=0)     # (N,)

    print(f"{TAG} Latent shape: {tuple(z_all.shape)}")

    # some quick relation stats for sanity
    rel_np = rel_all.numpy()
    for rid in range(5):
        cnt = int((rel_np == rid).sum())
        mu_r = z_all[rel_np == rid].mean(dim=0)
        print(f"{TAG} relation {rid}: {REL_NAMES[rid]:>7}, "
              f"count={cnt}, ||mu_r||={mu_r.norm().item():.3f}")

    return z_all.numpy(), rel_np


# ---------------------------------------------------------------------
# Train / eval logistic regression on z for different label fractions
# ---------------------------------------------------------------------
def run_latent_linear():
    print(f"{TAG} Using device: {DEVICE}")
    model = load_scene_model()
    model.to(DEVICE)
    model.eval()

    Z, y = encode_dataset(model, N_SAMPLES)      # Z: (N,256), y: (N,)

    # simple random split (80% train, 20% test)
    rng = np.random.RandomState(0)
    N = Z.shape[0]
    idx = np.arange(N)
    rng.shuffle(idx)

    split = int(0.8 * N)
    train_idx = idx[:split]
    test_idx  = idx[split:]

    X_train_full = Z[train_idx]
    y_train_full = y[train_idx]
    X_test       = Z[test_idx]
    y_test       = y[test_idx]

    print(f"{TAG} Split: train={len(y_train_full)}, test={len(y_test)}")

    summary_rows = []

    for frac in TRAIN_FRACTIONS:
        train_size = max(1, int(len(y_train_full) * frac))
        print(f"\n{TAG} === Training latent logistic with "
              f"train_fraction={frac:.2f}, train_size={train_size} ===")

        # pick a prefix subset for this fraction
        X_tr = X_train_full[:train_size]
        y_tr = y_train_full[:train_size]

        # standardize features
        scaler = StandardScaler()
        X_tr_std = scaler.fit_transform(X_tr)
        X_te_std = scaler.transform(X_test)

        clf = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=2000,
            n_jobs=-1,
        )
        clf.fit(X_tr_std, y_tr)

        y_tr_pred = clf.predict(X_tr_std)
        y_te_pred = clf.predict(X_te_std)

        acc_tr = accuracy_score(y_tr, y_tr_pred) * 100.0
        acc_te = accuracy_score(y_test, y_te_pred) * 100.0

        print(f"{TAG}   train acc = {acc_tr:.2f}%")
        print(f"{TAG}   test  acc = {acc_te:.2f}%")

        summary_rows.append((frac, train_size, acc_tr, acc_te))

        # for the full-data run, also print confusion matrix
        if abs(frac - 1.0) < 1e-6:
            cm = confusion_matrix(y_test, y_te_pred, labels=list(range(5)))
            print("\n======================================================================")
            print(f"{TAG} Confusion matrix for frac=1.00 (test_acc={acc_te:.2f}%)")
            print("rows=true, cols=pred:")
            header = "         " + " ".join(f"{name:>8}" for name in REL_NAMES)
            print(header)
            for i, row in enumerate(cm):
                row_str = " ".join(f"{v:8d}" for v in row)
                print(f"{REL_NAMES[i]:>8} {row_str}")
            print("======================================================================")

    # print summary table
    print("\n======================================================================")
    print(f"{TAG} Summary (latent logistic, new fit per fraction)")
    print("======================================================================")
    print(" frac  | train_size | train_acc(%) | test_acc(%)")
    print("-------+------------+-------------+------------")
    for frac, sz, acc_tr, acc_te in summary_rows:
        print(f" {frac:0.2f} | {sz:10d} | {acc_tr:11.2f} | {acc_te:10.2f}")
    print("======================================================================")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    run_latent_linear()
