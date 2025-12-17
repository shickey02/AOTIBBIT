#!/usr/bin/env python3
# geomlang_relation_classifier_tx_ty.py
#
# Quick baseline:
#   - encode a synthetic dataset
#   - compute global generator coords (t_x, t_y) for each sample
#   - train a simple linear classifier on (t_x, t_y) -> relation
#   - report train/test accuracy and confusion matrix.

import os
import math
import numpy as np

import torch
from torch.utils.data import DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from geomlang_global_coords_latent256 import (
    GeomEdges64Dataset,
    SceneModelEdges256,
    load_scene_model,
    REL_LEFT, REL_RIGHT, REL_ABOVE, REL_BELOW, REL_OVERLAP, REL_NAMES,
)

TAG        = "[clsTxTy]"
LATENT_DIM = 256
N_SAMPLES  = 6000
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR    = "outputs_edges_relscale256"
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE = 128


def encode_dataset(model, n_samples=N_SAMPLES):
    print(f"{TAG} Encoding dataset: N={n_samples}")
    ds = GeomEdges64Dataset(n_samples)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_rel = []
    all_z   = []

    model.eval()
    with torch.no_grad():
        for imgs, rel, scale, s_r, s_b in dl:
            imgs = imgs.to(DEVICE)
            z    = model.encode(imgs)
            all_rel.append(rel.cpu())
            all_z.append(z.cpu())

    rel = torch.cat(all_rel, dim=0)  # (N,)
    z   = torch.cat(all_z,   dim=0)  # (N,D)
    print(f"{TAG} Latent shape: {tuple(z.shape)}")
    return rel, z


def compute_generators(z, rel):
    mus = []
    rel_np = rel.numpy()
    for rid in range(5):
        mask = (rel_np == rid)
        z_r = z[mask]
        mu_r = z_r.mean(dim=0)
        print(f"{TAG} relation {rid}: {REL_NAMES[rid]:>7}, "
              f"count={mask.sum()}, ||mu_r||={mu_r.norm():.3f}")
        mus.append(mu_r)

    mu_left, mu_right, mu_above, mu_below, mu_overlap = mus

    v_LR = mu_right - mu_left
    v_AB = mu_below - mu_above

    g_x = v_LR / v_LR.norm()
    g_y = v_AB / v_AB.norm()

    cos_xy = torch.dot(g_x, g_y).item()
    angle = math.degrees(math.acos(max(min(cos_xy, 1.0), -1.0)))
    print(f"{TAG} ||mu_right - mu_left|| = {v_LR.norm():.4f}")
    print(f"{TAG} ||mu_below - mu_above|| = {v_AB.norm():.4f}")
    print(f"{TAG} angle(g_x, g_y) = {angle:.2f}°")

    return g_x, g_y, mu_overlap


def project_to_tx_ty(z, g_x, g_y, mu_overlap):
    """
    Least-squares coords t_x, t_y s.t.
        z - mu0 ≈ t_x g_x + t_y g_y
    using the same Gram inverse trick as global_coords script.
    """
    z   = z.to(DEVICE)
    mu0 = mu_overlap.to(DEVICE)
    gx  = g_x.to(DEVICE)
    gy  = g_y.to(DEVICE)

    z_c = z - mu0.unsqueeze(0)    # (N,D)

    a = torch.matmul(z_c, gx)     # <z_c, g_x>
    b = torch.matmul(z_c, gy)     # <z_c, g_y>

    s = torch.dot(gx, gy).item()
    denom = 1.0 - s * s
    if denom <= 0:
        raise RuntimeError(f"Gram matrix singular, denom={denom}")

    t_x = (a - s * b) / denom
    t_y = (b - s * a) / denom

    print(f"{TAG} t_x stats: mean={t_x.mean().item():.4f}, std={t_x.std().item():.4f}")
    print(f"{TAG} t_y stats: mean={t_y.mean().item():.4f}, std={t_y.std().item():.4f}")

    return t_x.cpu().numpy(), t_y.cpu().numpy()


def main():
    print(f"{TAG} Using device: {DEVICE}")
    model = load_scene_model()
    model.to(DEVICE)

    rel, z = encode_dataset(model, N_SAMPLES)
    g_x, g_y, mu_overlap = compute_generators(z, rel)
    t_x, t_y = project_to_tx_ty(z, g_x, g_y, mu_overlap)

    X = np.stack([t_x, t_y], axis=1)   # (N,2)
    y = rel.numpy()

    # simple train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Logistic regression classifier on (t_x, t_y)
    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=2000,
    )
    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    y_test_pred  = clf.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc  = accuracy_score(y_test,  y_test_pred)

    print(f"{TAG} Train accuracy (t_x,t_y only) = {train_acc*100:.2f}%")
    print(f"{TAG} Test  accuracy (t_x,t_y only) = {test_acc*100:.2f}%")

    cm = confusion_matrix(y_test, y_test_pred, labels=[0,1,2,3,4])
    print(f"{TAG} Confusion matrix (rows=true, cols=pred):")
    print("       ", " ".join(f"{REL_NAMES[i]:>8}" for i in range(5)))
    for i in range(5):
        row = " ".join(f"{cm[i,j]:8d}" for j in range(5))
        print(f"{REL_NAMES[i]:>8} {row}")

if __name__ == "__main__":
    main()
