#!/usr/bin/env python3
# geomlang_relation_tx_ty_curv_linear.py
#
# Train a logistic regression relation classifier on a 6D feature:
#   [t_x, t_y, c1, c2, c3, c4]
#
# where:
#   - (t_x, t_y) are global generator coordinates in span{g_x, g_y}
#   - c1..c4 are coordinates in a curvature basis (selected residual PCs).
#
# This reuses the generator + curvature machinery to probe how much
# relational information is captured by the "group plane + curvature"
# subspace, versus the full 256-D latent.

import os
import math
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from geomlang_global_coords_latent256 import (
    GeomEdges64Dataset,
    SceneModelEdges256,
    load_scene_model,
    REL_LEFT, REL_RIGHT, REL_ABOVE, REL_BELOW, REL_OVERLAP, REL_NAMES,
)

TAG        = "[clsTxTyCurv]"
IMG_SIZE   = 64
LATENT_DIM = 256
N_SAMPLES  = 6000
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR    = "outputs_edges_relscale256"
os.makedirs(OUT_DIR, exist_ok=True)

# residual PCA config
RESID_DIM  = 32
CURV_IDXS  = [0, 1, 2, 5]   # pick same PCs as curvature basis (dim=4)

# training fractions for data-efficiency curve
TRAIN_FRACTIONS = [0.10, 0.25, 0.50, 1.00]


# ----------------- helpers -----------------

def encode_dataset(model, n_samples=N_SAMPLES):
    print(f"{TAG} Encoding dataset: N={n_samples}")
    ds = GeomEdges64Dataset(n_samples)
    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)

    all_z   = []
    all_rel = []

    model.eval()
    with torch.no_grad():
        for imgs, rel, scale, s_r, s_b in dl:
            imgs = imgs.to(DEVICE)
            z    = model.encode(imgs)
            all_z.append(z.cpu())
            all_rel.append(rel.cpu())

    z   = torch.cat(all_z,   dim=0)  # (N, D)
    rel = torch.cat(all_rel, dim=0)  # (N,)

    print(f"{TAG} Latent shape: {tuple(z.shape)}")
    return z, rel


def compute_generators(z, rel):
    """Return g_x, g_y, and relation means mu_r (all CPU tensors)."""
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
    angle  = math.degrees(math.acos(max(min(cos_xy, 1.0), -1.0)))
    print(f"{TAG} ||mu_right - mu_left|| = {v_LR.norm():.4f}")
    print(f"{TAG} ||mu_below - mu_above|| = {v_AB.norm():.4f}")
    print(f"{TAG} angle(g_x, g_y) = {angle:.2f}°")

    return g_x, g_y, mus


def compute_tx_ty_and_residuals(z, mu_overlap, g_x, g_y):
    """
    For each latent z_i, compute:
      - t_x, t_y from least-squares projection onto span{g_x,g_y}
      - residual r_perp in full latent (z_c - projection).

    Returns:
      t_x, t_y : (N,) numpy
      r_perp   : (N, D) numpy
    """
    z_dev = z.to(DEVICE)
    mu0   = mu_overlap.to(DEVICE)
    gx    = g_x.to(DEVICE)
    gy    = g_y.to(DEVICE)

    z_c = z_dev - mu0.unsqueeze(0)    # (N, D)

    # Gram matrix inverse for possibly non-orthogonal g_x, g_y
    s = torch.dot(gx, gy).item()
    denom = 1.0 - s * s
    if denom <= 0:
        raise RuntimeError("Gram matrix singular in compute_tx_ty_and_residuals")

    a = torch.matmul(z_c, gx)   # (N,)
    b = torch.matmul(z_c, gy)   # (N,)

    t_x = (a - s * b) / denom
    t_y = (b - s * a) / denom

    proj_plane = (
        t_x.unsqueeze(1) * gx.unsqueeze(0)
        + t_y.unsqueeze(1) * gy.unsqueeze(0)
    )  # (N, D)

    r_perp = (z_c - proj_plane).cpu().numpy()  # (N, D)
    t_x_np = t_x.cpu().numpy()
    t_y_np = t_y.cpu().numpy()

    print(f"{TAG} t_x stats: mean={t_x_np.mean():.4f}, std={t_x_np.std():.4f}")
    print(f"{TAG} t_y stats: mean={t_y_np.mean():.4f}, std={t_y_np.std():.4f}")

    rn = np.linalg.norm(r_perp, axis=1)
    print(f"{TAG} Residual ||r_perp|| stats: mean={rn.mean():.4f}, std={rn.std():.4f}, "
          f"min={rn.min():.4f}, max={rn.max():.4f}")

    return t_x_np, t_y_np, r_perp


def build_curvature_basis(r_perp):
    """
    Run PCA on residuals and return curvature basis in full latent:
      H_full: (D, k_basis) as a torch tensor, columns are curvature directions.
    """
    N, D = r_perp.shape
    print(f"{TAG} Running residual PCA on shape {r_perp.shape} with RESID_DIM={RESID_DIM}")

    pca = PCA(n_components=RESID_DIM)
    r_low = pca.fit_transform(r_perp)        # (N, RESID_DIM)
    comps = pca.components_                  # (RESID_DIM, D)

    print(f"{TAG} Residual PCA variance (first 10 PCs):")
    for i, var in enumerate(pca.explained_variance_ratio_[:10]):
        print(f"   PC{i+1:02d}: {var*100:5.2f}%")

    H_full_np = comps[CURV_IDXS, :].T        # (D, k_basis)
    H_full    = torch.from_numpy(H_full_np).float()
    print(f"{TAG} Using residual PCs {CURV_IDXS} as curvature basis (dim={H_full.shape[1]})")

    return H_full


def compute_curvature_coords(r_perp, H_full):
    """
    For each residual r_i (in R^D), compute curvature coordinates c_i:
      c_i = H_full^T r_i, where H_full is (D, k_basis).
    Returns:
      C : (N, k_basis) numpy
    """
    H = H_full.to(DEVICE)          # (D, k)
    r = torch.from_numpy(r_perp).float().to(DEVICE)  # (N, D)

    with torch.no_grad():
        C = torch.matmul(r, H)     # (N, k)

    C_np = C.cpu().numpy()
    print(f"{TAG} Curvature coords stats per dim:")
    for j in range(C_np.shape[1]):
        cj = C_np[:, j]
        print(f"   c{j+1}: mean={cj.mean():.4f}, std={cj.std():.4f}, "
              f"min={cj.min():.4f}, max={cj.max():.4f}")
    return C_np

def train_logistic_features(X, y, frac):
    N = X.shape[0]

    # --- SPECIAL CASE: frac == 1.0 → use a proper held-out test set ---
    if abs(frac - 1.0) < 1e-8:
        # 80/20 split
        test_size  = int(0.20 * N)
        train_size = N - test_size

        idx = np.random.permutation(N)
        train_idx = idx[:train_size]
        test_idx  = idx[train_size:]

    else:
        # --- Normal case: small-fraction splits ---
        train_size = int(frac * N)
        idx = np.random.permutation(N)
        train_idx = idx[:train_size]
        test_idx  = idx[train_size:]
        if len(test_idx) == 0:
            raise RuntimeError("Empty test set — try increasing frac or patching the code.")

    # Extract datasets
    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    # Train classifier
    clf = LogisticRegression(max_iter=5000, multi_class="auto")
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train) * 100.0
    test_acc  = clf.score(X_test,  y_test)  * 100.0

    return clf, train_acc, test_acc, (X_test, y_test)


def print_confusion(y_true, y_pred, title_prefix=""):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4])
    print()
    print("="*70)
    print(f"{TAG} {title_prefix}Confusion matrix (rows=true, cols=pred):")
    header = "          " + "  ".join([f"{name:>7}" for name in REL_NAMES])
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{REL_NAMES[i]:>7} "
        for val in row:
            row_str += f"{val:9d}"
        print(row_str)
    print("="*70)
    print()


# ----------------- main -----------------

def main():
    print(f"{TAG} Using device: {DEVICE}")
    model = load_scene_model()
    model.to(DEVICE)

    # 1) encode full dataset to z + relation labels
    z, rel = encode_dataset(model, N_SAMPLES)  # z: (N,D), rel: (N,)

    # 2) global generators + means
    g_x, g_y, mus = compute_generators(z, rel)
    mu_overlap = mus[REL_OVERLAP]

    # 3) compute t_x, t_y, and residuals r_perp in full latent
    t_x, t_y, r_perp = compute_tx_ty_and_residuals(z, mu_overlap, g_x, g_y)

    # 4) build curvature basis from residual PCA
    H_full = build_curvature_basis(r_perp)   # (D, k_basis)

    # 5) curvature coordinates for each sample
    C = compute_curvature_coords(r_perp, H_full)  # (N, k_basis) -> here k_basis=4

    # 6) build final feature matrix [t_x, t_y, c1..ck]
    X = np.concatenate(
        [t_x.reshape(-1, 1),
         t_y.reshape(-1, 1),
         C],
        axis=1
    )
    y = rel.numpy()
    N, F = X.shape
    print(f"{TAG} Feature matrix shape: {X.shape} (N={N}, F={F})")

    # 7) train logistic at different fractions
    print()
    print("="*70)
    print(f"{TAG} Training logistic on (t_x, t_y, c1..c{C.shape[1]}) features")
    print("="*70)

    results = []
    best_model = None
    best_fraction = None
    best_test_acc = -1.0
    best_split = None

    for frac in TRAIN_FRACTIONS:
        clf, train_acc, test_acc, (X_test, y_test) = train_logistic_features(X, y, frac)
        results.append((frac, X.shape[0]*frac, train_acc, test_acc))
        print(f"{TAG} frac={frac:.2f} | train_acc={train_acc:.2f}% | test_acc={test_acc:.2f}%")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_fraction = frac
            best_model = clf
            best_split = (X_test, y_test)

    # 8) print summary table
    print()
    print("="*70)
    print(f"{TAG} Summary (logistic on 6D [t_x,t_y,c1..c4])")
    print("="*70)
    print(" frac  | train_size | train_acc(%) | test_acc(%)")
    print("-------+------------+-------------+-----------")
    for frac, train_size, train_acc, test_acc in results:
        print(f" {frac:0.2f} | {int(train_size):10d} | {train_acc:11.2f} | {test_acc:9.2f}")
    print("="*70)

    # 9) confusion matrix for best fraction
    if best_model is not None:
        X_test, y_test = best_split
        y_pred = best_model.predict(X_test)
        print_confusion(y_test, y_pred,
                        title_prefix=f"(t_x,t_y,curv) frac={best_fraction:.2f}, test_acc={best_test_acc:.2f}% ")

    print(f"{TAG} Done.")


if __name__ == "__main__":
    main()
