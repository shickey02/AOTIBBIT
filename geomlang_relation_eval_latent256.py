#!/usr/bin/env python3
# geomlang_relation_eval_latent256.py
#
# Compare different latent feature spaces:
#   - full 256-D z
#   - 2-D (t_x, t_y)
#   - 6-D (t_x, t_y, c1..c4)
# on several relation classification tasks, and visualize the geometry.
# Also probe an "overlap axis" orthogonal to the main relation plane.

import os
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# -----------------------
# Config
# -----------------------
IMG_SIZE    = 64
LATENT_DIM  = 256
RESID_DIM   = 32
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR     = "outputs_edges_relscale256"
SCENE_MODEL_PATH = os.path.join(OUT_DIR, "scene_model_edges_relscale256.pt")
DATA_LATENTS_PATH = os.path.join(OUT_DIR, "encoded_dataset_latents.npy")
DATA_LABELS_PATH  = os.path.join(OUT_DIR, "encoded_dataset_labels.npy")

REL_NAMES = ["left_of", "right_of", "above", "below", "overlap"]

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------
# Helpers
# -----------------------

def softmax_np(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()

def load_scene_model():
    checkpoint = torch.load(SCENE_MODEL_PATH, map_location=DEVICE)
    model = checkpoint["model"]
    model.to(DEVICE)
    model.eval()
    return model

def load_dataset_latents():
    # Assumes:
    #   np.save(DATA_LATENTS_PATH, Z)  where Z.shape = (N, 256)
    #   np.save(DATA_LABELS_PATH, rel_labels) with ints in [0..4]
    Z = np.load(DATA_LATENTS_PATH)
    y = np.load(DATA_LABELS_PATH)
    return Z, y

# -----------------------
# Geometric decomposition
# -----------------------

def compute_relation_axes_and_residuals(Z, y):
    """
    Given:
        Z: (N, D) latent vectors
        y: (N,) relation labels [0..4]
    Returns:
        g_x, g_y: (D,) unit vectors for left-right and above-below
        Z_resid:  (N, D) residuals after removing those 2 axes
    """
    Z_t = torch.from_numpy(Z).float().to(DEVICE)

    masks = {r: (y == r) for r in range(5)}
    mu = {}
    for r in range(5):
        mu[r] = Z_t[masks[r]].mean(dim=0)

    g_left  = mu[0]
    g_right = mu[1]
    g_above = mu[2]
    g_below = mu[3]

    g_x = (g_right - g_left)
    g_y = (g_below - g_above)

    g_x = g_x / g_x.norm()
    g_y = g_y / g_y.norm()

    # remove their components from Z to get residuals
    proj_x = (Z_t @ g_x)[:, None] * g_x[None, :]
    proj_y = (Z_t @ g_y)[:, None] * g_y[None, :]
    Z_resid = Z_t - proj_x - proj_y

    return g_x.cpu().numpy(), g_y.cpu().numpy(), Z_resid.cpu().numpy()

def compute_curvature_pcs(Z_resid, resid_dim=RESID_DIM):
    """
    PCA of residual subspace; returns components and projected coords.
    """
    pca = PCA(n_components=resid_dim, random_state=0)
    Z_resid_pca = pca.fit_transform(Z_resid)
    return pca, Z_resid_pca

def compute_overlap_axis(Z, y, g_x, g_y):
    """
    Compute an "overlap axis" g_overlap:
        g_overlap ∝ μ_overlap - μ_nonoverlap,
    then orthogonalize against g_x, g_y and normalize.

    Returns:
        g_overlap: (D,) numpy array (unit vector).
    """
    Z_t = torch.from_numpy(Z).float().to(DEVICE)
    g_x_t = torch.from_numpy(g_x).float().to(DEVICE)
    g_y_t = torch.from_numpy(g_y).float().to(DEVICE)

    mask_overlap = (y == 4)
    mask_non = ~mask_overlap

    mu_overlap    = Z_t[mask_overlap].mean(dim=0)
    mu_nonoverlap = Z_t[mask_non].mean(dim=0)

    g_o = mu_overlap - mu_nonoverlap

    # orthogonalize to g_x and g_y
    g_o = g_o - (g_o @ g_x_t) * g_x_t - (g_o @ g_y_t) * g_y_t
    g_o = g_o / g_o.norm()

    return g_o.cpu().numpy()

def build_features(Z, y):
    """
    Returns:
        X_2d  : (N, 2)       -> [t_x, t_y]
        X_6d  : (N, 6)       -> [t_x, t_y, c1, c2, c3, c4]
        X_z   : (N, 256)     -> raw latent
        g_x, g_y, Z_resid, Z_resid_pca, u (overlap coordinate)
    """

    # relation axes + residuals
    g_x, g_y, Z_resid = compute_relation_axes_and_residuals(Z, y)

    # projected coordinates on g_x, g_y
    t_x = Z @ g_x          # (N,)
    t_y = Z @ g_y          # (N,)

    # curvature PCs
    pca_resid, Z_resid_pca = compute_curvature_pcs(Z_resid, resid_dim=RESID_DIM)

    # choose a "curvature basis" – same indices as before
    # (0,1) carry most variance, plus 2 and 5 for extra structure
    c1 = Z_resid_pca[:, 0]
    c2 = Z_resid_pca[:, 1]
    c3 = Z_resid_pca[:, 2]
    c4 = Z_resid_pca[:, 5]

    X_2d = np.stack([t_x, t_y], axis=1)
    X_6d = np.stack([t_x, t_y, c1, c2, c3, c4], axis=1)
    X_z  = Z.astype(np.float32)

    # overlap axis (orthogonal to relation plane)
    g_overlap = compute_overlap_axis(Z, y, g_x, g_y)
    u = Z @ g_overlap  # (N,)

    # quick stats (just so we see what’s going on)
    def feat_stats(name, arr):
        print(f"[eval256] Feature set {name}: shape={arr.shape}")

    feat_stats("2D (t_x,t_y)", X_2d)
    feat_stats("6D (t_x,t_y,c1..c4)", X_6d)
    feat_stats("z (256D raw)", X_z)

    return X_2d, X_6d, X_z, g_x, g_y, Z_resid, Z_resid_pca, u

# -----------------------
# Visualization helpers
# -----------------------

def plot_2d_scatter(X_2d, y, title, out_path, rel_names=REL_NAMES):
    """
    X_2d: (N,2)
    y:    (N,)
    """
    plt.figure(figsize=(6, 6))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    for r in range(len(rel_names)):
        mask = (y == r)
        plt.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            s=8, alpha=0.5, label=rel_names[r], c=colors[r]
        )
    plt.xlabel("t_x (left/right axis)")
    plt.ylabel("t_y (above/below axis)")
    plt.title(title)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[eval256] Saved 2D scatter -> {out_path}")

def plot_3d_scatter(X_3d, y, title, out_path, rel_names=REL_NAMES):
    """
    X_3d: (N,3)
    y:    (N,)
    """
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

    for r in range(len(rel_names)):
        mask = (y == r)
        ax.scatter(
            X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2],
            s=8, alpha=0.5, c=colors[r], label=rel_names[r]
        )

    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.set_zlabel("dim 3")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[eval256] Saved 3D scatter -> {out_path}")

def plot_2d_binary(X_2d, y_bin, title, out_path,
                   label0="class0", label1="class1",
                   xlabel="dim1", ylabel="dim2"):
    """
    For binary labels y_bin ∈ {0,1}, scatter plot in 2D.
    """
    plt.figure(figsize=(6, 5))
    mask0 = (y_bin == 0)
    mask1 = (y_bin == 1)

    plt.scatter(X_2d[mask0, 0], X_2d[mask0, 1],
                s=8, alpha=0.5, label=label0, c='tab:blue')
    plt.scatter(X_2d[mask1, 0], X_2d[mask1, 1],
                s=8, alpha=0.5, label=label1, c='tab:orange')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[eval256] Saved binary 2D scatter -> {out_path}")

# -----------------------
# Tasks
# -----------------------

def make_binary_labels(y, mode):
    """
    mode = 'x_side' or 'y_side'
    x_side: 1 if left_of/right_of, 0 otherwise
    y_side: 1 if above/below,      0 otherwise
    """
    if mode == "x_side":
        return np.isin(y, [0, 1]).astype(np.int64)
    elif mode == "y_side":
        return np.isin(y, [2, 3]).astype(np.int64)
    else:
        raise ValueError(mode)

def eval_logistic(X, y, name, multi_class=True,
                  label_names=None, show_confusion=False):
    """
    Train/test split + logistic regression on feature matrix X for labels y.
    Prints confusion matrices for both multi-class and binary tasks if requested.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    if multi_class:
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=500,
                multi_class="multinomial",
                solver="lbfgs"
            ),
        )
    else:
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=500,
                solver="lbfgs"
            ),
        )

    clf.fit(X_train, y_train)
    acc_train = clf.score(X_train, y_train) * 100.0
    acc_test  = clf.score(X_test,  y_test) * 100.0

    task_type = "multi" if multi_class else "binary"
    print(f"[eval256] {name:20s} ({task_type}) "
          f"| train_acc={acc_train:5.2f}% | test_acc={acc_test:5.2f}%")

    if show_confusion:
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        # default labels if none provided
        if label_names is None:
            if multi_class:
                label_names = [str(i) for i in range(cm.shape[0])]
            else:
                label_names = ["class0", "class1"]

        print(f"\n[eval256] Confusion matrix for {name} (rows=true, cols=pred):")
        header = "           " + "  ".join(f"{ln:>12s}" for ln in label_names)
        print(header)
        for i, ln in enumerate(label_names):
            row = "  ".join(f"{cm[i, j]:12d}" for j in range(len(label_names)))
            print(f"{ln:>11s}  {row}")
        print("")

# -----------------------
# Main
# -----------------------

def main():
    print(f"[eval256] Using device: {DEVICE}")

    # load dataset latents and relation labels
    Z, y = load_dataset_latents()
    print(f"[eval256] Z shape: {Z.shape}, y shape: {y.shape}")

    X_2d, X_6d, X_z, g_x, g_y, Z_resid, Z_resid_pca, u = build_features(Z, y)

    # ----- Visualization of geometry -----
    # 1) Raw 2D relation plane
    plot_2d_scatter(
        X_2d, y,
        title="Relation plane: (t_x, t_y)",
        out_path=os.path.join(OUT_DIR, "latent256_2D_tx_ty.png"),
    )

    # 2) 3D slice of 6D: (t_x, t_y, c1)
    X_3d_tx_ty_c1 = X_6d[:, [0, 1, 2]]  # t_x, t_y, c1
    plot_3d_scatter(
        X_3d_tx_ty_c1, y,
        title="3D slice: (t_x, t_y, c1)",
        out_path=os.path.join(OUT_DIR, "latent256_3D_tx_ty_c1.png"),
    )

    # 3) PCA(3) of full 6D feature space
    pca3 = PCA(n_components=3, random_state=0)
    X_6d_pca3 = pca3.fit_transform(X_6d)
    plot_3d_scatter(
        X_6d_pca3, y,
        title="6D feature space PCA(3)",
        out_path=os.path.join(OUT_DIR, "latent256_3D_6D_PCA.png"),
    )

    # ----- Task 1: 5-way relation classification -----
    print("\n[eval256] === Task 1: 5-way relation classification ===")
    eval_logistic(
        X_2d, y,
        "2D    (t_x,t_y)",
        multi_class=True,
        label_names=REL_NAMES,
        show_confusion=True,
    )
    eval_logistic(
        X_6d, y,
        "6D    (t,c1..c4)",
        multi_class=True,
        label_names=REL_NAMES,
        show_confusion=True,
    )
    eval_logistic(
        X_z, y,
        "256D  (z raw)",
        multi_class=True,
        label_names=REL_NAMES,
        show_confusion=True,
    )

    # ----- Task 2: X-side (left/right vs others) -----
    print("\n[eval256] === Task 2: X-side (left/right vs others) ===")
    y_x = make_binary_labels(y, "x_side")
    eval_logistic(
        X_2d, y_x,
        "2D    (t_x,t_y)",
        multi_class=False,
        label_names=["other", "left_or_right"],
        show_confusion=True,
    )
    eval_logistic(
        X_6d, y_x,
        "6D    (t,c1..c4)",
        multi_class=False,
        label_names=["other", "left_or_right"],
        show_confusion=True,
    )
    eval_logistic(
        X_z,  y_x,
        "256D  (z raw)",
        multi_class=False,
        label_names=["other", "left_or_right"],
        show_confusion=True,
    )

    # ----- Task 3: Y-side (above/below vs others) -----
    print("\n[eval256] === Task 3: Y-side (above/below vs others) ===")
    y_y = make_binary_labels(y, "y_side")
    eval_logistic(
        X_2d, y_y,
        "2D    (t_x,t_y)",
        multi_class=False,
        label_names=["other", "above_or_below"],
        show_confusion=True,
    )
    eval_logistic(
        X_6d, y_y,
        "6D    (t,c1..c4)",
        multi_class=False,
        label_names=["other", "above_or_below"],
        show_confusion=True,
    )
    eval_logistic(
        X_z,  y_y,
        "256D  (z raw)",
        multi_class=False,
        label_names=["other", "above_or_below"],
        show_confusion=True,
    )

    # ----- Task 4: Left vs Right (restricted) -----
    print("\n[eval256] === Task 4: Left vs Right (restricted to left/right samples) ===")
    mask_lr = np.isin(y, [0, 1])
    X_2d_lr = X_2d[mask_lr]
    X_6d_lr = X_6d[mask_lr]
    X_z_lr  = X_z[mask_lr]
    y_lr    = y[mask_lr]  # 0 = left_of, 1 = right_of

    eval_logistic(
        X_2d_lr, y_lr,
        "2D    (t_x,t_y)",
        multi_class=False,
        label_names=["left_of", "right_of"],
        show_confusion=True,
    )
    eval_logistic(
        X_6d_lr, y_lr,
        "6D    (t,c1..c4)",
        multi_class=False,
        label_names=["left_of", "right_of"],
        show_confusion=True,
    )
    eval_logistic(
        X_z_lr,  y_lr,
        "256D  (z raw)",
        multi_class=False,
        label_names=["left_of", "right_of"],
        show_confusion=True,
    )

    # ----- Task 5: Above vs Below (restricted) -----
    print("\n[eval256] === Task 5: Above vs Below (restricted to above/below samples) ===")
    mask_ab = np.isin(y, [2, 3])
    X_2d_ab = X_2d[mask_ab]
    X_6d_ab = X_6d[mask_ab]
    X_z_ab  = X_z[mask_ab]
    y_ab    = (y[mask_ab] == 3).astype(np.int64)  # 0 = above, 1 = below

    eval_logistic(
        X_2d_ab, y_ab,
        "2D    (t_x,t_y)",
        multi_class=False,
        label_names=["above", "below"],
        show_confusion=True,
    )
    eval_logistic(
        X_6d_ab, y_ab,
        "6D    (t,c1..c4)",
        multi_class=False,
        label_names=["above", "below"],
        show_confusion=True,
    )
    eval_logistic(
        X_z_ab,  y_ab,
        "256D  (z raw)",
        multi_class=False,
        label_names=["above", "below"],
        show_confusion=True,
    )

    # ----- Task 6: Overlap vs Non-Overlap (baseline) -----
    print("\n[eval256] === Task 6: Overlap vs Non-Overlap ===")
    y_overlap = (y == 4).astype(np.int64)  # 1 = overlap, 0 = non-overlap

    eval_logistic(
        X_2d, y_overlap,
        "2D    (t_x,t_y)",
        multi_class=False,
        label_names=["non_overlap", "overlap"],
        show_confusion=True,
    )
    eval_logistic(
        X_6d, y_overlap,
        "6D    (t,c1..c4)",
        multi_class=False,
        label_names=["non_overlap", "overlap"],
        show_confusion=True,
    )
    eval_logistic(
        X_z,  y_overlap,
        "256D  (z raw)",
        multi_class=False,
        label_names=["non_overlap", "overlap"],
        show_confusion=True,
    )

    # ----- Task 7: Overlap-axis probe (u) -----
    print("\n[eval256] === Task 7: Overlap-axis probe (u) ===")

    # 1D: u only
    X_u_1d = u[:, None]

    eval_logistic(
        X_u_1d, y_overlap,
        "1D    (u only)",
        multi_class=False,
        label_names=["non_overlap", "overlap"],
        show_confusion=True,
    )

    # 3D: [u, t_x, t_y]
    X_u_tx_ty = np.concatenate([u[:, None], X_2d], axis=1)
    eval_logistic(
        X_u_tx_ty, y_overlap,
        "3D    (u,t_x,t_y)",
        multi_class=False,
        label_names=["non_overlap", "overlap"],
        show_confusion=True,
    )

    # Visualization: (t_x, u) colored by overlap vs non-overlap
    X_tx_u = np.stack([X_2d[:, 0], u], axis=1)
    plot_2d_binary(
        X_tx_u, y_overlap,
        title="Overlap axis vs left/right axis",
        out_path=os.path.join(OUT_DIR, "latent256_2D_tx_u_overlap.png"),
        label0="non_overlap",
        label1="overlap",
        xlabel="t_x (left/right)",
        ylabel="u (overlap axis)",
    )

if __name__ == "__main__":
    main()
