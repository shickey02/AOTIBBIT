#!/usr/bin/env python3
# geomlang_curvature_field_latent256.py
#
# Global LR/AB generator coordinates (t_x, t_y) + curvature fiber
# coordinates (u_1, u_2) for the LATENT_DIM=256 edges+relscale model.
#
# We:
#   1) Fit g_x, g_y from relation means (right-left, below-above).
#   2) Decompose z - mu0 = t_x g_x + t_y g_y + r_perp.
#   3) PCA the residuals r_perp and take the top 2 residual PCs as a
#      curvature basis h1, h2.
#   4) Project r_perp onto h1, h2 to get u_1, u_2.
#   5) Plot (t_x, t_y) colored by u_1, u_2, and curvature magnitude.

import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

# -----------------------
# Config
# -----------------------
IMG_SIZE   = 64
NUM_CH     = 3          # red, blue, edges
LATENT_DIM = 256

N_SAMPLES  = 6000
BATCH_SIZE = 128

OUT_DIR         = "outputs_edges_relscale256"
CKPT_SCENEMODEL = os.path.join(OUT_DIR, "scene_model_edges_relscale256.pt")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REL_LEFT, REL_RIGHT, REL_ABOVE, REL_BELOW, REL_OVERLAP = range(5)
REL_NAMES = ["left_of", "right_of", "above", "below", "overlap"]
REL_COLORS = {
    REL_LEFT:    "tab:blue",
    REL_RIGHT:   "tab:orange",
    REL_ABOVE:   "tab:green",
    REL_BELOW:   "tab:red",
    REL_OVERLAP: "tab:purple",
}

# -----------------------
# Scene generation (same as other 64x64 edges scripts)
# -----------------------

def draw_circle(mask, cx, cy, radius):
    H, W = mask.shape
    yy, xx = np.ogrid[:H, :W]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    mask[dist2 <= radius ** 2] = 1.0

def draw_square(mask, cx, cy, half_size):
    H, W = mask.shape
    x0 = max(0, cx - half_size)
    x1 = min(W, cx + half_size + 1)
    y0 = max(0, cy - half_size)
    y1 = min(H, cy + half_size + 1)
    mask[y0:y1, x0:x1] = 1.0

def make_edges(red_mask, blue_mask):
    union = (red_mask > 0.5) | (blue_mask > 0.5)
    union = union.astype(np.float32)

    H, W = union.shape
    interior = np.zeros_like(union)
    interior[1:-1, 1:-1] = (
        union[1:-1, 1:-1] *
        union[:-2, 1:-1] *
        union[2:, 1:-1] *
        union[1:-1, :-2] *
        union[1:-1, 2:]
    )
    edges = union - interior
    edges[edges < 0] = 0.0
    return edges

def relation_from_centers(cx_r, cy_r, cx_b, cy_b, tol=2.0):
    dx = cx_b - cx_r
    dy = cy_b - cy_r
    if abs(dx) > abs(dy) + tol:
        return REL_LEFT if dx > 0 else REL_RIGHT
    elif abs(dy) > abs(dx) + tol:
        return REL_ABOVE if dy > 0 else REL_BELOW
    else:
        return REL_OVERLAP

def scale_class_from_size(s):
    if s <= 7:
        return 0
    elif s <= 11:
        return 1
    else:
        return 2

class GeomEdges64Dataset(Dataset):
    def __init__(self, n_samples):
        super().__init__()
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        H = W = IMG_SIZE
        margin = 6

        shape_red  = np.random.randint(0, 2)   # 0 circle, 1 square
        shape_blue = np.random.randint(0, 2)

        base = np.random.randint(5, 13)
        size_red  = int(np.clip(base + np.random.randint(-2, 3), 4, 14))
        size_blue = int(np.clip(base + np.random.randint(-2, 3), 4, 14))

        def sample_center(s):
            cx = np.random.randint(margin + s, W - margin - s)
            cy = np.random.randint(margin + s, H - margin - s)
            return cx, cy

        cx_r, cy_r = sample_center(size_red)
        cx_b, cy_b = sample_center(size_blue)

        red  = np.zeros((H, W), dtype=np.float32)
        blue = np.zeros((H, W), dtype=np.float32)

        if shape_red == 0:
            draw_circle(red, cx_r, cy_r, size_red)
        else:
            draw_square(red, cx_r, cy_r, size_red)

        if shape_blue == 0:
            draw_circle(blue, cx_b, cy_b, size_blue)
        else:
            draw_square(blue, cx_b, cy_b, size_blue)

        edges = make_edges(red, blue)
        img   = np.stack([red, blue, edges], axis=0)

        rel   = relation_from_centers(cx_r, cy_r, cx_b, cy_b)
        scale = scale_class_from_size((size_red + size_blue) * 0.5)
        shape_r_lbl = shape_red
        shape_b_lbl = shape_blue

        return (
            torch.from_numpy(img),
            torch.tensor(rel,         dtype=torch.long),
            torch.tensor(scale,       dtype=torch.long),
            torch.tensor(shape_r_lbl, dtype=torch.long),
            torch.tensor(shape_b_lbl, dtype=torch.long),
        )

# -----------------------
# Model (same architecture as other 256 scripts)
# -----------------------

class Encoder(nn.Module):
    # 64 -> 32 -> 16 -> 8 -> 4
    def __init__(self, in_channels=NUM_CH, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32,  kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32,          64,  kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64,          128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128,         256, kernel_size=4, stride=2, padding=1)  # 4x4
        self.fc    = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z

class Decoder(nn.Module):
    def __init__(self, out_channels=NUM_CH, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64,  kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64,  32,  kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32,  out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, 4, 4)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))
        return x

class SceneModelEdges256(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        # heads for compatibility (unused here)
        self.rel_head      = nn.Linear(LATENT_DIM, 5)
        self.scale_head    = nn.Linear(LATENT_DIM, 3)
        self.shape_r_head  = nn.Linear(LATENT_DIM, 2)
        self.shape_b_head  = nn.Linear(LATENT_DIM, 2)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

def load_scene_model():
    print(f"[curvField256] Loading SceneModel from {CKPT_SCENEMODEL}")
    ckpt = torch.load(CKPT_SCENEMODEL, map_location=DEVICE)
    model = SceneModelEdges256()
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.to(DEVICE)
    model.eval()
    return model

# -----------------------
# Main
# -----------------------

def main():
    from sklearn.decomposition import PCA

    print(f"[curvField256] Using device: {DEVICE}")
    model = load_scene_model()

    # 1) Generate dataset & encode
    print(f"[curvField256] Generating dataset: N={N_SAMPLES}")
    ds = GeomEdges64Dataset(N_SAMPLES)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_z = []
    all_rel = []

    with torch.no_grad():
        for imgs, rel, scale, s_r, s_b in dl:
            imgs = imgs.to(DEVICE)
            z = model.encode(imgs)
            all_z.append(z.cpu())
            all_rel.append(rel)

    Z   = torch.cat(all_z, dim=0).numpy()           # (N, 256)
    rel = torch.cat(all_rel, dim=0).numpy()         # (N,)

    N, D = Z.shape
    print(f"[curvField256] Latent shape: {Z.shape}")

    # 2) Relation means, global mean, LR/AB generators
    mu0 = Z.mean(axis=0)

    mu_rel = {}
    for r in range(5):
        mask = (rel == r)
        mu_rel[r] = Z[mask].mean(axis=0)
        print(f"[curvField256] relation {r}={REL_NAMES[r]:>7}, "
              f"count={mask.sum()}, ||mu_r||={np.linalg.norm(mu_rel[r]-mu0):.3f}")

    g_x = mu_rel[REL_RIGHT] - mu_rel[REL_LEFT]
    g_y = mu_rel[REL_BELOW] - mu_rel[REL_ABOVE]

    # normalize
    g_x = g_x / np.linalg.norm(g_x)
    g_y = g_y / np.linalg.norm(g_y)

    angle = math.degrees(math.acos(
        np.clip(np.dot(g_x, g_y), -1.0, 1.0)
    ))
    print(f"[curvField256] angle(g_x, g_y) = {angle:.2f}°")

    # 3) Decompose into plane + residual
    dZ   = Z - mu0[None, :]
    t_x  = dZ @ g_x
    t_y  = dZ @ g_y
    plane = np.outer(t_x, g_x) + np.outer(t_y, g_y)
    Rperp = dZ - plane

    rnorm = np.linalg.norm(Rperp, axis=1)
    dnorm = np.linalg.norm(dZ, axis=1)
    frac  = rnorm / np.maximum(dnorm, 1e-8)
    print("[curvField256] Residual norms ||r_perp|| stats:")
    print(f"   mean = {rnorm.mean():.4f}, std = {rnorm.std():.4f}, "
          f"min = {rnorm.min():.4f}, max = {rnorm.max():.4f}")
    print("[curvField256] Relative residual ||r_perp|| / ||z-mu0|| stats:")
    print(f"   mean = {frac.mean():.4f}, std = {frac.std():.4f}, "
          f"min = {frac.min():.4f}, max = {frac.max():.4f}")

    # 4) PCA on residuals; take first 2 PCs as curvature basis h1,h2
    max_components = 32
    n_comp = min(max_components, D, N)
    print("[curvField256] PCA on residual subspace...")
    pca = PCA(n_components=n_comp, svd_solver="full", random_state=0)
    pca.fit(Rperp)
    evr = pca.explained_variance_ratio_
    for i, r in enumerate(evr, 1):
        print(f"  PC{i:02d}: {100*r:5.2f}% variance")
    print(f"  cumulative PC1..PC{n_comp}: {100*evr.cumsum()[-1]:.2f}%")

    h1 = pca.components_[0]      # already unit + orthogonal
    h2 = pca.components_[1]

    # 5) Fiber coords u1, u2 in curvature basis
    u1 = Rperp @ h1
    u2 = Rperp @ h2
    umag = np.sqrt(u1**2 + u2**2)

    # -----------------------
    # Plots
    # -----------------------
    # A. Reference scatter: relations colored, no curvature
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=120)
    for r in range(5):
        mask = (rel == r)
        ax.scatter(t_x[mask], t_y[mask], s=6, alpha=0.5,
                   color=REL_COLORS[r], label=REL_NAMES[r])
    ax.axhline(0.0, color="k", lw=0.5)
    ax.axvline(0.0, color="k", lw=0.5)
    ax.set_xlabel("t_x (LR generator coord)")
    ax.set_ylabel("t_y (AB generator coord)")
    ax.set_title("Global generator coordinates (latent256)\ncolor = relation")
    ax.legend(markerscale=2, fontsize=8)
    plt.tight_layout()
    out_ref = os.path.join(OUT_DIR, "curvature_field_reference_latent256.png")
    plt.savefig(out_ref)
    print(f"[curvField256] Saved reference scatter -> {out_ref}")
    plt.close(fig)

    # Helper for curvature-colored scatters
    def plot_curv_field(values, title_suffix, fname, cmap="coolwarm"):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=120)
        sc = ax.scatter(t_x, t_y, c=values, s=6, cmap=cmap, alpha=0.8)
        ax.axhline(0.0, color="k", lw=0.5)
        ax.axvline(0.0, color="k", lw=0.5)
        ax.set_xlabel("t_x (LR generator coord)")
        ax.set_ylabel("t_y (AB generator coord)")
        ax.set_title(f"Curvature field in {title_suffix} (latent256)")
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(title_suffix)
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, fname)
        plt.savefig(out_path)
        print(f"[curvField256] Saved {title_suffix} field -> {out_path}")
        plt.close(fig)

    # B. u1 field
    plot_curv_field(u1, "u1 (projection on h1)",
                    "curvature_field_u1_latent256.png")

    # C. u2 field
    plot_curv_field(u2, "u2 (projection on h2)",
                    "curvature_field_u2_latent256.png")

    # D. curvature magnitude
    plot_curv_field(umag, "||u|| = sqrt(u1^2 + u2^2)",
                    "curvature_field_umag_latent256.png",
                    cmap="viridis")

    print("[curvField256] Done.")

if __name__ == "__main__":
    main()
