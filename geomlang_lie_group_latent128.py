#!/usr/bin/env python3
# geomlang_lie_group_latent128.py
#
# Lie-group style analysis of relations in the edges+relscale model
# with LATENT_DIM = 128 (same architecture as the 64x64 edges model).
#
# Uses the checkpoint: outputs_edges_relscale/scene_model_edges_relscale.pt

import os, math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# -----------------------
# Config
# -----------------------
IMG_SIZE   = 64          # training resolution
NUM_CH     = 3           # red, blue, edges
LATENT_DIM = 128

N_SAMPLES  = 6000
BATCH_SIZE = 128

OUT_DIR         = "outputs_edges_relscale"
CKPT_SCENEMODEL = os.path.join(OUT_DIR, "scene_model_edges_relscale.pt")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REL_LEFT, REL_RIGHT, REL_ABOVE, REL_BELOW, REL_OVERLAP = range(5)
REL_NAMES = ["left_of", "right_of", "above", "below", "overlap"]

# -----------------------
# Scene generation (same as training)
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

        shape_red  = np.random.randint(0, 2)
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
# Model – 64x64 edges architecture with LATENT_DIM=128
# -----------------------

class Encoder128(nn.Module):
    def __init__(self, in_channels=NUM_CH, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32,  kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32,          64,  kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64,          128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128,         256, kernel_size=4, stride=2, padding=1)  # 4x4
        self.fc    = nn.Linear(256 * 4 * 4, latent_dim)  # 4096 -> 128

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z

class Decoder128(nn.Module):
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

class SceneModelEdges128(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder128()
        self.decoder = Decoder128()
        self.rel_head      = nn.Linear(LATENT_DIM, 5)
        self.scale_head    = nn.Linear(LATENT_DIM, 3)
        self.shape_r_head  = nn.Linear(LATENT_DIM, 2)
        self.shape_b_head  = nn.Linear(LATENT_DIM, 2)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

def load_scene_model():
    print(f"[lie128] Loading SceneModel from {CKPT_SCENEMODEL}")
    ckpt = torch.load(CKPT_SCENEMODEL, map_location=DEVICE)
    model = SceneModelEdges128()
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.to(DEVICE)
    model.eval()
    return model

# -----------------------
# Linear algebra helpers
# -----------------------

def pca_on_cloud(v, name, max_components=32):
    from sklearn.decomposition import PCA
    N, D = v.shape
    n_comp = min(max_components, D, N)
    pca = PCA(n_components=n_comp, svd_solver="full", random_state=0)
    pca.fit(v)
    evr = pca.explained_variance_ratio_

    print(f"\n[lie128] PCA on {name} difference cloud:")
    for i, r in enumerate(evr, 1):
        print(f"  PC{i:02d}: {100*r:5.2f}% variance")
    print(f"  cumulative PC1..PC{n_comp}: {100*evr.cumsum()[-1]:.2f}%")

    eff_rank = int((evr > 0.01).sum())
    print(f"  effective rank (>=1% each): {eff_rank}")
    return pca, eff_rank

def principal_angles(U, V):
    Uq, _ = np.linalg.qr(U)
    Vq, _ = np.linalg.qr(V)
    M = Uq.T @ Vq
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    angles = np.arccos(s)
    return np.sort(angles)[::-1]

# -----------------------
# Main
# -----------------------

def main():
    print(f"[lie128] Using device: {DEVICE}")
    model = load_scene_model()

    print(f"[lie128] Generating dataset: N={N_SAMPLES}")
    ds = GeomEdges64Dataset(N_SAMPLES)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_rel = []
    all_z   = []

    with torch.no_grad():
        for imgs, rel, scale, s_r, s_b in dl:
            imgs = imgs.to(DEVICE)
            rel  = rel.to(DEVICE)
            z    = model.encode(imgs)

            all_rel.append(rel.cpu())
            all_z.append(z.cpu())

    all_rel = torch.cat(all_rel, dim=0)
    all_z   = torch.cat(all_z,   dim=0)

    rel_np = all_rel.numpy()
    z_np   = all_z.numpy()

    for r in range(5):
        n = int((rel_np == r).sum())
        print(f"[lie128] relation {r}: {REL_NAMES[r]:>7}, count={n}")

    rng = np.random.default_rng(0)
    idx_left  = np.where(rel_np == REL_LEFT)[0]
    idx_right = np.where(rel_np == REL_RIGHT)[0]
    idx_above = np.where(rel_np == REL_ABOVE)[0]
    idx_below = np.where(rel_np == REL_BELOW)[0]

    def make_pairs(a, b, max_pairs=None):
        n = min(len(a), len(b))
        if max_pairs is not None:
            n = min(n, max_pairs)
        if n == 0:
            return np.array([], int), np.array([], int)
        a = rng.permutation(a)[:n]
        b = rng.permutation(b)[:n]
        return a, b

    left_idx,  right_idx = make_pairs(idx_left,  idx_right)
    above_idx, below_idx = make_pairs(idx_above, idx_below)

    z_left   = z_np[left_idx]
    z_right  = z_np[right_idx]
    z_above  = z_np[above_idx]
    z_below  = z_np[below_idx]

    v_lr = z_right - z_left
    v_ab = z_below - z_above

    for name, v in [("left→right", v_lr), ("above→below", v_ab)]:
        norms = np.linalg.norm(v, axis=1)
        print(f"\n[lie128] {name} norms:")
        print(f"  mean={norms.mean():.4f}, std={norms.std():.4f}, "
              f"min={norms.min():.4f}, max={norms.max():.4f}")

    pca_lr, rank_lr = pca_on_cloud(v_lr, "left→right")
    pca_ab, rank_ab = pca_on_cloud(v_ab, "above→below")

    k = min(rank_lr, rank_ab, 16)
    U_lr = pca_lr.components_[:k].T
    U_ab = pca_ab.components_[:k].T

    angles = principal_angles(U_lr, U_ab)
    print("\n[lie128] Principal angles between LR and AB subspaces (deg):")
    for i, a in enumerate(angles, 1):
        print(f"  θ{i:02d} = {a * 180.0 / math.pi:6.2f}°")

    # Simple PCA visualizations
    lr_2d = pca_lr.transform(v_lr)[:, :2]
    ab_2d = pca_ab.transform(v_ab)[:, :2]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
    ax[0].scatter(lr_2d[:, 0], lr_2d[:, 1], s=5, alpha=0.4)
    ax[0].set_title("v_LR cloud (first 2 PCs)")
    ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2")

    ax[1].scatter(ab_2d[:, 0], ab_2d[:, 1], s=5, alpha=0.4, color="tab:orange")
    ax[1].set_title("v_AB cloud (first 2 PCs)")
    ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2")

    plt.tight_layout()
    out_scatter = os.path.join(OUT_DIR, "lie_group_diffcloud_pca_latent128.png")
    plt.savefig(out_scatter)
    print(f"[lie128] Saved diff-cloud PCA scatter -> {out_scatter}")

    evr_lr = pca_lr.explained_variance_ratio_
    evr_ab = pca_ab.explained_variance_ratio_

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
    ax[0].plot(np.arange(1, len(evr_lr)+1), evr_lr, marker="o")
    ax[0].set_title("Scree: LR difference cloud")
    ax[0].set_xlabel("PC index"); ax[0].set_ylabel("variance ratio")

    ax[1].plot(np.arange(1, len(evr_ab)+1), evr_ab, marker="o", color="tab:orange")
    ax[1].set_title("Scree: AB difference cloud")
    ax[1].set_xlabel("PC index"); ax[1].set_ylabel("variance ratio")

    plt.tight_layout()
    out_scree = os.path.join(OUT_DIR, "lie_group_scree_latent128.png")
    plt.savefig(out_scree)
    print(f"[lie128] Saved scree plots -> {out_scree}")

    print("\n[lie128] Done.")

if __name__ == "__main__":
    main()
