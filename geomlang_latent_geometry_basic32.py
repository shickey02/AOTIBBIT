#!/usr/bin/env python3
# geomlang_latent_geometry_basic32.py
#
# Advanced latent geometry analysis for basic32:
# - same synthetic dataset generation as arithmetic script
# - compute per-relation latent clusters
# - build difference vectors for left↔right and above↔below
# - run PCA over these difference clouds (curvature / dimensionality)
# - fit linear operators A_LR, A_AB (z_right ≈ A_LR z_left, etc.)
# - evaluate mapping error and commutator norm ||[A_LR, A_AB]||_F
# - visualize original vs A*z decoded

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.decomposition import PCA

# --------------------
# Config
# --------------------
IMG_SIZE     = 32
NUM_CHANNELS = 3
LATENT_DIM   = 48

N_SAMPLES    = 6000
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR         = "outputs_basic32"
CKPT_SCENEMODEL = os.path.join(OUT_DIR, "scene_model_basic32.pt")
os.makedirs(OUT_DIR, exist_ok=True)

REL_NAMES = ["left_of", "right_of", "above", "below", "overlap"]

# --------------------
# Drawing & generation (same as in arithmetic script)
# --------------------
def draw_circle(grid, cx, cy, radius, channel_idx):
    h, w = grid.shape[1], grid.shape[2]
    for y in range(h):
        for x in range(w):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                grid[channel_idx, y, x] = 1.0


def draw_square(grid, cx, cy, half_size, channel_idx):
    h, w = grid.shape[1], grid.shape[2]
    x0 = max(0, cx - half_size)
    x1 = min(w, cx + half_size + 1)
    y0 = max(0, cy - half_size)
    y1 = min(h, cy + half_size + 1)
    grid[channel_idx, y0:y1, x0:x1] = 1.0


def draw_shape(grid, shape_id, cx, cy, size, channel_idx):
    if shape_id == 0:
        draw_circle(grid, cx, cy, size, channel_idx)
    else:
        draw_square(grid, cx, cy, size, channel_idx)


def generate_scene():
    H = W = IMG_SIZE
    margin = 4

    shape_red  = np.random.randint(0, 2)
    shape_blue = np.random.randint(0, 2)
    base = np.random.randint(4, 7)
    r_red  = int(np.clip(base + np.random.randint(-1, 2), 3, 7))
    r_blue = int(np.clip(base + np.random.randint(-1, 2), 3, 7))

    img = np.zeros((NUM_CHANNELS, H, W), dtype=np.float32)

    rel_idx = np.random.randint(0, len(REL_NAMES))

    def rand_x(size):
        return np.random.randint(margin + size, W - margin - size)
    def rand_y(size):
        return np.random.randint(margin + size, H - margin - size)

    if rel_idx == 0:  # left_of
        cy = rand_y(max(r_red, r_blue))
        cx_red  = np.random.randint(margin + r_red,  W//2 - margin)
        cx_blue = np.random.randint(W//2 + margin,  W - margin - r_blue)
        cy_red = cy_blue = cy
    elif rel_idx == 1:  # right_of
        cy = rand_y(max(r_red, r_blue))
        cx_blue = np.random.randint(margin + r_blue, W//2 - margin)
        cx_red  = np.random.randint(W//2 + margin, W - margin - r_red)
        cy_red = cy_blue = cy
    elif rel_idx == 2:  # above
        cx = rand_x(max(r_red, r_blue))
        cy_red  = np.random.randint(margin + r_red,  H//2 - margin)
        cy_blue = np.random.randint(H//2 + margin,  H - margin - r_blue)
        cx_red = cx_blue = cx
    elif rel_idx == 3:  # below
        cx = rand_x(max(r_red, r_blue))
        cy_blue = np.random.randint(margin + r_blue, H//2 - margin)
        cy_red  = np.random.randint(H//2 + margin, H - margin - r_red)
        cx_red = cx_blue = cx
    else:  # overlap/near
        cx = rand_x(max(r_red, r_blue))
        cy = rand_y(max(r_red, r_blue))
        jitter = np.random.randint(-2, 3, size=2)
        cx_red, cy_red = cx, cy
        cx_blue = int(np.clip(cx + jitter[0], margin + r_blue, W - margin - r_blue))
        cy_blue = int(np.clip(cy + jitter[1], margin + r_blue, H - margin - r_blue))

    draw_shape(img, shape_red,  cx_red,  cy_red,  r_red,  0)
    draw_shape(img, shape_blue, cx_blue, cy_blue, r_blue, 2)

    return img, rel_idx


def generate_dataset(n_samples):
    imgs = []
    rels = []
    for _ in range(n_samples):
        img, r = generate_scene()
        imgs.append(img)
        rels.append(r)
    return np.stack(imgs, 0), np.array(rels, dtype=np.int64)

# --------------------
# SceneModel (AE) – same as before
# --------------------
class Encoder(nn.Module):
    def __init__(self, in_channels=NUM_CHANNELS, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, 2, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 2, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.fc    = nn.Linear(64 * 4 * 4, latent_dim)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self, out_channels=NUM_CHANNELS, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc      = nn.Linear(latent_dim, 64 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(16, out_channels, 4, 2, 1)
    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 64, 4, 4)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        return x


class SceneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.rel_head        = nn.Linear(LATENT_DIM, 6)
        self.scale_head      = nn.Linear(LATENT_DIM, 3)
        self.shape_red_head  = nn.Linear(LATENT_DIM, 2)
        self.shape_blue_head = nn.Linear(LATENT_DIM, 2)
    def encode(self, x):
        return self.encoder(x)
    def decode(self, z):
        return self.decoder(z)


def load_scene_model():
    print(f"[geom] Loading SceneModel from {CKPT_SCENEMODEL}")
    ckpt = torch.load(CKPT_SCENEMODEL, map_location=DEVICE)
    model = SceneModel()
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt, strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.to(DEVICE)
    model.eval()
    return model

# --------------------
# Helpers
# --------------------
def to_rgb(img_chw):
    r = img_chw[0]
    b = img_chw[2]
    h, w = r.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[..., 0] = r
    rgb[..., 2] = b
    return rgb


def fit_linear_operator(z_src, z_tgt, n_epochs=500, lr=5e-3):
    """
    Fit A (D x D) and b (D) such that A z_src + b ≈ z_tgt
    via simple gradient descent.
    """
    z_src_t = torch.from_numpy(z_src).float().to(DEVICE)  # [N,D]
    z_tgt_t = torch.from_numpy(z_tgt).float().to(DEVICE)

    A = torch.randn(LATENT_DIM, LATENT_DIM, device=DEVICE) * 0.01
    b = torch.zeros(LATENT_DIM, device=DEVICE)

    A.requires_grad_(True)
    b.requires_grad_(True)

    opt = torch.optim.Adam([A, b], lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, n_epochs + 1):
        opt.zero_grad()
        pred = z_src_t @ A.T + b  # [N,D]
        loss = loss_fn(pred, z_tgt_t)
        loss.backward()
        opt.step()
        if epoch % 100 == 0 or epoch == 1:
            print(f"[geom]   epoch {epoch:3d}/{n_epochs}, loss={loss.item():.5f}")

    with torch.no_grad():
        A_final = A.detach().cpu().numpy()
        b_final = b.detach().cpu().numpy()
        pred_final = z_src @ A_final.T + b_final
        mse = ((pred_final - z_tgt) ** 2).mean()

    return A_final, b_final, mse

# --------------------
# Main
# --------------------
def main():
    print(f"[geom] Using device: {DEVICE}")
    ae = load_scene_model()

    print(f"[geom] Generating dataset: N={N_SAMPLES}")
    imgs_np, rels = generate_dataset(N_SAMPLES)

    # encode
    x = torch.from_numpy(imgs_np).float().to(DEVICE)
    with torch.no_grad():
        z = ae.encode(x).cpu().numpy()

    # clusters
    zs_by_rel = []
    for r in range(len(REL_NAMES)):
        mask = (rels == r)
        cluster = z[mask]
        zs_by_rel.append(cluster)
        print(f"[geom] relation {r}: {REL_NAMES[r]:>7s}, count={cluster.shape[0]}")

    # build difference clouds for left↔right and above↔below
    rng = np.random.default_rng(0)
    def paired_diffs(rel_a, rel_b, n_pairs=2000):
        za = zs_by_rel[rel_a]
        zb = zs_by_rel[rel_b]
        n = min(len(za), len(zb), n_pairs)
        idx_a = rng.choice(len(za), size=n, replace=False)
        idx_b = rng.choice(len(zb), size=n, replace=False)
        diffs = zb[idx_b] - za[idx_a]
        return diffs, za[idx_a], zb[idx_b]

    d_lr, z_left,  z_right  = paired_diffs(0, 1)
    d_ab, z_above, z_below  = paired_diffs(2, 3)

    # PCA on difference clouds
    def pca_report(diffs, name):
        pca = PCA(n_components=min(10, diffs.shape[1]))
        pca.fit(diffs)
        evr = pca.explained_variance_ratio_
        print(f"\n[geom] PCA on {name} difference cloud:")
        for i, v in enumerate(evr):
            print(f"  PC{i+1}: {v*100:.2f}% variance")
        cum = evr.cumsum()
        print(f"  cumulative PC1..PC4: {cum[:4].sum()*100:.2f}%")
        return pca

    pca_lr = pca_report(d_lr, "left→right")
    pca_ab = pca_report(d_ab, "above→below")

    # Fit linear operators
    print("\n[geom] Fitting linear operator A_LR (z_left → z_right)")
    A_LR, b_LR, mse_LR = fit_linear_operator(z_left, z_right, n_epochs=500, lr=5e-3)
    print(f"[geom] Final MSE z_right ≈ A_LR z_left + b: {mse_LR:.5f}")

    print("\n[geom] Fitting linear operator A_AB (z_above → z_below)")
    A_AB, b_AB, mse_AB = fit_linear_operator(z_above, z_below, n_epochs=500, lr=5e-3)
    print(f"[geom] Final MSE z_below ≈ A_AB z_above + b: {mse_AB:.5f}")

    # Commutator norm
    print("\n[geom] Computing commutator norm ||[A_LR, A_AB]||_F")
    ALR = torch.from_numpy(A_LR)
    AAB = torch.from_numpy(A_AB)
    comm = ALR @ AAB - AAB @ ALR
    comm_norm = torch.linalg.norm(comm, ord="fro").item()
    print(f"[geom] ||[A_LR, A_AB]||_F = {comm_norm:.5f}")

    # --------------------
    # Visualization grid: original vs A*z
    # --------------------
    ae.eval()
    with torch.no_grad():
        # pick a handful of left and above examples
        n_rows = 6
        idx_left_vis  = rng.choice(z_left.shape[0],  size=min(n_rows, z_left.shape[0]),  replace=False)
        idx_above_vis = rng.choice(z_above.shape[0], size=min(n_rows, z_above.shape[0]), replace=False)

        z_left_vis   = z_left[idx_left_vis]
        z_right_pred = z_left_vis @ A_LR.T + b_LR

        z_above_vis  = z_above[idx_above_vis]
        z_below_pred = z_above_vis @ A_AB.T + b_AB

        z_left_t   = torch.from_numpy(z_left_vis).float().to(DEVICE)
        z_rp_t     = torch.from_numpy(z_right_pred).float().to(DEVICE)
        z_above_t  = torch.from_numpy(z_above_vis).float().to(DEVICE)
        z_bp_t     = torch.from_numpy(z_below_pred).float().to(DEVICE)

        img_left   = ae.decode(z_left_t).cpu().numpy()
        img_rpred  = ae.decode(z_rp_t).cpu().numpy()
        img_above  = ae.decode(z_above_t).cpu().numpy()
        img_bpred  = ae.decode(z_bp_t).cpu().numpy()

    ncols = 4
    nrows = max(img_left.shape[0], img_above.shape[0])
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * 2.0, nrows * 2.0 / 3.0),
        dpi=120,
    )
    fig.suptitle("Linear operator geometry: z → A z (basic32)", fontsize=14)

    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            ax.axis("off")
            if r == 0:
                if c == 0:
                    ax.set_title("left (src)", fontsize=8)
                elif c == 1:
                    ax.set_title("A_LR z_left", fontsize=8)
                elif c == 2:
                    ax.set_title("above (src)", fontsize=8)
                else:
                    ax.set_title("A_AB z_above", fontsize=8)

            if c == 0 and r < img_left.shape[0]:
                ax.imshow(to_rgb(img_left[r]))
            elif c == 1 and r < img_rpred.shape[0]:
                ax.imshow(to_rgb(img_rpred[r]))
            elif c == 2 and r < img_above.shape[0]:
                ax.imshow(to_rgb(img_above[r]))
            elif c == 3 and r < img_bpred.shape[0]:
                ax.imshow(to_rgb(img_bpred[r]))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_grid = os.path.join(OUT_DIR, "latent_geometry_operators_basic32.png")
    plt.savefig(out_grid)
    print(f"[geom] Saved operator grid -> {out_grid}")


if __name__ == "__main__":
    main()
