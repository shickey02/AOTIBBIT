#!/usr/bin/env python3
# geomlang_operator_group_basic32.py
#
# Operator algebra for relation maps in latent space.
#
# - Fits four affine maps:
#     A_LR: left  -> right
#     A_RL: right -> left
#     A_AB: above -> below
#     A_BA: below -> above
# - Tests approximate inverses:
#     A_RL(A_LR z_left)  ≈ z_left
#     A_BA(A_AB z_above) ≈ z_above
# - Tests commutativity on samples:
#     A_AB(A_LR z) vs A_LR(A_AB z)
# - Visualizes orbits as decoded images.
#
# Run from repo root:
#   python bbit_geomlang/geomlang_operator_group_basic32.py

import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ----------------- config -----------------
IMG_SIZE     = 32
NUM_CHANNELS = 3
LATENT_DIM   = 48

DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_SCENEMODEL  = os.path.join("outputs_basic32", "scene_model_basic32.pt")
OUT_DIR          = "outputs_basic32"
os.makedirs(OUT_DIR, exist_ok=True)

N_DATA   = 6000
BATCH_SZ = 256
EPOCHS   = 500
LR       = 1e-3

REL_LEFT   = 0
REL_RIGHT  = 1
REL_ABOVE  = 2
REL_BELOW  = 3
REL_OVER   = 4

# ----------------- drawing + dataset -----------------
def relation_from_centers(rx, ry, bx, by, tol=1):
    if rx + tol < bx - tol:
        return REL_LEFT
    if rx - tol > bx + tol:
        return REL_RIGHT
    if ry + tol < by - tol:
        return REL_ABOVE
    if ry - tol > by + tol:
        return REL_BELOW
    return REL_OVER

def draw_circle(grid, cx, cy, radius, ch):
    h, w = grid.shape[1], grid.shape[2]
    for y in range(h):
        for x in range(w):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                grid[ch, y, x] = 1.0

def draw_square(grid, cx, cy, half, ch):
    h, w = grid.shape[1], grid.shape[2]
    x0 = max(0, cx - half)
    x1 = min(w, cx + half + 1)
    y0 = max(0, cy - half)
    y1 = min(h, cy + half + 1)
    grid[ch, y0:y1, x0:x1] = 1.0

def draw_shape(grid, shape_id, cx, cy, size, ch):
    if shape_id == 0:
        draw_circle(grid, cx, cy, size, ch)
    else:
        draw_square(grid, cx, cy, size, ch)

def sample_scene():
    H = W = IMG_SIZE
    margin = 4
    shape_red = np.random.randint(0, 2)
    shape_blue = np.random.randint(0, 2)

    base = np.random.randint(4, 7)
    r_red = int(np.clip(base + np.random.randint(-1, 2), 3, 7))
    r_blue = int(np.clip(base + np.random.randint(-1, 2), 3, 7))

    def sample_center(r):
        cx = np.random.randint(margin + r, W - margin - r)
        cy = np.random.randint(margin + r, H - margin - r)
        return cx, cy

    # avoid degenerate complete overlap
    while True:
        rx, ry = sample_center(r_red)
        bx, by = sample_center(r_blue)
        if abs(rx - bx) > 2 or abs(ry - by) > 2:
            break

    grid = np.zeros((NUM_CHANNELS, H, W), dtype=np.float32)
    draw_shape(grid, shape_red, rx, ry, r_red, 0)
    draw_shape(grid, shape_blue, bx, by, r_blue, 2)

    rel = relation_from_centers(rx, ry, bx, by)
    return grid, rel

def generate_dataset(N):
    imgs = []
    rels = []
    for _ in range(N):
        g, r = sample_scene()
        imgs.append(g)
        rels.append(r)
    return np.stack(imgs, 0), np.array(rels, dtype=np.int64)

# ----------------- model -----------------
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
        self.fc     = nn.Linear(latent_dim, 64 * 4 * 4)
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

def load_scene_model(path=CKPT_SCENEMODEL):
    ckpt = torch.load(path, map_location=DEVICE)
    if isinstance(ckpt, SceneModel):
        model = ckpt
    else:
        model = SceneModel()
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
    model.to(DEVICE)
    model.eval()
    return model

# ----------------- affine operator -----------------
class AffineOp(nn.Module):
    """z ↦ A z + b"""
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)  # includes bias

    def forward(self, z):
        return self.linear(z)

def fit_affine(source, target, name):
    """source, target: [N, D]"""
    ds = TensorDataset(source, target)
    loader = DataLoader(ds, batch_size=BATCH_SZ, shuffle=True)
    op = AffineOp(LATENT_DIM).to(DEVICE)
    opt = torch.optim.Adam(op.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    print(f"[group] Fitting {name} on {len(ds)} pairs")
    for epoch in range(1, EPOCHS + 1):
        op.train()
        loss_sum = 0.0
        for z_s, z_t in loader:
            z_s = z_s.to(DEVICE)
            z_t = z_t.to(DEVICE)
            opt.zero_grad()
            pred = op(z_s)
            loss = loss_fn(pred, z_t)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * z_s.size(0)
        loss_mean = loss_sum / len(ds)
        if epoch % 100 == 1 or epoch == EPOCHS:
            print(f"[group]   {name} epoch {epoch:3d}/{EPOCHS}, MSE={loss_mean:.5f}")

    # final eval loss
    op.eval()
    with torch.no_grad():
        pred = op(source.to(DEVICE))
        mse  = F.mse_loss(pred, target.to(DEVICE)).item()
    print(f"[group] Final MSE for {name}: {mse:.5f}\n")
    return op

# ----------------- helpers -----------------
def to_rgb(img_chw):
    r = img_chw[0]
    b = img_chw[2]
    H, W = r.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    rgb[..., 0] = r
    rgb[..., 2] = b
    return rgb

# ----------------- main analysis -----------------
def main():
    print(f"[group] Using device: {DEVICE}")
    ae = load_scene_model()

    print(f"[group] Generating dataset: N={N_DATA}")
    imgs_np, rel_np = generate_dataset(N_DATA)
    x = torch.from_numpy(imgs_np).float().to(DEVICE)

    # encode to latents
    with torch.no_grad():
        z = ae.encode(x)  # [N, D]

    # index by relation
    idx_left   = np.where(rel_np == REL_LEFT)[0]
    idx_right  = np.where(rel_np == REL_RIGHT)[0]
    idx_above  = np.where(rel_np == REL_ABOVE)[0]
    idx_below  = np.where(rel_np == REL_BELOW)[0]

    print(f"[group] counts: left={len(idx_left)}, right={len(idx_right)}, "
          f"above={len(idx_above)}, below={len(idx_below)}")

    # align pairs by truncation
    n_lr  = min(len(idx_left),  len(idx_right))
    n_ab  = min(len(idx_above), len(idx_below))

    z_left   = z[idx_left[:n_lr]]
    z_right  = z[idx_right[:n_lr]]
    z_above  = z[idx_above[:n_ab]]
    z_below  = z[idx_below[:n_ab]]

    # fit four operators
    A_LR = fit_affine(z_left,  z_right, "A_LR (left → right)")
    A_RL = fit_affine(z_right, z_left,  "A_RL (right → left)")
    A_AB = fit_affine(z_above, z_below, "A_AB (above → below)")
    A_BA = fit_affine(z_below, z_above, "A_BA (below → above)")

    # ---------- inverse tests ----------
    print("[group] Testing inverse consistency...")

    with torch.no_grad():
        # left-right-left
        z_lr  = A_LR(z_left.to(DEVICE))
        z_lrl = A_RL(z_lr)
        mse_lrl = F.mse_loss(z_lrl, z_left.to(DEVICE)).item()

        # right-left-right
        z_rl  = A_RL(z_right.to(DEVICE))
        z_rlr = A_LR(z_rl)
        mse_rlr = F.mse_loss(z_rlr, z_right.to(DEVICE)).item()

        # above-below-above
        z_abv = A_AB(z_above.to(DEVICE))
        z_abab = A_BA(z_abv)
        mse_abab = F.mse_loss(z_abab, z_above.to(DEVICE)).item()

        # below-above-below
        z_bel = A_BA(z_below.to(DEVICE))
        z_baba = A_AB(z_bel)
        mse_baba = F.mse_loss(z_baba, z_below.to(DEVICE)).item()

    print(f"[group] MSE A_RL(A_LR z_left)  vs z_left : {mse_lrl:.4f}")
    print(f"[group] MSE A_LR(A_RL z_right) vs z_right: {mse_rlr:.4f}")
    print(f"[group] MSE A_BA(A_AB z_above)  vs z_above: {mse_abab:.4f}")
    print(f"[group] MSE A_AB(A_BA z_below) vs z_below: {mse_baba:.4f}")

    # ---------- commutator in image space ----------
    print("\n[group] Visualizing commutator A_AB ∘ A_LR vs A_LR ∘ A_AB")

    n_vis = 6
    rng = np.random.default_rng(0)
    vis_idx = rng.choice(len(z_left), size=n_vis, replace=False)

    with torch.no_grad():
        z0      = z_left[vis_idx].to(DEVICE)
        z_lr    = A_LR(z0)
        z_ab    = A_AB(z0)
        z_lr_ab = A_AB(z_lr)    # A_AB(A_LR z)
        z_ab_lr = A_LR(z_ab)    # A_LR(A_AB z)

        x0      = ae.decode(z0).cpu().numpy()
        x_lr    = ae.decode(z_lr).cpu().numpy()
        x_ab    = ae.decode(z_ab).cpu().numpy()
        x_lr_ab = ae.decode(z_lr_ab).cpu().numpy()
        x_ab_lr = ae.decode(z_ab_lr).cpu().numpy()

    # per-sample mean-absolute diff between orders
    diff = np.abs(x_lr_ab - x_ab_lr).mean(axis=(1, 2, 3))
    print("[group] Mean |A_AB(A_LR z) - A_LR(A_AB z)| (per sample):")
    print("        ", diff)

    # ---------- grid figure ----------
    fig, axes = plt.subplots(
        nrows=n_vis,
        ncols=5,
        figsize=(5 * 2.2, n_vis * 2.2),
        dpi=120,
    )

    fig.suptitle("Operator algebra (basic32): A_AB ∘ A_LR vs A_LR ∘ A_AB", fontsize=14)

    for r in range(n_vis):
        imgs = [
            x0[r],
            x_lr[r],
            x_ab[r],
            x_lr_ab[r],
            x_ab_lr[r],
        ]
        for c in range(5):
            ax = axes[r, c]
            ax.axis("off")
            if r == 0:
                titles = ["z (left)", "A_LR z", "A_AB z",
                          "A_AB(A_LR z)", "A_LR(A_AB z)"]
                ax.set_title(titles[c], fontsize=8)
            ax.imshow(to_rgb(imgs[c]))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(OUT_DIR, "operator_group_commutator_grid_basic32.png")
    plt.savefig(out_path)
    print(f"[group] Saved commutator grid -> {out_path}")

if __name__ == "__main__":
    main()
