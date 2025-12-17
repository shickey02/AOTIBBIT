#!/usr/bin/env python3
# geomlang_linear_consistency_basic32.py
#
# Step 1: Test how "linear" the learned relation operators are in latent space.
#
# What this does:
#   1. Load the trained SceneModel AE from outputs_basic32/scene_model_basic32.pt
#   2. Regenerate a basic32 dataset and encode it to latents
#   3. Fit two affine maps:
#        A_LR : z_left  -> z_right
#        A_AB : z_above -> z_below
#   4. For many random latent pairs (z1, z2) it measures
#        δ = A(z1 + z2) - A(z1) - A(z2)
#      and collects the L2 norm ||δ||.
#      For a *perfect* linear map with no bias, δ would be ~0.
#      For an affine map Wx + b, δ is ~ -b, so ||δ|| ≈ ||b|| (constant).
#   5. Prints summary stats and saves a histogram + a small grid of decoded
#      "worst-case" examples.
#
# Run from project root:
#   python bbit_geomlang/geomlang_linear_consistency_basic32.py

import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt

# -----------------------
# Config
# -----------------------
IMG_SIZE      = 32
NUM_CHANNELS  = 3
LATENT_DIM    = 48

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR = "outputs_basic32"
os.makedirs(OUT_DIR, exist_ok=True)

CKPT_SCENEMODEL = os.path.join(OUT_DIR, "scene_model_basic32.pt")

N_SAMPLES    = 6000   # dataset size for fitting operators
BATCH_SIZE   = 256
EPOCHS_FIT   = 500
LR_FIT       = 1e-3

N_PAIR_SAMPLES = 20000  # how many (z1, z2) pairs to sample for linearity test

REL_NAMES = ["left_of", "right_of", "above", "below", "overlap"]


# -----------------------
# Simple basic32 generator (same family as before)
# -----------------------
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


def sample_scene():
    """
    Sample one static scene:
      - red + blue blob (circle or square)
      - one of five relations
    Returns:
      img: [3,H,W] float32
      rel_id: int in [0..4]
    """
    H = W = IMG_SIZE
    margin = 4

    grid = np.zeros((NUM_CHANNELS, H, W), dtype=np.float32)

    # shapes and sizes
    shape_red = np.random.randint(0, 2)
    shape_blue = np.random.randint(0, 2)

    base = np.random.randint(4, 7)
    r_red = int(np.clip(base + np.random.randint(-1, 2), 3, 7))
    r_blue = int(np.clip(base + np.random.randint(-1, 2), 3, 7))

    # sample relation
    rel_id = np.random.randint(0, 5)

    # place shapes according to relation
    # first choose a ref center for red
    cx_r = np.random.randint(margin + r_red, W - margin - r_red)
    cy_r = np.random.randint(margin + r_red, H - margin - r_red)

    dx = np.random.randint(r_red + r_blue + 2, r_red + r_blue + 8)
    dy = np.random.randint(r_red + r_blue + 2, r_red + r_blue + 8)

    if rel_id == 0:  # left_of: red is left of blue
        cx_b = min(W - margin - r_blue, cx_r + dx)
        cy_b = cy_r
    elif rel_id == 1:  # right_of: red is right of blue
        cx_b = max(margin + r_blue, cx_r - dx)
        cy_b = cy_r
    elif rel_id == 2:  # above
        cx_b = cx_r
        cy_b = min(H - margin - r_blue, cy_r + dy)
    elif rel_id == 3:  # below
        cx_b = cx_r
        cy_b = max(margin + r_blue, cy_r - dy)
    else:  # overlap
        cx_b = cx_r + np.random.randint(-2, 3)
        cy_b = cy_r + np.random.randint(-2, 3)

    draw_shape(grid, shape_red, cx_r, cy_r, r_red, 0)  # red
    draw_shape(grid, shape_blue, cx_b, cy_b, r_blue, 2)  # blue
    return grid, rel_id


def generate_dataset(n=N_SAMPLES):
    imgs = np.zeros((n, NUM_CHANNELS, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    rels = np.zeros((n,), dtype=np.int64)
    for i in range(n):
        img, rel = sample_scene()
        imgs[i] = img
        rels[i] = rel
    return imgs, rels


# -----------------------
# SceneModel (AE) – same arch as before
# -----------------------
class Encoder(nn.Module):
    def __init__(self, in_channels=NUM_CHANNELS, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1)  # 16x16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)           # 8x8
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)           # 4x4
        self.fc = nn.Linear(64 * 4 * 4, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self, out_channels=NUM_CHANNELS, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(16, out_channels, kernel_size=4, stride=2, padding=1)

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
        # heads not used, but kept for checkpoint compatibility
        self.rel_head = nn.Linear(LATENT_DIM, 6)
        self.scale_head = nn.Linear(LATENT_DIM, 3)
        self.shape_red_head = nn.Linear(LATENT_DIM, 2)
        self.shape_blue_head = nn.Linear(LATENT_DIM, 2)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


def load_scene_model(path=CKPT_SCENEMODEL):
    print(f"[lin] Loading SceneModel from {path}")
    ckpt = torch.load(path, map_location=DEVICE)
    if isinstance(ckpt, SceneModel):
        model = ckpt
    else:
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


# -----------------------
# Fit affine operators A_LR and A_AB
# -----------------------
class AffineMap(nn.Module):
    """Simple affine map: z_out = W z + b."""
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)  # includes bias

    def forward(self, z):
        return self.linear(z)


def fit_operator(z_src, z_tgt, name="A"):
    """
    Fit an affine map z_tgt ≈ A(z_src) by MSE.
    z_src, z_tgt: [N, D] tensors.
    """
    D = z_src.size(1)
    model = AffineMap(D).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR_FIT)
    loss_fn = nn.MSELoss()

    ds = TensorDataset(z_src, z_tgt)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(1, EPOCHS_FIT + 1):
        model.train()
        total = 0.0
        for xs, ys in dl:
            xs = xs.to(DEVICE)
            ys = ys.to(DEVICE)
            opt.zero_grad()
            pred = model(xs)
            loss = loss_fn(pred, ys)
            loss.backward()
            opt.step()
            total += loss.item() * xs.size(0)
        mse = total / len(ds)
        if epoch % 100 == 1 or epoch == EPOCHS_FIT:
            print(f"[lin] {name} epoch {epoch:3d}/{EPOCHS_FIT}, MSE={mse:.5f}")

    # final train MSE for logging
    with torch.no_grad():
        pred = model(z_src.to(DEVICE))
        mse_final = loss_fn(pred, z_tgt.to(DEVICE)).item()
    print(f"[lin] Final MSE {name}: {mse_final:.5f}")
    return model


# -----------------------
# Linearity test
# -----------------------
@torch.no_grad()
def test_linearity(model, z_pool, name="A"):
    """
    Given an affine operator model(z) = Wz + b and a pool of latents z_pool [N,D],
    sample many random pairs (z1, z2) and measure:
        δ = A(z1 + z2) - A(z1) - A(z2)
        err = ||δ||_2
    Also compare with the bias norm ||b||, which is what you'd expect for an
    ideal affine map.
    """
    z_pool = z_pool.to(DEVICE)
    N, D = z_pool.shape

    # sample indices of pairs
    idx1 = np.random.randint(0, N, size=N_PAIR_SAMPLES)
    idx2 = np.random.randint(0, N, size=N_PAIR_SAMPLES)

    errs = np.zeros(N_PAIR_SAMPLES, dtype=np.float32)

    linear = model.linear
    W = linear.weight.detach()        # [D, D]
    b = linear.bias.detach()          # [D]
    b_norm = float(b.norm().cpu().item())

    print(f"[lin] Testing linearity for {name}")
    print(f"[lin]   ||bias|| = {b_norm:.4f}")

    for i in range(N_PAIR_SAMPLES):
        z1 = z_pool[idx1[i]]
        z2 = z_pool[idx2[i]]

        z_sum = z1 + z2

        Az1 = model(z1.unsqueeze(0))[0]
        Az2 = model(z2.unsqueeze(0))[0]
        Az_sum = model(z_sum.unsqueeze(0))[0]

        delta = Az_sum - Az1 - Az2
        errs[i] = delta.norm().cpu().item()

    mean_err = float(errs.mean())
    std_err = float(errs.std())
    max_err = float(errs.max())
    min_err = float(errs.min())

    print(f"[lin] Linearity error stats for {name}:")
    print(f"       mean ||δ|| = {mean_err:.4f}")
    print(f"       std  ||δ|| = {std_err:.4f}")
    print(f"       min  ||δ|| = {min_err:.4f}")
    print(f"       max  ||δ|| = {max_err:.4f}")
    print("       (for a perfectly affine map, ||δ|| should be ~||bias||, a constant)")

    # Histogram plot
    plt.figure(figsize=(6, 4), dpi=120)
    plt.hist(errs, bins=50, alpha=0.8)
    plt.axvline(b_norm, color="red", linestyle="--", label="||bias||")
    plt.xlabel("||A(z1+z2) - A(z1) - A(z2)||")
    plt.ylabel("count")
    plt.title(f"Linearity deviation histogram for {name} (basic32)")
    plt.legend()
    out_path = os.path.join(OUT_DIR, f"linear_consistency_hist_{name}_basic32.png")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"[lin] Saved histogram -> {out_path}")

    return errs, b_norm


@torch.no_grad()
def visualize_worst_cases(ae, model, z_pool, errs, name="A", n_examples=6):
    """
    Decode a few worst-case pairs (largest ||δ||) to see what they look like.
    Creates a grid with:
        z1 source, A(z1)
        z2 source, A(z2)
        z1+z2 synthetic, A(z1+z2)
    """
    N, D = z_pool.shape
    z_pool = z_pool.to(DEVICE)

    # get indices of worst errors
    worst_idx = np.argsort(-errs)[:n_examples]

    def to_rgb(img_chw):
        r = img_chw[0]
        b = img_chw[2]
        H, W = r.shape
        rgb = np.zeros((H, W, 3), dtype=np.float32)
        rgb[..., 0] = r
        rgb[..., 2] = b
        return rgb

    fig, axes = plt.subplots(
        nrows=n_examples,
        ncols=6,
        figsize=(6 * 1.2, n_examples * 1.2),
        dpi=120,
    )
    fig.suptitle(f"Linearity worst-case examples for {name} (basic32)", fontsize=12)

    for row, idx in enumerate(worst_idx):
        z1 = z_pool[np.random.randint(0, N)]
        z2 = z_pool[np.random.randint(0, N)]
        z_sum = z1 + z2

        Az1 = model(z1.unsqueeze(0))[0]
        Az2 = model(z2.unsqueeze(0))[0]
        Az_sum = model(z_sum.unsqueeze(0))[0]

        imgs = []
        for z in [z1, Az1, z2, Az2, z_sum, Az_sum]:
            img = ae.decode(z.unsqueeze(0))[0].detach().cpu().numpy()
            imgs.append(to_rgb(img))

        titles = ["z1", "A(z1)", "z2", "A(z2)", "z1+z2", "A(z1+z2)"]
        for col in range(6):
            ax = axes[row, col]
            ax.axis("off")
            if row == 0:
                ax.set_title(titles[col], fontsize=8)
            ax.imshow(imgs[col])

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(OUT_DIR, f"linear_consistency_examples_{name}_basic32.png")
    plt.savefig(out_path)
    print(f"[lin] Saved examples grid -> {out_path}")


# -----------------------
# Main
# -----------------------
def main():
    print(f"[lin] Using device: {DEVICE}")

    # 1) Load AE
    ae = load_scene_model(CKPT_SCENEMODEL)

    # 2) Generate dataset and encode
    print(f"[lin] Generating dataset: N={N_SAMPLES}")
    imgs, rels = generate_dataset(N_SAMPLES)

    x = torch.from_numpy(imgs).float().to(DEVICE)
    with torch.no_grad():
        z_flat = ae.encode(x)       # [N, D]

    rels = np.asarray(rels)
    z_flat_cpu = z_flat.cpu()

    # split by relation
    left_mask  = rels == 0
    right_mask = rels == 1
    above_mask = rels == 2
    below_mask = rels == 3

    z_left  = z_flat_cpu[left_mask]
    z_right = z_flat_cpu[right_mask]
    z_above = z_flat_cpu[above_mask]
    z_below = z_flat_cpu[below_mask]

    # make sure sizes match (truncate to min length for pairing)
    n_lr = min(len(z_left), len(z_right))
    n_ab = min(len(z_above), len(z_below))

    z_left  = z_left[:n_lr]
    z_right = z_right[:n_lr]
    z_above = z_above[:n_ab]
    z_below = z_below[:n_ab]

    print(f"[lin] relation 0: left_of  count={len(z_left)}")
    print(f"[lin] relation 1: right_of count={len(z_right)}")
    print(f"[lin] relation 2: above    count={len(z_above)}")
    print(f"[lin] relation 3: below    count={len(z_below)}")

    # 3) Fit operators
    A_LR = fit_operator(z_left,  z_right, name="A_LR (left→right)")
    A_AB = fit_operator(z_above, z_below, name="A_AB (above→below)")

    # 4) Linearity tests
    errs_lr, bnorm_lr = test_linearity(A_LR, z_left,  name="A_LR")
    errs_ab, bnorm_ab = test_linearity(A_AB, z_above, name="A_AB")

    # 5) Visualize a handful of worst cases
    visualize_worst_cases(ae, A_LR, z_left,  errs_lr, name="A_LR", n_examples=6)
    visualize_worst_cases(ae, A_AB, z_above, errs_ab, name="A_AB", n_examples=6)


if __name__ == "__main__":
    main()
