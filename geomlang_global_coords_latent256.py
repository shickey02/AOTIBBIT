#!/usr/bin/env python3
# geomlang_global_coords_latent256.py
#
# Compute global generator coordinates (t_x, t_y) for every sample
# using the 256-latent edges+relscale model.
#
# We:
#   - generate N_SAMPLES synthetic 64x64 scenes
#   - encode to z in R^256
#   - estimate LR/AB generators g_x, g_y from relation-wise means
#   - for each z, solve the least-squares projection
#         z - mu0 ≈ t_x g_x + t_y g_y
#   - store t_x, t_y, residual norm ||r_perp||
#   - visualize (t_x, t_y) scatter and residual histogram

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
IMG_SIZE   = 64          # matches training architecture
NUM_CH     = 3           # red, blue, edges
LATENT_DIM = 256

N_SAMPLES  = 6000
BATCH_SIZE = 128

OUT_DIR         = "outputs_edges_relscale256"
CKPT_SCENEMODEL = os.path.join(OUT_DIR, "scene_model_edges_relscale256.pt")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[coords256] Using device: {DEVICE}")

REL_LEFT, REL_RIGHT, REL_ABOVE, REL_BELOW, REL_OVERLAP = range(5)
REL_NAMES = ["left_of", "right_of", "above", "below", "overlap"]

# -----------------------
# Scene generation (64x64) – identical to lie_group_latent256
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
    """Static 64x64 scenes with red/blue fill + edge channel."""

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
# Model – same as lie_group_latent256
# -----------------------

class Encoder(nn.Module):
    # 64 -> 32 -> 16 -> 8 -> 4
    def __init__(self, in_channels=NUM_CH, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32,  kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32,          64,  kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64,          128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128,         256, kernel_size=4, stride=2, padding=1)  # 4x4
        self.fc    = nn.Linear(256 * 4 * 4, latent_dim)  # 4096 -> 256

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
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)  # 256*4*4 = 4096
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 8
        self.deconv2 = nn.ConvTranspose2d(128, 64,  kernel_size=4, stride=2, padding=1)  # 16
        self.deconv3 = nn.ConvTranspose2d(64,  32,  kernel_size=4, stride=2, padding=1)  # 32
        self.deconv4 = nn.ConvTranspose2d(32,  out_channels, kernel_size=4, stride=2, padding=1)  # 64

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
        # classification heads (unused here, but keep for checkpoint compat)
        self.rel_head      = nn.Linear(LATENT_DIM, 5)
        self.scale_head    = nn.Linear(LATENT_DIM, 3)
        self.shape_r_head  = nn.Linear(LATENT_DIM, 2)
        self.shape_b_head  = nn.Linear(LATENT_DIM, 2)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


def load_scene_model():
    print(f"[coords256] Loading SceneModel from {CKPT_SCENEMODEL}")
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
    model = load_scene_model()

    # 1) Generate dataset and encode to latent
    print(f"[coords256] Generating dataset: N={N_SAMPLES}")
    ds = GeomEdges64Dataset(N_SAMPLES)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_rel = []
    all_z   = []

    with torch.no_grad():
        for imgs, rel, scale, s_r, s_b in dl:
            imgs = imgs.to(DEVICE)
            z    = model.encode(imgs)
            all_rel.append(rel.cpu())
            all_z.append(z.cpu())

    all_rel = torch.cat(all_rel, dim=0)        # (N,)
    all_z   = torch.cat(all_z,   dim=0)        # (N, D)
    N, D    = all_z.shape
    print(f"[coords256] Latent shape: {all_z.shape}")

    rel_np = all_rel.numpy()
    for r in range(5):
        cnt = int((rel_np == r).sum())
        print(f"[coords256] relation {r}: {REL_NAMES[r]:>7}, count={cnt}")

    # 2) Relation-wise means and generators
    z = all_z.to(DEVICE)

    mu_list = []
    for r in range(5):
        mask = (all_rel == r)
        mu_r = z[mask.to(DEVICE)].mean(dim=0)
        mu_list.append(mu_r)
    mu = torch.stack(mu_list, dim=0)  # (5, D)

    mu_left   = mu[REL_LEFT]
    mu_right  = mu[REL_RIGHT]
    mu_above  = mu[REL_ABOVE]
    mu_below  = mu[REL_BELOW]
    mu_overlap = mu[REL_OVERLAP]

    v_LR = mu_right - mu_left
    v_AB = mu_below - mu_above

    print(f"[coords256] ||mu_right - mu_left|| = {v_LR.norm().item():.4f}")
    print(f"[coords256] ||mu_below - mu_above|| = {v_AB.norm().item():.4f}")

    g_x = v_LR / v_LR.norm()
    g_y = v_AB / v_AB.norm()

    cos_xy = torch.dot(g_x, g_y).item()
    angle_xy = math.degrees(math.acos(max(min(cos_xy, 1.0), -1.0)))
    print(f"[coords256] angle(g_x, g_y) = {angle_xy:.2f}° (cos = {cos_xy:.4f})")

    # 3) Global coordinates via least-squares projection onto span{g_x,g_y}
    #    We want t = (t_x, t_y) minimizing || z - mu0 - t_x g_x - t_y g_y ||.
    #    Closed form because g_x,g_y are two vectors in R^D:
    #
    #    Let s = <g_x, g_y>. Then the Gram matrix is:
    #        G = [[1, s],
    #             [s, 1]]
    #    G^{-1} = 1/(1-s^2) * [[1, -s], [-s, 1]]
    #
    #    Define a = <z_c, g_x>, b = <z_c, g_y>, where z_c = z - mu0.
    #    Then:
    #        t_x = (a - s b) / (1 - s^2)
    #        t_y = (b - s a) / (1 - s^2)

    mu0 = mu_overlap.to(DEVICE)  # center at overlap-mean
    z_c = z - mu0.unsqueeze(0)   # (N, D)

    a = torch.matmul(z_c, g_x)   # (N,)
    b = torch.matmul(z_c, g_y)   # (N,)

    s = cos_xy
    denom = 1.0 - s * s
    if denom <= 0:
        raise RuntimeError(f"Gram matrix is singular: 1 - s^2 = {denom}")

    t_x = (a - s * b) / denom
    t_y = (b - s * a) / denom

    # reconstruct projection and residual
    proj = t_x.unsqueeze(1) * g_x.unsqueeze(0) + t_y.unsqueeze(1) * g_y.unsqueeze(0)  # (N,D)
    r_perp = z_c - proj
    r_norm = r_perp.norm(dim=1)              # (N,)
    zc_norm = z_c.norm(dim=1)

    print("[coords256] Residual norms ||r_perp|| stats:")
    rn = r_norm.cpu().numpy()
    znn = zc_norm.cpu().numpy()
    print(f"   mean = {rn.mean():.4f}, std = {rn.std():.4f}, "
          f"min = {rn.min():.4f}, max = {rn.max():.4f}")
    ratio = rn / (znn + 1e-8)
    print(f"[coords256] Relative residual ||r_perp|| / ||z-mu0|| stats:")
    print(f"   mean = {ratio.mean():.4f}, std = {ratio.std():.4f}, "
          f"min = {ratio.min():.4f}, max = {ratio.max():.4f}")

    # 4) Save raw data
    out_npz = os.path.join(OUT_DIR, "global_coords_latent256.npz")
    np.savez(
        out_npz,
        t_x=t_x.cpu().numpy(),
        t_y=t_y.cpu().numpy(),
        r_norm=r_norm.cpu().numpy(),
        rel=all_rel.numpy(),
    )
    print(f"[coords256] Saved coordinates -> {out_npz}")

    # 5) Visualization: (t_x, t_y) scatter colored by relation
    colors = {
        0: "tab:blue",    # left_of
        1: "tab:orange",  # right_of
        2: "tab:green",   # above
        3: "tab:red",     # below
        4: "tab:purple",  # overlap
    }

    tx_np = t_x.cpu().numpy()
    ty_np = -t_y.cpu().numpy()

    plt.figure(figsize=(8, 8), dpi=120)
    for r in range(5):
        mask = (rel_np == r)
        plt.scatter(tx_np[mask], ty_np[mask],
                    s=8, alpha=0.5, label=REL_NAMES[r], c=colors[r])
    plt.axhline(0, color="k", linewidth=0.5, alpha=0.5)
    plt.axvline(0, color="k", linewidth=0.5, alpha=0.5)
    plt.xlabel("t_x (LR generator coord)")
    plt.ylabel("t_y (AB generator coord)")
    plt.title("Global generator coordinates (latent256)")
    plt.legend(markerscale=2)
    plt.tight_layout()
    scatter_path = os.path.join(OUT_DIR, "global_coords_scatter_latent256.png")
    plt.savefig(scatter_path)
    plt.close()
    print(f"[coords256] Saved scatter -> {scatter_path}")

    # 6) Histogram of residual norms
    plt.figure(figsize=(6, 4), dpi=120)
    plt.hist(rn, bins=40)
    plt.xlabel("||r_perp||")
    plt.ylabel("count")
    plt.title("Residual norm after projection onto span{g_x, g_y} (latent256)")
    plt.tight_layout()
    hist_path = os.path.join(OUT_DIR, "global_coords_residual_hist_latent256.png")
    plt.savefig(hist_path)
    plt.close()
    print(f"[coords256] Saved residual histogram -> {hist_path}")

    print("[coords256] Done.")


if __name__ == "__main__":
    main()
