#!/usr/bin/env python3
# geomlang_noncomm_loops_latent256.py
#
# Visualize non-commutativity of the learned LR/AB generators via
# explicit rectangular loops and XY vs YX paths in latent space,
# decoded back to images.
#
# Outputs:
#   outputs_edges_relscale256/noncomm_loops_seed_<name>_paths_latent256.png
#   outputs_edges_relscale256/noncomm_loops_seed_<name>_loop_latent256.png

import os
import math
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
TAG          = "[loop256]"
IMG_SIZE     = 64
NUM_CH       = 3          # red, blue, edges
LATENT_DIM   = 256
N_SAMPLES    = 6000
BATCH_SIZE   = 128
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# step sizes in generator coordinates
EPS_X        = 0.6
EPS_Y        = 0.6

OUT_DIR         = "outputs_edges_relscale256"
CKPT_SCENEMODEL = os.path.join(OUT_DIR, "scene_model_edges_relscale256.pt")

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Relation constants
# ---------------------------------------------------------------------
REL_LEFT, REL_RIGHT, REL_ABOVE, REL_BELOW, REL_OVERLAP = range(5)
REL_NAMES = ["left_of", "right_of", "above", "below", "overlap"]

# ---------------------------------------------------------------------
# Scene generation (same as other 64x64 edges+relscale scripts)
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# Model – same architecture as other latent256 edges+relscale scripts
# ---------------------------------------------------------------------
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
        # classification heads (unused, kept for checkpoint compatibility)
        self.rel_head      = nn.Linear(LATENT_DIM, 5)
        self.scale_head    = nn.Linear(LATENT_DIM, 3)
        self.shape_r_head  = nn.Linear(LATENT_DIM, 2)
        self.shape_b_head  = nn.Linear(LATENT_DIM, 2)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


def load_scene_model():
    print(f"{TAG} Loading SceneModel from {CKPT_SCENEMODEL}")
    ckpt = torch.load(CKPT_SCENEMODEL, map_location=DEVICE)
    model = SceneModelEdges256()
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.to(DEVICE)
    model.eval()
    return model

# ---------------------------------------------------------------------
# Utility: load model
# ---------------------------------------------------------------------
def load_model():
    print(f"{TAG} Using device: {DEVICE}")
    model = load_scene_model()
    return model

# ---------------------------------------------------------------------
# Encode a synthetic dataset once so we have latents + relations
# ---------------------------------------------------------------------
def encode_dataset(model, n_samples=N_SAMPLES):
    print(f"{TAG} Generating dataset: N={n_samples}")
    ds = GeomEdges64Dataset(n_samples)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_z   = []
    all_rel = []

    with torch.no_grad():
        for imgs, rel, scale, s_r, s_b in dl:
            imgs = imgs.to(DEVICE)
            z = model.encode(imgs).cpu()
            all_z.append(z)
            all_rel.append(rel.cpu())

    z_all   = torch.cat(all_z,   dim=0)  # [N, D]
    rel_all = torch.cat(all_rel, dim=0)  # [N]
    print(f"{TAG} Latent shape: {tuple(z_all.shape)}")
    return z_all, rel_all

# ---------------------------------------------------------------------
# Compute global generators g_x, g_y from relation means
# ---------------------------------------------------------------------
def compute_global_generators(z, rel_ids):
    mus = []
    for rid in range(5):
        mask = (rel_ids == rid)
        rel_z = z[mask]
        mu_r  = rel_z.mean(0)
        print(
            f"{TAG} relation {rid}={REL_NAMES[rid]:>7}, "
            f"count={int(mask.sum())}, ||mu_r||={mu_r.norm():.3f}"
        )
        mus.append(mu_r)

    mu_left, mu_right, mu_above, mu_below, mu_overlap = mus

    g_x_raw = mu_right - mu_left
    g_y_raw = mu_below - mu_above

    g_x = g_x_raw / g_x_raw.norm()
    g_y = g_y_raw / g_y_raw.norm()

    angle = math.degrees(
        math.acos(torch.clamp(torch.dot(g_x, g_y), -1.0, 1.0))
    )
    print(f"{TAG} ||mu_right - mu_left|| = {g_x_raw.norm():.4f}")
    print(f"{TAG} ||mu_below - mu_above|| = {g_y_raw.norm():.4f}")
    print(f"{TAG} angle(g_x, g_y) = {angle:.2f}°")

    return g_x.to(DEVICE), g_y.to(DEVICE), mus

# ---------------------------------------------------------------------
# Flow operators: integrate along g_x / g_y with decode+encode step
# ---------------------------------------------------------------------
def flow_step(model, z, direction_vec, eps):
    """
    One small step along direction_vec with projection back to manifold:
        z' = E(D(z + eps * v))
    z: [latent_dim] on DEVICE
    direction_vec: [latent_dim] on DEVICE
    """
    z_prop = z + eps * direction_vec
    with torch.no_grad():
        x = model.decode(z_prop.unsqueeze(0))  # [1, C, H, W]
        z_new = model.encode(x).squeeze(0)
    return z_new


def path_xy(model, z0, g_x, g_y, eps_x, eps_y):
    """z0 -> +x -> +y"""
    z0 = z0.to(DEVICE)
    zs = [z0]
    z1 = flow_step(model, zs[-1], g_x, +eps_x)
    zs.append(z1)
    z2 = flow_step(model, zs[-1], g_y, +eps_y)
    zs.append(z2)
    return zs  # [z0, after x, after xy]


def path_yx(model, z0, g_x, g_y, eps_x, eps_y):
    """z0 -> +y -> +x"""
    z0 = z0.to(DEVICE)
    zs = [z0]
    z1 = flow_step(model, zs[-1], g_y, +eps_y)
    zs.append(z1)
    z2 = flow_step(model, zs[-1], g_x, +eps_x)
    zs.append(z2)
    return zs


def loop_rectangle(model, z0, g_x, g_y, eps_x, eps_y):
    """
    Commutator-style loop:
        z0 --(+x)--> z1 --(+y)--> z2 --(-x)--> z3 --(-y)--> z4
    Returns list [z0, z1, z2, z3, z4]
    """
    z0 = z0.to(DEVICE)
    zs = [z0]
    z1 = flow_step(model, zs[-1], g_x, +eps_x)
    zs.append(z1)
    z2 = flow_step(model, zs[-1], g_y, +eps_y)
    zs.append(z2)
    z3 = flow_step(model, zs[-1], g_x, -eps_x)
    zs.append(z3)
    z4 = flow_step(model, zs[-1], g_y, -eps_y)
    zs.append(z4)
    return zs

# ---------------------------------------------------------------------
# Decode a list of latents into numpy images
# ---------------------------------------------------------------------
def decode_strip(model, zs):
    zs = torch.stack(zs, dim=0)  # [K, D]
    with torch.no_grad():
        x = model.decode(zs.to(DEVICE))  # [K, C, H, W]
    x = x.cpu().numpy()
    if x.shape[1] == 1:
        imgs = x[:, 0, :, :]           # grayscale
        cmap = "gray"
    else:
        imgs = np.transpose(x, (0, 2, 3, 1))  # [K,H,W,C]
        cmap = None
    return imgs, cmap

# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------
def plot_paths(seed_name, imgs_xy, imgs_yx, cmap_xy,
               imgs_loop, cmap_loop, out_prefix):
    """
    imgs_xy: [3, H, W] or [3, H, W, C]
    imgs_yx: same
    imgs_loop: [5, ...]
    """

    # --- 2-row XY vs YX strip ---
    K = 3

    fig, axes = plt.subplots(2, K, figsize=(2.2 * K, 4.4))
    fig.suptitle(f"Non-commutative paths around seed '{seed_name}'")

    for r, (imgs, label) in enumerate(((imgs_xy, "XY"), (imgs_yx, "YX"))):
        for c in range(K):
            ax = axes[r, c]
            ax.axis("off")
            if imgs.ndim == 3:   # [K,H,W]
                ax.imshow(imgs[c], cmap=cmap_xy, vmin=0, vmax=1)
            else:                # [K,H,W,C]
                ax.imshow(np.clip(imgs[c], 0, 1))
            ax.set_title(f"{label} step {c}")

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f"{out_prefix}_paths_latent256.png")
    print(f"{TAG} Saving {out_path}")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    # --- loop strip (single row of 5) ---
    K = 5
    fig, axes = plt.subplots(1, K, figsize=(2.2 * K, 2.4))
    fig.suptitle(f"Rectangle loop around seed '{seed_name}'")

    for c in range(K):
        ax = axes[c]
        ax.axis("off")
        if imgs_loop.ndim == 3:
            ax.imshow(imgs_loop[c], cmap=cmap_loop, vmin=0, vmax=1)
        else:
            ax.imshow(np.clip(imgs_loop[c], 0, 1))
        if c == 0:
            ax.set_title("start")
        elif c == 4:
            ax.set_title("end")

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f"{out_prefix}_loop_latent256.png")
    print(f"{TAG} Saving {out_path}")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    model = load_model()

    # Encode dataset and compute generators
    z_all, rel_ids = encode_dataset(model, N_SAMPLES)
    g_x, g_y, mus   = compute_global_generators(z_all, rel_ids)

    # Choose a few seed indices:
    # - a typical left_of
    # - a typical above
    # - an overlap-ish one near the center
    seeds = {}

    # left_of seed (closest to mu_left)
    mu_left    = mus[REL_LEFT]
    left_mask  = (rel_ids == REL_LEFT)
    left_z     = z_all[left_mask]
    left_d     = torch.norm(left_z - mu_left, dim=1)
    left_loc   = int(torch.argmin(left_d))
    left_inds  = torch.nonzero(left_mask, as_tuple=False).squeeze(1)
    left_idx_global = int(left_inds[left_loc].item())
    seeds["left_of"] = left_idx_global

    # above seed
    mu_above   = mus[REL_ABOVE]
    above_mask = (rel_ids == REL_ABOVE)
    above_z    = z_all[above_mask]
    above_d    = torch.norm(above_z - mu_above, dim=1)
    above_loc  = int(torch.argmin(above_d))
    above_inds = torch.nonzero(above_mask, as_tuple=False).squeeze(1)
    above_idx_global = int(above_inds[above_loc].item())
    seeds["above"] = above_idx_global

    # overlap seed (near mu_overlap)
    mu_overlap  = mus[REL_OVERLAP]
    overlap_mask = (rel_ids == REL_OVERLAP)
    overlap_z    = z_all[overlap_mask]
    overlap_d    = torch.norm(overlap_z - mu_overlap, dim=1)
    overlap_loc  = int(torch.argmin(overlap_d))
    overlap_inds = torch.nonzero(overlap_mask, as_tuple=False).squeeze(1)
    overlap_idx_global = int(overlap_inds[overlap_loc].item())
    seeds["overlap"] = overlap_idx_global

    print(f"{TAG} Chosen seeds (global indices): {seeds}")

    # For each seed, build XY, YX and loop paths, decode, and plot
    for name, idx in seeds.items():
        rel_name = REL_NAMES[int(rel_ids[idx].item())]
        print(f"{TAG} Processing seed '{name}' idx={idx}, rel={rel_name}")
        z0 = z_all[idx].to(DEVICE)

        # XY and YX paths
        zs_xy = path_xy(model, z0, g_x, g_y, EPS_X, EPS_Y)
        zs_yx = path_yx(model, z0, g_x, g_y, EPS_X, EPS_Y)
        # rectangle loop
        zs_loop = loop_rectangle(model, z0, g_x, g_y, EPS_X, EPS_Y)

        imgs_xy, cmap_xy       = decode_strip(model, zs_xy)
        imgs_yx, _             = decode_strip(model, zs_yx)
        imgs_loop, cmap_loop   = decode_strip(model, zs_loop)

        # ensure np arrays of consistent shape
        imgs_xy   = np.asarray(imgs_xy)
        imgs_yx   = np.asarray(imgs_yx)
        imgs_loop = np.asarray(imgs_loop)

        out_prefix = f"noncomm_loops_seed_{name}"
        plot_paths(name, imgs_xy, imgs_yx, cmap_xy, imgs_loop, cmap_loop, out_prefix)

        # print latent endpoint discrepancy norms
        end_xy   = zs_xy[-1]
        end_yx   = zs_yx[-1]
        end_loop = zs_loop[-1]
        print(f"{TAG}  ||end_XY - end_YX|| = {torch.norm(end_xy - end_yx).item():.4f}")
        print(f"{TAG}  ||end_loop - z0||   = {torch.norm(end_loop - z0).item():.4f}")


if __name__ == "__main__":
    main()
