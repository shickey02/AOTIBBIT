#!/usr/bin/env python3
# geomlang_curvature_basis_latent256.py
#
# Step 1: Explicit curvature basis for the 256-d edges+relscale model.
#
# 1) Recompute residual PCA (orthogonal to the {g_x, g_y} relation plane).
# 2) Choose a small set of residual PCs as curvature basis {h_i}.
# 3) Project commutators [F_y, F_x] onto {h_i} and report energy fractions.
# 4) Decode +/- steps along each h_i from a seed code for qualitative inspection.

import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -----------------------
# Config
# -----------------------
IMG_SIZE   = 64
NUM_CH     = 3
LATENT_DIM = 256

N_SAMPLES  = 6000
BATCH_SIZE = 128

OUT_DIR         = "outputs_edges_relscale256"
CKPT_SCENEMODEL = os.path.join(OUT_DIR, "scene_model_edges_relscale256.pt")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REL_LEFT, REL_RIGHT, REL_ABOVE, REL_BELOW, REL_OVERLAP = range(5)
REL_NAMES = ["left_of", "right_of", "above", "below", "overlap"]

# which residual PCs to use as curvature basis
# indices are 0-based: 0->PC1, 1->PC2, 2->PC3, 5->PC6
CURV_PC_IDX = [0, 1, 2, 5]

# commutator finite-diff eps
EPS = 0.4
N_COMM_SEEDS = 400

# -----------------------
# Scene generation (same as other 256 scripts)
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
# Model
# -----------------------

class Encoder(nn.Module):
    # 64 -> 32 -> 16 -> 8 -> 4
    def __init__(self, in_channels=NUM_CH, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32,  kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32,          64,  kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64,          128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128,         256, kernel_size=4, stride=2, padding=1)
        self.fc    = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

class Decoder(nn.Module):
    def __init__(self, out_channels=NUM_CH, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc      = nn.Linear(latent_dim, 256 * 4 * 4)
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
        self.rel_head      = nn.Linear(LATENT_DIM, 5)
        self.scale_head    = nn.Linear(LATENT_DIM, 3)
        self.shape_r_head  = nn.Linear(LATENT_DIM, 2)
        self.shape_b_head  = nn.Linear(LATENT_DIM, 2)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

def load_scene_model():
    print(f"[curv256] Loading SceneModel from {CKPT_SCENEMODEL}")
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
# Helpers
# -----------------------

def orthonormalize(vs):
    """QR-orthonormalize columns of vs (D x k)."""
    Q, _ = np.linalg.qr(vs)
    return Q

def decode_to_rgb(img_tensor):
    """
    img_tensor: (3, H, W) torch tensor in [0,1]
    Returns (H,W,3) numpy for plotting.
    """
    img = img_tensor.detach().cpu().numpy()
    red, green, edge = img[0], img[1], img[2]
    H, W = red.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    rgb[..., 0] = red + 0.6 * edge   # red + edges
    rgb[..., 1] = green + 0.6 * edge # green + edges
    # slight blue tint for edges, but small so background stays dark
    rgb[..., 2] = 0.6 * edge
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb

# -----------------------
# Main
# -----------------------

def main():
    print(f"[curv256] Using device: {DEVICE}")
    model = load_scene_model()

    # --- 1) Generate dataset and latents ---
    print(f"[curv256] Generating dataset: N={N_SAMPLES}")
    ds = GeomEdges64Dataset(N_SAMPLES)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_z   = []
    all_rel = []
    all_imgs = []

    with torch.no_grad():
        for imgs, rel, scale, s_r, s_b in dl:
            imgs = imgs.to(DEVICE)
            z = model.encode(imgs)
            all_z.append(z.cpu())
            all_rel.append(rel.cpu())
            all_imgs.append(imgs.cpu())

    all_z   = torch.cat(all_z, dim=0).numpy()      # (N,D)
    all_rel = torch.cat(all_rel, dim=0).numpy()    # (N,)
    all_imgs = torch.cat(all_imgs, dim=0)          # (N,3,64,64)

    N, D = all_z.shape
    print(f"[curv256] Latent shape: {all_z.shape}")

    # --- relation means and generator plane ---
    idx_left  = np.where(all_rel == REL_LEFT)[0]
    idx_right = np.where(all_rel == REL_RIGHT)[0]
    idx_above = np.where(all_rel == REL_ABOVE)[0]
    idx_below = np.where(all_rel == REL_BELOW)[0]
    idx_overlap = np.where(all_rel == REL_OVERLAP)[0]

    mu_left   = all_z[idx_left].mean(axis=0)
    mu_right  = all_z[idx_right].mean(axis=0)
    mu_above  = all_z[idx_above].mean(axis=0)
    mu_below  = all_z[idx_below].mean(axis=0)

    g_x = mu_right - mu_left
    g_y = mu_below - mu_above

    gx_norm = np.linalg.norm(g_x)
    gy_norm = np.linalg.norm(g_y)
    print(f"[curv256] ||mu_right - mu_left|| = {gx_norm:.4f}")
    print(f"[curv256] ||mu_below - mu_above|| = {gy_norm:.4f}")

    cos_angle = np.dot(g_x, g_y) / (gx_norm * gy_norm)
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    angle_deg = math.degrees(math.acos(cos_angle))
    print(f"[curv256] angle(g_x, g_y) = {angle_deg:.2f}°")

    # Orthonormal basis for relation plane
    g_x_u = g_x / gx_norm
    g_y_ortho = g_y - np.dot(g_y, g_x_u) * g_x_u
    g_y_u = g_y_ortho / np.linalg.norm(g_y_ortho)
    U_plane = np.stack([g_x_u, g_y_u], axis=1)  # D x 2

    mu0 = 0.25 * (mu_left + mu_right + mu_above + mu_below)

    # --- 2) Residual projection + PCA ---
    z_centered = all_z - mu0[None, :]
    proj_plane = z_centered @ U_plane @ U_plane.T  # N x D
    r_perp = z_centered - proj_plane               # N x D

    r_norm = np.linalg.norm(r_perp, axis=1)
    z_norm = np.linalg.norm(z_centered, axis=1)
    rel_residual = r_norm / z_norm

    print(f"[curv256] Residual norms ||r_perp|| stats:")
    print(f"   mean = {r_norm.mean():.4f}, std = {r_norm.std():.4f}, "
          f"min = {r_norm.min():.4f}, max = {r_norm.max():.4f}")
    print(f"[curv256] Relative residual ||r_perp||/||z-mu0|| stats:")
    print(f"   mean = {rel_residual.mean():.4f}, std = {rel_residual.std():.4f}, "
          f"min = {rel_residual.min():.4f}, max = {rel_residual.max():.4f}")

    print("[curv256] PCA on residual subspace...")
    max_components = min(32, D, N)
    pca = PCA(n_components=max_components, svd_solver="full", random_state=0)
    pca.fit(r_perp)
    evr = pca.explained_variance_ratio_

    for i, r in enumerate(evr, 1):
        print(f"  PC{i:02d}: {100*r:5.2f}% variance")
    print(f"  cumulative PC1..PC{len(evr)}: {100*evr.cumsum()[-1]:.2f}%")

    eff_rank = int((evr > 0.01).sum())
    print(f"  effective residual rank (>=1% each): {eff_rank}")

    # curvature basis from chosen PCs
    print(f"[curv256] Using residual PCs {CURV_PC_IDX} as curvature basis.")
    H_raw = pca.components_[CURV_PC_IDX]   # shape (k, D)
    H = orthonormalize(H_raw.T)            # D x k  (QR re-ortho, just to be safe)
    k = H.shape[1]
    print(f"[curv256] curvature basis dim = {k}")

    # -------------------------------
    # 3) Commutators and projections
    # -------------------------------
    rng = np.random.default_rng(0)
    seed_idx = rng.choice(N, size=N_COMM_SEEDS, replace=False)

    gx_vec = g_x_u * gx_norm     # original scale is fine
    gy_vec = g_y_u * gy_norm

    c_list      = []
    c_perp_list = []
    coeff_list  = []

    model.eval()
    with torch.no_grad():
        for idx in seed_idx:
            z0 = all_z[idx]

            def apply_flow(z, dir_vec):
                z1 = z + EPS * dir_vec
                z2 = z - EPS * dir_vec
                z1_t = torch.from_numpy(z1[None, :]).to(DEVICE).float()
                z2_t = torch.from_numpy(z2[None, :]).to(DEVICE).float()
                x1 = model.decode(z1_t)
                x2 = model.decode(z2_t)
                z1_back = model.encode(x1).cpu().numpy()[0]
                z2_back = model.encode(x2).cpu().numpy()[0]
                return (z1_back - z2_back) / (2.0 * EPS)

            v_x = apply_flow(z0, gx_vec)
            v_y = apply_flow(z0, gy_vec)

            # flows in sequence
            z_xy = z0 + EPS * gx_vec
            v_y_at_xy = apply_flow(z_xy, gy_vec)

            z_yx = z0 + EPS * gy_vec
            v_x_at_yx = apply_flow(z_yx, gx_vec)

            c = (v_y_at_xy - v_x_at_yx) / EPS   # approx [F_y, F_x]
            c_list.append(c)

            # project c onto residual subspace (remove plane part)
            c_plane = (c @ U_plane) @ U_plane.T
            c_perp = c - c_plane
            c_perp_list.append(c_perp)

            # coefficients in curvature basis H
            alphas = H.T @ c_perp   # (k,)
            coeff_list.append(alphas)

    c_arr      = np.stack(c_list, axis=0)
    c_perp_arr = np.stack(c_perp_list, axis=0)
    coeff_arr  = np.stack(coeff_list, axis=0)  # (M, k)
    M = c_arr.shape[0]

    c_norm = np.linalg.norm(c_arr, axis=1)
    c_perp_norm = np.linalg.norm(c_perp_arr, axis=1)
    frac_perp = c_perp_norm / c_norm

    print("\n[curv256] Global commutator stats:")
    print(f"  ||c|| mean={c_norm.mean():.4f}, std={c_norm.std():.4f}, "
          f"min={c_norm.min():.4f}, max={c_norm.max():.4f}")
    print(f"  ||c_perp|| mean={c_perp_norm.mean():.4f}, std={c_perp_norm.std():.4f}")
    print(f"  frac_perp mean={frac_perp.mean():.4f}, std={frac_perp.std():.4f}, "
          f"min={frac_perp.min():.4f}, max={frac_perp.max():.4f}")

    # energy fractions for curvature basis
    energy_total = (c_perp_norm ** 2).sum()
    energy_hi = np.zeros(k, dtype=np.float64)

    per_seed_frac = np.zeros((M, k), dtype=np.float64)

    for i in range(M):
        alphas = coeff_arr[i]  # (k,)
        e_i = alphas ** 2
        e_all = e_i.sum()
        energy_hi += e_i
        if c_perp_norm[i] > 0:
            per_seed_frac[i] = e_i / (c_perp_norm[i] ** 2)

    print("\n[curv256] Global fraction of commutator residual energy in curvature basis:")
    for j in range(k):
        frac = energy_hi[j] / energy_total
        print(f"  h{j+1}: {100*frac:5.2f}%")

    frac_basis_total = energy_hi.sum() / energy_total
    print(f"  Total in span{{h_i}}: {100*frac_basis_total:5.2f}%")

    print("\n[curv256] Per-seed mean fraction ||P_{h_j} c_perp||^2 / ||c_perp||^2:")
    for j in range(k):
        vals = per_seed_frac[:, j]
        print(f"  h{j+1}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
              f"min={vals.min():.4f}, max={vals.max():.4f}")

    # simple barplot of global fractions
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=120)
    x = np.arange(1, k+1)
    ax.bar(x, energy_hi / energy_total)
    ax.set_xlabel("basis index h_i")
    ax.set_ylabel("fraction of total commutator residual energy")
    ax.set_title("Comm. residual energy in curvature basis (latent256)")
    ax.set_xticks(x)
    plt.tight_layout()
    out_bar = os.path.join(OUT_DIR, "curvature_basis_energy_latent256.png")
    plt.savefig(out_bar)
    print(f"[curv256] Saved curvature basis energy plot -> {out_bar}")

    # --------------------------------
    # 4) Decode +/- steps along h_i
    # --------------------------------
    print("[curv256] Making qualitative grid for +/- steps along h_i...")

    # pick one overlap seed to visualize (it tends to be symmetric-ish)
    if len(idx_overlap) == 0:
        seed_vis_idx = 0
        print("[curv256] WARNING: no overlap relation found; using idx=0 for visualization.")
    else:
        seed_vis_idx = int(idx_overlap[0])

    z_seed = all_z[seed_vis_idx]
    img_seed = all_imgs[seed_vis_idx]

    steps = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    n_steps = len(steps)

    fig, axes = plt.subplots(k, n_steps, figsize=(1.6*n_steps, 1.6*k), dpi=160)

    if k == 1:
        axes = np.expand_dims(axes, axis=0)  # ensure 2D

    with torch.no_grad():
        for row in range(k):
            h = H[:, row]
            for col, t in enumerate(steps):
                z_mod = z_seed + t * h
                z_t = torch.from_numpy(z_mod[None, :]).to(DEVICE).float()
                x_dec = model.decode(z_t)[0]
                rgb = decode_to_rgb(x_dec)
                ax = axes[row, col]
                ax.imshow(rgb, interpolation="nearest")
                ax.axis("off")
                ax.set_title(f"h{row+1}, t={t:g}", fontsize=7)

    plt.suptitle("Curvature basis directions h_i (latent256): rows=i, cols=step t", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_grid = os.path.join(OUT_DIR, "curvature_basis_grid_latent256.png")
    plt.savefig(out_grid)
    print(f"[curv256] Saved curvature basis decode grid -> {out_grid}")

    print("\n[curv256] Done.")

if __name__ == "__main__":
    main()
