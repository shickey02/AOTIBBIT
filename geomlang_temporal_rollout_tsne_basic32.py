#!/usr/bin/env python3
# geomlang_temporal_rollout_tsne_basic32.py
#
# Time-extended t-SNE visualization of temporal rollouts on the basic32 manifold.
#
# - Loads the trained SceneModel (AE) from outputs_basic32/scene_model_basic32.pt
# - Loads the FutureGRU from outputs_basic32/future_gru_rollout_basic32.pt
#   (trains a new one if it doesn't exist)
# - Generates temporal clips with moving red/blue blobs
# - Encodes all frames -> latents
# - Builds a t-SNE manifold from:
#       * a subsample of static latents
#       * the GRU rollout trajectories (latent z at t4..t8)
# - Plots:
#       * static latents as a light background cloud
#       * each GRU rollout as a polyline from t4 → t8
#
# Run:
#   python bbit_geomlang/geomlang_temporal_rollout_tsne_basic32.py

import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# --------------------
# Config
# --------------------
IMG_SIZE      = 32
NUM_CHANNELS  = 3
LATENT_DIM    = 48

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CKPT_SCENEMODEL = os.path.join("outputs_basic32", "scene_model_basic32.pt")
CKPT_FUTURE_GRU = os.path.join("outputs_basic32", "future_gru_rollout_basic32.pt")
OUT_DIR         = "outputs_basic32"
os.makedirs(OUT_DIR, exist_ok=True)

# temporal setup must match the rollout script
CLIP_LEN   = 8   # total frames per clip
HIST_LEN   = 4   # context length (we'll start rollouts at t4)
FUTURE_LEN = 4   # t5..t8

N_TRAIN_CLIPS = 2000
N_TEST_CLIPS  = 500

BATCH_SIZE = 128
EPOCHS_GRU = 20
LR_GRU     = 1e-3

# t-SNE config
MAX_STATIC_TSNE_POINTS = 4000   # subsample static latents
N_ROLLOUTS_TSNE        = 80     # how many rollout trajectories to plot


# --------------------
# Drawing primitives (same as other basic32 scripts)
# --------------------
def draw_circle(grid, cx, cy, radius, channel_idx):
    """Draw a filled circle onto grid[channel_idx]."""
    h, w = grid.shape[1], grid.shape[2]
    for y in range(h):
        for x in range(w):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                grid[channel_idx, y, x] = 1.0


def draw_square(grid, cx, cy, half_size, channel_idx):
    """Draw a filled square onto grid[channel_idx]."""
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


# --------------------
# Temporal dataset generation
# --------------------
def generate_temporal_clip(T=CLIP_LEN):
    """
    Generate one temporal clip of length T with moving red/blue blobs.
    Returns: [T, 3, H, W] in [0,1]
    """
    H = W = IMG_SIZE
    margin = 4

    # sample shapes and sizes
    shape_red = np.random.randint(0, 2)   # 0 circle, 1 square
    shape_blue = np.random.randint(0, 2)

    base = np.random.randint(4, 7)  # 4..6
    r_red = int(np.clip(base + np.random.randint(-1, 2), 3, 7))
    r_blue = int(np.clip(base + np.random.randint(-1, 2), 3, 7))

    def sample_center(r):
        cx = np.random.randint(margin + r, W - margin - r)
        cy = np.random.randint(margin + r, H - margin - r)
        return float(cx), float(cy)

    cx_r, cy_r = sample_center(r_red)
    cx_b, cy_b = sample_center(r_blue)

    def sample_velocity():
        # avoid exactly zero velocity
        while True:
            vx = np.random.uniform(-1.5, 1.5)
            vy = np.random.uniform(-1.5, 1.5)
            if abs(vx) + abs(vy) > 0.2:
                return vx, vy

    vx_r, vy_r = sample_velocity()
    vx_b, vy_b = sample_velocity()

    frames = np.zeros((T, NUM_CHANNELS, H, W), dtype=np.float32)

    for t in range(T):
        grid = np.zeros((NUM_CHANNELS, H, W), dtype=np.float32)

        draw_shape(grid, shape_red, int(round(cx_r)), int(round(cy_r)), r_red, 0)
        draw_shape(grid, shape_blue, int(round(cx_b)), int(round(cy_b)), r_blue, 2)

        frames[t] = grid

        def step_pos(cx, cy, vx, vy, r):
            cx_new = cx + vx
            cy_new = cy + vy
            # reflect at borders
            if cx_new < margin + r:
                cx_new = margin + r + (margin + r - cx_new)
                vx *= -1
            elif cx_new > W - margin - r:
                cx_new = W - margin - r - (cx_new - (W - margin - r))
                vx *= -1
            if cy_new < margin + r:
                cy_new = margin + r + (margin + r - cy_new)
                vy *= -1
            elif cy_new > H - margin - r:
                cy_new = H - margin - r - (cy_new - (H - margin - r))
                vy *= -1
            return cx_new, cy_new, vx, vy

        cx_r, cy_r, vx_r, vy_r = step_pos(cx_r, cy_r, vx_r, vy_r, r_red)
        cx_b, cy_b, vx_b, vy_b = step_pos(cx_b, cy_b, vx_b, vy_b, r_blue)

    return frames


def generate_temporal_dataset(num_clips, T=CLIP_LEN):
    clips = np.stack([generate_temporal_clip(T) for _ in range(num_clips)], axis=0)
    return clips  # [N, T, 3, H, W]


# --------------------
# SceneModel (AE) – same architecture as basic32
# --------------------
class Encoder(nn.Module):
    def __init__(self, in_channels=NUM_CHANNELS, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, 2, 1)   # 16x16
        self.conv2 = nn.Conv2d(16, 32, 3, 2, 1)            # 8x8
        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)            # 4x4
        self.fc   = nn.Linear(64 * 4 * 4, latent_dim)

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
        # classifier heads exist just so checkpoint loading is easy
        self.rel_head        = nn.Linear(LATENT_DIM, 6)
        self.scale_head      = nn.Linear(LATENT_DIM, 3)
        self.shape_red_head  = nn.Linear(LATENT_DIM, 2)
        self.shape_blue_head = nn.Linear(LATENT_DIM, 2)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


def load_scene_model(path=CKPT_SCENEMODEL):
    print(f"[tsne] Loading SceneModel from {path}")
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


# --------------------
# Future GRU (same as rollout script)
# --------------------
class FutureGRU(nn.Module):
    def __init__(self, dim_latent, hidden=128):
        super().__init__()
        self.gru = nn.GRU(dim_latent, hidden, batch_first=True)
        self.fc  = nn.Linear(hidden, dim_latent)

    def forward(self, z_hist):
        # z_hist: [B, T_hist, D]
        out, _ = self.gru(z_hist)
        last = out[:, -1, :]        # [B, hidden]
        return self.fc(last)        # [B, D]


def build_gru_dataset(z_clips, hist_len=HIST_LEN):
    """
    z_clips: [N, T, D] (CPU tensor)
    Returns X:[M,hist_len,D], Y:[M,D] for next-step prediction.
    """
    N, T, D = z_clips.shape
    X, Y = [], []
    for i in range(N):
        for t in range(T - hist_len):
            X.append(z_clips[i, t:t+hist_len])
            Y.append(z_clips[i, t+hist_len])
    return torch.stack(X, 0), torch.stack(Y, 0)


def train_future_gru(z_train, z_test):
    X_train, Y_train = build_gru_dataset(z_train)
    X_test,  Y_test  = build_gru_dataset(z_test)

    train_loader = DataLoader(TensorDataset(X_train, Y_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test, Y_test),
                              batch_size=BATCH_SIZE, shuffle=False)

    model = FutureGRU(LATENT_DIM, 128).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR_GRU)
    loss_fn = nn.MSELoss()

    for epoch in range(1, EPOCHS_GRU + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                test_loss += loss.item() * xb.size(0)
        test_loss /= len(test_loader.dataset)

        print(f"[tsne] GRU epoch {epoch:2d}/{EPOCHS_GRU} | train MSE={train_loss:.6f} | test MSE={test_loss:.6f}")

    torch.save(model.state_dict(), CKPT_FUTURE_GRU)
    print(f"[tsne] Saved FutureGRU -> {CKPT_FUTURE_GRU}")
    return model


def load_or_train_future_gru(z_train, z_test):
    if os.path.exists(CKPT_FUTURE_GRU):
        print(f"[tsne] Loading FutureGRU from {CKPT_FUTURE_GRU}")
        model = FutureGRU(LATENT_DIM, 128)
        state = torch.load(CKPT_FUTURE_GRU, map_location=DEVICE)
        model.load_state_dict(state, strict=False)
        model.to(DEVICE)
        model.eval()
        return model
    else:
        print("[tsne] No FutureGRU checkpoint found; training a new one...")
        return train_future_gru(z_train, z_test)


# --------------------
# t-SNE rollout construction
# --------------------
def build_tsne_with_rollouts(z_clips_test, gru):
    """
    z_clips_test: CPU tensor [N,T,D]
    gru: FutureGRU (on DEVICE)
    Returns:
        coords_static: [M,2] static background
        traj_coords: list of np.ndarray, each [FUTURE_LEN+1,2]
    """
    N, T, D = z_clips_test.shape
    assert T == CLIP_LEN

    # ---- Static latent cloud (subsample)
    all_z = z_clips_test.reshape(-1, D).numpy()
    total = all_z.shape[0]
    rng = np.random.default_rng(0)

    if total > MAX_STATIC_TSNE_POINTS:
        idx_static = rng.choice(total, size=MAX_STATIC_TSNE_POINTS, replace=False)
        static_z = all_z[idx_static]
    else:
        static_z = all_z

    # ---- Build GRU rollout trajectories in latent space
    # Choose which clips to roll out
    n_rollouts = min(N_ROLLOUTS_TSNE, N)
    rollout_indices = rng.choice(N, size=n_rollouts, replace=False)

    traj_latent_lists = []    # per rollout: [FUTURE_LEN+1, D]

    gru.eval()
    for idx in rollout_indices:
        z_clip = z_clips_test[idx]          # [T,D] (CPU)
        z_hist = z_clip[:HIST_LEN].clone()  # [HIST_LEN,D]

        # store start (t4 latent)
        latents = [z_hist[-1].numpy().copy()]

        for _ in range(FUTURE_LEN):
            z_in = z_hist.unsqueeze(0).to(DEVICE)   # [1,HIST_LEN,D]
            with torch.no_grad():
                z_next = gru(z_in).squeeze(0).cpu() # [D]
            latents.append(z_next.numpy().copy())
            # autoregressive update
            z_hist = torch.cat([z_hist[1:], z_next.unsqueeze(0)], dim=0)

        traj_latent_lists.append(np.stack(latents, axis=0))  # [FUTURE_LEN+1,D]

    # ---- Build single matrix for t-SNE
    tsne_points = []
    # 1) static background
    for i in range(static_z.shape[0]):
        tsne_points.append(static_z[i])
    # 2) all rollout latents
    traj_index_lists = []   # per rollout: list of indices in tsne_points
    current_idx = len(tsne_points)
    for traj in traj_latent_lists:
        idxs = list(range(current_idx, current_idx + traj.shape[0]))
        traj_index_lists.append(idxs)
        for j in range(traj.shape[0]):
            tsne_points.append(traj[j])
        current_idx += traj.shape[0]

    Z = np.stack(tsne_points, axis=0).astype(np.float32)
    print(f"[tsne] Running t-SNE on {Z.shape[0]} points (static={static_z.shape[0]}, rollout_latents={Z.shape[0]-static_z.shape[0]})")

    # IMPORTANT: do NOT pass n_iter to avoid version issues.
    tsne = TSNE(
        n_components=2,
        perplexity=30.0,
        init="random",
        random_state=0,
    )
    coords_all = tsne.fit_transform(Z)

    coords_static = coords_all[:static_z.shape[0]]
    traj_coords = []
    for idxs in traj_index_lists:
        traj_coords.append(coords_all[idxs])

    return coords_static, traj_coords


def plot_tsne_rollouts(coords_static, traj_coords):
    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    ax.set_title("t-SNE manifold with multi-step GRU rollouts (basic32)")

    # background static cloud
    ax.scatter(
        coords_static[:, 0],
        coords_static[:, 1],
        s=4,
        alpha=0.15,
        label="static latents",
    )

    # rollouts: draw polylines from start (t4) to t8
    first_start = True
    first_end   = True
    for traj in traj_coords:
        xs = traj[:, 0]
        ys = traj[:, 1]
        ax.plot(xs, ys, linewidth=0.8, alpha=0.8, color="black")

        # mark start and end
        if first_start:
            ax.scatter(xs[0], ys[0], s=18, marker="o", color="green", label="rollout start (t4)")
            first_start = False
        else:
            ax.scatter(xs[0], ys[0], s=18, marker="o", color="green")

        if first_end:
            ax.scatter(xs[-1], ys[-1], s=22, marker="x", color="red", label="rollout end (t8)")
            first_end = False
        else:
            ax.scatter(xs[-1], ys[-1], s=22, marker="x", color="red")

    ax.legend(loc="best", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "temporal_rollout_tsne_basic32.png")
    plt.savefig(out_path, dpi=120)
    print(f"[tsne] Saved t-SNE rollout figure -> {out_path}")


# --------------------
# Main
# --------------------
def main():
    print(f"[tsne] Using device: {DEVICE}")

    # Load AE
    ae = load_scene_model(CKPT_SCENEMODEL)

    # Generate temporal dataset
    print(f"[tsne] Generating temporal dataset: train={N_TRAIN_CLIPS}, test={N_TEST_CLIPS}, T={CLIP_LEN}")
    clips_train = generate_temporal_dataset(N_TRAIN_CLIPS, CLIP_LEN)
    clips_test  = generate_temporal_dataset(N_TEST_CLIPS, CLIP_LEN)

    # Encode to latents
    def encode_clips(clips_np):
        N, T, C, H, W = clips_np.shape
        x = torch.from_numpy(clips_np).float().to(DEVICE)
        x_flat = x.view(-1, C, H, W)
        with torch.no_grad():
            z_flat = ae.encode(x_flat)
        return z_flat.view(N, T, LATENT_DIM).cpu()

    print("[tsne] Encoding train clips...")
    z_clips_train = encode_clips(clips_train)
    print("[tsne] Encoding test clips...")
    z_clips_test  = encode_clips(clips_test)

    # Load or train GRU
    gru = load_or_train_future_gru(z_clips_train, z_clips_test)

    # Build t-SNE with multi-step rollouts
    coords_static, traj_coords = build_tsne_with_rollouts(z_clips_test, gru)

    # Plot
    plot_tsne_rollouts(coords_static, traj_coords)


if __name__ == "__main__":
    main()
