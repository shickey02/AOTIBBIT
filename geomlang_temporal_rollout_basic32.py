#!/usr/bin/env python3
# geomlang_temporal_rollout_basic32.py
#
# Multi-step temporal rollout analysis for the 32x32 geomlang model.
#
# Run:
#   python bbit_geomlang/geomlang_temporal_rollout_basic32.py

import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt

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

# Temporal setup
CLIP_LEN   = 8   # total frames per clip
HIST_LEN   = 4   # context length
FUTURE_LEN = 4   # how many steps we roll out
assert CLIP_LEN == HIST_LEN + FUTURE_LEN

N_TRAIN_CLIPS = 2000
N_TEST_CLIPS  = 500

BATCH_SIZE = 128
EPOCHS_GRU = 20
LR_GRU     = 1e-3

REL_NAMES = ["left_of", "right_of", "above", "below", "overlap"]

# --------------------
# Drawing primitives
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
    Generate a single temporal clip of length T with moving red and blue blobs.

    Returns:
        clip: np.ndarray of shape [T, 3, H, W] in [0,1]
    """
    H = W = IMG_SIZE
    margin = 4

    # sample shapes and sizes
    shape_red = np.random.randint(0, 2)   # 0 circle, 1 square
    shape_blue = np.random.randint(0, 2)

    base = np.random.randint(4, 7)  # 4..6
    r_red = int(np.clip(base + np.random.randint(-1, 2), 3, 7))
    r_blue = int(np.clip(base + np.random.randint(-1, 2), 3, 7))

    # initial centers
    def sample_center(r):
        cx = np.random.randint(margin + r, W - margin - r)
        cy = np.random.randint(margin + r, H - margin - r)
        return float(cx), float(cy)

    cx_r, cy_r = sample_center(r_red)
    cx_b, cy_b = sample_center(r_blue)

    # velocities (pixels per step)
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

    for _ in range(T):
        grid = np.zeros((NUM_CHANNELS, H, W), dtype=np.float32)

        # draw shapes
        draw_shape(grid, shape_red, int(round(cx_r)), int(round(cy_r)), r_red, 0)  # red in channel 0
        draw_shape(grid, shape_blue, int(round(cx_b)), int(round(cy_b)), r_blue, 2)  # blue in channel 2

        frames[_] = grid

        # update centers with reflection at borders
        def step_pos(cx, cy, vx, vy, r):
            cx_new = cx + vx
            cy_new = cy + vy
            # reflect horizontally
            if cx_new < margin + r:
                cx_new = margin + r + (margin + r - cx_new)
                vx *= -1
            elif cx_new > W - margin - r:
                cx_new = W - margin - r - (cx_new - (W - margin - r))
                vx *= -1
            # reflect vertically
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
    # [N, T, C, H, W]
    return clips


# --------------------
# SceneModel (autoencoder) – same architecture as basic32 training
# --------------------
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
        z = self.fc(x)
        return z


class Decoder(nn.Module):
    def __init__(self, out_channels=NUM_CHANNELS, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 8x8
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)  # 16x16
        self.deconv3 = nn.ConvTranspose2d(16, out_channels, kernel_size=4, stride=2, padding=1)  # 32x32

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
        # classifier heads not used here, but defined so the checkpoint loads cleanly
        self.rel_head = nn.Linear(LATENT_DIM, 6)
        self.scale_head = nn.Linear(LATENT_DIM, 3)
        self.shape_red_head = nn.Linear(LATENT_DIM, 2)
        self.shape_blue_head = nn.Linear(LATENT_DIM, 2)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


def load_scene_model(path=CKPT_SCENEMODEL):
    print(f"[rollout] Loading SceneModel from {path}")
    ckpt = torch.load(path, map_location=DEVICE)
    model = SceneModel()
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt, strict=False)
    else:
        # assume raw state_dict
        model.load_state_dict(ckpt, strict=False)
    model.to(DEVICE)
    model.eval()
    return model


# --------------------
# GRU for future prediction
# --------------------
class FutureGRU(nn.Module):
    def __init__(self, dim_latent, hidden=128):
        super().__init__()
        self.gru = nn.GRU(dim_latent, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, dim_latent)

    def forward(self, z_hist):
        # z_hist: [B, T_hist, D]
        out, _ = self.gru(z_hist)
        last = out[:, -1, :]  # [B, hidden]
        return self.fc(last)  # [B, D]


def build_gru_dataset(z_clips, hist_len=HIST_LEN):
    """
    z_clips: [N, T, D]
    Returns:
        X: [M, hist_len, D]
        Y: [M, D]  (future at t+hist_len)
    using sliding windows along time.
    """
    N, T, D = z_clips.shape
    samples_X = []
    samples_Y = []
    for i in range(N):
        for t in range(T - hist_len):
            samples_X.append(z_clips[i, t:t + hist_len])
            samples_Y.append(z_clips[i, t + hist_len])
    X = torch.stack(samples_X, dim=0)
    Y = torch.stack(samples_Y, dim=0)
    return X, Y


def train_future_gru(z_clips_train, z_clips_test):
    X_train, Y_train = build_gru_dataset(z_clips_train, HIST_LEN)
    X_test, Y_test = build_gru_dataset(z_clips_test, HIST_LEN)

    train_ds = TensorDataset(X_train, Y_train)
    test_ds = TensorDataset(X_test, Y_test)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = FutureGRU(dim_latent=LATENT_DIM, hidden=128).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR_GRU)
    loss_fn = nn.MSELoss()

    for epoch in range(1, EPOCHS_GRU + 1):
        model.train()
        train_loss = 0.0
        for Xb, Yb in train_loader:
            Xb = Xb.to(DEVICE)
            Yb = Yb.to(DEVICE)
            opt.zero_grad()
            pred = model(Xb)
            loss = loss_fn(pred, Yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * Xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for Xb, Yb in test_loader:
                Xb = Xb.to(DEVICE)
                Yb = Yb.to(DEVICE)
                pred = model(Xb)
                loss = loss_fn(pred, Yb)
                test_loss += loss.item() * Xb.size(0)
        test_loss /= len(test_loader.dataset)

        print(f"[rollout] Epoch {epoch:2d}/{EPOCHS_GRU} | train MSE={train_loss:.6f} | test MSE={test_loss:.6f}")

    torch.save(model.state_dict(), CKPT_FUTURE_GRU)
    print(f"[rollout] Saved FutureGRU -> {CKPT_FUTURE_GRU}")
    return model


def load_or_train_future_gru(z_clips_train, z_clips_test):
    if os.path.exists(CKPT_FUTURE_GRU):
        print(f"[rollout] Loading FutureGRU from {CKPT_FUTURE_GRU}")
        model = FutureGRU(dim_latent=LATENT_DIM, hidden=128)
        state = torch.load(CKPT_FUTURE_GRU, map_location=DEVICE)
        model.load_state_dict(state, strict=False)
        model.to(DEVICE)
        model.eval()
        return model
    else:
        print("[rollout] No FutureGRU checkpoint found, training a new one...")
        return train_future_gru(z_clips_train, z_clips_test)


# --------------------
# Geometry helpers
# --------------------
def compute_centroid(img_tensor, channel_idx):
    """
    Compute centroid (x,y) in pixel coordinates for the given channel of [C,H,W] tensor in [0,1].
    Uses intensity-weighted average to be robust to smooth edges.
    """
    x = img_tensor[channel_idx]
    H, W = x.shape
    x_np = x.detach().cpu().numpy()
    yy, xx = np.mgrid[0:H, 0:W]
    weights = x_np
    total = weights.sum()
    if total <= 1e-6:
        # fallback to center of image (should not really happen)
        return W / 2.0, H / 2.0
    cx = float((weights * xx).sum() / total)
    cy = float((weights * yy).sum() / total)
    return cx, cy


def relation_from_centroids(cx_r, cy_r, cx_b, cy_b, tol=1.0):
    """
    Simple geometric relation between red and blue based on centroids.
    Returns an integer 0..4 corresponding to REL_NAMES.
    """
    if cx_r + tol < cx_b - tol:
        return 0  # left_of
    if cx_r - tol > cx_b + tol:
        return 1  # right_of
    if cy_r + tol < cy_b - tol:
        return 2  # above
    if cy_r - tol > cy_b + tol:
        return 3  # below
    return 4  # overlap / near


# --------------------
# Rollout + metrics
# --------------------
@torch.no_grad()
def rollout_and_metrics(ae, gru, clips_test, n_rows_vis=8):
    """
    ae: SceneModel
    gru: FutureGRU
    clips_test: np.ndarray [N,T,C,H,W]
    """
    N, T, C, H, W = clips_test.shape

    # encode all clips to latents
    clips_torch = torch.from_numpy(clips_test).float().to(DEVICE)
    clips_flat = clips_torch.view(-1, C, H, W)
    z_flat = ae.encode(clips_flat)
    z_clips = z_flat.view(N, T, LATENT_DIM)

    max_h = FUTURE_LEN
    red_err = np.zeros(max_h, dtype=np.float64)
    blue_err = np.zeros(max_h, dtype=np.float64)
    rel_correct = np.zeros(max_h, dtype=np.int64)
    rel_total = np.zeros(max_h, dtype=np.int64)

    # for visualization, pick a few random clips
    rng = np.random.default_rng(0)
    vis_indices = rng.choice(N, size=min(n_rows_vis, N), replace=False)

    vis_true = []
    vis_pred = []

    for idx in range(N):
        # latent history: first HIST_LEN frames
        z_hist = z_clips[idx, :HIST_LEN].clone()  # [HIST_LEN, D]
        clip_np = clips_test[idx]                 # [T,C,H,W]
        preds_imgs = []

        for h_step in range(1, max_h + 1):
            # current window: shape [1, HIST_LEN, D]
            z_in = z_hist.unsqueeze(0)
            z_next = gru(z_in).squeeze(0)  # [D]

            # decode predicted frame
            img_pred = ae.decode(z_next.unsqueeze(0)).squeeze(0)  # [C,H,W]

            # ground-truth frame at this horizon
            t_true = HIST_LEN - 1 + h_step  # index in 0..T-1
            if t_true >= T:
                break
            img_true = torch.from_numpy(clip_np[t_true]).to(DEVICE)

            # centroids
            cx_r_pred, cy_r_pred = compute_centroid(img_pred, 0)
            cx_b_pred, cy_b_pred = compute_centroid(img_pred, 2)
            cx_r_true, cy_r_true = compute_centroid(img_true, 0)
            cx_b_true, cy_b_true = compute_centroid(img_true, 2)

            dr = math.hypot(cx_r_pred - cx_r_true, cy_r_pred - cy_r_true)
            db = math.hypot(cx_b_pred - cx_b_true, cy_b_pred - cy_b_true)

            red_err[h_step - 1] += dr
            blue_err[h_step - 1] += db

            # relations
            rel_true = relation_from_centroids(cx_r_true, cy_r_true, cx_b_true, cy_b_true)
            rel_pred = relation_from_centroids(cx_r_pred, cy_r_pred, cx_b_pred, cy_b_pred)
            if rel_true == rel_pred:
                rel_correct[h_step - 1] += 1
            rel_total[h_step - 1] += 1

            # update history with autoregressive prediction
            z_hist = torch.cat([z_hist[1:], z_next.unsqueeze(0)], dim=0)

            # store for visualization only if this is a chosen clip
            if idx in vis_indices:
                preds_imgs.append(img_pred.detach().cpu().numpy())

        if idx in vis_indices:
            vis_true.append(clip_np.copy())
            vis_pred.append(np.stack(preds_imgs, axis=0))

    # normalize errors
    den = np.maximum(rel_total, 1).astype(np.float64)
    red_err /= den
    blue_err /= den
    rel_acc = rel_correct / den

    print("\n[rollout] Centroid error vs horizon (pixels):")
    for h in range(max_h):
        print(f"  +{h+1} step(s): red={red_err[h]:.3f} px, blue={blue_err[h]:.3f} px")

    print("\n[rollout] Relation accuracy vs horizon:")
    for h in range(max_h):
        print(f"  +{h+1} step(s): acc={100.0*rel_acc[h]:.2f}%  (N={rel_total[h]})")

    # -------------
    # Visualization grid
    # -------------
    n_rows = len(vis_indices)
    if n_rows == 0:
        return

    vis_true = np.stack(vis_true, axis=0)  # [R,T,C,H,W]
    vis_pred = np.stack(vis_pred, axis=0)  # [R,F,C,H,W]
    R, T, C, H, W = vis_true.shape
    _, F_vis, _, _, _ = vis_pred.shape

    ncols = HIST_LEN + FUTURE_LEN + FUTURE_LEN + 1  # context, true futures, GRU futures, diff
    fig, axes = plt.subplots(
        nrows=R,
        ncols=ncols,
        figsize=(ncols * 1.2, R * 1.2),
        dpi=120,
    )

    fig.suptitle(
        "Temporal rollout (basic32): context, true futures, GRU futures, final |GRU-True|",
        fontsize=12,
    )

    def to_rgb(img_chw):
        r = img_chw[0]
        b = img_chw[2]
        H, W = r.shape
        rgb = np.zeros((H, W, 3), dtype=np.float32)
        rgb[..., 0] = r
        rgb[..., 2] = b
        return rgb

    for row in range(R):
        clip = vis_true[row]      # [T,C,H,W]
        preds = vis_pred[row]     # [F_vis,C,H,W]

        # final diff heatmap between last predicted and last true frame
        last_true = clip[HIST_LEN - 1 + F_vis]
        last_pred = preds[F_vis - 1]
        diff = np.abs(last_pred - last_true).mean(axis=0)

        for col in range(ncols):
            ax = axes[row, col]
            ax.axis("off")

            if row == 0:
                if col < HIST_LEN:
                    ax.set_title(f"t={col+1}", fontsize=8)
                elif col < HIST_LEN + FUTURE_LEN:
                    ax.set_title(f"True t{col+1}", fontsize=8)
                elif col < HIST_LEN + FUTURE_LEN + FUTURE_LEN:
                    k = col - (HIST_LEN + FUTURE_LEN) + 1
                    ax.set_title(f"GRU t{HIST_LEN+k}", fontsize=8)
                else:
                    ax.set_title("|GRU-True| last", fontsize=8)

            if col < HIST_LEN:
                img = clip[col]
                ax.imshow(to_rgb(img))
            elif col < HIST_LEN + FUTURE_LEN:
                t_idx = col
                if t_idx < T:
                    img = clip[t_idx]
                    ax.imshow(to_rgb(img))
            elif col < HIST_LEN + FUTURE_LEN + FUTURE_LEN:
                k = col - (HIST_LEN + FUTURE_LEN)  # 0..FUTURE_LEN-1
                if k < F_vis:
                    img = preds[k]
                    ax.imshow(to_rgb(img))
            else:
                im = ax.imshow(diff, cmap="magma",
                               vmin=0.0, vmax=max(1e-3, diff.max()))
                if row == 0:
                    cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
                    fig.colorbar(im, cax=cax)

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    out_path = os.path.join(OUT_DIR, "temporal_rollout_grid_basic32.png")
    plt.savefig(out_path)
    print(f"[rollout] Saved rollout grid -> {out_path}")


# --------------------
# Main
# --------------------
def main():
    print(f"[rollout] Using device: {DEVICE}")

    # Load AE
    ae = load_scene_model(CKPT_SCENEMODEL)

    # Generate temporal dataset
    print(f"[rollout] Generating temporal dataset: "
          f"train={N_TRAIN_CLIPS}, test={N_TEST_CLIPS}, T={CLIP_LEN}")
    clips_train = generate_temporal_dataset(N_TRAIN_CLIPS, CLIP_LEN)
    clips_test = generate_temporal_dataset(N_TEST_CLIPS, CLIP_LEN)

    # Encode clips to latents
    def encode_clips(clips_np):
        N, T, C, H, W = clips_np.shape
        x = torch.from_numpy(clips_np).float().to(DEVICE)
        x_flat = x.view(-1, C, H, W)
        with torch.no_grad():
            z_flat = ae.encode(x_flat)
        return z_flat.view(N, T, LATENT_DIM).cpu()

    print("[rollout] Encoding train clips...")
    z_clips_train = encode_clips(clips_train)
    print("[rollout] Encoding test clips...")
    z_clips_test = encode_clips(clips_test)

    # Train or load GRU
    gru = load_or_train_future_gru(z_clips_train, z_clips_test)

    # Rollout + metrics
    rollout_and_metrics(ae, gru, clips_test, n_rows_vis=8)


if __name__ == "__main__":
    main()
