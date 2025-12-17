#!/usr/bin/env python3
# geomlang_basic32_temporal_geom.py
#
# Temporal experiment on top of the basic32-geom model:
#   - Build synthetic "clips" by linear interpolation in latent space
#   - Train a GRU: z[0..3] -> z[4]
#   - Visualize:
#       * Grid of decoded frames:
#           t=1..4, True T, Linear-extrap T, GRU T, |GRU-True|
#       * Trajectory plot of red/blue centroids:
#           true positions over t=1..5 + GRU-pred endpoint (x marker)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from geomlang_basic32_geom import (
    IMG_SIZE,
    LATENT_DIM,
    SceneModel,
    tensor_to_rgb,
    OUT_DIR,
    CKPT_PATH,
    LATENTS_PATH,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_GRID = os.path.join(OUT_DIR, "temporal_future_grid_basic32_geom.png")
OUT_TRAJ = os.path.join(OUT_DIR, "temporal_trajectories_basic32_geom.png")

SEQ_LEN  = 5   # t=1..5 (we use 1..4 as history, 5 as future)
N_ROWS   = 8   # rows in the viz grid
BATCH    = 128
EPOCHS   = 25
STEPS_PER_EPOCH = 80
LR       = 1e-3

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# GRU model
# ---------------------------------------------------------------------

class FutureGRU(nn.Module):
    """
    GRU that maps a sequence of latents z[0..T-2] -> z_T.
    """
    def __init__(self, dim_latent, hidden=256, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=dim_latent,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden, dim_latent)

    def forward(self, z_hist):
        # z_hist: [B, T_hist, D]
        out, _ = self.gru(z_hist)
        last = out[:, -1, :]  # [B, hidden]
        return self.fc(last)  # [B, D]


# ---------------------------------------------------------------------
# Helpers: load AE + latents
# ---------------------------------------------------------------------

def load_scene_model():
    print(f"[temporal-basic32] Loading AE from {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model = SceneModel().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model


def load_latents():
    data = np.load(LATENTS_PATH)
    z = torch.from_numpy(data["z"]).float().to(DEVICE)     # [N,D]
    print(f"[temporal-basic32] Loaded latents: N={z.shape[0]}, D={z.shape[1]}")
    return z


# ---------------------------------------------------------------------
# Latent interpolation utilities
# ---------------------------------------------------------------------

@torch.no_grad()
def build_interpolant_seq(z_all, idx_start, idx_end, T=SEQ_LEN):
    """
    Linear latent interpolation between two dataset points.
    Returns [T,D].
    """
    z0 = z_all[idx_start]
    zT = z_all[idx_end]
    alphas = torch.linspace(0.0, 1.0, T, device=z_all.device)  # [T]
    seq = (1.0 - alphas.unsqueeze(-1)) * z0.unsqueeze(0) + \
          alphas.unsqueeze(-1)         * zT.unsqueeze(0)
    return seq  # [T,D]


def sample_interpolant_batch(z_all, batch_size, T=SEQ_LEN):
    """
    Build a batch of interpolant sequences.

    Returns:
        z_hist:  [B,T-1,D] (t=1..T-1)
        z_true:  [B,D]     (t=T)
    """
    N, D = z_all.shape
    idx_start = torch.randint(0, N, (batch_size,), device=z_all.device)
    idx_end   = torch.randint(0, N, (batch_size,), device=z_all.device)

    seqs = []
    for i in range(batch_size):
        seqs.append(build_interpolant_seq(z_all, idx_start[i], idx_end[i], T=T))
    seqs = torch.stack(seqs, dim=0)        # [B,T,D]

    z_hist = seqs[:, :T-1, :]              # [B,T-1,D]
    z_true = seqs[:, -1, :]                # [B,D]
    return z_hist, z_true


# ---------------------------------------------------------------------
# Centroid helpers for trajectories
# ---------------------------------------------------------------------

def channel_centroid(img_ch):
    """
    img_ch: [H,W] tensor
    returns (cx, cy) floats in pixel coords.
    """
    H, W = img_ch.shape
    ys = torch.arange(H, device=img_ch.device, dtype=torch.float32)
    xs = torch.arange(W, device=img_ch.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    m = img_ch.sum()
    if m <= 1e-6:
        return W / 2.0, H / 2.0

    cx = (img_ch * xx).sum() / m
    cy = (img_ch * yy).sum() / m
    return cx.item(), cy.item()


def frame_centers(img_3ch):
    """
    img_3ch: [3,H,W] tensor
    returns (cx_r, cy_r), (cx_b, cy_b)
    """
    r = img_3ch[0]
    b = img_3ch[1]
    return channel_centroid(r), channel_centroid(b)


# ---------------------------------------------------------------------
# Training GRU
# ---------------------------------------------------------------------

def train_future_gru(z_all):
    model = FutureGRU(dim_latent=z_all.shape[1], hidden=256, num_layers=1).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for _ in range(STEPS_PER_EPOCH):
            z_hist, z_true = sample_interpolant_batch(z_all, BATCH, T=SEQ_LEN)
            pred = model(z_hist)
            loss = F.mse_loss(pred, z_true)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            z_hist_val, z_true_val = sample_interpolant_batch(z_all, BATCH, T=SEQ_LEN)
            val_pred = model(z_hist_val)
            val_loss = F.mse_loss(val_pred, z_true_val).item()

        print(
            f"[temporal-basic32] Epoch {epoch}/{EPOCHS} | "
            f"train MSE={total_loss / STEPS_PER_EPOCH:.6f} | "
            f"val MSE={val_loss:.6f}"
        )

    return model


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------

@torch.no_grad()
def visualize(ae_model, future_gru, z_all):
    N, D = z_all.shape
    rng = np.random.default_rng(123)

    # Sample 2 indices per row (start/end)
    idx_pairs = rng.integers(0, N, size=(N_ROWS, 2))

    fig, axes = plt.subplots(
        nrows=N_ROWS,
        ncols=7,           # t1..t4, True T, Linear T, GRU T, |GRU-True|
        figsize=(7 * 1.4, N_ROWS * 1.4),
        dpi=120,
    )
    if N_ROWS == 1:
        axes = np.expand_dims(axes, 0)

    fig.suptitle("Latent dynamics (basic32-geom): t=1..4, True T, Linear, GRU, |GRU-True|")

    # Storage for trajectories
    red_trajs_true  = []
    blue_trajs_true = []
    red_pred_Ts     = []
    blue_pred_Ts    = []

    for row in range(N_ROWS):
        i_start, i_end = idx_pairs[row]
        seq = build_interpolant_seq(z_all, i_start, i_end, T=SEQ_LEN)  # [T,D]

        z_hist   = seq[:SEQ_LEN-1]   # [4,D]
        z_true_T = seq[-1]           # [D]

        # Linear extrapolation: constant-velocity step from z3->z4
        z_lin_T = z_hist[-1] + (z_hist[-1] - z_hist[-2])

        # GRU prediction
        z_gru_T = future_gru(z_hist.unsqueeze(0)).squeeze(0)

        # Decode frames
        def decode_latents(z_batch):
            z_batch = z_batch.to(DEVICE)
            imgs = ae_model.decode(z_batch)
            return imgs

        imgs_hist = decode_latents(z_hist)            # [4,3,H,W]
        img_true  = decode_latents(z_true_T.unsqueeze(0)).squeeze(0)  # [3,H,W]
        img_lin   = decode_latents(z_lin_T.unsqueeze(0)).squeeze(0)   # [3,H,W]
        img_gru   = decode_latents(z_gru_T.unsqueeze(0)).squeeze(0)   # [3,H,W]

        diff = (img_gru - img_true).abs().mean(dim=0).cpu().numpy()   # [H,W]

        col_titles = ["t=1", "t=2", "t=3", "t=4", "True T", "Linear", "GRU", "|GRU-True|"]
        for col in range(8):
            if col < 7:
                ax = axes[row, col]
            else:
                # add extra column for diff heatmap
                continue

        # Actually we created 7 cols, but we want 8th for diff.
        # So we re-layout: 0..3 hist, 4 true, 5 linear, 6 GRU, and
        # we use an extra axes object for diff appended later.
        # Simpler: allocate 8 columns from the start.

    plt.close(fig)  # we'll rebuild properly below

    # Rebuild grid with correct 8 columns
    fig, axes = plt.subplots(
        nrows=N_ROWS,
        ncols=8,
        figsize=(8 * 1.4, N_ROWS * 1.4),
        dpi=120,
    )
    if N_ROWS == 1:
        axes = np.expand_dims(axes, 0)

    col_titles = ["t=1", "t=2", "t=3", "t=4", "True T", "Linear", "GRU", "|GRU-True|"]

    for row in range(N_ROWS):
        i_start, i_end = idx_pairs[row]
        seq = build_interpolant_seq(z_all, i_start, i_end, T=SEQ_LEN)

        z_hist   = seq[:SEQ_LEN-1]
        z_true_T = seq[-1]
        z_lin_T  = z_hist[-1] + (z_hist[-1] - z_hist[-2])
        z_gru_T  = future_gru(z_hist.unsqueeze(0)).squeeze(0)

        def decode_latents(z_batch):
            z_batch = z_batch.to(DEVICE)
            imgs = ae_model.decode(z_batch)
            return imgs

        imgs_hist = decode_latents(z_hist)            # [4,3,H,W]
        img_true  = decode_latents(z_true_T.unsqueeze(0)).squeeze(0)
        img_lin   = decode_latents(z_lin_T.unsqueeze(0)).squeeze(0)
        img_gru   = decode_latents(z_gru_T.unsqueeze(0)).squeeze(0)

        diff = (img_gru - img_true).abs().mean(dim=0).cpu().numpy()

        # Collect trajectories (true t=1..5 + GRU endpoint)
        red_traj_true  = []
        blue_traj_true = []
        # t=1..4
        for t in range(SEQ_LEN-1):
            rc, bc = frame_centers(imgs_hist[t])
            red_traj_true.append(rc)
            blue_traj_true.append(bc)
        # true T (t=5)
        rc_T, bc_T = frame_centers(img_true)
        red_traj_true.append(rc_T)
        blue_traj_true.append(bc_T)

        rc_pred, bc_pred = frame_centers(img_gru)
        red_trajs_true.append(red_traj_true)
        blue_trajs_true.append(blue_traj_true)
        red_pred_Ts.append(rc_pred)
        blue_pred_Ts.append(bc_pred)

        # --- plot row ---
        for col in range(8):
            ax = axes[row, col]
            ax.axis("off")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10)

            if col < 4:
                rgb = tensor_to_rgb(imgs_hist[col])
                ax.imshow(rgb)
            elif col == 4:
                ax.imshow(tensor_to_rgb(img_true))
            elif col == 5:
                ax.imshow(tensor_to_rgb(img_lin))
            elif col == 6:
                ax.imshow(tensor_to_rgb(img_gru))
            else:
                im = ax.imshow(
                    diff,
                    cmap="inferno",
                    vmin=0.0,
                    vmax=max(1e-3, diff.max()),
                )
                if row == 0:
                    # add colorbar on the right once
                    cax = fig.add_axes([0.93, 0.1, 0.02, 0.8])
                    fig.colorbar(im, cax=cax)

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(OUT_GRID)
    plt.close(fig)
    print(f"[temporal-basic32] Saved grid -> {OUT_GRID}")

    # -----------------------------------------------------------------
    # Trajectory figure
    # -----------------------------------------------------------------
    fig2, axes2 = plt.subplots(
        nrows=N_ROWS,
        ncols=1,
        figsize=(4, 1.6 * N_ROWS),
        dpi=120,
    )
    if N_ROWS == 1:
        axes2 = [axes2]

    for row in range(N_ROWS):
        ax = axes2[row]
        ax.set_xlim(0, IMG_SIZE)
        ax.set_ylim(IMG_SIZE, 0)  # invert y to match image coords
        ax.set_xticks([])
        ax.set_yticks([])

        r_traj = red_trajs_true[row]
        b_traj = blue_trajs_true[row]

        rx = [p[0] for p in r_traj]
        ry = [p[1] for p in r_traj]
        bx = [p[0] for p in b_traj]
        by = [p[1] for p in b_traj]

        ax.plot(rx, ry, "-o", linewidth=1.0, markersize=3, label="red true")
        ax.plot(bx, by, "-o", linewidth=1.0, markersize=3, label="blue true")

        r_pred = red_pred_Ts[row]
        b_pred = blue_pred_Ts[row]

        ax.scatter([r_pred[0]], [r_pred[1]], marker="x", s=40)
        ax.scatter([b_pred[0]], [b_pred[1]], marker="x", s=40)

        if row == 0:
            ax.legend(loc="upper right", fontsize=6)

    plt.tight_layout()
    plt.savefig(OUT_TRAJ)
    plt.close(fig2)
    print(f"[temporal-basic32] Saved trajectories -> {OUT_TRAJ}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print(">>")
    print(f"[temporal-basic32] Using device: {DEVICE}")

    ae = load_scene_model()
    z_all = load_latents()

    future_gru = train_future_gru(z_all)

    visualize(ae, future_gru, z_all)


if __name__ == "__main__":
    main()
