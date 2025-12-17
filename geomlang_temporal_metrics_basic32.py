#!/usr/bin/env python3
# geomlang_temporal_metrics_basic32.py
#
# Step A + B: measure geometric accuracy of GRU dynamics on the
# basic32 autoencoder AND test whether GRU predictions preserve
# high-level relations (left/right/above/below/overlapping).
#
# Uses your trained SceneModel from geomlang_multishape_relscale.py.

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from geomlang_multishape_relscale import (
    SceneModel,
    LATENT_DIM,
    IMG_SIZE,
    NUM_CHANNELS as NUM_CH,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LEN    = 5          # t=1..5 (we show 1..4, predict 5)
N_SEQ      = 4000
TRAIN_FRAC = 0.8
BATCH_SIZE = 64
EPOCHS     = 20
LR         = 1e-3

AE_CKPT  = os.path.join("outputs_basic32", "scene_model_basic32.pt")
OUT_GRID = os.path.join("outputs_basic32",
                        "temporal_future_grid_basic32_metrics.png")

REL_NAMES = ["left_of", "right_of", "above", "below", "overlap"]


# ---------------------------------------------------------------------
# Drawing primitives (same as basic32 style)
# ---------------------------------------------------------------------

def draw_circle(grid, cx, cy, radius, ch_idx):
    h, w = grid.shape[1], grid.shape[2]
    for y in range(h):
        for x in range(w):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                grid[ch_idx, y, x] = 1.0


def draw_square(grid, cx, cy, half_size, ch_idx):
    h, w = grid.shape[1], grid.shape[2]
    x0 = max(0, cx - half_size)
    x1 = min(w, cx + half_size + 1)
    y0 = max(0, cy - half_size)
    y1 = min(h, cy + half_size + 1)
    grid[ch_idx, y0:y1, x0:x1] = 1.0


def draw_shape(grid, is_circle, cx, cy, size, ch_idx):
    if is_circle:
        draw_circle(grid, cx, cy, size, ch_idx)
    else:
        draw_square(grid, cx, cy, size, ch_idx)


# ---------------------------------------------------------------------
# Use your trained SceneModel as AE
# ---------------------------------------------------------------------

class AEWrapper(nn.Module):
    def __init__(self, scene_model: SceneModel):
        super().__init__()
        self.encoder = scene_model.encoder
        self.decoder = scene_model.decoder
        self.latent_dim = LATENT_DIM

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


def load_ae():
    if not os.path.exists(AE_CKPT):
        raise FileNotFoundError(
            f"AE checkpoint not found: {AE_CKPT}"
        )

    raw = torch.load(AE_CKPT, map_location=DEVICE)

    model = SceneModel()
    if isinstance(raw, dict) and "state_dict" in raw:
        model.load_state_dict(raw["state_dict"])
    elif isinstance(raw, dict):
        model.load_state_dict(raw)
    else:
        model = raw

    model.to(DEVICE).eval()
    ae = AEWrapper(model).to(DEVICE).eval()
    print(f"[metrics] Loaded SceneModel from {AE_CKPT} (latent_dim={ae.latent_dim})")
    return ae


# ---------------------------------------------------------------------
# Temporal dataset: simple straight-ish motion clips
# ---------------------------------------------------------------------

def generate_temporal_dataset(num_seq=N_SEQ, T=SEQ_LEN, seed=0):
    rng = np.random.default_rng(seed)
    clips = np.zeros((num_seq, T, NUM_CH, IMG_SIZE, IMG_SIZE), dtype=np.float32)

    for n in range(num_seq):
        red_is_circle  = rng.integers(0, 2)
        blue_is_circle = rng.integers(0, 2)

        r_red  = rng.integers(4, 7)
        r_blue = rng.integers(4, 7)

        margin = 4
        cx_red  = rng.integers(margin, IMG_SIZE - margin)
        cy_red  = rng.integers(margin, IMG_SIZE - margin)
        cx_blue = rng.integers(margin, IMG_SIZE - margin)
        cy_blue = rng.integers(margin, IMG_SIZE - margin)

        vx_red  = rng.uniform(-1.5, 1.5)
        vy_red  = rng.uniform(-1.5, 1.5)
        vx_blue = rng.uniform(-1.5, 1.5)
        vy_blue = rng.uniform(-1.5, 1.5)

        for t in range(T):
            img = np.zeros((NUM_CH, IMG_SIZE, IMG_SIZE), dtype=np.float32)

            draw_shape(img, red_is_circle,
                       int(round(cx_red)), int(round(cy_red)),
                       r_red, ch_idx=0)
            draw_shape(img, blue_is_circle,
                       int(round(cx_blue)), int(round(cy_blue)),
                       r_blue, ch_idx=2)

            clips[n, t] = img

            cx_red  = np.clip(cx_red  + vx_red,  margin, IMG_SIZE - margin)
            cy_red  = np.clip(cy_red  + vy_red,  margin, IMG_SIZE - margin)
            cx_blue = np.clip(cx_blue + vx_blue, margin, IMG_SIZE - margin)
            cy_blue = np.clip(cy_blue + vy_blue, margin, IMG_SIZE - margin)

    return torch.from_numpy(clips)   # [N,T,C,H,W]


# ---------------------------------------------------------------------
# GRU model
# ---------------------------------------------------------------------

class FutureGRU(nn.Module):
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
        out, _ = self.gru(z_hist)
        last = out[:, -1, :]
        return self.fc(last)


# ---------------------------------------------------------------------
# Centroid + relation utilities
# ---------------------------------------------------------------------

def compute_centroid(mask):
    mask = mask.astype(np.float32)
    if mask.sum() <= 0:
        h, w = mask.shape
        return (w / 2.0, h / 2.0)
    ys, xs = np.nonzero(mask > 0.5)
    if len(xs) == 0:
        h, w = mask.shape
        return (w / 2.0, h / 2.0)
    return xs.mean(), ys.mean()


def centroid_errors(true_img, pred_img):
    true_np = true_img.detach().cpu().numpy()
    pred_np = pred_img.detach().cpu().numpy()

    r_true = true_np[0]
    r_pred = pred_np[0]
    b_true = true_np[2]
    b_pred = pred_np[2]

    cx_rt, cy_rt = compute_centroid(r_true)
    cx_rp, cy_rp = compute_centroid(r_pred)
    cx_bt, cy_bt = compute_centroid(b_true)
    cx_bp, cy_bp = compute_centroid(b_pred)

    err_red  = math.hypot(cx_rt - cx_rp, cy_rt - cy_rp)
    err_blue = math.hypot(cx_bt - cx_bp, cy_bt - cy_bp)
    return err_red, err_blue


def relation_from_img(img_tensor, overlap_margin=2.0):
    """
    Map a decoded frame -> one of 5 coarse relation labels.

    Strategy:
      - compute centroids of red and blue
      - dx = red.x - blue.x, dy = red.y - blue.y
      - if both |dx|,|dy| <= overlap_margin -> 'overlap'
      - else if |dx| > |dy| -> left/right
      - else -> above/below
    """
    x = img_tensor.detach().cpu().numpy()
    r = x[0]
    b = x[2]

    cx_r, cy_r = compute_centroid(r)
    cx_b, cy_b = compute_centroid(b)

    dx = cx_r - cx_b
    dy = cy_r - cy_b

    if abs(dx) <= overlap_margin and abs(dy) <= overlap_margin:
        return 4  # overlap

    if abs(dx) > abs(dy):
        return 0 if dx < 0 else 1   # left_of / right_of
    else:
        return 2 if dy < 0 else 3   # above / below


def tensor_to_rgb(img_tensor):
    x = img_tensor.detach().cpu().numpy()
    r = np.clip(x[0], 0.0, 1.0)
    b = np.clip(x[2], 0.0, 1.0)
    h, w = r.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[..., 0] = r
    rgb[..., 2] = b
    return rgb


# ---------------------------------------------------------------------
# Encode clips with AE
# ---------------------------------------------------------------------

@torch.no_grad()
def encode_clips(ae, clips):
    N, T, C, H, W = clips.shape
    clips_flat = clips.view(N * T, C, H, W).to(DEVICE)
    z_flat = ae.encode(clips_flat)
    D = z_flat.shape[-1]
    return z_flat.view(N, T, D)


# ---------------------------------------------------------------------
# GRU training
# ---------------------------------------------------------------------

def train_gru(ae, clips):
    N = clips.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)
    n_train = int(TRAIN_FRAC * N)
    train_idx = idx[:n_train]
    test_idx  = idx[n_train:]

    clips_train = clips[train_idx]
    clips_test  = clips[test_idx]

    with torch.no_grad():
        z_train = encode_clips(ae, clips_train)
        z_test  = encode_clips(ae, clips_test)

    N_tr, T, D = z_train.shape
    print(f"[metrics] Encoded latents: train={N_tr}, test={z_test.shape[0]}, D={D}")

    gru = FutureGRU(D, hidden=256, num_layers=1).to(DEVICE)
    opt = torch.optim.Adam(gru.parameters(), lr=LR)

    def batch_iter(z_all):
        N_all = z_all.shape[0]
        idx_all = np.arange(N_all)
        np.random.shuffle(idx_all)
        for i in range(0, N_all, BATCH_SIZE):
            j = idx_all[i:i+BATCH_SIZE]
            yield z_all[j]

    for epoch in range(1, EPOCHS + 1):
        gru.train()
        train_losses = []
        for z_seq in batch_iter(z_train):
            z_seq = z_seq.to(DEVICE)
            z_hist = z_seq[:, :SEQ_LEN-1, :]
            z_true = z_seq[:, SEQ_LEN-1, :]

            z_pred = gru(z_hist)
            loss = F.mse_loss(z_pred, z_true)

            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(loss.item())


        gru.eval()
        with torch.no_grad():
            z_hist_te = z_test[:, :SEQ_LEN-1, :].to(DEVICE)
            z_true_te = z_test[:, SEQ_LEN-1, :].to(DEVICE)
            z_pred_te = gru(z_hist_te)
            test_loss = F.mse_loss(z_pred_te, z_true_te).item()

        print(f"[metrics] Epoch {epoch:2d}/{EPOCHS} | "
              f"train MSE={np.mean(train_losses):.6f} | "
              f"test  MSE={test_loss:.6f}")

    return gru, clips_test, z_test


# ---------------------------------------------------------------------
# Relation accuracy on the whole test set
# ---------------------------------------------------------------------

@torch.no_grad()
def relation_metrics(ae, gru, z_test):
    N_te = z_test.shape[0]

    z_hist   = z_test[:, :SEQ_LEN-1, :].to(DEVICE)
    z_true_T = z_test[:, SEQ_LEN-1, :].to(DEVICE)
    z_pred_T = gru(z_hist)

    # Baseline: copy last observed frame
    z_last = z_hist[:, -1, :]

    # Decode all three
    imgs_true = ae.decode(z_true_T)
    imgs_pred = ae.decode(z_pred_T)
    imgs_last = ae.decode(z_last)

    labels_true = []
    labels_gru  = []
    labels_last = []

    for i in range(N_te):
        lt = relation_from_img(imgs_true[i])
        lg = relation_from_img(imgs_pred[i])
        ll = relation_from_img(imgs_last[i])

        labels_true.append(lt)
        labels_gru.append(lg)
        labels_last.append(ll)

    labels_true = np.array(labels_true)
    labels_gru  = np.array(labels_gru)
    labels_last = np.array(labels_last)

    acc_gru  = (labels_gru == labels_true).mean()
    acc_last = (labels_last == labels_true).mean()

    print("\n[relations] Relation accuracy on test set:")
    print(f"  Baseline (copy last frame): {acc_last*100:.2f}%")
    print(f"  GRU predicted future:       {acc_gru*100:.2f}%")

    # Optional tiny confusion for GRU
    num_classes = len(REL_NAMES)
    conf = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(labels_true, labels_gru):
        conf[t, p] += 1

    print("\n[relations] Confusion matrix (rows = true, cols = GRU pred):")
    header = "         " + " ".join(f"{name[:3]:>5}" for name in REL_NAMES)
    print(header)
    for i in range(num_classes):
        row_str = f"{REL_NAMES[i][:7]:>7}:"
        for j in range(num_classes):
            row_str += f"{conf[i, j]:5d}"
        print(row_str)


# ---------------------------------------------------------------------
# Visualization (uses only a subset of test sequences)
# ---------------------------------------------------------------------

@torch.no_grad()
def visualize_and_metrics(ae, gru, clips_test, z_test):
    N_te = clips_test.shape[0]
    rng = np.random.default_rng(123)
    N_ROWS = 8
    indices = rng.choice(N_te, size=N_ROWS, replace=False)

    z_seq   = z_test[indices].to(DEVICE)
    clips_s = clips_test[indices]

    z_hist    = z_seq[:, :SEQ_LEN-1, :]
    z_true_T  = z_seq[:, SEQ_LEN-1, :]
    z_pred_T  = gru(z_hist)

    imgs_hist = ae.decode(z_hist.reshape(-1, z_hist.shape[-1]))
    imgs_hist = imgs_hist.view(N_ROWS, SEQ_LEN-1, NUM_CH, IMG_SIZE, IMG_SIZE)
    imgs_true = ae.decode(z_true_T)
    imgs_pred = ae.decode(z_pred_T)

    err_red_list  = []
    err_blue_list = []
    diffs = []
    diff_max = 0.0

    for r in range(N_ROWS):
        er, eb = centroid_errors(imgs_true[r], imgs_pred[r])
        err_red_list.append(er)
        err_blue_list.append(eb)

        diff = (imgs_pred[r] - imgs_true[r]).abs().mean(dim=0)
        d_np = diff.detach().cpu().numpy()
        diffs.append(d_np)
        diff_max = max(diff_max, d_np.max())

    print("\n[metrics] Per-row centroid errors (pixels):")
    for i, (er, eb) in enumerate(zip(err_red_list, err_blue_list)):
        print(f"  row {i:2d}: red={er:.2f} px, blue={eb:.2f} px")

    er_arr = np.array(err_red_list)
    eb_arr = np.array(err_blue_list)
    print("\n[metrics] Global centroid error stats:")
    print(f"  Red  mean={er_arr.mean():.3f}, std={er_arr.std():.3f}, "
          f"min={er_arr.min():.3f}, max={er_arr.max():.3f}")
    print(f"  Blue mean={eb_arr.mean():.3f}, std={eb_arr.std():.3f}, "
          f"min={eb_arr.min():.3f}, max={eb_arr.max():.3f}")

    # ---- grid: 7 columns: t=1..4, True T, GRU, |GRU-True| ----
    fig, axes = plt.subplots(
        nrows=N_ROWS,
        ncols=7,
        figsize=(7 * 1.6, N_ROWS * 1.6),
        dpi=120,
    )
    fig.suptitle("Latent dynamics (basic32): t=1..4, True T, GRU, |GRU-True|",
                 fontsize=14)

    col_titles = ["t=1", "t=2", "t=3", "t=4", "True T", "GRU", "|GRU-True|"]

    for row in range(N_ROWS):
        for col in range(7):
            ax = axes[row, col]
            ax.axis("off")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10)

            if col < 4:
                rgb = tensor_to_rgb(imgs_hist[row, col])
                ax.imshow(rgb)
            elif col == 4:
                rgb = tensor_to_rgb(imgs_true[row])
                ax.imshow(rgb)
            elif col == 5:
                rgb = tensor_to_rgb(imgs_pred[row])
                ax.imshow(rgb)
            else:
                im = ax.imshow(
                    diffs[row],
                    cmap="magma",
                    vmin=0.0,
                    vmax=max(1e-3, diff_max),
                )

        axes[row, 0].set_ylabel(
            f"R:{err_red_list[row]:.1f}px\nB:{err_blue_list[row]:.1f}px",
            fontsize=7,
            rotation=0,
            labelpad=24,
            ha="right",
            va="center",
        )

    cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    fig.colorbar(im, cax=cax)

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    os.makedirs(os.path.dirname(OUT_GRID), exist_ok=True)
    plt.savefig(OUT_GRID)
    print(f"[metrics] Saved grid -> {OUT_GRID}")

    # Also run relation-level metrics on the entire test set
    relation_metrics(ae, gru, z_test)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("[metrics] Using device:", DEVICE)
    ae = load_ae()
    clips = generate_temporal_dataset()
    gru, clips_test, z_test = train_gru(ae, clips)
    visualize_and_metrics(ae, gru, clips_test, z_test)


if __name__ == "__main__":
    main()
