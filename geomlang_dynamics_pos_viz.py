#!/usr/bin/env python3
# geomlang_dynamics_pos_viz.py
#
# Visualize latent dynamics with *explicit 2D object positions*:
#   - Build latent interpolant sequences z_1..z_4..z_T
#   - Predict z_pred_T with the trained future-GRU
#   - Decode all frames with the autoencoder
#   - Compute centers-of-mass for red & blue objects in each frame
#   - Overlay positions on the images:
#       * t=1..4: true centers
#       * True T: true centers
#       * Pred T: predicted centers (circles) + true centers (crosses)
#   - Also show a |Pred-True| pixelwise diff heatmap
#
# Output: outputs_edges/dynamics_future_grid_pos.png

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LATENTS_PATH = os.path.join("outputs_edges", "latents_dump.npz")
AE_CKPT_CANDIDATES = [
    os.path.join("outputs_edges", "conv_autoencoder_edges.pt"),
    os.path.join("outputs",      "conv_autoencoder_edges.pt"),
]
GRU_CKPT_PATH = os.path.join("outputs_edges", "future_gru_dynamics.pt")
OUT_PATH      = os.path.join("outputs_edges", "dynamics_future_grid_pos.png")

SEQ_LEN = 5      # t=1..5 (we'll use 1..4 as history, 5 as future)
N_ROWS  = 8      # number of sequences to visualize
IMG_SIZE = 64    # images are 64x64 in this project


# ---------------------------------------------------------------------
#  Autoencoder (must match conv_autoencoder_edges) + loader
# ---------------------------------------------------------------------

class ConvAutoencoderEdges(nn.Module):
    """
    This matches the conv autoencoder you used for the edges experiment:
      - Input:  [B,3,64,64]
          ch0: red fill
          ch1: blue fill
          ch2: edges
      - Latent: [B, D]
      - Output: [B,3,64,64] in [0,1]
    """

    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        # encoder: [3,64,64] -> [latent_dim]
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),   # 32 x 32 x 32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),  # 64 x 16 x 16
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), # 128 x 8 x 8
            nn.ReLU(inplace=True),
        )
        self.enc_fc = nn.Linear(128 * 8 * 8, latent_dim)

        # decoder: [latent_dim] -> [3,64,64]
        self.dec_fc = nn.Linear(latent_dim, 128 * 8 * 8)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 64 x 16 x 16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 32 x 32 x 32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),    # 3 x 64 x 64
            nn.Sigmoid(),                          # data in [0,1]
        )

    def encode(self, x):
        h = self.enc(x)
        h = h.view(h.size(0), -1)
        z = self.enc_fc(h)
        return z

    def decode(self, z):
        h = self.dec_fc(z)
        h = h.view(z.size(0), 128, 8, 8)
        x_rec = self.dec(h)
        return x_rec

    def forward(self, x):
        return self.decode(self.encode(x))


def find_existing_ckpt(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Could not find any autoencoder checkpoint in {candidates}. "
        f"Move or copy conv_autoencoder_edges.pt into one of those paths."
    )


def load_ae():
    path = find_existing_ckpt(AE_CKPT_CANDIDATES)
    print(f"[pos-viz] Loading AE from: {path}")
    ckpt = torch.load(path, map_location=DEVICE)

    if isinstance(ckpt, nn.Module):
        ae = ckpt
        latent_dim = getattr(ae, "latent_dim", 128)
    else:
        config = ckpt.get("config", {})
        latent_dim = config.get("latent_dim", 128)
        ae = ConvAutoencoderEdges(latent_dim=latent_dim)
        state = ckpt.get("ae", ckpt.get("model", ckpt.get("model_state_dict", ckpt)))
        ae.load_state_dict(state, strict=False)

    ae.to(DEVICE)
    ae.eval()
    print(f"[pos-viz] AE latent_dim = {latent_dim}")
    return ae, latent_dim


# ---------------------------------------------------------------------
#  GRU dynamics model + loader
# ---------------------------------------------------------------------

class FutureGRU(nn.Module):
    """GRU that takes z[0..T-2] and predicts z_T."""

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


def load_future_gru(dim_latent):
    print(f"[pos-viz] Loading FutureGRU from: {GRU_CKPT_PATH}")
    ckpt = torch.load(GRU_CKPT_PATH, map_location=DEVICE)
    cfg = ckpt.get("config", {})
    hidden = cfg.get("hidden_dim", cfg.get("hidden", 256))
    num_layers = cfg.get("num_layers", 1)

    model = FutureGRU(dim_latent, hidden=hidden, num_layers=num_layers)
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "model" in ckpt:
        state = ckpt["model"]
    elif "gru" in ckpt:
        state = ckpt["gru"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=False)
    model.to(DEVICE)
    model.eval()
    return model


# ---------------------------------------------------------------------
#  Latent utilities
# ---------------------------------------------------------------------

def load_latents():
    data = np.load(LATENTS_PATH)
    z = torch.from_numpy(data["z"]).float().to(DEVICE)       # [N,D]
    rel = torch.from_numpy(data["rel"]).long().to(DEVICE)    # [N] (unused here)
    scale = torch.from_numpy(data["scale"]).long().to(DEVICE)# [N] (unused here)
    print(f"[pos-viz] Loaded latents: N={z.shape[0]}, D={z.shape[1]}")
    return z, rel, scale


@torch.no_grad()
def build_interpolant_seq(z_all, idx_start, idx_end, T=5):
    """
    Linear latent interpolation between two dataset points.
    Returns [T,D].
    """
    z0 = z_all[idx_start]
    zT = z_all[idx_end]
    alphas = torch.linspace(0.0, 1.0, T, device=z_all.device)  # [T]
    seq = (1.0 - alphas.unsqueeze(-1)) * z0.unsqueeze(0) + \
          alphas.unsqueeze(-1) * zT.unsqueeze(0)
    return seq  # [T,D]


@torch.no_grad()
def decode_latents(ae, z_batch):
    """
    z_batch: [..., D] -> imgs [..., 3, 64, 64]
    """
    orig_shape = z_batch.shape[:-1]
    z_flat = z_batch.reshape(-1, z_batch.shape[-1])
    imgs = ae.decode(z_flat)
    imgs = imgs.view(*orig_shape, 3, IMG_SIZE, IMG_SIZE)
    return imgs


def tensor_to_rgb(img_tensor):
    """
    Input: [3,H,W] with channels:
        0: red fill
        1: blue fill
        2: edges
    Output: [H,W,3] float in [0,1] for plotting:

      - red object = red
      - blue object = blue
      - edges = white overlay
    """
    x = img_tensor.detach().cpu().numpy()
    r = x[0]
    b = x[1]
    e = x[2]

    H, W = r.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)

    # red object
    rgb[..., 0] += np.clip(r, 0.0, 1.0)
    # blue object
    rgb[..., 2] += np.clip(b, 0.0, 1.0)
    # edges as white overlay
    edge = np.clip(e, 0.0, 1.0)
    rgb[..., 0] = np.clip(rgb[..., 0] + 0.7 * edge, 0.0, 1.0)
    rgb[..., 1] = np.clip(rgb[..., 1] + 0.7 * edge, 0.0, 1.0)
    rgb[..., 2] = np.clip(rgb[..., 2] + 0.7 * edge, 0.0, 1.0)

    return rgb


# ---------------------------------------------------------------------
#  Center-of-mass helpers
# ---------------------------------------------------------------------

def channel_com(img_ch):
    """
    Center of mass of a single channel [H,W] tensor in image coordinates.
    Returns (cx, cy) in pixel coordinates.
    """
    arr = img_ch.detach().cpu().numpy().astype(np.float64)
    arr = arr - arr.min()
    maxv = arr.max()
    if maxv > 0:
        arr /= maxv

    total = arr.sum()
    if total < 1e-6:
        # fallback: center of image if channel is empty
        return IMG_SIZE / 2.0, IMG_SIZE / 2.0

    ys, xs = np.indices(arr.shape)
    cx = float((xs * arr).sum() / total)
    cy = float((ys * arr).sum() / total)
    return cx, cy


def compute_positions(img_tensor):
    """
    img_tensor: [3,H,W]
    Returns:
      (cx_r, cy_r), (cx_b, cy_b)
    using channels 0 (red fill) and 1 (blue fill).
    """
    red_ch = img_tensor[0]
    blue_ch = img_tensor[1]
    cx_r, cy_r = channel_com(red_ch)
    cx_b, cy_b = channel_com(blue_ch)
    return (cx_r, cy_r), (cx_b, cy_b)


# ---------------------------------------------------------------------
#  Main visualization
# ---------------------------------------------------------------------

@torch.no_grad()
def main():
    print(">>")
    print(f"[pos-viz] Using device: {DEVICE}")

    # load base data/models
    z_all, rel_all, scale_all = load_latents()
    N, D = z_all.shape

    ae, latent_dim = load_ae()
    assert latent_dim == D, f"AE latent_dim {latent_dim} != z dim {D}"

    future_gru = load_future_gru(D)

    # sample random pairs for interpolation
    rng = np.random.default_rng(42)
    indices = rng.choice(N, size=2 * N_ROWS, replace=False)  # 2 endpoints per row

    n_cols = 7  # t1, t2, t3, t4, True T, Pred T, Diff
    fig, axes = plt.subplots(
        nrows=N_ROWS,
        ncols=n_cols,
        figsize=(n_cols * 1.5, N_ROWS * 1.5),
        dpi=120,
    )
    fig.suptitle("Latent dynamics (positions): t=1..4, True T, Pred T, |Pred−True|",
                 fontsize=14)

    col_titles = ["t=1", "t=2", "t=3", "t=4",
                  "True T", "Pred T", "|Pred−True|"]

    for row in range(N_ROWS):
        i_start = indices[2 * row]
        i_end   = indices[2 * row + 1]

        # latent interpolation
        seq = build_interpolant_seq(z_all, i_start, i_end, T=SEQ_LEN)   # [T,D]
        z_hist = seq[:SEQ_LEN - 1]   # [4,D]
        z_true_T = seq[SEQ_LEN - 1]  # [D]

        # GRU prediction
        z_pred_T = future_gru(z_hist.unsqueeze(0)).squeeze(0)  # [D]

        # decode all relevant latents
        imgs_hist = decode_latents(ae, z_hist.unsqueeze(0)).squeeze(0)  # [4,3,64,64]
        img_true  = decode_latents(ae, z_true_T.unsqueeze(0)).squeeze(0)  # [3,64,64]
        img_pred  = decode_latents(ae, z_pred_T.unsqueeze(0)).squeeze(0)  # [3,64,64]

        # diff heatmap
        diff = (img_pred - img_true).abs().mean(dim=0)  # [64,64]
        diff_np = diff.detach().cpu().numpy()

        # true vs predicted positions at T
        (cx_r_true, cy_r_true), (cx_b_true, cy_b_true) = compute_positions(img_true)
        (cx_r_pred, cy_r_pred), (cx_b_pred, cy_b_pred) = compute_positions(img_pred)

        # draw row
        for col in range(n_cols):
            ax = axes[row, col]
            ax.axis("off")

            if row == 0:
                ax.set_title(col_titles[col], fontsize=10)

            if col < 4:
                # history frames t=1..4
                img = imgs_hist[col]
                (cx_r, cy_r), (cx_b, cy_b) = compute_positions(img)
                rgb = tensor_to_rgb(img)
                ax.imshow(rgb, origin="upper")
                # overlay centers
                ax.scatter([cx_r], [cy_r], s=20, edgecolors="red",
                           facecolors="none", linewidths=1.2)
                ax.scatter([cx_b], [cy_b], s=20, edgecolors="blue",
                           facecolors="none", linewidths=1.2)

            elif col == 4:
                # True T
                rgb = tensor_to_rgb(img_true)
                ax.imshow(rgb, origin="upper")
                ax.scatter([cx_r_true], [cy_r_true], s=25, edgecolors="red",
                           facecolors="none", linewidths=1.5)
                ax.scatter([cx_b_true], [cy_b_true], s=25, edgecolors="blue",
                           facecolors="none", linewidths=1.5)

            elif col == 5:
                # Pred T with *both* true (cross) and predicted (circle) centers
                rgb = tensor_to_rgb(img_pred)
                ax.imshow(rgb, origin="upper")

                # predicted centers: circles
                ax.scatter([cx_r_pred], [cy_r_pred], s=30, edgecolors="red",
                           facecolors="none", linewidths=1.5)
                ax.scatter([cx_b_pred], [cy_b_pred], s=30, edgecolors="blue",
                           facecolors="none", linewidths=1.5)

                # true centers: crosses
                ax.scatter([cx_r_true], [cy_r_true], s=30, c="red", marker="x",
                           linewidths=1.5)
                ax.scatter([cx_b_true], [cy_b_true], s=30, c="blue", marker="x",
                           linewidths=1.5)

            else:
                # diff heatmap
                im = ax.imshow(diff_np, cmap="inferno",
                               vmin=0.0, vmax=max(1e-3, diff_np.max()))
                if row == 0:
                    # put a single colorbar on the right
                    cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
                    fig.colorbar(im, cax=cax)

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.savefig(OUT_PATH)
    print(f"[pos-viz] Saved grid -> {OUT_PATH}")


if __name__ == "__main__":
    main()
