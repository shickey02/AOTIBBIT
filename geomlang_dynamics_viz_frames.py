#!/usr/bin/env python3
# geomlang_dynamics_viz_frames.py
#
# Visualize latent dynamics on the edges model:
#   - Decode t=1..4 interpolant frames
#   - Decode true final frame (T)
#   - Decode nearest-manifold frame to GRU prediction
#   - Decode raw GRU-predicted latent
#   - Show diff heatmap |Pred - True|
#
# Output: outputs_edges/dynamics_future_grid_nn.png

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from geomlang_edges_relscale import (
    ConvAutoencoderEdges,
    LATENT_DIM,
    IMG_SIZE as TRAIN_IMG_SIZE,
)

# ---------------------------------------------------------------------
# Config / paths
# ---------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LATENTS_PATH   = os.path.join("outputs_edges", "latents_dump.npz")
AE_CKPT_PATH   = os.path.join("outputs_edges", "conv_autoencoder_edges.pt")
GRU_CKPT_PATH  = os.path.join("outputs_edges", "future_gru_dynamics.pt")
OUT_PATH       = os.path.join("outputs_edges", "dynamics_future_grid_nn.png")

SEQ_LEN        = 5     # t=0..4 (we’ll use 0..3 as history, 4 as future)
N_ROWS         = 8     # number of sequences to visualize

IMG_SIZE       = TRAIN_IMG_SIZE  # should be 64


# ---------------------------------------------------------------------
# Autoencoder loader / decoder
# ---------------------------------------------------------------------

def load_ae():
    """
    Load ConvAutoencoderEdges from outputs_edges/conv_autoencoder_edges.pt
    as saved by geomlang_edges_relscale.py.
    """
    if not os.path.exists(AE_CKPT_PATH):
        raise FileNotFoundError(
            f"AE checkpoint not found at {AE_CKPT_PATH}.\n"
            f"Run geomlang_edges_relscale.py first to train and save it."
        )

    print(f"[viz] Loading AE from: {AE_CKPT_PATH}")
    ckpt = torch.load(AE_CKPT_PATH, map_location=DEVICE)

    cfg = ckpt.get("config", {})
    latent_dim = cfg.get("latent_dim", LATENT_DIM)

    ae = ConvAutoencoderEdges(latent_dim=latent_dim)
    state = ckpt.get("ae_state_dict", ckpt)
    ae.load_state_dict(state, strict=False)

    ae.to(DEVICE)
    ae.eval()
    return ae


def tensor_to_rgb(img_tensor):
    """
    Input: [3,H,W] with channels:
        0: red fill
        1: blue fill
        2: edges (white)
    Output: [H,W,3] float in [0,1] for plotting.

    - red object: pure red
    - blue object: pure blue
    - edges: white overlay
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
# Future-GRU dynamics model
# ---------------------------------------------------------------------

class FutureGRU(nn.Module):
    """
    Simple GRU that takes a sequence of latents z[0..T_hist-1] and predicts z_T.
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


def load_future_gru(dim_latent):
    if not os.path.exists(GRU_CKPT_PATH):
        raise FileNotFoundError(
            f"Future GRU checkpoint not found at {GRU_CKPT_PATH}.\n"
            f"Run geomlang_dynamics.py first to train and save it."
        )

    ckpt = torch.load(GRU_CKPT_PATH, map_location=DEVICE)
    cfg = ckpt.get("config", {})
    hidden = cfg.get("hidden_dim", cfg.get("hidden", 256))
    num_layers = cfg.get("num_layers", 1)

    model = FutureGRU(dim_latent, hidden=hidden, num_layers=num_layers)
    # choose appropriate state key
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
    print(f"[viz] Loaded FutureGRU from {GRU_CKPT_PATH}")
    return model


# ---------------------------------------------------------------------
# Latent utilities
# ---------------------------------------------------------------------

def load_latents():
    if not os.path.exists(LATENTS_PATH):
        raise FileNotFoundError(
            f"Latents file not found at {LATENTS_PATH}.\n"
            f"Run geomlang_edges_relscale.py first to generate it."
        )
    data = np.load(LATENTS_PATH)
    z = torch.from_numpy(data["z"]).float().to(DEVICE)       # [N,D]
    rel = torch.from_numpy(data["rel"]).long().to(DEVICE)    # [N]
    scale = torch.from_numpy(data["scale"]).long().to(DEVICE)# [N]
    print(f"[viz] Loaded latents: N={z.shape[0]}, D={z.shape[1]}")
    return z, rel, scale


@torch.no_grad()
def build_interpolant_seq(z_all, idx_start, idx_end, T=5):
    """
    Linear latent interpolation between two dataset points.
    Returns sequence [T,D].
    """
    z0 = z_all[idx_start]
    zT = z_all[idx_end]
    alphas = torch.linspace(0.0, 1.0, T, device=z_all.device)  # [T]
    seq = (1.0 - alphas.unsqueeze(-1)) * z0.unsqueeze(0) + \
          alphas.unsqueeze(-1) * zT.unsqueeze(0)
    return seq  # [T,D]


@torch.no_grad()
def nearest_manifold_latent(z_pred, z_all):
    """
    Given predicted latent z_pred [D], find the closest dataset latent.
    """
    z_cpu = z_all.detach().cpu()
    zp = z_pred.detach().cpu().unsqueeze(0)  # [1,D]
    d2 = torch.sum((z_cpu - zp) ** 2, dim=1)  # [N]
    idx = torch.argmin(d2).item()
    return z_all[idx], idx


def decode_latents(ae, z_batch):
    """
    z_batch: [..., D] → images [..., 3, IMG_SIZE, IMG_SIZE]
    Uses the loaded autoencoder.
    """
    orig_shape = z_batch.shape[:-1]
    z_flat = z_batch.reshape(-1, z_batch.shape[-1]).to(DEVICE)
    with torch.no_grad():
        imgs = ae.decode(z_flat)
    imgs = imgs.view(*orig_shape, 3, IMG_SIZE, IMG_SIZE)
    return imgs


# ---------------------------------------------------------------------
# Main visualization
# ---------------------------------------------------------------------

@torch.no_grad()
def main():
    print("")
    print(f"[viz] Using device: {DEVICE}")

    # --- load models + latents ---
    z_all, rel_all, scale_all = load_latents()
    N, D = z_all.shape

    ae = load_ae()
    future_gru = load_future_gru(D)

    # --- sample random pairs to visualize ---
    rng = np.random.default_rng(42)
    indices = rng.choice(N, size=2 * N_ROWS, replace=False)  # 2 per row

    fig, axes = plt.subplots(
        nrows=N_ROWS,
        ncols=8,
        figsize=(8 * 1.5, N_ROWS * 1.5),
        dpi=120,
    )
    fig.suptitle(
        "Latent dynamics: t=1..4, True T, NN T, Pred T, |Pred−True|",
        fontsize=14
    )

    col_titles = ["t=1", "t=2", "t=3", "t=4",
                  "True T", "NN T", "Pred T", "|Pred−True|"]

    # We'll add colorbar for diff in the top-right heatmap once
    diff_im_for_cbar = None

    for row in range(N_ROWS):
        i_start = indices[2 * row]
        i_end   = indices[2 * row + 1]

        # Build interpolant sequence in latent space
        seq = build_interpolant_seq(z_all, i_start, i_end, T=SEQ_LEN)  # [T,D]
        z_hist   = seq[:SEQ_LEN - 1]   # [4,D]
        z_true_T = seq[SEQ_LEN - 1]    # [D]

        # GRU prediction
        z_pred_T = future_gru(z_hist.unsqueeze(0)).squeeze(0)  # [D]

        # Nearest-manifold latent
        z_nn_T, idx_nn = nearest_manifold_latent(z_pred_T, z_all)

        # Decode all relevant latents
        imgs_hist = decode_latents(ae, z_hist.unsqueeze(0)).squeeze(0)  # [4,3,H,W]
        img_true  = decode_latents(ae, z_true_T.unsqueeze(0)).squeeze(0)  # [3,H,W]
        img_pred  = decode_latents(ae, z_pred_T.unsqueeze(0)).squeeze(0)  # [3,H,W]
        img_nn    = decode_latents(ae, z_nn_T.unsqueeze(0)).squeeze(0)    # [3,H,W]

        # Diff heatmap (per-pixel mean over channels)
        diff = (img_pred - img_true).abs().mean(dim=0)  # [H,W]
        diff_np = diff.detach().cpu().numpy()

        for col in range(8):
            ax = axes[row, col]
            ax.axis("off")

            if row == 0:
                ax.set_title(col_titles[col], fontsize=10)

            if col < 4:
                # history frames
                rgb = tensor_to_rgb(imgs_hist[col])
                ax.imshow(rgb)
            elif col == 4:
                # true final frame
                rgb = tensor_to_rgb(img_true)
                ax.imshow(rgb)
            elif col == 5:
                # nearest-manifold frame
                rgb = tensor_to_rgb(img_nn)
                ax.imshow(rgb)
            elif col == 6:
                # raw predicted frame
                rgb = tensor_to_rgb(img_pred)
                ax.imshow(rgb)
            else:
                # diff heatmap
                im = ax.imshow(
                    diff_np,
                    cmap="inferno",
                    vmin=0.0,
                    vmax=max(1e-3, diff_np.max()),
                )
                if diff_im_for_cbar is None:
                    diff_im_for_cbar = im

    # single colorbar for diff, on the right
    if diff_im_for_cbar is not None:
        cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
        fig.colorbar(diff_im_for_cbar, cax=cax)

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.savefig(OUT_PATH)
    print(f"[viz] Saved grid -> {OUT_PATH}")


if __name__ == "__main__":
    main()
