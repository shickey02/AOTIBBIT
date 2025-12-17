#!/usr/bin/env python3
# geomlang_dynamics_manifold.py
#
# Train a future-prediction GRU on *manifold-projected* latent sequences
# and visualize its behaviour on canonical left/right and above/below
# relation transitions.
#
# Assumes you already trained the edges autoencoder + heads using
# geomlang_edges_relscale.py and have:
#   - outputs_edges/conv_autoencoder_edges.pt
#   - outputs_edges/latents_dump.npz   (contains z, rel, scale)
#
# Output:
#   - outputs_edges/future_gru_manifold.pt
#   - outputs_edges/dynamics_future_grid_manifold.png

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LATENTS_PATH = os.path.join("outputs_edges", "latents_dump.npz")
AE_CKPT_PATH = os.path.join("outputs_edges", "conv_autoencoder_edges.pt")
GRU_CKPT_PATH = os.path.join("outputs_edges", "future_gru_manifold.pt")
OUT_GRID_PATH = os.path.join("outputs_edges", "dynamics_future_grid_manifold.png")

IMG_SIZE = 64   # matches edges training; change if your AE used a different size

SEQ_LEN = 5         # t=1..5  (we use 1..4 as history, 5 as future)
N_SEQ = 4000        # number of training sequences
N_EPOCHS = 20
BATCH_SIZE = 64
LR = 1e-3
N_ROWS_VIZ = 8      # sequences to visualize in the grid


# ------------------------------
#  Autoencoder architecture
# ------------------------------

class ConvAutoencoderEdges(nn.Module):
    """
    Must match the architecture used when training conv_autoencoder_edges.pt.
    This is the same small conv AE we've been using for viz.
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


def load_ae(z_dim: int):
    """
    Load the trained AE from checkpoint; z_dim is taken from latents_dump.npz.
    """
    if not os.path.exists(AE_CKPT_PATH):
        raise FileNotFoundError(
            f"Missing autoencoder checkpoint at {AE_CKPT_PATH}.\n"
            f"Run geomlang_edges_relscale.py first to train it."
        )

    ckpt = torch.load(AE_CKPT_PATH, map_location=DEVICE)

    ae = ConvAutoencoderEdges(latent_dim=z_dim)
    # checkpoint may store various keys; try common ones
    if isinstance(ckpt, dict):
        state = ckpt.get("ae", ckpt.get("model", ckpt.get("model_state_dict", ckpt)))
    else:
        # someone saved the full module
        ae = ckpt

    if isinstance(ckpt, dict):
        ae.load_state_dict(state, strict=False)

    ae.to(DEVICE).eval()
    print(f"[manifold] Loaded AE from {AE_CKPT_PATH} (latent_dim={z_dim})")
    return ae


def tensor_to_rgb(img_tensor):
    """
    Convert [3,H,W] model output into an RGB image for plotting.

    Channel semantics:
        0: red fill
        1: blue fill
        2: edges

    We render:
        - red: pure red
        - blue: pure blue
        - edges: white overlay
    """
    x = img_tensor.detach().cpu().numpy()
    r = np.clip(x[0], 0.0, 1.0)
    b = np.clip(x[1], 0.0, 1.0)
    e = np.clip(x[2], 0.0, 1.0)

    H, W = r.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)

    rgb[..., 0] += r         # red object
    rgb[..., 2] += b         # blue object

    # edges as white overlay
    rgb[..., 0] = np.clip(rgb[..., 0] + 0.7 * e, 0.0, 1.0)
    rgb[..., 1] = np.clip(rgb[..., 1] + 0.7 * e, 0.0, 1.0)
    rgb[..., 2] = np.clip(rgb[..., 2] + 0.7 * e, 0.0, 1.0)

    return rgb


# ------------------------------
#  Future GRU
# ------------------------------

class FutureGRU(nn.Module):
    """
    GRU over latent sequences: z[0..T-2] -> predict z_T.
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


def load_latents():
    data = np.load(LATENTS_PATH)
    z = torch.from_numpy(data["z"]).float().to(DEVICE)
    rel = torch.from_numpy(data["rel"]).long().to(DEVICE)
    scale = torch.from_numpy(data["scale"]).long().to(DEVICE)
    print(f"[manifold] Loaded latents: N={z.shape[0]}, D={z.shape[1]}")
    return z, rel, scale


# ------------------------------
#  Manifold helpers
# ------------------------------

def project_to_manifold(ae: ConvAutoencoderEdges, z: torch.Tensor, steps: int = 1):
    """
    Differentiable autoencoder-manifold projection:
        z -> decode -> encode -> z'
    Repeating pulls z back toward the AE's latent manifold.
    z can be [D] or [B,D].

    Gradients will flow w.r.t. z; AE weights are fixed because they are
    not in the optimizer.
    """
    if z.dim() == 1:
        z_in = z.unsqueeze(0)
        squeeze_back = True
    else:
        z_in = z
        squeeze_back = False

    for _ in range(steps):
        x = ae.decode(z_in)     # uses fixed AE weights
        z_in = ae.encode(x)     # still in graph, so dL/dz flows

    if squeeze_back:
        return z_in.squeeze(0)
    return z_in



@torch.no_grad()
def build_manifold_sequences(ae, z_all, num_seq=N_SEQ, T=SEQ_LEN):
    """
    Build num_seq sequences of length T in latent space by:
      - choosing random endpoints z0, zT from the dataset
      - linearly interpolating between them
      - projecting each intermediate step back to the AE manifold
    Returns: [num_seq, T, D]
    """
    N, D = z_all.shape
    seqs = torch.zeros(num_seq, T, D, device=DEVICE)

    alphas = torch.linspace(0.0, 1.0, T, device=DEVICE)

    for i in range(num_seq):
        i0 = np.random.randint(0, N)
        iT = np.random.randint(0, N)
        z0 = z_all[i0]
        zT = z_all[iT]

        z_steps = []
        for a in alphas:
            z = (1.0 - a) * z0 + a * zT
            z_proj = project_to_manifold(ae, z, steps=1)
            z_steps.append(z_proj)
        seqs[i] = torch.stack(z_steps, dim=0)

    return seqs


@torch.no_grad()
def nearest_manifold_latent(z_pred, z_all):
    """
    Given predicted latent z_pred [D], find the closest latent in z_all.
    """
    z_cpu = z_all.detach().cpu()
    zp = z_pred.detach().cpu().unsqueeze(0)  # [1,D]
    d2 = torch.sum((z_cpu - zp) ** 2, dim=1)  # [N]
    idx = torch.argmin(d2).item()
    return z_all[idx], idx


# ------------------------------
#  Training the GRU
# ------------------------------

def train_future_gru(ae, z_all):
    N, D = z_all.shape

    # Build manifold-projected sequences
    seqs = build_manifold_sequences(ae, z_all, num_seq=N_SEQ, T=SEQ_LEN)
    # Split train / test
    n_train = int(0.8 * N_SEQ)
    train_seqs = seqs[:n_train]
    test_seqs = seqs[n_train:]

    def make_loader(seqs):
        hist = seqs[:, :SEQ_LEN-1, :]      # [B,4,D]
        fut = seqs[:, SEQ_LEN-1, :]        # [B,D]
        ds = torch.utils.data.TensorDataset(hist, fut)
        return torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    train_loader = make_loader(train_seqs)
    test_loader = make_loader(test_seqs)

    model = FutureGRU(D, hidden=256, num_layers=1).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_count = 0

        for z_hist, z_true in train_loader:
            z_hist = z_hist.to(DEVICE)
            z_true = z_true.to(DEVICE)

            opt.zero_grad()
            z_pred_raw = model(z_hist)
            # project prediction back to manifold before measuring loss
            z_pred = project_to_manifold(ae, z_pred_raw, steps=1)
            loss = F.mse_loss(z_pred, z_true)
            loss.backward()
            opt.step()

            train_loss += loss.item() * z_hist.size(0)
            train_count += z_hist.size(0)

        model.eval()
        test_loss = 0.0
        test_count = 0
        with torch.no_grad():
            for z_hist, z_true in test_loader:
                z_hist = z_hist.to(DEVICE)
                z_true = z_true.to(DEVICE)
                z_pred_raw = model(z_hist)
                z_pred = project_to_manifold(ae, z_pred_raw, steps=1)
                loss = F.mse_loss(z_pred, z_true)
                test_loss += loss.item() * z_hist.size(0)
                test_count += z_hist.size(0)

        print(f"[manifold] Epoch {epoch:2d}/{N_EPOCHS} | "
              f"train MSE={train_loss/train_count:.6f} | "
              f"test MSE={test_loss/test_count:.6f}")

    # save checkpoint
    ckpt = {
        "config": {
            "latent_dim": D,
            "hidden_dim": 256,
            "num_layers": 1,
            "seq_len": SEQ_LEN,
        },
        "model_state_dict": model.state_dict(),
    }
    os.makedirs(os.path.dirname(GRU_CKPT_PATH), exist_ok=True)
    torch.save(ckpt, GRU_CKPT_PATH)
    print(f"[manifold] Saved GRU checkpoint -> {GRU_CKPT_PATH}")

    return model


# ------------------------------
#  Visualization grid
# ------------------------------

@torch.no_grad()
def decode_latents(ae, z_batch):
    """
    z_batch: [..., D] -> images [..., 3, H, W]
    """
    orig_shape = z_batch.shape[:-1]
    z_flat = z_batch.reshape(-1, z_batch.shape[-1])
    imgs = ae.decode(z_flat)
    imgs = imgs.view(*orig_shape, 3, IMG_SIZE, IMG_SIZE)
    return imgs


@torch.no_grad()
def build_rel_index(rel_all):
    """
    Build index lists for each relation label.
    """
    rel_index = {}
    for r in torch.unique(rel_all).tolist():
        rel_index[int(r)] = (rel_all == r).nonzero(as_tuple=True)[0].cpu().numpy()
    return rel_index


@torch.no_grad()
def make_viz_grid(ae, future_gru, z_all, rel_all):
    """
    Create a grid similar to the previous one, but using manifold-projected
    sequences and more interpretable relation transitions (left<->right,
    above<->below).
    """
    D = z_all.shape[1]
    rel_index = build_rel_index(rel_all)

    fig, axes = plt.subplots(
        nrows=N_ROWS_VIZ,
        ncols=8,
        figsize=(8 * 1.5, N_ROWS_VIZ * 1.5),
        dpi=120,
    )
    fig.suptitle("Latent dynamics (manifold): t=1..4, True T, NN T, Pred T, |Pred−True|",
                 fontsize=14)

    col_titles = ["t=1", "t=2", "t=3", "t=4", "True T", "NN T", "Pred T", "|Pred−True|"]

    # alternate between horizontal and vertical relation changes
    rel_pairs = [(0, 1), (2, 3)]  # (left_of,right_of), (above,below)

    for row in range(N_ROWS_VIZ):
        r0, r1 = rel_pairs[row % len(rel_pairs)]
        idx0_pool = rel_index.get(r0, None)
        idx1_pool = rel_index.get(r1, None)
        if idx0_pool is None or idx1_pool is None:
            # fallback: random indices if we don't have those relations
            i_start = np.random.randint(0, z_all.shape[0])
            i_end = np.random.randint(0, z_all.shape[0])
        else:
            i_start = int(np.random.choice(idx0_pool))
            i_end = int(np.random.choice(idx1_pool))

        z0 = z_all[i_start]
        zT = z_all[i_end]

        # build one manifold-projected sequence between them
        alphas = torch.linspace(0.0, 1.0, SEQ_LEN, device=DEVICE)
        z_steps = []
        for a in alphas:
            z = (1.0 - a) * z0 + a * zT
            z_proj = project_to_manifold(ae, z, steps=1)
            z_steps.append(z_proj)
        seq = torch.stack(z_steps, dim=0)     # [T,D]

        z_hist = seq[:SEQ_LEN-1]             # [4,D]
        z_true_T = seq[SEQ_LEN-1]            # [D]

        # model prediction
        z_pred_raw = future_gru(z_hist.unsqueeze(0)).squeeze(0)   # [D]
        z_pred = project_to_manifold(ae, z_pred_raw, steps=1)

        # nearest-neighbor latent
        z_nn, idx_nn = nearest_manifold_latent(z_pred, z_all)

        # decode frames
        imgs_hist = decode_latents(ae, z_hist.unsqueeze(0)).squeeze(0)  # [4,3,H,W]
        img_true = decode_latents(ae, z_true_T.unsqueeze(0)).squeeze(0)
        img_pred = decode_latents(ae, z_pred.unsqueeze(0)).squeeze(0)
        img_nn = decode_latents(ae, z_nn.unsqueeze(0)).squeeze(0)

        # diff heatmap
        diff = (img_pred - img_true).abs().mean(dim=0).detach().cpu().numpy()

        for col in range(8):
            ax = axes[row, col]
            ax.axis("off")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10)

            if col < 4:
                rgb = tensor_to_rgb(imgs_hist[col])
                ax.imshow(rgb)
            elif col == 4:  # true T
                ax.imshow(tensor_to_rgb(img_true))
            elif col == 5:  # NN T
                ax.imshow(tensor_to_rgb(img_nn))
            elif col == 6:  # Pred T
                ax.imshow(tensor_to_rgb(img_pred))
            else:
                im = ax.imshow(diff, cmap="inferno",
                               vmin=0.0, vmax=max(1e-3, diff.max()))
                if row == 0:
                    # single shared colorbar
                    cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
                    fig.colorbar(im, cax=cax)

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    os.makedirs(os.path.dirname(OUT_GRID_PATH), exist_ok=True)
    plt.savefig(OUT_GRID_PATH)
    print(f"[manifold] Saved viz grid -> {OUT_GRID_PATH}")


# ------------------------------
#  Main
# ------------------------------

def main():
    print(f"[manifold] Using device: {DEVICE}")

    # 1) Load latents + AE
    z_all, rel_all, scale_all = load_latents()
    N, D = z_all.shape

    ae = load_ae(z_dim=D)

    # 2) Train GRU on manifold-projected sequences
    if os.path.exists(GRU_CKPT_PATH):
        print(f"[manifold] Found existing GRU at {GRU_CKPT_PATH}, loading instead of retraining.")
        ckpt = torch.load(GRU_CKPT_PATH, map_location=DEVICE)
        model = FutureGRU(D, hidden=ckpt["config"]["hidden_dim"],
                          num_layers=ckpt["config"]["num_layers"])
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(DEVICE).eval()
    else:
        model = train_future_gru(ae, z_all)

    # 3) Visualization grid
    model.eval()
    make_viz_grid(ae, model, z_all, rel_all)


if __name__ == "__main__":
    main()
