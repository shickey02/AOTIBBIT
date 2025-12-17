#!/usr/bin/env python3
# geomlang_temporal_basic32.py
#
# Builds on geomlang_multishape_relscale.py ("basic32"):
#   - Trains a clean SceneModel (encoder/decoder + heads)
#   - Extracts latents z for many scenes
#   - Constructs synthetic temporal sequences in latent space
#       * left_of -> right_of
#       * above   -> below
#       * inside  -> overlapping
#       * plus some random relation pairs
#   - Compares three predictors for the future latent z_T:
#       1. Last-frame baseline        (z_4)
#       2. Linear constant-velocity   (z_4 + (z_4 - z_3))
#       3. GRU                        (learned)
#   - Decodes everything back to RGB frames and saves:
#       outputs_basic32/temporal_future_grid_basic32.png
#   - Also makes a t-SNE arrow plot of GRU flow on the manifold:
#       outputs_basic32/temporal_tsne_arrows_basic32.png
#
# Run from your project root as:
#   python bbit_geomlang/geomlang_temporal_basic32.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# ---- import your rebuilt spatial world ----
# If your file has a different name, fix this import.
from geomlang_multishape_relscale import (
    IMG_SIZE,
    NUM_CHANNELS,
    LATENT_DIM,
    RELATION_NAMES,
    NUM_REL,
    generate_dataset,
    SceneModel,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "outputs_basic32"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------
# 1. Train / load spatial model
# -------------------------------

CKPT_PATH = os.path.join(OUT_DIR, "scene_model_basic32.pt")


def train_spatial_model():
    print(f"[temporal] Using device: {DEVICE}")

    # Slightly fewer samples/epochs than your big run to keep it fast.
    scenes, rel_labels, scale_labels, shape_r, shape_b = generate_dataset(
        num_samples=4000, seed=0
    )

    dataset = TensorDataset(scenes, rel_labels, scale_labels, shape_r, shape_b)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = SceneModel().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 250
    for epoch in range(EPOCHS):
        model.train()
        tot_recon = tot_rel = tot_scale = tot_sr = tot_sb = 0.0
        tot = 0
        correct_rel = correct_scale = correct_sr = correct_sb = 0

        for x, rel, scale, sr, sb in loader:
            x = x.to(DEVICE)
            rel = rel.to(DEVICE)
            scale = scale.to(DEVICE)
            sr = sr.to(DEVICE)
            sb = sb.to(DEVICE)

            opt.zero_grad()
            recon, z, rel_logits, scale_logits, sr_logits, sb_logits = model(x)

            bce = F.binary_cross_entropy(recon, x)
            rel_ce = F.cross_entropy(rel_logits, rel)
            scale_ce = F.cross_entropy(scale_logits, scale)
            sr_ce = F.cross_entropy(sr_logits, sr)
            sb_ce = F.cross_entropy(sb_logits, sb)

            loss = bce + rel_ce + scale_ce + sr_ce + sb_ce
            loss.backward()
            opt.step()

            bsz = x.size(0)
            tot += bsz
            tot_recon += bce.item() * bsz
            tot_rel += rel_ce.item() * bsz
            tot_scale += scale_ce.item() * bsz
            tot_sr += sr_ce.item() * bsz
            tot_sb += sb_ce.item() * bsz

            correct_rel += (rel_logits.argmax(dim=1) == rel).sum().item()
            correct_scale += (scale_logits.argmax(dim=1) == scale).sum().item()
            correct_sr += (sr_logits.argmax(dim=1) == sr).sum().item()
            correct_sb += (sb_logits.argmax(dim=1) == sb).sum().item()

        if epoch % 50 == 0 or epoch == EPOCHS - 1:
            print(
                f"[temporal] Epoch {epoch}/{EPOCHS-1} | "
                f"Recon {tot_recon/tot:.4f} | "
                f"RelCE {tot_rel/tot:.4f} | "
                f"ScaleCE {tot_scale/tot:.4f} | "
                f"ShapeRCE {tot_sr/tot:.4f} | "
                f"ShapeBCE {tot_sb/tot:.4f} | "
                f"Acc_rel {100*correct_rel/tot:.1f}% | "
                f"Acc_scale {100*correct_scale/tot:.1f}% | "
                f"Acc_shapeR {100*correct_sr/tot:.1f}% | "
                f"Acc_shapeB {100*correct_sb/tot:.1f}%"
            )

    torch.save(model.state_dict(), CKPT_PATH)
    print(f"[temporal] Saved spatial checkpoint -> {CKPT_PATH}")
    return model, dataset


def load_or_train_spatial():
    if os.path.exists(CKPT_PATH):
        print(f"[temporal] Loading spatial model from {CKPT_PATH}")
        model = SceneModel().to(DEVICE)
        model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
        model.eval()

        # Re-generate dataset (same distribution; we don't care about exact
        # per-sample identity for dynamics).
        scenes, rel_labels, scale_labels, shape_r, shape_b = generate_dataset(
            num_samples=4000, seed=0
        )
        dataset = TensorDataset(scenes, rel_labels, scale_labels, shape_r, shape_b)
        return model, dataset
    else:
        return train_spatial_model()


# --------------------------------
# 2. Build temporal latent dataset
# --------------------------------

def encode_all_latents(model, dataset):
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    all_z = []
    all_rel = []
    with torch.no_grad():
        for x, rel, scale, sr, sb in loader:
            x = x.to(DEVICE)
            z = model.encode(x)
            all_z.append(z.cpu())
            all_rel.append(rel)
    all_z = torch.cat(all_z, dim=0)      # [N,D]
    all_rel = torch.cat(all_rel, dim=0)  # [N]
    print(f"[temporal] Encoded latents: N={all_z.shape[0]}, D={all_z.shape[1]}")
    return all_z, all_rel


def sample_index_for_rel(rel_labels, rel_id, rng):
    idxs = torch.nonzero(rel_labels == rel_id, as_tuple=False).squeeze(1)
    if len(idxs) == 0:
        # Fallback: any index
        return int(rng.integers(0, len(rel_labels)))
    return int(idxs[int(rng.integers(0, len(idxs)))])


def build_temporal_sequences(z_all, rel_all, num_seqs=1200, T=5, seed=123):
    """
    Build sequences by linear interpolation in latent space between
    scenes of specific relation pairs.

    Returns:
        seqs [num_seqs, T, D]
        info list of (rel_start, rel_end)
    """
    rng = np.random.default_rng(seed)
    N, D = z_all.shape
    seqs = []
    info = []

    # canonical relation transitions
    transitions = [
        (RELATION_NAMES.index("left_of"), RELATION_NAMES.index("right_of")),
        (RELATION_NAMES.index("above"), RELATION_NAMES.index("below")),
    ]
    # some inside/overlapping if they exist
    if "inside" in RELATION_NAMES and "overlapping" in RELATION_NAMES:
        transitions.append(
            (RELATION_NAMES.index("inside"), RELATION_NAMES.index("overlapping"))
        )

    num_per_pair = num_seqs // (len(transitions) + 1)

    def interp(z0, z1, T):
        alphas = torch.linspace(0.0, 1.0, T, device=z_all.device).unsqueeze(1)
        return (1 - alphas) * z0.unsqueeze(0) + alphas * z1.unsqueeze(0)  # [T,D]

    # structured transitions
    for rel_start, rel_end in transitions:
        for _ in range(num_per_pair):
            i0 = sample_index_for_rel(rel_all, rel_start, rng)
            i1 = sample_index_for_rel(rel_all, rel_end, rng)
            seq = interp(z_all[i0], z_all[i1], T)
            seqs.append(seq)
            info.append((rel_start, rel_end))

    # some purely random pairs (extra spice)
    remaining = num_seqs - len(seqs)
    for _ in range(remaining):
        i0 = int(rng.integers(0, N))
        i1 = int(rng.integers(0, N))
        seq = interp(z_all[i0], z_all[i1], T)
        seqs.append(seq)
        info.append((-1, -1))

    seqs = torch.stack(seqs, dim=0)  # [num_seqs, T, D]
    print(f"[temporal] Built temporal sequences: {seqs.shape}")
    return seqs.to(DEVICE), info


# --------------------------------------
# 3. Temporal models: baseline / linear / GRU
# --------------------------------------

class FutureGRU(nn.Module):
    def __init__(self, dim_latent, hidden=256):
        super().__init__()
        self.gru = nn.GRU(
            input_size=dim_latent, hidden_size=hidden, num_layers=1, batch_first=True
        )
        self.fc = nn.Linear(hidden, dim_latent)

    def forward(self, z_hist):
        # z_hist: [B, T_hist, D]
        out, _ = self.gru(z_hist)
        last = out[:, -1, :]
        return self.fc(last)  # [B,D]


def temporal_dataset_from_seqs(seqs, T_hist=4):
    """
    seqs: [N, T, D]
    Returns:
        z_hist: [N, T_hist, D]
        z_target: [N, D]  (future z_T at step T_hist)
    """
    T = seqs.shape[1]
    assert T_hist < T
    z_hist = seqs[:, :T_hist, :]
    z_target = seqs[:, T_hist, :]
    return z_hist, z_target


def train_gru(z_hist, z_target, dim_latent, epochs=40):
    dataset = TensorDataset(z_hist, z_target)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = FutureGRU(dim_latent).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        tot = 0.0
        count = 0
        for zh, zt in loader:
            zh = zh.to(DEVICE)
            zt = zt.to(DEVICE)
            opt.zero_grad()
            pred = model(zh)
            loss = mse(pred, zt)
            loss.backward()
            opt.step()
            tot += loss.item() * zh.size(0)
            count += zh.size(0)
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"[temporal] GRU epoch {epoch}/{epochs-1} | MSE {tot/count:.6f}")
    return model


@torch.no_grad()
def evaluate_methods(z_hist, z_target, gru_model):
    """
    Compare:
        - last-frame baseline
        - constant-velocity linear extrapolation
        - GRU
    Returns:
        dict of mse values
    """
    mse = nn.MSELoss()

    # baseline: copy last history frame
    z_last = z_hist[:, -1, :]

    # linear: constant velocity in latent space
    z_prev = z_hist[:, -2, :]
    v = z_last - z_prev
    z_lin = z_last + v

    # GRU
    z_gru = gru_model(z_hist)

    m_last = mse(z_last, z_target).item()
    m_lin = mse(z_lin, z_target).item()
    m_gru = mse(z_gru, z_target).item()

    print(
        "[temporal] MSE (latent): "
        f"last={m_last:.6f}, linear={m_lin:.6f}, GRU={m_gru:.6f}"
    )
    return {
        "last": m_last,
        "linear": m_lin,
        "gru": m_gru,
        "z_last": z_last,
        "z_lin": z_lin,
        "z_gru": z_gru,
    }


# --------------------------------------
# 4. Decode & visualize frame trajectories
# --------------------------------------

def channels_to_rgb(img_tensor):
    """
    Your basic32 scenes are RGB: channel 0 red, 1 green(?), 2 blue.
    We'll assume:
      - red object in red channel
      - blue object in blue channel
      - green mostly unused
    We'll just return the tensor as-is (clipped to [0,1]).
    """
    x = img_tensor.detach().cpu().numpy()
    x = np.clip(x, 0.0, 1.0)
    x = np.transpose(x, (1, 2, 0))  # CHW -> HWC
    return x


@torch.no_grad()
def make_frame_grid(model, seqs, z_target, results, num_rows=8, out_path=None):
    """
    Build a grid:
        t1,t2,t3,t4, TrueT, LinearPred, GRUPred, |GRU-True|
    """
    model.eval()
    decoder = model.decode

    z_hist = seqs[:, :4, :]  # [N,4,D]
    z_lin = results["z_lin"]
    z_gru = results["z_gru"]

    N = seqs.shape[0]
    num_rows = min(num_rows, N)

    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=8,
        figsize=(10, 1.3 * num_rows),
        dpi=160,
    )
    fig.suptitle("Latent dynamics (basic32): t=1..4, True T, Linear, GRU, |GRU–True|")

    titles = ["t=1", "t=2", "t=3", "t=4", "True T", "Linear", "GRU", "|GRU-True|"]

    for r in range(num_rows):
        # decode history
        hist = decoder(z_hist[r]).cpu()  # [4, C, H, W]
        true = decoder(z_target[r].unsqueeze(0)).squeeze(0).cpu()
        lin = decoder(z_lin[r].unsqueeze(0)).squeeze(0).cpu()
        gru = decoder(z_gru[r].unsqueeze(0)).squeeze(0).cpu()

        diff = (gru - true).abs().mean(dim=0).numpy()

        for c in range(8):
            ax = axes[r, c]
            ax.axis("off")
            if r == 0:
                ax.set_title(titles[c], fontsize=8)

            if c < 4:
                img = channels_to_rgb(hist[c])
                ax.imshow(img)
            elif c == 4:
                ax.imshow(channels_to_rgb(true))
            elif c == 5:
                ax.imshow(channels_to_rgb(lin))
            elif c == 6:
                ax.imshow(channels_to_rgb(gru))
            else:
                im = ax.imshow(
                    diff,
                    cmap="magma",
                    vmin=0.0,
                    vmax=max(1e-3, diff.max()),
                )
                if r == 0:
                    # attach a colorbar only once
                    cax = fig.add_axes([0.92, 0.1, 0.015, 0.8])
                    fig.colorbar(im, cax=cax)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    if out_path is not None:
        plt.savefig(out_path)
        print(f"[temporal] Saved frame grid -> {out_path}")
    else:
        plt.show()
    plt.close(fig)


# --------------------------------------
# 5. t-SNE vector field viz
# --------------------------------------

@torch.no_grad()
def make_tsne_arrows(z_all, z_hist, results, out_path=None, max_arrows=300):
    """
    t-SNE over a subset of latents, plus GRU arrows.
    """
    z_last = z_hist[:, -1, :]
    z_gru = results["z_gru"]

    # subsample to avoid clutter
    N_arrows = min(max_arrows, z_last.shape[0])
    idx = torch.randperm(z_last.shape[0])[:N_arrows]
    z_last_s = z_last[idx].cpu()
    z_gru_s = z_gru[idx].cpu()

    # background static points
    N_bg = min(2000, z_all.shape[0])
    idx_bg = torch.randperm(z_all.shape[0])[:N_bg]
    z_bg = z_all[idx_bg].cpu()

    Z = torch.cat([z_bg, z_last_s, z_gru_s], dim=0).numpy()
    print(f"[temporal] Running t-SNE on {Z.shape[0]} points...")
    tsne = TSNE(
        n_components=2,
        perplexity=40.0,
        learning_rate="auto",
        init="pca",
    )
    Z2 = tsne.fit_transform(Z)

    # unpack
    bg_2d = Z2[:N_bg]
    last_2d = Z2[N_bg : N_bg + N_arrows]
    gru_2d = Z2[N_bg + N_arrows :]

    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    ax.set_title("t-SNE manifold with GRU temporal flow (basic32)")

    ax.scatter(bg_2d[:, 0], bg_2d[:, 1], s=5, alpha=0.25, label="static latents")

    for i in range(N_arrows):
        x0, y0 = last_2d[i]
        x1, y1 = gru_2d[i]
        ax.arrow(
            x0,
            y0,
            x1 - x0,
            y1 - y0,
            head_width=0.8,
            head_length=1.2,
            length_includes_head=True,
            alpha=0.7,
            linewidth=0.7,
        )

    ax.legend(loc="best")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path)
        print(f"[temporal] Saved t-SNE arrows -> {out_path}")
    else:
        plt.show()
    plt.close(fig)


# --------------------------------------
# 6. Main
# --------------------------------------

def main():
    # 1) get spatial model + dataset
    model, dataset = load_or_train_spatial()

    # 2) latents for many static scenes
    z_all, rel_all = encode_all_latents(model, dataset)

    # 3) temporal sequences in latent space
    seqs, info = build_temporal_sequences(z_all.to(DEVICE), rel_all.to(DEVICE))

    # 4) future-prediction dataset
    z_hist, z_target = temporal_dataset_from_seqs(seqs, T_hist=4)

    # 5) train GRU
    gru_model = train_gru(z_hist, z_target, dim_latent=LATENT_DIM, epochs=40)

    # 6) evaluate methods
    results = evaluate_methods(z_hist, z_target, gru_model)

    # 7) frame grid viz
    grid_path = os.path.join(OUT_DIR, "temporal_future_grid_basic32.png")
    make_frame_grid(model, seqs, z_target, results, num_rows=8, out_path=grid_path)

    # 8) t-SNE arrows
    tsne_path = os.path.join(OUT_DIR, "temporal_tsne_arrows_basic32.png")
    make_tsne_arrows(z_all, z_hist, results, out_path=tsne_path)


if __name__ == "__main__":
    main()
