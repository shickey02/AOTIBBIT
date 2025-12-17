#!/usr/bin/env python3
# geomlang_tsne_temporal_overlay.py
#
# 7.1 – Overlay static relation structure and temporal dynamics on same t-SNE.
#
# - Loads latents_dump.npz (z, rel, scale)
# - Loads future_gru_dynamics.pt trained in geomlang_dynamics.py
# - Builds a t-SNE embedding for a subset of latents
# - For each embedded point, constructs a synthetic motion sequence that
#   *ends* at that latent, runs it through the GRU, and measures prediction
#   error at the final frame.
# - Plots:
#     Left  panel: t-SNE colored by relation label (static geometry)
#     Right panel: t-SNE colored by temporal prediction error (dynamics)
#
# Output: outputs_edges/tsne_static_vs_temporal.png

import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

LATENTS_PATH = os.path.join("outputs_edges", "latents_dump.npz")
DYNAMICS_CKPT = os.path.join("outputs_edges", "future_gru_dynamics.pt")
OUT_FIG = os.path.join("outputs_edges", "tsne_static_vs_temporal.png")

N_TSNE_POINTS = 3000   # how many latents to embed
SEQ_LEN = 5            # must match geomlang_dynamics
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_latents(path=LATENTS_PATH):
    data = np.load(path)
    z = torch.from_numpy(data["z"]).float()          # [N, D]
    rel = torch.from_numpy(data["rel"]).long()       # [N]
    scale = torch.from_numpy(data["scale"]).long()   # [N]
    return z, rel, scale


class FutureGRUDynamics(nn.Module):
    """
    Same architecture as in geomlang_dynamics.py:
    GRU over latent sequences, predict next latent.
    """
    def __init__(self, dim_latent, hidden=256):
        super().__init__()
        self.gru = nn.GRU(dim_latent, hidden, batch_first=True)
        # align with checkpoint naming (fc_out)
        self.fc_out = nn.Linear(hidden, dim_latent)

    def forward(self, seq):
        # seq: [B, T-1, D]
        out, _ = self.gru(seq)   # [B, T-1, H]
        last = out[:, -1, :]     # [B, H]
        pred = self.fc_out(last)    # [B, D]
        return pred


def load_dynamics_model(path, dim_latent):
    ckpt = torch.load(path, map_location=DEVICE)
    # Try to read a config if present, else default
    hidden = 256
    if "config" in ckpt:
        hidden = ckpt["config"].get("hidden_dim", hidden)
    model = FutureGRUDynamics(dim_latent, hidden=hidden).to(DEVICE)
    state_key = None
    for key_candidate in ("model_state_dict", "model", "state_dict"):
        if key_candidate in ckpt:
            state_key = key_candidate
            break
    if state_key is None:
        raise KeyError(f"No model state found in checkpoint keys: {ckpt.keys()}")
    model.load_state_dict(ckpt[state_key])
    model.eval()
    return model


@torch.no_grad()
def compute_temporal_errors(z, idxs, model, seq_len=SEQ_LEN):
    """
    For each latent index j in idxs, build one synthetic motion sequence
    that ENDS at z[j], by interpolating from a random start latent k.
    Run [start..second-to-last] through GRU and predict final latent.
    Compute MSE between prediction and true z[j].

    Returns: errors: np.ndarray [len(idxs)]
    """
    z = z.to(DEVICE)
    N, D = z.shape
    L = len(idxs)
    errors = np.zeros(L, dtype=np.float32)

    for n, j in enumerate(idxs):
        # pick random start index != j
        k = j
        while k == j:
            k = np.random.randint(0, N)

        z_start = z[k]
        z_end = z[j]

        # Build linear interpolation sequence from start -> end
        T = seq_len
        alphas = torch.linspace(0.0, 1.0, T, device=DEVICE)  # [T]
        seq = (1.0 - alphas.view(T, 1)) * z_start.view(1, D) + \
              alphas.view(T, 1) * z_end.view(1, D)            # [T, D]

        # Input to GRU: first T-1 frames
        inp = seq[:-1, :].unsqueeze(0)  # [1, T-1, D]
        target = seq[-1, :].unsqueeze(0)  # [1, D]

        pred = model(inp)  # [1, D]
        mse = torch.mean((pred - target) ** 2).item()
        errors[n] = mse

    return errors


def make_tsne(z, idxs):
    """
    Compute a t-SNE embedding for z[idxs].
    """
    z_np = z[idxs].numpy()
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        init="pca",
        random_state=42,
    )
    emb = tsne.fit_transform(z_np)  # [M, 2]
    return emb


def plot_static_vs_temporal(emb, rel_subset, errors, out_path=OUT_FIG):
    """
    emb: [M,2]  t-SNE coords
    rel_subset: [M] relation labels for same points
    errors: [M] temporal MSE for same points
    """
    rel_np = rel_subset.numpy()
    M = emb.shape[0]

    # Normalize errors for color (lower error = more predictable)
    eps = 1e-8
    log_err = np.log(errors + eps)
    # invert so that lower MSE -> higher score
    score = -log_err
    score = (score - score.min()) / (score.max() - score.min() + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_static, ax_temp = axes

    # --- Left: static relation clusters ---
    scatter = ax_static.scatter(
        emb[:, 0], emb[:, 1],
        c=rel_np,
        cmap="tab10",
        s=10,
        alpha=0.9
    )
    ax_static.set_title("Static relation clusters (t-SNE)")
    ax_static.set_xticks([])
    ax_static.set_yticks([])

    # Build legend for relation ids
    handles = []
    labels = []
    for r in sorted(np.unique(rel_np)):
        handles.append(plt.Line2D([0], [0], marker='o', linestyle='',
                                  color=scatter.cmap(scatter.norm(r)), markersize=6))
        labels.append(f"rel={r}")
    ax_static.legend(handles, labels, loc="best", fontsize=8, frameon=False)

    # --- Right: temporal predictability ---
    temp_sc = ax_temp.scatter(
        emb[:, 0], emb[:, 1],
        c=score,
        cmap="viridis",
        s=10,
        alpha=0.9
    )
    ax_temp.set_title("Temporal predictability (future-GRU MSE)")
    ax_temp.set_xticks([])
    ax_temp.set_yticks([])

    cbar = fig.colorbar(temp_sc, ax=ax_temp, fraction=0.046, pad=0.04)
    cbar.set_label("Low ⟵ 1 / log(MSE) ⟶ High", fontsize=8)

    plt.suptitle("7.1 – Static vs temporal structure in latent space", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"[7.1] Saved t-SNE static+temporal overlay -> {out_path}")


def main():
    set_seed(42)
    print(f"[7.1] Using device: {DEVICE}")

    # --- Load latents + labels ---
    z, rel, scale = load_latents(LATENTS_PATH)
    N, D = z.shape
    print(f"[7.1] Loaded latents: N={N}, D={D}, rel range={rel.min().item()}/{rel.max().item()}")

    # --- Pick subset of points for t-SNE ---
    N_use = min(N_TSNE_POINTS, N)
    idxs = np.random.choice(N, size=N_use, replace=False)
    idxs = np.sort(idxs)  # just for determinism
    rel_subset = rel[idxs]

    # --- Compute t-SNE embedding ---
    print("[7.1] Running t-SNE on subset...")
    emb = make_tsne(z, idxs)

    # --- Load dynamics model & compute temporal errors ---
    print("[7.1] Loading dynamics model...")
    model = load_dynamics_model(DYNAMICS_CKPT, D)

    print("[7.1] Computing temporal prediction errors (per endpoint)...")
    errors = compute_temporal_errors(z, idxs, model, seq_len=SEQ_LEN)
    print(f"[7.1] Temporal errors: min={errors.min():.4f}, max={errors.max():.4f}")

    # --- Plot overlay ---
    plot_static_vs_temporal(emb, rel_subset, errors, out_path=OUT_FIG)


if __name__ == "__main__":
    main()
