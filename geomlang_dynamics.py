#!/usr/bin/env python3
# geomlang_dynamics.py
#
# 7.0 – Latent dynamics via 5.4-style interpolations.
#
# We:
#   - Load latent codes (z) from outputs_edges/latents_dump.npz
#   - Build sequences by linearly interpolating z_start -> z_end
#   - Train a GRU to predict the future latent z_T from the first T-1 frames
#   - Run t-SNE on original latents, true futures, and imagined futures
#     to see how the manifold geometry changes when we add "temporal imagination".

import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# -----------------------
# Config
# -----------------------
LATENTS_PATH      = os.path.join("outputs_edges", "latents_dump.npz")
OUT_DIR           = "outputs_edges"
SEQ_LEN           = 5              # T
N_SEQS            = 4000           # total interpolant sequences to generate
BATCH_SIZE        = 128
HIDDEN_DIM        = 256
EPOCHS            = 20
LR                = 1e-3
TSNE_N_BASE       = 1000           # how many base latents to sample for TSNE
TSNE_N_FUTURE     = 1000           # how many futures (true & pred) to include
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------
# Data loading
# -----------------------
def load_latents(path=LATENTS_PATH, device=DEVICE):
    data = np.load(path)
    z = torch.from_numpy(data["z"]).float().to(device)      # [N, D]
    rel = torch.from_numpy(data["rel"]).long().to(device)   # [N]
    scale = torch.from_numpy(data["scale"]).long().to(device)
    print(f"[7.0] Loaded latents from {path}")
    print(f"[7.0] z shape: {z.shape}, rel range: {rel.min().item()}/{rel.max().item()}, "
          f"scale range: {scale.min().item()}/{scale.max().item()}")
    return z, rel, scale


# -----------------------
# Interpolant sequence dataset
# -----------------------
class InterpolantDataset(Dataset):
    """
    Each item is:
        seq_in:  [T-1, D]  (frames 0..T-2)
        target:  [D]       (frame T-1, i.e. the end of the interpolation)
        meta:    (idx_start, idx_end)
    """

    def __init__(self, z_all, n_seqs=4000, T=5):
        super().__init__()
        self.T = T
        self.z_all = z_all.cpu()  # keep a CPU copy for building sequences
        N, D = self.z_all.shape
        self.D = D

        # sample random start/end index pairs
        idx_start = np.random.randint(0, N, size=n_seqs)
        idx_end   = np.random.randint(0, N, size=n_seqs)

        # avoid trivial identical start/end in the majority of cases
        mask_same = (idx_start == idx_end)
        # resample those that are same
        while mask_same.any():
            idx_end[mask_same] = np.random.randint(0, N, size=mask_same.sum())
            mask_same = (idx_start == idx_end)

        self.idx_start = idx_start
        self.idx_end   = idx_end

        # precompute sequences as numpy for simplicity
        self.seqs = np.zeros((n_seqs, T, D), dtype=np.float32)
        for i in range(n_seqs):
            z0 = self.z_all[idx_start[i]].numpy()
            z1 = self.z_all[idx_end[i]].numpy()
            for t in range(T):
                alpha = t / (T - 1)  # 0 .. 1
                self.seqs[i, t] = (1.0 - alpha) * z0 + alpha * z1

        print(f"[7.0] Built {n_seqs} interpolant sequences (T={T})")

    def __len__(self):
        return self.seqs.shape[0]

    def __getitem__(self, idx):
        seq = self.seqs[idx]           # [T, D]
        seq_in = seq[:-1]              # [T-1, D]
        target = seq[-1]               # [D]
        return (
            torch.from_numpy(seq_in), 
            torch.from_numpy(target),
            int(self.idx_start[idx]),
            int(self.idx_end[idx]),
        )


# -----------------------
# GRU Future Predictor
# -----------------------
class FutureGRU(nn.Module):
    """
    GRU that takes a sequence of latents [B, T-1, D]
    and predicts the latent at T (future frame).
    """
    def __init__(self, latent_dim, hidden_dim=256, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x: [B, T-1, D]
        out, h = self.gru(x)   # out: [B, T-1, H]
        last = out[:, -1, :]   # [B, H]
        z_pred = self.fc_out(last)  # [B, D]
        return z_pred


# -----------------------
# Training & evaluation
# -----------------------
def train_future_gru(model, train_loader, test_loader, epochs=20, lr=1e-3, device=DEVICE):
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0
        for seq_in, target, _, _ in train_loader:
            seq_in = seq_in.to(device).float()
            target = target.to(device).float()

            optim.zero_grad()
            pred = model(seq_in)
            loss = loss_fn(pred, target)
            loss.backward()
            optim.step()

            train_loss += loss.item() * seq_in.size(0)
            n_train += seq_in.size(0)

        train_loss /= max(n_train, 1)

        # quick eval MSE on test
        model.eval()
        test_loss = 0.0
        n_test = 0
        with torch.no_grad():
            for seq_in, target, _, _ in test_loader:
                seq_in = seq_in.to(device).float()
                target = target.to(device).float()
                pred = model(seq_in)
                loss = loss_fn(pred, target)
                test_loss += loss.item() * seq_in.size(0)
                n_test += seq_in.size(0)
        test_loss /= max(n_test, 1)

        print(f"[7.0] Epoch {epoch:2d}/{epochs} | train MSE={train_loss:.6f} | test MSE={test_loss:.6f}")

    return model


# -----------------------
# t-SNE visualization
# -----------------------
def tsne_with_futures(z_all, model, dataset, out_dir=OUT_DIR, n_base=1000, n_future=1000, device=DEVICE):
    model.eval()
    z_all_cpu = z_all.cpu().numpy()
    N, D = z_all_cpu.shape

    # sample base points from the original latent cloud
    n_base = min(n_base, N)
    base_idx = np.random.choice(N, size=n_base, replace=False)
    z_base = z_all_cpu[base_idx]  # [n_base, D]

    # sample sequences from dataset for futures
    n_future = min(n_future, len(dataset))
    future_idx = np.random.choice(len(dataset), size=n_future, replace=False)

    z_true_fut = []
    z_pred_fut = []
    for idx in future_idx:
        seq_in, target, _, _ = dataset[idx]
        seq_in = seq_in.unsqueeze(0).to(device).float()   # [1, T-1, D]
        with torch.no_grad():
            pred = model(seq_in)[0].cpu().numpy()
        z_true_fut.append(target.numpy())
        z_pred_fut.append(pred)

    z_true_fut = np.stack(z_true_fut, axis=0)  # [n_future, D]
    z_pred_fut = np.stack(z_pred_fut, axis=0)  # [n_future, D]

    # joint array for t-SNE: base + true future + predicted future
    X = np.concatenate([z_base, z_true_fut, z_pred_fut], axis=0)
    labels = np.array(
        ["base"] * n_base + ["true_future"] * n_future + ["pred_future"] * n_future
    )

    print(f"[7.0] Running t-SNE on {X.shape[0]} points...")
    tsne = TSNE(
        n_components=2,
        perplexity=40,
        learning_rate=200,
        verbose=1,
        init="pca",
        random_state=42,
    )
    X_2d = tsne.fit_transform(X)

    # split back
    base_2d = X_2d[:n_base]
    true_2d = X_2d[n_base : n_base + n_future]
    pred_2d = X_2d[n_base + n_future :]

    # plot
    plt.figure(figsize=(10, 8))
    plt.scatter(
        base_2d[:, 0], base_2d[:, 1],
        s=10, alpha=0.3, label="Base latents"
    )
    plt.scatter(
        true_2d[:, 0], true_2d[:, 1],
        s=40, alpha=0.9, marker="o", label="True futures"
    )
    plt.scatter(
        pred_2d[:, 0], pred_2d[:, 1],
        s=40, alpha=0.9, marker="x", label="Imagined futures"
    )
    plt.legend()
    plt.title("7.0 – t-SNE of base latents vs true / imagined futures")
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "tsne_dynamics_futures.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[7.0] Saved t-SNE figure -> {out_path}")


# -----------------------
# Main
# -----------------------
def main():
    set_seed(42)
    print(f"[7.0] Using device: {DEVICE}")

    # 1) Load static latents
    z_all, rel, scale = load_latents(LATENTS_PATH, DEVICE)
    N, D = z_all.shape

    # 2) Build interpolant sequences dataset
    dataset = InterpolantDataset(z_all, n_seqs=N_SEQS, T=SEQ_LEN)

    # 3) Train/test split
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    indices = np.random.permutation(n_total)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_subset = torch.utils.data.Subset(dataset, train_idx)
    test_subset  = torch.utils.data.Subset(dataset, test_idx)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)

    # 4) Model + training
    model = FutureGRU(latent_dim=D, hidden_dim=HIDDEN_DIM)
    model = train_future_gru(model, train_loader, test_loader, epochs=EPOCHS, lr=LR, device=DEVICE)

    # save checkpoint
    os.makedirs(OUT_DIR, exist_ok=True)
    ckpt_path = os.path.join(OUT_DIR, "future_gru_dynamics.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "latent_dim": D,
                "hidden_dim": HIDDEN_DIM,
                "seq_len": SEQ_LEN,
                "n_seqs": N_SEQS,
            },
        },
        ckpt_path,
    )
    print(f"[7.0] Saved future GRU checkpoint -> {ckpt_path}")

    # 5) t-SNE visualization of manifold + futures
    tsne_with_futures(z_all, model, dataset, out_dir=OUT_DIR,
                      n_base=TSNE_N_BASE, n_future=TSNE_N_FUTURE, device=DEVICE)


if __name__ == "__main__":
    main()
