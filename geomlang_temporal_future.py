#!/usr/bin/env python3
# geomlang_temporal_future.py
#
# 6.3 – Future prediction from latent sequences.
#
# Uses the *static* latent dataset (no true time) to:
#   - Build random sequences of latents z_t (i.i.d. over t)
#   - Train a GRU to:
#       (a) decode relation & scale at each frame
#       (b) predict the relation & scale of the final frame ("future@T")
#
# Because the sequences are random over time (no real dynamics),
# the *best achievable* future@T accuracy is basically chance.
# That itself is a geometric result: future is not encoded in past latents.
#
# Outputs:
#   - outputs_edges/temporal_future_gru.pt  (trained weights)
#   - outputs_edges/temporal_future_stats.npz (metrics per epoch)
#
# Run:
#   python bbit_geomlang/geomlang_temporal_future.py

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------
# Config
# -----------------------
LATENTS_PATH   = os.path.join("outputs_edges", "latents_dump.npz")
OUT_DIR        = "outputs_edges"
CKPT_PATH      = os.path.join(OUT_DIR, "temporal_future_gru.pt")
STATS_PATH     = os.path.join(OUT_DIR, "temporal_future_stats.npz")

SEQ_LEN        = 5            # T
BATCH_SIZE     = 128
EPOCHS         = 20
LR             = 1e-3
HIDDEN_GRU     = 256
LAMBDA_FUTURE  = 1.0          # weight for future prediction loss
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------
# Utils
# -----------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_latents(path: str, device):
    data = np.load(path)
    z = torch.from_numpy(data["z"]).float().to(device)
    rel = torch.from_numpy(data["rel"]).long().to(device)
    scale = torch.from_numpy(data["scale"]).long().to(device)
    print(f"[6.3] Loaded latents from {path}")
    print(f"[6.3] z shape: {z.shape}, rel min/max: {rel.min().item()}/{rel.max().item()}, "
          f"scale min/max: {scale.min().item()}/{scale.max().item()}")
    return z, rel, scale


class RandomSequenceDataset(Dataset):
    """
    Build random sequences by shuffling all indices and chunking into length T.
    This makes time artificially i.i.d. (no real dynamics), which is *good* here:
    we can empirically see that future is not predictable from past.
    """

    def __init__(self, z: torch.Tensor, rel: torch.Tensor, scale: torch.Tensor, T: int):
        super().__init__()
        self.z = z
        self.rel = rel
        self.scale = scale
        self.T = T
        self.indices = self._build_indices(z.size(0), T)

    def _build_indices(self, N: int, T: int):
        idxs = np.arange(N)
        np.random.shuffle(idxs)
        n_seqs = N // T
        idxs = idxs[: n_seqs * T]
        seqs = idxs.reshape(n_seqs, T)
        print(f"[6.3] Built {n_seqs} random sequences, T={T}")
        return torch.from_numpy(seqs).long()

    def __len__(self):
        return self.indices.size(0)

    def __getitem__(self, idx):
        seq_idx = self.indices[idx]  # [T]
        z_seq = self.z[seq_idx]      # [T, D]
        rel_seq = self.rel[seq_idx]  # [T]
        scale_seq = self.scale[seq_idx]  # [T]
        return z_seq, rel_seq, scale_seq


class GRUFutureHead(nn.Module):
    """
    GRU over latent sequences, with:
      - frame-wise heads for rel & scale
      - future@T heads for rel & scale (based on final hidden state)
    """
    def __init__(self, dim_latent: int, hidden: int, n_rel: int, n_scale: int):
        super().__init__()
        self.gru = nn.GRU(input_size=dim_latent,
                          hidden_size=hidden,
                          num_layers=1,
                          batch_first=True)

        self.rel_head = nn.Linear(hidden, n_rel)
        self.scale_head = nn.Linear(hidden, n_scale)

        self.future_rel_head = nn.Linear(hidden, n_rel)
        self.future_scale_head = nn.Linear(hidden, n_scale)

    def forward(self, z_seq: torch.Tensor):
        """
        z_seq: [B, T, D]
        Returns:
          rel_logits:   [B, T, n_rel]
          scale_logits: [B, T, n_scale]
          fut_rel_logits:   [B, n_rel]
          fut_scale_logits: [B, n_scale]
        """
        h_seq, h_last = self.gru(z_seq)    # h_seq: [B,T,H], h_last: [1,B,H]
        h_T = h_last.squeeze(0)           # [B,H]

        rel_logits = self.rel_head(h_seq)
        scale_logits = self.scale_head(h_seq)

        fut_rel_logits = self.future_rel_head(h_T)
        fut_scale_logits = self.future_scale_head(h_T)

        return rel_logits, scale_logits, fut_rel_logits, fut_scale_logits


# -----------------------
# Train / Eval
# -----------------------
def train_epoch(model, loader, optim, criterion, lambda_future):
    model.train()
    total_loss = 0.0
    total_frames = 0

    correct_rel = 0
    correct_scale = 0

    correct_fut_rel = 0
    correct_fut_scale = 0
    total_seqs = 0

    for z_seq, rel_seq, scale_seq in loader:
        z_seq = z_seq.to(DEVICE)          # [B,T,D]
        rel_seq = rel_seq.to(DEVICE)      # [B,T]
        scale_seq = scale_seq.to(DEVICE)  # [B,T]

        B, T, D = z_seq.shape

        optim.zero_grad()

        rel_logits, scale_logits, fut_rel_logits, fut_scale_logits = model(z_seq)

        # frame-wise losses
        frame_rel_loss = criterion(rel_logits.view(B * T, -1),
                                   rel_seq.view(B * T))
        frame_scale_loss = criterion(scale_logits.view(B * T, -1),
                                     scale_seq.view(B * T))

        # future@T targets = labels of last frame
        target_fut_rel = rel_seq[:, -1]    # [B]
        target_fut_scale = scale_seq[:, -1]  # [B]

        fut_rel_loss = criterion(fut_rel_logits, target_fut_rel)
        fut_scale_loss = criterion(fut_scale_logits, target_fut_scale)

        loss = frame_rel_loss + frame_scale_loss \
               + lambda_future * (fut_rel_loss + fut_scale_loss)

        loss.backward()
        optim.step()

        total_loss += loss.item() * B
        total_frames += B * T
        total_seqs += B

        # frame-wise accuracy
        pred_rel = rel_logits.argmax(dim=-1)      # [B,T]
        pred_scale = scale_logits.argmax(dim=-1)  # [B,T]
        correct_rel += (pred_rel == rel_seq).sum().item()
        correct_scale += (pred_scale == scale_seq).sum().item()

        # future@T accuracy
        pred_fut_rel = fut_rel_logits.argmax(dim=-1)     # [B]
        pred_fut_scale = fut_scale_logits.argmax(dim=-1) # [B]
        correct_fut_rel += (pred_fut_rel == target_fut_rel).sum().item()
        correct_fut_scale += (pred_fut_scale == target_fut_scale).sum().item()

    avg_loss = total_loss / total_seqs
    acc_rel = 100.0 * correct_rel / total_frames
    acc_scale = 100.0 * correct_scale / total_frames
    acc_fut_rel = 100.0 * correct_fut_rel / total_seqs
    acc_fut_scale = 100.0 * correct_fut_scale / total_seqs

    return avg_loss, acc_rel, acc_scale, acc_fut_rel, acc_fut_scale


@torch.no_grad()
def eval_epoch(model, loader, criterion, lambda_future):
    model.eval()
    total_loss = 0.0
    total_frames = 0

    correct_rel = 0
    correct_scale = 0

    correct_fut_rel = 0
    correct_fut_scale = 0
    total_seqs = 0

    for z_seq, rel_seq, scale_seq in loader:
        z_seq = z_seq.to(DEVICE)
        rel_seq = rel_seq.to(DEVICE)
        scale_seq = scale_seq.to(DEVICE)

        B, T, D = z_seq.shape

        rel_logits, scale_logits, fut_rel_logits, fut_scale_logits = model(z_seq)

        frame_rel_loss = criterion(rel_logits.view(B * T, -1),
                                   rel_seq.view(B * T))
        frame_scale_loss = criterion(scale_logits.view(B * T, -1),
                                     scale_seq.view(B * T))

        target_fut_rel = rel_seq[:, -1]
        target_fut_scale = scale_seq[:, -1]

        fut_rel_loss = criterion(fut_rel_logits, target_fut_rel)
        fut_scale_loss = criterion(fut_scale_logits, target_fut_scale)

        loss = frame_rel_loss + frame_scale_loss \
               + lambda_future * (fut_rel_loss + fut_scale_loss)

        total_loss += loss.item() * B
        total_frames += B * T
        total_seqs += B

        pred_rel = rel_logits.argmax(dim=-1)
        pred_scale = scale_logits.argmax(dim=-1)
        correct_rel += (pred_rel == rel_seq).sum().item()
        correct_scale += (pred_scale == scale_seq).sum().item()

        pred_fut_rel = fut_rel_logits.argmax(dim=-1)
        pred_fut_scale = fut_scale_logits.argmax(dim=-1)
        correct_fut_rel += (pred_fut_rel == target_fut_rel).sum().item()
        correct_fut_scale += (pred_fut_scale == target_fut_scale).sum().item()

    avg_loss = total_loss / total_seqs
    acc_rel = 100.0 * correct_rel / total_frames
    acc_scale = 100.0 * correct_scale / total_frames
    acc_fut_rel = 100.0 * correct_fut_rel / total_seqs
    acc_fut_scale = 100.0 * correct_fut_scale / total_seqs

    return avg_loss, acc_rel, acc_scale, acc_fut_rel, acc_fut_scale


# -----------------------
# Main
# -----------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(42)

    print(f"[6.3] Using device: {DEVICE}")

    z, rel, scale = load_latents(LATENTS_PATH, DEVICE)
    N, D = z.shape
    n_rel = int(rel.max().item()) + 1
    n_scale = int(scale.max().item()) + 1

    dataset = RandomSequenceDataset(z, rel, scale, T=SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = GRUFutureHead(dim_latent=D,
                          hidden=HIDDEN_GRU,
                          n_rel=n_rel,
                          n_scale=n_scale).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # For plotting later
    train_loss_hist = []
    train_rel_acc_hist = []
    train_scale_acc_hist = []
    train_fut_rel_acc_hist = []
    train_fut_scale_acc_hist = []

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_rel_acc, tr_scale_acc, tr_fut_rel_acc, tr_fut_scale_acc = train_epoch(
            model, loader, optimizer, criterion, LAMBDA_FUTURE
        )

        train_loss_hist.append(tr_loss)
        train_rel_acc_hist.append(tr_rel_acc)
        train_scale_acc_hist.append(tr_scale_acc)
        train_fut_rel_acc_hist.append(tr_fut_rel_acc)
        train_fut_scale_acc_hist.append(tr_fut_scale_acc)

        print(
            f"[6.3] Epoch {epoch:2d}/{EPOCHS} | "
            f"Loss={tr_loss:.4f} | "
            f"RelAcc={tr_rel_acc:.2f}% | ScaleAcc={tr_scale_acc:.2f}% | "
            f"Future@T RelAcc={tr_fut_rel_acc:.2f}% | Future@T ScaleAcc={tr_fut_scale_acc:.2f}%"
        )

    # Save model
    torch.save(
        {
            "config": {
                "dim_latent": D,
                "hidden_gru": HIDDEN_GRU,
                "n_rel": n_rel,
                "n_scale": n_scale,
                "lambda_future": LAMBDA_FUTURE,
                "seq_len": SEQ_LEN,
                "epochs": EPOCHS,
            },
            "state_dict": model.state_dict(),
        },
        CKPT_PATH,
    )
    print(f"[6.3] Saved GRU future model -> {CKPT_PATH}")

    # Save stats
    np.savez(
        STATS_PATH,
        train_loss=np.array(train_loss_hist),
        train_rel_acc=np.array(train_rel_acc_hist),
        train_scale_acc=np.array(train_scale_acc_hist),
        train_future_rel_acc=np.array(train_fut_rel_acc_hist),
        train_future_scale_acc=np.array(train_fut_scale_acc_hist),
    )
    print(f"[6.3] Saved training stats -> {STATS_PATH}")


if __name__ == "__main__":
    main()
