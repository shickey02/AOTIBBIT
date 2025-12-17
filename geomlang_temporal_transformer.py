#!/usr/bin/env python3
# geomlang_temporal_transformer.py
#
# 6.2 – Temporal reasoning with order-sensitive encoder (GRU) and relation-change sequences.
#
# Loads frozen latents (z, rel, scale) from outputs_edges/latents_dump.npz,
# builds synthetic sequences where relations/scale can change over time,
# and trains a small GRU head with positional embeddings to predict per-timestep
# relation/scale plus a future target from the first K frames.
#
# Usage:
#   python bbit_geomlang/geomlang_temporal_transformer.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

LATENTS_PATH   = os.path.join("outputs_edges", "latents_dump.npz")
CKPT_OUT       = os.path.join("outputs_edges", "temporal_gru_head.pt")

BATCH_SIZE     = 128
SEQ_LEN        = 6
FUTURE_K       = 3          # use first K frames to predict frame T-1
N_SEQS         = 8000       # total sequences
EPOCHS         = 15
HIDDEN_GRU     = 256
HIDDEN_MLP     = 256
LR             = 1e-3

DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED           = 42


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_latents(path=LATENTS_PATH, device=DEVICE):
    data = np.load(path)
    z = torch.from_numpy(data["z"]).float().to(device)       # [N,D]
    rel = torch.from_numpy(data["rel"]).long().to(device)    # [N]
    scale = torch.from_numpy(data["scale"]).long().to(device)# [N]
    return z, rel, scale


class RelationSeqDataset(Dataset):
    """
    Build sequences by interpolating between latents of start/end relation/scale.
    rel labels per frame: start for first half, end for second half.
    scale labels per frame: start for first half, end for second half.
    """
    def __init__(self, z, rel, scale, seq_len=6, n_seqs=8000):
        super().__init__()
        self.z = z
        self.rel = rel
        self.scale = scale
        self.seq_len = seq_len
        self.n_seqs = n_seqs

        # buckets per relation
        self.rel_to_idx = {}
        for r in torch.unique(rel).tolist():
            self.rel_to_idx[int(r)] = (rel == r).nonzero(as_tuple=False).squeeze(1)
        # buckets per scale
        self.scale_to_idx = {}
        for s in torch.unique(scale).tolist():
            self.scale_to_idx[int(s)] = (scale == s).nonzero(as_tuple=False).squeeze(1)

        self.rel_values = list(self.rel_to_idx.keys())
        self.scale_values = list(self.scale_to_idx.keys())

    def __len__(self):
        return self.n_seqs

    def __getitem__(self, idx):
        T = self.seq_len
        # pick start/end relations and scales (can be same)
        r_start = int(np.random.choice(self.rel_values))
        r_end   = int(np.random.choice(self.rel_values))
        s_start = int(np.random.choice(self.scale_values))
        s_end   = int(np.random.choice(self.scale_values))

        # pick seeds
        z_start = self._sample_from_rel_scale(r_start, s_start)
        z_end   = self._sample_from_rel_scale(r_end,   s_end)

        # interpolate latents
        alphas = torch.linspace(0.0, 1.0, steps=T, device=self.z.device).unsqueeze(1)  # [T,1]
        z_seq = (1 - alphas) * z_start + alphas * z_end  # [T,D]

        # labels: first half start, second half end
        rel_seq = torch.tensor(
            [r_start if t < T//2 else r_end for t in range(T)],
            device=self.z.device, dtype=torch.long
        )
        scale_seq = torch.tensor(
            [s_start if t < T//2 else s_end for t in range(T)],
            device=self.z.device, dtype=torch.long
        )
        return z_seq, rel_seq, scale_seq

    def _sample_from_rel_scale(self, rel_val, scale_val):
        # intersect buckets
        idx_rel = self.rel_to_idx[rel_val]
        idx_scale = self.scale_to_idx[scale_val]
        common = torch.tensor(np.intersect1d(idx_rel.cpu().numpy(), idx_scale.cpu().numpy()), device=self.z.device)
        if common.numel() == 0:
            # fallback: ignore scale
            common = idx_rel
        choice = common[torch.randint(0, common.numel(), (1,))]
        return self.z[choice]  # [1,D]


class GRUHead(nn.Module):
    def __init__(self, dim_latent, n_rel, n_scale, hidden_gru=256, hidden_mlp=256, seq_len=6):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(seq_len, dim_latent))
        self.mlp_in = nn.Sequential(
            nn.Linear(dim_latent, hidden_mlp),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(hidden_mlp, hidden_gru, batch_first=True)
        self.head_rel = nn.Linear(hidden_gru, n_rel)
        self.head_scale = nn.Linear(hidden_gru, n_scale)

    def forward(self, z_seq):
        # z_seq: [B,T,D]
        B, T, D = z_seq.shape
        pos = self.pos_emb[:T].unsqueeze(0)  # [1,T,D]
        x = z_seq + pos
        x = self.mlp_in(x)
        h, _ = self.gru(x)  # [B,T,H]
        rel_logits = self.head_rel(h)
        scale_logits = self.head_scale(h)
        return rel_logits, scale_logits, h


def evaluate(loader, model, device):
    model.eval()
    correct_rel = correct_scale = total = 0
    with torch.no_grad():
        for z_seq, r_seq, s_seq in loader:
            z_seq = z_seq.to(device)
            r_seq = r_seq.to(device)
            s_seq = s_seq.to(device)
            rel_logits, scale_logits, _ = model(z_seq)
            rel_pred = rel_logits.argmax(dim=-1)
            scale_pred = scale_logits.argmax(dim=-1)
            correct_rel += (rel_pred == r_seq).sum().item()
            correct_scale += (scale_pred == s_seq).sum().item()
            total += r_seq.numel()
    return 100.0 * correct_rel / total, 100.0 * correct_scale / total


def evaluate_future(loader, model, k_frames, device):
    model.eval()
    correct_rel = correct_scale = total = 0
    with torch.no_grad():
        for z_seq, r_seq, s_seq in loader:
            z_seq = z_seq.to(device)
            r_seq = r_seq.to(device)
            s_seq = s_seq.to(device)
            rel_logits, scale_logits, h = model(z_seq)
            # use hidden at step k_frames-1 to predict final labels
            h_k = h[:, k_frames-1, :]  # [B, H]
            rel_last = model.head_rel(h_k)
            scale_last = model.head_scale(h_k)
            rel_pred = rel_last.argmax(dim=-1)
            scale_pred = scale_last.argmax(dim=-1)
            rel_true = r_seq[:, -1]
            scale_true = s_seq[:, -1]
            correct_rel += (rel_pred == rel_true).sum().item()
            correct_scale += (scale_pred == scale_true).sum().item()
            total += rel_true.numel()
    return 100.0 * correct_rel / total, 100.0 * correct_scale / total


def main():
    set_seed(SEED)
    print(f"[6.2] Using device: {DEVICE}")
    z, rel, scale = load_latents(LATENTS_PATH, DEVICE)
    N, D = z.shape
    n_rel = int(torch.max(rel).item()) + 1
    n_scale = int(torch.max(scale).item()) + 1

    dataset = RelationSeqDataset(z, rel, scale, seq_len=SEQ_LEN, n_seqs=N_SEQS)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = GRUHead(D, n_rel, n_scale, hidden_gru=HIDDEN_GRU, hidden_mlp=HIDDEN_MLP, seq_len=SEQ_LEN).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    print("[6.2] Training GRU head on relation-change sequences...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = total_ce_rel = total_ce_scale = 0.0
        total = 0
        for z_seq, r_seq, s_seq in loader:
            z_seq = z_seq.to(DEVICE)
            r_seq = r_seq.to(DEVICE)
            s_seq = s_seq.to(DEVICE)
            opt.zero_grad()
            rel_logits, scale_logits, _ = model(z_seq)
            B, T, _ = rel_logits.shape
            loss_rel = F.cross_entropy(rel_logits.view(B*T, n_rel), r_seq.view(-1))
            loss_scale = F.cross_entropy(scale_logits.view(B*T, n_scale), s_seq.view(-1))
            loss = loss_rel + loss_scale
            loss.backward()
            opt.step()
            total_loss += loss.item() * B
            total_ce_rel += loss_rel.item() * B
            total_ce_scale += loss_scale.item() * B
            total += B
        if epoch % 5 == 0 or epoch == 1:
            acc_rel, acc_scale = evaluate(loader, model, DEVICE)
            acc_rel_f, acc_scale_f = evaluate_future(loader, model, FUTURE_K, DEVICE)
            print(f"[6.2] Epoch {epoch:2d}/{EPOCHS} | Loss={total_loss/total:.4f} | "
                  f"RelAcc={acc_rel:.2f}% | ScaleAcc={acc_scale:.2f}% | "
                  f"Future@T RelAcc={acc_rel_f:.2f}% | Future@T ScaleAcc={acc_scale_f:.2f}%")

    torch.save({
        "model": model.state_dict(),
        "config": {
            "dim_latent": D,
            "n_rel": n_rel,
            "n_scale": n_scale,
            "hidden_gru": HIDDEN_GRU,
            "hidden_mlp": HIDDEN_MLP,
            "seq_len": SEQ_LEN,
            "future_k": FUTURE_K,
        }
    }, CKPT_OUT)
    print(f"[6.2] Saved GRU head -> {CKPT_OUT}")


if __name__ == "__main__":
    main()
