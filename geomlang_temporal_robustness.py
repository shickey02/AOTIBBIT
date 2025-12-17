#!/usr/bin/env python3
# geomlang_temporal_robustness.py
#
# 6.1 – Robustness of temporal adapter/head to shuffling and frame drop.
#
# Loads:
#   - outputs_edges/latents_dump.npz (z, rel, scale)
#   - outputs_edges/temporal_adapter_head_perm0.01.pt (or another checkpoint)
# Rebuilds latent sequences and measures per-frame accuracy under:
#   - normal order
#   - shuffled order
#   - random frame drop (mask out frames)

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

LATENTS_PATH   = os.path.join("outputs_edges", "latents_dump.npz")
CKPT_PATH      = os.path.join("outputs_edges", "temporal_adapter_head_perm0.01.pt")
SEQ_LEN        = 5
BATCH_SIZE     = 128
SEQS_PER_CLASS = 256
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_latents(path=LATENTS_PATH, device=DEVICE):
    data = np.load(path)
    z = torch.from_numpy(data["z"]).float().to(device)
    rel = torch.from_numpy(data["rel"]).long().to(device)
    scale = torch.from_numpy(data["scale"]).long().to(device)
    return z, rel, scale


class TemporalLatentDataset(Dataset):
    def __init__(self, z, rel, scale, T=5, n_per_class=256):
        super().__init__()
        self.z = z
        self.rel = rel
        self.scale = scale
        self.indices = self._build(z, rel, scale, T, n_per_class)

    def _build(self, z, rel, scale, T, n_per_class):
        z = z.detach().cpu()
        rel = rel.detach().cpu().numpy()
        scale = scale.detach().cpu().numpy()
        combos = {}
        for i in range(z.shape[0]):
            combos.setdefault((int(rel[i]), int(scale[i])), []).append(i)
        seqs = []
        for (r, s), idxs in combos.items():
            idxs = np.array(idxs)
            if len(idxs) < T:
                continue
            np.random.shuffle(idxs)
            max_seqs = min(n_per_class, len(idxs) // T)
            for j in range(max_seqs):
                start = j * T
                end = start + T
                seq = idxs[start:end]
                if len(seq) == T:
                    seqs.append(seq)
        return np.array(seqs, dtype=np.int64)

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, idx):
        idx_seq = self.indices[idx]
        return (
            self.z[idx_seq],       # [T, D]
            self.rel[idx_seq],     # [T]
            self.scale[idx_seq],   # [T]
        )


class TemporalAdapter(nn.Module):
    def __init__(self, dim_latent, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_latent, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim_latent),
        )

    def forward(self, z):
        return z + self.net(z)


class TemporalHead(nn.Module):
    def __init__(self, dim_latent, n_rel, n_scale, hidden=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_latent, hidden),
            nn.ReLU(inplace=True),
        )
        self.rel_head = nn.Linear(hidden, n_rel)
        self.scale_head = nn.Linear(hidden, n_scale)

    def forward(self, z_prime):
        B, T, D = z_prime.shape
        x = z_prime.view(B * T, D)
        x = self.mlp(x)
        rel_logits = self.rel_head(x).view(B, T, -1)
        scale_logits = self.scale_head(x).view(B, T, -1)
        return rel_logits, scale_logits


def load_adapter_head(path, dim_latent, n_rel, n_scale):
    ckpt = torch.load(path, map_location=DEVICE)
    adapter = TemporalAdapter(dim_latent, hidden=ckpt["config"].get("hidden_adapt", 256)).to(DEVICE)
    head = TemporalHead(dim_latent, n_rel, n_scale, hidden=ckpt["config"].get("hidden_head", 256)).to(DEVICE)
    adapter.load_state_dict(ckpt["adapter"])
    head.load_state_dict(ckpt["head"])
    adapter.eval()
    head.eval()
    return adapter, head


@torch.no_grad()
def eval_setting(loader, adapter, head, shuffle=False, drop_prob=0.0):
    correct_rel = correct_scale = total = 0
    for z_seq, r_seq, s_seq in loader:
        z_seq = z_seq.to(DEVICE)
        r_seq = r_seq.to(DEVICE)
        s_seq = s_seq.to(DEVICE)

        # optionally shuffle time
        if shuffle:
            perm = torch.stack([torch.randperm(z_seq.size(1)) for _ in range(z_seq.size(0))]).to(DEVICE)
            z_seq = torch.gather(z_seq, 1, perm.unsqueeze(-1).expand_as(z_seq))
            r_seq = torch.gather(r_seq, 1, perm)
            s_seq = torch.gather(s_seq, 1, perm)

        # optionally drop frames (mask out)
        if drop_prob > 0.0:
            mask = (torch.rand_like(r_seq.float()) > drop_prob).float().unsqueeze(-1)  # [B,T,1]
            z_seq = z_seq * mask  # zero out dropped frames

        z_prime = adapter(z_seq)
        rel_logits, scale_logits = head(z_prime)
        rel_pred = rel_logits.argmax(dim=-1)
        scale_pred = scale_logits.argmax(dim=-1)

        correct_rel += (rel_pred == r_seq).sum().item()
        correct_scale += (scale_pred == s_seq).sum().item()
        total += r_seq.numel()

    acc_rel = 100.0 * correct_rel / total
    acc_scale = 100.0 * correct_scale / total
    return acc_rel, acc_scale


def main():
    set_seed(42)
    print(f"[6.1] Using device: {DEVICE}")

    z, rel, scale = load_latents(LATENTS_PATH, DEVICE)
    N, D = z.shape
    n_rel = int(torch.max(rel).item()) + 1
    n_scale = int(torch.max(scale).item()) + 1

    dataset = TemporalLatentDataset(z, rel, scale, T=SEQ_LEN, n_per_class=SEQS_PER_CLASS)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    adapter, head = load_adapter_head(CKPT_PATH, D, n_rel, n_scale)

    # Normal order
    acc_rel_norm, acc_scale_norm = eval_setting(loader, adapter, head, shuffle=False, drop_prob=0.0)
    # Shuffled
    acc_rel_shuffle, acc_scale_shuffle = eval_setting(loader, adapter, head, shuffle=True, drop_prob=0.0)
    # Drop 30% frames
    acc_rel_drop, acc_scale_drop = eval_setting(loader, adapter, head, shuffle=False, drop_prob=0.3)

    print("\n[6.1] Robustness summary (per-frame acc):")
    print(f"  Normal   : RelAcc={acc_rel_norm:.2f}% | ScaleAcc={acc_scale_norm:.2f}%")
    print(f"  Shuffled : RelAcc={acc_rel_shuffle:.2f}% | ScaleAcc={acc_scale_shuffle:.2f}%")
    print(f"  Drop30%  : RelAcc={acc_rel_drop:.2f}% | ScaleAcc={acc_scale_drop:.2f}%")


if __name__ == "__main__":
    main()
