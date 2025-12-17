#!/usr/bin/env python3
# geomlang_temporal.py
#
# 6.0 – Temporal geometry on *frozen* latents
#
# This script:
#   - Loads z + labels from outputs_edges/latents_dump.npz
#   - Reconstructs semantic directions (vert / horiz / inout / scale)
#   - Builds synthetic temporal sequences where (rel, scale) is constant
#   - Trains a TemporalAdapter(z) + TemporalHead(z') with:
#         L = CE_rel + CE_scale + λ_perm * L_perm
#     where L_perm penalizes changes over time in projections along
#     those semantic axes.
#
# IMPORTANT ASSUMPTIONS (adjust if needed):
#   - latents_dump.npz has keys:
#         "z"      : float32 array [N, D]
#         "rel"    : int64 array [N] with:
#               0: left_of
#               1: right_of
#               2: above
#               3: below
#               4: inside
#               5: overlapping
#         "scale"  : int64 array [N] with:
#               0: red_larger
#               1: red_smaller
#               2: similar
#
#   If the keys differ, run:
#       import numpy as np
#       d = np.load("outputs_edges/latents_dump.npz")
#       print(d.files)
#   and tweak LOAD_LATENTS() accordingly.

import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------
# Config
# -----------------------
LATENTS_PATH    = os.path.join("outputs_edges", "latents_dump.npz")

BATCH_SIZE      = 64
SEQ_LEN         = 5          # T
N_SEQS_PER_CLASS= 256        # how many sequences per (rel, scale) combo (best-effort)
EPOCHS          = 20

HIDDEN_ADAPT    = 256
HIDDEN_HEAD     = 256

LAMBDA_PERM     = 1e-2       # strength of permanence penalty (normalized)
LAMBDA_PERM_SWEEP = [0.0, 1e-3, 1e-2, 1e-1]  # tradeoff scan; set to None to disable sweep
SEED            = 42

DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------
# Utils
# -----------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True) + eps)


# -----------------------
# Load latents + labels
# -----------------------
def load_latents(path: str, device=DEVICE):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find '{path}'. "
            f"Run geomlang_edges_relscale.py first so it saves latents_dump.npz."
        )

    data = np.load(path)
    print("[6.0] Loaded latents_dump.npz with keys:", data.files)

    # Adjust these names if your file uses different keys.
    if "z" not in data or "rel" not in data or "scale" not in data:
        raise KeyError(
            "Expected keys 'z', 'rel', 'scale' in latents_dump.npz. "
            f"Found: {data.files}. Edit load_latents() to match your keys."
        )

    z_np     = data["z"]      # [N, D]
    rel_np   = data["rel"]    # [N]
    scale_np = data["scale"]  # [N]

    z     = torch.from_numpy(z_np).float().to(device)
    rel   = torch.from_numpy(rel_np).long().to(device)
    scale = torch.from_numpy(scale_np).long().to(device)

    N, D = z.shape
    print(f"[6.0] z shape: N={N}, D={D}")
    print(f"[6.0] rel unique labels:", torch.unique(rel).cpu().tolist())
    print(f"[6.0] scale unique labels:", torch.unique(scale).cpu().tolist())

    return z, rel, scale


# -----------------------
# Semantic directions
# -----------------------
def compute_semantic_axes(z, rel, scale):
    """
    Compute mean latents per relation and scale label,
    then build semantic directions:

        vert_axis  ~ above - below
        horiz_axis ~ right_of - left_of
        inout_axis ~ inside - overlapping
        scale_axis ~ red_larger - red_smaller

    Assumes label indices:
        rel:   0=left_of, 1=right_of, 2=above, 3=below, 4=inside, 5=overlapping
        scale: 0=red_larger, 1=red_smaller, 2=similar
    """

    n_rel   = int(torch.max(rel).item()) + 1
    n_scale = int(torch.max(scale).item()) + 1

    D = z.shape[1]

    # Means per relation
    rel_means = []
    for r in range(n_rel):
        mask = (rel == r)
        if mask.sum() == 0:
            rel_means.append(torch.zeros(D, device=z.device))
        else:
            rel_means.append(z[mask].mean(dim=0))
    rel_means = torch.stack(rel_means, dim=0)  # [n_rel, D]

    # Means per scale
    scale_means = []
    for s in range(n_scale):
        mask = (scale == s)
        if mask.sum() == 0:
            scale_means.append(torch.zeros(D, device=z.device))
        else:
            scale_means.append(z[mask].mean(dim=0))
    scale_means = torch.stack(scale_means, dim=0)  # [n_scale, D]

    # Relation label indices (based on your earlier logs)
    LEFT, RIGHT, ABOVE, BELOW, INSIDE, OVERLAP = 0, 1, 2, 3, 4, 5
    RED_LARGER, RED_SMALLER, SIMILAR = 0, 1, 2

    vert_axis  = normalize(rel_means[ABOVE]   - rel_means[BELOW])    # up/down
    horiz_axis = normalize(rel_means[RIGHT]   - rel_means[LEFT])     # left/right
    inout_axis = normalize(rel_means[INSIDE]  - rel_means[OVERLAP])  # containment
    scale_axis = normalize(scale_means[RED_LARGER] - scale_means[RED_SMALLER])

    axes = torch.stack([vert_axis, horiz_axis, inout_axis, scale_axis], dim=0)
    names = ["vertical (above-below)",
             "horizontal (right-left)",
             "containment (inside-overlapping)",
             "scale (red_larger-red_smaller)"]

    print("[6.0] Semantic axes norms:",
          [float(a.norm().item()) for a in axes])

    return axes, names


# -----------------------
# Temporal dataset (on z)
# -----------------------
class TemporalLatentDataset(Dataset):
    """
    Each item is:
        z_seq      : [T, D]
        rel_seq    : [T]
        scale_seq  : [T]
    where z_seq consists of frames with the same (rel, scale) label
    gathered from different examples.
    """

    def __init__(self, z, rel, scale, seq_len=5, n_seqs_per_class=256):
        super().__init__()
        self.z = z
        self.rel = rel
        self.scale = scale
        self.seq_len = seq_len

        self.indices = self._build_sequences(z, rel, scale, seq_len, n_seqs_per_class)

    def _build_sequences(self, z, rel, scale, T, n_seqs_per_class):
        z = z.detach().cpu()
        rel = rel.detach().cpu().numpy()
        scale = scale.detach().cpu().numpy()

        N = z.shape[0]
        all_indices = np.arange(N)

        # Unique (rel, scale) combos
        combos = {}
        for i in range(N):
            key = (int(rel[i]), int(scale[i]))
            combos.setdefault(key, []).append(i)

        sequences = []

        print("[6.0] Building sequences...")
        for (r, s), idxs in combos.items():
            idxs = np.array(idxs)
            if len(idxs) < T:
                continue

            np.random.shuffle(idxs)
            # We can generate multiple disjoint sequences per class
            max_seqs_here = min(n_seqs_per_class, len(idxs) // T)
            for j in range(max_seqs_here):
                start = j * T
                end = start + T
                if end > len(idxs):
                    break
                seq = idxs[start:end]
                if len(seq) == T:
                    sequences.append(seq)

        print(f"[6.0] Built {len(sequences)} sequences (T={T})")
        return np.array(sequences, dtype=np.int64)  # [N_seq, T]

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, idx):
        idx_seq = self.indices[idx]   # [T]
        z_seq = self.z[idx_seq]       # [T, D]
        r = self.rel[idx_seq]         # [T]
        s = self.scale[idx_seq]       # [T]
        return z_seq, r, s


# -----------------------
# Temporal adapter + head
# -----------------------
class TemporalAdapter(nn.Module):
    """
    Residual adapter on z:
        z' = z + f(z)
    """

    def __init__(self, dim_latent, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_latent, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim_latent),
        )

    def forward(self, z):
        # z: [..., D]
        return z + self.net(z)


class TemporalHead(nn.Module):
    """
    Simple per-frame head:
        - shared MLP -> relation logits & scale logits
    """

    def __init__(self, dim_latent, n_rel, n_scale, hidden=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_latent, hidden),
            nn.ReLU(inplace=True),
        )
        self.rel_head = nn.Linear(hidden, n_rel)
        self.scale_head = nn.Linear(hidden, n_scale)

    def forward(self, z_prime):
        """
        z_prime: [B, T, D]
        returns:
            rel_logits   : [B, T, n_rel]
            scale_logits : [B, T, n_scale]
        """
        B, T, D = z_prime.shape
        x = z_prime.view(B * T, D)
        x = self.mlp(x)
        rel_logits = self.rel_head(x).view(B, T, -1)
        scale_logits = self.scale_head(x).view(B, T, -1)
        return rel_logits, scale_logits


# -----------------------
# Permanence loss
# -----------------------
def permanence_loss(z_prime, axes):
    """
    z_prime: [B, T, D]  (adapter output)
    axes   : [K, D]     (semantic unit vectors)

    L_perm = average over axes of mean squared Δ projection over time.
    """

    B, T, D = z_prime.shape
    K = axes.shape[0]

    # [B, T, D] -> [B, T, D] (already)
    losses = []
    for k in range(K):
        a = axes[k]                       # [D]
        # [B, T, D] @ [D] -> [B, T]
        proj = torch.matmul(z_prime, a)   # [B, T]
        delta = proj[:, 1:] - proj[:, :-1]  # [B, T-1]
        losses.append((delta ** 2).mean())

    return sum(losses) / len(losses)


def permanence_loss_with_axes(z_prime: torch.Tensor, axes: torch.Tensor):
    """
    Return mean loss and per-axis losses (L1 and L2).
    """
    l1s = []
    l2s = []
    for k in range(axes.size(0)):
        a = axes[k]
        proj = torch.matmul(z_prime, a)
        delta = proj[:, 1:] - proj[:, :-1]
        l1s.append(delta.abs().mean())
        l2s.append((delta ** 2).mean())
    mean_l2 = sum(l2s) / len(l2s)
    return mean_l2, [x.item() for x in l1s], [x.item() for x in l2s]


def projection_drift(z_prime: torch.Tensor, axes: torch.Tensor):
    """
    Mean |Δ projection| per axis over time.
    """
    drifts = []
    for k in range(axes.size(0)):
        a = axes[k]
        proj = torch.matmul(z_prime, a)
        delta = proj[:, 1:] - proj[:, :-1]
        drifts.append(delta.abs().mean().item())
    return drifts


# -----------------------
# Evaluation helpers
# -----------------------
def evaluate_static(z, rel, scale, adapter, head, batch_size=256):
    """
    Evaluate relation/scale accuracy using single frames (T=1),
    on all latents z.
    """
    adapter.eval()
    head.eval()

    n_rel   = int(torch.max(rel).item()) + 1
    n_scale = int(torch.max(scale).item()) + 1

    N = z.shape[0]
    correct_rel = 0
    correct_scale = 0
    total = 0

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(N, start + batch_size)
            z_batch = z[start:end]                 # [B, D]
            r_batch = rel[start:end]
            s_batch = scale[start:end]

            z_prime = adapter(z_batch)             # [B, D]
            z_prime = z_prime.unsqueeze(1)         # [B, 1, D]

            rel_logits, scale_logits = head(z_prime)  # [B, 1, C]
            rel_pred   = rel_logits.argmax(dim=-1).squeeze(1)   # [B]
            scale_pred = scale_logits.argmax(dim=-1).squeeze(1) # [B]

            correct_rel   += (rel_pred == r_batch).sum().item()
            correct_scale += (scale_pred == s_batch).sum().item()
            total += (end - start)

    acc_rel   = 100.0 * correct_rel / total
    acc_scale = 100.0 * correct_scale / total
    return acc_rel, acc_scale


# -----------------------
# Main
# -----------------------
def main():
    set_seed(SEED)
    print(f"[6.0] Using device: {DEVICE}")

    # 1) Load latents and labels.
    z, rel, scale = load_latents(LATENTS_PATH, DEVICE)
    N, D = z.shape
    n_rel   = int(torch.max(rel).item()) + 1
    n_scale = int(torch.max(scale).item()) + 1

    # 2) Compute semantic axes.
    axes, axis_names = compute_semantic_axes(z, rel, scale)

    # 3) Build temporal dataset + loader.
    dataset = TemporalLatentDataset(
        z, rel, scale,
        seq_len=SEQ_LEN,
        n_seqs_per_class=N_SEQS_PER_CLASS
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    # 4) Sweep over permanence weights (or single value)
    lambdas = LAMBDA_PERM_SWEEP if LAMBDA_PERM_SWEEP else [LAMBDA_PERM]

    for lam in lambdas:
        print(f"\n[6.0] ==== Lambda_perm = {lam} ====")

        # fresh adapter/head
        adapter = TemporalAdapter(D, hidden=HIDDEN_ADAPT).to(DEVICE)
        head    = TemporalHead(D, n_rel, n_scale, hidden=HIDDEN_HEAD).to(DEVICE)
        params = list(adapter.parameters()) + list(head.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)

        # Baseline static accuracy before temporal training.
        print("[6.0] Evaluating static accuracy before temporal training...")
        acc_rel_0, acc_scale_0 = evaluate_static(z, rel, scale, adapter, head)
        print(f"[6.0] Baseline (untrained head) – RelAcc: {acc_rel_0:.2f}% | ScaleAcc: {acc_scale_0:.2f}%")

        # Baseline drift (adapter as identity)
        with torch.no_grad():
            z_id = z.view(-1, 1, D).expand(-1, SEQ_LEN, -1)  # fake sequence copies
            drift_before = projection_drift(z_id, axes)
            print(f"[6.0] Drift before training (mean |Δ| per axis): {drift_before}")

        # Train temporal adapter/head with permanence.
        print("[6.0] Starting temporal training...")
        for epoch in range(1, EPOCHS + 1):
            adapter.train()
            head.train()

            total_loss = 0.0
            total_ce_rel = 0.0
            total_ce_scale = 0.0
            total_perm = 0.0
            per_axis_l1 = None

            correct_rel = 0
            correct_scale = 0
            total_frames = 0

            for z_seq, r_seq, s_seq in loader:
                B, T, D_ = z_seq.shape
                assert D_ == D

                z_seq = z_seq.to(DEVICE)
                r_seq = r_seq.to(DEVICE)
                s_seq = s_seq.to(DEVICE)

                opt.zero_grad()

                z_prime = adapter(z_seq)  # [B, T, D]

                rel_logits, scale_logits = head(z_prime)  # [B, T, n_rel], [B, T, n_scale]

                # Flatten for CE
                rel_flat   = r_seq.view(-1)
                scale_flat = s_seq.view(-1)
                rel_logit_flat   = rel_logits.view(B * T, n_rel)
                scale_logit_flat = scale_logits.view(B * T, n_scale)

                ce_rel   = F.cross_entropy(rel_logit_flat, rel_flat)
                ce_scale = F.cross_entropy(scale_logit_flat, scale_flat)

                # Permanence loss on z'
                perm_mean, perm_l1_axes, perm_l2_axes = permanence_loss_with_axes(z_prime, axes)
                per_axis_l1 = perm_l1_axes

                loss = ce_rel + ce_scale + lam * perm_mean
                loss.backward()
                opt.step()

                total_loss += loss.item() * B
                total_ce_rel += ce_rel.item() * B
                total_ce_scale += ce_scale.item() * B
                total_perm += perm_mean.item() * B

                # Seq accuracy (per-frame)
                with torch.no_grad():
                    rel_pred = rel_logits.argmax(dim=-1)     # [B, T]
                    scale_pred = scale_logits.argmax(dim=-1) # [B, T]

                    correct_rel   += (rel_pred == r_seq).sum().item()
                    correct_scale += (scale_pred == s_seq).sum().item()
                    total_frames  += B * T

            avg_loss     = total_loss / len(dataset)
            avg_ce_rel   = total_ce_rel / len(dataset)
            avg_ce_scale = total_ce_scale / len(dataset)
            avg_perm     = total_perm / len(dataset)

            acc_rel_seq   = 100.0 * correct_rel / total_frames
            acc_scale_seq = 100.0 * correct_scale / total_frames

            print(
                f"[6.0] Epoch {epoch:2d}/{EPOCHS} | "
                f"Loss={avg_loss:.4f} | CE_rel={avg_ce_rel:.4f} | CE_scale={avg_ce_scale:.4f} | "
                f"Perm={avg_perm:.4f} | SeqRelAcc={acc_rel_seq:.2f}% | SeqScaleAcc={acc_scale_seq:.2f}%"
            )
            if per_axis_l1 is not None and epoch == EPOCHS:
                print(f"[6.0] Per-axis mean |Δ| at end: {per_axis_l1}")

        # Final static accuracy with trained temporal adapter/head.
        print("[6.0] Evaluating static accuracy AFTER temporal training...")
        acc_rel_1, acc_scale_1 = evaluate_static(z, rel, scale, adapter, head)
        print(f"[6.0] After training – RelAcc: {acc_rel_1:.2f}% | ScaleAcc: {acc_scale_1:.2f}%")

        # Drift after training
        with torch.no_grad():
            # Use a few batches to estimate drift
            drift_list = []
            for z_seq, _, _ in loader:
                z_seq = z_seq.to(DEVICE)
                z_prime = adapter(z_seq)
                drift_list.append(projection_drift(z_prime, axes))
                if len(drift_list) >= 5:
                    break
            drift_after = np.mean(np.array(drift_list), axis=0).tolist()
            print(f"[6.0] Drift after training (mean |Δ| per axis): {drift_after}")
            print(f"[6.0] Drift improvement (before → after):")
            for name, b, a in zip(axis_names, drift_before, drift_after):
                print(f"    {name}: {b:.3f} → {a:.3f}")

        # Save adapter + head for later analysis.
        out_dir = "outputs_edges"
        os.makedirs(out_dir, exist_ok=True)
        torch.save(
            {
                "adapter": adapter.state_dict(),
                "head": head.state_dict(),
                "config": {
                    "dim_latent": D,
                    "n_rel": n_rel,
                    "n_scale": n_scale,
                    "hidden_adapt": HIDDEN_ADAPT,
                    "hidden_head": HIDDEN_HEAD,
                    "lambda_perm": lam,
                    "seq_len": SEQ_LEN,
                    "n_seqs_per_class": N_SEQS_PER_CLASS,
                },
            },
            os.path.join(out_dir, f"temporal_adapter_head_perm{lam}.pt"),
        )
        print(f"[6.0] Saved temporal adapter+head -> outputs_edges/temporal_adapter_head_perm{lam}.pt")


if __name__ == "__main__":
    main()
