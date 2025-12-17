#!/usr/bin/env python3
# geomlang_temporal_imagine.py
#
# 6.3 – Imagined future frame visualization.
#
# Uses latents_dump.npz to train a tiny GRU that predicts the
# future relation/scale given the first T-1 frames, then renders
# canonical red/blue-shape scenes from the labels and shows:
#   [context frames ...] [true future] [imagined future]
# for a few random sequences.

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
from geomlang_viz_utils import channels_to_crisp_rgb

# -----------------------
# Config
# -----------------------
LATENTS_PATH     = os.path.join("outputs_edges", "latents_dump.npz")
SEQ_LEN          = 5          # total length T
BATCH_SIZE       = 128
SEQS_PER_CLASS   = 256        # how many sequences per (rel, scale) combo
N_EPOCHS         = 15
LR               = 1e-3
FUTURE_LAMBDA    = 1.0        # weight on future losses
N_VISUAL_SEQS    = 4          # how many sequences to visualize
IMG_SIZE         = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------
# Utilities
# -----------------------

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


def build_sequences(z, rel, scale, T=5, n_per_class=256):
    """
    Group by (rel, scale) and build sequences of length T by
    slicing shuffled indices within each combo.
    Returns:
        z_seq:     [M, T, D]
        rel_seq:   [M, T]
        scale_seq: [M, T]
    """
    z_cpu = z.detach().cpu()
    rel_np = rel.detach().cpu().numpy()
    scale_np = scale.detach().cpu().numpy()

    combos = {}
    N = z_cpu.shape[0]
    for i in range(N):
        key = (int(rel_np[i]), int(scale_np[i]))
        combos.setdefault(key, []).append(i)

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

    seqs = np.array(seqs, dtype=np.int64)  # [M, T]
    idx_tensor = torch.from_numpy(seqs).long()

    z_seq = z_cpu[idx_tensor]         # [M, T, D]
    rel_seq = rel[idx_tensor]         # [M, T]
    scale_seq = scale[idx_tensor]     # [M, T]

    return z_seq, rel_seq, scale_seq


# -----------------------
# Future GRU model
# -----------------------

class FutureGRU(nn.Module):
    def __init__(self, dim_latent, n_rel, n_scale, hidden=256, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=dim_latent,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
        )
        # Frame-wise heads
        self.head_rel_frame = nn.Linear(hidden, n_rel)
        self.head_scale_frame = nn.Linear(hidden, n_scale)
        # Future-at-T heads
        self.head_rel_future = nn.Linear(hidden, n_rel)
        self.head_scale_future = nn.Linear(hidden, n_scale)

    def forward(self, z_seq):
        """
        z_seq: [B, T, D]
        Returns:
            frame_rel_logits:   [B, T, n_rel]
            frame_scale_logits: [B, T, n_scale]
            future_rel_logits:  [B, n_rel]
            future_scale_logits:[B, n_scale]
        """
        out, _ = self.gru(z_seq)          # out: [B, T, H]
        last = out[:, -1, :]              # [B, H]

        frame_rel = self.head_rel_frame(out)
        frame_scale = self.head_scale_frame(out)

        future_rel = self.head_rel_future(last)
        future_scale = self.head_scale_future(last)

        return frame_rel, frame_scale, future_rel, future_scale

    @torch.no_grad()
    def predict_future(self, z_prefix):
        """
        Predict future labels from a prefix [B, Tp, D] (Tp <= T).
        """
        out, _ = self.gru(z_prefix)
        last = out[:, -1, :]
        future_rel = self.head_rel_future(last).argmax(dim=-1)      # [B]
        future_scale = self.head_scale_future(last).argmax(dim=-1)  # [B]
        return future_rel, future_scale


# -----------------------
# Canonical scene renderer
# -----------------------

REL_NAMES = {
    0: "red_left_blue_right",
    1: "red_right_blue_left",
    2: "red_above_blue",
    3: "red_below_blue",
    4: "red_inside_blue",
    5: "red_overlap_blue",
}

SCALE_NAMES = {
    0: "red_big_blue_small",
    1: "similar_size",
    2: "red_small_blue_big",
}


def draw_rect(img, cx, cy, size, channel):
    """
    Draw a filled square (with a thin edge) on img[channel, :, :].
    img: [3, H, W]
    """
    H, W = img.shape[1], img.shape[2]
    half = size // 2
    y0 = max(cy - half, 0)
    y1 = min(cy + half, H)
    x0 = max(cx - half, 0)
    x1 = min(cx + half, W)

    # fill
    img[channel, y0:y1, x0:x1] = 0.8
    # simple edge
    img[channel, y0:y1, x0:min(x0 + 1, W)] = 1.0
    img[channel, y0:y1, max(x1 - 1, 0):x1] = 1.0
    img[channel, y0:min(y0 + 1, H), x0:x1] = 1.0
    img[channel, max(y1 - 1, 0):y1, x0:x1] = 1.0


def render_scene(rel_label, scale_label, img_size=IMG_SIZE):
    """
    Render a canonical red/blue two-shape scene from (rel_label, scale_label).
    Returns float array [3, H, W] in [0,1].
    """
    img = np.zeros((3, img_size, img_size), dtype=np.float32)
    c = img_size // 2
    offset = img_size // 4

    # Map scale to sizes
    big = img_size // 3
    med = img_size // 4
    small = img_size // 6

    if scale_label == 0:   # red big, blue small
        red_size, blue_size = big, small
    elif scale_label == 1: # similar
        red_size, blue_size = med, med
    else:                  # red small, blue big
        red_size, blue_size = small, big

    # Default centers
    red_cx, red_cy = c - offset, c
    blue_cx, blue_cy = c + offset, c

    # Relation layout
    if rel_label == 0:      # red left, blue right
        red_cx, red_cy = c - offset, c
        blue_cx, blue_cy = c + offset, c
    elif rel_label == 1:    # red right, blue left
        red_cx, red_cy = c + offset, c
        blue_cx, blue_cy = c - offset, c
    elif rel_label == 2:    # red above blue
        red_cx, red_cy = c, c - offset
        blue_cx, blue_cy = c, c + offset
    elif rel_label == 3:    # red below blue
        red_cx, red_cy = c, c + offset
        blue_cx, blue_cy = c, c - offset
    elif rel_label == 4:    # red inside blue
        blue_cx, blue_cy = c, c
        red_cx, red_cy = c, c
        # ensure blue is at least as big
        blue_size = max(blue_size, red_size + img_size // 12)
    elif rel_label == 5:    # overlapping
        red_cx, red_cy = c - offset // 2, c
        blue_cx, blue_cy = c + offset // 2, c

    # Draw red on channel 0, blue on channel 1
    draw_rect(img, red_cx, red_cy, red_size, channel=0)
    draw_rect(img, blue_cx, blue_cy, blue_size, channel=1)
    # Edge channel (2) can be simple OR of red/blue edges – here just copy max
    img[2] = np.maximum(img[0], img[1])

    return img


# -----------------------
# Training
# -----------------------

def train_future_model(z_seq, rel_seq, scale_seq, n_rel, n_scale):
    dataset = TensorDataset(z_seq, rel_seq, scale_seq)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    dim_latent = z_seq.size(-1)
    model = FutureGRU(dim_latent, n_rel, n_scale, hidden=256).to(DEVICE)

    ce = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct_rel = correct_scale = 0
        correct_future_rel = correct_future_scale = 0
        total_frames = total_future = 0

        for z_b, r_b, s_b in loader:
            z_b = z_b.to(DEVICE)          # [B, T, D]
            r_b = r_b.to(DEVICE)          # [B, T]
            s_b = s_b.to(DEVICE)          # [B, T]

            frame_rel_logits, frame_scale_logits, future_rel_logits, future_scale_logits = model(z_b)

            # Frame-wise losses
            B, T, _ = frame_rel_logits.shape
            loss_rel = ce(frame_rel_logits.view(B * T, -1), r_b.view(-1))
            loss_scale = ce(frame_scale_logits.view(B * T, -1), s_b.view(-1))

            # Future@T losses (target is label at last frame)
            target_rel_T = r_b[:, -1]
            target_scale_T = s_b[:, -1]
            loss_f_rel = ce(future_rel_logits, target_rel_T)
            loss_f_scale = ce(future_scale_logits, target_scale_T)

            loss = loss_rel + loss_scale + FUTURE_LAMBDA * (loss_f_rel + loss_f_scale)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * B

            # Accuracies
            with torch.no_grad():
                frame_rel_pred = frame_rel_logits.argmax(dim=-1)
                frame_scale_pred = frame_scale_logits.argmax(dim=-1)
                correct_rel += (frame_rel_pred == r_b).sum().item()
                correct_scale += (frame_scale_pred == s_b).sum().item()
                total_frames += B * T

                future_rel_pred = future_rel_logits.argmax(dim=-1)
                future_scale_pred = future_scale_logits.argmax(dim=-1)
                correct_future_rel += (future_rel_pred == target_rel_T).sum().item()
                correct_future_scale += (future_scale_pred == target_scale_T).sum().item()
                total_future += B

        avg_loss = total_loss / len(dataset)
        frame_rel_acc = 100.0 * correct_rel / total_frames
        frame_scale_acc = 100.0 * correct_scale / total_frames
        fut_rel_acc = 100.0 * correct_future_rel / total_future
        fut_scale_acc = 100.0 * correct_future_scale / total_future

        print(
            f"[6.3-imagine] Epoch {epoch:2d}/{N_EPOCHS} | "
            f"Loss={avg_loss:.4f} | "
            f"Frame RelAcc={frame_rel_acc:.2f}% | Frame ScaleAcc={frame_scale_acc:.2f}% | "
            f"Future@T RelAcc={fut_rel_acc:.2f}% | Future@T ScaleAcc={fut_scale_acc:.2f}%"
        )

    return model


# -----------------------
# Visualization
# -----------------------

def visualize_imagination(model, z_seq, rel_seq, scale_seq, save_path):
    """
    Pick a few random sequences, show:
      [context frames] [true future] [imagined future]
    using canonical renderer.
    """
    model.eval()
    M, T, _ = z_seq.shape

    # pick random sequence indices
    idxs = np.random.choice(M, size=min(N_VISUAL_SEQS, M), replace=False)

    fig, axes = plt.subplots(
        len(idxs), T + 2,
        figsize=(2.2 * (T + 2), 2.2 * len(idxs))
    )

    if len(idxs) == 1:
        axes = axes[np.newaxis, :]  # make it 2D

    for row_idx, seq_idx in enumerate(idxs):
        z_s = z_seq[seq_idx:seq_idx + 1].to(DEVICE)   # [1, T, D]
        r_s = rel_seq[seq_idx]                        # [T]
        s_s = scale_seq[seq_idx]                      # [T]

        # Context = first T-1 frames
        z_prefix = z_s[:, :-1, :]                     # [1, T-1, D]
        with torch.no_grad():
            pred_rel_T, pred_scale_T = model.predict_future(z_prefix)
        pred_rel_T = int(pred_rel_T.item())
        pred_scale_T = int(pred_scale_T.item())

        true_rel_T = int(r_s[-1].item())
        true_scale_T = int(s_s[-1].item())

        print(
            f"[6.3-imagine] Seq {seq_idx}: "
            f"true (rel={true_rel_T}:{REL_NAMES.get(true_rel_T,'?')}, "
            f"scale={true_scale_T}:{SCALE_NAMES.get(true_scale_T,'?')}) | "
            f"pred (rel={pred_rel_T}:{REL_NAMES.get(pred_rel_T,'?')}, "
            f"scale={pred_scale_T}:{SCALE_NAMES.get(pred_scale_T,'?')})"
        )

        # Render context frames from their labels
        for t in range(T - 1):
            r_t = int(r_s[t].item())
            s_t = int(s_s[t].item())
            img = render_scene(r_t, s_t)
            ax = axes[row_idx, t]
            ax.imshow(np.transpose(img, (1, 2, 0)))
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(f"t={t+1}")

        # True future
        img_true = render_scene(true_rel_T, true_scale_T)
        ax_true = axes[row_idx, T - 1]
        ax_true.imshow(np.transpose(img_true, (1, 2, 0)))
        ax_true.set_xticks([])
        ax_true.set_yticks([])
        if row_idx == 0:
            ax_true.set_title("True T")

        # Imagined future
        img_pred = render_scene(pred_rel_T, pred_scale_T)
        ax_pred = axes[row_idx, T]
        ax_pred.imshow(np.transpose(img_pred, (1, 2, 0)))
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])
        if row_idx == 0:
            ax_pred.set_title("Pred T")

        # blank last column for spacing / or could duplicate pred
        axes[row_idx, T + 1].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[6.3-imagine] Saved imagined futures grid to: {save_path}")


# -----------------------
# Main
# -----------------------

def main():
    set_seed(42)
    print(f"[6.3-imagine] Using device: {DEVICE}")

    z, rel, scale = load_latents(LATENTS_PATH)
    N, D = z.shape
    n_rel = int(rel.max().item()) + 1
    n_scale = int(scale.max().item()) + 1

    print(
        f"[6.3-imagine] Loaded latents from {LATENTS_PATH}\n"
        f"  z: {z.shape}, rel range: {rel.min().item()}–{rel.max().item()}, "
        f"scale range: {scale.min().item()}–{scale.max().item()}\n"
        f"  n_rel={n_rel}, n_scale={n_scale}"
    )

    z_seq, rel_seq, scale_seq = build_sequences(
        z, rel, scale,
        T=SEQ_LEN,
        n_per_class=SEQS_PER_CLASS
    )
    print(
        f"[6.3-imagine] Built {z_seq.shape[0]} sequences, "
        f"T={SEQ_LEN}, latent dim={z_seq.shape[-1]}"
    )

    # Train future model
    model = train_future_model(z_seq, rel_seq, scale_seq, n_rel, n_scale)

    # Visualize imagined futures
    img_path = os.path.join("outputs_edges", "temporal_future_imagine.png")
    visualize_imagination(model, z_seq, rel_seq, scale_seq, img_path)


if __name__ == "__main__":
    main()
