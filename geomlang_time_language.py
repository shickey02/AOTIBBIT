#!/usr/bin/env python3
"""
geomlang_time_language.py

BBIT + Time + Language Bridge

We:
  - Generate 3D voxel scenes over time (T frames) with a moving primitive.
  - Motions:
      0 = fall      (downwards in -y)
      1 = rise      (upwards in +y)
      2 = slide_lr  (left -> right in +x)
      3 = slide_rl  (right -> left in -x)
  - Train:
      * encoder: sequence -> latent z (R^24)
      * decoder: z -> sequence
      * classifier: z -> motion label (4-way)
  - Then learn a language bridge:
      * E_motion: nn.Embedding(4, D_word)
      * mapper: z -> e_pred
    with MSE(e_pred, E_motion[label]) loss.

This aligns the motion manifold with a small "verb" space.
"""

import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

GRID_SIZE = 16
T_FRAMES = 8
MOTIONS = ["fall", "rise", "slide_lr", "slide_rl"]


# ------------------------------
# Primitive generator (single frame)
# ------------------------------

def make_empty():
    return np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.float32)


def make_sphere(radius_frac=0.25, center=None):
    grid = make_empty()
    if center is None:
        center = ((GRID_SIZE - 1) / 2.0,) * 3
    cx, cy, cz = center
    r = GRID_SIZE * radius_frac
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            for z in range(GRID_SIZE):
                dx = x - cx
                dy = y - cy
                dz = z - cz
                if math.sqrt(dx * dx + dy * dy + dz * dz) <= r:
                    grid[x, y, z] = 1.0
    return grid


def make_cube(size_frac=0.35, center=None):
    grid = make_empty()
    if center is None:
        center = ((GRID_SIZE - 1) / 2.0,) * 3
    cx, cy, cz = center
    size = int(GRID_SIZE * size_frac)
    half = size // 2
    x0 = max(0, int(cx - half))
    x1 = min(GRID_SIZE, int(cx + half))
    y0 = max(0, int(cy - half))
    y1 = min(GRID_SIZE, int(cy + half))
    z0 = max(0, int(cz - half))
    z1 = min(GRID_SIZE, int(cz + half))
    grid[x0:x1, y0:y1, z0:z1] = 1.0
    return grid


def sample_primitive(center, shape_type=None):
    if shape_type is None:
        shape_type = np.random.choice(["sphere", "cube"])
    if shape_type == "sphere":
        radius = np.random.uniform(0.18, 0.30)
        return make_sphere(radius_frac=radius, center=center)
    else:
        size = np.random.uniform(0.25, 0.4)
        return make_cube(size_frac=size, center=center)


# ------------------------------
# Motion sequence generator
# ------------------------------

def generate_motion_sequence(motion_id, T=T_FRAMES):
    """
    Return sequence of shape (T, X, Y, Z) for one primitive moving
    according to the motion type.
    """
    seq = []
    # Start near the center, with some jitter
    base_center = np.array([
        GRID_SIZE // 2 + np.random.randint(-2, 3),
        GRID_SIZE // 2 + np.random.randint(-2, 3),
        GRID_SIZE // 2 + np.random.randint(-2, 3),
    ])
    shape_type = np.random.choice(["sphere", "cube"])
    # motion step size
    step = np.random.randint(1, 3)

    for t in range(T):
        if motion_id == 0:  # fall: downwards in -y
            center = base_center + np.array([0, -step * t, 0])
        elif motion_id == 1:  # rise: upwards in +y
            center = base_center + np.array([0, step * t, 0])
        elif motion_id == 2:  # slide_lr: +x
            center = base_center + np.array([step * t, 0, 0])
        else:  # slide_rl: -x
            center = base_center + np.array([-step * t, 0, 0])

        center = np.clip(center, 2, GRID_SIZE - 3)
        frame = sample_primitive(tuple(center), shape_type=shape_type)
        seq.append(frame)

    return np.stack(seq, axis=0)  # (T, X, Y, Z)


def generate_motion_dataset(n_per_motion=300):
    sequences = []
    labels = []
    for motion_id in range(len(MOTIONS)):
        for _ in range(n_per_motion):
            seq = generate_motion_sequence(motion_id)
            sequences.append(seq)
            labels.append(motion_id)
    sequences_np = np.stack(sequences, axis=0)  # (N, T, X, Y, Z)
    labels_np = np.array(labels, dtype=np.int64)
    return sequences_np, labels_np


# ------------------------------
# Models
# ------------------------------

class TimeEncoder(nn.Module):
    def __init__(self, latent_dim=24, T=T_FRAMES, grid_size=GRID_SIZE):
        super().__init__()
        self.T = T
        self.grid_size = grid_size
        in_dim = T * grid_size * grid_size * grid_size
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x):
        # x: (B, T, X, Y, Z)
        b = x.shape[0]
        x_flat = x.view(b, -1)
        z = self.net(x_flat)
        return z


class TimeDecoder(nn.Module):
    def __init__(self, latent_dim=24, T=T_FRAMES, grid_size=GRID_SIZE):
        super().__init__()
        self.T = T
        self.grid_size = grid_size
        out_dim = T * grid_size * grid_size * grid_size
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, z):
        # z: (B, latent_dim)
        b = z.shape[0]
        x = self.net(z)
        x = x.view(b, self.T, self.grid_size, self.grid_size, self.grid_size)
        return x


class MotionModel(nn.Module):
    def __init__(self, latent_dim=24, T=T_FRAMES, grid_size=GRID_SIZE, n_motions=4):
        super().__init__()
        self.encoder = TimeEncoder(latent_dim=latent_dim, T=T, grid_size=grid_size)
        self.decoder = TimeDecoder(latent_dim=latent_dim, T=T, grid_size=grid_size)
        self.classifier = nn.Linear(latent_dim, n_motions)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        logits = self.classifier(z)
        return recon, logits, z


# ------------------------------
# Train motion model
# ------------------------------

def train_motion_model(num_epochs=800, latent_dim=24, lr=1e-3, beta=0.4, device="cpu"):
    seq_np, labels_np = generate_motion_dataset(n_per_motion=300)
    seq_t = torch.from_numpy(seq_np).to(device)
    labels_t = torch.from_numpy(labels_np).to(device)

    model = MotionModel(latent_dim=latent_dim,
                        T=T_FRAMES,
                        grid_size=GRID_SIZE,
                        n_motions=len(MOTIONS)).to(device)

    bce = nn.BCELoss()
    ce = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        opt.zero_grad()
        recon, logits, z = model(seq_t)
        loss_recon = bce(recon, seq_t)
        loss_cls = ce(logits, labels_t)
        loss = loss_recon + beta * loss_cls
        loss.backward()
        opt.step()

        if epoch % 100 == 0 or epoch == num_epochs - 1:
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                acc = (preds == labels_t).float().mean().item()
            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"Recon: {loss_recon.item():.6f} | "
                f"MotionCE: {loss_cls.item():.6f} | "
                f"Total: {loss.item():.6f} | "
                f"Acc: {acc*100:.2f}%"
            )

    model.eval()
    with torch.no_grad():
        recon, logits, z = model(seq_t)
        loss_recon = bce(recon, seq_t).item()
        loss_cls = ce(logits, labels_t).item()
        preds = logits.argmax(dim=1)
        acc = (preds == labels_t).float().mean().item()

    print("\nFinal motion metrics:")
    print(f"  Recon BCE: {loss_recon:.6f}")
    print(f"  Motion CE: {loss_cls:.6f}")
    print(f"  Accuracy:  {acc*100:.2f}%")

    z_np = z.cpu().numpy()
    labels_np = labels_np
    means = []
    for m in range(len(MOTIONS)):
        mask = (labels_np == m)
        mean_z = z_np[mask].mean(axis=0)
        means.append(mean_z)
        print(f"Mean latent for motion {m} ({MOTIONS[m]}):")
        print("  ", mean_z)

    print("\nPairwise distances between motion means (L2):")
    for i in range(len(MOTIONS)):
        for j in range(i + 1, len(MOTIONS)):
            d = np.linalg.norm(means[i] - means[j])
            print(f"  {MOTIONS[i]} <-> {MOTIONS[j]}: {d:.4f}")

    return model, seq_np, labels_np, means, z.detach(), labels_t


# ------------------------------
# Language bridge for motion
# ------------------------------

class ZToWordMapper(nn.Module):
    def __init__(self, latent_dim=24, word_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, word_dim),
        )

    def forward(self, z):
        return self.net(z)


def train_motion_language_bridge(z_all,
                                 labels_all,
                                 word_dim=8,
                                 num_epochs=400,
                                 lr=1e-3,
                                 device="cpu"):
    latent_dim = z_all.shape[1]
    z_all = z_all.to(device)
    labels_all = labels_all.to(device)

    E_motion = nn.Embedding(len(MOTIONS), word_dim).to(device)
    mapper = ZToWordMapper(latent_dim=latent_dim, word_dim=word_dim).to(device)

    params = list(E_motion.parameters()) + list(mapper.parameters())
    opt = optim.Adam(params, lr=lr)
    mse = nn.MSELoss()

    for epoch in range(num_epochs):
        mapper.train()
        E_motion.train()
        opt.zero_grad()

        e_true = E_motion(labels_all)
        e_pred = mapper(z_all)
        loss = mse(e_pred, e_true)
        loss.backward()
        opt.step()

        if epoch % 100 == 0 or epoch == num_epochs - 1:
            print(f"[MotionLang] Epoch {epoch}/{num_epochs} | MSE: {loss.item():.6f}")

    E_motion.eval()
    mapper.eval()

    with torch.no_grad():
        motion_embs = E_motion(torch.arange(len(MOTIONS), device=device))
        motion_embs_np = motion_embs.cpu().numpy()

    print("\nMotion word embedding vectors:")
    for i, name in enumerate(MOTIONS):
        print(f"  {name}: {motion_embs_np[i]}")

    print("\nPairwise distances in motion word-embedding space (L2):")
    for i in range(len(MOTIONS)):
        for j in range(i + 1, len(MOTIONS)):
            d = np.linalg.norm(motion_embs_np[i] - motion_embs_np[j])
            print(f"  {MOTIONS[i]} <-> {MOTIONS[j]}: {d:.4f}")

    # Relation vectors in latent vs word space for fall->rise, slide_lr->slide_rl
    z_np = z_all.cpu().numpy()
    labels_np = labels_all.cpu().numpy()
    z_means = []
    for m in range(len(MOTIONS)):
        mask = (labels_np == m)
        z_means.append(z_np[mask].mean(axis=0))
    z_means = np.stack(z_means, axis=0)

    def rel_vec_z(a, b):
        return z_means[b] - z_means[a]

    def rel_vec_w(a, b):
        return motion_embs_np[b] - motion_embs_np[a]

    idx_fall = MOTIONS.index("fall")
    idx_rise = MOTIONS.index("rise")
    idx_lr = MOTIONS.index("slide_lr")
    idx_rl = MOTIONS.index("slide_rl")

    v_fr_z = rel_vec_z(idx_fall, idx_rise)
    v_fr_w = rel_vec_w(idx_fall, idx_rise)
    v_sl_z = rel_vec_z(idx_lr, idx_rl)
    v_sl_w = rel_vec_w(idx_lr, idx_rl)

    def cos_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

    print("\n[fall -> rise] relation vector:")
    print("  latent space norm:", np.linalg.norm(v_fr_z))
    print("  word space norm:  ", np.linalg.norm(v_fr_w))

    print("\n[slide_lr -> slide_rl] relation vector:")
    print("  latent space norm:", np.linalg.norm(v_sl_z))
    print("  word space norm:  ", np.linalg.norm(v_sl_w))

    print("\nCosine similarity between motion directions in word space:")
    print("  cos(v_fr_w, v_sl_w):", cos_sim(v_fr_w, v_sl_w))

    return E_motion, mapper


# ------------------------------
# Main
# ------------------------------

def main():
    device = "cpu"
    print(f"Using device: {device}")
    os.makedirs("outputs_time_lang", exist_ok=True)

    # 1) Train motion model
    model, seq_np, labels_np, means, z_all, labels_t = train_motion_model(
        num_epochs=800,
        latent_dim=24,
        lr=1e-3,
        beta=0.4,
        device=device,
    )

    # 2) Train language bridge on motion latents
    print("\n=== Training motion language bridge ===")
    E_motion, mapper = train_motion_language_bridge(
        z_all=z_all,
        labels_all=labels_t,
        word_dim=8,
        num_epochs=400,
        lr=1e-3,
        device=device,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
