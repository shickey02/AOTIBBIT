#!/usr/bin/env python3
"""
geomlang_sentences.py

BBIT – unified "sentence" model:
  - Two objects in a 3D grid, over time.
  - One spatial relation:   left_of, right_of, above, below, inside, overlapping
  - One motion type:        fall, rise, slide_lr, slide_rl
  - One object moves according to the motion; the other stays fixed.

We train:
  * encoder:   sequence -> latent z (R^32)
  * decoder:   z -> sequence
  * rel_head:  z -> relation label (6-way)
  * mot_head:  z -> motion label   (4-way)

Then we inspect:
  - how well z encodes both relation + motion
  - cluster structure in latent space (means + pairwise distances)
"""

import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

GRID_SIZE = 16
T_FRAMES = 6

RELATIONS = ["left_of", "right_of", "above", "below", "inside", "overlapping"]
MOTIONS   = ["fall", "rise", "slide_lr", "slide_rl"]


# ------------------------------
# Basic voxel primitives
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
                if math.sqrt(dx*dx + dy*dy + dz*dz) <= r:
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


def sample_primitive(center, shape_type=None, size_jitter=True):
    if shape_type is None:
        shape_type = np.random.choice(["sphere", "cube"])
    if shape_type == "sphere":
        radius = np.random.uniform(0.18, 0.30) if size_jitter else 0.25
        return make_sphere(radius_frac=radius, center=center)
    else:
        size = np.random.uniform(0.25, 0.4) if size_jitter else 0.35
        return make_cube(size_frac=size, center=center)


# ------------------------------
# Relation placement (static first frame)
# ------------------------------

def place_two_objects_with_relation(rel_id):
    """
    Return:
      vol_A, vol_B, center_A, center_B
    ensuring the *initial* centers satisfy the chosen relation.

    RELATIONS:
      0: left_of       (A left of B)
      1: right_of      (A right of B)
      2: above         (A above B)
      3: below         (A below B)
      4: inside        (A inside B)
      5: overlapping   (A overlapping B)
    """
    rel_name = RELATIONS[rel_id]

    # Margin to avoid clipping
    margin = 4

    # Start with B roughly in the middle
    center_B = np.array([
        np.random.randint(margin, GRID_SIZE - margin),
        np.random.randint(margin, GRID_SIZE - margin),
        np.random.randint(margin, GRID_SIZE - margin),
    ])

    shape_A = np.random.choice(["sphere", "cube"])
    shape_B = np.random.choice(["sphere", "cube"])

    # Basic offsets
    dx = np.random.randint(3, 5)
    dy = np.random.randint(3, 5)
    dz = np.random.randint(0, 3)

    if rel_name == "left_of":
        center_A = center_B - np.array([dx, 0, 0])
    elif rel_name == "right_of":
        center_A = center_B + np.array([dx, 0, 0])
    elif rel_name == "above":
        center_A = center_B + np.array([0, dy, 0])
    elif rel_name == "below":
        center_A = center_B - np.array([0, dy, 0])
    elif rel_name == "inside":
        # Make B the container, A the contained
        # Put B near center, A close to B
        center_B = np.array([
            GRID_SIZE // 2 + np.random.randint(-2, 3),
            GRID_SIZE // 2 + np.random.randint(-2, 3),
            GRID_SIZE // 2 + np.random.randint(-2, 3),
        ])
        center_A = center_B + np.random.randint(-1, 2, size=3)
        shape_B = "cube"
        shape_A = "sphere"
    else:  # overlapping
        # Put A slightly offset so they intersect
        center_A = center_B + np.array([
            np.random.randint(-2, 3),
            np.random.randint(-2, 3),
            np.random.randint(-2, 3),
        ])

    # Clip to safe bounds
    center_A = np.clip(center_A, margin, GRID_SIZE - margin - 1)
    center_B = np.clip(center_B, margin, GRID_SIZE - margin - 1)

    vol_A = sample_primitive(tuple(center_A), shape_type=shape_A)
    vol_B = sample_primitive(tuple(center_B), shape_type=shape_B)

    return vol_A, vol_B, center_A, center_B


# ------------------------------
# Motion evolution over time
# ------------------------------

def apply_motion(center, motion_id, t, step=None):
    """
    Evolve center over time t according to motion type.
    We keep things simple and discrete.

      0: fall      -> -y
      1: rise      -> +y
      2: slide_lr  -> +x
      3: slide_rl  -> -x
    """
    if step is None:
        step = np.random.randint(1, 3)

    dx, dy, dz = 0, 0, 0
    if motion_id == 0:  # fall
        dy = -step * t
    elif motion_id == 1:  # rise
        dy = step * t
    elif motion_id == 2:  # slide_lr
        dx = step * t
    else:  # slide_rl
        dx = -step * t

    new_center = center + np.array([dx, dy, dz])
    margin = 3
    new_center = np.clip(new_center, margin, GRID_SIZE - margin - 1)
    return new_center


# ------------------------------
# Sentence dataset: relation + motion in sequences
# ------------------------------

def generate_sentence_sequence(rel_id, mot_id, T=T_FRAMES):
    """
    Returns:
      seq: (T, X, Y, Z) with two objects:
        - B fixed
        - A moving according to motion
    """
    vol_A0, vol_B0, center_A0, center_B = place_two_objects_with_relation(rel_id)
    shape_A = np.random.choice(["sphere", "cube"])
    shape_B = np.random.choice(["sphere", "cube"])
    # Recompute A0, B0 with explicit shape choice for consistency
    vol_A0 = sample_primitive(tuple(center_A0), shape_type=shape_A)
    vol_B0 = sample_primitive(tuple(center_B),  shape_type=shape_B)

    seq = []
    step = np.random.randint(1, 3)
    center_A = center_A0.copy()
    for t in range(T):
        if t == 0:
            cA = center_A0
        else:
            cA = apply_motion(center_A0, mot_id, t, step=step)
        vol_A = sample_primitive(tuple(cA), shape_type=shape_A)
        vol_B = sample_primitive(tuple(center_B), shape_type=shape_B)

        frame = np.clip(vol_A + vol_B, 0.0, 1.0)
        seq.append(frame)

    return np.stack(seq, axis=0)  # (T, X, Y, Z)


def generate_sentence_dataset(n_per_combo=150):
    """
    Dataset size: len(RELATIONS) * len(MOTIONS) * n_per_combo
    """
    sequences = []
    rel_labels = []
    mot_labels = []

    for r in range(len(RELATIONS)):
        for m in range(len(MOTIONS)):
            for _ in range(n_per_combo):
                seq = generate_sentence_sequence(r, m)
                sequences.append(seq)
                rel_labels.append(r)
                mot_labels.append(m)

    sequences_np = np.stack(sequences, axis=0)  # (N, T, X, Y, Z)
    rel_np = np.array(rel_labels, dtype=np.int64)
    mot_np = np.array(mot_labels, dtype=np.int64)
    return sequences_np, rel_np, mot_np


# ------------------------------
# Model
# ------------------------------

class SentenceEncoder(nn.Module):
    def __init__(self, latent_dim=32, T=T_FRAMES, grid_size=GRID_SIZE):
        super().__init__()
        in_dim = T * grid_size * grid_size * grid_size
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )

    def forward(self, x):
        # x: (B, T, X, Y, Z)
        b = x.shape[0]
        x_flat = x.view(b, -1)
        return self.net(x_flat)


class SentenceDecoder(nn.Module):
    def __init__(self, latent_dim=32, T=T_FRAMES, grid_size=GRID_SIZE):
        super().__init__()
        out_dim = T * grid_size * grid_size * grid_size
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim),
            nn.Sigmoid(),
        )
        self.T = T
        self.grid_size = grid_size

    def forward(self, z):
        # z: (B, latent_dim)
        b = z.shape[0]
        x = self.net(z)
        x = x.view(b, self.T, self.grid_size, self.grid_size, self.grid_size)
        return x


class SentenceModel(nn.Module):
    def __init__(self,
                 latent_dim=32,
                 T=T_FRAMES,
                 grid_size=GRID_SIZE,
                 n_relations=6,
                 n_motions=4):
        super().__init__()
        self.encoder = SentenceEncoder(latent_dim=latent_dim,
                                       T=T,
                                       grid_size=grid_size)
        self.decoder = SentenceDecoder(latent_dim=latent_dim,
                                       T=T,
                                       grid_size=grid_size)
        self.rel_head = nn.Linear(latent_dim, n_relations)
        self.mot_head = nn.Linear(latent_dim, n_motions)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        rel_logits = self.rel_head(z)
        mot_logits = self.mot_head(z)
        return recon, rel_logits, mot_logits, z


# ------------------------------
# Training
# ------------------------------

def train_sentence_model(num_epochs=500,
                         latent_dim=32,
                         lr=1e-3,
                         alpha_rel=0.5,
                         beta_mot=0.5,
                         device="cpu"):
    seq_np, rel_np, mot_np = generate_sentence_dataset(n_per_combo=100)
    seq_t = torch.from_numpy(seq_np).to(device)
    rel_t = torch.from_numpy(rel_np).to(device)
    mot_t = torch.from_numpy(mot_np).to(device)

    model = SentenceModel(latent_dim=latent_dim,
                          T=T_FRAMES,
                          grid_size=GRID_SIZE,
                          n_relations=len(RELATIONS),
                          n_motions=len(MOTIONS)).to(device)

    bce = nn.BCELoss()
    ce = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        opt.zero_grad()

        recon, rel_logits, mot_logits, z = model(seq_t)

        loss_recon = bce(recon, seq_t)
        loss_rel = ce(rel_logits, rel_t)
        loss_mot = ce(mot_logits, mot_t)
        loss = loss_recon + alpha_rel * loss_rel + beta_mot * loss_mot

        loss.backward()
        opt.step()

        if epoch % 50 == 0 or epoch == num_epochs - 1:
            with torch.no_grad():
                pred_rel = rel_logits.argmax(dim=1)
                pred_mot = mot_logits.argmax(dim=1)
                acc_rel = (pred_rel == rel_t).float().mean().item()
                acc_mot = (pred_mot == mot_t).float().mean().item()
            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"Recon: {loss_recon.item():.6f} | "
                f"RelCE: {loss_rel.item():.6f} | "
                f"MotCE: {loss_mot.item():.6f} | "
                f"Total: {loss.item():.6f} | "
                f"Acc_rel: {acc_rel*100:.2f}% | "
                f"Acc_mot: {acc_mot*100:.2f}%"
            )

    model.eval()
    with torch.no_grad():
        recon, rel_logits, mot_logits, z = model(seq_t)
        loss_recon = bce(recon, seq_t).item()
        loss_rel = ce(rel_logits, rel_t).item()
        loss_mot = ce(mot_logits, mot_t).item()
        pred_rel = rel_logits.argmax(dim=1)
        pred_mot = mot_logits.argmax(dim=1)
        acc_rel = (pred_rel == rel_t).float().mean().item()
        acc_mot = (pred_mot == mot_t).float().mean().item()

    print("\nFinal sentence model metrics:")
    print(f"  Recon BCE:  {loss_recon:.6f}")
    print(f"  Rel CE:     {loss_rel:.6f}  (acc {acc_rel*100:.2f}%)")
    print(f"  Motion CE:  {loss_mot:.6f}  (acc {acc_mot*100:.2f}%)")

    # Examine latent clusters by relation
    z_np = z.cpu().numpy()
    rel_np = rel_np
    mot_np = mot_np

    print("\nMean latent by relation:")
    rel_means = []
    for r in range(len(RELATIONS)):
        mask = (rel_np == r)
        mean_z = z_np[mask].mean(axis=0)
        rel_means.append(mean_z)
        print(f"  {RELATIONS[r]}:")
        print("   ", mean_z)
    print("\nPairwise distances between relation means (L2):")
    for i in range(len(RELATIONS)):
        for j in range(i+1, len(RELATIONS)):
            d = np.linalg.norm(rel_means[i] - rel_means[j])
            print(f"  {RELATIONS[i]} <-> {RELATIONS[j]}: {d:.4f}")

    print("\nMean latent by motion:")
    mot_means = []
    for m in range(len(MOTIONS)):
        mask = (mot_np == m)
        mean_z = z_np[mask].mean(axis=0)
        mot_means.append(mean_z)
        print(f"  {MOTIONS[m]}:")
        print("   ", mean_z)
    print("\nPairwise distances between motion means (L2):")
    for i in range(len(MOTIONS)):
        for j in range(i+1, len(MOTIONS)):
            d = np.linalg.norm(mot_means[i] - mot_means[j])
            print(f"  {MOTIONS[i]} <-> {MOTIONS[j]}: {d:.4f}")

    return model, seq_np, rel_np, mot_np, z


def main():
    device = "cpu"
    print(f"Using device: {device}")
    os.makedirs("outputs_sentences", exist_ok=True)

    model, seq_np, rel_np, mot_np, z = train_sentence_model(
        num_epochs=500,
        latent_dim=32,
        lr=1e-3,
        alpha_rel=0.5,
        beta_mot=0.5,
        device=device,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
