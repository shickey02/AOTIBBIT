#!/usr/bin/env python3
"""
geomlang_sentences_viz.py

Builds on geomlang_sentences:
  - Trains the sentence model on two-object sequences (relation + motion).
  - Visualizes:
      * a single (relation, motion) scene as T frames of 2D projections
      * an interpolation in latent space between two different "sentences"
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

GRID_SIZE = 16
T_FRAMES = 6

RELATIONS = ["left_of", "right_of", "above", "below", "inside", "overlapping"]
MOTIONS   = ["fall", "rise", "slide_lr", "slide_rl"]


# ---------- primitives (same as before) ----------

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


# ---------- relation placement ----------

def place_two_objects_with_relation(rel_id):
    rel_name = RELATIONS[rel_id]
    margin = 4

    center_B = np.array([
        np.random.randint(margin, GRID_SIZE - margin),
        np.random.randint(margin, GRID_SIZE - margin),
        np.random.randint(margin, GRID_SIZE - margin),
    ])

    shape_A = np.random.choice(["sphere", "cube"])
    shape_B = np.random.choice(["sphere", "cube"])

    dx = np.random.randint(3, 5)
    dy = np.random.randint(3, 5)

    if rel_name == "left_of":
        center_A = center_B - np.array([dx, 0, 0])
    elif rel_name == "right_of":
        center_A = center_B + np.array([dx, 0, 0])
    elif rel_name == "above":
        center_A = center_B + np.array([0, dy, 0])
    elif rel_name == "below":
        center_A = center_B - np.array([0, dy, 0])
    elif rel_name == "inside":
        center_B = np.array([
            GRID_SIZE // 2 + np.random.randint(-2, 3),
            GRID_SIZE // 2 + np.random.randint(-2, 3),
            GRID_SIZE // 2 + np.random.randint(-2, 3),
        ])
        center_A = center_B + np.random.randint(-1, 2, size=3)
        shape_B = "cube"
        shape_A = "sphere"
    else:  # overlapping
        center_A = center_B + np.array([
            np.random.randint(-2, 3),
            np.random.randint(-2, 3),
            np.random.randint(-2, 3),
        ])

    center_A = np.clip(center_A, margin, GRID_SIZE - margin - 1)
    center_B = np.clip(center_B, margin, GRID_SIZE - margin - 1)

    vol_A = sample_primitive(tuple(center_A), shape_type=shape_A)
    vol_B = sample_primitive(tuple(center_B), shape_type=shape_B)
    return vol_A, vol_B, center_A, center_B, shape_A, shape_B


# ---------- motion ----------

def apply_motion(center, motion_id, t, step=None):
    if step is None:
        step = np.random.randint(1, 3)
    dx = dy = dz = 0
    if motion_id == 0:   # fall
        dy = -step * t
    elif motion_id == 1: # rise
        dy = step * t
    elif motion_id == 2: # slide_lr
        dx = step * t
    else:                # slide_rl
        dx = -step * t

    new_center = center + np.array([dx, dy, dz])
    margin = 3
    new_center = np.clip(new_center, margin, GRID_SIZE - margin - 1)
    return new_center


def generate_sentence_sequence(rel_id, mot_id, T=T_FRAMES):
    vol_A0, vol_B0, center_A0, center_B, shape_A, shape_B = place_two_objects_with_relation(rel_id)
    seq = []
    step = np.random.randint(1, 3)

    for t in range(T):
        if t == 0:
            cA = center_A0
        else:
            cA = apply_motion(center_A0, mot_id, t, step=step)
        vol_A = sample_primitive(tuple(cA), shape_type=shape_A)
        vol_B = sample_primitive(tuple(center_B), shape_type=shape_B)
        frame = np.clip(vol_A + vol_B, 0.0, 1.0)
        seq.append(frame)
    return np.stack(seq, axis=0)


def generate_sentence_dataset(n_per_combo=50):
    sequences, rel_labels, mot_labels = [], [], []
    for r in range(len(RELATIONS)):
        for m in range(len(MOTIONS)):
            for _ in range(n_per_combo):
                sequences.append(generate_sentence_sequence(r, m))
                rel_labels.append(r)
                mot_labels.append(m)
    seq_np = np.stack(sequences, axis=0)
    return seq_np, np.array(rel_labels, dtype=np.int64), np.array(mot_labels, dtype=np.int64)


# ---------- model ----------

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
        b = x.shape[0]
        x_flat = x.view(b, -1)
        return self.net(x_flat)


class SentenceDecoder(nn.Module):
    def __init__(self, latent_dim=32, T=T_FRAMES, grid_size=GRID_SIZE):
        super().__init__()
        out_dim = T * grid_size * grid_size * grid_size
        self.T = T
        self.grid_size = grid_size
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, z):
        b = z.shape[0]
        x = self.net(z)
        return x.view(b, self.T, self.grid_size, self.grid_size, self.grid_size)


class SentenceModel(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = SentenceEncoder(latent_dim=latent_dim)
        self.decoder = SentenceDecoder(latent_dim=latent_dim)
        self.rel_head = nn.Linear(latent_dim, len(RELATIONS))
        self.mot_head = nn.Linear(latent_dim, len(MOTIONS))

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        rel_logits = self.rel_head(z)
        mot_logits = self.mot_head(z)
        return recon, rel_logits, mot_logits, z


# ---------- helpers: training & visualization ----------

def train_sentence_model(device="cpu", epochs=300, latent_dim=32):
    seq_np, rel_np, mot_np = generate_sentence_dataset(n_per_combo=40)
    seq_t = torch.from_numpy(seq_np).to(device)
    rel_t = torch.from_numpy(rel_np).to(device)
    mot_t = torch.from_numpy(mot_np).to(device)

    model = SentenceModel(latent_dim=latent_dim).to(device)
    bce = nn.BCELoss()
    ce = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        recon, rel_logits, mot_logits, z = model(seq_t)

        loss_recon = bce(recon, seq_t)
        loss_rel = ce(rel_logits, rel_t)
        loss_mot = ce(mot_logits, mot_t)
        loss = loss_recon + 0.5 * loss_rel + 0.5 * loss_mot

        loss.backward()
        opt.step()

        if epoch % 50 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                pred_rel = rel_logits.argmax(dim=1)
                pred_mot = mot_logits.argmax(dim=1)
                acc_rel = (pred_rel == rel_t).float().mean().item()
                acc_mot = (pred_mot == mot_t).float().mean().item()
            print(
                f"Epoch {epoch}/{epochs} | "
                f"Recon: {loss_recon.item():.6f} | "
                f"RelCE: {loss_rel.item():.6f} | "
                f"MotCE: {loss_mot.item():.6f} | "
                f"Acc_rel: {acc_rel*100:.2f}% | Acc_mot: {acc_mot*100:.2f}%"
            )

    return model, seq_np, rel_np, mot_np


def seq_to_projection_images(seq, out_dir, prefix):
    """
    seq: (T, X, Y, Z)
    Save one PNG per timestep, max-projected along Z.
    """
    os.makedirs(out_dir, exist_ok=True)
    T, X, Y, Z = seq.shape
    for t in range(T):
        frame = seq[t]        # (X, Y, Z)
        proj = frame.max(axis=2)  # (X, Y)
        proj = (proj * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(proj)
        img = img.resize((128, 128), resample=Image.NEAREST)
        img.save(os.path.join(out_dir, f"{prefix}_t{t:02d}.png"))


def main():
    device = "cpu"
    print(f"Using device: {device}")
    os.makedirs("outputs_sentences_viz", exist_ok=True)

    model, seq_np, rel_np, mot_np = train_sentence_model(device=device, epochs=300, latent_dim=32)
    model.eval()

    # --- pick two specific sentences to inspect and interpolate between ---
    # Example: A = (left_of, fall), B = (inside, rise)
    target_A = (RELATIONS.index("left_of"), MOTIONS.index("fall"))
    target_B = (RELATIONS.index("inside"), MOTIONS.index("rise"))

    def pick_example(rel_id, mot_id):
        mask = (rel_np == rel_id) & (mot_np == mot_id)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            raise RuntimeError("No example for that combo.")
        return idxs[0]

    idx_A = pick_example(*target_A)
    idx_B = pick_example(*target_B)

    seq_A = seq_np[idx_A:idx_A+1]  # (1, T, X, Y, Z)
    seq_B = seq_np[idx_B:idx_B+1]

    with torch.no_grad():
        z_A = model.encoder(torch.from_numpy(seq_A).to(device)).cpu().numpy()[0]
        z_B = model.encoder(torch.from_numpy(seq_B).to(device)).cpu().numpy()[0]

    print(f"Sentence A: {RELATIONS[target_A[0]]} + {MOTIONS[target_A[1]]}")
    print(f"Sentence B: {RELATIONS[target_B[0]]} + {MOTIONS[target_B[1]]}")

    # Save original sequences
    seq_to_projection_images(seq_A[0], "outputs_sentences_viz", "A_orig")
    seq_to_projection_images(seq_B[0], "outputs_sentences_viz", "B_orig")

    # --- interpolate in latent space ---
    n_steps = 7
    for i, t in enumerate(np.linspace(0.0, 1.0, n_steps)):
        z_interp = (1 - t) * z_A + t * z_B
        z_interp_t = torch.from_numpy(z_interp[None, :]).float().to(device)
        with torch.no_grad():
            recon_seq = model.decoder(z_interp_t).cpu().numpy()[0]
        seq_to_projection_images(
            recon_seq,
            "outputs_sentences_viz",
            f"interp_{i:02d}_t{t:.2f}".replace(".", "_")
        )

    print("Saved projections to outputs_sentences_viz/")

if __name__ == "__main__":
    main()
