#!/usr/bin/env python3
"""
geomlang_relations_extended.py

Phase 4b + Language Bridge: Extended Relational BBIT

We generate 16x16x16 voxel scenes with TWO primitive shapes and a
TAGGED SPATIAL / TOPOLOGICAL RELATION between them:

  0 = left_of      (B left of A)
  1 = right_of
  2 = above
  3 = below
  4 = inside       (B clearly inside A)
  5 = overlapping  (A and B intersect significantly but not nested)

We train:

  - encoder: scene -> latent z (R^16)
  - decoder: z -> scene
  - classifier: z -> 6-way relation logits

Joint loss:
  loss_rel = BCE(reconstruction) + beta * CE(relation)

Then we add a toy LANGUAGE BRIDGE:

  - A learned word embedding E_word: 6 words x D_word (D_word=8)
  - A mapping M: z -> e_pred in R^D_word

We train M and E_word so that:
  e_pred ~ E_word[label]   (MSE loss)

This aligns the latent geometry with a "word space" for the relations.
We print relation vectors both in latent space and word space.
"""

import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

GRID_SIZE = 16
RELATIONS = ["left_of", "right_of", "above", "below", "inside", "overlapping"]


# ------------------------------
# Primitive generators
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


def make_cube(size_frac=0.4, center=None):
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


def make_cylinder(axis="z", radius_frac=0.2, center=None, length_frac=0.9):
    grid = make_empty()
    if center is None:
        center = ((GRID_SIZE - 1) / 2.0,) * 3
    cx, cy, cz = center
    r = GRID_SIZE * radius_frac
    L = int(GRID_SIZE * length_frac)

    if axis == "z":
        z0 = max(0, int(cz - L / 2))
        z1 = min(GRID_SIZE, int(cz + L / 2))
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                dx = x - cx
                dy = y - cy
                if math.sqrt(dx*dx + dy*dy) <= r:
                    grid[x, y, z0:z1] = 1.0
    elif axis == "x":
        x0 = max(0, int(cx - L / 2))
        x1 = min(GRID_SIZE, int(cx + L / 2))
        for y in range(GRID_SIZE):
            for z in range(GRID_SIZE):
                dy = y - cy
                dz = z - cz
                if math.sqrt(dy*dy + dz*dz) <= r:
                    grid[x0:x1, y, z] = 1.0
    else:  # "y"
        y0 = max(0, int(cy - L / 2))
        y1 = min(GRID_SIZE, int(cy + L / 2))
        for x in range(GRID_SIZE):
            for z in range(GRID_SIZE):
                dx = x - cx
                dz = z - cz
                if math.sqrt(dx*dx + dz*dz) <= r:
                    grid[x, y0:y1, z] = 1.0
    return grid


def make_L_shape(thickness=2, length_frac=0.5, center=None):
    grid = make_empty()
    if center is None:
        center = ((GRID_SIZE - 1) / 2.0,) * 3
    cx, cy, cz = map(int, center)
    L = int(GRID_SIZE * length_frac)
    half = L // 2

    # Horizontal bar (x-axis)
    x0 = max(0, cx - half)
    x1 = min(GRID_SIZE, cx + half)
    y0 = max(0, cy - thickness // 2)
    y1 = min(GRID_SIZE, cy + thickness // 2)
    z0 = max(0, cz - thickness // 2)
    z1 = min(GRID_SIZE, cz + thickness // 2)
    grid[x0:x1, y0:y1, z0:z1] = 1.0

    # Vertical bar (y-axis)
    x0 = max(0, cx - thickness // 2)
    x1 = min(GRID_SIZE, cx + thickness // 2)
    y0 = max(0, cy - half)
    y1 = min(GRID_SIZE, cy + half)
    grid[x0:x1, y0:y1, z0:z1] = 1.0

    return grid


def sample_primitive_at(center,
                        shape_type=None,
                        size_scale=None,
                        radius_scale=None):
    """
    Random primitive but forced to use a given center.
    Optional shape_type / size_scale / radius_scale to control scale.
    """
    if shape_type is None:
        shape_type = np.random.choice(["sphere", "cube", "cyl", "L"])

    if shape_type == "sphere":
        if radius_scale is None:
            radius_frac = np.random.uniform(0.18, 0.35)
        else:
            radius_frac = radius_scale
        grid = make_sphere(radius_frac=radius_frac, center=center)

    elif shape_type == "cube":
        if size_scale is None:
            size_frac = np.random.uniform(0.25, 0.55)
        else:
            size_frac = size_scale
        grid = make_cube(size_frac=size_frac, center=center)

    elif shape_type == "cyl":
        axis = np.random.choice(["x", "y", "z"])
        if radius_scale is None:
            radius_frac = np.random.uniform(0.12, 0.25)
        else:
            radius_frac = radius_scale
        length_frac = np.random.uniform(0.5, 1.0)
        grid = make_cylinder(
            axis=axis,
            radius_frac=radius_frac,
            center=center,
            length_frac=length_frac,
        )

    else:  # "L"
        thickness = np.random.randint(1, 4)
        if size_scale is None:
            length_frac = np.random.uniform(0.4, 0.8)
        else:
            length_frac = size_scale
        grid = make_L_shape(
            thickness=thickness,
            length_frac=length_frac,
            center=center,
        )

    return grid


# ------------------------------
# Dataset with extended relations
# ------------------------------

def generate_relational_scenes_extended(n_per_relation=220):
    """
    Generate scenes with two primitives and one of six relations:

      0 = left_of (B left of A)
      1 = right_of
      2 = above
      3 = below
      4 = inside (B inside A)
      5 = overlapping (A and B intersect significantly)
    """
    scenes = []
    labels = []

    base_center = np.array([GRID_SIZE // 2,
                            GRID_SIZE // 2,
                            GRID_SIZE // 2])

    # offsets in (x,y,z) for the directional relations
    offset_config = {
        0: np.array([-4, 0, 0]),   # left_of: B left of A
        1: np.array([+4, 0, 0]),   # right_of
        2: np.array([0, +4, 0]),   # above
        3: np.array([0, -4, 0]),   # below
    }

    # 0-3: directional relations
    for rel_id in range(4):
        offset = offset_config[rel_id]
        for _ in range(n_per_relation):
            jitter_A = np.random.randint(-1, 2, size=3)
            jitter_B = np.random.randint(-1, 2, size=3)
            center_A = base_center + jitter_A
            center_B = base_center + offset + jitter_B

            center_A = np.clip(center_A, 3, GRID_SIZE - 4)
            center_B = np.clip(center_B, 3, GRID_SIZE - 4)

            grid = make_empty()
            prim_A = sample_primitive_at(tuple(center_A))
            prim_B = sample_primitive_at(tuple(center_B))
            grid = np.maximum(grid, prim_A)
            grid = np.maximum(grid, prim_B)

            scenes.append(grid)
            labels.append(rel_id)

    # 4: inside (B inside A)
    for _ in range(n_per_relation):
        center_A = base_center + np.random.randint(-1, 2, size=3)
        center_A = np.clip(center_A, 4, GRID_SIZE - 5)
        center_B = center_A + np.random.randint(-1, 2, size=3)
        center_B = np.clip(center_B, 4, GRID_SIZE - 5)

        grid = make_empty()

        # A: big container
        big_shape = np.random.choice(["sphere", "cube"])
        if big_shape == "sphere":
            prim_A = sample_primitive_at(tuple(center_A),
                                         shape_type="sphere",
                                         radius_scale=np.random.uniform(0.4, 0.55))
        else:
            prim_A = sample_primitive_at(tuple(center_A),
                                         shape_type="cube",
                                         size_scale=np.random.uniform(0.5, 0.7))

        # B: smaller thing inside
        small_shape = np.random.choice(["sphere", "cube", "cyl", "L"])
        if small_shape == "sphere":
            prim_B = sample_primitive_at(tuple(center_B),
                                         shape_type="sphere",
                                         radius_scale=np.random.uniform(0.15, 0.25))
        elif small_shape == "cube":
            prim_B = sample_primitive_at(tuple(center_B),
                                         shape_type="cube",
                                         size_scale=np.random.uniform(0.2, 0.35))
        else:
            prim_B = sample_primitive_at(tuple(center_B),
                                         shape_type=small_shape,
                                         size_scale=np.random.uniform(0.3, 0.5))

        grid = np.maximum(grid, prim_A)
        grid = np.maximum(grid, prim_B)

        scenes.append(grid)
        labels.append(4)

    # 5: overlapping (centers close, similar scales)
    for _ in range(n_per_relation):
        center_A = base_center + np.random.randint(-1, 2, size=3)
        center_A = np.clip(center_A, 4, GRID_SIZE - 5)

        # small offset of 1-3 voxels on x/y/z to ensure intersection
        offset = np.random.randint(-3, 4, size=3)
        while np.all(offset == 0):
            offset = np.random.randint(-3, 4, size=3)
        center_B = center_A + offset
        center_B = np.clip(center_B, 4, GRID_SIZE - 5)

        grid = make_empty()

        shape_A = np.random.choice(["sphere", "cube", "cyl", "L"])
        shape_B = np.random.choice(["sphere", "cube", "cyl", "L"])
        scale_base = np.random.uniform(0.3, 0.5)

        if shape_A == "sphere":
            prim_A = sample_primitive_at(tuple(center_A),
                                         shape_type="sphere",
                                         radius_scale=scale_base)
        elif shape_A == "cube":
            prim_A = sample_primitive_at(tuple(center_A),
                                         shape_type="cube",
                                         size_scale=scale_base)
        else:
            prim_A = sample_primitive_at(tuple(center_A),
                                         shape_type=shape_A,
                                         size_scale=scale_base)

        if shape_B == "sphere":
            prim_B = sample_primitive_at(tuple(center_B),
                                         shape_type="sphere",
                                         radius_scale=scale_base)
        elif shape_B == "cube":
            prim_B = sample_primitive_at(tuple(center_B),
                                         shape_type="cube",
                                         size_scale=scale_base)
        else:
            prim_B = sample_primitive_at(tuple(center_B),
                                         shape_type=shape_B,
                                         size_scale=scale_base)

        grid = np.maximum(grid, prim_A)
        grid = np.maximum(grid, prim_B)

        scenes.append(grid)
        labels.append(5)

    scenes_np = np.stack(scenes, axis=0)
    labels_np = np.array(labels, dtype=np.int64)
    return scenes_np, labels_np


# ------------------------------
# Models
# ------------------------------

class Encoder(nn.Module):
    def __init__(self, latent_dim=16, grid_size=GRID_SIZE):
        super().__init__()
        in_dim = grid_size * grid_size * grid_size
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x):
        b = x.shape[0]
        x_flat = x.view(b, -1)
        z = self.net(x_flat)
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dim=16, grid_size=GRID_SIZE):
        super().__init__()
        out_dim = grid_size * grid_size * grid_size
        self.grid_size = grid_size
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, z):
        b = z.shape[0]
        x = self.net(z)
        x = x.view(b, self.grid_size, self.grid_size, self.grid_size)
        return x


class SceneRelModel(nn.Module):
    def __init__(self, latent_dim=16, grid_size=GRID_SIZE, n_relations=6):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim, grid_size=grid_size)
        self.decoder = Decoder(latent_dim=latent_dim, grid_size=grid_size)
        self.classifier = nn.Linear(latent_dim, n_relations)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        logits = self.classifier(z)
        return recon, logits, z


# ------------------------------
# Relation training
# ------------------------------

def train_relational_model_extended(num_epochs=900,
                                    latent_dim=16,
                                    lr=1e-3,
                                    beta=0.4,
                                    device="cpu"):
    scenes_np, labels_np = generate_relational_scenes_extended(
        n_per_relation=220
    )
    scenes_t = torch.from_numpy(scenes_np).to(device)
    labels_t = torch.from_numpy(labels_np).to(device)

    model = SceneRelModel(latent_dim=latent_dim,
                          grid_size=GRID_SIZE,
                          n_relations=len(RELATIONS)).to(device)

    bce = nn.BCELoss()
    ce = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        opt.zero_grad()
        recon, logits, z = model(scenes_t)
        loss_recon = bce(recon, scenes_t)
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
                f"RelCE: {loss_cls.item():.6f} | "
                f"Total: {loss.item():.6f} | "
                f"Acc: {acc*100:.2f}%"
            )

    model.eval()
    with torch.no_grad():
        recon, logits, z = model(scenes_t)
        loss_recon = bce(recon, scenes_t).item()
        loss_cls = ce(logits, labels_t).item()
        preds = logits.argmax(dim=1)
        acc = (preds == labels_t).float().mean().item()

    print("\nFinal relation metrics:")
    print(f"  Recon BCE: {loss_recon:.6f}")
    print(f"  Rel CE:    {loss_cls:.6f}")
    print(f"  Accuracy:  {acc*100:.2f}%")

    z_np = z.cpu().numpy()
    labels_np = labels_np  # alias
    means = []
    for rel_id in range(len(RELATIONS)):
        mask = (labels_np == rel_id)
        mean_z = z_np[mask].mean(axis=0)
        means.append(mean_z)
        print(f"Mean latent for relation {rel_id} ({RELATIONS[rel_id]}):")
        print("  ", mean_z)

    print("\nPairwise distances between relation means (L2):")
    for i in range(len(RELATIONS)):
        for j in range(i + 1, len(RELATIONS)):
            d = np.linalg.norm(means[i] - means[j])
            print(f"  {RELATIONS[i]} <-> {RELATIONS[j]}: {d:.4f}")

    return model, scenes_np, labels_np, means, z.detach(), labels_t


# ------------------------------
# Language bridge: z -> word embeddings
# ------------------------------

class ZToWordMapper(nn.Module):
    def __init__(self, latent_dim=16, word_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, word_dim),
        )

    def forward(self, z):
        return self.net(z)


def train_language_bridge(z_all,
                          labels_all,
                          word_dim=8,
                          num_epochs=400,
                          lr=1e-3,
                          device="cpu"):
    """
    z_all: (N, latent_dim) tensor
    labels_all: (N,) tensor of relation ids

    We learn:
      - E_word: nn.Embedding(6, word_dim)
      - mapper: z -> e_pred

    Loss = MSE(e_pred, E_word[label])
    """
    latent_dim = z_all.shape[1]
    z_all = z_all.to(device)
    labels_all = labels_all.to(device)

    E_word = nn.Embedding(len(RELATIONS), word_dim).to(device)
    mapper = ZToWordMapper(latent_dim=latent_dim,
                           word_dim=word_dim).to(device)

    params = list(E_word.parameters()) + list(mapper.parameters())
    opt = optim.Adam(params, lr=lr)
    mse = nn.MSELoss()

    for epoch in range(num_epochs):
        mapper.train()
        E_word.train()
        opt.zero_grad()

        e_true = E_word(labels_all)       # (N, word_dim)
        e_pred = mapper(z_all)            # (N, word_dim)
        loss = mse(e_pred, e_true)
        loss.backward()
        opt.step()

        if epoch % 100 == 0 or epoch == num_epochs - 1:
            print(f"[LangBridge] Epoch {epoch}/{num_epochs} | MSE: {loss.item():.6f}")

    # Inspect word embeddings
    E_word.eval()
    mapper.eval()

    with torch.no_grad():
        word_embs = E_word(torch.arange(len(RELATIONS), device=device))
        word_embs_np = word_embs.cpu().numpy()

    print("\nWord embedding vectors (toy language space):")
    for i, name in enumerate(RELATIONS):
        print(f"  {name}: {word_embs_np[i]}")

    print("\nPairwise distances in word-embedding space (L2):")
    for i in range(len(RELATIONS)):
        for j in range(i + 1, len(RELATIONS)):
            d = np.linalg.norm(word_embs_np[i] - word_embs_np[j])
            print(f"  {RELATIONS[i]} <-> {RELATIONS[j]}: {d:.4f}")

    # Also compute relation vectors in latent vs word space
    print("\nRelation vectors (latent space vs word space):")
    with torch.no_grad():
        # mean z per relation
        z_np = z_all.cpu().numpy()
        labels_np = labels_all.cpu().numpy()
        z_means = []
        for rel_id in range(len(RELATIONS)):
            mask = (labels_np == rel_id)
            z_means.append(z_np[mask].mean(axis=0))
        z_means = np.stack(z_means, axis=0)

    def rel_vec_z(a, b):
        return z_means[b] - z_means[a]

    def rel_vec_w(a, b):
        return word_embs_np[b] - word_embs_np[a]

    # Example: left_of vs right_of; above vs below
    idx_left = RELATIONS.index("left_of")
    idx_right = RELATIONS.index("right_of")
    idx_above = RELATIONS.index("above")
    idx_below = RELATIONS.index("below")

    v_lr_z = rel_vec_z(idx_left, idx_right)
    v_lr_w = rel_vec_w(idx_left, idx_right)
    v_ab_z = rel_vec_z(idx_below, idx_above)  # below -> above
    v_ab_w = rel_vec_w(idx_below, idx_above)

    def cos_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

    print("\n[left_of -> right_of] relation vector:")
    print("  latent space norm:", np.linalg.norm(v_lr_z))
    print("  word space norm:  ", np.linalg.norm(v_lr_w))
    print("  (no direct comparison is enforced, but they are both 'directions')")

    print("\n[below -> above] relation vector:")
    print("  latent space norm:", np.linalg.norm(v_ab_z))
    print("  word space norm:  ", np.linalg.norm(v_ab_w))

    print("\nCosine similarity between relation directions in word space:")
    print("  cos(v_lr_w, v_ab_w):", cos_sim(v_lr_w, v_ab_w))

    return E_word, mapper


# ------------------------------
# Main
# ------------------------------

def main():
    device = "cpu"
    print(f"Using device: {device}")
    os.makedirs("outputs_rel_ext", exist_ok=True)

    # 1) Train relation + geometry model
    model, scenes_np, labels_np, means, z_all, labels_t = \
        train_relational_model_extended(
            num_epochs=900,
            latent_dim=16,
            lr=1e-3,
            beta=0.4,
            device=device,
        )

    # 2) Train language bridge on the learned latents
    print("\n=== Training language bridge (geometry -> word space) ===")
    E_word, mapper = train_language_bridge(
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
