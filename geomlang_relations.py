#!/usr/bin/env python3
"""
geomlang_relations.py

Phase 4: Relational BBIT + relation-vector walks.

We generate 16x16x16 voxel scenes with TWO primitive shapes and a
TAGGED SPATIAL RELATION between them:

  0 = left_of   (shape B left of shape A)
  1 = right_of
  2 = above
  3 = below

We train:

  - encoder: scene -> latent z (R^16)
  - decoder: z -> scene
  - classifier: z -> 4-way relation logits

Joint loss:
  loss = BCE(reconstruction) + beta * CE(relation)

After training, we:
  - print reconstruction loss, relation accuracy
  - compute mean latent per relation and distances
  - choose example scenes and move them along relation vectors:
      v_lr = mean(right_of) - mean(left_of)
      v_ab = mean(above)    - mean(below)
    and save 3D visualizations + PLYs for:
      original, z+v_lr, z+v_ab
"""

import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

GRID_SIZE = 16
RELATIONS = ["left_of", "right_of", "above", "below"]


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


def sample_primitive_at(center):
    """Random primitive but forced to use a given center."""
    shape_type = np.random.choice(["sphere", "cube", "cyl", "L"])
    if shape_type == "sphere":
        radius_frac = np.random.uniform(0.18, 0.35)
        grid = make_sphere(radius_frac=radius_frac, center=center)
    elif shape_type == "cube":
        size_frac = np.random.uniform(0.25, 0.55)
        grid = make_cube(size_frac=size_frac, center=center)
    elif shape_type == "cyl":
        axis = np.random.choice(["x", "y", "z"])
        radius_frac = np.random.uniform(0.12, 0.25)
        length_frac = np.random.uniform(0.5, 1.0)
        grid = make_cylinder(
            axis=axis,
            radius_frac=radius_frac,
            center=center,
            length_frac=length_frac,
        )
    else:
        thickness = np.random.randint(1, 4)
        length_frac = np.random.uniform(0.4, 0.8)
        grid = make_L_shape(
            thickness=thickness,
            length_frac=length_frac,
            center=center,
        )
    return grid


# ------------------------------
# Dataset with explicit relations
# ------------------------------

def generate_relational_scenes(n_per_relation=250):
    """
    For each relation r in {left_of, right_of, above, below},
    generate n_per_relation scenes containing two primitives A and B
    such that B has that relation to A (in x/y plane).
    """
    scenes = []
    labels = []

    base_center = np.array([GRID_SIZE // 2, GRID_SIZE // 2, GRID_SIZE // 2])

    offset_config = {
        0: np.array([-4, 0, 0]),   # left_of: B left of A
        1: np.array([+4, 0, 0]),   # right_of
        2: np.array([0, +4, 0]),   # above
        3: np.array([0, -4, 0]),   # below
    }

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

    scenes_np = np.stack(scenes, axis=0)
    labels_np = np.array(labels, dtype=np.int64)
    return scenes_np, labels_np


# ------------------------------
# Model: autoencoder + classifier
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
    def __init__(self, latent_dim=16, grid_size=GRID_SIZE, n_relations=4):
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
# Visualization helpers
# ------------------------------

def save_voxels_3d(voxels, out_path_img, title="voxel_shape", threshold=0.5):
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa

        filled = voxels >= threshold
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.voxels(filled, edgecolor="k")
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.tight_layout()
        plt.savefig(out_path_img, dpi=150)
        plt.close()
        print(f"Saved 3D voxel visualization to {out_path_img}")
    except ImportError:
        print("matplotlib not installed; skipping 3D visualization.")


def save_voxels_as_ply(voxels, out_path_ply, threshold=0.5):
    xs, ys, zs = np.where(voxels >= threshold)
    n_points = len(xs)
    with open(out_path_ply, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for x, y, z in zip(xs, ys, zs):
            f.write(f"{x} {y} {z}\n")
    print(f"Saved PLY to {out_path_ply}")


# ------------------------------
# Training / evaluation
# ------------------------------

def train_relational_model(num_epochs=800, latent_dim=16, lr=1e-3,
                           beta=0.3, device="cpu"):
    scenes_np, labels_np = generate_relational_scenes(n_per_relation=250)
    scenes_t = torch.from_numpy(scenes_np).to(device)
    labels_t = torch.from_numpy(labels_np).to(device)

    model = SceneRelModel(latent_dim=latent_dim, grid_size=GRID_SIZE).to(device)
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

    # final eval
    model.eval()
    with torch.no_grad():
        recon, logits, z = model(scenes_t)
        loss_recon = bce(recon, scenes_t).item()
        loss_cls = ce(logits, labels_t).item()
        preds = logits.argmax(dim=1)
        acc = (preds == labels_t).float().mean().item()

    print("\nFinal metrics:")
    print(f"  Recon BCE: {loss_recon:.6f}")
    print(f"  Rel CE:    {loss_cls:.6f}")
    print(f"  Accuracy:  {acc*100:.2f}%")

    # mean latent per relation
    z_np = z.cpu().numpy()
    labels_np = labels_np  # local alias
    means = []
    for rel_id in range(4):
        mask = (labels_np == rel_id)
        mean_z = z_np[mask].mean(axis=0)
        means.append(mean_z)
        print(f"Mean latent for relation {rel_id} ({RELATIONS[rel_id]}):")
        print("  ", mean_z)

    print("\nPairwise distances between relation means (L2):")
    for i in range(4):
        for j in range(i + 1, 4):
            d = np.linalg.norm(means[i] - means[j])
            print(f"  {RELATIONS[i]} <-> {RELATIONS[j]}: {d:.4f}")

    return model, scenes_np, labels_np, means


# ------------------------------
# Relation-vector walks
# ------------------------------

def relation_vector_walks(model, scenes_np, labels_np, means,
                          out_dir="outputs_rel", device="cpu"):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    # relation vectors
    mean_left, mean_right, mean_above, mean_below = means
    v_lr = mean_right - mean_left
    v_ab = mean_above - mean_below

    print("\nRelation vectors:")
    print("  v_lr (right_of - left_of):", v_lr)
    print("  v_ab (above - below):    ", v_ab)

    with torch.no_grad():
        scenes_t = torch.from_numpy(scenes_np).to(device)
        _, logits, z_all = model(scenes_t)
        preds_all = logits.argmax(dim=1).cpu().numpy()
        z_all_np = z_all.cpu().numpy()

    # pick a few example indices to visualize
    examples = []
    for rel_id in range(4):
        idxs = np.where(labels_np == rel_id)[0]
        if len(idxs) > 0:
            examples.append(idxs[0])

    alpha = 0.4  # how far to move along relation vectors

    for idx in examples:
        label = labels_np[idx]
        rel_name = RELATIONS[label]
        print(f"\nExample index {idx}, relation {rel_name}")

        scene = scenes_np[idx]
        z_orig = z_all_np[idx:idx+1]

        # choose which vectors to apply: always try both
        z_lr = z_orig + alpha * v_lr
        z_ab = z_orig + alpha * v_ab

        z_t = torch.from_numpy(z_orig).float().to(device)
        z_lr_t = torch.from_numpy(z_lr).float().to(device)
        z_ab_t = torch.from_numpy(z_ab).float().to(device)

        with torch.no_grad():
            scene_orig_rec = model.decoder(z_t)[0].cpu().numpy()
            logits_orig = model.classifier(z_t)[0]
            pred_orig = torch.argmax(logits_orig).item()

            scene_lr = model.decoder(z_lr_t)[0].cpu().numpy()
            logits_lr = model.classifier(z_lr_t)[0]
            pred_lr = torch.argmax(logits_lr).item()

            scene_ab = model.decoder(z_ab_t)[0].cpu().numpy()
            logits_ab = model.classifier(z_ab_t)[0]
            pred_ab = torch.argmax(logits_ab).item()

        print(f"  Pred(orig): {RELATIONS[pred_orig]}")
        print(f"  Pred(z+v_lr): {RELATIONS[pred_lr]}")
        print(f"  Pred(z+v_ab): {RELATIONS[pred_ab]}")

        base_name = f"idx{idx}_{rel_name}"
        # save 3D visuals + PLYs
        save_voxels_3d(scene_orig_rec, os.path.join(out_dir, base_name + "_orig.png"),
                       title=f"orig ({RELATIONS[pred_orig]})")
        save_voxels_as_ply(scene_orig_rec, os.path.join(out_dir, base_name + "_orig.ply"))

        save_voxels_3d(scene_lr, os.path.join(out_dir, base_name + "_plus_vlr.png"),
                       title=f"+v_lr ({RELATIONS[pred_lr]})")
        save_voxels_as_ply(scene_lr, os.path.join(out_dir, base_name + "_plus_vlr.ply"))

        save_voxels_3d(scene_ab, os.path.join(out_dir, base_name + "_plus_vab.png"),
                       title=f"+v_ab ({RELATIONS[pred_ab]})")
        save_voxels_as_ply(scene_ab, os.path.join(out_dir, base_name + "_plus_vab.ply"))


def main():
    device = "cpu"
    print(f"Using device: {device}")
    os.makedirs("outputs_rel", exist_ok=True)

    model, scenes_np, labels_np, means = train_relational_model(
        num_epochs=800,
        latent_dim=16,
        lr=1e-3,
        beta=0.3,
        device=device,
    )

    relation_vector_walks(
        model=model,
        scenes_np=scenes_np,
        labels_np=labels_np,
        means=means,
        out_dir="outputs_rel",
        device=device,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
