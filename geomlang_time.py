#!/usr/bin/env python3
"""
geomlang_time.py

Temporal BBIT: simple motion sequences.

We generate sequences of T=5 frames, each a 16x16x16 voxel grid, with:

  0 = fall      (sphere falls down toward a floor cube)
  1 = rise      (sphere rises away from the floor cube)
  2 = slide_lr  (sphere slides left->right past a central cube)
  3 = slide_rl  (sphere slides right->left past the cube)

Model:
  - Encoder: sequence (flattened) -> latent z_time (R^24)
  - Decoder: z_time -> reconstructed sequence
  - Classifier: z_time -> 4-way motion logits

Loss:
  loss = BCE(reconstruction) + beta * CE(motion_class)

We report:
  - reconstruction loss
  - classification accuracy
  - mean latent per motion class & distances

We also save one example sequence as 3D PNGs + PLY per frame.
"""

import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

GRID_SIZE = 16
T_STEPS = 5
MOTIONS = ["fall", "rise", "slide_lr", "slide_rl"]


# ------------------------------
# Primitive generators
# ------------------------------

def make_empty():
    return np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.float32)


def make_sphere(radius_frac=0.2, center=None):
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


# ------------------------------
# Motion sequence generator
# ------------------------------

def generate_motion_sequences(n_per_motion=250):
    """
    Generate sequences of T_STEPS frames with a static reference cube
    and a moving sphere, for four motion patterns:

      0 = fall     (sphere moves down in y)
      1 = rise     (sphere moves up in y)
      2 = slide_lr (sphere moves left -> right in x)
      3 = slide_rl (sphere moves right -> left in x)

    Returns:
      seqs_np: (N, T, G, G, G)
      labels_np: (N,)
    """
    seqs = []
    labels = []

    base_center = np.array([GRID_SIZE // 2,
                            GRID_SIZE // 2,
                            GRID_SIZE // 2])

    for motion_id in range(4):
        for _ in range(n_per_motion):
            # static reference cube near bottom center
            floor_center = base_center.copy()
            floor_center[1] = 3  # low Y
            floor_cube = make_cube(size_frac=0.5, center=tuple(floor_center))

            # jitter the base a bit per sequence
            jitter = np.random.randint(-1, 2, size=3)
            jitter[1] = 0  # don't move floor up/down too much
            floor_center_seq = floor_center + jitter
            floor_center_seq = np.clip(floor_center_seq, 3, GRID_SIZE - 3)
            floor_cube = make_cube(size_frac=0.5, center=tuple(floor_center_seq))

            # motion offsets over T frames
            if motion_id == 0:  # fall
                # start higher, move downward
                start_y = GRID_SIZE - 4
                end_y = floor_center_seq[1] + 3
                ys = np.linspace(start_y, end_y, T_STEPS)
                xs = np.full(T_STEPS, floor_center_seq[0] + np.random.randint(-1, 2))
            elif motion_id == 1:  # rise
                start_y = floor_center_seq[1] + 2
                end_y = GRID_SIZE - 3
                ys = np.linspace(start_y, end_y, T_STEPS)
                xs = np.full(T_STEPS, floor_center_seq[0] + np.random.randint(-1, 2))
            elif motion_id == 2:  # slide_lr
                start_x = 3
                end_x = GRID_SIZE - 4
                xs = np.linspace(start_x, end_x, T_STEPS)
                ys = np.full(T_STEPS, floor_center_seq[1] + 4)
            else:  # slide_rl
                start_x = GRID_SIZE - 4
                end_x = 3
                xs = np.linspace(start_x, end_x, T_STEPS)
                ys = np.full(T_STEPS, floor_center_seq[1] + 4)

            zs = np.full(T_STEPS, floor_center_seq[2])  # fixed depth

            # build frames
            frames = []
            for t in range(T_STEPS):
                grid = make_empty()
                # add floor cube
                grid = np.maximum(grid, floor_cube)

                cx = int(xs[t])
                cy = int(ys[t])
                cz = int(zs[t])
                cx = np.clip(cx, 2, GRID_SIZE - 3)
                cy = np.clip(cy, 2, GRID_SIZE - 3)
                cz = np.clip(cz, 2, GRID_SIZE - 3)

                sphere = make_sphere(radius_frac=0.18,
                                     center=(cx, cy, cz))
                grid = np.maximum(grid, sphere)
                frames.append(grid)

            seq = np.stack(frames, axis=0)  # (T, G, G, G)
            seqs.append(seq)
            labels.append(motion_id)

    seqs_np = np.stack(seqs, axis=0)
    labels_np = np.array(labels, dtype=np.int64)
    return seqs_np, labels_np


# ------------------------------
# Models
# ------------------------------

class TimeEncoder(nn.Module):
    def __init__(self, latent_dim=24, grid_size=GRID_SIZE, T=T_STEPS):
        super().__init__()
        in_dim = T * grid_size * grid_size * grid_size
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )

    def forward(self, x):
        b = x.shape[0]
        x_flat = x.view(b, -1)
        z = self.net(x_flat)
        return z


class TimeDecoder(nn.Module):
    def __init__(self, latent_dim=24, grid_size=GRID_SIZE, T=T_STEPS):
        super().__init__()
        self.grid_size = grid_size
        self.T = T
        out_dim = T * grid_size * grid_size * grid_size
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, z):
        b = z.shape[0]
        x = self.net(z)
        x = x.view(b, self.T, self.grid_size, self.grid_size, self.grid_size)
        return x


class TimeMotionModel(nn.Module):
    def __init__(self, latent_dim=24, grid_size=GRID_SIZE, T=T_STEPS, n_motions=4):
        super().__init__()
        self.encoder = TimeEncoder(latent_dim=latent_dim,
                                   grid_size=grid_size,
                                   T=T)
        self.decoder = TimeDecoder(latent_dim=latent_dim,
                                   grid_size=grid_size,
                                   T=T)
        self.classifier = nn.Linear(latent_dim, n_motions)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        logits = self.classifier(z)
        return recon, logits, z


# ------------------------------
# Visualization helpers
# ------------------------------

def save_voxels_3d(voxels, out_path_img, title="frame", threshold=0.5):
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa

        filled = voxels >= threshold
        fig = plt.figure(figsize=(4, 4))
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

def train_time_model(num_epochs=800,
                     latent_dim=24,
                     lr=1e-3,
                     beta=0.4,
                     device="cpu"):
    seqs_np, labels_np = generate_motion_sequences(n_per_motion=250)
    seqs_t = torch.from_numpy(seqs_np).to(device)
    labels_t = torch.from_numpy(labels_np).to(device)

    model = TimeMotionModel(latent_dim=latent_dim,
                            grid_size=GRID_SIZE,
                            T=T_STEPS,
                            n_motions=len(MOTIONS)).to(device)

    bce = nn.BCELoss()
    ce = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        opt.zero_grad()
        recon, logits, z = model(seqs_t)
        loss_recon = bce(recon, seqs_t)
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
        recon, logits, z = model(seqs_t)
        loss_recon = bce(recon, seqs_t).item()
        loss_cls = ce(logits, labels_t).item()
        preds = logits.argmax(dim=1)
        acc = (preds == labels_t).float().mean().item()

    print("\nFinal metrics:")
    print(f"  Recon BCE:   {loss_recon:.6f}")
    print(f"  Motion CE:   {loss_cls:.6f}")
    print(f"  Accuracy:    {acc*100:.2f}%")

    z_np = z.cpu().numpy()
    labels_np = labels_np  # alias
    means = []
    for motion_id in range(len(MOTIONS)):
        mask = (labels_np == motion_id)
        mean_z = z_np[mask].mean(axis=0)
        means.append(mean_z)
        print(f"Mean latent for motion {motion_id} ({MOTIONS[motion_id]}):")
        print("  ", mean_z)

    print("\nPairwise distances between motion means (L2):")
    for i in range(len(MOTIONS)):
        for j in range(i + 1, len(MOTIONS)):
            d = np.linalg.norm(means[i] - means[j])
            print(f"  {MOTIONS[i]} <-> {MOTIONS[j]}: {d:.4f}")

    return model, seqs_np, labels_np, means


# ------------------------------
# Visualize one example sequence
# ------------------------------

def visualize_example_sequence(model, seqs_np, labels_np,
                               out_dir="outputs_time", device="cpu"):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    # pick one index per motion
    indices = []
    for motion_id in range(len(MOTIONS)):
        idxs = np.where(labels_np == motion_id)[0]
        if len(idxs) > 0:
            indices.append(idxs[0])

    seqs_t = torch.from_numpy(seqs_np).to(device)
    with torch.no_grad():
        recon, logits, z_all = model(seqs_t)
        preds_all = logits.argmax(dim=1).cpu().numpy()

    for idx in indices:
        label = labels_np[idx]
        motion_name = MOTIONS[label]
        pred = preds_all[idx]
        print(f"\nExample index {idx}, motion {motion_name}, pred {MOTIONS[pred]}")

        orig_seq = seqs_np[idx]          # (T, G, G, G)
        recon_seq = recon[idx].cpu().numpy()

        for t in range(T_STEPS):
            orig_frame = orig_seq[t]
            recon_frame = recon_seq[t]

            base = f"idx{idx}_{motion_name}_t{t}"

            # original
            save_voxels_3d(orig_frame,
                           os.path.join(out_dir, base + "_orig.png"),
                           title=f"{motion_name} orig t={t}")
            save_voxels_as_ply(orig_frame,
                               os.path.join(out_dir, base + "_orig.ply"))

            # reconstruction
            save_voxels_3d(recon_frame,
                           os.path.join(out_dir, base + "_recon.png"),
                           title=f"{motion_name} recon t={t}")
            save_voxels_as_ply(recon_frame,
                               os.path.join(out_dir, base + "_recon.ply"))


def main():
    device = "cpu"
    print(f"Using device: {device}")
    os.makedirs("outputs_time", exist_ok=True)

    model, seqs_np, labels_np, means = train_time_model(
        num_epochs=800,
        latent_dim=24,
        lr=1e-3,
        beta=0.4,
        device=device,
    )

    visualize_example_sequence(
        model=model,
        seqs_np=seqs_np,
        labels_np=labels_np,
        out_dir="outputs_time",
        device=device,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
