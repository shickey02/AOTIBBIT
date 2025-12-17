#!/usr/bin/env python3
"""
geomlang_prototype.py

Phase 1 prototype for a geometric "thought language":
- Each concept = 3D voxel grid (16x16x16)
- Each concept has a latent code z in R^latent_dim
- A decoder network maps z -> voxel grid (shape)
- We test occlusion by optimizing z to match partial observations

Now with:
- Forced CPU (RTX 5070 too new for current PyTorch wheel)
- 3D voxel visualizations (matplotlib)
- PLY point cloud export for Blender
"""

import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

GRID_SIZE = 16  # 16x16x16


# ------------------------------
# 1. Utility: make simple 3D shapes
# ------------------------------

def make_empty():
    return np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.float32)


def make_sphere(radius_frac=0.3):
    """Sphere centered in the grid."""
    grid = make_empty()
    center = (GRID_SIZE - 1) / 2.0
    radius = GRID_SIZE * radius_frac
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            for z in range(GRID_SIZE):
                dx = x - center
                dy = y - center
                dz = z - center
                if math.sqrt(dx * dx + dy * dy + dz * dz) <= radius:
                    grid[x, y, z] = 1.0
    return grid


def make_cube(size_frac=0.5):
    """Axis-aligned cube centered in the grid."""
    grid = make_empty()
    size = int(GRID_SIZE * size_frac)
    start = (GRID_SIZE - size) // 2
    end = start + size
    grid[start:end, start:end, start:end] = 1.0
    return grid


def make_cylinder(axis="z", radius_frac=0.25):
    """Cylinder aligned along a given axis."""
    grid = make_empty()
    center = (GRID_SIZE - 1) / 2.0
    radius = GRID_SIZE * radius_frac

    if axis == "z":
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                dx = x - center
                dy = y - center
                if math.sqrt(dx * dx + dy * dy) <= radius:
                    grid[x, y, :] = 1.0
    elif axis == "x":
        for y in range(GRID_SIZE):
            for z in range(GRID_SIZE):
                dy = y - center
                dz = z - center
                if math.sqrt(dy * dy + dz * dz) <= radius:
                    grid[:, y, z] = 1.0
    else:  # axis == "y"
        for x in range(GRID_SIZE):
            for z in range(GRID_SIZE):
                dx = x - center
                dz = z - center
                if math.sqrt(dx * dx + dz * dz) <= radius:
                    grid[x, :, z] = 1.0
    return grid


def make_L_shape():
    """Simple L-shape: two perpendicular bars."""
    grid = make_empty()
    # Horizontal bar
    grid[GRID_SIZE // 4:3 * GRID_SIZE // 4, GRID_SIZE // 4, GRID_SIZE // 4] = 1.0
    # Vertical bar
    grid[GRID_SIZE // 4, GRID_SIZE // 4:3 * GRID_SIZE // 4, GRID_SIZE // 4] = 1.0
    return grid


def generate_concept_dataset():
    """Return a small list of concept voxel grids (N, 16,16,16)."""
    shapes = []
    shapes.append(make_sphere(radius_frac=0.3))
    shapes.append(make_cube(size_frac=0.5))
    shapes.append(make_cylinder(axis="z", radius_frac=0.25))
    shapes.append(make_L_shape())
    arr = np.stack(shapes, axis=0)  # (N, 16,16,16)
    return arr


# ------------------------------
# 2. Model: latent -> voxel grid decoder
# ------------------------------

class Decoder(nn.Module):
    def __init__(self, latent_dim=16, grid_size=GRID_SIZE):
        super().__init__()
        self.latent_dim = latent_dim
        self.grid_size = grid_size
        out_dim = grid_size * grid_size * grid_size

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
            nn.Sigmoid(),  # output in [0,1] as occupancy probs
        )

    def forward(self, z):
        # z: (batch, latent_dim)
        x = self.net(z)  # (batch, out_dim)
        x = x.view(-1, self.grid_size, self.grid_size, self.grid_size)
        return x


# ------------------------------
# 3. Training latent codes + decoder
# ------------------------------

def train_decoder(num_epochs=1500, latent_dim=16, lr=1e-3, device="cpu"):
    concepts = generate_concept_dataset()  # (N, 16,16,16)
    concepts_t = torch.from_numpy(concepts).to(device)  # (N, 16,16,16)

    num_concepts = concepts_t.shape[0]

    # One latent vector per concept, learnable
    latents = nn.Embedding(num_concepts, latent_dim)
    torch.nn.init.normal_(latents.weight, mean=0.0, std=0.1)

    decoder = Decoder(latent_dim=latent_dim).to(device)

    params = list(decoder.parameters()) + list(latents.parameters())
    optimizer = optim.Adam(params, lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        idx = torch.arange(num_concepts, device=device)
        z = latents(idx)  # (N, latent_dim)
        recon = decoder(z)  # (N, 16,16,16)

        loss = criterion(recon, concepts_t)

        loss.backward()
        optimizer.step()

        if epoch % 200 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch}/{num_epochs} | Loss: {loss.item():.6f}")

    return decoder, latents, concepts_t


# ------------------------------
# 4. Occlusion test: recover latent from partial observations
# ------------------------------

def occlude_voxels(voxel_grid, drop_prob=0.5):
    """
    Randomly "hide" some voxels as unknown.
    Return:
      observed: same shape, but zeros where occluded
      mask: 1 where observed, 0 where occluded
    """
    mask = (np.random.rand(*voxel_grid.shape) > drop_prob).astype(np.float32)
    observed = voxel_grid * mask
    return observed, mask


def recover_from_occlusion(decoder, target_voxels, mask, latent_dim=16, steps=500, lr=1e-2, device="cpu"):
    """
    Given a decoder and a partially observed voxel grid (with mask),
    optimize a latent vector z to best fit the observed voxels.
    """
    decoder.eval()

    target = torch.from_numpy(target_voxels).to(device)  # (16,16,16)
    mask_t = torch.from_numpy(mask).to(device)

    # Start from random latent
    z = torch.randn(1, latent_dim, device=device, requires_grad=True)
    optimizer = optim.Adam([z], lr=lr)
    criterion = nn.BCELoss()

    for step in range(steps):
        optimizer.zero_grad()
        recon = decoder(z)[0]  # (16,16,16)
        # Only compare on observed positions
        loss = criterion(recon * mask_t, target * mask_t)
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == steps - 1:
            print(f"  Recover step {step}/{steps} | Loss: {loss.item():.6f}")

    with torch.no_grad():
        final_recon = decoder(z)[0].cpu().numpy()
    return final_recon


# ------------------------------
# 5. Visualization helpers
# ------------------------------

def save_mid_slice_triplet(original, observed, recon, out_path):
    """Save a 2D mid-slice comparison (as before)."""
    try:
        import matplotlib.pyplot as plt

        mid = GRID_SIZE // 2
        plt.figure(figsize=(10, 3))

        plt.subplot(1, 3, 1)
        plt.title("Original (mid slice)")
        plt.imshow(original[:, :, mid], cmap="gray")

        plt.subplot(1, 3, 2)
        plt.title("Observed (mid slice)")
        plt.imshow(observed[:, :, mid], cmap="gray")

        plt.subplot(1, 3, 3)
        plt.title("Recon (mid slice)")
        plt.imshow(recon[:, :, mid], cmap="gray")

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved mid-slice visualization to {out_path}")
    except ImportError:
        print("matplotlib not installed; skipping 2D visualization.")


def save_voxels_3d(voxels, out_path_img, title="voxel_shape", threshold=0.5):
    """Save a simple 3D voxel plot using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

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
    """
    Export voxels as a point cloud PLY file.
    Each occupied voxel above threshold becomes one vertex.
    Can be imported into Blender (File -> Import -> PLY).
    """
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
            # You can leave coordinates in voxel space; Blender can scale/translate
            f.write(f"{x} {y} {z}\n")

    print(f"Saved PLY point cloud to {out_path_ply}")


# ------------------------------
# 6. Main glue: train & run one occlusion experiment
# ------------------------------

def main():
    # Force CPU because RTX 5070 (sm_120) is too new for current PyTorch wheel
    device = "cpu"
    print(f"Using device: {device}")

    os.makedirs("outputs", exist_ok=True)

    # 1) Train decoder + latent codes
    decoder, latents, concepts_t = train_decoder(
        num_epochs=1500,
        latent_dim=16,
        lr=1e-3,
        device=device,
    )

    # 2) Pick a concept to occlude & recover
    concept_idx = 0  # 0 = sphere, 1 = cube, etc.
    original = concepts_t[concept_idx].cpu().numpy()

    print("\nRunning occlusion + recovery test...")
    observed, mask = occlude_voxels(original, drop_prob=0.6)

    # 3) Recover latent from occluded shape
    recon = recover_from_occlusion(
        decoder=decoder,
        target_voxels=observed,
        mask=mask,
        latent_dim=16,
        steps=600,
        lr=5e-2,
        device=device,
    )

    # 4) Quantitative check
    diff = np.mean(np.abs(recon - original))
    print(f"\nMean absolute difference between recon and original: {diff:.4f}")

    # 5) Save visualizations
    save_mid_slice_triplet(
        original=original,
        observed=observed,
        recon=recon,
        out_path="outputs/occlusion_recon_2d.png",
    )

    save_voxels_3d(
        original,
        out_path_img="outputs/original_3d.png",
        title="Original concept"
    )
    save_voxels_3d(
        observed,
        out_path_img="outputs/observed_3d.png",
        title="Observed (occluded)"
    )
    save_voxels_3d(
        recon,
        out_path_img="outputs/recon_3d.png",
        title="Reconstruction"
    )

    # 6) Save PLY point clouds for Blender
    save_voxels_as_ply(
        original,
        out_path_ply="outputs/original.ply",
        threshold=0.5,
    )
    save_voxels_as_ply(
        recon,
        out_path_ply="outputs/recon.ply",
        threshold=0.5,
    )


if __name__ == "__main__":
    main()
