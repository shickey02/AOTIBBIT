#!/usr/bin/env python3
"""
geomlang_autoencoder.py

Phase 2 prototype for a geometric "thought language":

- Generate MANY 3D shapes (spheres, cubes, cylinders, L-shapes) with random parameters.
- Train an AUTOENCODER:
    encoder: voxel grid (16x16x16) -> latent vector z
    decoder: latent vector z -> voxel grid
- After training, we:
    1) Do occlusion-based reconstruction (as before) by optimizing z.
    2) Pick two shapes A and B, encode to z_A and z_B,
       and interpolate between them to see smooth MORPHING
       from one concept to another in latent space.

Includes:
- 2D mid-slice visualization
- 3D voxel visualization (PNG)
- PLY export for Blender

CPU-only (your RTX 5070 is too new for current PyTorch wheel).
"""

import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

GRID_SIZE = 16  # 16x16x16


# ------------------------------
# 1. Shape generators (now PARAMETRIC)
# ------------------------------

def make_empty():
    return np.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=np.float32)


def make_sphere(radius_frac=0.25, center=None):
    grid = make_empty()
    if center is None:
        center = ((GRID_SIZE - 1) / 2.0,) * 3
    cx, cy, cz = center
    radius = GRID_SIZE * radius_frac
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            for z in range(GRID_SIZE):
                dx = x - cx
                dy = y - cy
                dz = z - cz
                if math.sqrt(dx*dx + dy*dy + dz*dz) <= radius:
                    grid[x, y, z] = 1.0
    return grid


def make_cube(size_frac=0.4, center=None):
    grid = make_empty()
    if center is None:
        center = ((GRID_SIZE - 1) / 2.0,) * 3
    cx, cy, cz = center
    size = int(GRID_SIZE * size_frac)
    # compute bounds around center
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
    radius = GRID_SIZE * radius_frac
    length = int(GRID_SIZE * length_frac)

    if axis == "z":
        z0 = max(0, int(cz - length / 2))
        z1 = min(GRID_SIZE, int(cz + length / 2))
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                dx = x - cx
                dy = y - cy
                if math.sqrt(dx*dx + dy*dy) <= radius:
                    grid[x, y, z0:z1] = 1.0
    elif axis == "x":
        x0 = max(0, int(cx - length / 2))
        x1 = min(GRID_SIZE, int(cx + length / 2))
        for y in range(GRID_SIZE):
            for z in range(GRID_SIZE):
                dy = y - cy
                dz = z - cz
                if math.sqrt(dy*dy + dz*dz) <= radius:
                    grid[x0:x1, y, z] = 1.0
    else:  # "y"
        y0 = max(0, int(cy - length / 2))
        y1 = min(GRID_SIZE, int(cy + length / 2))
        for x in range(GRID_SIZE):
            for z in range(GRID_SIZE):
                dx = x - cx
                dz = z - cz
                if math.sqrt(dx*dx + dz*dz) <= radius:
                    grid[x, y0:y1, z] = 1.0
    return grid


def make_L_shape(thickness=2, length_frac=0.5, center=None):
    grid = make_empty()
    if center is None:
        center = ((GRID_SIZE - 1) / 2.0,) * 3
    cx, cy, cz = map(int, center)
    length = int(GRID_SIZE * length_frac)
    half = length // 2

    # Horizontal bar in x-axis
    x0 = max(0, cx - half)
    x1 = min(GRID_SIZE, cx + half)
    y0 = max(0, cy - thickness // 2)
    y1 = min(GRID_SIZE, cy + thickness // 2)
    z0 = max(0, cz - thickness // 2)
    z1 = min(GRID_SIZE, cz + thickness // 2)
    grid[x0:x1, y0:y1, z0:z1] = 1.0

    # Vertical bar in y-axis
    x0 = max(0, cx - thickness // 2)
    x1 = min(GRID_SIZE, cx + thickness // 2)
    y0 = max(0, cy - half)
    y1 = min(GRID_SIZE, cy + half)
    grid[x0:x1, y0:y1, z0:z1] = 1.0

    return grid


def random_center(margin=3):
    """Random center within the grid, leaving some margin so shapes don't clip too bad."""
    return (
        np.random.randint(margin, GRID_SIZE - margin),
        np.random.randint(margin, GRID_SIZE - margin),
        np.random.randint(margin, GRID_SIZE - margin),
    )


def generate_random_shapes(n_shapes=128):
    """
    Generate a dataset of random parametric shapes.
    Each sample = one shape in the 16x16x16 grid.
    """
    shapes = []
    for _ in range(n_shapes):
        shape_type = np.random.choice(["sphere", "cube", "cylinder", "L"])
        center = random_center()
        if shape_type == "sphere":
            radius_frac = np.random.uniform(0.18, 0.35)
            grid = make_sphere(radius_frac=radius_frac, center=center)
        elif shape_type == "cube":
            size_frac = np.random.uniform(0.25, 0.55)
            grid = make_cube(size_frac=size_frac, center=center)
        elif shape_type == "cylinder":
            axis = np.random.choice(["x", "y", "z"])
            radius_frac = np.random.uniform(0.12, 0.25)
            length_frac = np.random.uniform(0.5, 1.0)
            grid = make_cylinder(axis=axis, radius_frac=radius_frac,
                                 center=center, length_frac=length_frac)
        else:  # "L"
            thickness = np.random.randint(1, 4)
            length_frac = np.random.uniform(0.4, 0.8)
            grid = make_L_shape(thickness=thickness, length_frac=length_frac, center=center)

        shapes.append(grid)

    return np.stack(shapes, axis=0)  # (N, 16,16,16)


# ------------------------------
# 2. Autoencoder model
# ------------------------------

class Encoder(nn.Module):
    def __init__(self, latent_dim=16, grid_size=GRID_SIZE):
        super().__init__()
        in_dim = grid_size * grid_size * grid_size
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x):
        # x: (batch, 16,16,16)
        batch = x.shape[0]
        x = x.view(batch, -1)
        z = self.net(x)
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dim=16, grid_size=GRID_SIZE):
        super().__init__()
        out_dim = grid_size * grid_size * grid_size
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
            nn.Sigmoid(),
        )
        self.grid_size = grid_size

    def forward(self, z):
        # z: (batch, latent_dim)
        x = self.net(z)
        batch = x.shape[0]
        x = x.view(batch, self.grid_size, self.grid_size, self.grid_size)
        return x


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=16, grid_size=GRID_SIZE):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim, grid_size=grid_size)
        self.decoder = Decoder(latent_dim=latent_dim, grid_size=grid_size)

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec, z


# ------------------------------
# 3. Training the autoencoder
# ------------------------------

def train_autoencoder(num_epochs=1000, latent_dim=16, lr=1e-3, device="cpu"):
    shapes_np = generate_random_shapes(n_shapes=128)  # (N,16,16,16)
    shapes_t = torch.from_numpy(shapes_np).to(device)  # float32 already

    model = Autoencoder(latent_dim=latent_dim, grid_size=GRID_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        recon, _ = model(shapes_t)
        loss = criterion(recon, shapes_t)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch}/{num_epochs} | Loss: {loss.item():.6f}")

    return model, shapes_np


# ------------------------------
# 4. Occlusion + latent optimization (decoder-only)
# ------------------------------

def occlude_voxels(voxel_grid, drop_prob=0.5):
    mask = (np.random.rand(*voxel_grid.shape) > drop_prob).astype(np.float32)
    observed = voxel_grid * mask
    return observed, mask


def recover_from_occlusion_decoder(decoder, observed, mask, latent_dim=16,
                                   steps=500, lr=1e-2, device="cpu"):
    """
    Optimize a latent z so that decoder(z) matches the observed voxels
    in the locations where mask == 1.
    """
    decoder.eval()
    target = torch.from_numpy(observed).to(device)
    mask_t = torch.from_numpy(mask).to(device)

    z = torch.randn(1, latent_dim, device=device, requires_grad=True)
    optimizer = optim.Adam([z], lr=lr)
    criterion = nn.BCELoss()

    for step in range(steps):
        optimizer.zero_grad()
        recon = decoder(z)[0]
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
        print(f"Saved 2D mid-slice visualization to {out_path}")
    except ImportError:
        print("matplotlib not installed; skipping 2D visualization.")


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
    print(f"Saved PLY point cloud to {out_path_ply}")


# ------------------------------
# 6. Latent interpolation (concept morphing)
# ------------------------------

def interpolate_and_visualize(model, shapes_np, idx_a, idx_b, out_dir,
                              n_steps=5, threshold=0.5, device="cpu"):
    """
    Encode two shapes A,B; linearly interpolate between z_A and z_B;
    decode to get intermediate shapes and save 3D PNGs.
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        a = torch.from_numpy(shapes_np[idx_a:idx_a+1]).to(device)  # (1,16,16,16)
        b = torch.from_numpy(shapes_np[idx_b:idx_b+1]).to(device)

        z_a = model.encoder(a)  # (1,latent_dim)
        z_b = model.encoder(b)

        for i, t in enumerate(np.linspace(0.0, 1.0, n_steps)):
            z_t = (1 - t) * z_a + t * z_b
            x_t = model.decoder(z_t)[0].cpu().numpy()

            out_path_img = os.path.join(out_dir, f"interp_{i:02d}_t{t:.2f}.png")
            title = f"Interp t={t:.2f}"
            save_voxels_3d(x_t, out_path_img=out_path_img, title=title, threshold=threshold)


# ------------------------------
# 7. Main
# ------------------------------

def main():
    device = "cpu"
    print(f"Using device: {device}")
    os.makedirs("outputs_ae", exist_ok=True)

    # 1) Train autoencoder on random shapes
    model, shapes_np = train_autoencoder(
        num_epochs=800,
        latent_dim=16,
        lr=1e-3,
        device=device,
    )

    # 2) Occlusion test on one random shape
    idx = np.random.randint(0, shapes_np.shape[0])
    original = shapes_np[idx]
    print(f"\nRunning occlusion + recovery on shape index {idx}...")
    observed, mask = occlude_voxels(original, drop_prob=0.6)

    recon = recover_from_occlusion_decoder(
        decoder=model.decoder,
        observed=observed,
        mask=mask,
        latent_dim=16,
        steps=500,
        lr=5e-2,
        device=device,
    )

    diff = np.mean(np.abs(recon - original))
    print(f"Mean absolute difference between recon and original: {diff:.4f}")

    save_mid_slice_triplet(
        original=original,
        observed=observed,
        recon=recon,
        out_path="outputs_ae/occlusion_recon_2d.png",
    )
    save_voxels_3d(
        original,
        out_path_img="outputs_ae/original_3d.png",
        title="Original (AE)"
    )
    save_voxels_3d(
        observed,
        out_path_img="outputs_ae/observed_3d.png",
        title="Observed (AE)"
    )
    save_voxels_3d(
        recon,
        out_path_img="outputs_ae/recon_3d.png",
        title="Reconstruction (AE)"
    )
    save_voxels_as_ply(
        original,
        out_path_ply="outputs_ae/original.ply",
        threshold=0.5,
    )
    save_voxels_as_ply(
        recon,
        out_path_ply="outputs_ae/recon.ply",
        threshold=0.5,
    )

    # 3) Latent interpolation between two random shapes
    idx_a = np.random.randint(0, shapes_np.shape[0])
    idx_b = np.random.randint(0, shapes_np.shape[0])
    print(f"\nLatent interpolation between shapes {idx_a} and {idx_b}...")
    interpolate_and_visualize(
        model=model,
        shapes_np=shapes_np,
        idx_a=idx_a,
        idx_b=idx_b,
        out_dir="outputs_ae/interp",
        n_steps=7,
        threshold=0.5,
        device=device,
    )
    print("Done.")


if __name__ == "__main__":
    main()
