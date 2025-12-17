#!/usr/bin/env python3
# geomlang_noncomm_loop_projection_latent256.py
#
# For several seeds, run a commutator-style rectangle loop in latent space
# and decompose the net displacement Δz into:
#   - component in span{g_x, g_y}
#   - component in curvature basis (selected residual PCs)
#   - leftover residual.
#
# Prints norms / fractions for each seed.

import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from geomlang_global_coords_latent256 import (
    GeomEdges64Dataset,
    SceneModelEdges256,
    load_scene_model,
    REL_LEFT, REL_RIGHT, REL_ABOVE, REL_BELOW, REL_OVERLAP, REL_NAMES,
)

TAG        = "[loopProj256]"
IMG_SIZE   = 64
LATENT_DIM = 256
N_SAMPLES  = 6000
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR    = "outputs_edges_relscale256"
os.makedirs(OUT_DIR, exist_ok=True)

# rectangle step size in generator coords
EPS_X = 0.6
EPS_Y = 0.6

# how many residual PCs to keep before picking curvature basis
RESID_DIM = 32
# which of those PCs form the curvature basis
CURV_IDXS = [0, 1, 2, 5]   # matches earlier experiments (h1,h2,h3,h4)


# ----------------- helpers -----------------

def encode_dataset(model, n_samples=N_SAMPLES):
    print(f"{TAG} Encoding dataset: N={n_samples}")
    ds = GeomEdges64Dataset(n_samples)
    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)

    all_imgs = []
    all_rel  = []
    all_z    = []

    with torch.no_grad():
        for imgs, rel, scale, s_r, s_b in dl:
            imgs = imgs.to(DEVICE)
            z    = model.encode(imgs)
            all_imgs.append(imgs.cpu())
            all_rel.append(rel.cpu())
            all_z.append(z.cpu())

    imgs = torch.cat(all_imgs, dim=0)   # (N, C, H, W)
    rel  = torch.cat(all_rel,  dim=0)   # (N,)
    z    = torch.cat(all_z,    dim=0)   # (N, D)

    print(f"{TAG} Latent shape: {tuple(z.shape)}")
    return imgs, rel, z


def compute_generators(z, rel):
    """Return g_x, g_y, and relation means mu_r (all torch CPU tensors)."""
    mus = []
    rel_np = rel.numpy()
    for rid in range(5):
        mask = (rel_np == rid)
        z_r = z[mask]
        mu_r = z_r.mean(dim=0)
        print(f"{TAG} relation {rid}: {REL_NAMES[rid]:>7}, "
              f"count={mask.sum()}, ||mu_r||={mu_r.norm():.3f}")
        mus.append(mu_r)

    mu_left, mu_right, mu_above, mu_below, mu_overlap = mus

    v_LR = mu_right - mu_left
    v_AB = mu_below - mu_above

    g_x = v_LR / v_LR.norm()
    g_y = v_AB / v_AB.norm()

    cos_xy = torch.dot(g_x, g_y).item()
    angle = math.degrees(math.acos(max(min(cos_xy, 1.0), -1.0)))
    print(f"{TAG} ||mu_right - mu_left|| = {v_LR.norm():.4f}")
    print(f"{TAG} ||mu_below - mu_above|| = {v_AB.norm():.4f}")
    print(f"{TAG} angle(g_x, g_y) = {angle:.2f}°")

    return g_x, g_y, mus


def residuals_and_pca(z, mu_overlap, g_x, g_y):
    """Compute residual subspace and PCA basis."""
    z = z.to(DEVICE)
    mu0 = mu_overlap.to(DEVICE)

    z_c = z - mu0.unsqueeze(0)      # center at overlap
    # project onto generator plane
    gx = g_x.to(DEVICE)
    gy = g_y.to(DEVICE)
    s = torch.dot(gx, gy).item()
    denom = 1.0 - s * s
    if denom <= 0:
        raise RuntimeError("Gram matrix singular in residuals_and_pca")

    a = torch.matmul(z_c, gx)       # <z_c, g_x>
    b = torch.matmul(z_c, gy)       # <z_c, g_y>
    t_x = (a - s * b) / denom
    t_y = (b - s * a) / denom

    proj_plane = (
        t_x.unsqueeze(1) * gx.unsqueeze(0)
        + t_y.unsqueeze(1) * gy.unsqueeze(0)
    )

    r = z_c - proj_plane            # residuals, shape (N, D)
    r_cpu = r.cpu().numpy()
    print(f"{TAG} Residual norms stats: mean={np.linalg.norm(r_cpu, axis=1).mean():.4f}")

    # PCA in residual space
    pca = PCA(n_components=RESID_DIM)
    r_low = pca.fit_transform(r_cpu)             # (N, RESID_DIM)
    comps = pca.components_                      # (RESID_DIM, D)

    print(f"{TAG} Residual PCA variance (first 10 PCs):")
    for i, var in enumerate(pca.explained_variance_ratio_[:10]):
        print(f"   PC{i+1:02d}: {var*100:5.2f}%")

    # curvature basis in full latent: columns are D-vectors
    H_full_np = comps[CURV_IDXS, :].T            # (D, k_basis) in NumPy
    H_full = torch.from_numpy(H_full_np).float() # <- convert to Torch

    print(f"{TAG} Using residual PCs {CURV_IDXS} as curvature basis (dim={H_full.shape[1]})")

    return r, (t_x, t_y), H_full



def loop_rectangle(model, z0, g_x, g_y, eps_x, eps_y):
    """Same commutator loop as in geomlang_noncomm_loops_latent256.py."""
    def flow_step(z, direction, eps):
        z_prop = z + eps * direction
        with torch.no_grad():
            x = model.decode(z_prop.unsqueeze(0))
            z_new = model.encode(x).squeeze(0)
        return z_new

    z0 = z0.to(DEVICE)
    zs = [z0]
    z1 = flow_step(zs[-1], g_x, +eps_x)
    zs.append(z1)
    z2 = flow_step(zs[-1], g_y, +eps_y)
    zs.append(z2)
    z3 = flow_step(zs[-1], g_x, -eps_x)
    zs.append(z3)
    z4 = flow_step(zs[-1], g_y, -eps_y)
    zs.append(z4)
    return zs


def choose_seeds(z, rel, mus):
    """Pick one left_of, one above, one overlap seed near their means."""
    seeds = {}
    rel_np = rel.numpy()

    def closest_idx(target_mu, rid):
        mask = (rel_np == rid)
        idxs = np.nonzero(mask)[0]
        z_r = z[mask]
        d = torch.norm(z_r - target_mu, dim=1)
        local = int(torch.argmin(d))
        return int(idxs[local])

    mu_left, mu_right, mu_above, mu_below, mu_overlap = mus

    seeds["left_of"] = closest_idx(mu_left, REL_LEFT)
    seeds["above"]   = closest_idx(mu_above, REL_ABOVE)
    seeds["overlap"] = closest_idx(mu_overlap, REL_OVERLAP)

    print(f"{TAG} Seeds (global indices): {seeds}")
    return seeds


def decompose_delta(delta, g_x, g_y, H_full):
    """
    delta: (D,)
    g_x, g_y: (D,), assumed unit length but not orthogonal
    H_full: (D, k_basis), orthonormal columns in residual subspace
    Returns dict with norms and fractions.
    """
    D = delta.shape[0]
    gx = g_x
    gy = g_y

    # 1) component in span{g_x, g_y} using Gram inverse
    s = torch.dot(gx, gy).item()
    denom = 1.0 - s * s
    a = torch.dot(delta, gx).item()
    b = torch.dot(delta, gy).item()
    t_x = (a - s * b) / denom
    t_y = (b - s * a) / denom
    delta_plane = t_x * gx + t_y * gy

    # 2) residual part orthogonal to plane
    delta_res = delta - delta_plane

    # 3) project residual onto curvature basis
    H = H_full.to(delta.device)          # (D, k)
    coeffs = torch.matmul(H.t(), delta_res)      # (k,)
    delta_curv = torch.matmul(H, coeffs)         # (D,)
    delta_rest = delta_res - delta_curv          # leftover

    total_norm = delta.norm().item()
    plane_norm = delta_plane.norm().item()
    curv_norm  = delta_curv.norm().item()
    rest_norm  = delta_rest.norm().item()

    return dict(
        total=total_norm,
        plane=plane_norm,
        curv=curv_norm,
        rest=rest_norm,
        frac_plane=plane_norm / total_norm if total_norm > 0 else 0.0,
        frac_curv=curv_norm / total_norm if total_norm > 0 else 0.0,
        frac_rest=rest_norm / total_norm if total_norm > 0 else 0.0,
    )


# ----------------- main -----------------

def main():
    print(f"{TAG} Using device: {DEVICE}")
    model = load_scene_model()  # from geomlang_global_coords_latent256
    model.to(DEVICE)
    model.eval()

    imgs, rel, z = encode_dataset(model, N_SAMPLES)
    g_x, g_y, mus = compute_generators(z, rel)

    # residual PCA + curvature basis in full latent
    r, (t_x, t_y), H_full = residuals_and_pca(z, mus[4], g_x, g_y)

    # choose seeds and analyze each
    seeds = choose_seeds(z, rel, mus)

    for name, idx in seeds.items():
        print(f"{TAG} Seed '{name}' idx={idx}, rel={REL_NAMES[rel[idx]]}")
        z0 = z[idx].to(DEVICE)

        zs_loop = loop_rectangle(model, z0, g_x.to(DEVICE), g_y.to(DEVICE),
                                 EPS_X, EPS_Y)
        z_end = zs_loop[-1].detach().cpu()
        delta = z_end - z[idx]

        stats = decompose_delta(delta, g_x, g_y, H_full)

        print(f"{TAG}   ||Δz||          = {stats['total']:.4f}")
        print(f"{TAG}   plane   norm    = {stats['plane']:.4f} "
              f"({stats['frac_plane']*100:5.1f}%)")
        print(f"{TAG}   curvature norm  = {stats['curv']:.4f} "
              f"({stats['frac_curv']*100:5.1f}%)")
        print(f"{TAG}   leftover  norm  = {stats['rest']:.4f} "
              f"({stats['frac_rest']*100:5.1f}%)")

if __name__ == "__main__":
    main()
