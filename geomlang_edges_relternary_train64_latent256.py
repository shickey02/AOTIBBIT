#!/usr/bin/env python3
# geomlang_edges_relternary_train64_latent256.py
#
# 3-object + edges autoencoder (64x64) with LATENT_DIM=256
# Adds an ASYMMETRIC ternary head:
#   classify relation of A (red) relative to B (green) and C (blue)
#
# Channels:
#   0: A fill (red)
#   1: B fill (green)
#   2: C fill (blue)
#   3: edges (union outlines)
#
# Saves checkpoint to:
#   outputs_edges_relternary256/scene_model_edges_relternary256.pt

import os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------
# Config
# -----------------------
IMG_SIZE   = 64
NUM_CH     = 4
LATENT_DIM = 256

N_TRAIN    = 36000
N_VAL      = 6000
BATCH_SIZE = 128
N_EPOCHS   = 40
LR         = 1e-3

OUT_DIR         = "outputs_edges_relternary256"
CKPT_PATH       = os.path.join(OUT_DIR, "scene_model_edges_relternary256.pt")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Asymmetric ternary classes (A relative to B,C)
# -----------------------
T_BETWEEN, T_CLOSER_B, T_CLOSER_C, T_LEFT_BOTH, T_RIGHT_BOTH, T_ABOVE_BOTH, T_BELOW_BOTH = range(7)
T_NAMES = ["between", "closer_to_B", "closer_to_C", "left_of_both", "right_of_both", "above_both", "below_both"]

# Geometry tolerances
TOL_AXIS   = 2.0   # axis dominance / boundary tol
BAND_BETW  = 4.0   # how tightly A must stay near the line (in the minor axis) to count as "between"

# -----------------------
# Drawing helpers
# -----------------------
def draw_circle(mask, cx, cy, radius):
    H, W = mask.shape
    yy, xx = np.ogrid[:H, :W]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    mask[dist2 <= radius ** 2] = 1.0

def draw_square(mask, cx, cy, half_size):
    H, W = mask.shape
    x0 = max(0, cx - half_size)
    x1 = min(W, cx + half_size + 1)
    y0 = max(0, cy - half_size)
    y1 = min(H, cy + half_size + 1)
    mask[y0:y1, x0:x1] = 1.0

def make_edges(*masks):
    # union -> edge via 4-neighbor interior erosion
    union = np.zeros_like(masks[0], dtype=np.float32)
    for m in masks:
        union = np.maximum(union, (m > 0.5).astype(np.float32))

    interior = np.zeros_like(union)
    interior[1:-1, 1:-1] = (
        union[1:-1, 1:-1] *
        union[:-2, 1:-1] *
        union[2:, 1:-1] *
        union[1:-1, :-2] *
        union[1:-1, 2:]
    )
    edges = union - interior
    edges[edges < 0] = 0.0
    return edges

def _dist2(ax, ay, bx, by):
    dx = ax - bx
    dy = ay - by
    return dx*dx + dy*dy

def ternary_label_A_vs_BC(cxA, cyA, cxB, cyB, cxC, cyC,
                          tol_axis=TOL_AXIS, band_between=BAND_BETW):
    """
    Decide A's asymmetric role relative to (B,C).
    We first try to assign a "hard geometric role":
      - left/right/above/below of both (clear separation)
      - between B and C along dominant axis of segment BC (and near in minor axis)
    Otherwise we fall back to closer_to_B / closer_to_C.
    """

    # Hard "of both" tests
    minx = min(cxB, cxC)
    maxx = max(cxB, cxC)
    miny = min(cyB, cyC)
    maxy = max(cyB, cyC)

    if cxA < minx - tol_axis:
        return T_LEFT_BOTH
    if cxA > maxx + tol_axis:
        return T_RIGHT_BOTH
    if cyA < miny - tol_axis:
        return T_BELOW_BOTH
    if cyA > maxy + tol_axis:
        return T_ABOVE_BOTH

    # Between test: choose dominant axis of BC
    dxBC = cxC - cxB
    dyBC = cyC - cyB
    if abs(dxBC) >= abs(dyBC):
        # x-dominant: A.x between B.x and C.x, and A.y near their midpoint
        lo = min(cxB, cxC) + tol_axis
        hi = max(cxB, cxC) - tol_axis
        if lo <= cxA <= hi:
            midy = 0.5 * (cyB + cyC)
            if abs(cyA - midy) <= band_between:
                return T_BETWEEN
    else:
        # y-dominant
        lo = min(cyB, cyC) + tol_axis
        hi = max(cyB, cyC) - tol_axis
        if lo <= cyA <= hi:
            midx = 0.5 * (cxB + cxC)
            if abs(cxA - midx) <= band_between:
                return T_BETWEEN

    # Fallback: closer-to
    dAB = _dist2(cxA, cyA, cxB, cyB)
    dAC = _dist2(cxA, cyA, cxC, cyC)
    return T_CLOSER_B if dAB <= dAC else T_CLOSER_C

# -----------------------
# Dataset
# -----------------------
class GeomEdgesTernary64Dataset(Dataset):
    def __init__(self, n_samples, seed=None):
        super().__init__()
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        H = W = IMG_SIZE
        margin = 6
        rng = self.rng

        # shapes: 0 circle, 1 square
        shapeA = int(rng.integers(0, 2))
        shapeB = int(rng.integers(0, 2))
        shapeC = int(rng.integers(0, 2))

        # sizes: 4..14, softly correlated
        base = int(rng.integers(5, 13))
        sA = int(np.clip(base + int(rng.integers(-2, 3)), 4, 14))
        sB = int(np.clip(base + int(rng.integers(-2, 3)), 4, 14))
        sC = int(np.clip(base + int(rng.integers(-2, 3)), 4, 14))

        def sample_center(s):
            cx = int(rng.integers(margin + s, W - margin - s))
            cy = int(rng.integers(margin + s, H - margin - s))
            return cx, cy

        cxA, cyA = sample_center(sA)
        cxB, cyB = sample_center(sB)
        cxC, cyC = sample_center(sC)

        A = np.zeros((H, W), dtype=np.float32)
        B = np.zeros((H, W), dtype=np.float32)
        C = np.zeros((H, W), dtype=np.float32)

        if shapeA == 0: draw_circle(A, cxA, cyA, sA)
        else:           draw_square(A, cxA, cyA, sA)

        if shapeB == 0: draw_circle(B, cxB, cyB, sB)
        else:           draw_square(B, cxB, cyB, sB)

        if shapeC == 0: draw_circle(C, cxC, cyC, sC)
        else:           draw_square(C, cxC, cyC, sC)

        E = make_edges(A, B, C)
        img = np.stack([A, B, C, E], axis=0)

        t = ternary_label_A_vs_BC(cxA, cyA, cxB, cyB, cxC, cyC)

        return (
            torch.from_numpy(img),
            torch.tensor(t, dtype=torch.long),
            torch.tensor(shapeA, dtype=torch.long),
            torch.tensor(shapeB, dtype=torch.long),
            torch.tensor(shapeC, dtype=torch.long),
        )

# -----------------------
# Model
# -----------------------
class Encoder(nn.Module):
    def __init__(self, in_channels=NUM_CH, latent_dim=LATENT_DIM):
        super().__init__()
        # 64 -> 32 -> 16 -> 8 -> 4
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

class Decoder(nn.Module):
    def __init__(self, out_channels=NUM_CH, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, 4, 4)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))
        return x

class SceneModelEdgesTernary64_256(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.ternary_head = nn.Linear(LATENT_DIM, 7)
        # optional: keep shape heads for monitoring disentanglement
        self.shapeA_head = nn.Linear(LATENT_DIM, 2)
        self.shapeB_head = nn.Linear(LATENT_DIM, 2)
        self.shapeC_head = nn.Linear(LATENT_DIM, 2)

    def forward(self, x):
        z = self.encoder(x)
        rec = self.decoder(z)
        tlog = self.ternary_head(z)
        a = self.shapeA_head(z)
        b = self.shapeB_head(z)
        c = self.shapeC_head(z)
        return rec, tlog, a, b, c

    def encode(self, x): return self.encoder(x)
    def decode(self, z): return self.decoder(z)

# -----------------------
# Training
# -----------------------
def main():
    print(f"[train64-ternary-256] Using device: {DEVICE}")

    train_ds = GeomEdgesTernary64Dataset(N_TRAIN)
    val_ds   = GeomEdgesTernary64Dataset(N_VAL)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SceneModelEdgesTernary64_256().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    bce = nn.BCELoss()
    ce  = nn.CrossEntropyLoss()

    def run_epoch(loader, train=True):
        model.train(train)
        total = 0.0
        n = 0
        with torch.set_grad_enabled(train):
            for imgs, t, sA, sB, sC in loader:
                imgs = imgs.to(DEVICE)
                t    = t.to(DEVICE)
                sA   = sA.to(DEVICE)
                sB   = sB.to(DEVICE)
                sC   = sC.to(DEVICE)

                if train:
                    opt.zero_grad()

                rec, tlog, a, b, c = model(imgs)

                rec_loss = bce(rec, imgs)
                t_loss   = ce(tlog, t)
                sh_loss  = ce(a, sA) + ce(b, sB) + ce(c, sC)

                # Keep reconstruction dominant, but let the asymmetric task shape the latent
                loss = rec_loss + 0.35 * t_loss + 0.10 * sh_loss

                if train:
                    loss.backward()
                    opt.step()

                total += float(loss.item())
                n += 1
        return total / max(1, n)

    for epoch in range(1, N_EPOCHS + 1):
        tr = run_epoch(train_dl, train=True)
        va = run_epoch(val_dl,   train=False)
        print(f"[train64-ternary-256] Epoch {epoch:3d}/{N_EPOCHS} | train={tr:.4f} | val={va:.4f}")

    torch.save({"model_state_dict": model.state_dict()}, CKPT_PATH)
    print(f"[train64-ternary-256] Saved -> {CKPT_PATH}")

if __name__ == "__main__":
    main()
