#!/usr/bin/env python3
# geomlang_edges_relternary_train64_latent256_phase4.py
#
# Phase 4: Factorized "between" + "closeness" supervision (orthogonal targets)
#
# Targets:
#   - between_score b in [0,1]
#   - t_on_BC       t in [0,1]  (projection coordinate along BC segment)
#   - closer_sign   s in {0,1}  (A closer to B than C?)
#   - closer_mag    m in [-1,1] (scaled distance difference)
#   - overlap_B     oB in {0,1}
#   - overlap_C     oC in {0,1}
#
# Derived 7-way labels (for eval only):
#   overlap_B, overlap_C, between, left_of, right_of, closer_to_B, closer_to_C

import os, math, json, random
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------
# Config
# -----------------------
IMG_SIZE   = 64
LATENT_DIM = 256
NUM_CH     = 4  # A,B,C fills + union edges
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR    = "outputs_edges_relternary256_phase4"
CKPT_PATH  = os.path.join(OUT_DIR, "scene_model_edges_relternary256_phase4.pt")
os.makedirs(OUT_DIR, exist_ok=True)

# dataset sizes
N_TRAIN = 12000
N_VAL   = 2000
BATCH_SIZE = 256
EPOCHS  = 40
LR      = 3e-4

# augment
ROT90_PROB = 1.0

# shape params
SHAPES = (0, 0, 0)  # 0=circle (kept simple); you can expand later
SIZES  = (10, 10, 10)

# geometric tolerances / scales
SIGMA_PERP    = 2.0     # controls between_score falloff from line
SIGMA_OUTSIDE = 0.15    # controls between_score falloff outside segment in t-units
CLOSER_SIGMA  = 8.0     # controls closer_mag scaling
OVERLAP_TOL   = 3.0     # pixel tolerance for overlap
MIN_BC_DIST   = 14.0    # keep B and C separated

# loss weights (tune as needed)
W_RECON   = 1.0
W_BETWEEN = 1.0
W_TPROJ   = 1.0
W_CSIGN   = 1.0
W_CMAG    = 1.0
W_OV      = 1.0

TAG = "[train64-ternary-phase4-256]"


# -----------------------
# Geometry helpers
# -----------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def l2(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def proj_t_and_perp(A, B, C) -> Tuple[float, float]:
    """Return (t_raw, d_perp) where t is projection coord of A onto line BC,
    and d_perp is perpendicular distance to the infinite line BC.
    """
    Ax, Ay = A; Bx, By = B; Cx, Cy = C
    vx, vy = (Cx - Bx), (Cy - By)
    wx, wy = (Ax - Bx), (Ay - By)
    vv = vx*vx + vy*vy + 1e-9
    t = (wx*vx + wy*vy) / vv
    # perpendicular distance via area / |v|
    cross = abs(wx*vy - wy*vx)
    d_perp = cross / (math.sqrt(vv) + 1e-9)
    return float(t), float(d_perp)

def between_score(A, B, C, sigma_perp=SIGMA_PERP, sigma_out=SIGMA_OUTSIDE) -> float:
    t_raw, d_perp = proj_t_and_perp(A, B, C)
    # outside-ness: distance outside [0,1] in t units (0 if inside)
    out = 0.0
    if t_raw < 0.0: out = -t_raw
    if t_raw > 1.0: out = t_raw - 1.0
    b = math.exp(-(d_perp*d_perp)/(sigma_perp*sigma_perp + 1e-9)) * \
        math.exp(-(out*out)/(sigma_out*sigma_out + 1e-9))
    return float(clamp(b, 0.0, 1.0))

def closer_sign_and_mag(A, B, C, sigma=CLOSER_SIGMA) -> Tuple[int, float]:
    dB = l2(A, B)
    dC = l2(A, C)
    # sign: 1 if closer to B
    s = 1 if dB < dC else 0
    # magnitude in [-1,1]
    m = math.tanh((dC - dB) / (sigma + 1e-9))
    return int(s), float(m)

def overlap_flag(A, B, sizeA, sizeB, tol=OVERLAP_TOL) -> int:
    # circles: overlap if center distance <= radii sum + tol
    return 1 if l2(A, B) <= (sizeA + sizeB + tol) else 0

def maybe_rot90_triplet(A, B, C, p=ROT90_PROB):
    if random.random() > p:
        return A, B, C
    k = random.choice([0, 1, 2, 3])
    if k == 0:
        return A, B, C
    # rotate around image center (IMG_SIZE/2)
    cx = (IMG_SIZE - 1) / 2.0
    cy = (IMG_SIZE - 1) / 2.0
    def rot(pt):
        x, y = pt
        x0 = x - cx
        y0 = y - cy
        for _ in range(k):
            x0, y0 = -y0, x0
        return (int(round(x0 + cx)), int(round(y0 + cy)))
    return rot(A), rot(B), rot(C)


# -----------------------
# Rendering (simple circles + union edges)
# -----------------------
def draw_circle(mask, cx, cy, r):
    H, W = mask.shape
    y0 = max(0, cy - r); y1 = min(H, cy + r + 1)
    x0 = max(0, cx - r); x1 = min(W, cx + r + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist2 = (xx - cx)*(xx - cx) + (yy - cy)*(yy - cy)
    mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], (dist2 <= r*r).astype(np.float32))
    return mask

def make_edges(A_mask, B_mask, C_mask):
    union = ((A_mask > 0.5) | (B_mask > 0.5) | (C_mask > 0.5)).astype(np.uint8)
    # simple morph gradient via 4-neighborhood
    up    = np.pad(union[1:, :], ((0,1),(0,0)), mode="constant", constant_values=0)
    down  = np.pad(union[:-1,:], ((1,0),(0,0)), mode="constant", constant_values=0)
    left  = np.pad(union[:,1:], ((0,0),(0,1)), mode="constant", constant_values=0)
    right = np.pad(union[:,:-1],((0,0),(1,0)), mode="constant", constant_values=0)
    neigh = (up | down | left | right)
    edge = (union ^ neigh).astype(np.float32)
    return edge

def render_scene_ABC(Ax, Ay, Bx, By, Cx, Cy, sizeA, sizeB, sizeC):
    H = W = IMG_SIZE
    A = np.zeros((H, W), np.float32)
    B = np.zeros((H, W), np.float32)
    C = np.zeros((H, W), np.float32)
    draw_circle(A, Ax, Ay, sizeA)
    draw_circle(B, Bx, By, sizeB)
    draw_circle(C, Cx, Cy, sizeC)
    E = make_edges(A, B, C)
    x = np.stack([A, B, C, E], axis=0)  # C,H,W
    return x


# -----------------------
# Dataset
# -----------------------
class GeomEdgesTernary64DatasetPhase4(Dataset):
    def __init__(self, N: int, seed: int = 0):
        self.N = int(N)
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return self.N

    def _sample_BC(self):
        # sample B,C with minimum separation
        for _ in range(200):
            Bx = self.rng.randint(10, IMG_SIZE-10)
            By = self.rng.randint(10, IMG_SIZE-10)
            Cx = self.rng.randint(10, IMG_SIZE-10)
            Cy = self.rng.randint(10, IMG_SIZE-10)
            if math.hypot(Cx-Bx, Cy-By) >= MIN_BC_DIST:
                return (Bx, By), (Cx, Cy)
        # fallback
        return (18, 32), (46, 32)

    def __getitem__(self, idx: int):
        sizeA, sizeB, sizeC = SIZES

        B, C = self._sample_BC()

        # sample A anywhere reasonable
        Ax = int(self.rng.randint(8, IMG_SIZE-8))
        Ay = int(self.rng.randint(8, IMG_SIZE-8))
        A = (Ax, Ay)

        # augmentation rotation
        A2, B2, C2 = maybe_rot90_triplet(A, B, C, p=ROT90_PROB)

        # targets
        t_raw, d_perp = proj_t_and_perp(A2, B2, C2)
        t_clip = float(clamp(t_raw, 0.0, 1.0))
        b = between_score(A2, B2, C2)
        cs, cm = closer_sign_and_mag(A2, B2, C2)
        oB = overlap_flag(A2, B2, sizeA, sizeB)
        oC = overlap_flag(A2, C2, sizeA, sizeC)

        x = render_scene_ABC(A2[0], A2[1], B2[0], B2[1], C2[0], C2[1], sizeA, sizeB, sizeC)

        # return tensors
        # x: float32 [C,H,W]
        # targets: float32 scalars
        return (
            torch.from_numpy(x).float(),
            torch.tensor([b], dtype=torch.float32),
            torch.tensor([t_clip], dtype=torch.float32),
            torch.tensor([cs], dtype=torch.float32),
            torch.tensor([cm], dtype=torch.float32),
            torch.tensor([oB], dtype=torch.float32),
            torch.tensor([oC], dtype=torch.float32),
        )


# -----------------------
# Model (conv AE + multi-head)
# -----------------------
class Encoder64(nn.Module):
    def __init__(self, in_ch=NUM_CH, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 4, 2, 1), nn.ReLU(True),  # 32x32
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True),     # 16x16
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(True),    # 8x8
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(True),   # 4x4
        )
        self.fc = nn.Linear(256*4*4, latent_dim)

    def forward(self, x):
        h = self.net(x)
        h = h.view(h.size(0), -1)
        z = self.fc(h)
        return z

class Decoder64(nn.Module):
    def __init__(self, out_ch=NUM_CH, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256*4*4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(True),  # 8x8
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(True),   # 16x16
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True),    # 32x32
            nn.ConvTranspose2d(32, out_ch, 4, 2, 1),               # 64x64
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z).view(z.size(0), 256, 4, 4)
        x_hat = self.net(h)
        return x_hat

class SceneModelTernaryEdges64_256_Phase4(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder64(NUM_CH, LATENT_DIM)
        self.dec = Decoder64(NUM_CH, LATENT_DIM)

        # Heads
        self.between_head = nn.Sequential(nn.Linear(LATENT_DIM, 128), nn.ReLU(True), nn.Linear(128, 1))
        self.tproj_head   = nn.Sequential(nn.Linear(LATENT_DIM, 128), nn.ReLU(True), nn.Linear(128, 1))
        self.cs_head      = nn.Sequential(nn.Linear(LATENT_DIM, 128), nn.ReLU(True), nn.Linear(128, 1))
        self.cm_head      = nn.Sequential(nn.Linear(LATENT_DIM, 128), nn.ReLU(True), nn.Linear(128, 1))
        self.oB_head      = nn.Sequential(nn.Linear(LATENT_DIM, 128), nn.ReLU(True), nn.Linear(128, 1))
        self.oC_head      = nn.Sequential(nn.Linear(LATENT_DIM, 128), nn.ReLU(True), nn.Linear(128, 1))

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat


def main():
    print(f"{TAG} Using device: {DEVICE}")
    print(f"{TAG} Targets: between_score, t_on_BC, closer_sign, closer_mag, overlap_B, overlap_C")
    print(f"{TAG} Params: SIGMA_PERP={SIGMA_PERP}, SIGMA_OUTSIDE={SIGMA_OUTSIDE}, CLOSER_SIGMA={CLOSER_SIGMA}, OVERLAP_TOL={OVERLAP_TOL}")
    print(f"{TAG} Rotation90 prob = {ROT90_PROB}")

    train_ds = GeomEdgesTernary64DatasetPhase4(N_TRAIN, seed=0)
    val_ds   = GeomEdgesTernary64DatasetPhase4(N_VAL, seed=1)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SceneModelTernaryEdges64_256_Phase4().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    bce = nn.BCEWithLogitsLoss()
    reg = nn.SmoothL1Loss()

    def run_epoch(dl, train: bool):
        model.train(train)
        total = 0.0
        n = 0

        for x, b, t, cs, cm, oB, oC in dl:
            x  = x.to(DEVICE)
            b  = b.to(DEVICE)
            t  = t.to(DEVICE)
            cs = cs.to(DEVICE)
            cm = cm.to(DEVICE)
            oB = oB.to(DEVICE)
            oC = oC.to(DEVICE)

            z, x_hat = model(x)

            # recon
            loss_recon = F.mse_loss(x_hat, x)

            # heads
            b_logit  = model.between_head(z)
            t_logit  = model.tproj_head(z)
            cs_logit = model.cs_head(z)
            cm_raw   = model.cm_head(z)
            oB_logit = model.oB_head(z)
            oC_logit = model.oC_head(z)

            b_pred = torch.sigmoid(b_logit)        # [0,1]
            t_pred = torch.sigmoid(t_logit)        # [0,1]
            cm_pred = torch.tanh(cm_raw)           # [-1,1]

            loss_between = reg(b_pred, b)
            loss_tproj   = reg(t_pred, t)
            loss_csign   = bce(cs_logit, cs)
            loss_cmag    = reg(cm_pred, cm)
            loss_oB      = bce(oB_logit, oB)
            loss_oC      = bce(oC_logit, oC)

            loss = (W_RECON*loss_recon +
                    W_BETWEEN*loss_between +
                    W_TPROJ*loss_tproj +
                    W_CSIGN*loss_csign +
                    W_CMAG*loss_cmag +
                    W_OV*(loss_oB + loss_oC))

            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            total += float(loss.item()) * x.size(0)
            n += x.size(0)

        return total / max(1, n)

    best_val = 1e9
    for ep in range(1, EPOCHS+1):
        tr = run_epoch(train_dl, train=True)
        va = run_epoch(val_dl,   train=False)
        print(f"{TAG} Epoch {ep:3d}/{EPOCHS} | train={tr:.4f} | val={va:.4f}")

        if va < best_val:
            best_val = va
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": ep,
                "val_loss": va,
                "config": {
                    "IMG_SIZE": IMG_SIZE,
                    "LATENT_DIM": LATENT_DIM,
                    "NUM_CH": NUM_CH,
                }
            }, CKPT_PATH)

    print(f"{TAG} Saved -> {CKPT_PATH}")


if __name__ == "__main__":
    main()
