#!/usr/bin/env python3
# geomlang_edges_relternary_train64_latent256_phase7.py
#
# Phase 7: Fix left/right by canonicalizing BC and supervising lr_sign.
# Keeps AE structure from Phase 6 (ConvAEHeads), but changes targets/loss.
#
# Targets:
#   between_score (0..1) continuous
#   t_on_BC       (0..1) continuous
#   overlap_any   (0/1)  (max of overlap_B, overlap_C)
#   lr_sign       (0/1)  (0 = left of B→C, 1 = right of B→C)  [masked when "between"]

import os, math, random
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------
# Config
# -----------------------
IMG_SIZE   = 64
LATENT_DIM = 256
N_TRAIN    = 12000
N_VAL      = 2000
BATCH      = 128
EPOCHS     = 40
LR         = 2e-4

OUTDIR = "outputs_edges_relternary256_phase7"
CKPT   = os.path.join(OUTDIR, "scene_model_edges_relternary256_phase7.pt")

# Balanced buckets (same idea as Phase 6)
BUCKETS = ("between_clear", "between_overlap", "overlap_only", "non_between")
BUCKET_PROBS = (0.22, 0.22, 0.28, 0.28)

# Geometry params
R_A = 6
R_B = 7
R_C = 7

SIGMA_PERP     = 2.0
SIGMA_OUTSIDE  = 0.15
END_MARGIN_T   = 0.08

OVERLAP_TOL = 3.0

ROT90_PROB = 1.0

# Loss weights
W_RECON   = 1.0
W_BETW    = 1.0
W_TPROJ   = 0.6
W_OVERLAP = 0.6
W_LR      = 0.6

# Mask threshold: only learn lr when "not between"
BETWEEN_MASK_THR = 0.50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# Helpers
# -----------------------
def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def rot90_xy(x, y, n=1):
    cx = (IMG_SIZE - 1) / 2.0
    cy = (IMG_SIZE - 1) / 2.0
    dx = x - cx
    dy = y - cy
    n = n % 4
    for _ in range(n):
        dx, dy = -dy, dx
    return (dx + cx, dy + cy)

def dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def proj_t_and_perp(A, B, C):
    bx, by = B
    cx, cy = C
    ax, ay = A
    vx, vy = (cx - bx, cy - by)
    wx, wy = (ax - bx, ay - by)
    vv = vx*vx + vy*vy + 1e-9
    t = (wx*vx + wy*vy) / vv
    cross = abs(vx*wy - vy*wx)
    vnorm = math.sqrt(vv)
    perp = cross / (vnorm + 1e-9)
    seg_len = vnorm
    return t, perp, seg_len

def between_score(A, B, C):
    t, perp, seg_len = proj_t_and_perp(A, B, C)
    outside = 0.0
    if t < 0.0: outside = -t
    elif t > 1.0: outside = (t - 1.0)

    s_perp = math.exp(-(perp**2) / (2.0 * (SIGMA_PERP**2)))
    s_out  = math.exp(-(outside**2) / (2.0 * (SIGMA_OUTSIDE**2)))
    s = s_perp * s_out
    return float(s), float(clamp(t, 0.0, 1.0))

def overlap_flag(A, B, rA, rB):
    d = dist(A, B)
    return 1.0 if d <= (rA + rB - OVERLAP_TOL) else 0.0

def draw_scene(A, B, C):
    imgR = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
    imgB = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
    imgE = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
    dR = ImageDraw.Draw(imgR)
    dB = ImageDraw.Draw(imgB)
    dE = ImageDraw.Draw(imgE)

    def circle_bbox(p, r):
        return [p[0]-r, p[1]-r, p[0]+r, p[1]+r]

    dR.ellipse(circle_bbox(A, R_A), fill=255)
    dB.ellipse(circle_bbox(B, R_B), fill=255)
    dB.ellipse(circle_bbox(C, R_C), fill=255)

    dE.ellipse(circle_bbox(A, R_A), outline=255, width=1)
    dE.ellipse(circle_bbox(B, R_B), outline=255, width=1)
    dE.ellipse(circle_bbox(C, R_C), outline=255, width=1)

    arr = np.stack([
        np.array(imgR, dtype=np.float32) / 255.0,
        np.array(imgB, dtype=np.float32) / 255.0,
        np.array(imgE, dtype=np.float32) / 255.0,
    ], axis=0)
    return arr

def sample_BC():
    margin = 12
    while True:
        B = (random.randint(margin, IMG_SIZE-1-margin), random.randint(margin, IMG_SIZE-1-margin))
        C = (random.randint(margin, IMG_SIZE-1-margin), random.randint(margin, IMG_SIZE-1-margin))
        if dist(B, C) >= 18:
            return B, C

def sample_A_for_bucket(bucket, B, C):
    margin = 10
    for _ in range(2000):
        t0 = random.random()
        bx, by = B
        cx, cy = C
        vx, vy = (cx - bx, cy - by)

        vnorm = math.sqrt(vx*vx + vy*vy) + 1e-9
        nx, ny = (-vy / vnorm, vx / vnorm)

        px = bx + t0 * vx
        py = by + t0 * vy

        if bucket in ("between_clear", "between_overlap"):
            t0 = random.uniform(END_MARGIN_T, 1.0 - END_MARGIN_T)
            px = bx + t0 * vx
            py = by + t0 * vy
            perp = random.uniform(-2.0, 2.0)
        else:
            if random.random() < 0.55:
                perp = random.uniform(6.0, 18.0) * (1 if random.random() < 0.5 else -1)
            else:
                t0 = random.choice([random.uniform(-0.6, -0.05), random.uniform(1.05, 1.6)])
                px = bx + t0 * vx
                py = by + t0 * vy
                perp = random.uniform(-6.0, 6.0)

        ax = px + perp * nx
        ay = py + perp * ny
        ax = clamp(ax, margin, IMG_SIZE-1-margin)
        ay = clamp(ay, margin, IMG_SIZE-1-margin)
        A = (ax, ay)

        s_between, _ = between_score(A, B, C)
        oB = overlap_flag(A, B, R_A, R_B)
        oC = overlap_flag(A, C, R_A, R_C)

        if bucket == "between_clear":
            if s_between > 0.70 and (oB < 0.5 and oC < 0.5): return A
        elif bucket == "between_overlap":
            if s_between > 0.70 and (oB > 0.5 or oC > 0.5): return A
        elif bucket == "overlap_only":
            if s_between < 0.35 and (oB > 0.5 or oC > 0.5): return A
        elif bucket == "non_between":
            if s_between < 0.35 and (oB < 0.5 and oC < 0.5): return A

    return (random.randint(margin, IMG_SIZE-1-margin), random.randint(margin, IMG_SIZE-1-margin))

def maybe_rotate(A, B, C):
    if random.random() > ROT90_PROB:
        return A, B, C
    k = random.randint(0, 3)
    if k == 0:
        return A, B, C
    A2 = rot90_xy(A[0], A[1], k)
    B2 = rot90_xy(B[0], B[1], k)
    C2 = rot90_xy(C[0], C[1], k)
    return (A2[0], A2[1]), (B2[0], B2[1]), (C2[0], C2[1])

def canonicalize_BC(B, C):
    # Lexicographic canonicalization so "B→C" direction is consistent
    # (fixes left/right ambiguity across rotations)
    if (B[0] > C[0]) or (B[0] == C[0] and B[1] > C[1]):
        return C, B
    return B, C

def lr_sign(A, B, C):
    # sign of cross(BC, BA): positive => A is "left" of directed B→C (screen coords)
    bx, by = B
    cx, cy = C
    ax, ay = A
    v1x, v1y = (cx - bx, cy - by)
    v2x, v2y = (ax - bx, ay - by)
    cross = v1x * v2y - v1y * v2x
    # Map to binary: 0 = left, 1 = right (you can flip if you want)
    return 1.0 if cross < 0 else 0.0

# -----------------------
# Dataset
# -----------------------
class TernaryGeomDataset(Dataset):
    def __init__(self, n, seed=123):
        super().__init__()
        self.n = n
        self.seed = seed

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        random.seed(idx * 1000003 + 17)

        bucket = random.choices(BUCKETS, weights=BUCKET_PROBS, k=1)[0]
        B, C = sample_BC()
        A = sample_A_for_bucket(bucket, B, C)
        A, B, C = maybe_rotate(A, B, C)

        # Phase 7: canonicalize AFTER rotation so axis is stable globally
        B, C = canonicalize_BC(B, C)

        bscore, tproj = between_score(A, B, C)
        oB = overlap_flag(A, B, R_A, R_B)
        oC = overlap_flag(A, C, R_A, R_C)
        oAny = 1.0 if (oB > 0.5 or oC > 0.5) else 0.0

        lrs = lr_sign(A, B, C)

        x = draw_scene(A, B, C)
        x = torch.tensor(x, dtype=torch.float32)

        y = {
            "between_score": torch.tensor([bscore], dtype=torch.float32),
            "t_on_BC":       torch.tensor([tproj], dtype=torch.float32),
            "overlap_any":   torch.tensor([oAny], dtype=torch.float32),
            "lr_sign":       torch.tensor([lrs], dtype=torch.float32),
        }
        return x, y

# -----------------------
# Model (same AE backbone; redefine heads for phase7 targets)
# -----------------------
class ConvAEHeads(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(inplace=True)
        )
        self.fc_mu = nn.Linear(256*4*4, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, 256*4*4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid()
        )

        def mlp(out_dim):
            return nn.Sequential(
                nn.Linear(latent_dim, 256), nn.ReLU(inplace=True),
                nn.Linear(256, 128), nn.ReLU(inplace=True),
                nn.Linear(128, out_dim)
            )

        self.h_between = mlp(1)    # regression (sigmoid -> 0..1)
        self.h_tproj   = mlp(1)    # regression (sigmoid -> 0..1)
        self.h_overlap = mlp(1)    # logit
        self.h_lr      = mlp(1)    # logit

    def encode(self, x):
        h = self.enc(x)
        h = h.reshape(h.size(0), -1)
        return self.fc_mu(h)

    def decode(self, z):
        h = self.fc_dec(z).view(z.size(0), 256, 4, 4)
        return self.dec(h)

    def forward(self, x):
        z = self.encode(x)
        xhat = self.decode(z)
        return {
            "z": z,
            "xhat": xhat,
            "between": self.h_between(z),
            "tproj":   self.h_tproj(z),
            "overlap": self.h_overlap(z),
            "lr":      self.h_lr(z),
        }

# -----------------------
# Train
# -----------------------
def train():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"[train64-ternary-phase7-256] Using device: {DEVICE}")
    print("[train64-ternary-phase7-256] Targets: between_score, t_on_BC, overlap_any, lr_sign")
    print(f"[train64-ternary-phase7-256] Bucket probs: {dict(zip(BUCKETS, BUCKET_PROBS))}")
    print(f"[train64-ternary-phase7-256] Rotation90 prob = {ROT90_PROB} | BC canonicalization = ON")

    ds_tr = TernaryGeomDataset(N_TRAIN, seed=123)
    ds_va = TernaryGeomDataset(N_VAL, seed=456)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True, num_workers=0, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH, shuffle=False, num_workers=0)

    model = ConvAEHeads(LATENT_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    def step(batch, train_mode=True):
        x, y = batch
        x = x.to(DEVICE)

        yb = y["between_score"].to(DEVICE)   # (B,1)
        yt = y["t_on_BC"].to(DEVICE)         # (B,1)
        yo = y["overlap_any"].to(DEVICE)     # (B,1)
        yl = y["lr_sign"].to(DEVICE)         # (B,1)

        if train_mode:
            model.train()
            opt.zero_grad(set_to_none=True)
        else:
            model.eval()

        out = model(x)

        loss_recon = mse(out["xhat"], x)
        loss_between = mse(torch.sigmoid(out["between"]), yb)
        loss_tproj   = mse(torch.sigmoid(out["tproj"]), yt)
        loss_overlap = bce(out["overlap"], yo)

        # Mask lr loss to ONLY non-between examples
        bmask = (yb.squeeze(1) < BETWEEN_MASK_THR)
        if bmask.any():
            loss_lr = bce(out["lr"][bmask], yl[bmask])
        else:
            loss_lr = torch.tensor(0.0, device=DEVICE)

        loss = (
            W_RECON   * loss_recon +
            W_BETW    * loss_between +
            W_TPROJ   * loss_tproj +
            W_OVERLAP * loss_overlap +
            W_LR      * loss_lr
        )

        if train_mode:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()

        return float(loss.item())

    best_val = float("inf")
    for ep in range(1, EPOCHS+1):
        tr_losses = [step(batch, True) for batch in dl_tr]
        with torch.no_grad():
            va_losses = [step(batch, False) for batch in dl_va]

        tr = float(np.mean(tr_losses))
        va = float(np.mean(va_losses))
        print(f"[train64-ternary-phase7-256] Epoch {ep:3d}/{EPOCHS} | train={tr:.4f} | val={va:.4f}")

        if va < best_val:
            best_val = va
            torch.save({
                "model_state": model.state_dict(),
                "latent_dim": LATENT_DIM,
                "img_size": IMG_SIZE,
                "phase": 7,
            }, CKPT)

    print(f"[train64-ternary-phase7-256] Saved -> {CKPT}")

if __name__ == "__main__":
    set_seed(123)
    train()
