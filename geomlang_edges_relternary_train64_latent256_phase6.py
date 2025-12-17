#!/usr/bin/env python3
# geomlang_edges_relternary_train64_latent256_phase6.py
#
# Phase 6: Factorized ternary geometry with balanced sampling.
# - Inputs: 64x64 RGB+EDGE style (we use 3 channels: red fill, blue fill, edges)
# - Model: conv autoencoder + multi-head predictors
# - Targets (factorized):
#     between_score   (0..1 continuous)
#     t_on_BC         (0..1 continuous, projection of A onto segment BC)
#     closer_sign     (0/1: 1 means closer to C, 0 means closer to B)
#     closer_mag      (0..1 continuous)
#     overlap_B       (0/1)
#     overlap_C       (0/1)
#
# Key: We DO NOT force between to be mutually exclusive with overlap.
#
# Outputs:
#   outputs_edges_relternary256_phase6/scene_model_edges_relternary256_phase6.pt

import os, math, random, json
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
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

OUTDIR = "outputs_edges_relternary256_phase6"
CKPT   = os.path.join(OUTDIR, "scene_model_edges_relternary256_phase6.pt")

# Balanced buckets (to ensure between isn't rare)
# We sample scenes into one of these buckets by construction:
#   - "between_clear": between_score high AND no overlap
#   - "between_overlap": between_score high AND overlap_B or overlap_C
#   - "overlap_only": overlap_B or overlap_C but between_score low
#   - "non_between": between_score low and no overlap
BUCKETS = ("between_clear", "between_overlap", "overlap_only", "non_between")
BUCKET_PROBS = (0.22, 0.22, 0.28, 0.28)

# Geometry params
R_A = 6
R_B = 7
R_C = 7

# "Between" scoring parameters
SIGMA_PERP     = 2.0   # pixels
SIGMA_OUTSIDE  = 0.15  # fraction of segment length
END_MARGIN_T   = 0.08  # keep A away from endpoints for "clear" between

# Overlap
OVERLAP_TOL = 3.0  # pixels (distance between centers less than (rA+rB-OVERLAP_TOL) => overlap)

# Closer
CLOSER_SIGMA = 8.0  # pixels

ROT90_PROB = 1.0

# Loss weights (tune if needed)
W_RECON   = 1.0
W_BETW   = 1.0
W_TPROJ  = 0.6
W_CSIGN  = 0.6
W_CMAG   = 0.6
W_OB     = 0.6
W_OC     = 0.6

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
    # rotate around image center (IMG_SIZE/2, IMG_SIZE/2) by 90*n degrees
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
    # projection of A onto line BC (infinite), t in R (0 at B, 1 at C)
    bx, by = B
    cx, cy = C
    ax, ay = A
    vx, vy = (cx - bx, cy - by)
    wx, wy = (ax - bx, ay - by)
    vv = vx*vx + vy*vy + 1e-9
    t = (wx*vx + wy*vy) / vv
    # perpendicular distance to line
    # distance from point to line via area formula:
    # perp = |v x w| / |v|
    cross = abs(vx*wy - vy*wx)
    vnorm = math.sqrt(vv)
    perp = cross / (vnorm + 1e-9)
    seg_len = vnorm
    return t, perp, seg_len

def between_score(A, B, C):
    t, perp, seg_len = proj_t_and_perp(A, B, C)

    # penalty for being outside the segment (t<0 or t>1), scaled by segment length
    outside = 0.0
    if t < 0.0:
        outside = -t
    elif t > 1.0:
        outside = (t - 1.0)

    # Convert perp distance into [0..1]
    s_perp = math.exp(-(perp**2) / (2.0 * (SIGMA_PERP**2)))

    # Outside factor: outside is in "t units", scale by SIGMA_OUTSIDE (~fraction)
    s_out = math.exp(-(outside**2) / (2.0 * (SIGMA_OUTSIDE**2)))

    s = s_perp * s_out
    return float(s), float(clamp(t, 0.0, 1.0))

def overlap_flag(A, B, rA, rB):
    d = dist(A, B)
    return 1.0 if d <= (rA + rB - OVERLAP_TOL) else 0.0

def closer_targets(A, B, C):
    dB = dist(A, B)
    dC = dist(A, C)
    # sign: 0 => closer to B, 1 => closer to C
    sign = 1.0 if dC < dB else 0.0
    # magnitude: squash distance diff to [0..1] (1 means strongly closer)
    diff = abs(dB - dC)
    mag = 1.0 - math.exp(-(diff**2) / (2.0 * (CLOSER_SIGMA**2)))
    return float(sign), float(mag)

def draw_scene(A, B, C):
    # Channels:
    # 0 red fill for A
    # 1 blue fill for B & C (both blue)
    # 2 edges for A,B,C
    imgR = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
    imgB = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
    imgE = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)

    dR = ImageDraw.Draw(imgR)
    dB = ImageDraw.Draw(imgB)
    dE = ImageDraw.Draw(imgE)

    def circle_bbox(p, r):
        return [p[0]-r, p[1]-r, p[0]+r, p[1]+r]

    # fills
    dR.ellipse(circle_bbox(A, R_A), fill=255)
    dB.ellipse(circle_bbox(B, R_B), fill=255)
    dB.ellipse(circle_bbox(C, R_C), fill=255)

    # edges
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
    # sample B,C not too close, not too close to border
    margin = 12
    while True:
        B = (random.randint(margin, IMG_SIZE-1-margin), random.randint(margin, IMG_SIZE-1-margin))
        C = (random.randint(margin, IMG_SIZE-1-margin), random.randint(margin, IMG_SIZE-1-margin))
        if dist(B, C) >= 18:
            return B, C

def sample_A_for_bucket(bucket, B, C):
    # Construct A according to bucket; retry until constraints satisfied.
    margin = 10
    for _ in range(2000):
        # start with a t along segment and perpendicular offset
        t0 = random.random()
        bx, by = B
        cx, cy = C
        vx, vy = (cx - bx, cy - by)
        # unit normal
        vnorm = math.sqrt(vx*vx + vy*vy) + 1e-9
        nx, ny = (-vy / vnorm, vx / vnorm)

        # choose base point on segment
        px = bx + t0 * vx
        py = by + t0 * vy

        if bucket in ("between_clear", "between_overlap"):
            # keep away from endpoints
            t0 = random.uniform(END_MARGIN_T, 1.0 - END_MARGIN_T)
            px = bx + t0 * vx
            py = by + t0 * vy
            # small perp
            perp = random.uniform(-2.0, 2.0)
        else:
            # larger perp or outside segment
            if random.random() < 0.55:
                perp = random.uniform(6.0, 18.0) * (1 if random.random() < 0.5 else -1)
            else:
                # outside segment (t outside [0,1])
                t0 = random.choice([random.uniform(-0.6, -0.05), random.uniform(1.05, 1.6)])
                px = bx + t0 * vx
                py = by + t0 * vy
                perp = random.uniform(-6.0, 6.0)

        ax = px + perp * nx
        ay = py + perp * ny

        # clamp to image area
        ax = clamp(ax, margin, IMG_SIZE-1-margin)
        ay = clamp(ay, margin, IMG_SIZE-1-margin)
        A = (ax, ay)

        # compute factors
        s_between, t_clamped = between_score(A, B, C)
        oB = overlap_flag(A, B, R_A, R_B)
        oC = overlap_flag(A, C, R_A, R_C)

        if bucket == "between_clear":
            if s_between > 0.70 and (oB < 0.5 and oC < 0.5):
                return A
        elif bucket == "between_overlap":
            if s_between > 0.70 and (oB > 0.5 or oC > 0.5):
                return A
        elif bucket == "overlap_only":
            if s_between < 0.35 and (oB > 0.5 or oC > 0.5):
                return A
        elif bucket == "non_between":
            if s_between < 0.35 and (oB < 0.5 and oC < 0.5):
                return A

    # fallback: unconstrained random
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

# -----------------------
# Dataset
# -----------------------
class TernaryGeomDataset(Dataset):
    def __init__(self, n, seed=123):
        super().__init__()
        self.n = n
        self.rng = random.Random(seed)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Use global random but make deterministic per idx
        random.seed(idx * 1000003 + 17)

        # choose bucket
        bucket = random.choices(BUCKETS, weights=BUCKET_PROBS, k=1)[0]

        B, C = sample_BC()
        A = sample_A_for_bucket(bucket, B, C)
        A, B, C = maybe_rotate(A, B, C)

        # factors
        bscore, tproj = between_score(A, B, C)
        csign, cmag = closer_targets(A, B, C)
        oB = overlap_flag(A, B, R_A, R_B)
        oC = overlap_flag(A, C, R_A, R_C)

        x = draw_scene(A, B, C)
        x = torch.tensor(x, dtype=torch.float32)

        y = {
            "between_score": torch.tensor([bscore], dtype=torch.float32),
            "t_on_BC":       torch.tensor([tproj], dtype=torch.float32),
            "closer_sign":   torch.tensor([csign], dtype=torch.float32),  # BCE target
            "closer_mag":    torch.tensor([cmag], dtype=torch.float32),
            "overlap_B":     torch.tensor([oB], dtype=torch.float32),
            "overlap_C":     torch.tensor([oC], dtype=torch.float32),
        }
        return x, y

# -----------------------
# Model
# -----------------------
class ConvAEHeads(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(inplace=True),   # 32x32
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(inplace=True),  # 16x16
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(inplace=True), # 8x8
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(inplace=True) # 4x4
        )
        self.fc_mu = nn.Linear(256*4*4, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, 256*4*4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(inplace=True), # 8x8
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True),  # 16x16
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(inplace=True),   # 32x32
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid()              # 64x64
        )

        # Heads
        def mlp(out_dim):
            return nn.Sequential(
                nn.Linear(latent_dim, 256), nn.ReLU(inplace=True),
                nn.Linear(256, 128), nn.ReLU(inplace=True),
                nn.Linear(128, out_dim)
            )

        self.h_between = mlp(1)   # regression
        self.h_tproj   = mlp(1)   # regression
        self.h_csign   = mlp(1)   # logit (BCE)
        self.h_cmag    = mlp(1)   # regression
        self.h_oB      = mlp(1)   # logit
        self.h_oC      = mlp(1)   # logit

    def encode(self, x):
        h = self.enc(x)
        h = h.reshape(h.size(0), -1)
        z = self.fc_mu(h)
        return z

    def decode(self, z):
        h = self.fc_dec(z).view(z.size(0), 256, 4, 4)
        xhat = self.dec(h)
        return xhat

    def forward(self, x):
        z = self.encode(x)
        xhat = self.decode(z)
        out = {
            "z": z,
            "xhat": xhat,
            "between": self.h_between(z),
            "tproj":   self.h_tproj(z),
            "csign":   self.h_csign(z),
            "cmag":    self.h_cmag(z),
            "oB":      self.h_oB(z),
            "oC":      self.h_oC(z),
        }
        return out

# -----------------------
# Train
# -----------------------
def train():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"[train64-ternary-phase6-256] Using device: {DEVICE}")
    print("[train64-ternary-phase6-256] Targets: between_score, t_on_BC, closer_sign, closer_mag, overlap_B, overlap_C")
    print(f"[train64-ternary-phase6-256] Params: SIGMA_PERP={SIGMA_PERP}, SIGMA_OUTSIDE={SIGMA_OUTSIDE}, CLOSER_SIGMA={CLOSER_SIGMA}, OVERLAP_TOL={OVERLAP_TOL}")
    print(f"[train64-ternary-phase6-256] Bucket probs: {dict(zip(BUCKETS, BUCKET_PROBS))}")
    print(f"[train64-ternary-phase6-256] Rotation90 prob = {ROT90_PROB}")

    ds_tr = TernaryGeomDataset(N_TRAIN, seed=123)
    ds_va = TernaryGeomDataset(N_VAL, seed=456)
    dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True, num_workers=0, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH, shuffle=False, num_workers=0)

    model = ConvAEHeads(LATENT_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    l1  = nn.L1Loss()

    def step(batch, train_mode=True):
        x, y = batch
        x = x.to(DEVICE)
        yb = y["between_score"].to(DEVICE)
        yt = y["t_on_BC"].to(DEVICE)
        ys = y["closer_sign"].to(DEVICE)
        ym = y["closer_mag"].to(DEVICE)
        yoB = y["overlap_B"].to(DEVICE)
        yoC = y["overlap_C"].to(DEVICE)

        if train_mode:
            model.train()
            opt.zero_grad(set_to_none=True)
        else:
            model.eval()

        out = model(x)

        loss_recon = mse(out["xhat"], x)
        loss_between = mse(torch.sigmoid(out["between"]), yb)   # clamp to 0..1 via sigmoid
        loss_tproj   = mse(torch.sigmoid(out["tproj"]), yt)     # 0..1 via sigmoid
        loss_csign   = bce(out["csign"], ys)
        loss_cmag    = l1(torch.sigmoid(out["cmag"]), ym)       # 0..1 via sigmoid
        loss_oB      = bce(out["oB"], yoB)
        loss_oC      = bce(out["oC"], yoC)

        loss = (
            W_RECON  * loss_recon +
            W_BETW   * loss_between +
            W_TPROJ  * loss_tproj +
            W_CSIGN  * loss_csign +
            W_CMAG   * loss_cmag +
            W_OB     * loss_oB +
            W_OC     * loss_oC
        )

        if train_mode:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()

        return float(loss.item())

    best_val = float("inf")
    for ep in range(1, EPOCHS+1):
        tr_losses = []
        for batch in dl_tr:
            tr_losses.append(step(batch, train_mode=True))
        va_losses = []
        with torch.no_grad():
            for batch in dl_va:
                va_losses.append(step(batch, train_mode=False))

        tr = float(np.mean(tr_losses))
        va = float(np.mean(va_losses))
        print(f"[train64-ternary-phase6-256] Epoch {ep:3d}/{EPOCHS} | train={tr:.4f} | val={va:.4f}")

        if va < best_val:
            best_val = va
            torch.save({
                "model_state": model.state_dict(),
                "latent_dim": LATENT_DIM,
                "img_size": IMG_SIZE,
                "phase": 6,
            }, CKPT)

    print(f"[train64-ternary-phase6-256] Saved -> {CKPT}")

if __name__ == "__main__":
    set_seed(123)
    train()
