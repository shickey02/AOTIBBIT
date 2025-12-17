#!/usr/bin/env python3
# geomlang_export_latents_ternary_phase7_256.py
#
# Exports:
#   - Z (N,256)
#   - targets dict -> npz (between_score, t_on_BC, overlap_any, lr_sign)
#   - preds dict   -> npz (between_pred, tproj_pred, overlap_logit, lr_logit)
#
# Assumes training script saved:
#   outputs_edges_relternary256_phase7/scene_model_edges_relternary256_phase7.pt

import os, math, random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------
# Config
# -----------------------
IMG_SIZE   = 64
LATENT_DIM = 256
N_EVAL     = 6000
SEED       = 123
BATCH      = 256

OUTDIR = "outputs_edges_relternary256_phase7"
CKPT   = os.path.join(OUTDIR, "scene_model_edges_relternary256_phase7.pt")

Z_OUT   = os.path.join(OUTDIR, f"encoded_latents_seed{SEED}_N{N_EVAL}.npy")
T_OUT   = os.path.join(OUTDIR, f"encoded_targets_seed{SEED}_N{N_EVAL}.npz")
P_OUT   = os.path.join(OUTDIR, f"encoded_preds_seed{SEED}_N{N_EVAL}.npz")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# You MUST keep these identical to Phase 7 train script
# -----------------------
import math
from PIL import Image, ImageDraw

R_A = 6
R_B = 7
R_C = 7
SIGMA_PERP     = 2.0
SIGMA_OUTSIDE  = 0.15
END_MARGIN_T   = 0.08
OVERLAP_TOL    = 3.0
ROT90_PROB     = 1.0

BUCKETS = ("between_clear", "between_overlap", "overlap_only", "non_between")
BUCKET_PROBS = (0.22, 0.22, 0.28, 0.28)

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
    return t, perp

def between_score(A, B, C):
    t, perp = proj_t_and_perp(A, B, C)
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
    if (B[0] > C[0]) or (B[0] == C[0] and B[1] > C[1]):
        return C, B
    return B, C

def lr_sign(A, B, C):
    bx, by = B
    cx, cy = C
    ax, ay = A
    v1x, v1y = (cx - bx, cy - by)
    v2x, v2y = (ax - bx, ay - by)
    cross = v1x * v2y - v1y * v2x
    return 1.0 if cross < 0 else 0.0

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
# Model (must match Phase 7 train)
# -----------------------
class ConvAEHeads(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(256*4*4, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, 256*4*4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
        )

        def mlp(out_dim):
            return nn.Sequential(
                nn.Linear(latent_dim, 256), nn.ReLU(inplace=True),
                nn.Linear(256, 128), nn.ReLU(inplace=True),
                nn.Linear(128, out_dim),
            )

        self.h_between = mlp(1)  # reg -> sigmoid
        self.h_tproj   = mlp(1)  # reg -> sigmoid
        self.h_overlap = mlp(1)  # logit
        self.h_lr      = mlp(1)  # logit

    def encode(self, x):
        h = self.enc(x)
        h = h.reshape(h.size(0), -1)
        return self.fc_mu(h)

    def forward(self, x):
        z = self.encode(x)
        # decode not needed for export
        return {
            "z": z,
            "between": self.h_between(z),
            "tproj":   self.h_tproj(z),
            "overlap": self.h_overlap(z),
            "lr":      self.h_lr(z),
        }

def main():
    print(f"[export-ternary-phase7-256] device = {DEVICE}")
    print(f"[export-ternary-phase7-256] loading ckpt: {CKPT}")

    ck = torch.load(CKPT, map_location=DEVICE)
    model = ConvAEHeads(LATENT_DIM).to(DEVICE)
    model.load_state_dict(ck["model_state"])
    model.eval()

    ds = TernaryGeomDataset(N_EVAL, seed=SEED)
    eval_loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0)

    Z = []
    tb = []
    tt = []
    to = []
    tl = []

    pb = []
    pt = []
    po = []
    pl = []

    with torch.no_grad():
        for x, y in eval_loader:
            x = x.to(DEVICE)
            out = model(x)

            z = out["z"].detach().cpu().numpy()
            Z.append(z)

            tb.append(y["between_score"].numpy())
            tt.append(y["t_on_BC"].numpy())
            to.append(y["overlap_any"].numpy())
            tl.append(y["lr_sign"].numpy())

            pb.append(torch.sigmoid(out["between"]).cpu().numpy())
            pt.append(torch.sigmoid(out["tproj"]).cpu().numpy())
            po.append(out["overlap"].cpu().numpy())  # logits
            pl.append(out["lr"].cpu().numpy())       # logits

    Z = np.concatenate(Z, axis=0)
    tb = np.concatenate(tb, axis=0)
    tt = np.concatenate(tt, axis=0)
    to = np.concatenate(to, axis=0)
    tl = np.concatenate(tl, axis=0)

    pb = np.concatenate(pb, axis=0)
    pt = np.concatenate(pt, axis=0)
    po = np.concatenate(po, axis=0)
    pl = np.concatenate(pl, axis=0)

    os.makedirs(OUTDIR, exist_ok=True)
    np.save(Z_OUT, Z)
    np.savez(T_OUT,
             between_score=tb.squeeze(1),
             t_on_BC=tt.squeeze(1),
             overlap_any=to.squeeze(1),
             lr_sign=tl.squeeze(1))
    np.savez(P_OUT,
             between_pred=pb.squeeze(1),
             tproj_pred=pt.squeeze(1),
             overlap_logit=po.squeeze(1),
             lr_logit=pl.squeeze(1))

    print(f"[export-ternary-phase7-256] Z shape = {Z.shape}")
    print(f"[export-ternary-phase7-256] saved: {Z_OUT}")
    print(f"[export-ternary-phase7-256] saved: {T_OUT}")
    print(f"[export-ternary-phase7-256] saved: {P_OUT}")

if __name__ == "__main__":
    main()
