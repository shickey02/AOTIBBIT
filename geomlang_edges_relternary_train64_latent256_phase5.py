#!/usr/bin/env python3
# geomlang_edges_relternary_train64_latent256_phase5.py
#
# Phase 5 train:
# Factorized heads (continuous + bits) AND a derived "display label" that does NOT let overlap steal between.
#
# Targets:
#   - between_score  : [0..1] "how between" (line/segment closeness) independent of overlap
#   - t_on_BC        : [0..1] projection parameter along segment BC (clamped)
#   - closer_sign    : 0 if closer to B else 1 (closer to C)
#   - closer_mag     : [0..1] how much closer (normalized)
#   - overlap_B      : {0,1}
#   - overlap_C      : {0,1}
#
# Derived display labels (8-way):
#   0 A_left_of_BtoC
#   1 A_right_of_BtoC
#   2 A_between_clear      (between AND NOT overlap_any)
#   3 A_closer_to_B
#   4 A_closer_to_C
#   5 A_overlap_B          (overlap_B AND NOT between)
#   6 A_overlap_C          (overlap_C AND NOT between)
#   7 A_between_overlap    (between AND overlap_any)

import os, json, math, random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------
# Config
# -----------------------
TAG = "[train64-ternary-phase5-256]"

IMG_SIZE   = 64
NUM_CH     = 3            # red fill, blue fill, edges
LATENT_DIM = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR   = "outputs_edges_relternary256_phase5"
CKPT_PATH = os.path.join(OUT_DIR, "scene_model_edges_relternary256_phase5.pt")

# factorized params (same spirit as phase4, tweak freely)
SIGMA_PERP     = 2.0
SIGMA_OUTSIDE  = 0.15      # in t-space, penalty for being off-segment
CLOSER_SIGMA   = 8.0       # pixels-ish scale for closeness magnitude shaping
OVERLAP_TOL    = 3.0       # overlap threshold fudge

END_MARGIN     = 0.10      # in t-space for "between" validity window
BETWEEN_THRESH = 0.55      # threshold on between_score to call "between" for display label
CLOSER_THRESH  = 0.15      # threshold on closer_mag to decide closer-vs-left/right fallback

ROT90_PROB = 1.0

N_TRAIN   = 12000
N_VAL     = 2000
SEED_TRAIN = 123
SEED_VAL   = 456

BATCH_SIZE = 256
EPOCHS     = 40
LR         = 2e-3

# A-scan anchors (same style as your prior phases)
SCAN_CONFIGS = {
    "horiz_mid":   ((18, 32), (46, 32)),
    "vert_mid":    ((32, 18), (32, 46)),
    "diag_mid":    ((20, 20), (44, 44)),
    "horiz_short": ((23, 32), (41, 32)),
    "horiz_off":   ((18, 22), (46, 40)),
}
SCAN_T_MIN = -0.35
SCAN_T_MAX = 1.35

# -----------------------
# Geometry helpers
# -----------------------
def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def proj_t_and_perp(A, B, C):
    """Return (t_clamped, perp_dist_pixels, outside_t_dist)."""
    Ax, Ay = A; Bx, By = B; Cx, Cy = C
    vx, vy = (Cx - Bx), (Cy - By)
    wx, wy = (Ax - Bx), (Ay - By)
    vv = vx*vx + vy*vy + 1e-9
    t = (wx*vx + wy*vy) / vv

    # perp distance to infinite line
    # distance = |cross(v, w)| / |v|
    cross = vx*wy - vy*wx
    vlen = math.sqrt(vv)
    d_perp = abs(cross) / (vlen + 1e-9)

    # outside measure in t-space
    if t < 0.0:
        outside = -t
    elif t > 1.0:
        outside = t - 1.0
    else:
        outside = 0.0

    return clamp01(t), float(d_perp), float(outside), float(t), float(cross)

def between_score(A, B, C, sigma_perp=SIGMA_PERP, sigma_out=SIGMA_OUTSIDE):
    t_clamp, d_perp, outside, t_raw, cross = proj_t_and_perp(A, B, C)
    s_perp = math.exp(- (d_perp / (sigma_perp + 1e-9))**2)
    s_out  = math.exp(- (outside / (sigma_out + 1e-9))**2)
    return float(s_perp * s_out), t_clamp, t_raw, cross

def overlap_flag(A, sizeA, B, sizeB, tol=OVERLAP_TOL):
    """Cheap overlap for disk-ish shapes: centers within (rA+rB-tol)."""
    Ax, Ay = A; Bx, By = B
    d = math.sqrt((Ax-Bx)**2 + (Ay-By)**2)
    return 1 if d <= (sizeA + sizeB - tol) else 0

def overlap_flag_from_masks(Ax, Ay, shapeA, sizeA, Bx, By, shapeB, sizeB):
    tmp = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    if shapeA == 0:
        mA = draw_square(tmp, Ax, Ay, sizeA)
    else:
        mA = draw_circle(tmp, Ax, Ay, sizeA)

    if shapeB == 0:
        mB = draw_square(tmp, Bx, By, sizeB)
    else:
        mB = draw_circle(tmp, Bx, By, sizeB)

    return 1 if np.any(mA & mB) else 0


def closer_targets(A, B, C, closer_sigma=CLOSER_SIGMA):
    Ax, Ay = A; Bx, By = B; Cx, Cy = C
    dB = math.sqrt((Ax-Bx)**2 + (Ay-By)**2)
    dC = math.sqrt((Ax-Cx)**2 + (Ay-Cy)**2)
    sign = 0 if dB <= dC else 1
    # magnitude: squashed difference
    diff = abs(dB - dC)
    mag = 1.0 - math.exp(-diff / (closer_sigma + 1e-9))
    return int(sign), float(clamp01(mag))

def maybe_rot90_triplet(A, B, C, p=1.0):
    """Rotate whole scene by 90deg about image center with prob p (keeps distribution isotropic)."""
    if random.random() > p:
        return A, B, C
    cx = (IMG_SIZE - 1) / 2.0
    cy = (IMG_SIZE - 1) / 2.0

    def rot(pt):
        x, y = pt
        # translate
        x0 = x - cx
        y0 = y - cy
        # 90deg: (x,y)->(-y,x)
        xr = -y0
        yr =  x0
        return (int(round(xr + cx)), int(round(yr + cy)))

    return rot(A), rot(B), rot(C)

# -----------------------
# Rendering (fills + edges)
# -----------------------
def draw_circle(mask, cx, cy, r):
    H, W = mask.shape
    y, x = np.ogrid[:H, :W]
    return ((x - cx)**2 + (y - cy)**2) <= (r*r)

def draw_square(mask, cx, cy, r):
    H, W = mask.shape
    x0 = max(0, cx - r); x1 = min(W, cx + r + 1)
    y0 = max(0, cy - r); y1 = min(H, cy + r + 1)
    m = np.zeros_like(mask, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m

def edges_from_mask(m):
    # simple 4-neighbor outline
    up    = np.pad(m[:-1, :], ((1,0),(0,0)), constant_values=False)
    down  = np.pad(m[1:, :],  ((0,1),(0,0)), constant_values=False)
    left  = np.pad(m[:, :-1], ((0,0),(1,0)), constant_values=False)
    right = np.pad(m[:, 1:],  ((0,0),(0,1)), constant_values=False)
    erode = m & up & down & left & right
    edge  = m & (~erode)
    return edge

def render_scene_ABC(Ax, Ay, Bx, By, Cx, Cy, shapeA, shapeB, shapeC, sizeA, sizeB, sizeC):
    H = W = IMG_SIZE
    red  = np.zeros((H, W), dtype=np.float32)
    blue = np.zeros((H, W), dtype=np.float32)

    # A is red, B is blue, C is blue (same channel) — matches your earlier 2-shape convention
    if shapeA == 0:
        mA = draw_square(red, Ax, Ay, sizeA)
    else:
        mA = draw_circle(red, Ax, Ay, sizeA)

    if shapeB == 0:
        mB = draw_square(blue, Bx, By, sizeB)
    else:
        mB = draw_circle(blue, Bx, By, sizeB)

    if shapeC == 0:
        mC = draw_square(blue, Cx, Cy, sizeC)
    else:
        mC = draw_circle(blue, Cx, Cy, sizeC)

    red[mA] = 1.0
    blue[mB] = 1.0
    blue[mC] = 1.0

    eA = edges_from_mask(mA)
    eB = edges_from_mask(mB)
    eC = edges_from_mask(mC)
    edges = (eA | eB | eC).astype(np.float32)

    img = np.stack([red, blue, edges], axis=0)  # CHW
    return img

# -----------------------
# Derived display label (Phase 5 fix)
# -----------------------
REL5_NAMES = [
    "A_left_of_BtoC",
    "A_right_of_BtoC",
    "A_between_clear",
    "A_closer_to_B",
    "A_closer_to_C",
    "A_overlap_B",
    "A_overlap_C",
    "A_between_overlap",
]

def derived_label_from_factors(side_cross, between_s, t_clamp, closer_sign, closer_mag, oB, oC):
    overlap_any = 1 if (oB or oC) else 0
    between_ok = (between_s >= BETWEEN_THRESH) and (t_clamp >= END_MARGIN) and (t_clamp <= (1.0 - END_MARGIN))

    # ---- critical: between is decided BEFORE overlap, but overlap is still preserved in a separate between_overlap class
    if between_ok:
        return 7 if overlap_any else 2  # between_overlap vs between_clear

    # not between:
    if oB:
        return 5
    if oC:
        return 6

    # closer if strong enough; else left/right by side
    if closer_mag >= CLOSER_THRESH:
        return 3 if closer_sign == 0 else 4

    return 0 if side_cross > 0 else 1

# -----------------------
# Dataset
# -----------------------
class GeomEdgesTernary64DatasetPhase5(Dataset):
    def __init__(self, n, seed=0):
        self.n = int(n)
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        H = W = IMG_SIZE

        # shapes (0=square, 1=circle) — keep simple / stable
        shapeA = int(self.rng.randint(0, 2))
        shapeB = int(self.rng.randint(0, 2))
        shapeC = int(self.rng.randint(0, 2))

        # sizes
        sizeA = int(self.rng.randint(8, 13))
        sizeB = int(self.rng.randint(8, 13))
        sizeC = int(self.rng.randint(8, 13))

        # centers with margins
        margin = 6
        def sample_center(r):
            x = int(self.rng.randint(margin + r, W - margin - r))
            y = int(self.rng.randint(margin + r, H - margin - r))
            return (x, y)

        A = sample_center(sizeA)
        B = sample_center(sizeB)
        C = sample_center(sizeC)

        # rotate 90 with prob
        # (we need python random for p; sync by using idx-seeded if desired)
        # We'll just do deterministic-ish by using rng draw:
        if self.rng.rand() < ROT90_PROB:
            # implement same 90deg about center
            cx = (IMG_SIZE - 1) / 2.0
            cy = (IMG_SIZE - 1) / 2.0
            def rot(pt):
                x, y = pt
                x0 = x - cx; y0 = y - cy
                xr = -y0; yr = x0
                return (int(round(xr + cx)), int(round(yr + cy)))
            A, B, C = rot(A), rot(B), rot(C)

        # factor targets
        bscore, t_clamp, t_raw, cross = between_score(A, B, C)
        csign, cmag = closer_targets(A, B, C)
        oB = overlap_flag_from_masks(A[0], A[1], shapeA, sizeA, B[0], B[1], shapeB, sizeB)
        oC = overlap_flag_from_masks(A[0], A[1], shapeA, sizeA, C[0], C[1], shapeC, sizeC)


        # derived display label (for debug/optional classification reporting)
        y_disp = derived_label_from_factors(cross, bscore, t_clamp, csign, cmag, oB, oC)

        img = render_scene_ABC(
            A[0], A[1], B[0], B[1], C[0], C[1],
            shapeA, shapeB, shapeC, sizeA, sizeB, sizeC
        )

        # return:
        #   img: float32 CHW
        #   targets: float32 tensor for continuous + bits
        targets = {
            "between_score": np.float32(bscore),
            "t_on_BC":       np.float32(t_clamp),
            "closer_sign":   np.int64(csign),
            "closer_mag":    np.float32(cmag),
            "overlap_B":     np.int64(oB),
            "overlap_C":     np.int64(oC),
            "disp_label":    np.int64(y_disp),
        }

        x = torch.from_numpy(img).float()
        return x, targets

def collate_phase5(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)

    # pack targets
    between = torch.tensor([b[1]["between_score"] for b in batch], dtype=torch.float32)
    tproj   = torch.tensor([b[1]["t_on_BC"]       for b in batch], dtype=torch.float32)
    csign   = torch.tensor([b[1]["closer_sign"]   for b in batch], dtype=torch.long)
    cmag    = torch.tensor([b[1]["closer_mag"]    for b in batch], dtype=torch.float32)
    oB      = torch.tensor([b[1]["overlap_B"]     for b in batch], dtype=torch.long)
    oC      = torch.tensor([b[1]["overlap_C"]     for b in batch], dtype=torch.long)
    ydisp   = torch.tensor([b[1]["disp_label"]    for b in batch], dtype=torch.long)

    tdict = {
        "between_score": between,
        "t_on_BC": tproj,
        "closer_sign": csign,
        "closer_mag": cmag,
        "overlap_B": oB,
        "overlap_C": oC,
        "disp_label": ydisp,
    }
    return xs, tdict

# -----------------------
# Model
# -----------------------
class Encoder(nn.Module):
    def __init__(self, in_ch=NUM_CH, z_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 4, 2, 1), nn.ReLU(True),     # 32x32
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True),        # 16x16
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(True),       # 8x8
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(True),      # 4x4
        )
        self.fc = nn.Linear(256 * 4 * 4, z_dim)

    def forward(self, x):
        h = self.net(x)
        h = h.view(h.size(0), -1)
        z = self.fc(h)
        return z

class Decoder(nn.Module):
    def __init__(self, out_ch=NUM_CH, z_dim=LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(z_dim, 256 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(True),  # 8x8
            nn.ConvTranspose2d(128, 64,  4, 2, 1), nn.ReLU(True),  # 16x16
            nn.ConvTranspose2d(64,  32,  4, 2, 1), nn.ReLU(True),  # 32x32
            nn.ConvTranspose2d(32,  out_ch, 4, 2, 1),              # 64x64
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.fc(z).view(z.size(0), 256, 4, 4)
        xhat = self.net(h)
        return xhat

class SceneModelTernaryEdges64_256_Phase5(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.dec = Decoder()

        # heads
        self.between_head = nn.Sequential(nn.Linear(LATENT_DIM, 128), nn.ReLU(True), nn.Linear(128, 1))
        self.tproj_head   = nn.Sequential(nn.Linear(LATENT_DIM, 128), nn.ReLU(True), nn.Linear(128, 1))

        self.closer_sign_head = nn.Sequential(nn.Linear(LATENT_DIM, 128), nn.ReLU(True), nn.Linear(128, 2))
        self.closer_mag_head  = nn.Sequential(nn.Linear(LATENT_DIM, 128), nn.ReLU(True), nn.Linear(128, 1))

        self.overlapB_head = nn.Sequential(nn.Linear(LATENT_DIM, 128), nn.ReLU(True), nn.Linear(128, 2))
        self.overlapC_head = nn.Sequential(nn.Linear(LATENT_DIM, 128), nn.ReLU(True), nn.Linear(128, 2))

    def encode(self, x): return self.enc(x)
    def decode(self, z): return self.dec(z)

    def forward(self, x):
        z = self.encode(x)
        xhat = self.decode(z)

        between = torch.sigmoid(self.between_head(z)).squeeze(1)
        tproj   = torch.sigmoid(self.tproj_head(z)).squeeze(1)

        csign_logits = self.closer_sign_head(z)
        cmag = torch.sigmoid(self.closer_mag_head(z)).squeeze(1)

        oB_logits = self.overlapB_head(z)
        oC_logits = self.overlapC_head(z)

        return {
            "z": z,
            "xhat": xhat,
            "between": between,
            "tproj": tproj,
            "csign_logits": csign_logits,
            "cmag": cmag,
            "oB_logits": oB_logits,
            "oC_logits": oC_logits,
        }

# -----------------------
# Training
# -----------------------
def loss_fn(out, x, t):
    # reconstruction
    rec = F.mse_loss(out["xhat"], x)

    # factor heads
    between = F.mse_loss(out["between"], t["between_score"])
    # gate tproj loss by between-ness so it matters only when near the segment
    w = t["between_score"].detach()
    tproj = ((out["tproj"] - t["t_on_BC"])**2 * w).sum() / (w.sum() + 1e-9)


    csign   = F.cross_entropy(out["csign_logits"], t["closer_sign"])
    cmag    = F.l1_loss(out["cmag"], t["closer_mag"])

    oB      = F.cross_entropy(out["oB_logits"], t["overlap_B"])
    oC      = F.cross_entropy(out["oC_logits"], t["overlap_C"])

    # weights (tune as needed)
    total = (
        1.0 * rec +
        1.0 * between +
        0.5 * tproj +
        0.5 * csign +
        0.5 * cmag +
        0.5 * oB +
        0.5 * oC
    )
    parts = {
        "total": float(total.detach().cpu()),
        "rec": float(rec.detach().cpu()),
        "between": float(between.detach().cpu()),
        "tproj": float(tproj.detach().cpu()),
        "csign": float(csign.detach().cpu()),
        "cmag": float(cmag.detach().cpu()),
        "oB": float(oB.detach().cpu()),
        "oC": float(oC.detach().cpu()),
    }
    return total, parts

@torch.no_grad()
def val_epoch(model, dl):
    model.eval()
    acc = []
    for x, t in dl:
        x = x.to(DEVICE)
        for k in t: t[k] = t[k].to(DEVICE)
        out = model(x)
        total, _ = loss_fn(out, x, t)
        acc.append(float(total.detach().cpu()))
    return float(np.mean(acc))

def train():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"{TAG} Using device: {DEVICE}")
    print(f"{TAG} Targets: between_score, t_on_BC, closer_sign, closer_mag, overlap_B, overlap_C")
    print(f"{TAG} Params: SIGMA_PERP={SIGMA_PERP}, SIGMA_OUTSIDE={SIGMA_OUTSIDE}, CLOSER_SIGMA={CLOSER_SIGMA}, OVERLAP_TOL={OVERLAP_TOL}")
    print(f"{TAG} Rotation90 prob = {ROT90_PROB}")

    ds_tr = GeomEdgesTernary64DatasetPhase5(N_TRAIN, seed=SEED_TRAIN)
    ds_va = GeomEdgesTernary64DatasetPhase5(N_VAL,   seed=SEED_VAL)

    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, collate_fn=collate_phase5)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_phase5)

    model = SceneModelTernaryEdges64_256_Phase5().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = 1e9
    history = []

    for ep in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for x, t in dl_tr:
            x = x.to(DEVICE)
            for k in t: t[k] = t[k].to(DEVICE)

            out = model(x)
            total, _ = loss_fn(out, x, t)

            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()
            losses.append(float(total.detach().cpu()))

        tr = float(np.mean(losses))
        va = val_epoch(model, dl_va)
        history.append({"epoch": ep, "train": tr, "val": va})

        print(f"{TAG} Epoch {ep:3d}/{EPOCHS} | train={tr:.4f} | val={va:.4f}")

        if va < best_val:
            best_val = va
            torch.save(
                {"model_state_dict": model.state_dict(), "history": history, "best_val": best_val},
                CKPT_PATH
            )

    print(f"{TAG} Saved -> {CKPT_PATH}")
    meta = {
        "IMG_SIZE": IMG_SIZE,
        "NUM_CH": NUM_CH,
        "LATENT_DIM": LATENT_DIM,
        "REL5_NAMES": REL5_NAMES,
        "params": {
            "SIGMA_PERP": SIGMA_PERP,
            "SIGMA_OUTSIDE": SIGMA_OUTSIDE,
            "CLOSER_SIGMA": CLOSER_SIGMA,
            "OVERLAP_TOL": OVERLAP_TOL,
            "END_MARGIN": END_MARGIN,
            "BETWEEN_THRESH": BETWEEN_THRESH,
            "CLOSER_THRESH": CLOSER_THRESH,
        }
    }
    with open(os.path.join(OUT_DIR, "phase5_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"{TAG} done.")

if __name__ == "__main__":
    train()
