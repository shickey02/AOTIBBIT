#!/usr/bin/env python3
# geomlang_edges_relternary_train64_latent256_phase1.py
#
# Phase 1 upgrades for ternary training:
#   - Identity channels: A, B, C are fixed channels (no permutation shortcuts)
#   - Asymmetric labels: relation is defined "about A" relative to directed segment B->C
#   - Clean boundaries: tolerance + resample to avoid ambiguous boundary cases
#   - Rotation augmentation: random 0/90/180/270 degrees by rotating centers before rendering
#
# Output:
#   outputs_edges_relternary256_phase1/scene_model_edges_relternary256_phase1.pt

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------
# Config
# -----------------------
IMG_SIZE   = 64
NUM_CH     = 4          # A fill, B fill, C fill, edges
LATENT_DIM = 256

N_TRAIN    = 24000
N_VAL      = 6000
BATCH_SIZE = 128
N_EPOCHS   = 40
LR         = 1e-3

OUT_DIR         = "outputs_edges_relternary256_phase1"
CKPT_SCENEMODEL = os.path.join(OUT_DIR, "scene_model_edges_relternary256_phase1.pt")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Phase-1 label space (7 classes)
# -----------------------
# These are asymmetric by construction: A is the query, B->C defines an oriented reference.
#
# 0: A_left_of_BtoC        (cross > tol)
# 1: A_right_of_BtoC       (cross < -tol)
# 2: A_between_B_and_C      (projection between + near line)
# 3: A_closer_to_B          (d(A,B) + tol < d(A,C))
# 4: A_closer_to_C          (d(A,C) + tol < d(A,B))
# 5: A_overlap_B            (A overlaps B strongly)
# 6: A_overlap_C            (A overlaps C strongly)
#
# Priority order matters: overlap first, then between, then side, then closer.
REL_A_LEFT, REL_A_RIGHT, REL_A_BETWEEN, REL_A_CLOSER_B, REL_A_CLOSER_C, REL_A_OVER_B, REL_A_OVER_C = range(7)
REL_NAMES = [
    "A_left_of_BtoC",
    "A_right_of_BtoC",
    "A_between_BC",
    "A_closer_to_B",
    "A_closer_to_C",
    "A_overlap_B",
    "A_overlap_C",
]

# -----------------------
# Geometry tolerances (Phase 1 boundary hardening)
# -----------------------
CROSS_TOL      = 2.0     # tolerance for side-of-line cross product (in pixel^2-ish units after scaling)
CLOSER_TOL     = 2.0     # tolerance for "closer" comparisons (pixels)
BETWEEN_LINE_TOL = 2.0   # distance-to-line tolerance for BETWEEN
OVERLAP_TOL    = 3.0     # if centers are within this many pixels, treat as overlap-ish

# Dataset sampling
MARGIN = 6
SIZE_MIN, SIZE_MAX = 4, 14

# If a sample falls into an ambiguous boundary region, we resample up to this many tries.
MAX_RESAMPLE_TRIES = 50

# Random 90-degree rotation augmentation
ROTATE90_PROB = 1.0  # set <1.0 if you only want it sometimes

# -----------------------
# Shape drawing / edges
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
    # union edges across all objects (boolean union first)
    union = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        union |= (m > 0.5)

    union_f = union.astype(np.float32)

    interior = np.zeros_like(union_f)
    interior[1:-1, 1:-1] = (
        union_f[1:-1, 1:-1] *
        union_f[:-2, 1:-1] *
        union_f[2:, 1:-1] *
        union_f[1:-1, :-2] *
        union_f[1:-1, 2:]
    )
    edges = union_f - interior
    edges[edges < 0] = 0.0
    return edges


# -----------------------
# Rotation augmentation (centers only, no image interpolation)
# -----------------------
def rot90_point(cx, cy, k, W, H):
    """
    Rotate point (cx,cy) around image center using k*90 degrees CCW.
    Assumes coordinates in pixel space [0..W-1], [0..H-1].
    """
    if k % 4 == 0:
        return cx, cy
    if k % 4 == 1:
        # (x,y) -> (y, W-1-x) for CCW when W==H
        return cy, (W - 1 - cx)
    if k % 4 == 2:
        return (W - 1 - cx), (H - 1 - cy)
    # k==3
    return (H - 1 - cy), cx

# -----------------------
# Ternary asymmetric label
# -----------------------
def _dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def _dist_point_to_line(A, B, C):
    """
    Distance from point A to infinite line through B->C.
    """
    x0,y0 = A
    x1,y1 = B
    x2,y2 = C
    dx = x2 - x1
    dy = y2 - y1
    denom = math.sqrt(dx*dx + dy*dy) + 1e-8
    # |(A-B) x (C-B)| / |C-B|
    cross = abs((x0-x1)*dy - (y0-y1)*dx)
    return cross / denom

def _proj_t(A, B, C):
    """
    Projection parameter t of A onto segment B->C:
      t=0 at B, t=1 at C (infinite line projection)
    """
    x0,y0 = A
    x1,y1 = B
    x2,y2 = C
    vx = x2 - x1
    vy = y2 - y1
    wx = x0 - x1
    wy = y0 - y1
    vv = vx*vx + vy*vy + 1e-8
    return (wx*vx + wy*vy) / vv

def relation_ternary_asym(A, B, C):
    """
    Return one of 7 classes. Also returns a boolean 'ambiguous' flag
    when the sample is too close to decision boundaries.
    """
    dAB = _dist(A,B)
    dAC = _dist(A,C)

    # overlap first
    if dAB <= OVERLAP_TOL:
        return REL_A_OVER_B, False
    if dAC <= OVERLAP_TOL:
        return REL_A_OVER_C, False

    # between: projection within [0,1] and close to line
    t = _proj_t(A,B,C)
    line_d = _dist_point_to_line(A,B,C)
    if (0.0 <= t <= 1.0) and (line_d <= BETWEEN_LINE_TOL):
        return REL_A_BETWEEN, False

    # side-of directed line B->C via signed cross product
    # cross = (A-B) x (C-B) = (Ax-Bx)*(Cy-By) - (Ay-By)*(Cx-Bx)
    cross = (A[0]-B[0])*(C[1]-B[1]) - (A[1]-B[1])*(C[0]-B[0])
    if cross > CROSS_TOL:
        return REL_A_LEFT, False
    if cross < -CROSS_TOL:
        return REL_A_RIGHT, False

    # closer-to with tolerance
    if dAB + CLOSER_TOL < dAC:
        return REL_A_CLOSER_B, False
    if dAC + CLOSER_TOL < dAB:
        return REL_A_CLOSER_C, False

    # If we reach here, we are in an ambiguous boundary region.
    # Mark ambiguous so dataset can resample (Phase 1 boundary hardening).
    return REL_A_CLOSER_B, True  # placeholder label, will be resampled anyway

# -----------------------
# Dataset (identity channels + asymmetry + rotation + resampling)
# -----------------------
class GeomEdgesTernary64Phase1(Dataset):
    def __init__(self, n_samples, seed=None):
        super().__init__()
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n_samples

    def _sample_center(self, s, W, H):
        cx = int(self.rng.integers(MARGIN + s, W - MARGIN - s))
        cy = int(self.rng.integers(MARGIN + s, H - MARGIN - s))
        return cx, cy

    def __getitem__(self, idx):
        H = W = IMG_SIZE
        rng = self.rng

        for _try in range(MAX_RESAMPLE_TRIES):
            # Shapes per object: 0 circle, 1 square
            shape_A = int(rng.integers(0, 2))
            shape_B = int(rng.integers(0, 2))
            shape_C = int(rng.integers(0, 2))

            base = int(rng.integers(5, 13))  # 5..12
            size_A = int(np.clip(base + int(rng.integers(-2, 3)), SIZE_MIN, SIZE_MAX))
            size_B = int(np.clip(base + int(rng.integers(-2, 3)), SIZE_MIN, SIZE_MAX))
            size_C = int(np.clip(base + int(rng.integers(-2, 3)), SIZE_MIN, SIZE_MAX))

            Ax, Ay = self._sample_center(size_A, W, H)
            Bx, By = self._sample_center(size_B, W, H)
            Cx, Cy = self._sample_center(size_C, W, H)

            # Rotation augmentation (by rotating centers; preserves crisp rasterization)
            if rng.random() < ROTATE90_PROB:
                k = int(rng.integers(0, 4))
                Ax, Ay = rot90_point(Ax, Ay, k, W, H)
                Bx, By = rot90_point(Bx, By, k, W, H)
                Cx, Cy = rot90_point(Cx, Cy, k, W, H)

            rel, ambiguous = relation_ternary_asym((Ax,Ay), (Bx,By), (Cx,Cy))
            if ambiguous:
                continue  # resample until not on a boundary

            A = np.zeros((H, W), dtype=np.float32)
            B = np.zeros((H, W), dtype=np.float32)
            C = np.zeros((H, W), dtype=np.float32)

            if shape_A == 0: draw_circle(A, Ax, Ay, size_A)
            else:            draw_square(A, Ax, Ay, size_A)

            if shape_B == 0: draw_circle(B, Bx, By, size_B)
            else:            draw_square(B, Bx, By, size_B)

            if shape_C == 0: draw_circle(C, Cx, Cy, size_C)
            else:            draw_square(C, Cx, Cy, size_C)

            edges = make_edges(A, B, C)

            img = np.stack([A, B, C, edges], axis=0)  # [4,64,64]
            return (
                torch.from_numpy(img),
                torch.tensor(rel, dtype=torch.long),
                torch.tensor(shape_A, dtype=torch.long),
                torch.tensor(shape_B, dtype=torch.long),
                torch.tensor(shape_C, dtype=torch.long),
            )

        # Fallback: if we somehow failed many times, emit a sample without resampling
        # (rare, but prevents DataLoader hard failures)
        shape_A = int(rng.integers(0, 2))
        shape_B = int(rng.integers(0, 2))
        shape_C = int(rng.integers(0, 2))
        size_A = 10; size_B = 10; size_C = 10
        Ax, Ay = 32, 32
        Bx, By = 24, 32
        Cx, Cy = 40, 32
        rel, _ = relation_ternary_asym((Ax,Ay),(Bx,By),(Cx,Cy))
        A = np.zeros((H, W), dtype=np.float32)
        B = np.zeros((H, W), dtype=np.float32)
        C = np.zeros((H, W), dtype=np.float32)
        draw_circle(A, Ax, Ay, size_A)
        draw_circle(B, Bx, By, size_B)
        draw_circle(C, Cx, Cy, size_C)
        edges = make_edges(A,B,C)
        img = np.stack([A,B,C,edges], axis=0)
        return (
            torch.from_numpy(img),
            torch.tensor(rel, dtype=torch.long),
            torch.tensor(shape_A, dtype=torch.long),
            torch.tensor(shape_B, dtype=torch.long),
            torch.tensor(shape_C, dtype=torch.long),
        )

# -----------------------
# Model
# -----------------------
class Encoder(nn.Module):
    def __init__(self, in_channels=NUM_CH, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1)  # 64->32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)           # 32->16
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)          # 16->8
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)         # 8->4
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
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 4->8
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # 8->16
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)    # 16->32
        self.deconv4 = nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1)  # 32->64

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, 4, 4)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        return torch.sigmoid(self.deconv4(x))

class SceneModelTernaryPhase1_256(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.rel_head   = nn.Linear(LATENT_DIM, 7)
        self.shapeA_head = nn.Linear(LATENT_DIM, 2)
        self.shapeB_head = nn.Linear(LATENT_DIM, 2)
        self.shapeC_head = nn.Linear(LATENT_DIM, 2)

    def forward(self, x):
        z = self.encoder(x)
        rec = self.decoder(z)
        rel_logits = self.rel_head(z)
        a_log = self.shapeA_head(z)
        b_log = self.shapeB_head(z)
        c_log = self.shapeC_head(z)
        return rec, rel_logits, a_log, b_log, c_log

    def encode(self, x): return self.encoder(x)
    def decode(self, z): return self.decoder(z)

# -----------------------
# Training
# -----------------------
def main():
    print(f"[train64-ternary-phase1-256] Using device: {DEVICE}")
    print("[train64-ternary-phase1-256] Labels:", REL_NAMES)
    print(f"[train64-ternary-phase1-256] Tols: CROSS={CROSS_TOL}, CLOSER={CLOSER_TOL}, BETWEEN_LINE={BETWEEN_LINE_TOL}, OVERLAP={OVERLAP_TOL}")
    print(f"[train64-ternary-phase1-256] Rotation90 prob = {ROTATE90_PROB}")

    train_ds = GeomEdgesTernary64Phase1(N_TRAIN, seed=0)
    val_ds   = GeomEdgesTernary64Phase1(N_VAL,   seed=123)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SceneModelTernaryPhase1_256().to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    bce = nn.BCELoss()
    ce  = nn.CrossEntropyLoss()

    def run_epoch(loader, train=True):
        model.train() if train else model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.set_grad_enabled(train):
            for imgs, rel, sA, sB, sC in loader:
                imgs = imgs.to(DEVICE)
                rel  = rel.to(DEVICE)
                sA   = sA.to(DEVICE)
                sB   = sB.to(DEVICE)
                sC   = sC.to(DEVICE)

                if train:
                    opt.zero_grad()

                rec, rel_log, a_log, b_log, c_log = model(imgs)

                rec_loss = bce(rec, imgs)
                rel_loss = ce(rel_log, rel)
                a_loss   = ce(a_log, sA)
                b_loss   = ce(b_log, sB)
                c_loss   = ce(c_log, sC)

                # Keep reconstruction dominant, but make relation head matter.
                cls_loss = rel_loss + 0.25*(a_loss + b_loss + c_loss)
                loss = rec_loss + 0.35 * cls_loss

                if train:
                    loss.backward()
                    opt.step()

                total_loss += float(loss.item())
                n_batches += 1

        return total_loss / max(1, n_batches)

    for epoch in range(1, N_EPOCHS + 1):
        train_loss = run_epoch(train_dl, train=True)
        val_loss   = run_epoch(val_dl,   train=False)
        print(f"[train64-ternary-phase1-256] Epoch {epoch:3d}/{N_EPOCHS} | train={train_loss:.4f} | val={val_loss:.4f}")

    ckpt = {
        "model_state_dict": model.state_dict(),
        "rel_names": REL_NAMES,
        "tols": {
            "CROSS_TOL": CROSS_TOL,
            "CLOSER_TOL": CLOSER_TOL,
            "BETWEEN_LINE_TOL": BETWEEN_LINE_TOL,
            "OVERLAP_TOL": OVERLAP_TOL,
        },
        "num_ch": NUM_CH,
    }
    torch.save(ckpt, CKPT_SCENEMODEL)
    print(f"[train64-ternary-phase1-256] Saved -> {CKPT_SCENEMODEL}")

if __name__ == "__main__":
    main()
