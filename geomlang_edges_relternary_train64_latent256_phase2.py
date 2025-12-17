#!/usr/bin/env python3
# geomlang_edges_relternary_train64_latent256_phase2.py
#
# Phase 2 trainer: ternary (A,B,C) relation learning + A-scan minibatch injection.
# Goal: fix the "dataset looks good but scan field fails" issue by training on scans
# while still training on the random dataset.
#
# Outputs:
#   outputs_edges_relternary256_phase2/scene_model_edges_relternary256_phase2.pt

import os, math, json
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

# Channels: A, B, C, edges
NUM_CH = 4

N_TRAIN    = 24000
N_VAL      = 6000
BATCH_SIZE = 128
N_EPOCHS   = 40
LR         = 1e-3

OUT_DIR = "outputs_edges_relternary256_phase2"
CKPT_PATH = os.path.join(OUT_DIR, "scene_model_edges_relternary256_phase2.pt")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TAG = "[train64-ternary-phase2-256]"

# -----------------------
# Labels (7-way)
# -----------------------
REL_A_LEFT_OF_BC   = 0
REL_A_RIGHT_OF_BC  = 1
REL_A_BETWEEN_BC   = 2
REL_A_CLOSER_TO_B  = 3
REL_A_CLOSER_TO_C  = 4
REL_A_OVERLAP_B    = 5
REL_A_OVERLAP_C    = 6

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
# Tolerances / geometry knobs
# -----------------------
CROSS_TOL        = 2.0   # for left/right of directed BC (cross product threshold)
CLOSER_TOL       = 2.0   # for closer-to comparisons
BETWEEN_LINE_TOL = 2.0   # distance-to-line threshold for "between"
OVERLAP_PAD      = 3.0   # overlap threshold padding in distance test

# Rotation augmentation (90° multiples)
ROT90_PROB = 1.0

# -----------------------
# Phase 2: A-scan minibatch injection
# -----------------------
SCAN_EVERY_STEPS = 2           # inject scan batch every N steps (1-4 typical)
SCAN_BATCH_SIZE  = BATCH_SIZE
SCAN_CAND_N      = 401         # candidate A positions along scan before balancing
SCAN_T_MIN       = -0.25       # extend beyond segment BC
SCAN_T_MAX       = 1.25
SCAN_USE_BALANCE = True

SCAN_REC_W       = 1.00        # reconstruction weight for scan batch
SCAN_REL_W       = 0.50        # relation head weight for scan batch
SCAN_SHAPE_W     = 0.25        # shape heads weight inside scan batch

# Scan anchors (same idea as your eval suite)
SCAN_CONFIGS = {
    "horiz_mid":   ((18, 32), (46, 32)),
    "vert_mid":    ((32, 18), (32, 46)),
    "diag_mid":    ((20, 20), (44, 44)),
    "horiz_short": ((23, 32), (41, 32)),
    "horiz_off":   ((18, 22), (46, 40)),
}

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
    # robust union edges even if masks are float32
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

def _rot90_point(cx, cy, k, H, W):
    k = k % 4
    if k == 0:
        return cx, cy
    if k == 1:
        return cy, (W - 1 - cx)
    if k == 2:
        return (W - 1 - cx), (H - 1 - cy)
    return (H - 1 - cy), cx

def maybe_rot90_triplet(Axy, Bxy, Cxy, p=1.0):
    if p <= 0:
        return Axy, Bxy, Cxy
    if np.random.rand() < p:
        k = np.random.randint(0, 4)
        H = W = IMG_SIZE
        Ax, Ay = _rot90_point(Axy[0], Axy[1], k, H, W)
        Bx, By = _rot90_point(Bxy[0], Bxy[1], k, H, W)
        Cx, Cy = _rot90_point(Cxy[0], Cxy[1], k, H, W)
        return (Ax, Ay), (Bx, By), (Cx, Cy)
    return Axy, Bxy, Cxy

# -----------------------
# Geometry label rule
# -----------------------
def dist(a, b):
    dx = float(a[0] - b[0])
    dy = float(a[1] - b[1])
    return math.sqrt(dx*dx + dy*dy)

def point_line_distance(A, B, C):
    # distance from point A to line through B->C
    x0, y0 = float(A[0]), float(A[1])
    x1, y1 = float(B[0]), float(B[1])
    x2, y2 = float(C[0]), float(C[1])
    vx, vy = (x2 - x1), (y2 - y1)
    wx, wy = (x0 - x1), (y0 - y1)
    denom = math.sqrt(vx*vx + vy*vy) + 1e-9
    # area-based distance
    cross = abs(vx*wy - vy*wx)
    return cross / denom

def projection_t(A, B, C):
    # t for projection of A onto line B->C: B + t*(C-B)
    x0, y0 = float(A[0]), float(A[1])
    x1, y1 = float(B[0]), float(B[1])
    x2, y2 = float(C[0]), float(C[1])
    vx, vy = (x2 - x1), (y2 - y1)
    wx, wy = (x0 - x1), (y0 - y1)
    vv = vx*vx + vy*vy + 1e-9
    return (wx*vx + wy*vy) / vv

def ternary_label_from_centers(Axy, Bxy, Cxy, sizeA, sizeB, sizeC):
    # Priority ordering matters.
    # 1) Overlaps
    if dist(Axy, Bxy) <= (sizeA + sizeB - OVERLAP_PAD):
        return REL_A_OVERLAP_B
    if dist(Axy, Cxy) <= (sizeA + sizeC - OVERLAP_PAD):
        return REL_A_OVERLAP_C

    # 2) Between: near line AND projection in [0,1]
    d_line = point_line_distance(Axy, Bxy, Cxy)
    t = projection_t(Axy, Bxy, Cxy)
    if (d_line <= BETWEEN_LINE_TOL) and (0.0 <= t <= 1.0):
        return REL_A_BETWEEN_BC

    # 3) Closer comparisons (only if difference is meaningful)
    dB = dist(Axy, Bxy)
    dC = dist(Axy, Cxy)
    if dB + CLOSER_TOL < dC:
        return REL_A_CLOSER_TO_B
    if dC + CLOSER_TOL < dB:
        return REL_A_CLOSER_TO_C

    # 4) Left/right of directed segment B->C (cross sign)
    bx, by = float(Bxy[0]), float(Bxy[1])
    cx, cy = float(Cxy[0]), float(Cxy[1])
    ax, ay = float(Axy[0]), float(Axy[1])
    vx, vy = (cx - bx), (cy - by)
    wx, wy = (ax - bx), (ay - by)
    cross = vx*wy - vy*wx

    if cross > CROSS_TOL:
        return REL_A_LEFT_OF_BC
    if cross < -CROSS_TOL:
        return REL_A_RIGHT_OF_BC

    # Fallback: if ambiguous, decide by cross sign (or default to left)
    return REL_A_LEFT_OF_BC if cross >= 0 else REL_A_RIGHT_OF_BC

# -----------------------
# Rendering
# -----------------------
def render_scene_ABC(cxA, cyA, cxB, cyB, cxC, cyC, shapeA, shapeB, shapeC, sizeA, sizeB, sizeC):
    H = W = IMG_SIZE
    A = np.zeros((H, W), dtype=np.float32)
    B = np.zeros((H, W), dtype=np.float32)
    C = np.zeros((H, W), dtype=np.float32)

    if shapeA == 0: draw_circle(A, cxA, cyA, sizeA)
    else:           draw_square(A, cxA, cyA, sizeA)

    if shapeB == 0: draw_circle(B, cxB, cyB, sizeB)
    else:           draw_square(B, cxB, cyB, sizeB)

    if shapeC == 0: draw_circle(C, cxC, cyC, sizeC)
    else:           draw_square(C, cxC, cyC, sizeC)

    edges = make_edges(A, B, C)

    # Channel order must match model
    img = np.stack([A, B, C, edges], axis=0).astype(np.float32)  # [4,H,W]
    return img

# -----------------------
# Dataset (random scenes)
# -----------------------
class GeomEdgesTernary64Dataset(Dataset):
    def __init__(self, n_samples, seed=None):
        super().__init__()
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        rng = self.rng
        H = W = IMG_SIZE
        margin = 6

        # shapes
        shapeA = int(rng.integers(0, 2))
        shapeB = int(rng.integers(0, 2))
        shapeC = int(rng.integers(0, 2))

        # sizes (keep similar scale)
        base = int(rng.integers(5, 13))  # 5..12
        sizeA = int(np.clip(base + int(rng.integers(-2, 3)), 4, 14))
        sizeB = int(np.clip(base + int(rng.integers(-2, 3)), 4, 14))
        sizeC = int(np.clip(base + int(rng.integers(-2, 3)), 4, 14))

        def sample_center(s):
            cx = int(rng.integers(margin + s, W - margin - s))
            cy = int(rng.integers(margin + s, H - margin - s))
            return cx, cy

        Ax, Ay = sample_center(sizeA)
        Bx, By = sample_center(sizeB)
        Cx, Cy = sample_center(sizeC)

        (Ax, Ay), (Bx, By), (Cx, Cy) = maybe_rot90_triplet((Ax, Ay), (Bx, By), (Cx, Cy), p=ROT90_PROB)

        rel = ternary_label_from_centers((Ax, Ay), (Bx, By), (Cx, Cy), sizeA, sizeB, sizeC)

        img = render_scene_ABC(Ax, Ay, Bx, By, Cx, Cy, shapeA, shapeB, shapeC, sizeA, sizeB, sizeC)

        return (
            torch.from_numpy(img),
            torch.tensor(rel, dtype=torch.long),
            torch.tensor(shapeA, dtype=torch.long),
            torch.tensor(shapeB, dtype=torch.long),
            torch.tensor(shapeC, dtype=torch.long),
        )

# -----------------------
# A-scan minibatch generator
# -----------------------
def make_ascan_minibatch(batch_size: int, scan_name: str, margin: int = 6):
    (Bx, By), (Cx, Cy) = SCAN_CONFIGS[scan_name]
    H = W = IMG_SIZE

    # Fix shapes/sizes for Phase2 geometry field training
    shapeA = 0; shapeB = 0; shapeC = 0
    sizeA  = 10; sizeB  = 10; sizeC  = 10

    ts = np.linspace(SCAN_T_MIN, SCAN_T_MAX, SCAN_CAND_N)

    cand = []
    cand_y = []

    for t in ts:
        Ax = int(round(Bx + t * (Cx - Bx)))
        Ay = int(round(By + t * (Cy - By)))

        Ax = int(np.clip(Ax, margin + sizeA, W - margin - sizeA))
        Ay = int(np.clip(Ay, margin + sizeA, H - margin - sizeA))

        (Ax2, Ay2), (Bx2, By2), (Cx2, Cy2) = maybe_rot90_triplet((Ax, Ay), (Bx, By), (Cx, Cy), p=ROT90_PROB)

        rel = ternary_label_from_centers((Ax2, Ay2), (Bx2, By2), (Cx2, Cy2), sizeA, sizeB, sizeC)

        cand.append((Ax2, Ay2, Bx2, By2, Cx2, Cy2))
        cand_y.append(int(rel))

    cand_y = np.array(cand_y, dtype=np.int64)
    n_classes = len(REL_NAMES)

    if SCAN_USE_BALANCE:
        per_c = max(1, batch_size // n_classes)
        idxs = []
        for c in range(n_classes):
            where = np.where(cand_y == c)[0]
            if len(where) == 0:
                continue
            pick = np.random.choice(where, size=per_c, replace=(len(where) < per_c))
            idxs.append(pick)
        if len(idxs) == 0:
            chosen = np.random.choice(len(cand), size=batch_size, replace=True)
        else:
            chosen = np.concatenate(idxs, axis=0)
            if len(chosen) < batch_size:
                pad = np.random.choice(len(cand), size=(batch_size - len(chosen)), replace=True)
                chosen = np.concatenate([chosen, pad], axis=0)
            chosen = chosen[:batch_size]
    else:
        chosen = np.random.choice(len(cand), size=batch_size, replace=True)

    imgs = []
    rels = []
    sA = []; sB = []; sC = []
    for i in chosen:
        Ax2, Ay2, Bx2, By2, Cx2, Cy2 = cand[i]
        img = render_scene_ABC(Ax2, Ay2, Bx2, By2, Cx2, Cy2, shapeA, shapeB, shapeC, sizeA, sizeB, sizeC)
        imgs.append(img)
        rels.append(int(cand_y[i]))
        sA.append(shapeA); sB.append(shapeB); sC.append(shapeC)

    x = torch.from_numpy(np.stack(imgs, axis=0)).float()
    y = torch.tensor(rels, dtype=torch.long)
    sA = torch.tensor(sA, dtype=torch.long)
    sB = torch.tensor(sB, dtype=torch.long)
    sC = torch.tensor(sC, dtype=torch.long)
    return x, y, sA, sB, sC

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
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # 4x4
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
        x = self.fc(z).view(z.size(0), 256, 4, 4)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        return torch.sigmoid(self.deconv4(x))

class SceneModelTernaryEdges64_256(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.rel_head  = nn.Linear(LATENT_DIM, len(REL_NAMES))
        self.shapeA    = nn.Linear(LATENT_DIM, 2)
        self.shapeB    = nn.Linear(LATENT_DIM, 2)
        self.shapeC    = nn.Linear(LATENT_DIM, 2)

    def forward(self, x):
        z = self.encoder(x)
        rec = self.decoder(z)
        rel = self.rel_head(z)
        sA  = self.shapeA(z)
        sB  = self.shapeB(z)
        sC  = self.shapeC(z)
        return rec, rel, sA, sB, sC

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

# -----------------------
# Training
# -----------------------
def main():
    print(f"{TAG} Using device: {DEVICE}")
    print(f"{TAG} Labels: {REL_NAMES}")
    print(f"{TAG} Tols: CROSS={CROSS_TOL}, CLOSER={CLOSER_TOL}, BETWEEN_LINE={BETWEEN_LINE_TOL}, OVERLAP_PAD={OVERLAP_PAD}")
    print(f"{TAG} Rotation90 prob = {ROT90_PROB}")
    print(f"{TAG} Scan inject: every {SCAN_EVERY_STEPS} steps | batch={SCAN_BATCH_SIZE} | balance={SCAN_USE_BALANCE}")

    train_ds = GeomEdgesTernary64Dataset(N_TRAIN)
    val_ds   = GeomEdgesTernary64Dataset(N_VAL, seed=123)  # deterministic-ish val

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SceneModelTernaryEdges64_256().to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    bce = nn.BCELoss()
    ce  = nn.CrossEntropyLoss()

    scan_names = list(SCAN_CONFIGS.keys())

    def run_epoch(loader, train=True):
        if train:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        n_batches = 0
        step = 0

        with torch.set_grad_enabled(train):
            for imgs, rel, sA, sB, sC in loader:
                step += 1

                imgs = imgs.to(DEVICE)
                rel  = rel.to(DEVICE)
                sA   = sA.to(DEVICE)
                sB   = sB.to(DEVICE)
                sC   = sC.to(DEVICE)

                if train:
                    opt.zero_grad()

                rec, rel_log, sA_log, sB_log, sC_log = model(imgs)

                rec_loss = bce(rec, imgs)
                rel_loss = ce(rel_log, rel)
                sA_loss  = ce(sA_log, sA)
                sB_loss  = ce(sB_log, sB)
                sC_loss  = ce(sC_log, sC)

                cls_loss = rel_loss + sA_loss + sB_loss + sC_loss
                loss = rec_loss + 0.25 * cls_loss

                # ---- Inject scan minibatch (train only) ----
                if train and (step % SCAN_EVERY_STEPS == 0):
                    scan_name = scan_names[np.random.randint(0, len(scan_names))]
                    x_scan, y_scan, sA_scan, sB_scan, sC_scan = make_ascan_minibatch(
                        batch_size=SCAN_BATCH_SIZE,
                        scan_name=scan_name,
                        margin=6,
                    )
                    x_scan  = x_scan.to(DEVICE)
                    y_scan  = y_scan.to(DEVICE)
                    sA_scan = sA_scan.to(DEVICE)
                    sB_scan = sB_scan.to(DEVICE)
                    sC_scan = sC_scan.to(DEVICE)

                    rec_s, rel_s, sA_s, sB_s, sC_s = model(x_scan)

                    rec_loss_s = bce(rec_s, x_scan)
                    rel_loss_s = ce(rel_s, y_scan)
                    sA_loss_s  = ce(sA_s, sA_scan)
                    sB_loss_s  = ce(sB_s, sB_scan)
                    sC_loss_s  = ce(sC_s, sC_scan)

                    loss = loss + (SCAN_REC_W * rec_loss_s) + (SCAN_REL_W * rel_loss_s) + (SCAN_SHAPE_W * (sA_loss_s + sB_loss_s + sC_loss_s))

                if train:
                    loss.backward()
                    opt.step()

                total_loss += float(loss.item())
                n_batches += 1

        return total_loss / max(1, n_batches)

    for epoch in range(1, N_EPOCHS + 1):
        tr = run_epoch(train_dl, train=True)
        va = run_epoch(val_dl, train=False)
        print(f"{TAG} Epoch {epoch:3d}/{N_EPOCHS} | train={tr:.4f} | val={va:.4f}")

    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": {
            "IMG_SIZE": IMG_SIZE,
            "NUM_CH": NUM_CH,
            "LATENT_DIM": LATENT_DIM,
            "REL_NAMES": REL_NAMES,
            "tols": dict(CROSS=CROSS_TOL, CLOSER=CLOSER_TOL, BETWEEN_LINE=BETWEEN_LINE_TOL, OVERLAP_PAD=OVERLAP_PAD),
            "scan": dict(SCAN_EVERY_STEPS=SCAN_EVERY_STEPS, SCAN_BATCH_SIZE=SCAN_BATCH_SIZE, SCAN_USE_BALANCE=SCAN_USE_BALANCE),
        }
    }
    torch.save(ckpt, CKPT_PATH)
    print(f"{TAG} Saved -> {CKPT_PATH}")

if __name__ == "__main__":
    main()
