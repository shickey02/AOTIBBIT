#!/usr/bin/env python3
# geomlang_edges_relternary_train64_latent256_phase3.py
#
# Phase 3: "Between system first" by explicitly projecting A onto segment BC.
#
# Key upgrades vs Phase2:
# - "between" label is defined by projection u on BC + perpendicular tube distance
# - Auxiliary head predicts (u, d_perp) from latent; loss applied on BETWEEN samples
# - Optional A-scan minibatches to harden manifold behavior (same scan anchors)
#
# Channels:
#   0 = A fill
#   1 = B fill
#   2 = C fill
#   3 = edges (union of A/B/C outlines)
#
# Output:
#   outputs_edges_relternary256_phase3/
#     scene_model_edges_relternary256_phase3.pt

import os, math, json, random
import numpy as np
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

# -----------------------
# Config
# -----------------------
IMG_SIZE   = 64
NUM_CH     = 4
LATENT_DIM = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = "outputs_edges_relternary256_phase3"
os.makedirs(OUT_DIR, exist_ok=True)

CKPT_PATH = os.path.join(OUT_DIR, "scene_model_edges_relternary256_phase3.pt")

REL_NAMES = [
    "A_left_of_BtoC",
    "A_right_of_BtoC",
    "A_between_BC",
    "A_closer_to_B",
    "A_closer_to_C",
    "A_overlap_B",
    "A_overlap_C",
]

# ---- Phase 3 tolerances ----
# "Between" is the star: segment projection + tube thickness
BETWEEN_LINE_TOL = 2.0      # thickness of tube around BC line
BETWEEN_END_MARGIN = 0.10   # keep away from u near 0 or 1 to avoid closer/endpoint ambiguity
CLOSER_TOL = 2.0
OVERLAP_TOL = 3.0

# ---- Data/Training ----
TRAIN_N = 12000
VAL_N   = 2000
EPOCHS  = 40
BATCH_SIZE = 256
LR = 2e-4

ROT90_PROB = 1.0  # keep consistent with your earlier runs

# ---- A-scan minibatch hardening (optional but recommended) ----
USE_ASCAN_MINIBATCH = True
ASCAN_EVERY_N_STEPS = 12     # frequency inside epoch
ASCAN_PER_NAME = 128         # samples per scan name per minibatch (kept small)
ASCAN_SHAPES = (0, 0, 0)     # fixed shapes during scans
ASCAN_SIZES  = (10, 10, 10)
ASCAN_MARGIN = 6

# Scan anchors (same style as your Phase2)
SCAN_T_MIN, SCAN_T_MAX = -0.35, 1.35
SCAN_CONFIGS: Dict[str, Tuple[Tuple[int,int], Tuple[int,int]]] = {
    "horiz_mid":   ((18,32), (46,32)),
    "vert_mid":    ((32,18), (32,46)),
    "diag_mid":    ((20,20), (44,44)),
    "horiz_short": ((23,32), (41,32)),
    "horiz_off":   ((18,22), (46,40)),
}


# ============================================================
# Geometry helpers (Phase 3 "between" projection)
# ============================================================
def proj_point_to_segment(A, B, C, eps=1e-9):
    """
    Project point A onto segment BC.
    Returns:
      u      : scalar on infinite line where P = B + u*(C-B)
      u_clamp: clamped u in [0,1]
      P      : projection on infinite line
      Pseg   : closest point on segment
      d_perp : perpendicular distance from A to infinite line BC
      d_seg  : distance from A to segment BC
    """
    Ax, Ay = float(A[0]), float(A[1])
    Bx, By = float(B[0]), float(B[1])
    Cx, Cy = float(C[0]), float(C[1])

    vx, vy = (Cx - Bx), (Cy - By)
    wx, wy = (Ax - Bx), (Ay - By)

    vv = vx*vx + vy*vy
    if vv < eps:
        u = 0.0
        u_clamp = 0.0
        P = (Bx, By)
        Pseg = (Bx, By)
        d_perp = math.hypot(Ax - Bx, Ay - By)
        d_seg  = d_perp
        return u, u_clamp, P, Pseg, float(d_perp), float(d_seg)

    u = (wx*vx + wy*vy) / vv
    u_clamp = float(np.clip(u, 0.0, 1.0))

    Px, Py = (Bx + u*vx, By + u*vy)
    Qx, Qy = (Bx + u_clamp*vx, By + u_clamp*vy)

    cross = abs(vx*wy - vy*wx)
    d_perp = cross / (math.sqrt(vv) + eps)

    d_seg = math.hypot(Ax - Qx, Ay - Qy)
    return float(u), float(u_clamp), (Px, Py), (Qx, Qy), float(d_perp), float(d_seg)


def is_between_by_projection(A, B, C,
                            between_line_tol=BETWEEN_LINE_TOL,
                            end_margin=BETWEEN_END_MARGIN):
    u, _, _, _, d_perp, _ = proj_point_to_segment(A, B, C)
    if not (end_margin <= u <= 1.0 - end_margin):
        return False
    if d_perp > between_line_tol:
        return False
    return True


def dist2(P, Q):
    dx = float(P[0]-Q[0])
    dy = float(P[1]-Q[1])
    return dx*dx + dy*dy


def overlaps(A, B, sizeA, sizeB, tol=OVERLAP_TOL):
    # Simple square overlap in center-distance terms (fast + stable)
    # overlap if center distance <= (sizeA + sizeB)/2 + tol
    thresh = (0.5*(sizeA + sizeB) + tol)
    return dist2(A, B) <= (thresh*thresh)


def side_of_line(P, B, C):
    # sign of cross((C-B), (P-B))
    return (C[0]-B[0])*(P[1]-B[1]) - (C[1]-B[1])*(P[0]-B[0])


def ternary_label_from_centers(A, B, C, sizeA, sizeB, sizeC):
    # Overlaps have priority
    if overlaps(A, B, sizeA, sizeB, tol=OVERLAP_TOL):
        return REL_NAMES.index("A_overlap_B")
    if overlaps(A, C, sizeA, sizeC, tol=OVERLAP_TOL):
        return REL_NAMES.index("A_overlap_C")

    # Phase 3: Between BEFORE closer (so it doesn't get stolen)
    if is_between_by_projection(A, B, C, BETWEEN_LINE_TOL, BETWEEN_END_MARGIN):
        return REL_NAMES.index("A_between_BC")

    # left/right relative to directed line B->C
    s = side_of_line(A, B, C)
    if s > 0:
        # call this "left" relative to direction B->C
        left_idx = REL_NAMES.index("A_left_of_BtoC")
        right_idx = REL_NAMES.index("A_right_of_BtoC")
        lr_label = left_idx
    elif s < 0:
        left_idx = REL_NAMES.index("A_left_of_BtoC")
        right_idx = REL_NAMES.index("A_right_of_BtoC")
        lr_label = right_idx
    else:
        # exactly on the line: fall back to closer
        lr_label = None

    # closer-to-B or closer-to-C
    dB = math.sqrt(dist2(A, B))
    dC = math.sqrt(dist2(A, C))
    if abs(dB - dC) <= CLOSER_TOL:
        # ambiguous → if LR exists, use it; else choose closer by tie-break
        if lr_label is not None:
            return lr_label
    if dB < dC:
        return REL_NAMES.index("A_closer_to_B")
    else:
        return REL_NAMES.index("A_closer_to_C")


# ============================================================
# Rendering helpers
# ============================================================
def draw_square(mask, cx, cy, r):
    H, W = mask.shape
    x0 = max(0, cx - r); x1 = min(W, cx + r + 1)
    y0 = max(0, cy - r); y1 = min(H, cy + r + 1)
    mask[y0:y1, x0:x1] = 1.0


def mask_edges(m):
    # cheap outline: pixel is edge if it is 1 and has a 0 neighbor (4-neigh)
    H, W = m.shape
    e = np.zeros_like(m)
    # interior check by shifts
    up    = np.zeros_like(m); up[1:,:]  = m[:-1,:]
    down  = np.zeros_like(m); down[:-1,:]= m[1:,:]
    left  = np.zeros_like(m); left[:,1:]= m[:,:-1]
    right = np.zeros_like(m); right[:,:-1]= m[:,1:]
    interior = (up > 0.5) & (down > 0.5) & (left > 0.5) & (right > 0.5) & (m > 0.5)
    e[(m > 0.5) & (~interior)] = 1.0
    return e.astype(np.float32)


def make_edges(A_m, B_m, C_m):
    union = (A_m > 0.5) | (B_m > 0.5) | (C_m > 0.5)
    # edge of union is ok, but we want outlines of each too, so OR them
    eA = mask_edges(A_m)
    eB = mask_edges(B_m)
    eC = mask_edges(C_m)
    eU = mask_edges(union.astype(np.float32))
    return ((eA > 0.5) | (eB > 0.5) | (eC > 0.5) | (eU > 0.5)).astype(np.float32)


def render_scene_ABC(Ax, Ay, Bx, By, Cx, Cy,
                     shapeA, shapeB, shapeC,
                     sizeA, sizeB, sizeC):
    H = W = IMG_SIZE
    A = np.zeros((H,W), np.float32)
    B = np.zeros((H,W), np.float32)
    C = np.zeros((H,W), np.float32)

    # Phase scripts generally used squares; keep deterministic + fast
    draw_square(A, Ax, Ay, sizeA)
    draw_square(B, Bx, By, sizeB)
    draw_square(C, Cx, Cy, sizeC)

    E = make_edges(A, B, C)

    x = np.stack([A, B, C, E], axis=0)  # [4,H,W]
    return x


def maybe_rot90_triplet(A, B, C, p=ROT90_PROB):
    if random.random() > p:
        return A, B, C

    # rotate 90 degrees around image center
    cx = (IMG_SIZE - 1) / 2.0
    cy = (IMG_SIZE - 1) / 2.0

    def rot(P):
        x, y = P
        # shift to origin
        dx, dy = x - cx, y - cy
        # 90deg: (dx,dy)->(-dy,dx)
        rx, ry = -dy, dx
        return (int(round(rx + cx)), int(round(ry + cy)))

    return rot(A), rot(B), rot(C)


# ============================================================
# Dataset
# ============================================================
class GeomEdgesTernary64Dataset(torch.utils.data.Dataset):
    def __init__(self, n: int, seed: int = 0):
        self.n = n
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return self.n

    def _sample_centers(self):
        # Keep centers away from border by margin for sizes up to ~12
        margin = 8
        Ax = self.rng.randint(margin, IMG_SIZE - margin)
        Ay = self.rng.randint(margin, IMG_SIZE - margin)
        Bx = self.rng.randint(margin, IMG_SIZE - margin)
        By = self.rng.randint(margin, IMG_SIZE - margin)
        Cx = self.rng.randint(margin, IMG_SIZE - margin)
        Cy = self.rng.randint(margin, IMG_SIZE - margin)
        return (Ax,Ay), (Bx,By), (Cx,Cy)

    def __getitem__(self, idx):
        # Shapes / sizes
        shapeA = 0; shapeB = 0; shapeC = 0
        sizeA = int(self.rng.choice([8,10,12]))
        sizeB = int(self.rng.choice([8,10,12]))
        sizeC = int(self.rng.choice([8,10,12]))

        # Rejection sample for *some* balance: bias a bit toward between
        # (Phase2 showed between collapsing; we want more of it)
        for _ in range(200):
            A, B, C = self._sample_centers()
            A2, B2, C2 = maybe_rot90_triplet(A, B, C, p=ROT90_PROB)

            rel = ternary_label_from_centers(A2, B2, C2, sizeA, sizeB, sizeC)

            # Heuristic acceptance: upweight between samples
            if rel == REL_NAMES.index("A_between_BC"):
                if self.rng.rand() < 0.85:
                    break
            else:
                if self.rng.rand() < 0.45:
                    break

        Ax, Ay = A2
        Bx, By = B2
        Cx, Cy = C2

        img = render_scene_ABC(Ax, Ay, Bx, By, Cx, Cy, shapeA, shapeB, shapeC, sizeA, sizeB, sizeC)

        # Phase 3 targets: (u, d_perp) for projection onto BC
        u, _, _, _, d_perp, _ = proj_point_to_segment((Ax,Ay), (Bx,By), (Cx,Cy))
        u = np.float32(u)
        d_perp = np.float32(d_perp)

        # Return sizes too (kept for compatibility with your earlier pipelines)
        return (
            torch.from_numpy(img).float(),
            torch.tensor(rel, dtype=torch.long),
            torch.tensor(sizeA, dtype=torch.long),
            torch.tensor(sizeB, dtype=torch.long),
            torch.tensor(sizeC, dtype=torch.long),
            torch.tensor(u, dtype=torch.float32),
            torch.tensor(d_perp, dtype=torch.float32),
        )


# ============================================================
# Model (conv encoder/decoder + relation head + between head)
# ============================================================
class SceneModelTernaryEdges64_256(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(NUM_CH, 32, 4, 2, 1), nn.ReLU(True),   # 64->32
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True),       # 32->16
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(True),      # 16->8
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(True),     # 8->4
        )
        self.fc_mu = nn.Linear(256*4*4, LATENT_DIM)

        # Decoder
        self.fc_dec = nn.Linear(LATENT_DIM, 256*4*4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(True),  # 4->8
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(True),   # 8->16
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True),    # 16->32
            nn.ConvTranspose2d(32, NUM_CH, 4, 2, 1),               # 32->64
            nn.Sigmoid(),
        )

        # Relation head
        self.rel_head = nn.Sequential(
            nn.Linear(LATENT_DIM, 256), nn.ReLU(True),
            nn.Linear(256, len(REL_NAMES))
        )

        # Phase 3: predict (u, d_perp) from latent
        self.between_head = nn.Sequential(
            nn.Linear(LATENT_DIM, 128), nn.ReLU(True),
            nn.Linear(128, 2)
        )

    def encode(self, x):
        h = self.enc(x)
        h = h.view(h.size(0), -1)
        z = self.fc_mu(h)
        return z

    def decode(self, z):
        h = self.fc_dec(z).view(z.size(0), 256, 4, 4)
        xhat = self.dec(h)
        return xhat

    def forward(self, x):
        z = self.encode(x)
        xhat = self.decode(z)
        logits = self.rel_head(z)
        bd = self.between_head(z)  # [u_pred, dperp_pred]
        return xhat, logits, bd, z


# ============================================================
# A-scan minibatch generator (small batches)
# ============================================================
def make_ascan_minibatch(scan_name: str, n: int):
    (Bx, By), (Cx, Cy) = SCAN_CONFIGS[scan_name]
    H = W = IMG_SIZE
    sizeA, sizeB, sizeC = ASCAN_SIZES
    shapeA, shapeB, shapeC = ASCAN_SHAPES

    # random t values across the scan range
    ts = np.random.uniform(SCAN_T_MIN, SCAN_T_MAX, size=(n,)).astype(np.float32)

    imgs = []
    labels = []
    u_true = []
    dperp_true = []

    for t in ts:
        Ax = int(round(Bx + t * (Cx - Bx)))
        Ay = int(round(By + t * (Cy - By)))

        Ax = int(np.clip(Ax, ASCAN_MARGIN + sizeA, W - ASCAN_MARGIN - sizeA))
        Ay = int(np.clip(Ay, ASCAN_MARGIN + sizeA, H - ASCAN_MARGIN - sizeA))

        A2, B2, C2 = maybe_rot90_triplet((Ax,Ay), (Bx,By), (Cx,Cy), p=1.0)

        rel = ternary_label_from_centers(A2, B2, C2, sizeA, sizeB, sizeC)
        img = render_scene_ABC(A2[0],A2[1], B2[0],B2[1], C2[0],C2[1],
                               shapeA,shapeB,shapeC, sizeA,sizeB,sizeC)
        u, _, _, _, dperp, _ = proj_point_to_segment(A2, B2, C2)

        imgs.append(img)
        labels.append(rel)
        u_true.append(u)
        dperp_true.append(dperp)

    x = torch.from_numpy(np.stack(imgs, axis=0)).float()
    y = torch.tensor(labels, dtype=torch.long)
    u = torch.tensor(u_true, dtype=torch.float32)
    d = torch.tensor(dperp_true, dtype=torch.float32)
    return x, y, u, d


# ============================================================
# Train
# ============================================================
def main():
    print(f"[train64-ternary-phase3-256] Using device: {DEVICE}")
    print(f"[train64-ternary-phase3-256] Labels: {REL_NAMES}")
    print(f"[train64-ternary-phase3-256] Tols: BETWEEN_LINE={BETWEEN_LINE_TOL}, END_MARGIN={BETWEEN_END_MARGIN}, CLOSER={CLOSER_TOL}, OVERLAP={OVERLAP_TOL}")
    print(f"[train64-ternary-phase3-256] Rotation90 prob = {ROT90_PROB}")
    print(f"[train64-ternary-phase3-256] A-scan minibatch = {USE_ASCAN_MINIBATCH}")

    train_ds = GeomEdgesTernary64Dataset(TRAIN_N, seed=0)
    val_ds   = GeomEdgesTernary64Dataset(VAL_N, seed=1)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl   = torch.utils.data.DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SceneModelTernaryEdges64_256().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = 1e9

    between_idx = REL_NAMES.index("A_between_BC")

    def step_batch(imgs, rel, u_true, dperp_true, train: bool):
        if train:
            model.train()
        else:
            model.eval()

        imgs = imgs.to(DEVICE)
        rel  = rel.to(DEVICE)
        u_true = u_true.to(DEVICE)
        dperp_true = dperp_true.to(DEVICE)

        with torch.set_grad_enabled(train):
            xhat, logits, bd, z = model(imgs)

            # Reconstruction (edges are important: give edges a bit more weight)
            # BCE over all channels is fine; you can tune weights later.
            recon = F.binary_cross_entropy(xhat, imgs)

            # Relation classification
            cls = F.cross_entropy(logits, rel)

            # Phase 3: between regression on BETWEEN samples only
            u_pred = bd[:, 0]
            d_pred = bd[:, 1]

            m_between = (rel == between_idx).float()
            # clamp targets to keep gradients sane
            u_t = torch.clamp(u_true, -0.5, 1.5)
            d_t = torch.clamp(dperp_true, 0.0, 20.0)

            Lu = (m_between * (u_pred - u_t).abs()).sum() / (m_between.sum() + 1e-6)
            Ld = (m_between * (d_pred - d_t).abs()).sum() / (m_between.sum() + 1e-6)

            # Total loss (conservative weights)
            loss = recon + cls + 0.50*Lu + 0.10*Ld

            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        return float(loss.detach().cpu()), float(recon.detach().cpu()), float(cls.detach().cpu()), float(Lu.detach().cpu()), float(Ld.detach().cpu())

    for ep in range(1, EPOCHS+1):
        # ---- train ----
        tl = 0.0
        steps = 0
        for batch in train_dl:
            imgs, rel, sA, sB, sC, u_true, dperp_true = batch
            loss, recon, cls, Lu, Ld = step_batch(imgs, rel, u_true, dperp_true, train=True)
            tl += loss
            steps += 1

            # Optional A-scan minibatch (hardens manifold + between geometry)
            if USE_ASCAN_MINIBATCH and (steps % ASCAN_EVERY_N_STEPS == 0):
                # mix a few scan names
                for name in SCAN_CONFIGS.keys():
                    x_s, y_s, u_s, d_s = make_ascan_minibatch(name, ASCAN_PER_NAME)
                    _ = step_batch(x_s, y_s, u_s, d_s, train=True)

        tl /= max(1, steps)

        # ---- val ----
        model.eval()
        vl = 0.0
        vsteps = 0
        with torch.no_grad():
            for batch in val_dl:
                imgs, rel, sA, sB, sC, u_true, dperp_true = batch
                loss, recon, cls, Lu, Ld = step_batch(imgs, rel, u_true, dperp_true, train=False)
                vl += loss
                vsteps += 1
        vl /= max(1, vsteps)

        print(f"[train64-ternary-phase3-256] Epoch {ep:3d}/{EPOCHS} | train={tl:.4f} | val={vl:.4f}")

        # save best
        if vl < best_val:
            best_val = vl
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": ep,
                "val_loss": vl,
                "config": {
                    "IMG_SIZE": IMG_SIZE,
                    "NUM_CH": NUM_CH,
                    "LATENT_DIM": LATENT_DIM,
                    "REL_NAMES": REL_NAMES,
                    "BETWEEN_LINE_TOL": BETWEEN_LINE_TOL,
                    "BETWEEN_END_MARGIN": BETWEEN_END_MARGIN,
                    "CLOSER_TOL": CLOSER_TOL,
                    "OVERLAP_TOL": OVERLAP_TOL,
                }
            }, CKPT_PATH)

    print(f"[train64-ternary-phase3-256] Saved -> {CKPT_PATH}")


if __name__ == "__main__":
    main()
