#!/usr/bin/env python3
# geomlang_edges_relgraph3_train64_latent256.py
#
# 3-object version of your edges+relscale trainer.
# - 64x64 synthetic scenes
# - Channels: obj0, obj1, obj2, edges  => NUM_CH = 4
# - Predicts a relation graph: (0,1), (0,2), (1,2) each in 5 classes
# - Keeps a legacy rel_head for edge(0,1) so older probing scripts can still work if needed.
#
# Saves checkpoint to:
#   outputs_edges_relgraph3_256/scene_model_edges_relgraph3_256.pt

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------
# Config
# -----------------------
IMG_SIZE   = 64
NUM_CH     = 4          # obj0, obj1, obj2, edges
LATENT_DIM = 256

N_TRAIN    = 24000
N_VAL      = 6000
BATCH_SIZE = 128
N_EPOCHS   = 40
LR         = 1e-3

OUT_DIR         = "outputs_edges_relgraph3_256"
CKPT_SCENEMODEL = os.path.join(OUT_DIR, "scene_model_edges_relgraph3_256.pt")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REL_LEFT, REL_RIGHT, REL_ABOVE, REL_BELOW, REL_OVERLAP = range(5)
REL_NAMES = ["left_of", "right_of", "above", "below", "overlap"]

# Edge ordering for the relation graph:
# y_graph[0] = rel(0,1), y_graph[1] = rel(0,2), y_graph[2] = rel(1,2)
EDGE_PAIRS = [(0, 1), (0, 2), (1, 2)]
N_EDGES = len(EDGE_PAIRS)

# -----------------------
# Shape drawing / scene generation
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
    # edges of union over all objects
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

def relation_from_centers(cx_a, cy_a, cx_b, cy_b, tol=2.0):
    # dx,dy defined as (b - a) like your original rule
    dx = cx_b - cx_a
    dy = cy_b - cy_a

    if abs(dx) > abs(dy) + tol:
        return REL_LEFT if dx > 0 else REL_RIGHT
    elif abs(dy) > abs(dx) + tol:
        return REL_ABOVE if dy > 0 else REL_BELOW
    else:
        return REL_OVERLAP

def scale_class_from_size(s):
    if s <= 7:
        return 0
    elif s <= 11:
        return 1
    else:
        return 2

class GeomEdges64Graph3Dataset(Dataset):
    """
    Returns:
      img:        float [4,64,64]  (obj0, obj1, obj2, edges)
      rel_graph:  long  [3]        relations for (0,1), (0,2), (1,2)
      scale:      long  scalar     coarse overall size class (mean of sizes)
      shapes:     long  [3]        0 circle / 1 square per object
    """
    def __init__(self, n_samples, seed=None, tol=2.0, margin=6):
        super().__init__()
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)
        self.tol = float(tol)
        self.margin = int(margin)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        H = W = IMG_SIZE
        rng = self.rng
        margin = self.margin

        # 3 objects, each circle or square
        shapes = [int(rng.integers(0, 2)) for _ in range(3)]

        # sizes: base coupled (keeps scale coherence) + per-object jitter
        base = int(rng.integers(5, 13))  # 5..12
        sizes = [
            int(np.clip(base + int(rng.integers(-2, 3)), 4, 14))
            for _ in range(3)
        ]

        def sample_center(s):
            cx = int(rng.integers(margin + s, W - margin - s))
            cy = int(rng.integers(margin + s, H - margin - s))
            return cx, cy

        centers = [sample_center(sizes[i]) for i in range(3)]
        (cx0, cy0), (cx1, cy1), (cx2, cy2) = centers

        obj0 = np.zeros((H, W), dtype=np.float32)
        obj1 = np.zeros((H, W), dtype=np.float32)
        obj2 = np.zeros((H, W), dtype=np.float32)
        objs = [obj0, obj1, obj2]

        for i in range(3):
            cx, cy = centers[i]
            s = sizes[i]
            if shapes[i] == 0:
                draw_circle(objs[i], cx, cy, s)
            else:
                draw_square(objs[i], cx, cy, s)

        edges = make_edges(obj0, obj1, obj2)

        img = np.stack([obj0, obj1, obj2, edges], axis=0)  # [4,64,64]

        # Relation graph
        rels = []
        for a, b in EDGE_PAIRS:
            cxa, cya = centers[a]
            cxb, cyb = centers[b]
            rels.append(relation_from_centers(cxa, cya, cxb, cyb, tol=self.tol))
        rel_graph = np.array(rels, dtype=np.int64)  # [3]

        scale = scale_class_from_size(float(np.mean(sizes)))

        return (
            torch.from_numpy(img),
            torch.from_numpy(rel_graph).long(),
            torch.tensor(scale, dtype=torch.long),
            torch.tensor(shapes, dtype=torch.long),
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
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # 4x4
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z

class Decoder(nn.Module):
    def __init__(self, out_channels=NUM_CH, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 8
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   # 16
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)    # 32
        self.deconv4 = nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1)  # 64

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, 4, 4)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))
        return x

class SceneModelEdges64Graph3_256(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(in_channels=NUM_CH, latent_dim=LATENT_DIM)
        self.decoder = Decoder(out_channels=NUM_CH, latent_dim=LATENT_DIM)

        # Graph head: 3 edges * 5 classes = 15 logits
        self.rel_graph_head = nn.Linear(LATENT_DIM, N_EDGES * 5)

        # Keep a legacy single rel head for edge(0,1) if you want old style probes
        self.rel_head = nn.Linear(LATENT_DIM, 5)

        # Optional auxiliary heads (keep the spirit of your original setup)
        self.scale_head  = nn.Linear(LATENT_DIM, 3)
        self.shape0_head = nn.Linear(LATENT_DIM, 2)
        self.shape1_head = nn.Linear(LATENT_DIM, 2)
        self.shape2_head = nn.Linear(LATENT_DIM, 2)

    def forward(self, x):
        z = self.encoder(x)
        rec = self.decoder(z)

        rel_graph_logits = self.rel_graph_head(z).view(-1, N_EDGES, 5)  # [B,3,5]
        rel01_logits = self.rel_head(z)

        scale_logits = self.scale_head(z)
        s0 = self.shape0_head(z)
        s1 = self.shape1_head(z)
        s2 = self.shape2_head(z)

        return rec, rel_graph_logits, rel01_logits, scale_logits, s0, s1, s2

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

# -----------------------
# Training
# -----------------------

def main():
    print(f"[train64-graph3-256] Using device: {DEVICE}")

    train_ds = GeomEdges64Graph3Dataset(N_TRAIN)
    val_ds   = GeomEdges64Graph3Dataset(N_VAL)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SceneModelEdges64Graph3_256().to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    bce = nn.BCELoss()
    ce  = nn.CrossEntropyLoss()

    def run_epoch(loader, train=True):
        model.train() if train else model.eval()

        total_loss = 0.0
        n_batches = 0

        with torch.set_grad_enabled(train):
            for imgs, rel_graph, scale, shapes in loader:
                imgs = imgs.to(DEVICE)                          # [B,4,64,64]
                rel_graph = rel_graph.to(DEVICE)                # [B,3]
                scale = scale.to(DEVICE)                        # [B]
                shapes = shapes.to(DEVICE)                      # [B,3]

                if train:
                    opt.zero_grad()

                rec, relg_log, rel01_log, scale_log, s0_log, s1_log, s2_log = model(imgs)

                rec_loss = bce(rec, imgs)

                # graph loss: sum CE over the 3 edges
                graph_loss = 0.0
                for e in range(N_EDGES):
                    graph_loss = graph_loss + ce(relg_log[:, e, :], rel_graph[:, e])

                # legacy edge(0,1) should match rel_graph[:,0]
                rel01_loss = ce(rel01_log, rel_graph[:, 0])

                scale_loss = ce(scale_log, scale)

                s0_loss = ce(s0_log, shapes[:, 0])
                s1_loss = ce(s1_log, shapes[:, 1])
                s2_loss = ce(s2_log, shapes[:, 2])

                cls_loss = graph_loss + rel01_loss + scale_loss + s0_loss + s1_loss + s2_loss
                loss = rec_loss + 0.25 * cls_loss

                if train:
                    loss.backward()
                    opt.step()

                total_loss += float(loss.item())
                n_batches += 1

        return total_loss / max(1, n_batches)

    for epoch in range(1, N_EPOCHS + 1):
        train_loss = run_epoch(train_dl, train=True)
        val_loss   = run_epoch(val_dl,   train=False)

        print(f"[train64-graph3-256] Epoch {epoch:3d}/{N_EPOCHS} "
              f"| train loss={train_loss:.4f} | val loss={val_loss:.4f}")

    ckpt = {"model_state_dict": model.state_dict()}
    torch.save(ckpt, CKPT_SCENEMODEL)
    print(f"[train64-graph3-256] Saved SceneModel -> {CKPT_SCENEMODEL}")

if __name__ == "__main__":
    main()
