#!/usr/bin/env python3
# geomlang_relation_cnn_baseline.py
#
# Pixel-space baseline for relation classification.
# Trains a small convnet directly on 64x64x3 images from GeomEdges64Dataset
# and compares performance as we vary the number of labeled training samples.
#
# Fractions of the train set used: [0.1, 0.25, 0.5, 1.0]
# Train/test split is 80/20, fixed once per run.

import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from sklearn.metrics import confusion_matrix

from geomlang_global_coords_latent256 import (
    GeomEdges64Dataset,
    REL_LEFT, REL_RIGHT, REL_ABOVE, REL_BELOW, REL_OVERLAP, REL_NAMES,
)

TAG        = "[cnnRel]"
IMG_SIZE   = 64
NUM_CH     = 3
N_CLASSES  = 5

N_SAMPLES  = 6000
TRAIN_FRAC = 0.8          # 80% train, 20% test
BATCH_SIZE = 64
EPOCHS_FULL = 10          # epochs for the largest (100%) run
LR         = 1e-3

# Fractions of the *train* split to test sample-efficiency
TRAIN_FRACTIONS = [0.10, 0.25, 0.50, 1.00]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# Simple convnet for 64x64x3 -> 5 classes
# ---------------------------------------------------------------------
class RelationCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 64x64
        self.conv1 = nn.Conv2d(NUM_CH, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(2, 2)   # halves H,W each time

        # 64x64 -> pool -> 32x32 -> pool -> 16x16 -> pool -> 8x8
        # channels: 3 -> 32 -> 64 -> 128
        feat_dim = 128 * 8 * 8

        self.fc1 = nn.Linear(feat_dim, 256)
        self.fc2 = nn.Linear(256, N_CLASSES)

    def forward(self, x):
        # x: [B,3,64,64]
        x = self.pool(F.relu(self.conv1(x)))  # [B,32,32,32]
        x = self.pool(F.relu(self.conv2(x)))  # [B,64,16,16]
        x = self.pool(F.relu(self.conv3(x)))  # [B,128,8,8]
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

# ---------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------
def make_datasets(n_samples=N_SAMPLES, train_frac=TRAIN_FRAC, seed=0):
    """
    Returns train_dataset, test_dataset, train_indices, test_indices
    where the underlying base dataset is GeomEdges64Dataset(n_samples).
    """
    print(f"{TAG} Building GeomEdges64Dataset with N={n_samples}")
    base_ds = GeomEdges64Dataset(n_samples)

    rng = np.random.RandomState(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    n_train = int(train_frac * n_samples)
    train_idx = indices[:n_train]
    test_idx  = indices[n_train:]

    train_ds = Subset(base_ds, train_idx.tolist())
    test_ds  = Subset(base_ds, test_idx.tolist())

    print(f"{TAG} Split: train={len(train_ds)}, test={len(test_ds)}")
    return base_ds, train_ds, test_ds, train_idx, test_idx

def collate_rel_only(batch):
    """
    batch is list of (img, rel, scale, shape_r, shape_b).
    We keep only img and rel.
    """
    imgs, rels = [], []
    for img, rel, scale, s_r, s_b in batch:
        imgs.append(img)
        rels.append(rel)
    imgs = torch.stack(imgs, dim=0)          # [B,3,64,64]
    rels = torch.stack(rels, dim=0)          # [B]
    return imgs, rels

# ---------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------
def train_one_model(train_ds, test_ds, epochs, frac_label):
    """
    Train a RelationCNN on given train/test subsets.
    Returns train_acc, test_acc, confusion (for test set).
    """
    model = RelationCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, collate_fn=collate_rel_only
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=collate_rel_only
    )

    print(f"{TAG} === Training CNN with train_fraction={frac_label:.2f}, "
          f"train_size={len(train_ds)}, epochs={epochs} ===")

    for ep in range(1, epochs + 1):
        model.train()
        total, correct, running_loss = 0, 0, 0.0
        for imgs, rels in train_loader:
            imgs = imgs.to(DEVICE)
            rels = rels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, rels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == rels).sum().item()
            total += imgs.size(0)

        train_acc = correct / total * 100.0
        avg_loss = running_loss / total
        print(f"{TAG}  Epoch {ep:02d} | loss={avg_loss:.4f} | "
              f"train_acc={train_acc:.2f}%")

    # final train accuracy
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for imgs, rels in train_loader:
            imgs = imgs.to(DEVICE)
            rels = rels.to(DEVICE)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            correct += (preds == rels).sum().item()
            total += imgs.size(0)
    final_train_acc = correct / total * 100.0

    # test accuracy + confusion
    total, correct = 0, 0
    all_true, all_pred = [], []
    with torch.no_grad():
        for imgs, rels in test_loader:
            imgs = imgs.to(DEVICE)
            rels = rels.to(DEVICE)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            correct += (preds == rels).sum().item()
            total += imgs.size(0)
            all_true.append(rels.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
    final_test_acc = correct / total * 100.0

    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)
    cm = confusion_matrix(all_true, all_pred, labels=list(range(N_CLASSES)))

    return final_train_acc, final_test_acc, cm

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print(f"{TAG} Using device: {DEVICE}")
    torch.manual_seed(0)
    np.random.seed(0)

    base_ds, train_ds_full, test_ds, train_idx, test_idx = make_datasets()

    # use the same test set for all fractions
    results = []

    for frac in TRAIN_FRACTIONS:
        n_train_full = len(train_ds_full)
        n_sub = max(64, int(n_train_full * frac))  # at least some batches
        sub_indices = train_idx[:n_sub]
        train_sub = Subset(base_ds, sub_indices.tolist())

        # scale epochs a bit so tiny subsets don't overfit too insanely fast
        # (simple heuristic: fewer epochs for full data, more for tiny)
        if frac >= 1.0:
            epochs = EPOCHS_FULL
        elif frac >= 0.5:
            epochs = EPOCHS_FULL + 2
        elif frac >= 0.25:
            epochs = EPOCHS_FULL + 4
        else:
            epochs = EPOCHS_FULL + 6

        train_acc, test_acc, cm = train_one_model(
            train_sub, test_ds, epochs, frac
        )
        results.append((frac, len(train_sub), train_acc, test_acc, cm))

        print(f"{TAG} === Fraction={frac:.2f}, "
              f"train_size={len(train_sub)} DONE ===")
        print(f"{TAG}   Final train acc = {train_acc:.2f}%")
        print(f"{TAG}   Final test  acc = {test_acc:.2f}%")
        print()

    print("\n" + "=" * 70)
    print(f"{TAG} Summary (single convnet trained fresh per fraction)")
    print("=" * 70)
    print(" frac  | train_size | train_acc(%) | test_acc(%)")
    print("-------+------------+-------------+------------")
    for frac, n_tr, tr_acc, te_acc, _ in results:
        print(f" {frac:4.2f} | {n_tr:10d} | {tr_acc:11.2f} | {te_acc:10.2f}")

    # Print confusion matrix only for the largest fraction (1.0)
    full_entry = [r for r in results if abs(r[0] - 1.0) < 1e-6][0]
    _, _, _, full_test_acc, cm_full = full_entry
    print("\n" + "=" * 70)
    print(f"{TAG} Confusion matrix for frac=1.00 (test_acc={full_test_acc:.2f}%)")
    print("rows=true, cols=pred:")
    header = "         " + " ".join(f"{name:>8}" for name in REL_NAMES)
    print(header)
    for i, row in enumerate(cm_full):
        label = f"{REL_NAMES[i]:>8}"
        row_str = " ".join(f"{v:8d}" for v in row)
        print(f"{label} {row_str}")

if __name__ == "__main__":
    main()
