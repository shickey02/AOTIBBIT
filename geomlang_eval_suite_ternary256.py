#!/usr/bin/env python3
# geomlang_eval_suite_ternary256.py
#
# Evaluation suite for geomlang_edges_relternary_train64_latent256.py
#
# - Loads trained checkpoint
# - Dataset eval: acc, confusion, per-class F1, confidence summary
# - A-scan heatmaps: fix (B,C) anchors, sweep A across plane
# - Saves PNGs + JSON summary into outputs_edges_relternary256/

import os, json, math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

try:
    from torchvision.utils import make_grid, save_image
    _HAS_TV = True
except Exception:
    _HAS_TV = False

# ---- Import training definitions (model + dataset + rules) ----
from geomlang_edges_relternary_train64_latent256 import (
    IMG_SIZE, NUM_CH, LATENT_DIM,
    T_BETWEEN, T_CLOSER_B, T_CLOSER_C, T_LEFT_BOTH, T_RIGHT_BOTH, T_ABOVE_BOTH, T_BELOW_BOTH,
    T_NAMES,
    TOL_AXIS, BAND_BETW,
    GeomEdgesTernary64Dataset,
    SceneModelEdgesTernary64_256,
    draw_circle, draw_square, make_edges,
    ternary_label_A_vs_BC,
)

OUT_DIR = "outputs_edges_relternary256"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TAG = "[eval-ternary256]"

CKPT_PATH = os.path.join(OUT_DIR, "scene_model_edges_relternary256.pt")

# -----------------------
# Eval dataset settings
# -----------------------
EVAL_N = 6000
EVAL_SEED = 123
BATCH_SIZE = 256

# -----------------------
# A-scan settings
# -----------------------
SCAN_N = 201              # 151/201 recommended
SCAN_MARGIN = 6           # keep drawable region safe
A_SIZE = 10
B_SIZE = 10
C_SIZE = 10
A_SHAPE = 0               # 0 circle, 1 square
B_SHAPE = 0
C_SHAPE = 0

# If true, we "clip" A to drawable bounds (keeps scan complete)
# If false, we skip points that would clip and mark them -1.
SCAN_CLIP = True

# -----------------------
# Utilities
# -----------------------
def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def f1_per_class(cm):
    # rows=true, cols=pred
    f1 = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = (2*tp + fp + fn)
        f1.append(float((2*tp / denom) if denom > 0 else 0.0))
    return f1

def save_confusion_png(cm, labels, out_path, title):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, origin="upper")
    plt.colorbar(label="count")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("pred")
    plt.ylabel("true")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_grid_png(arr, out_path, title, is_labels=False, n_classes=None):
    plt.figure(figsize=(6.5, 6.5))
    if is_labels:
        assert n_classes is not None
        cmap = plt.get_cmap("tab20", n_classes)
        plt.imshow(arr, origin="lower", cmap=cmap, vmin=-0.5, vmax=(n_classes - 0.5))
        plt.colorbar(ticks=list(range(n_classes)), label="class")
    else:
        plt.imshow(arr, origin="lower")
        plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

@torch.no_grad()
def classify_batch(model, x_bchw):
    """
    x_bchw: float tensor [B,NUM_CH,H,W] on DEVICE
    returns: pred_np[B], conf_np[B], probs_np[B,C]
    """
    z = model.encode(x_bchw)
    logits = model.ternary_head(z)
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    return (pred.detach().cpu().numpy(),
            conf.detach().cpu().numpy(),
            probs.detach().cpu().numpy())

def render_scene_ABC(cxA, cyA, cxB, cyB, cxC, cyC,
                     shapeA, shapeB, shapeC,
                     sA, sB, sC):
    H = W = IMG_SIZE
    A = np.zeros((H, W), dtype=np.float32)
    B = np.zeros((H, W), dtype=np.float32)
    C = np.zeros((H, W), dtype=np.float32)

    if shapeA == 0: draw_circle(A, cxA, cyA, sA)
    else:           draw_square(A, cxA, cyA, sA)

    if shapeB == 0: draw_circle(B, cxB, cyB, sB)
    else:           draw_square(B, cxB, cyB, sB)

    if shapeC == 0: draw_circle(C, cxC, cyC, sC)
    else:           draw_square(C, cxC, cyC, sC)

    E = make_edges(A, B, C)
    img = np.stack([A, B, C, E], axis=0)  # [4,H,W]
    return img

# -----------------------
# Dataset evaluation
# -----------------------
@torch.no_grad()
def eval_on_dataset(model):
    ds = GeomEdgesTernary64Dataset(EVAL_N, seed=EVAL_SEED)
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    y_true = []
    y_pred = []
    confs = []

    # optional: also capture "hardness" via entropy
    entropies = []

    for imgs, t, sA, sB, sC in dl:
        imgs = imgs.to(DEVICE)
        t_np = t.numpy()

        pred_np, conf_np, probs_np = classify_batch(model, imgs)

        y_true.append(t_np)
        y_pred.append(pred_np)
        confs.append(conf_np)

        # entropy per sample: -sum p log p
        eps = 1e-9
        ent = -np.sum(probs_np * np.log(probs_np + eps), axis=1)
        entropies.append(ent)

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    confs  = np.concatenate(confs,  axis=0)
    entropies = np.concatenate(entropies, axis=0)

    acc = float((y_true == y_pred).mean())
    cm = confusion_matrix(y_true, y_pred, n_classes=len(T_NAMES))
    f1 = f1_per_class(cm)

    return {
        "acc": acc,
        "cm": cm,
        "f1_per_class": f1,
        "mean_conf": float(confs.mean()),
        "min_conf": float(confs.min()),
        "p05_conf": float(np.quantile(confs, 0.05)),
        "mean_entropy": float(entropies.mean()),
        "p95_entropy": float(np.quantile(entropies, 0.95)),
        "counts_true": np.bincount(y_true, minlength=len(T_NAMES)).tolist(),
        "counts_pred": np.bincount(y_pred, minlength=len(T_NAMES)).tolist(),
    }

# -----------------------
# A-scan evaluation (fix B,C; sweep A)
# -----------------------
@torch.no_grad()
def eval_A_scan(model, cxB, cyB, cxC, cyC, tag_name):
    H = W = IMG_SIZE

    xs = np.linspace(SCAN_MARGIN, W - 1 - SCAN_MARGIN, SCAN_N)
    ys = np.linspace(SCAN_MARGIN, H - 1 - SCAN_MARGIN, SCAN_N)

    labels_true = np.full((SCAN_N, SCAN_N), -1, dtype=np.int64)
    labels_pred = np.full((SCAN_N, SCAN_N), -1, dtype=np.int64)
    conf_grid   = np.zeros((SCAN_N, SCAN_N), dtype=np.float32)
    acc_grid    = np.zeros((SCAN_N, SCAN_N), dtype=np.float32)

    batch_imgs = []
    batch_pos  = []
    batch_true = []

    def _clip_center(cx, cy, s):
        cx = int(np.clip(cx, SCAN_MARGIN + s, W - 1 - (SCAN_MARGIN + s)))
        cy = int(np.clip(cy, SCAN_MARGIN + s, H - 1 - (SCAN_MARGIN + s)))
        return cx, cy

    # ensure B,C are drawable
    cxB2, cyB2 = _clip_center(cxB, cyB, B_SIZE)
    cxC2, cyC2 = _clip_center(cxC, cyC, C_SIZE)

    def flush():
        if not batch_imgs:
            return
        x = torch.from_numpy(np.stack(batch_imgs, axis=0)).float().to(DEVICE)
        pred_np, conf_np, _ = classify_batch(model, x)
        for (iy, ix), t, p, c in zip(batch_pos, batch_true, pred_np, conf_np):
            labels_true[iy, ix] = int(t)
            labels_pred[iy, ix] = int(p)
            conf_grid[iy, ix]   = float(c)
            acc_grid[iy, ix]    = float(1.0 if int(t) == int(p) else 0.0)
        batch_imgs.clear()
        batch_pos.clear()
        batch_true.clear()

    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            cxA = int(round(x))
            cyA = int(round(y))

            if SCAN_CLIP:
                cxA, cyA = _clip_center(cxA, cyA, A_SIZE)
            else:
                # skip if would clip (mark as -1)
                if not (SCAN_MARGIN + A_SIZE <= cxA <= W - 1 - (SCAN_MARGIN + A_SIZE) and
                        SCAN_MARGIN + A_SIZE <= cyA <= H - 1 - (SCAN_MARGIN + A_SIZE)):
                    continue

            t = ternary_label_A_vs_BC(cxA, cyA, cxB2, cyB2, cxC2, cyC2,
                                      tol_axis=TOL_AXIS, band_between=BAND_BETW)

            img = render_scene_ABC(cxA, cyA, cxB2, cyB2, cxC2, cyC2,
                                   A_SHAPE, B_SHAPE, C_SHAPE,
                                   A_SIZE, B_SIZE, C_SIZE)

            batch_imgs.append(img)
            batch_pos.append((iy, ix))
            batch_true.append(t)

            if len(batch_imgs) >= BATCH_SIZE:
                flush()

    flush()

    # compute scan acc only on valid points
    valid = (labels_true >= 0)
    scan_acc = float(acc_grid[valid].mean()) if valid.any() else 0.0

    # save images
    true_path = os.path.join(OUT_DIR, f"ternary_scan_{tag_name}_true.png")
    pred_path = os.path.join(OUT_DIR, f"ternary_scan_{tag_name}_pred.png")
    conf_path = os.path.join(OUT_DIR, f"ternary_scan_{tag_name}_conf.png")
    acc_path  = os.path.join(OUT_DIR, f"ternary_scan_{tag_name}_acc.png")

    save_grid_png(labels_true, true_path, f"A-scan true labels ({tag_name})", is_labels=True, n_classes=len(T_NAMES))
    save_grid_png(labels_pred, pred_path, f"A-scan predicted labels ({tag_name})", is_labels=True, n_classes=len(T_NAMES))
    save_grid_png(conf_grid,   conf_path, f"A-scan confidence ({tag_name})", is_labels=False)
    save_grid_png(acc_grid,    acc_path,  f"A-scan accuracy mean={scan_acc:.4f} ({tag_name})", is_labels=False)

    # optional decoded montage: sample a few points across the scan
    montage_path = None
    if _HAS_TV:
        # pick representative points: center, near B, near C, far left, far right
        picks = []
        picks.append((SCAN_N//2, SCAN_N//2))
        picks.append((SCAN_N//2, max(0, SCAN_N//2 - 45)))
        picks.append((SCAN_N//2, min(SCAN_N-1, SCAN_N//2 + 45)))
        picks.append((max(0, SCAN_N//2 - 45), SCAN_N//2))
        picks.append((min(SCAN_N-1, SCAN_N//2 + 45), SCAN_N//2))

        imgs = []
        for (iy, ix) in picks:
            # reconstruct A center from scan coordinates
            cxA = int(round(xs[ix]))
            cyA = int(round(ys[iy]))
            cxA, cyA = _clip_center(cxA, cyA, A_SIZE)

            img = render_scene_ABC(cxA, cyA, cxB2, cyB2, cxC2, cyC2,
                                   A_SHAPE, B_SHAPE, C_SHAPE,
                                   A_SIZE, B_SIZE, C_SIZE)
            imgs.append(img)

        x = torch.from_numpy(np.stack(imgs, axis=0)).float().to(DEVICE)
        z = model.encode(x)
        rec = model.decode(z).detach().cpu()

        # make a montage that shows input over recon: stack along batch dimension
        x_cpu = x.detach().cpu()
        both = torch.cat([x_cpu, rec], dim=0)  # [2B,4,H,W]
        grid = make_grid(both, nrow=len(imgs), padding=2)
        montage_path = os.path.join(OUT_DIR, f"ternary_scan_{tag_name}_montage_in_out.png")
        save_image(grid, montage_path)

    return {
        "tag": tag_name,
        "B": [int(cxB2), int(cyB2)],
        "C": [int(cxC2), int(cyC2)],
        "scan_acc": scan_acc,
        "paths": {
            "true": true_path, "pred": pred_path, "conf": conf_path, "acc": acc_path,
            "montage": montage_path
        }
    }

# -----------------------
# Main
# -----------------------
def main():
    print(f"{TAG} device = {DEVICE}")
    print(f"{TAG} loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    if "model_state_dict" not in ckpt:
        raise RuntimeError(f"{TAG} checkpoint missing model_state_dict")

    model = SceneModelEdgesTernary64_256().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ---- dataset eval ----
    print(f"{TAG} dataset eval (N={EVAL_N}, seed={EVAL_SEED})")
    ds_metrics = eval_on_dataset(model)

    cm_path = os.path.join(OUT_DIR, "ternary_confusion_matrix.png")
    save_confusion_png(ds_metrics["cm"], T_NAMES, cm_path, "Ternary (A vs B,C) Confusion Matrix")
    print(f"{TAG} saved: {cm_path}")

    # ---- A-scan evals (multiple anchor configurations) ----
    # We want several “anchor geometries” for B,C:
    # 1) horizontal bar
    # 2) vertical bar
    # 3) diagonal bar
    # 4) shorter bar (closer anchors)
    # 5) off-center anchors (translation generalization)
    H = W = IMG_SIZE
    cx0 = W // 2
    cy0 = H // 2

    configs = [
        ("horiz_mid",   (cx0 - 14, cy0), (cx0 + 14, cy0)),
        ("vert_mid",    (cx0, cy0 - 14), (cx0, cy0 + 14)),
        ("diag_mid",    (cx0 - 12, cy0 - 12), (cx0 + 12, cy0 + 12)),
        ("horiz_short", (cx0 - 9,  cy0), (cx0 + 9,  cy0)),
        ("horiz_off",   (cx0 - 14, cy0 - 10), (cx0 + 14, cy0 + 8)),
    ]

    scan_summaries = []
    for name, (cxB, cyB), (cxC, cyC) in configs:
        print(f"{TAG} A-scan: {name}  B=({cxB},{cyB}) C=({cxC},{cyC})  SCAN_N={SCAN_N}")
        scan_summaries.append(eval_A_scan(model, cxB, cyB, cxC, cyC, name))

    # ---- summary json ----
    summary = {
        "device": str(DEVICE),
        "ckpt_path": CKPT_PATH,
        "dataset_eval": {
            "N": EVAL_N,
            "seed": EVAL_SEED,
            "acc": ds_metrics["acc"],
            "f1_per_class": ds_metrics["f1_per_class"],
            "mean_conf": ds_metrics["mean_conf"],
            "min_conf": ds_metrics["min_conf"],
            "p05_conf": ds_metrics["p05_conf"],
            "mean_entropy": ds_metrics["mean_entropy"],
            "p95_entropy": ds_metrics["p95_entropy"],
            "counts_true": ds_metrics["counts_true"],
            "counts_pred": ds_metrics["counts_pred"],
            "confusion_matrix": ds_metrics["cm"].tolist(),
            "confusion_png": cm_path,
        },
        "A_scan": {
            "SCAN_N": SCAN_N,
            "SCAN_MARGIN": SCAN_MARGIN,
            "SCAN_CLIP": SCAN_CLIP,
            "sizes": {"A": A_SIZE, "B": B_SIZE, "C": C_SIZE},
            "shapes": {"A": A_SHAPE, "B": B_SHAPE, "C": C_SHAPE},
            "rule": {"TOL_AXIS": float(TOL_AXIS), "BAND_BETW": float(BAND_BETW)},
            "configs": scan_summaries,
        }
    }

    summary_path = os.path.join(OUT_DIR, "ternary_eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"{TAG} summary saved: {summary_path}")

    # ---- console highlights ----
    print(f"{TAG} DATASET acc = {ds_metrics['acc']:.4f}")
    print(f"{TAG} DATASET f1 per class = {[round(x,4) for x in ds_metrics['f1_per_class']]}")
    for s in scan_summaries:
        print(f"{TAG} SCAN {s['tag']} acc = {s['scan_acc']:.4f}")

    if not _HAS_TV:
        print(f"{TAG} torchvision not available -> skipping montage images")

if __name__ == "__main__":
    main()
