#!/usr/bin/env python3
# geomlang_eval_suite256.py
#
# Evaluation suite for geomlang_edges_relscale_train64_latent256.py models.
# - Loads trained checkpoint
# - Evaluates relation accuracy + confusion + per-class F1 on a deterministic dataset
# - Runs a dense dx/dy grid test with ground-truth labels (matches your rule)
# - NEW (5.4): Cross-manifold transitions in latent space (centroid-to-centroid)
#   * left_of <-> right_of
#   * above <-> below
#   * overlap -> left_of / above (sanity)
# - Saves PNGs + per-transition JSON reports + an updated JSON summary

import os, json, math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Optional: montage grids if torchvision is installed
try:
    from torchvision.utils import make_grid, save_image
    _HAS_TV = True
except Exception:
    _HAS_TV = False

from geomlang_edges_relscale_train64_latent256 import (
    IMG_SIZE, NUM_CH, LATENT_DIM,
    REL_LEFT, REL_RIGHT, REL_ABOVE, REL_BELOW, REL_OVERLAP,
    REL_NAMES,
    GeomEdges64Dataset,
    SceneModelEdges64_256,
)

OUT_DIR = "outputs_edges_relscale256"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TAG = "[eval256]"

CKPT_PATH = os.path.join(OUT_DIR, "scene_model_edges_relscale256.pt")

# ---------- Eval dataset settings ----------
EVAL_N = 6000
EVAL_SEED = 123
BATCH_SIZE = 256

# ---------- Grid test settings ----------
# We generate two shapes with fixed sizes and vary the BLUE center relative to RED center:
# dx = cx_b - cx_r, dy = cy_b - cy_r
GRID_N = 201           # 101/151/201 are good
DX_RANGE = (-24, 24)   # keep inside image; adjust if you want harder extrapolation
DY_RANGE = (-24, 24)
TOL = 2.0              # must match relation_from_centers tol

# Fix sizes/shapes for grid test to isolate relation geometry.
GRID_MARGIN = 6
RED_SHAPE = 0   # 0 circle, 1 square
BLUE_SHAPE = 0
RED_SIZE = 10
BLUE_SIZE = 10

# ---------- Cross-manifold transition settings (5.4) ----------
TRANSITION_STEPS = 41
TRANSITION_SAMPLES_FOR_CENTROIDS = 2048   # total samples used to estimate centroids (across dataset; counts per class will vary)
TRANSITION_SAVE_DECODED = True

# ---------- Drawing helpers (match training) ----------
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

def make_edges(red_mask, blue_mask):
    union = (red_mask > 0.5) | (blue_mask > 0.5)
    union = union.astype(np.float32)
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

def relation_from_centers(cx_r, cy_r, cx_b, cy_b, tol=TOL):
    dx = cx_b - cx_r
    dy = cy_b - cy_r
    if abs(dx) > abs(dy) + tol:
        return REL_LEFT if dx > 0 else REL_RIGHT
    elif abs(dy) > abs(dx) + tol:
        return REL_ABOVE if dy > 0 else REL_BELOW
    else:
        return REL_OVERLAP

def render_scene(cx_r, cy_r, cx_b, cy_b, shape_r, shape_b, size_r, size_b):
    H = W = IMG_SIZE
    red  = np.zeros((H, W), dtype=np.float32)
    blue = np.zeros((H, W), dtype=np.float32)

    if shape_r == 0:
        draw_circle(red, cx_r, cy_r, size_r)
    else:
        draw_square(red, cx_r, cy_r, size_r)

    if shape_b == 0:
        draw_circle(blue, cx_b, cy_b, size_b)
    else:
        draw_square(blue, cx_b, cy_b, size_b)

    edges = make_edges(red, blue)
    img = np.stack([red, blue, edges], axis=0)  # [3,H,W]
    return img

# ---------- Metrics ----------
def confusion_matrix(y_true, y_pred, n_classes=5):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def f1_per_class(cm):
    # cm rows = true, cols = pred
    f1 = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = (2*tp + fp + fn)
        f1_c = (2*tp / denom) if denom > 0 else 0.0
        f1.append(float(f1_c))
    return f1

def save_confusion_png(cm, out_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, origin="upper")
    plt.colorbar(label="count")
    plt.xticks(range(5), REL_NAMES, rotation=45, ha="right")
    plt.yticks(range(5), REL_NAMES)
    plt.xlabel("pred")
    plt.ylabel("true")
    plt.title("Relation Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_grid_png(arr, out_path, title, is_labels=False):
    plt.figure(figsize=(6, 6))
    if is_labels:
        cmap = plt.get_cmap("tab10", 5)
        plt.imshow(arr, origin="lower", cmap=cmap, vmin=-0.5, vmax=4.5)
        plt.colorbar(ticks=range(5), label="relation")
    else:
        plt.imshow(arr, origin="lower")
        plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# ---------- Cross-manifold helpers ----------
@torch.no_grad()
def _classify_from_imgs(model, x_bchw: torch.Tensor):
    """
    x_bchw: [B,3,H,W] float tensor on DEVICE
    returns: (pred_np[B], conf_np[B])
    """
    z = model.encode(x_bchw)
    logits = model.rel_head(z)
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    return pred.detach().cpu().numpy(), conf.detach().cpu().numpy()

@torch.no_grad()
def _estimate_class_centroids(model, n_use: int):
    """
    Uses deterministic dataset (EVAL_SEED) to estimate z-centroids per relation class.
    n_use is the total number of samples to consume from the eval dataset.
    Returns: centroids dict {class_id: np.array([LATENT_DIM])}, counts dict.
    """
    ds = GeomEdges64Dataset(EVAL_N, seed=EVAL_SEED)
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    sums = {c: np.zeros((LATENT_DIM,), dtype=np.float64) for c in range(5)}
    cnts = {c: 0 for c in range(5)}
    used = 0

    for imgs, rel, *_ in dl:
        imgs = imgs.to(DEVICE)
        rel_np = rel.numpy().astype(np.int64)
        z = model.encode(imgs).detach().cpu().numpy()  # [B, D]

        for i in range(z.shape[0]):
            c = int(rel_np[i])
            sums[c] += z[i].astype(np.float64)
            cnts[c] += 1
            used += 1
            if used >= n_use:
                break
        if used >= n_use:
            break

    centroids = {}
    for c in range(5):
        if cnts[c] == 0:
            raise RuntimeError(f"{TAG} centroid estimate failed: class {c} had 0 samples")
        centroids[c] = (sums[c] / float(cnts[c])).astype(np.float32)

    return centroids, cnts

def _linspace_alphas(n: int):
    return np.linspace(0.0, 1.0, n, dtype=np.float32)

@torch.no_grad()
def run_latent_transition(model, z_a: np.ndarray, z_b: np.ndarray, name: str, out_dir: str):
    """
    Interpolates z_a -> z_b, decodes each step, reclassifies decoded image.
    Saves montage + json report.
    """
    alphas = _linspace_alphas(TRANSITION_STEPS)
    Z = (1.0 - alphas[:, None]) * z_a[None, :] + alphas[:, None] * z_b[None, :]  # [T,D]
    z_t = torch.from_numpy(Z).to(DEVICE)

    # Decode
    x = model.decode(z_t).clamp(0, 1)  # [T,3,H,W]

    # Reclassify from decoded images (key sanity check)
    pred, conf = _classify_from_imgs(model, x)

    # Find first index where prediction differs from pred[0]
    start = int(pred[0])
    first_change = None
    for i in range(len(pred)):
        if int(pred[i]) != start:
            first_change = i
            break

    montage_path = os.path.join(out_dir, f"eval256_transition_{name}.png")
    if TRANSITION_SAVE_DECODED:
        if _HAS_TV:
            grid = make_grid(x.detach().cpu(), nrow=min(TRANSITION_STEPS, 12), padding=2)
            save_image(grid, montage_path)
        else:
            # Fallback: vertical strip of RGB frames
            x_np = x.detach().cpu().numpy()      # [T,3,H,W]
            rgb = np.transpose(x_np, (0, 2, 3, 1))  # [T,H,W,3]
            big = np.concatenate(list(rgb), axis=0)  # [T*H, W, 3]
            plt.figure(figsize=(6, 2 + 0.06 * big.shape[0]))
            plt.imshow(big)
            plt.axis("off")
            plt.title(f"transition {name} (vertical strip)")
            plt.tight_layout()
            plt.savefig(montage_path, dpi=200)
            plt.close()

    report = {
        "name": name,
        "steps": int(TRANSITION_STEPS),
        "start_pred": int(pred[0]),
        "end_pred": int(pred[-1]),
        "first_change_idx": None if first_change is None else int(first_change),
        "pred_labels": [int(x) for x in pred.tolist()],
        "conf": [float(x) for x in conf.tolist()],
        "mean_conf": float(np.mean(conf)),
        "min_conf": float(np.min(conf)),
    }
    report_path = os.path.join(out_dir, f"eval256_transition_{name}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return montage_path, report_path, report

# ---------- Core evaluation ----------
@torch.no_grad()
def eval_on_dataset(model):
    ds = GeomEdges64Dataset(EVAL_N, seed=EVAL_SEED)
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    y_true = []
    y_pred = []
    confs = []

    for imgs, rel, scale, s_r, s_b in dl:
        imgs = imgs.to(DEVICE)
        rel = rel.numpy()

        z = model.encode(imgs)
        logits = model.rel_head(z)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

        y_true.append(rel)
        y_pred.append(pred.cpu().numpy())
        confs.append(conf.cpu().numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    confs  = np.concatenate(confs,  axis=0)

    acc = float((y_true == y_pred).mean())
    cm = confusion_matrix(y_true, y_pred, n_classes=5)
    f1 = f1_per_class(cm)

    return {
        "acc": acc,
        "cm": cm,
        "f1_per_class": f1,
        "mean_conf": float(confs.mean()),
        "min_conf": float(confs.min()),
        "p05_conf": float(np.quantile(confs, 0.05)),
    }

@torch.no_grad()
def eval_dxdy_grid(model):
    H = W = IMG_SIZE

    # choose a safe central anchor for red so blue can move around without leaving frame
    cx_r = W // 2
    cy_r = H // 2

    # ensure our fixed sizes won't clip with the chosen margin
    max_s = max(RED_SIZE, BLUE_SIZE)
    if not (GRID_MARGIN + max_s <= cx_r <= W - GRID_MARGIN - max_s):
        raise RuntimeError("Red center too close to boundary for chosen sizes/margins.")
    if not (GRID_MARGIN + max_s <= cy_r <= H - GRID_MARGIN - max_s):
        raise RuntimeError("Red center too close to boundary for chosen sizes/margins.")

    dx_vals = np.linspace(DX_RANGE[0], DX_RANGE[1], GRID_N)
    dy_vals = np.linspace(DY_RANGE[0], DY_RANGE[1], GRID_N)

    labels_true = np.zeros((GRID_N, GRID_N), dtype=np.int64)
    labels_pred = np.zeros((GRID_N, GRID_N), dtype=np.int64)
    conf_grid   = np.zeros((GRID_N, GRID_N), dtype=np.float32)

    batch_imgs = []
    batch_pos = []

    def flush_batch():
        if not batch_imgs:
            return
        x = torch.from_numpy(np.stack(batch_imgs, axis=0)).float().to(DEVICE)  # [B,3,H,W]
        z = model.encode(x)
        logits = model.rel_head(z)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        pred = pred.cpu().numpy()
        conf = conf.cpu().numpy()
        for (iy, ix), p, c in zip(batch_pos, pred, conf):
            labels_pred[iy, ix] = int(p)
            conf_grid[iy, ix]   = float(c)
        batch_imgs.clear()
        batch_pos.clear()

    for iy, dy in enumerate(dy_vals):
        for ix, dx in enumerate(dx_vals):
            cx_b = int(round(cx_r + dx))
            cy_b = int(round(cy_r + dy))

            # Clip blue center so it stays drawable given size/margins
            cx_b = int(np.clip(cx_b, GRID_MARGIN + BLUE_SIZE, W - GRID_MARGIN - BLUE_SIZE))
            cy_b = int(np.clip(cy_b, GRID_MARGIN + BLUE_SIZE, H - GRID_MARGIN - BLUE_SIZE))

            t = relation_from_centers(cx_r, cy_r, cx_b, cy_b, tol=TOL)
            labels_true[iy, ix] = int(t)

            img = render_scene(cx_r, cy_r, cx_b, cy_b, RED_SHAPE, BLUE_SHAPE, RED_SIZE, BLUE_SIZE)
            batch_imgs.append(img)
            batch_pos.append((iy, ix))

            if len(batch_imgs) >= BATCH_SIZE:
                flush_batch()

    flush_batch()

    acc_grid = (labels_true == labels_pred).astype(np.float32)
    acc = float(acc_grid.mean())

    return {
        "dx_vals": dx_vals,
        "dy_vals": dy_vals,
        "labels_true": labels_true,
        "labels_pred": labels_pred,
        "conf_grid": conf_grid,
        "acc_grid": acc_grid,
        "acc": acc,
    }

def main():
    print(f"{TAG} device = {DEVICE}")

    # ---- Load model ----
    print(f"{TAG} loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    if "model_state_dict" not in ckpt:
        raise RuntimeError(f"{TAG} checkpoint missing model_state_dict")

    model = SceneModelEdges64_256().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ---- Eval on deterministic dataset ----
    print(f"{TAG} running dataset eval (N={EVAL_N}, seed={EVAL_SEED})")
    ds_metrics = eval_on_dataset(model)

    cm_path = os.path.join(OUT_DIR, "eval256_confusion_matrix.png")
    save_confusion_png(ds_metrics["cm"], cm_path)
    print(f"{TAG} saved: {cm_path}")

    # ---- Eval on dx/dy grid ----
    print(f"{TAG} running dx/dy grid eval (GRID_N={GRID_N}, dx={DX_RANGE}, dy={DY_RANGE}, tol={TOL})")
    grid = eval_dxdy_grid(model)

    true_path = os.path.join(OUT_DIR, f"eval256_dxdy_true_labels_N{GRID_N}.png")
    pred_path = os.path.join(OUT_DIR, f"eval256_dxdy_pred_labels_N{GRID_N}.png")
    conf_path = os.path.join(OUT_DIR, f"eval256_dxdy_conf_N{GRID_N}.png")
    acc_path  = os.path.join(OUT_DIR, f"eval256_dxdy_acc_N{GRID_N}.png")

    save_grid_png(grid["labels_true"], true_path, "dx/dy grid: ground-truth labels", is_labels=True)
    save_grid_png(grid["labels_pred"], pred_path, "dx/dy grid: predicted labels", is_labels=True)
    save_grid_png(grid["conf_grid"],   conf_path,  "dx/dy grid: confidence (max softmax)", is_labels=False)
    save_grid_png(grid["acc_grid"],    acc_path,   f"dx/dy grid: accuracy (mean={grid['acc']:.4f})", is_labels=False)

    print(f"{TAG} saved: {true_path}")
    print(f"{TAG} saved: {pred_path}")
    print(f"{TAG} saved: {conf_path}")
    print(f"{TAG} saved: {acc_path}")

    # ---- Base summary JSON ----
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
            "confusion_matrix": ds_metrics["cm"].tolist(),
        },
        "dxdy_grid_eval": {
            "GRID_N": GRID_N,
            "DX_RANGE": DX_RANGE,
            "DY_RANGE": DY_RANGE,
            "TOL": TOL,
            "fixed_shapes": {"red": RED_SHAPE, "blue": BLUE_SHAPE},
            "fixed_sizes": {"red": RED_SIZE, "blue": BLUE_SIZE},
            "acc": grid["acc"],
        }
    }

    summary_path = os.path.join(OUT_DIR, "eval256_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"{TAG} summary saved: {summary_path}")

    # ---- Cross-manifold transitions (5.4) ----
    print(f"{TAG} estimating class centroids (use_total={TRANSITION_SAMPLES_FOR_CENTROIDS}, seed={EVAL_SEED})")
    centroids, cnts = _estimate_class_centroids(model, n_use=TRANSITION_SAMPLES_FOR_CENTROIDS)
    print(f"{TAG} centroid counts used: {cnts}")

    transitions = [
        ("left_to_right",  REL_LEFT,  REL_RIGHT),
        ("right_to_left",  REL_RIGHT, REL_LEFT),
        ("above_to_below", REL_ABOVE, REL_BELOW),
        ("below_to_above", REL_BELOW, REL_ABOVE),
        ("overlap_to_left",  REL_OVERLAP, REL_LEFT),
        ("overlap_to_above", REL_OVERLAP, REL_ABOVE),
    ]

    trans_reports = {}
    for name, a, b in transitions:
        print(f"{TAG} transition: {REL_NAMES[a]} -> {REL_NAMES[b]} ({name})")
        montage_path, report_path, rep = run_latent_transition(
            model, centroids[a], centroids[b], name=name, out_dir=OUT_DIR
        )
        print(f"{TAG} saved: {montage_path}")
        print(f"{TAG} saved: {report_path}")
        trans_reports[name] = rep

    # Update summary with transitions and overwrite summary file
    summary["transitions"] = {
        "steps": int(TRANSITION_STEPS),
        "centroid_estimation": {
            "use_total_samples": int(TRANSITION_SAMPLES_FOR_CENTROIDS),
            "counts_used": {str(k): int(v) for k, v in cnts.items()},
        },
        "reports": {
            k: {
                "start_pred": v["start_pred"],
                "end_pred": v["end_pred"],
                "first_change_idx": v["first_change_idx"],
                "mean_conf": v["mean_conf"],
                "min_conf": v["min_conf"],
            } for k, v in trans_reports.items()
        }
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"{TAG} summary updated with transitions: {summary_path}")

    # ---- Console highlights ----
    print(f"{TAG} DATASET acc = {ds_metrics['acc']:.4f}")
    print(f"{TAG} DATASET f1 per class = {[round(x,4) for x in ds_metrics['f1_per_class']]}")
    print(f"{TAG} GRID acc = {grid['acc']:.4f}")

if __name__ == "__main__":
    main()
