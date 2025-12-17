#!/usr/bin/env python3
# geomlang_eval_suite_ternary_phase2_256.py
#
# Phase 2 eval:
# - Dataset metrics (accuracy + classification report)
# - Confusion matrix PNGs (counts + row-normalized)
# - A-scans (horiz/vert/diag/etc) -> true/pred/conf/acc PNGs
# - Saves JSON summary
#
# Uses the same scan anchors as trainer.

import os, json
import numpy as np
import torch
import matplotlib.pyplot as plt

from typing import Dict, Tuple

try:
    from sklearn.metrics import classification_report, f1_score, confusion_matrix
    _HAS_SK = True
except Exception:
    _HAS_SK = False

from geomlang_edges_relternary_train64_latent256_phase2 import (
    IMG_SIZE, NUM_CH, DEVICE,
    OUT_DIR, CKPT_PATH,
    REL_NAMES,
    GeomEdgesTernary64Dataset,
    SceneModelTernaryEdges64_256,
    render_scene_ABC,
    maybe_rot90_triplet,
    ternary_label_from_centers,
    SCAN_CONFIGS,
    SCAN_T_MIN, SCAN_T_MAX,
)

TAG = "[eval-ternary-phase2-256]"

EVAL_N = 6000
EVAL_SEED = 123
BATCH_SIZE = 256

SCAN_N = 201
SCAN_MARGIN = 6
SCAN_SHAPES = (0, 0, 0)  # A,B,C shapes fixed for scan eval
SCAN_SIZES  = (10, 10, 10)

def save_grid_png(arr, out_path, title, is_labels=False, n_classes=7):
    plt.figure(figsize=(6, 6))
    if is_labels:
        cmap = plt.get_cmap("tab10", n_classes)
        plt.imshow(arr, origin="lower", cmap=cmap, vmin=-0.5, vmax=n_classes - 0.5)
        plt.colorbar(ticks=range(n_classes), label="label")
    else:
        plt.imshow(arr, origin="lower")
        plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def _confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Fallback confusion matrix if sklearn isn't available."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm

def save_confusion_matrix_png(cm: np.ndarray, out_path: str, title: str, labels: list, normalize_rows: bool):
    """
    cm: [C,C] with rows=true, cols=pred
    normalize_rows: if True, convert each row to probabilities (sum to 1 where row sum > 0)
    """
    cm_plot = cm.astype(np.float32)
    if normalize_rows:
        row_sums = cm_plot.sum(axis=1, keepdims=True)
        # avoid division by zero
        cm_plot = np.divide(cm_plot, np.maximum(row_sums, 1e-12))

    plt.figure(figsize=(9, 8))
    im = plt.imshow(cm_plot, origin="upper")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.title(title)
    plt.xlabel("pred")
    plt.ylabel("true")

    # ticks
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)

    # annotate (counts or proportions)
    # keep it light to avoid clutter: only annotate if classes <= 12
    if len(labels) <= 12:
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm_plot[i, j]
                if normalize_rows:
                    s = f"{val:.2f}"
                else:
                    s = str(int(cm[i, j]))
                plt.text(j, i, s, ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

@torch.no_grad()
def classify_batch(model, x_bchw):
    z = model.encode(x_bchw)
    logits = model.rel_head(z)
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    return pred.detach().cpu().numpy(), conf.detach().cpu().numpy()

@torch.no_grad()
def eval_dataset(model):
    ds = GeomEdgesTernary64Dataset(EVAL_N, seed=EVAL_SEED)
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    y_true = []
    y_pred = []
    confs = []

    for imgs, rel, sA, sB, sC in dl:
        imgs = imgs.to(DEVICE)
        pred, conf = classify_batch(model, imgs)
        y_true.append(rel.numpy())
        y_pred.append(pred)
        confs.append(conf)

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    confs  = np.concatenate(confs,  axis=0)

    acc = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro")) if _HAS_SK else None
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted")) if _HAS_SK else None

    report = classification_report(
        y_true, y_pred,
        target_names=REL_NAMES,
        digits=4,
        zero_division=0
    ) if _HAS_SK else None

    # confusion matrix
    n_classes = len(REL_NAMES)
    if _HAS_SK:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    else:
        cm = _confusion_matrix_np(y_true, y_pred, n_classes)

    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "conf_mean": float(confs.mean()),
        "conf_min": float(confs.min()),
        "conf_max": float(confs.max()),
        "report": report,
        "cm": cm,
        "y_true": y_true,
        "y_pred": y_pred,
    }

@torch.no_grad()
def eval_ascan(model, scan_name: str):
    (Bx, By), (Cx, Cy) = SCAN_CONFIGS[scan_name]
    H = W = IMG_SIZE
    shapeA, shapeB, shapeC = SCAN_SHAPES
    sizeA, sizeB, sizeC = SCAN_SIZES

    ts = np.linspace(SCAN_T_MIN, SCAN_T_MAX, SCAN_N)

    y_true = np.zeros((SCAN_N,), dtype=np.int64)
    y_pred = np.zeros((SCAN_N,), dtype=np.int64)
    confs  = np.zeros((SCAN_N,), dtype=np.float32)

    batch_imgs = []
    batch_idx = []

    def flush():
        if not batch_imgs:
            return
        x = torch.from_numpy(np.stack(batch_imgs, axis=0)).float().to(DEVICE)
        pred, conf = classify_batch(model, x)
        for i, p, c in zip(batch_idx, pred, conf):
            y_pred[i] = int(p)
            confs[i]  = float(c)
        batch_imgs.clear()
        batch_idx.clear()

    for i, t in enumerate(ts):
        Ax = int(round(Bx + t * (Cx - Bx)))
        Ay = int(round(By + t * (Cy - By)))
        Ax = int(np.clip(Ax, SCAN_MARGIN + sizeA, W - SCAN_MARGIN - sizeA))
        Ay = int(np.clip(Ay, SCAN_MARGIN + sizeA, H - SCAN_MARGIN - sizeA))

        (Ax2, Ay2), (Bx2, By2), (Cx2, Cy2) = maybe_rot90_triplet((Ax, Ay), (Bx, By), (Cx, Cy), p=1.0)

        y_true[i] = int(ternary_label_from_centers((Ax2, Ay2), (Bx2, By2), (Cx2, Cy2), sizeA, sizeB, sizeC))
        img = render_scene_ABC(Ax2, Ay2, Bx2, By2, Cx2, Cy2, shapeA, shapeB, shapeC, sizeA, sizeB, sizeC)

        batch_imgs.append(img)
        batch_idx.append(i)
        if len(batch_imgs) >= BATCH_SIZE:
            flush()

    flush()

    acc_line = (y_true == y_pred).astype(np.float32)
    return {
        "t_vals": ts,
        "y_true": y_true,
        "y_pred": y_pred,
        "conf": confs,
        "acc_line": acc_line,
        "mean_acc": float(acc_line.mean()),
        "mean_conf": float(confs.mean()),
    }

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"{TAG} device = {DEVICE}")
    print(f"{TAG} loading checkpoint: {CKPT_PATH}")

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    if "model_state_dict" not in ckpt:
        raise RuntimeError(f"{TAG} checkpoint missing model_state_dict")

    model = SceneModelTernaryEdges64_256().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ---- Dataset eval ----
    print(f"{TAG} dataset eval (N={EVAL_N}, seed={EVAL_SEED})")
    ds = eval_dataset(model)

    print(f"{TAG} DATASET acc = {ds['acc']:.4f}")
    if ds["macro_f1"] is not None:
        print(f"{TAG} DATASET macro_f1 = {ds['macro_f1']:.4f} | weighted_f1 = {ds['weighted_f1']:.4f}")
    print(f"{TAG} conf mean={ds['conf_mean']:.4f} min={ds['conf_min']:.4f} max={ds['conf_max']:.4f}")
    if ds["report"] is not None:
        print(ds["report"])

    # ---- Confusion matrix PNGs (counts + normalized) ----
    cm = ds["cm"]
    cm_counts_path = os.path.join(OUT_DIR, "phase2_confusion_matrix_counts.png")
    cm_norm_path   = os.path.join(OUT_DIR, "phase2_confusion_matrix_row_norm.png")

    save_confusion_matrix_png(
        cm,
        cm_counts_path,
        title="Phase2 Confusion (counts): (true rows, pred cols)",
        labels=REL_NAMES,
        normalize_rows=False,
    )
    save_confusion_matrix_png(
        cm,
        cm_norm_path,
        title="Phase2 Confusion (row-normalized): (true rows, pred cols)",
        labels=REL_NAMES,
        normalize_rows=True,
    )
    print(f"{TAG} saved: {cm_counts_path}")
    print(f"{TAG} saved: {cm_norm_path}")

    # ---- Scans ----
    scans = {}
    for name in SCAN_CONFIGS.keys():
        print(f"{TAG} A-scan: {name}  B={SCAN_CONFIGS[name][0]} C={SCAN_CONFIGS[name][1]}  SCAN_N={SCAN_N}")
        res = eval_ascan(model, name)
        scans[name] = {
            "mean_acc": res["mean_acc"],
            "mean_conf": res["mean_conf"],
        }

        # Save 1D "images" as 2D for easy viewing: [1, N]
        true_img = res["y_true"][None, :]
        pred_img = res["y_pred"][None, :]
        conf_img = res["conf"][None, :]
        acc_img  = res["acc_line"][None, :]

        true_path = os.path.join(OUT_DIR, f"phase2_scan_{name}_true.png")
        pred_path = os.path.join(OUT_DIR, f"phase2_scan_{name}_pred.png")
        conf_path = os.path.join(OUT_DIR, f"phase2_scan_{name}_conf.png")
        acc_path  = os.path.join(OUT_DIR, f"phase2_scan_{name}_acc.png")

        save_grid_png(true_img, true_path, f"A-scan {name}: true labels", is_labels=True, n_classes=len(REL_NAMES))
        save_grid_png(pred_img, pred_path, f"A-scan {name}: predicted labels", is_labels=True, n_classes=len(REL_NAMES))
        save_grid_png(conf_img, conf_path, f"A-scan {name}: confidence", is_labels=False)
        save_grid_png(acc_img,  acc_path,  f"A-scan {name}: accuracy (mean={res['mean_acc']:.4f})", is_labels=False)

        print(f"{TAG} SCAN {name} mean_acc={res['mean_acc']:.4f} mean_conf={res['mean_conf']:.4f}")

    # ---- Summary ----
    summary = {
        "ckpt_path": CKPT_PATH,
        "dataset": {
            "acc": ds["acc"],
            "macro_f1": ds["macro_f1"],
            "weighted_f1": ds["weighted_f1"],
            "conf_mean": ds["conf_mean"],
            "conf_min": ds["conf_min"],
            "conf_max": ds["conf_max"],
            "report": ds["report"],
            "confusion_matrix_counts": ds["cm"].tolist(),
        },
        "scans": scans,
        "artifacts": {
            "confusion_counts_png": cm_counts_path,
            "confusion_row_norm_png": cm_norm_path,
        }
    }
    out_json = os.path.join(OUT_DIR, "phase2_eval_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"{TAG} saved summary: {out_json}")
    print(f"{TAG} done.")

if __name__ == "__main__":
    main()
