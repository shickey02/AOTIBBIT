#!/usr/bin/env python3
# geomlang_eval_suite_ternary_phase1_256.py
#
# Eval suite for ternary Phase1 model (A,B,C) with 7 labels:
#  ['A_left_of_BtoC','A_right_of_BtoC','A_between_BC',
#   'A_closer_to_B','A_closer_to_C','A_overlap_B','A_overlap_C']
#
# Outputs:
#  - confusion matrix + macro/weighted F1
#  - A-position grid "scans" (B,C fixed; A swept over grid)
#    showing predicted label, confidence, and correctness.

import os, json, math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score

# -----------------------
# Paths / config
# -----------------------
OUT_DIR   = "outputs_edges_relternary256_phase1"
CKPT_PATH = os.path.join(OUT_DIR, "scene_model_edges_relternary256_phase1.pt")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TAG = "[eval-ternary-phase1-256]"

# Dataset eval
N_DATASET = 6000
SEED      = 123
BATCH     = 256

# Grid eval (A sweep)
GRID_N = 201  # 201x201 heatmaps
# We’ll generate a full grid of A positions; margins keep shapes on-canvas
MARGIN = 6

# -----------------------
# Import your training code (dataset + model + labels)
# -----------------------
# You trained with:
#   geomlang_edges_relternary_train64_latent256_phase1.py
# so we import from that file to guarantee identical generation + labels.
from geomlang_edges_relternary_train64_latent256_phase1 import (
    IMG_SIZE,
    GeomEdgesTernary64Phase1,   # <-- if your class name differs, see note below
    SceneModelTernaryPhase1_256,    # <-- if your model class name differs, see note below
    REL_NAMES,
)
LABEL_NAMES = REL_NAMES

# -----------------------
# Helpers
# -----------------------
def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

def plot_confusion(cm, labels, title, path):
    plt.figure(figsize=(8,7))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)
    plt.xlabel("pred")
    plt.ylabel("true")
    save_fig(path)

def softmax_conf(logits):
    # returns (pred, conf)
    probs = F.softmax(logits, dim=-1)
    conf, pred = probs.max(dim=-1)
    return pred, conf

# -----------------------
# A-grid scene generator (B,C fixed; A moves)
# -----------------------
# We want to reuse your exact drawing code + label function for correctness.
# Easiest: instantiate a dataset-like generator by calling the same internal
# helpers your dataset uses.
#
# Many of your phase1 train scripts already have helper functions like:
#   draw_circle / draw_square / make_edges / label_from_centers(...)
# But since names can differ, we’ll create a tiny "probe dataset" by
# borrowing the dataset class and injecting fixed centers.
#
# If your dataset doesn’t support this directly, this fallback implementation
# calls dataset internals by re-implementing minimal drawing in-place.
#
# To keep this robust, we’ll attempt to import helpers from the training file;
# if not present, we’ll use fallback drawing.

try:
    from geomlang_edges_relternary_train64_latent256_phase1 import (
        draw_circle, draw_square, make_edges, label_phase1_from_centers
    )
    HAVE_HELPERS = True
except Exception:
    HAVE_HELPERS = False

def _fallback_draw_circle(mask, cx, cy, r):
    H, W = mask.shape
    yy, xx = np.ogrid[:H, :W]
    dist2 = (xx - cx)**2 + (yy - cy)**2
    mask[dist2 <= r*r] = 1.0

def _fallback_draw_square(mask, cx, cy, s):
    H, W = mask.shape
    x0 = max(0, cx - s); x1 = min(W, cx + s + 1)
    y0 = max(0, cy - s); y1 = min(H, cy + s + 1)
    mask[y0:y1, x0:x1] = 1.0

def _fallback_make_edges(A, B, C):
    union = ((A > 0.5) | (B > 0.5) | (C > 0.5)).astype(np.float32)
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

def _fallback_label_phase1(cxA, cyA, cxB, cyB, cxC, cyC,
                          tol_cross=2.0, tol_closer=2.0, tol_between_line=2.0, tol_overlap=3.0):
    # This is a fallback. Ideally you have label_phase1_from_centers in train script.
    # If you do, we’ll use it. If not, this gives a reasonable approximation:
    #
    # - overlap_B/C if distance < tol_overlap
    # - closer_to_B/C if |dB - dC| > tol_closer
    # - between_BC if A is near line segment BC (distance-to-line < tol_between_line) AND projection within segment
    # - left/right of directed BC based on cross product sign
    #
    # NOTE: If your official labeling differs, prefer the imported helper.
    dxBC = cxC - cxB
    dyBC = cyC - cyB
    dxBA = cxA - cxB
    dyBA = cyA - cyB

    # overlaps (center-proxy)
    dAB = math.hypot(cxA - cxB, cyA - cyB)
    dAC = math.hypot(cxA - cxC, cyA - cyC)
    if dAB <= tol_overlap:
        return 5  # A_overlap_B
    if dAC <= tol_overlap:
        return 6  # A_overlap_C

    # closer
    if abs(dAB - dAC) > tol_closer:
        return 3 if dAB < dAC else 4  # closer_to_B else closer_to_C

    # between: distance from A to segment BC
    denom = (dxBC*dxBC + dyBC*dyBC)
    if denom < 1e-6:
        # degenerate BC, fall back to overlap-ish behavior
        return 2
    t = ((cxA - cxB)*dxBC + (cyA - cyB)*dyBC) / denom
    t_clamped = max(0.0, min(1.0, t))
    px = cxB + t_clamped*dxBC
    py = cyB + t_clamped*dyBC
    dist_line = math.hypot(cxA - px, cyA - py)
    if dist_line <= tol_between_line and 0.0 <= t <= 1.0:
        return 2  # A_between_BC

    # left/right of directed BC via cross product sign
    cross = dxBC*dyBA - dyBC*dxBA
    if abs(cross) <= tol_cross:
        # near-colinear; default to between-ish if nothing else
        return 2
    return 0 if cross > 0 else 1  # A_left_of_BtoC else A_right_of_BtoC

def make_probe_img(cxA, cyA, cxB, cyB, cxC, cyC,
                   shapeA=0, shapeB=0, shapeC=0,
                   sizeA=8, sizeB=8, sizeC=8):
    H = W = IMG_SIZE
    A = np.zeros((H,W), np.float32)
    B = np.zeros((H,W), np.float32)
    C = np.zeros((H,W), np.float32)

    if HAVE_HELPERS:
        # use the exact drawing funcs from your training script
        if shapeA == 0: draw_circle(A, cxA, cyA, sizeA)
        else:           draw_square(A, cxA, cyA, sizeA)
        if shapeB == 0: draw_circle(B, cxB, cyB, sizeB)
        else:           draw_square(B, cxB, cyB, sizeB)
        if shapeC == 0: draw_circle(C, cxC, cyC, sizeC)
        else:           draw_square(C, cxC, cyC, sizeC)
        edges = make_edges(A, B, C)
        y = label_phase1_from_centers(cxA, cyA, cxB, cyB, cxC, cyC)
    else:
        if shapeA == 0: _fallback_draw_circle(A, cxA, cyA, sizeA)
        else:           _fallback_draw_square(A, cxA, cyA, sizeA)
        if shapeB == 0: _fallback_draw_circle(B, cxB, cyB, sizeB)
        else:           _fallback_draw_square(B, cxB, cyB, sizeB)
        if shapeC == 0: _fallback_draw_circle(C, cxC, cyC, sizeC)
        else:           _fallback_draw_square(C, cxC, cyC, sizeC)
        edges = _fallback_make_edges(A, B, C)
        y = _fallback_label_phase1(cxA, cyA, cxB, cyB, cxC, cyC)

    img = np.stack([A, B, C, edges], axis=0)  # NOTE: expects your model uses 4ch (A,B,C,edges)
    return img, int(y)

def run_a_grid_scan(model, name, Bxy, Cxy,
                    shapeA=0, shapeB=0, shapeC=0,
                    sizeA=8, sizeB=8, sizeC=8):
    """
    Sweeps A over the image plane; holds B,C fixed.
    Saves:
      - pred label map (int)
      - conf map (float)
      - correctness map (0/1)
    """
    cxB, cyB = Bxy
    cxC, cyC = Cxy

    H = W = IMG_SIZE
    xs = np.linspace(MARGIN, W - 1 - MARGIN, GRID_N).astype(np.int32)
    ys = np.linspace(MARGIN, H - 1 - MARGIN, GRID_N).astype(np.int32)

    pred_map = np.zeros((GRID_N, GRID_N), np.int32)
    conf_map = np.zeros((GRID_N, GRID_N), np.float32)
    acc_map  = np.zeros((GRID_N, GRID_N), np.float32)

    # batch A positions for speed
    batch_imgs = []
    batch_true = []
    batch_pos  = []

    def flush_batch():
        if not batch_imgs:
            return
        x = torch.from_numpy(np.stack(batch_imgs, axis=0)).float().to(DEVICE)
        with torch.no_grad():
            rec, rel_logits = model(x)[:2] if isinstance(model(x), (tuple, list)) else (None, model(x))
            pred, conf = softmax_conf(rel_logits)
        pred = pred.cpu().numpy()
        conf = conf.cpu().numpy()

        for (i, j), y_true, y_pred, y_conf in zip(batch_pos, batch_true, pred, conf):
            pred_map[i, j] = int(y_pred)
            conf_map[i, j] = float(y_conf)
            acc_map[i, j]  = 1.0 if int(y_pred) == int(y_true) else 0.0

        batch_imgs.clear()
        batch_true.clear()
        batch_pos.clear()

    for i, cyA in enumerate(ys):
        for j, cxA in enumerate(xs):
            img, y_true = make_probe_img(cxA, cyA, cxB, cyB, cxC, cyC,
                                         shapeA=shapeA, shapeB=shapeB, shapeC=shapeC,
                                         sizeA=sizeA, sizeB=sizeB, sizeC=sizeC)
            batch_imgs.append(img)
            batch_true.append(y_true)
            batch_pos.append((i, j))

            if len(batch_imgs) >= BATCH:
                flush_batch()

    flush_batch()

    # Save plots
    # Pred labels (categorical heatmap)
    plt.figure(figsize=(7,6))
    plt.imshow(pred_map, interpolation="nearest")
    plt.title(f"A-scan predicted labels ({name})")
    plt.colorbar(label="class")
    save_fig(os.path.join(OUT_DIR, f"phase1_scan_{name}_pred.png"))

    # Confidence heatmap
    plt.figure(figsize=(7,6))
    plt.imshow(conf_map, vmin=0.0, vmax=1.0, interpolation="nearest")
    plt.title(f"A-scan confidence ({name})")
    plt.colorbar(label="p(max)")
    save_fig(os.path.join(OUT_DIR, f"phase1_scan_{name}_conf.png"))

    # Accuracy heatmap
    mean_acc = float(acc_map.mean())
    plt.figure(figsize=(7,6))
    plt.imshow(acc_map, vmin=0.0, vmax=1.0, interpolation="nearest")
    plt.title(f"A-scan accuracy mean={mean_acc:.4f} ({name})")
    plt.colorbar(label="correct")
    save_fig(os.path.join(OUT_DIR, f"phase1_scan_{name}_acc.png"))

    return {
        "name": name,
        "B": [int(cxB), int(cyB)],
        "C": [int(cxC), int(cyC)],
        "grid_n": int(GRID_N),
        "mean_acc": mean_acc,
        "mean_conf": float(conf_map.mean()),
        "min_conf": float(conf_map.min()),
        "max_conf": float(conf_map.max()),
    }

# -----------------------
# Main
# -----------------------
def main():
    print(f"{TAG} loading checkpoint: {CKPT_PATH}")

    # Load model
    model = SceneModelTernaryPhase1_256().to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    if "model_state_dict" not in ckpt:
        raise RuntimeError(f"{TAG} missing model_state_dict in ckpt")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.eval()
    # ---------------- Dataset eval ----------------
    print(f"{TAG} dataset eval (N={N_DATASET}, seed={SEED})")
    ds = GeomEdgesTernary64Phase1(N_DATASET, seed=SEED)
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0)
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0)

    y_true_all = []
    y_pred_all = []
    conf_all   = []

    with torch.no_grad():
        for batch in dl:
            # Expected dataset return: (img, rel, sA, sB, sC) or (img, rel, ...)
            imgs = batch[0].float().to(DEVICE)
            y    = batch[1].long().cpu().numpy()

            out = model(imgs)
            # Expect forward returns (rec, rel_logits, ...) or at least (rec, rel_logits)
            if isinstance(out, (tuple, list)):
                rel_logits = out[1]
            else:
                rel_logits = out

            pred, conf = softmax_conf(rel_logits)
            y_pred = pred.cpu().numpy()
            y_conf = conf.cpu().numpy()

            y_true_all.append(y)
            y_pred_all.append(y_pred)
            conf_all.append(y_conf)

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    confs  = np.concatenate(conf_all)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))

    print(f"{TAG} DATASET acc = {acc:.4f}")
    print(f"{TAG} DATASET macro_f1 = {macro_f1:.4f} | weighted_f1 = {weighted_f1:.4f}")
    print(f"{TAG} conf mean={confs.mean():.4f} min={confs.min():.4f} max={confs.max():.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABEL_NAMES))))
    plot_confusion(cm, LABEL_NAMES, "Phase1 Confusion (true rows, pred cols)",
                   os.path.join(OUT_DIR, "phase1_confusion_matrix.png"))

    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(LABEL_NAMES))),
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0
    )
    print(report)

    # ---------------- A-grid scans ----------------
    # These mirror the pattern you used earlier (horiz_mid / vert_mid / diag_mid / etc.)
    scans = []

    # Middle horizontal BC
    scans.append(("horiz_mid",   (18, 32), (46, 32)))
    # Middle vertical BC
    scans.append(("vert_mid",    (32, 18), (32, 46)))
    # Diagonal BC
    scans.append(("diag_mid",    (20, 20), (44, 44)))
    # Shorter horizontal BC (tighter segment)
    scans.append(("horiz_short", (23, 32), (41, 32)))
    # Off-axis BC
    scans.append(("horiz_off",   (18, 22), (46, 40)))

    scan_summaries = []
    for name, Bxy, Cxy in scans:
        print(f"{TAG} A-scan: {name}  B={Bxy} C={Cxy}  GRID_N={GRID_N}")
        s = run_a_grid_scan(model, name, Bxy, Cxy)
        print(f"{TAG} SCAN {name} mean_acc={s['mean_acc']:.4f} mean_conf={s['mean_conf']:.4f}")
        scan_summaries.append(s)

    summary = {
        "dataset": {
            "N": int(N_DATASET),
            "seed": int(SEED),
            "acc": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "conf_mean": float(confs.mean()),
            "conf_min": float(confs.min()),
            "conf_max": float(confs.max()),
        },
        "label_names": LABEL_NAMES,
        "scan_summaries": scan_summaries,
    }

    out_json = os.path.join(OUT_DIR, "phase1_eval_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"{TAG} saved summary: {out_json}")
    print(f"{TAG} done.")

if __name__ == "__main__":
    main()
