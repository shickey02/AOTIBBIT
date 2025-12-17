#!/usr/bin/env python3
# geomlang_eval_suite_ternary_phase8_256.py
#
# Phase 8: Threshold calibration on Phase 7 outputs
# - Uses Phase 7 exported .npz files (since Phase 8 = no training)
# - Builds y_true from *target factor keys* (between_score, overlap_any, lr_sign)
#   using Phase 7 dataset construction thresholds:
#       between_hi = 0.70 (between buckets)
#       between_lo = 0.35 (non-between buckets)
#       overlap_true = 0.50
# - Sweeps Tb/To on predictions to best match y_true by macro-F1.

import os, json
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix, f1_score

OUTDIR = "outputs_edges_relternary256_phase8"
os.makedirs(OUTDIR, exist_ok=True)

LATENTS = "outputs_edges_relternary256_phase7/encoded_latents_seed123_N6000.npy"
TARGETS = "outputs_edges_relternary256_phase7/encoded_targets_seed123_N6000.npz"
PREDS   = "outputs_edges_relternary256_phase7/encoded_preds_seed123_N6000.npz"

LABELS = [
    "A_left_of_BtoC",
    "A_right_of_BtoC",
    "A_between_clear",
    "A_between_overlap",
    "A_overlap_only",
]

TAG = "[eval-ternary-phase8-256]"

def _ensure_1d(x, name):
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim != 1:
        raise ValueError(f"{TAG} {name} should be 1D (N,) but got shape {x.shape}")
    return x

def sigmoid(x):
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-x))

def derive_true_from_targets(between_t, overlap_t, lr_t,
                             BETWEEN_HI=0.70, BETWEEN_LO=0.35, OVERLAP_TO=0.50):
    """
    Reconstructs y_true classes from how Phase 7 data was *constructed*:
      - between_clear:  between >= BETWEEN_HI and not overlap
      - between_overlap:between >= BETWEEN_HI and overlap
      - overlap_only:   between <= BETWEEN_LO and overlap
      - left/right:     between <= BETWEEN_LO and not overlap (lr decides)
    Anything in the "gray zone" (BETWEEN_LO < between < BETWEEN_HI) is assigned
    by a reasonable tie-break:
      - if overlap => overlap_only
      - else => left/right by lr
    """
    between_t = _ensure_1d(between_t, "between_score(target)")
    overlap_t = _ensure_1d(overlap_t, "overlap_any(target)")
    lr_t      = _ensure_1d(lr_t,      "lr_sign(target)")

    y = np.zeros((between_t.shape[0],), dtype=np.int64)

    is_overlap = overlap_t >= OVERLAP_TO
    is_between_hi = between_t >= BETWEEN_HI
    is_non_between_lo = between_t <= BETWEEN_LO

    # Between classes
    y[np.logical_and(is_between_hi, np.logical_not(is_overlap))] = 2  # between_clear
    y[np.logical_and(is_between_hi, is_overlap)] = 3                  # between_overlap

    # Overlap-only in low-between region
    y[np.logical_and(is_non_between_lo, is_overlap)] = 4              # overlap_only

    # Left/right in low-between region with no overlap
    base_lr = np.logical_and(is_non_between_lo, np.logical_not(is_overlap))
    y[np.logical_and(base_lr, lr_t < 0.5)] = 0  # left
    y[np.logical_and(base_lr, lr_t >= 0.5)] = 1 # right

    # Gray zone tie-break
    gray = np.logical_and(~is_between_hi, ~is_non_between_lo)
    gray_overlap = np.logical_and(gray, is_overlap)
    gray_no_overlap = np.logical_and(gray, ~is_overlap)

    y[gray_overlap] = 4
    y[np.logical_and(gray_no_overlap, lr_t < 0.5)] = 0
    y[np.logical_and(gray_no_overlap, lr_t >= 0.5)] = 1

    return y

def derive_pred_labels(between_p, overlap_p, lr_p, Tb, To):
    """
    Pred-derived labels with sweepable thresholds Tb/To.
    """
    y = np.zeros((between_p.shape[0],), dtype=np.int64)

    is_between = between_p >= Tb
    is_overlap = overlap_p >= To
    lr_right   = lr_p >= 0.5

    y[np.logical_and(is_between, ~is_overlap)] = 2
    y[np.logical_and(is_between, is_overlap)]  = 3
    y[np.logical_and(~is_between, is_overlap)] = 4

    base = np.logical_and(~is_between, ~is_overlap)
    y[np.logical_and(base, ~lr_right)] = 0
    y[np.logical_and(base, lr_right)]  = 1

    return y

def main():
    # Load files (latents not required for eval, but keep for parity)
    _ = np.load(LATENTS)

    targets = np.load(TARGETS)
    preds   = np.load(PREDS)

    print(f"{TAG} target keys: {list(targets.files)}")
    print(f"{TAG} pred keys:   {list(preds.files)}")

    # ---- Targets (ground truth factors) ----
    between_t = _ensure_1d(targets["between_score"], "between_score(target)")
    overlap_t = _ensure_1d(targets["overlap_any"],   "overlap_any(target)")
    lr_t      = _ensure_1d(targets["lr_sign"],       "lr_sign(target)")

    # ---- Predictions ----
    between_p = _ensure_1d(preds["between_pred"], "between_pred(pred)")
    tproj_p   = _ensure_1d(preds["tproj_pred"],   "tproj_pred(pred)")  # unused in label derive, but kept
    overlap_k = _ensure_1d(preds["overlap_logit"],"overlap_logit(pred)")
    lr_k      = _ensure_1d(preds["lr_logit"],     "lr_logit(pred)")

    # Convert logits -> probs for overlap/lr
    overlap_p = sigmoid(overlap_k)
    lr_p      = sigmoid(lr_k)

    # between_pred is already 0..1 in your archive; keep safe
    between_p = np.clip(between_p.astype(np.float32), 0.0, 1.0)
    tproj_p   = np.clip(tproj_p.astype(np.float32),   0.0, 1.0)

    # ---- True derived labels (from target factors) ----
    y_true = derive_true_from_targets(
        between_t, overlap_t, lr_t,
        BETWEEN_HI=0.70, BETWEEN_LO=0.35, OVERLAP_TO=0.50
    )

    # ---- Head sanity metrics ----
    between_mse = float(np.mean((between_p - between_t) ** 2))
    overlap_acc = float(np.mean((overlap_p >= 0.5) == (overlap_t >= 0.5)))
    lr_acc      = float(np.mean((lr_p      >= 0.5) == (lr_t      >= 0.5)))

    print(f"{TAG} HEAD between_mse={between_mse:.5f} overlap_acc={overlap_acc:.4f} lr_acc={lr_acc:.4f}")

    # ---- Sweep thresholds to best match y_true ----
    best = {"macro_f1": -1.0, "Tb": None, "To": None}

    Tb_vals = np.linspace(0.35, 0.90, 45)
    To_vals = np.linspace(0.20, 0.90, 45)

    for Tb in Tb_vals:
        for To in To_vals:
            y_pred = derive_pred_labels(between_p, overlap_p, lr_p, Tb=float(Tb), To=float(To))
            f1 = f1_score(y_true, y_pred, average="macro")
            if f1 > best["macro_f1"]:
                best.update({"macro_f1": float(f1), "Tb": float(Tb), "To": float(To)})

    print(f"{TAG} BEST THRESHOLDS: {best}")

    # ---- Final eval using best thresholds ----
    Tb = best["Tb"]
    To = best["To"]
    y_pred = derive_pred_labels(between_p, overlap_p, lr_p, Tb=Tb, To=To)

    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        digits=4,
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))

    with open(os.path.join(OUTDIR, "phase8_thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    with open(os.path.join(OUTDIR, "phase8_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    np.save(os.path.join(OUTDIR, "phase8_confusion_matrix.npy"), cm)

    summary = {
        "between_mse": between_mse,
        "overlap_acc@0.5": overlap_acc,
        "lr_acc@0.5": lr_acc,
        "best": best,
    }
    with open(os.path.join(OUTDIR, "phase8_eval_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(report)
    print(f"{TAG} saved thresholds + report + confusion matrix + summary -> {OUTDIR}")

if __name__ == "__main__":
    main()
