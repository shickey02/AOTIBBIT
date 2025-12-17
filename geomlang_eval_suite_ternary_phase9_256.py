#!/usr/bin/env python3
# geomlang_eval_suite_ternary_phase9_256.py
#
# Phase 9 eval:
# - Reads phase9_y_true.npy + phase9_y_pred.npy
# - Writes report + confusion matrices (counts + row-normalized) + JSON summary

import os, json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

OUTDIR = "outputs_edges_relternary256_phase9"
os.makedirs(OUTDIR, exist_ok=True)

Y_TRUE = os.path.join(OUTDIR, "phase9_y_true.npy")
Y_PRED = os.path.join(OUTDIR, "phase9_y_pred.npy")

LABELS = [
    "A_left_of_BtoC",
    "A_right_of_BtoC",
    "A_between_clear",
    "A_between_overlap",
    "A_overlap_only",
]

def _save_cm_png(cm, labels, path, title, normalize_rows=False):
    cm = cm.astype(np.float32)
    if normalize_rows:
        denom = cm.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        cm_show = cm / denom
    else:
        cm_show = cm

    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()
    im = ax.imshow(cm_show, interpolation="nearest")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # annotate (lightweight)
    for i in range(cm_show.shape[0]):
        for j in range(cm_show.shape[1]):
            val = cm_show[i, j]
            if normalize_rows:
                s = f"{val:.2f}"
            else:
                s = f"{int(cm[i, j])}"
            ax.text(j, i, s, ha="center", va="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def main():
    assert os.path.exists(Y_TRUE), f"Missing: {Y_TRUE}"
    assert os.path.exists(Y_PRED), f"Missing: {Y_PRED}"

    y_true = np.load(Y_TRUE).astype(int)
    y_pred = np.load(Y_PRED).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    macro = float(f1_score(y_true, y_pred, average="macro"))
    weighted = float(f1_score(y_true, y_pred, average="weighted"))

    report = classification_report(
        y_true, y_pred,
        target_names=LABELS,
        digits=4,
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))

    with open(os.path.join(OUTDIR, "phase9_report.txt"), "w") as f:
        f.write(report)

    _save_cm_png(
        cm, LABELS,
        os.path.join(OUTDIR, "phase9_confusion_matrix_counts.png"),
        title="Phase 9 Confusion Matrix (Counts)",
        normalize_rows=False
    )

    _save_cm_png(
        cm, LABELS,
        os.path.join(OUTDIR, "phase9_confusion_matrix_row_norm.png"),
        title="Phase 9 Confusion Matrix (Row-Normalized)",
        normalize_rows=True
    )

    summary = {
        "phase": 9,
        "accuracy": acc,
        "macro_f1": macro,
        "weighted_f1": weighted,
        "support_true": np.bincount(y_true, minlength=len(LABELS)).tolist(),
        "support_pred": np.bincount(y_pred, minlength=len(LABELS)).tolist(),
        "confusion_matrix_counts": cm.tolist(),
    }

    with open(os.path.join(OUTDIR, "phase9_eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("[eval-ternary-phase9-256] acc =", f"{acc:.4f}",
          "| macro_f1 =", f"{macro:.4f}",
          "| weighted_f1 =", f"{weighted:.4f}")
    print(report)
    print("[eval-ternary-phase9-256] saved ->", OUTDIR)

if __name__ == "__main__":
    main()
