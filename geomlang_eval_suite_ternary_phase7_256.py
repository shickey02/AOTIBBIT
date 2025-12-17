#!/usr/bin/env python3
# geomlang_eval_suite_ternary_phase7_256.py

import os, json
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score

OUTDIR = "outputs_edges_relternary256_phase7"
T_PATH = os.path.join(OUTDIR, "encoded_targets_seed123_N6000.npz")
P_PATH = os.path.join(OUTDIR, "encoded_preds_seed123_N6000.npz")

SUMMARY_PATH = os.path.join(OUTDIR, "phase7_eval_summary.json")

# Thresholds
BETW_THR = 0.70
OVER_THR = 0.0   # overlap_any already 0/1 in targets; preds is logit
LR_MASK_BETW_THR = 0.50

REL_NAMES = [
    "A_left_of_BtoC",
    "A_right_of_BtoC",
    "A_between_clear",
    "A_between_overlap",
    "A_overlap_only",
]

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def derive_label(between, overlap_any, lr_sign):
    # between, overlap_any in [0..1], lr_sign in {0,1}
    if between >= BETW_THR:
        return 3 if overlap_any >= 0.5 else 2  # between_overlap vs between_clear
    if overlap_any >= 0.5:
        return 4  # overlap_only (not-between but overlapping)
    # not between, not overlap → left/right
    return 1 if lr_sign >= 0.5 else 0

def main():
    assert os.path.exists(T_PATH), f"Missing targets: {T_PATH}"
    assert os.path.exists(P_PATH), f"Missing preds: {P_PATH}"

    T = np.load(T_PATH)
    P = np.load(P_PATH)

    y_between = T["between_score"].astype(np.float32)
    y_tproj   = T["t_on_BC"].astype(np.float32)
    y_overlap = T["overlap_any"].astype(np.float32)
    y_lr      = T["lr_sign"].astype(np.float32)

    p_between = P["between_pred"].astype(np.float32)
    p_tproj   = P["tproj_pred"].astype(np.float32)
    p_overlap = sigmoid(P["overlap_logit"].astype(np.float32))
    p_lr      = sigmoid(P["lr_logit"].astype(np.float32))

    # Head metrics
    between_mse = float(np.mean((p_between - y_between)**2))
    tproj_mse   = float(np.mean((p_tproj - y_tproj)**2))
    overlap_acc = float(np.mean((p_overlap >= 0.5) == (y_overlap >= 0.5)))

    lr_mask = (y_between < LR_MASK_BETW_THR)
    if np.any(lr_mask):
        lr_acc = float(np.mean((p_lr[lr_mask] >= 0.5) == (y_lr[lr_mask] >= 0.5)))
    else:
        lr_acc = None

    # Derived labels
    y_true = np.array([derive_label(y_between[i], y_overlap[i], y_lr[i]) for i in range(len(y_between))], dtype=np.int64)
    y_pred = np.array([derive_label(p_between[i], p_overlap[i], (p_lr[i] >= 0.5)) for i in range(len(y_between))], dtype=np.int64)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(REL_NAMES))))
    report = classification_report(y_true, y_pred, target_names=REL_NAMES, digits=4, zero_division=0)

    print("[eval-ternary-phase7-256] HEAD between_mse=%.5f tproj_mse=%.5f overlap_acc=%.4f lr_acc=%s"
          % (between_mse, tproj_mse, overlap_acc, "None" if lr_acc is None else f"{lr_acc:.4f}"))
    print("[eval-ternary-phase7-256] DERIVED acc = %.4f" % acc)
    print("[eval-ternary-phase7-256] DERIVED macro_f1 = %.4f | weighted_f1 = %.4f" % (macro_f1, weighted_f1))
    print(report)

    summary = {
        "between_mse": between_mse,
        "tproj_mse": tproj_mse,
        "overlap_acc": overlap_acc,
        "lr_acc_masked": lr_acc,
        "derived_acc": acc,
        "derived_macro_f1": macro_f1,
        "derived_weighted_f1": weighted_f1,
        "confusion_matrix": cm.tolist(),
        "rel_names": REL_NAMES,
        "thresholds": {
            "BETW_THR": BETW_THR,
            "LR_MASK_BETW_THR": LR_MASK_BETW_THR,
        }
    }

    os.makedirs(OUTDIR, exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[eval-ternary-phase7-256] saved summary: {SUMMARY_PATH}")
    print("[eval-ternary-phase7-256] done.")

if __name__ == "__main__":
    main()
