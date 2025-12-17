#!/usr/bin/env python3
# geomlang_eval_suite_ternary_phase6_256.py
#
# Phase 6 Eval:
# - Per-head metrics (MSE/MAE/Acc)
# - Derived label confusion matrix (optional) that ALLOWS between+overlap (A_between_overlap)
# - Binary between accuracy + overlap accuracy
# - Robust sklearn report even when some classes have 0 support
# - JSON-safe summary writer

import os, json
import numpy as np
import torch

try:
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    _HAS_SK = True
except Exception:
    _HAS_SK = False

import matplotlib.pyplot as plt

from geomlang_edges_relternary_train64_latent256_phase6 import (
    ConvAEHeads, TernaryGeomDataset, LATENT_DIM, OUTDIR
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = os.path.join(OUTDIR, "scene_model_edges_relternary256_phase6.pt")

N = 6000
SEED = 123

# Derived label set (kept for debugging only — factors are the real objective)
REL_NAMES = [
    "A_left_of_BtoC",
    "A_right_of_BtoC",
    "A_closer_to_B",
    "A_closer_to_C",
    "A_overlap_B",
    "A_overlap_C",
    "A_between_clear",
    "A_between_overlap",
]

def _jsonify(o):
    import numpy as _np
    if isinstance(o, _np.ndarray):
        return o.tolist()
    if isinstance(o, (_np.float32, _np.float64)):
        return float(o)
    if isinstance(o, (_np.int32, _np.int64)):
        return int(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

def derive_label(t_between, tproj, csign, oB, oC,
                 THR_BETWEEN=0.70, THR_OVER=0.50, THR_C=0.50):
    """
    IMPORTANT: between is NOT excluded by overlap.

    Priority is just for a single confusion matrix view:
      1) Between clear/overlap
      2) Left/Right (based on tproj)
      3) Closer sign
      4) Overlap B/C

    This means the confusion matrix isn't "truth" — it's a visualization.
    """
    is_between = (t_between >= THR_BETWEEN) and (tproj >= 0.0) and (tproj <= 1.0)
    is_overlap = (oB >= THR_OVER) or (oC >= THR_OVER)

    if is_between and is_overlap:
        return REL_NAMES.index("A_between_overlap")
    if is_between and (not is_overlap):
        return REL_NAMES.index("A_between_clear")

    # left/right from tproj: if projection near B side => left, else right
    # (This is only meaningful in your canonical B->C direction; rotations exist in data.)
    if tproj < 0.5:
        return REL_NAMES.index("A_left_of_BtoC")
    else:
        return REL_NAMES.index("A_right_of_BtoC")

def plot_confusion(cm, names, outpath, title):
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(names))
    plt.xticks(ticks, names, rotation=45, ha="right")
    plt.yticks(ticks, names)
    plt.tight_layout()
    plt.xlabel("pred")
    plt.ylabel("true")
    fig.savefig(outpath, dpi=180)
    plt.close(fig)

@torch.no_grad()
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"[eval-ternary-phase6-256] device = {DEVICE}")
    print(f"[eval-ternary-phase6-256] loading checkpoint: {CKPT}")

    ck = torch.load(CKPT, map_location=DEVICE)
    model = ConvAEHeads(LATENT_DIM).to(DEVICE)
    model.load_state_dict(ck["model_state"])
    model.eval()

    ds = TernaryGeomDataset(N, seed=SEED)
    print(f"[eval-ternary-phase6-256] dataset eval (N={N}, seed={SEED})")

    # Collect factor targets & preds
    tb = np.zeros((N,), np.float32)
    tt = np.zeros((N,), np.float32)
    tcs = np.zeros((N,), np.float32)
    tcm = np.zeros((N,), np.float32)
    toB = np.zeros((N,), np.float32)
    toC = np.zeros((N,), np.float32)

    pb = np.zeros((N,), np.float32)
    pt = np.zeros((N,), np.float32)
    pcs = np.zeros((N,), np.float32)
    pcm = np.zeros((N,), np.float32)
    poB = np.zeros((N,), np.float32)
    poC = np.zeros((N,), np.float32)

    for i in range(N):
        x, y = ds[i]
        x = x.unsqueeze(0).to(DEVICE)
        out = model(x)

        tb[i]  = float(y["between_score"].item())
        tt[i]  = float(y["t_on_BC"].item())
        tcs[i] = float(y["closer_sign"].item())
        tcm[i] = float(y["closer_mag"].item())
        toB[i] = float(y["overlap_B"].item())
        toC[i] = float(y["overlap_C"].item())

        pb[i]  = float(torch.sigmoid(out["between"]).item())
        pt[i]  = float(torch.sigmoid(out["tproj"]).item())
        pcs[i] = float(torch.sigmoid(out["csign"]).item())
        pcm[i] = float(torch.sigmoid(out["cmag"]).item())
        poB[i] = float(torch.sigmoid(out["oB"]).item())
        poC[i] = float(torch.sigmoid(out["oC"]).item())

    # Head metrics
    between_mse = float(np.mean((pb - tb)**2))
    tproj_mse   = float(np.mean((pt - tt)**2))
    csign_acc   = float(np.mean((pcs >= 0.5) == (tcs >= 0.5)))
    cmag_mae    = float(np.mean(np.abs(pcm - tcm)))
    oB_acc      = float(np.mean((poB >= 0.5) == (toB >= 0.5)))
    oC_acc      = float(np.mean((poC >= 0.5) == (toC >= 0.5)))

    print(f"[eval-ternary-phase6-256] HEAD between_mse={between_mse:.5f} tproj_mse={tproj_mse:.5f} csign_acc={csign_acc:.4f} cmag_mae={cmag_mae:.4f} oB_acc={oB_acc:.4f} oC_acc={oC_acc:.4f}")

    # Binary geometry checks (these are the real, non-contradictory “is it between / is it overlapping?”)
    true_between = (tb >= 0.70).astype(np.int32)
    pred_between = (pb >= 0.70).astype(np.int32)
    between_acc = float(np.mean(true_between == pred_between))

    true_overlap = ((toB >= 0.5) | (toC >= 0.5)).astype(np.int32)
    pred_overlap = ((poB >= 0.5) | (poC >= 0.5)).astype(np.int32)
    overlap_acc = float(np.mean(true_overlap == pred_overlap))

    print(f"[eval-ternary-phase6-256] BINARY between_acc={between_acc:.4f} overlap_acc={overlap_acc:.4f}")

    # Derived-label confusion (debug view)
    y_true = np.zeros((N,), np.int32)
    y_pred = np.zeros((N,), np.int32)

    for i in range(N):
        y_true[i] = derive_label(tb[i], tt[i], tcs[i], toB[i], toC[i])
        y_pred[i] = derive_label(pb[i], pt[i], pcs[i], poB[i], poC[i])

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(REL_NAMES)))) if _HAS_SK else None
    if cm is not None:
        plot_confusion(
            cm, REL_NAMES,
            os.path.join(OUTDIR, "phase6_confusion_matrix_counts.png"),
            "Phase6 Derived Confusion (counts): (true rows, pred cols)"
        )

        # row-normalized
        cm_row = cm.astype(np.float32)
        row_sums = cm_row.sum(axis=1, keepdims=True) + 1e-9
        cm_row = cm_row / row_sums
        plot_confusion(
            cm_row, REL_NAMES,
            os.path.join(OUTDIR, "phase6_confusion_matrix_row_norm.png"),
            "Phase6 Derived Confusion (row-normalized): (true rows, pred cols)"
        )

    # classification report robust to 0-support classes
    report = None
    macro_f1 = None
    weighted_f1 = None
    acc = float(np.mean(y_true == y_pred))
    if _HAS_SK:
        labels = list(range(len(REL_NAMES)))
        report = classification_report(
            y_true, y_pred,
            labels=labels,
            target_names=REL_NAMES,
            digits=4,
            zero_division=0
        )
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0))

    print(f"[eval-ternary-phase6-256] DERIVED acc = {acc:.4f}")
    if macro_f1 is not None:
        print(f"[eval-ternary-phase6-256] DERIVED macro_f1 = {macro_f1:.4f} | weighted_f1 = {weighted_f1:.4f}")
    if report is not None:
        print(report)

    summary = {
        "phase": 6,
        "N": N,
        "seed": SEED,
        "head": {
            "between_mse": between_mse,
            "tproj_mse": tproj_mse,
            "csign_acc": csign_acc,
            "cmag_mae": cmag_mae,
            "oB_acc": oB_acc,
            "oC_acc": oC_acc
        },
        "binary": {
            "between_acc": between_acc,
            "overlap_acc": overlap_acc
        },
        "derived": {
            "acc": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1
        }
    }

    out_json = os.path.join(OUTDIR, "phase6_eval_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_jsonify)

    print(f"[eval-ternary-phase6-256] saved: {out_json}")
    print("[eval-ternary-phase6-256] done.")

if __name__ == "__main__":
    main()
