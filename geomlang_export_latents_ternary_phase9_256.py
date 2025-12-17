#!/usr/bin/env python3
# geomlang_export_latents_ternary_phase9_256.py
#
# Phase 9 export:
# - Reads Phase 7 encoded_latents / encoded_targets / encoded_preds
# - Reads Phase 8 calibrated thresholds (Tb, To)
# - Derives:
#     y_pred_calibrated (using Tb/To on preds)
#     y_true_reference  (using "true" generator cuts on targets)
# - Saves stable labels for downstream geometry work.

import os, json
import numpy as np

OUTDIR = "outputs_edges_relternary256_phase9"
os.makedirs(OUTDIR, exist_ok=True)

LATENTS = "outputs_edges_relternary256_phase7/encoded_latents_seed123_N6000.npy"
TARGETS = "outputs_edges_relternary256_phase7/encoded_targets_seed123_N6000.npz"
PREDS   = "outputs_edges_relternary256_phase7/encoded_preds_seed123_N6000.npz"

PHASE8_THRESH = "outputs_edges_relternary256_phase8/phase8_thresholds.json"

LABELS = [
    "A_left_of_BtoC",
    "A_right_of_BtoC",
    "A_between_clear",
    "A_between_overlap",
    "A_overlap_only",
]

# "True" reference cut (how we interpret factor targets into discrete buckets)
# These should match how the Phase 7 dataset was bucketed:
TRUE_BETWEEN_T = 0.70
TRUE_OVERLAP_T = 0.50

def sigmoid(x):
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-x))

def _ensure_1d(x, name):
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim != 1:
        raise ValueError(f"{name} should be 1D (N,) but got shape {x.shape}")
    return x

def _npz_get(npz, wanted, aliases):
    keys = list(npz.files)
    for k in [wanted] + list(aliases):
        if k in npz.files:
            return npz[k]
    raise KeyError(f"Could not find '{wanted}' in npz. Tried: {[wanted]+list(aliases)}. Available keys: {keys}")

def derive_from_preds(between_pred, overlap_prob, lr_prob, Tb, To):
    # label ids: 0 left, 1 right, 2 between_clear, 3 between_overlap, 4 overlap_only
    y = []
    for b, o, lr in zip(between_pred, overlap_prob, lr_prob):
        if b >= Tb:
            y.append(3 if o >= To else 2)
        else:
            if o >= To:
                y.append(4)
            else:
                y.append(0 if lr < 0.5 else 1)
    return np.array(y, dtype=np.int64)

def derive_from_targets(between_score, overlap_any, lr_sign):
    # lr_sign in targets is 0/1 (0 = left-of, 1 = right-of) under BC-canonicalization
    y = []
    for b, o, lr in zip(between_score, overlap_any, lr_sign):
        if b >= TRUE_BETWEEN_T:
            y.append(3 if o >= TRUE_OVERLAP_T else 2)
        else:
            if o >= TRUE_OVERLAP_T:
                y.append(4)
            else:
                y.append(0 if lr < 0.5 else 1)
    return np.array(y, dtype=np.int64)

def main():
    print("[export-ternary-phase9-256] loading:")
    print("  latents:", LATENTS)
    print("  targets:", TARGETS)
    print("  preds  :", PREDS)
    print("  thresh :", PHASE8_THRESH)

    assert os.path.exists(LATENTS), f"Missing: {LATENTS}"
    assert os.path.exists(TARGETS), f"Missing: {TARGETS}"
    assert os.path.exists(PREDS),   f"Missing: {PREDS}"
    assert os.path.exists(PHASE8_THRESH), f"Missing: {PHASE8_THRESH}"

    Z = np.load(LATENTS)
    targets = np.load(TARGETS)
    preds   = np.load(PREDS)

    print("[export-ternary-phase9-256] target keys:", list(targets.files))
    print("[export-ternary-phase9-256] pred keys:  ", list(preds.files))
    print("[export-ternary-phase9-256] Z shape =", Z.shape)

    # ---- Load thresholds ----
    with open(PHASE8_THRESH, "r") as f:
        th = json.load(f)
    Tb = float(th["Tb"])
    To = float(th["To"])
    print(f"[export-ternary-phase9-256] using calibrated Tb={Tb:.6f} To={To:.6f}")

    # ---- Pull preds (Phase 7 archive keys) ----
    between_pred = _npz_get(preds, "between_pred", ["between", "between_score_pred", "between_prob"])
    overlap_logit = _npz_get(preds, "overlap_logit", ["overlap", "overlap_any_logit", "o_any_logit", "overlap_pred_logit"])
    lr_logit      = _npz_get(preds, "lr_logit", ["lr", "lr_sign_logit", "lr_pred_logit", "left_right_logit"])

    between_pred = _ensure_1d(between_pred, "between_pred")
    overlap_logit = _ensure_1d(overlap_logit, "overlap_logit")
    lr_logit      = _ensure_1d(lr_logit, "lr_logit")

    overlap_prob = sigmoid(overlap_logit)
    lr_prob      = sigmoid(lr_logit)

    # (between_pred is already in 0..1 in your Phase 7 export; if not, sigmoid it)
    if between_pred.min() < -0.05 or between_pred.max() > 1.05:
        between_pred = sigmoid(between_pred)

    # ---- Pull targets ----
    between_score = _npz_get(targets, "between_score", ["between", "between_true"])
    overlap_any   = _npz_get(targets, "overlap_any", ["overlap", "o_any"])
    lr_sign       = _npz_get(targets, "lr_sign", ["lr", "left_right", "lr_true"])

    between_score = _ensure_1d(between_score, "between_score")
    overlap_any   = _ensure_1d(overlap_any, "overlap_any")
    lr_sign       = _ensure_1d(lr_sign, "lr_sign")

    # ---- Derive labels ----
    y_pred = derive_from_preds(between_pred, overlap_prob, lr_prob, Tb, To)
    y_true = derive_from_targets(between_score, overlap_any, lr_sign)

    # ---- Save outputs ----
    np.save(os.path.join(OUTDIR, "phase9_y_true.npy"), y_true)
    np.save(os.path.join(OUTDIR, "phase9_y_pred.npy"), y_pred)

    # convenience name used by later scripts (treat calibrated labels as "official")
    np.save(os.path.join(OUTDIR, "encoded_labels_seed123_N6000.npy"), y_pred)

    with open(os.path.join(OUTDIR, "phase9_thresholds_used.json"), "w") as f:
        json.dump({"Tb": Tb, "To": To, "labels": LABELS,
                   "TRUE_BETWEEN_T": TRUE_BETWEEN_T, "TRUE_OVERLAP_T": TRUE_OVERLAP_T}, f, indent=2)

    print("[export-ternary-phase9-256] saved ->", OUTDIR)
    print("[export-ternary-phase9-256] y_true counts:", np.bincount(y_true, minlength=len(LABELS)))
    print("[export-ternary-phase9-256] y_pred counts:", np.bincount(y_pred, minlength=len(LABELS)))

if __name__ == "__main__":
    main()
