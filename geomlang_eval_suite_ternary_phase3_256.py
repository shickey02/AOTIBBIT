#!/usr/bin/env python3
# geomlang_eval_suite_ternary_phase3_256.py
#
# Phase 3 eval:
# - Dataset metrics + classification report
# - Confusion matrices (counts + row-normalized)
# - A-scans -> true/pred/conf/acc PNGs
# - JSON summary

import os, json
import numpy as np
import torch
import matplotlib.pyplot as plt

try:
    from sklearn.metrics import classification_report, f1_score, confusion_matrix
    _HAS_SK = True
except Exception:
    _HAS_SK = False

from geomlang_edges_relternary_train64_latent256_phase3 import (
    IMG_SIZE, DEVICE,
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

TAG = "[eval-ternary-phase3-256]"

EVAL_N = 6000
EVAL_SEED = 123
BATCH_SIZE = 256

SCAN_N = 201
SCAN_MARGIN = 6
SCAN_SHAPES = (0, 0, 0)
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

def save_confusion(cm, out_path, title, normalize_rows=False):
    cm2 = cm.astype(np.float32)
    if normalize_rows:
        denom = cm2.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        cm2 = cm2 / denom

    plt.figure(figsize=(9, 7))
    plt.imshow(cm2, aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("pred")
    plt.ylabel("true")
    plt.xticks(range(len(REL_NAMES)), REL_NAMES, rotation=35, ha="right")
    plt.yticks(range(len(REL_NAMES)), REL_NAMES)

    # annotate
    for i in range(cm2.shape[0]):
        for j in range(cm2.shape[1]):
            if normalize_rows:
                txt = f"{cm2[i,j]:.2f}"
            else:
                txt = f"{int(cm2[i,j])}"
            plt.text(j, i, txt, ha="center", va="center", fontsize=9)

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

    for batch in dl:
        imgs, rel, sA, sB, sC, u_true, dperp_true = batch
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

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(REL_NAMES)))) if _HAS_SK else None

    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "conf_mean": float(confs.mean()),
        "conf_min": float(confs.min()),
        "conf_max": float(confs.max()),
        "report": report,
        "cm": cm,
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

    # ---- Confusion matrices ----
    if ds["cm"] is not None:
        cm = ds["cm"]
        out_counts = os.path.join(OUT_DIR, "phase3_confusion_matrix_counts.png")
        out_norm   = os.path.join(OUT_DIR, "phase3_confusion_matrix_row_norm.png")
        save_confusion(cm, out_counts, "Phase3 Confusion (counts): (true rows, pred cols)", normalize_rows=False)
        save_confusion(cm, out_norm,   "Phase3 Confusion (row-normalized): (true rows, pred cols)", normalize_rows=True)
        print(f"{TAG} saved: {out_counts}")
        print(f"{TAG} saved: {out_norm}")

    # ---- Scans ----
    scans = {}
    for name in SCAN_CONFIGS.keys():
        print(f"{TAG} A-scan: {name}  B={SCAN_CONFIGS[name][0]} C={SCAN_CONFIGS[name][1]}  SCAN_N={SCAN_N}")
        res = eval_ascan(model, name)
        scans[name] = {"mean_acc": res["mean_acc"], "mean_conf": res["mean_conf"]}

        true_img = res["y_true"][None, :]
        pred_img = res["y_pred"][None, :]
        conf_img = res["conf"][None, :]
        acc_img  = res["acc_line"][None, :]

        save_grid_png(true_img, os.path.join(OUT_DIR, f"phase3_scan_{name}_true.png"),
                      f"A-scan {name}: true labels", is_labels=True, n_classes=len(REL_NAMES))
        save_grid_png(pred_img, os.path.join(OUT_DIR, f"phase3_scan_{name}_pred.png"),
                      f"A-scan {name}: predicted labels", is_labels=True, n_classes=len(REL_NAMES))
        save_grid_png(conf_img, os.path.join(OUT_DIR, f"phase3_scan_{name}_conf.png"),
                      f"A-scan {name}: confidence", is_labels=False)
        save_grid_png(acc_img,  os.path.join(OUT_DIR, f"phase3_scan_{name}_acc.png"),
                      f"A-scan {name}: accuracy (mean={res['mean_acc']:.4f})", is_labels=False)

        print(f"{TAG} SCAN {name} mean_acc={res['mean_acc']:.4f} mean_conf={res['mean_conf']:.4f}")

    # ---- Summary ----
    summary = {
        "ckpt_path": CKPT_PATH,
        "dataset": {
            k: v for k, v in ds.items() if k != "cm"
        },
        "scans": scans,
    }
    out_json = os.path.join(OUT_DIR, "phase3_eval_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"{TAG} saved summary: {out_json}")
    print(f"{TAG} done.")

if __name__ == "__main__":
    main()
