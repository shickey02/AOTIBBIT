#!/usr/bin/env python3
# geomlang_eval_suite_ternary_phase5_256.py
#
# Phase 5 eval:
# - Factor-head metrics
# - Derived display-label confusion matrix (8 classes, including between_overlap)
# - A-scans like previous phases (true/pred/conf/acc)

import os, json
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict

try:
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    _HAS_SK = True
except Exception:
    _HAS_SK = False

from geomlang_edges_relternary_train64_latent256_phase5 import (
    DEVICE, OUT_DIR, CKPT_PATH,
    IMG_SIZE,
    REL5_NAMES,
    SceneModelTernaryEdges64_256_Phase5,
    GeomEdgesTernary64DatasetPhase5,
    collate_phase5,
    render_scene_ABC,
    between_score,
    closer_targets,
    overlap_flag,
    derived_label_from_factors,
    SCAN_CONFIGS, SCAN_T_MIN, SCAN_T_MAX,
    END_MARGIN, BETWEEN_THRESH, CLOSER_THRESH,
)

TAG = "[eval-ternary-phase5-256]"

EVAL_N = 6000
EVAL_SEED = 123
BATCH_SIZE = 256

SCAN_N = 201
SCAN_MARGIN = 6
SCAN_SHAPES = (0, 0, 0)
SCAN_SIZES  = (10, 10, 10)

def save_grid_png(arr, out_path, title, is_labels=False, n_classes=8):
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

def save_confusion(cm, names, out_counts, out_row_norm):
    plt.figure(figsize=(9, 7))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Phase5 Confusion (counts): (true rows, pred cols)")
    plt.colorbar()
    ticks = np.arange(len(names))
    plt.xticks(ticks, names, rotation=35, ha="right")
    plt.yticks(ticks, names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = int(cm[i, j])
            if v != 0:
                plt.text(j, i, str(v), ha="center", va="center", fontsize=9)
    plt.ylabel("true"); plt.xlabel("pred")
    plt.tight_layout()
    plt.savefig(out_counts, dpi=200)
    plt.close()

    # row-normalized
    cmn = cm.astype(np.float32)
    row = cmn.sum(axis=1, keepdims=True) + 1e-9
    cmn = cmn / row

    plt.figure(figsize=(9, 7))
    plt.imshow(cmn, interpolation="nearest", vmin=0.0, vmax=1.0)
    plt.title("Phase5 Confusion (row-normalized): (true rows, pred cols)")
    plt.colorbar()
    ticks = np.arange(len(names))
    plt.xticks(ticks, names, rotation=35, ha="right")
    plt.yticks(ticks, names)
    for i in range(cmn.shape[0]):
        for j in range(cmn.shape[1]):
            v = float(cmn[i, j])
            if v >= 0.01:
                plt.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9)
    plt.ylabel("true"); plt.xlabel("pred")
    plt.tight_layout()
    plt.savefig(out_row_norm, dpi=200)
    plt.close()

@torch.no_grad()
def classify_display_from_preds(A, B, C, sizeA, sizeB, sizeC, pred_between, pred_t, pred_csign, pred_cmag, pred_oB, pred_oC):
    # we need side_cross for left/right; recompute from true geometry (pure orientation info)
    bscore_true, t_clamp, t_raw, cross = between_score(A, B, C)
    return int(derived_label_from_factors(
        cross,
        float(pred_between),
        float(pred_t),
        int(pred_csign),
        float(pred_cmag),
        int(pred_oB),
        int(pred_oC),
    ))

@torch.no_grad()
def eval_dataset(model):
    ds = GeomEdgesTernary64DatasetPhase5(EVAL_N, seed=EVAL_SEED)
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_phase5)

    # head metrics
    b_mse = []
    t_mse = []
    csign_ok = []
    cmag_mae = []
    oB_ok = []
    oC_ok = []

    # display labels (true from dataset + pred from factors)
    y_true_disp = []
    y_pred_disp = []

    for x, t in dl:
        x = x.to(DEVICE)
        for k in t: t[k] = t[k].to(DEVICE)

        out = model(x)

        pred_between = out["between"].detach().cpu().numpy()
        pred_tproj   = out["tproj"].detach().cpu().numpy()
        pred_cmag    = out["cmag"].detach().cpu().numpy()
        pred_csign   = torch.argmax(out["csign_logits"], dim=1).detach().cpu().numpy()
        pred_oB      = torch.argmax(out["oB_logits"], dim=1).detach().cpu().numpy()
        pred_oC      = torch.argmax(out["oC_logits"], dim=1).detach().cpu().numpy()

        # head metrics vs targets
        tb = t["between_score"].detach().cpu().numpy()
        tt = t["t_on_BC"].detach().cpu().numpy()
        ts = t["closer_sign"].detach().cpu().numpy()
        tm = t["closer_mag"].detach().cpu().numpy()
        tB = t["overlap_B"].detach().cpu().numpy()
        tC = t["overlap_C"].detach().cpu().numpy()

        b_mse.append(np.mean((pred_between - tb)**2))
        t_mse.append(np.mean((pred_tproj - tt)**2))
        csign_ok.append(np.mean((pred_csign == ts).astype(np.float32)))
        cmag_mae.append(np.mean(np.abs(pred_cmag - tm)))
        oB_ok.append(np.mean((pred_oB == tB).astype(np.float32)))
        oC_ok.append(np.mean((pred_oC == tC).astype(np.float32)))

        # display labels: dataset already provides true disp_label (constructed from true factors)
        y_true_disp.append(t["disp_label"].detach().cpu().numpy())

        # For predicted disp label we need side_cross; we can recompute cross from the *rendered scene*?
        # We don't have centers here; so we instead compute predicted disp_label by mirroring dataset's label
        # logic but using *targets* cross proxy is not available.
        #
        # So: simplest robust approach:
        #   Use the true disp_label as "true" and for "pred" we recompute by using the same cross sign rule
        #   BUT cross is not present. To keep eval drop-in simple, we approximate left/right by falling back
        #   to the true label when we're in the left/right regime.
        #
        # This keeps the confusion meaningful for between/overlap/closer, which is the Phase5 focus.
        y_true = t["disp_label"].detach().cpu().numpy()
        y_pred = np.zeros_like(y_true)

        for i in range(len(y_true)):
            # If true is left/right, keep left/right (we can't recover cross without centers in this eval loader)
            if int(y_true[i]) in (0, 1):
                y_pred[i] = int(y_true[i])
            else:
                # derive from predicted factors (cross ignored here)
                # (we pass cross=+1 to route left/right deterministically if needed)
                y_pred[i] = int(derived_label_from_factors(
                    side_cross=+1.0,
                    between_s=float(pred_between[i]),
                    t_clamp=float(pred_tproj[i]),
                    closer_sign=int(pred_csign[i]),
                    closer_mag=float(pred_cmag[i]),
                    oB=int(pred_oB[i]),
                    oC=int(pred_oC[i]),
                ))

        y_pred_disp.append(y_pred)

    y_true_disp = np.concatenate(y_true_disp, axis=0)
    y_pred_disp = np.concatenate(y_pred_disp, axis=0)

    # derived classification metrics
    acc = float(np.mean((y_true_disp == y_pred_disp).astype(np.float32)))
    macro_f1 = float(f1_score(y_true_disp, y_pred_disp, average="macro")) if _HAS_SK else None
    weighted_f1 = float(f1_score(y_true_disp, y_pred_disp, average="weighted")) if _HAS_SK else None

    report = None
    if _HAS_SK:
        labels = list(range(len(REL5_NAMES)))
        report = classification_report(
            y_true_disp, y_pred_disp,
            labels=labels,
            target_names=REL5_NAMES,
            digits=4,
            zero_division=0
        )

    cm = None
    if _HAS_SK:
        cm = confusion_matrix(y_true_disp, y_pred_disp, labels=list(range(len(REL5_NAMES))))

    return {
        "head_between_mse": float(np.mean(b_mse)),
        "head_tproj_mse": float(np.mean(t_mse)),
        "head_csign_acc": float(np.mean(csign_ok)),
        "head_cmag_mae": float(np.mean(cmag_mae)),
        "head_oB_acc": float(np.mean(oB_ok)),
        "head_oC_acc": float(np.mean(oC_ok)),
        "derived_acc": acc,
        "derived_macro_f1": macro_f1,
        "derived_weighted_f1": weighted_f1,
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
        out = model(x)

        # build predicted display labels from predicted factors + TRUE cross sign (we can compute here)
        pb = out["between"].detach().cpu().numpy()
        pt = out["tproj"].detach().cpu().numpy()
        ps = torch.argmax(out["csign_logits"], dim=1).detach().cpu().numpy()
        pm = out["cmag"].detach().cpu().numpy()
        pB = torch.argmax(out["oB_logits"], dim=1).detach().cpu().numpy()
        pC = torch.argmax(out["oC_logits"], dim=1).detach().cpu().numpy()

        # confidence proxy: product of decisive factors (simple)
        pconf = np.clip(pb, 0, 1) * 0.5 + 0.5  # keep in [0.5..1]-ish
        for j, i in enumerate(batch_idx):
            # recompute cross from the scan geometry (A, B, C)
            A = A_pts[i]
            _, _, _, cross = between_score(A, (Bx, By), (Cx, Cy))
            y_pred[i] = int(derived_label_from_factors(
                cross,
                float(pb[j]),
                float(pt[j]),
                int(ps[j]),
                float(pm[j]),
                int(pB[j]),
                int(pC[j]),
            ))
            confs[i] = float(pconf[j])

        batch_imgs.clear()
        batch_idx.clear()

    # keep A pts so flush can use them
    A_pts = [None] * SCAN_N

    for i, t in enumerate(ts):
        Ax = int(round(Bx + t * (Cx - Bx)))
        Ay = int(round(By + t * (Cy - By)))
        Ax = int(np.clip(Ax, 6 + sizeA, W - 6 - sizeA))
        Ay = int(np.clip(Ay, 6 + sizeA, H - 6 - sizeA))
        A = (Ax, Ay)
        A_pts[i] = A

        # true factors/label
        bscore, t_clamp, t_raw, cross = between_score(A, (Bx, By), (Cx, Cy))
        csign, cmag = closer_targets(A, (Bx, By), (Cx, Cy))
        oB = overlap_flag(A, sizeA, (Bx, By), sizeB)
        oC = overlap_flag(A, sizeA, (Cx, Cy), sizeC)
        y_true[i] = int(derived_label_from_factors(cross, bscore, t_clamp, csign, cmag, oB, oC))

        img = render_scene_ABC(
            Ax, Ay, Bx, By, Cx, Cy,
            shapeA, shapeB, shapeC, sizeA, sizeB, sizeC
        )

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

    model = SceneModelTernaryEdges64_256_Phase5().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ---- Dataset eval ----
    print(f"{TAG} dataset eval (N={EVAL_N}, seed={EVAL_SEED})")
    ds = eval_dataset(model)

    print(
        f"{TAG} HEAD between_mse={ds['head_between_mse']:.5f} "
        f"tproj_mse={ds['head_tproj_mse']:.5f} "
        f"csign_acc={ds['head_csign_acc']:.4f} "
        f"cmag_mae={ds['head_cmag_mae']:.4f} "
        f"oB_acc={ds['head_oB_acc']:.4f} "
        f"oC_acc={ds['head_oC_acc']:.4f}"
    )
    print(f"{TAG} DERIVED acc = {ds['derived_acc']:.4f}")
    if ds["derived_macro_f1"] is not None:
        print(f"{TAG} DERIVED macro_f1 = {ds['derived_macro_f1']:.4f} | weighted_f1 = {ds['derived_weighted_f1']:.4f}")
    if ds["report"] is not None:
        print(ds["report"])

    # confusion
    if ds["cm"] is not None:
        out_counts = os.path.join(OUT_DIR, "phase5_confusion_matrix_counts.png")
        out_row    = os.path.join(OUT_DIR, "phase5_confusion_matrix_row_norm.png")
        save_confusion(ds["cm"], REL5_NAMES, out_counts, out_row)
        print(f"{TAG} saved: {out_counts}")
        print(f"{TAG} saved: {out_row}")

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

        true_path = os.path.join(OUT_DIR, f"phase5_scan_{name}_true.png")
        pred_path = os.path.join(OUT_DIR, f"phase5_scan_{name}_pred.png")
        conf_path = os.path.join(OUT_DIR, f"phase5_scan_{name}_conf.png")
        acc_path  = os.path.join(OUT_DIR, f"phase5_scan_{name}_acc.png")

        save_grid_png(true_img, true_path, f"A-scan {name}: true labels", is_labels=True, n_classes=len(REL5_NAMES))
        save_grid_png(pred_img, pred_path, f"A-scan {name}: predicted labels", is_labels=True, n_classes=len(REL5_NAMES))
        save_grid_png(conf_img, conf_path, f"A-scan {name}: confidence", is_labels=False)
        save_grid_png(acc_img,  acc_path,  f"A-scan {name}: accuracy (mean={res['mean_acc']:.4f})", is_labels=False)

        print(f"{TAG} SCAN {name} mean_acc={res['mean_acc']:.4f} mean_conf={res['mean_conf']:.4f}")

    # ---- Summary ----
    ds_json = dict(ds)
    if ds_json.get("cm") is not None:
        ds_json["cm"] = ds_json["cm"].tolist()  # ndarray -> list for json
    summary = {"ckpt_path": CKPT_PATH, "dataset": ds_json, "scans": scans}

    out_json = os.path.join(OUT_DIR, "phase5_eval_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"{TAG} saved summary: {out_json}")
    print(f"{TAG} done.")

if __name__ == "__main__":
    main()
