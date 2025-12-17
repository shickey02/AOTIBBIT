#!/usr/bin/env python3
# geomlang_eval_suite_ternary_phase4_256.py
#
# Phase 4 eval:
# - Head metrics: between MSE, tproj MSE, closer_sign acc, closer_mag MAE, overlap accs
# - Derived 7-class confusion matrices (counts + row-norm) like prior phases
# - A-scans (horiz/vert/diag/etc) saved as thicker bands (not 1-pixel lines)
# - Saves JSON summary

import os, json
import numpy as np
import torch
import matplotlib.pyplot as plt

try:
    from sklearn.metrics import confusion_matrix, classification_report, f1_score
    _HAS_SK = True
except Exception:
    _HAS_SK = False

from geomlang_edges_relternary_train64_latent256_phase4 import (
    IMG_SIZE, DEVICE,
    OUT_DIR, CKPT_PATH,
    GeomEdgesTernary64DatasetPhase4,
    SceneModelTernaryEdges64_256_Phase4,
    render_scene_ABC,
    maybe_rot90_triplet,
    proj_t_and_perp,
    between_score as true_between_score,
    closer_sign_and_mag as true_closer,
    overlap_flag as true_overlap,
    SIZES,
)

TAG = "[eval-ternary-phase4-256]"

EVAL_N = 6000
EVAL_SEED = 123
BATCH_SIZE = 256

# scans (same anchors you used)
SCAN_CONFIGS = {
    "horiz_mid":   ((18, 32), (46, 32)),
    "vert_mid":    ((32, 18), (32, 46)),
    "diag_mid":    ((20, 20), (44, 44)),
    "horiz_short": ((23, 32), (41, 32)),
    "horiz_off":   ((18, 22), (46, 40)),
}
SCAN_N = 201
SCAN_MARGIN = 6
SCAN_SIZES = SIZES

# derived-label thresholds
TAU_OV      = 0.5
TAU_BETWEEN = 0.55
LR_MARGIN   = 0.15   # around t=0.5
# if not overlap/between, decide left/right if far from center; else decide closer
TAU_LR_USE  = 0.20   # abs(t-0.5) > this => left/right; else closer

REL_NAMES = [
    "A_left_of_BtoC",
    "A_right_of_BtoC",
    "A_between_BC",
    "A_closer_to_B",
    "A_closer_to_C",
    "A_overlap_B",
    "A_overlap_C",
]

def save_heat(arr, out_path, title, vmin=None, vmax=None, is_labels=False):
    plt.figure(figsize=(7, 3))
    plt.imshow(arr, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_cm(cm, out_path, title, labels):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, origin="upper")
    plt.title(title)
    plt.xlabel("pred")
    plt.ylabel("true")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.colorbar()
    # annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if isinstance(val, (float, np.floating)):
                txt = f"{val:.2f}"
            else:
                txt = str(int(val))
            plt.text(j, i, txt, ha="center", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

@torch.no_grad()
def forward_heads(model, x_bchw):
    z = model.encode(x_bchw)
    b  = torch.sigmoid(model.between_head(z))  # [0,1]
    t  = torch.sigmoid(model.tproj_head(z))    # [0,1]
    cs = torch.sigmoid(model.cs_head(z))       # [0,1]
    cm = torch.tanh(model.cm_head(z))          # [-1,1]
    oB = torch.sigmoid(model.oB_head(z))       # [0,1]
    oC = torch.sigmoid(model.oC_head(z))       # [0,1]
    return b, t, cs, cm, oB, oC

def derive_label_from_preds(b, t, cs, oB, oC):
    # priority: overlaps -> between -> left/right -> closer
    # b,t,cs,oB,oC are floats (pred probs)
    if oB >= TAU_OV and oB >= oC:
        return 5
    if oC >= TAU_OV and oC > oB:
        return 6
    if b >= TAU_BETWEEN:
        return 2
    # left/right from t along BC
    if abs(t - 0.5) > TAU_LR_USE:
        if t < 0.5 - LR_MARGIN:
            return 0
        if t > 0.5 + LR_MARGIN:
            return 1
    # else closer
    return 3 if cs >= 0.5 else 4

def derive_label_true(A, B, C, sizeA, sizeB, sizeC):
    # compute true targets and apply same derived decision logic
    b = true_between_score(A, B, C)
    t_raw, _ = proj_t_and_perp(A, B, C)
    t = float(np.clip(t_raw, 0.0, 1.0))
    cs, _ = true_closer(A, B, C)
    oB = true_overlap(A, B, sizeA, sizeB)
    oC = true_overlap(A, C, sizeA, sizeC)
    return derive_label_from_preds(b, t, float(cs), float(oB), float(oC))

@torch.no_grad()
def eval_dataset(model):
    ds = GeomEdgesTernary64DatasetPhase4(EVAL_N, seed=EVAL_SEED)
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # head targets / preds
    b_t, t_t, cs_t, cm_t, oB_t, oC_t = [], [], [], [], [], []
    b_p, t_p, cs_p, cm_p, oB_p, oC_p = [], [], [], [], [], []

    # derived labels
    y_true = []
    y_pred = []

    for x, b, t, cs, cm, oB, oC in dl:
        x = x.to(DEVICE)
        bp, tp, csp, cmp, oBp, oCp = forward_heads(model, x)

        b_t.append(b.numpy());   t_t.append(t.numpy());   cs_t.append(cs.numpy());   cm_t.append(cm.numpy());   oB_t.append(oB.numpy());   oC_t.append(oC.numpy())
        b_p.append(bp.cpu().numpy()); t_p.append(tp.cpu().numpy()); cs_p.append(csp.cpu().numpy()); cm_p.append(cmp.cpu().numpy()); oB_p.append(oBp.cpu().numpy()); oC_p.append(oCp.cpu().numpy())

        # derive labels (vectorized)
        bpv  = bp.squeeze(1).cpu().numpy()
        tpv  = tp.squeeze(1).cpu().numpy()
        cspv = csp.squeeze(1).cpu().numpy()
        oBpv = oBp.squeeze(1).cpu().numpy()
        oCpv = oCp.squeeze(1).cpu().numpy()

        # true derived labels from true targets already present in batch
        bt = b.squeeze(1).numpy()
        tt = t.squeeze(1).numpy()
        cst = cs.squeeze(1).numpy()
        oBt = oB.squeeze(1).numpy()
        oCt = oC.squeeze(1).numpy()

        for i in range(len(bpv)):
            y_pred.append(derive_label_from_preds(float(bpv[i]), float(tpv[i]), float(cspv[i]), float(oBpv[i]), float(oCpv[i])))
            y_true.append(derive_label_from_preds(float(bt[i]),  float(tt[i]),  float(cst[i]),  float(oBt[i]),  float(oCt[i])))

    # concat head arrays
    def cat1(xs): return np.concatenate(xs, axis=0).squeeze(1)

    bt = cat1(b_t); tt = cat1(t_t); cst = cat1(cs_t); cmt = cat1(cm_t); oBt = cat1(oB_t); oCt = cat1(oC_t)
    bp = cat1(b_p); tp = cat1(t_p); csp = cat1(cs_p); cmp = cat1(cm_p); oBp = cat1(oB_p); oCp = cat1(oC_p)

    # head metrics
    head = {
        "between_mse": float(np.mean((bp - bt)**2)),
        "tproj_mse":   float(np.mean((tp - tt)**2)),
        "csign_acc":   float(np.mean(((csp >= 0.5).astype(np.float32) == cst.astype(np.float32)).astype(np.float32))),
        "cmag_mae":    float(np.mean(np.abs(cmp - cmt))),
        "overlapB_acc":float(np.mean(((oBp >= 0.5).astype(np.float32) == oBt.astype(np.float32)).astype(np.float32))),
        "overlapC_acc":float(np.mean(((oCp >= 0.5).astype(np.float32) == oCt.astype(np.float32)).astype(np.float32))),
    }

    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)

    acc = float(np.mean(y_true == y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro")) if _HAS_SK else None
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted")) if _HAS_SK else None
    
    labels_all = list(range(len(REL_NAMES)))

    report = classification_report(
        y_true, y_pred,
        labels=labels_all,
        target_names=REL_NAMES,
        digits=4,
        zero_division=0
    ) if _HAS_SK else None

    cm_counts = confusion_matrix(
        y_true, y_pred,
        labels=labels_all
    ) if _HAS_SK else None

    
    cm_row = None
    if cm_counts is not None:
        row_sums = cm_counts.sum(axis=1, keepdims=True) + 1e-9
        cm_row = cm_counts / row_sums

    return {
        "head_metrics": head,
        "derived_acc": acc,
        "derived_macro_f1": macro_f1,
        "derived_weighted_f1": weighted_f1,
        "derived_report": report,
        "cm_counts": cm_counts,
        "cm_row_norm": cm_row,
    }

@torch.no_grad()
def eval_ascan(model, scan_name: str):
    (Bx, By), (Cx, Cy) = SCAN_CONFIGS[scan_name]
    H = W = IMG_SIZE
    sizeA, sizeB, sizeC = SCAN_SIZES

    ts = np.linspace(-0.25, 1.25, SCAN_N)

    y_true = np.zeros((SCAN_N,), dtype=np.int64)
    y_pred = np.zeros((SCAN_N,), dtype=np.int64)
    confs  = np.zeros((SCAN_N,), dtype=np.float32)

    batch_imgs = []
    batch_idx = []

    def flush():
        if not batch_imgs:
            return
        x = torch.from_numpy(np.stack(batch_imgs, axis=0)).float().to(DEVICE)
        b, t, cs, cm, oB, oC = forward_heads(model, x)
        b = b.squeeze(1).cpu().numpy()
        t = t.squeeze(1).cpu().numpy()
        cs = cs.squeeze(1).cpu().numpy()
        oB = oB.squeeze(1).cpu().numpy()
        oC = oC.squeeze(1).cpu().numpy()

        for i, bp, tp, csp, oBp, oCp in zip(batch_idx, b, t, cs, oB, oC):
            lab = derive_label_from_preds(float(bp), float(tp), float(csp), float(oBp), float(oCp))
            y_pred[i] = lab
            # confidence proxy: whichever rule fired
            if lab == 5: confs[i] = float(oBp)
            elif lab == 6: confs[i] = float(oCp)
            elif lab == 2: confs[i] = float(bp)
            elif lab == 0 or lab == 1: confs[i] = float(1.0 - abs(tp - 0.5))  # closer to extremes => lower; center => higher (rough)
            else: confs[i] = float(max(csp, 1.0 - csp))
        batch_imgs.clear()
        batch_idx.clear()

    for i, tlin in enumerate(ts):
        Ax = int(round(Bx + tlin * (Cx - Bx)))
        Ay = int(round(By + tlin * (Cy - By)))
        Ax = int(np.clip(Ax, SCAN_MARGIN + sizeA, W - SCAN_MARGIN - sizeA))
        Ay = int(np.clip(Ay, SCAN_MARGIN + sizeA, H - SCAN_MARGIN - sizeA))

        A = (Ax, Ay); B = (Bx, By); C = (Cx, Cy)
        A2, B2, C2 = maybe_rot90_triplet(A, B, C, p=1.0)

        y_true[i] = derive_label_true(A2, B2, C2, sizeA, sizeB, sizeC)
        img = render_scene_ABC(A2[0], A2[1], B2[0], B2[1], C2[0], C2[1], sizeA, sizeB, sizeC)

        batch_imgs.append(img)
        batch_idx.append(i)
        if len(batch_imgs) >= BATCH_SIZE:
            flush()

    flush()
    acc_line = (y_true == y_pred).astype(np.float32)
    return {
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

    model = SceneModelTernaryEdges64_256_Phase4().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ---- Dataset eval ----
    print(f"{TAG} dataset eval (N={EVAL_N}, seed={EVAL_SEED})")
    ds = eval_dataset(model)

    hm = ds["head_metrics"]
    print(f"{TAG} HEAD between_mse={hm['between_mse']:.5f} tproj_mse={hm['tproj_mse']:.5f} csign_acc={hm['csign_acc']:.4f} cmag_mae={hm['cmag_mae']:.4f} oB_acc={hm['overlapB_acc']:.4f} oC_acc={hm['overlapC_acc']:.4f}")

    print(f"{TAG} DERIVED acc = {ds['derived_acc']:.4f}")
    if ds["derived_macro_f1"] is not None:
        print(f"{TAG} DERIVED macro_f1 = {ds['derived_macro_f1']:.4f} | weighted_f1 = {ds['derived_weighted_f1']:.4f}")
    if ds["derived_report"] is not None:
        print(ds["derived_report"])

    # ---- Confusion matrices ----
    if ds["cm_counts"] is not None:
        p1 = os.path.join(OUT_DIR, "phase4_confusion_matrix_counts.png")
        p2 = os.path.join(OUT_DIR, "phase4_confusion_matrix_row_norm.png")
        save_cm(ds["cm_counts"], p1, "Phase4 Confusion (counts): (true rows, pred cols)", REL_NAMES)
        save_cm(ds["cm_row_norm"], p2, "Phase4 Confusion (row-normalized): (true rows, pred cols)", REL_NAMES)
        print(f"{TAG} saved: {p1}")
        print(f"{TAG} saved: {p2}")

    # ---- Scans ----
    scans = {}
    for name in SCAN_CONFIGS.keys():
        print(f"{TAG} A-scan: {name}  B={SCAN_CONFIGS[name][0]} C={SCAN_CONFIGS[name][1]}  SCAN_N={SCAN_N}")
        res = eval_ascan(model, name)
        scans[name] = {"mean_acc": res["mean_acc"], "mean_conf": res["mean_conf"]}
        print(f"{TAG} SCAN {name} mean_acc={res['mean_acc']:.4f} mean_conf={res['mean_conf']:.4f}")

        # Thicken the 1xN into HxN band for readability
        band_h = 32
        true_img = np.tile(res["y_true"][None, :], (band_h, 1))
        pred_img = np.tile(res["y_pred"][None, :], (band_h, 1))
        conf_img = np.tile(res["conf"][None, :],   (band_h, 1))
        acc_img  = np.tile(res["acc_line"][None,:],(band_h, 1))

        save_heat(true_img, os.path.join(OUT_DIR, f"phase4_scan_{name}_true.png"), f"A-scan {name}: true labels", vmin=-0.5, vmax=6.5)
        save_heat(pred_img, os.path.join(OUT_DIR, f"phase4_scan_{name}_pred.png"), f"A-scan {name}: predicted labels", vmin=-0.5, vmax=6.5)
        save_heat(conf_img, os.path.join(OUT_DIR, f"phase4_scan_{name}_conf.png"), f"A-scan {name}: confidence", vmin=0.0, vmax=1.0)
        save_heat(acc_img,  os.path.join(OUT_DIR, f"phase4_scan_{name}_acc.png"),  f"A-scan {name}: accuracy (mean={res['mean_acc']:.4f})", vmin=0.0, vmax=1.0)

    # ---- Summary ----
    summary = {
        "ckpt_path": CKPT_PATH,
        "head_metrics": ds["head_metrics"],
        "derived": {
            "acc": ds["derived_acc"],
            "macro_f1": ds["derived_macro_f1"],
            "weighted_f1": ds["derived_weighted_f1"],
        },
        "scans": scans,
        "thresholds": {
            "TAU_OV": TAU_OV,
            "TAU_BETWEEN": TAU_BETWEEN,
            "LR_MARGIN": LR_MARGIN,
            "TAU_LR_USE": TAU_LR_USE,
        }
    }
    out_json = os.path.join(OUT_DIR, "phase4_eval_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"{TAG} saved summary: {out_json}")
    print(f"{TAG} done.")

if __name__ == "__main__":
    main()
