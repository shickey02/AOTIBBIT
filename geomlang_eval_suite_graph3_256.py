#!/usr/bin/env python3
# geomlang_eval_suite_graph3_256.py
#
# Eval suite for geomlang_edges_relgraph3_train64_latent256.py
# - Loads checkpoint
# - Dataset eval: per-edge accuracy, per-edge confusion + F1
# - 3 dx/dy grids: test each edge relation geometry independently
# - Saves PNGs + JSON summary into outputs_edges_relgraph3_256/

import os, json
import numpy as np
import torch
import matplotlib.pyplot as plt

from geomlang_edges_relgraph3_train64_latent256 import (
    IMG_SIZE, NUM_CH, LATENT_DIM,
    REL_LEFT, REL_RIGHT, REL_ABOVE, REL_BELOW, REL_OVERLAP,
    REL_NAMES,
    EDGE_PAIRS, N_EDGES,
    GeomEdges64Graph3Dataset,
    SceneModelEdges64Graph3_256,
)

OUT_DIR = "outputs_edges_relgraph3_256"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TAG = "[eval-graph3]"

CKPT_PATH = os.path.join(OUT_DIR, "scene_model_edges_relgraph3_256.pt")

# ---------- Eval dataset settings ----------
EVAL_N = 6000
EVAL_SEED = 123
BATCH_SIZE = 256

# ---------- Grid settings ----------
GRID_N = 201
DX_RANGE = (-24, 24)
DY_RANGE = (-24, 24)
TOL = 2.0

GRID_MARGIN = 6
# fixed shapes/sizes for grid tests
OBJ_SHAPES = [0, 0, 0]   # 0 circle, 1 square
OBJ_SIZES  = [10, 10, 10]

# which edge to grid-test: 0=(0,1), 1=(0,2), 2=(1,2)
# we will test all 3.

# -----------------------
# Helpers (match training)
# -----------------------
def draw_circle(mask, cx, cy, radius):
    H, W = mask.shape
    yy, xx = np.ogrid[:H, :W]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    mask[dist2 <= radius ** 2] = 1.0

def draw_square(mask, cx, cy, half_size):
    H, W = mask.shape
    x0 = max(0, cx - half_size)
    x1 = min(W, cx + half_size + 1)
    y0 = max(0, cy - half_size)
    y1 = min(H, cy + half_size + 1)
    mask[y0:y1, x0:x1] = 1.0

def make_edges(*masks):
    union = np.zeros_like(masks[0], dtype=np.float32)
    for m in masks:
        union = np.maximum(union, (m > 0.5).astype(np.float32))

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

def relation_from_centers(cx_a, cy_a, cx_b, cy_b, tol=TOL):
    dx = cx_b - cx_a
    dy = cy_b - cy_a
    if abs(dx) > abs(dy) + tol:
        return REL_LEFT if dx > 0 else REL_RIGHT
    elif abs(dy) > abs(dx) + tol:
        return REL_ABOVE if dy > 0 else REL_BELOW
    else:
        return REL_OVERLAP

def render_scene_3(centers, shapes, sizes):
    H = W = IMG_SIZE
    objs = [np.zeros((H, W), dtype=np.float32) for _ in range(3)]
    for i in range(3):
        cx, cy = centers[i]
        s = sizes[i]
        if shapes[i] == 0:
            draw_circle(objs[i], cx, cy, s)
        else:
            draw_square(objs[i], cx, cy, s)
    edges = make_edges(objs[0], objs[1], objs[2])
    img = np.stack([objs[0], objs[1], objs[2], edges], axis=0)  # [4,H,W]
    return img

# -----------------------
# Metrics
# -----------------------
def confusion_matrix(y_true, y_pred, n_classes=5):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def f1_per_class(cm):
    f1 = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = (2*tp + fp + fn)
        f1_c = (2*tp / denom) if denom > 0 else 0.0
        f1.append(float(f1_c))
    return f1

def save_confusion_png(cm, out_path, title):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, origin="upper")
    plt.colorbar(label="count")
    plt.xticks(range(5), REL_NAMES, rotation=45, ha="right")
    plt.yticks(range(5), REL_NAMES)
    plt.xlabel("pred")
    plt.ylabel("true")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_grid_png(arr, out_path, title, is_labels=False):
    plt.figure(figsize=(6, 6))
    if is_labels:
        cmap = plt.get_cmap("tab10", 5)
        plt.imshow(arr, origin="lower", cmap=cmap, vmin=-0.5, vmax=4.5)
        plt.colorbar(ticks=range(5), label="relation")
    else:
        plt.imshow(arr, origin="lower")
        plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

@torch.no_grad()
def classify_rel_graph(model, x_bchw):
    """
    x_bchw: [B,4,H,W] float tensor on DEVICE
    returns:
      pred_edges: [B,3] int64 numpy
      conf_edges: [B,3] float32 numpy (max softmax per edge)
    """
    z = model.encode(x_bchw)
    logits = model.rel_graph_head(z).view(-1, N_EDGES, 5)
    probs = torch.softmax(logits, dim=2)  # [B,3,5]
    conf, pred = probs.max(dim=2)
    return pred.detach().cpu().numpy(), conf.detach().cpu().numpy()

# -----------------------
# Dataset evaluation
# -----------------------
@torch.no_grad()
def eval_on_dataset(model):
    ds = GeomEdges64Graph3Dataset(EVAL_N, seed=EVAL_SEED)
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    y_true_edges = [ [] for _ in range(N_EDGES) ]
    y_pred_edges = [ [] for _ in range(N_EDGES) ]
    conf_edges   = [ [] for _ in range(N_EDGES) ]

    for imgs, rel_graph, scale, shapes in dl:
        imgs = imgs.float().to(DEVICE)             # [B,4,64,64]
        rel_graph = rel_graph.cpu().numpy()        # [B,3]

        pred, conf = classify_rel_graph(model, imgs)  # [B,3], [B,3]

        for e in range(N_EDGES):
            y_true_edges[e].append(rel_graph[:, e])
            y_pred_edges[e].append(pred[:, e])
            conf_edges[e].append(conf[:, e])

    metrics = {}
    for e in range(N_EDGES):
        yt = np.concatenate(y_true_edges[e], axis=0)
        yp = np.concatenate(y_pred_edges[e], axis=0)
        cf = np.concatenate(conf_edges[e], axis=0)

        cm = confusion_matrix(yt, yp, n_classes=5)
        f1 = f1_per_class(cm)
        acc = float((yt == yp).mean())

        metrics[f"edge{e}"] = {
            "pair": EDGE_PAIRS[e],
            "acc": acc,
            "f1_per_class": f1,
            "mean_conf": float(cf.mean()),
            "min_conf": float(cf.min()),
            "p05_conf": float(np.quantile(cf, 0.05)),
            "confusion_matrix": cm,
        }

    # simple overall edge-avg accuracy
    avg_acc = float(np.mean([metrics[f"edge{e}"]["acc"] for e in range(N_EDGES)]))
    metrics["avg_edge_acc"] = avg_acc
    return metrics

# -----------------------
# dx/dy grid eval per edge
# -----------------------
@torch.no_grad()
def eval_dxdy_grid_for_edge(model, edge_idx):
    H = W = IMG_SIZE
    a, b = EDGE_PAIRS[edge_idx]

    # base centers: keep all 3 in safe central positions.
    # We'll move object b relative to object a, and keep the 3rd object fixed at center.
    centers = [(W//2, H//2), (W//2, H//2), (W//2, H//2)]
    fixed_idx = 3 - (a + b)  # because {0,1,2} sum = 3
    centers[fixed_idx] = (W//2, H//2)

    # choose anchor for object a (must be drawable)
    max_s = max(OBJ_SIZES)
    cx_a = W // 2
    cy_a = H // 2
    if not (GRID_MARGIN + max_s <= cx_a <= W - GRID_MARGIN - max_s):
        raise RuntimeError("Anchor too close to boundary.")
    if not (GRID_MARGIN + max_s <= cy_a <= H - GRID_MARGIN - max_s):
        raise RuntimeError("Anchor too close to boundary.")

    centers[a] = (cx_a, cy_a)

    dx_vals = np.linspace(DX_RANGE[0], DX_RANGE[1], GRID_N)
    dy_vals = np.linspace(DY_RANGE[0], DY_RANGE[1], GRID_N)

    labels_true = np.zeros((GRID_N, GRID_N), dtype=np.int64)
    labels_pred = np.zeros((GRID_N, GRID_N), dtype=np.int64)
    conf_grid   = np.zeros((GRID_N, GRID_N), dtype=np.float32)

    batch_imgs = []
    batch_pos = []

    def flush():
        if not batch_imgs:
            return
        x = torch.from_numpy(np.stack(batch_imgs, axis=0)).float().to(DEVICE)  # [B,4,H,W]
        pred, conf = classify_rel_graph(model, x)  # [B,3], [B,3]
        for (iy, ix), p, c in zip(batch_pos, pred[:, edge_idx], conf[:, edge_idx]):
            labels_pred[iy, ix] = int(p)
            conf_grid[iy, ix]   = float(c)
        batch_imgs.clear()
        batch_pos.clear()

    for iy, dy in enumerate(dy_vals):
        for ix, dx in enumerate(dx_vals):
            cx_b = int(round(cx_a + dx))
            cy_b = int(round(cy_a + dy))

            # clip so drawable
            s_b = OBJ_SIZES[b]
            cx_b = int(np.clip(cx_b, GRID_MARGIN + s_b, W - GRID_MARGIN - s_b))
            cy_b = int(np.clip(cy_b, GRID_MARGIN + s_b, H - GRID_MARGIN - s_b))

            centers[b] = (cx_b, cy_b)

            # ground truth for that pair
            t = relation_from_centers(cx_a, cy_a, cx_b, cy_b, tol=TOL)
            labels_true[iy, ix] = int(t)

            img = render_scene_3(centers, OBJ_SHAPES, OBJ_SIZES)
            batch_imgs.append(img)
            batch_pos.append((iy, ix))

            if len(batch_imgs) >= BATCH_SIZE:
                flush()

    flush()

    acc_grid = (labels_true == labels_pred).astype(np.float32)
    acc = float(acc_grid.mean())

    return {
        "dx_vals": dx_vals,
        "dy_vals": dy_vals,
        "labels_true": labels_true,
        "labels_pred": labels_pred,
        "conf_grid": conf_grid,
        "acc_grid": acc_grid,
        "acc": acc,
        "pair": (a, b),
    }

# -----------------------
# Main
# -----------------------
def main():
    print(f"{TAG} device = {DEVICE}")
    print(f"{TAG} loading checkpoint: {CKPT_PATH}")

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    if "model_state_dict" not in ckpt:
        raise RuntimeError(f"{TAG} checkpoint missing model_state_dict")

    model = SceneModelEdges64Graph3_256().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ---- Dataset eval ----
    print(f"{TAG} dataset eval (N={EVAL_N}, seed={EVAL_SEED})")
    ds_metrics = eval_on_dataset(model)

    # save per-edge confusion matrices
    for e in range(N_EDGES):
        cm = ds_metrics[f"edge{e}"]["confusion_matrix"]
        pair = ds_metrics[f"edge{e}"]["pair"]
        out_path = os.path.join(OUT_DIR, f"graph3_confusion_edge{e}_pair{pair[0]}{pair[1]}.png")
        save_confusion_png(cm, out_path, title=f"Graph3 Confusion: edge{e} pair{pair} (true rows, pred cols)")
        print(f"{TAG} saved: {out_path}")

    # ---- Grid eval per edge ----
    grid_summaries = {}
    for e in range(N_EDGES):
        pair = EDGE_PAIRS[e]
        print(f"{TAG} dx/dy grid eval for edge{e} pair{pair} (GRID_N={GRID_N})")
        grid = eval_dxdy_grid_for_edge(model, e)

        base = f"graph3_dxdy_edge{e}_pair{pair[0]}{pair[1]}_N{GRID_N}"
        true_path = os.path.join(OUT_DIR, base + "_true.png")
        pred_path = os.path.join(OUT_DIR, base + "_pred.png")
        conf_path = os.path.join(OUT_DIR, base + "_conf.png")
        acc_path  = os.path.join(OUT_DIR, base + "_acc.png")

        save_grid_png(grid["labels_true"], true_path, f"edge{e} {pair}: ground-truth labels", is_labels=True)
        save_grid_png(grid["labels_pred"], pred_path, f"edge{e} {pair}: predicted labels", is_labels=True)
        save_grid_png(grid["conf_grid"],   conf_path,  f"edge{e} {pair}: confidence (max softmax)", is_labels=False)
        save_grid_png(grid["acc_grid"],    acc_path,   f"edge{e} {pair}: accuracy (mean={grid['acc']:.4f})", is_labels=False)

        print(f"{TAG} saved: {true_path}")
        print(f"{TAG} saved: {pred_path}")
        print(f"{TAG} saved: {conf_path}")
        print(f"{TAG} saved: {acc_path}")

        grid_summaries[f"edge{e}"] = {"pair": pair, "acc": grid["acc"]}

    # ---- Summary JSON ----
    summary = {
        "device": str(DEVICE),
        "ckpt_path": CKPT_PATH,
        "dataset_eval": {
            "N": EVAL_N,
            "seed": EVAL_SEED,
            "avg_edge_acc": ds_metrics["avg_edge_acc"],
            "edges": {
                f"edge{e}": {
                    "pair": ds_metrics[f"edge{e}"]["pair"],
                    "acc": ds_metrics[f"edge{e}"]["acc"],
                    "f1_per_class": ds_metrics[f"edge{e}"]["f1_per_class"],
                    "mean_conf": ds_metrics[f"edge{e}"]["mean_conf"],
                    "min_conf": ds_metrics[f"edge{e}"]["min_conf"],
                    "p05_conf": ds_metrics[f"edge{e}"]["p05_conf"],
                    "confusion_matrix": ds_metrics[f"edge{e}"]["confusion_matrix"].tolist(),
                }
                for e in range(N_EDGES)
            }
        },
        "dxdy_grid_eval": {
            "GRID_N": GRID_N,
            "DX_RANGE": DX_RANGE,
            "DY_RANGE": DY_RANGE,
            "TOL": TOL,
            "fixed_shapes": OBJ_SHAPES,
            "fixed_sizes": OBJ_SIZES,
            "edges": grid_summaries,
        }
    }

    summary_path = os.path.join(OUT_DIR, "graph3_eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"{TAG} summary saved: {summary_path}")

    # ---- Console highlights ----
    print(f"{TAG} DATASET avg_edge_acc = {ds_metrics['avg_edge_acc']:.4f}")
    for e in range(N_EDGES):
        pair = ds_metrics[f"edge{e}"]["pair"]
        acc = ds_metrics[f"edge{e}"]["acc"]
        f1  = ds_metrics[f"edge{e}"]["f1_per_class"]
        print(f"{TAG} edge{e} pair{pair} acc={acc:.4f} f1={[round(x,4) for x in f1]}")
    for e in range(N_EDGES):
        print(f"{TAG} GRID edge{e} pair{EDGE_PAIRS[e]} acc={grid_summaries[f'edge{e}']['acc']:.4f}")

if __name__ == "__main__":
    main()
