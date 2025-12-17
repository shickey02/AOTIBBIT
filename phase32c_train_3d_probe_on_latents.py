#!/usr/bin/env python3
# phase32c_train_3d_probe_on_latents.py
#
# Trains a small probe on top of frozen latents:
#  - regress: dx,dy,dz,dist
#  - classify: left, above, front, ov2d
#
# Inputs:
#   --latents  latents.npy [N,D]
#   --labels   labels.npz  from phase32a
#
# Outputs:
#   outdir/
#     probe.pt
#     metrics.txt
#
# Notes:
# - Supports --val_frac
# - Handles ov2d class imbalance via BCE pos_weight (auto)
# - Saves checkpoint in a PyTorch 2.6 "weights_only"-friendly format
#
import os, argparse, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Model
# -------------------------
class Probe(nn.Module):
    def __init__(self, D, hidden=256, drop=0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(D, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(drop),
        )
        self.reg = nn.Linear(hidden, 4)   # dx,dy,dz,dist (normalized)
        self.cls = nn.Linear(hidden, 4)   # left,above,front,ov2d logits

    def forward(self, z):
        h = self.trunk(z)
        return self.reg(h), self.cls(h)

# -------------------------
# Metrics helpers
# -------------------------
def r2_score(y_true, y_pred, eps=1e-9):
    # y_true/y_pred: [N, K]
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2, axis=0) + eps
    r2 = 1.0 - (ss_res / ss_tot)
    return r2

def prf_from_counts(tp, fp, fn, eps=1e-12):
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    f1 = 2 * p * r / (p + r + eps)
    return float(p), float(r), float(f1)

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # ov2d imbalance handling
    ap.add_argument("--balance_ov2d", action="store_true",
                    help="Use BCE pos_weight for ov2d based on train split imbalance.")
    ap.add_argument("--ov2d_thr", type=float, default=0.50,
                    help="Threshold for ov2d in metrics reporting (training still uses logits).")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    print("[device]", device)

    Z = np.load(args.latents).astype(np.float32)  # [N,D]
    lab = np.load(args.labels)

    # targets
    y_reg = np.stack([lab["dx"], lab["dy"], lab["dz"], lab["dist"]], axis=1).astype(np.float32)
    y_cls = np.stack([lab["left"], lab["above"], lab["front"], lab["ov2d"]], axis=1).astype(np.int64)

    N, D = Z.shape
    print("[data] Z:", Z.shape, "y_reg:", y_reg.shape, "y_cls:", y_cls.shape)

    # split
    idx = np.arange(N)
    rng.shuffle(idx)
    n_val = max(2000, int(args.val_frac * N))
    val_idx = idx[:n_val]
    tr_idx  = idx[n_val:]

    def batch_iter(idxs, bs):
        for i in range(0, len(idxs), bs):
            j = idxs[i:i+bs]
            yield Z[j], y_reg[j], y_cls[j]

    # normalization (regression targets)
    reg_mean_np = y_reg[tr_idx].mean(axis=0, keepdims=True)
    reg_std_np  = y_reg[tr_idx].std(axis=0, keepdims=True) + 1e-6

    # torch tensors for "weights_only" safe saving
    reg_mean_t = torch.from_numpy(reg_mean_np.astype(np.float32))
    reg_std_t  = torch.from_numpy(reg_std_np.astype(np.float32))

    # ov2d pos_weight (train split)
    # pos_weight in BCEWithLogits is weight for positive examples: w = Nneg/Npos
    ov2d_pos_weight = None
    if args.balance_ov2d:
        pos = float(y_cls[tr_idx, 3].sum())
        neg = float(len(tr_idx) - pos)
        w = neg / max(1.0, pos)
        ov2d_pos_weight = w
        print(f"[pos_weight ov2d] pos={int(pos)} neg={int(neg)} w={w:.4f}")
    else:
        print("[pos_weight ov2d] disabled")

    model = Probe(D, hidden=args.hidden, drop=args.drop).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # loss weights
    # keep same mix you were using
    LAMBDA_CLS = 0.6

    def eval_split(idxs):
        model.eval()

        # regression accumulators in original units
        all_true = []
        all_pred = []

        # classification per-label
        cls_correct = np.zeros(4, dtype=np.int64)
        cls_total = 0

        # ov2d prf
        tp = fp = fn = 0

        with torch.no_grad():
            for xb, yr, yc in batch_iter(idxs, args.batch):
                xbt = torch.from_numpy(xb).to(device)
                yct = torch.from_numpy(yc).to(device)

                pred_reg_n, pred_cls_logits = model(xbt)

                # denormalize regression preds to original scale for MAE/R2
                pred_reg = (pred_reg_n.cpu().numpy() * reg_std_np) + reg_mean_np
                all_pred.append(pred_reg)
                all_true.append(yr)

                probs = torch.sigmoid(pred_cls_logits)
                pred = (probs > 0.5).long()

                for k in range(4):
                    cls_correct[k] += int((pred[:, k] == yct[:, k]).sum().item())
                cls_total += int(yct.shape[0])

                # ov2d PRF at custom threshold
                ovp = (probs[:, 3] > float(args.ov2d_thr)).long().cpu().numpy()
                ovt = yct[:, 3].long().cpu().numpy()
                tp += int(((ovp == 1) & (ovt == 1)).sum())
                fp += int(((ovp == 1) & (ovt == 0)).sum())
                fn += int(((ovp == 0) & (ovt == 1)).sum())

        Yt = np.concatenate(all_true, axis=0)
        Yp = np.concatenate(all_pred, axis=0)

        mae = np.mean(np.abs(Yp - Yt), axis=0)
        r2 = r2_score(Yt, Yp)
        r2_total = float(np.mean(r2))

        acc = (cls_correct / max(1, cls_total)).tolist()
        P, R, F1 = prf_from_counts(tp, fp, fn)

        return mae, r2, r2_total, acc, (P, R, F1), (tp, fp, fn)

    best_score = None
    lines = []

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0

        # training
        for xb, yr, yc in batch_iter(tr_idx, args.batch):
            xbt = torch.from_numpy(xb).to(device)

            # normalized regression targets
            yrt_n = torch.from_numpy(((yr - reg_mean_np) / reg_std_np).astype(np.float32)).to(device)

            # BCE expects float targets
            yct_f = torch.from_numpy(yc.astype(np.float32)).to(device)

            pred_reg_n, pred_cls_logits = model(xbt)

            loss_reg = F.mse_loss(pred_reg_n, yrt_n)

            if ov2d_pos_weight is not None:
                pw = torch.tensor([1.0, 1.0, 1.0, float(ov2d_pos_weight)], device=device)
                loss_cls = F.binary_cross_entropy_with_logits(pred_cls_logits, yct_f, pos_weight=pw)
            else:
                loss_cls = F.binary_cross_entropy_with_logits(pred_cls_logits, yct_f)

            loss = loss_reg + LAMBDA_CLS * loss_cls

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += float(loss.item()) * xbt.shape[0]
            seen += int(xbt.shape[0])

        tr_loss = total_loss / max(1, seen)

        # validation
        mae, r2, r2_total, acc, prf, counts = eval_split(val_idx)
        P, R, F1 = prf
        tp, fp, fn = counts

        msg = (
            f"[ep {ep:02d}] train_loss={tr_loss:.4f} "
            f"VAL: MAE(dx,dy,dz,dist)=({mae[0]:.4f}, {mae[1]:.4f}, {mae[2]:.4f}, {mae[3]:.4f}) "
            f"R2(dx,dy,dz,dist)=({r2[0]:.3f}, {r2[1]:.3f}, {r2[2]:.3f}, {r2[3]:.3f}) "
            f"R2_total={r2_total:.3f} "
            f"ACC(left,above,front,ov2d)=({acc[0]:.3f}, {acc[1]:.3f}, {acc[2]:.3f}, {acc[3]:.3f}) "
            f"OV2D@thr{args.ov2d_thr:.2f}(PRF)=({P:.3f}, {R:.3f}, {F1:.3f}) tp/fp/fn={tp}/{fp}/{fn}"
        )
        print(msg)
        lines.append(msg)

        # scoring: emphasize geometry, then ov2d f1 a bit
        score = (1.0 - r2_total) - 0.20 * F1
        if best_score is None or score < best_score:
            best_score = score
            ckpt = {
                "D": int(D),
                "hidden": int(args.hidden),
                "drop": float(args.drop),
                "state_dict": model.state_dict(),
                # store as torch tensors to avoid numpy pickle issues under PyTorch 2.6 defaults
                "reg_mean": reg_mean_t,
                "reg_std": reg_std_t,
                "ov2d_thr": float(args.ov2d_thr),
                "balance_ov2d": bool(args.balance_ov2d),
                "ov2d_pos_weight": float(ov2d_pos_weight) if ov2d_pos_weight is not None else None,
            }
            torch.save(ckpt, os.path.join(args.outdir, "probe.pt"))
            print("[save] probe.pt (best so far)")

    with open(os.path.join(args.outdir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("[ok] wrote metrics.txt and probe.pt")

if __name__ == "__main__":
    main()
