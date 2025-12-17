#!/usr/bin/env python3
# phase32g_operator_delta_mlp.py
#
# Train an "operator head" that maps delta in latent space to delta in 3D position:
#    input:  dz = z2 - z1
#    target: dxyz = [dx,dy,dz] difference between samples in REAL units
#
# Trains:
#   - linear baseline
#   - small MLP
#
# Reports MAE + R^2 (per-dim and total) in REAL units.

import os, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def r2_total(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean(axis=0, keepdims=True)) ** 2).sum() + 1e-12
    return float(1.0 - ss_res/ss_tot)

def r2_perdim(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0, keepdims=True)) ** 2).sum(axis=0) + 1e-12
    return (1.0 - ss_res/ss_tot).astype(np.float64)

class DeltaMLP(nn.Module):
    def __init__(self, D, hidden=256, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, 3),
        )
    def forward(self, dz):
        return self.net(dz)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--pairs", type=int, default=200000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--zscore", action="store_true", help="z-score latents using train set (recommended)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    print("[device]", device)

    Z = np.load(args.latents).astype(np.float32)  # [N,D]
    lab = np.load(args.labels)
    # IMPORTANT: labels.npz should include actual coordinates for each sample.
    # Phase32a usually stores x1,y1,z1,x2,y2,z2 or similar.
    # We handle both patterns:
    if all(k in lab for k in ["x1","y1","z1","x2","y2","z2"]):
        P1 = np.stack([lab["x1"], lab["y1"], lab["z1"]], axis=1).astype(np.float32)
        P2 = np.stack([lab["x2"], lab["y2"], lab["z2"]], axis=1).astype(np.float32)
        # "scene position" could mean relative vector from obj1 to obj2
        POS = (P2 - P1)  # [N,3]
        print("[labels] using (x2-x1,y2-y1,z2-z1) as per-sample position vector")
    elif all(k in lab for k in ["dx","dy","dz"]):
        POS = np.stack([lab["dx"], lab["dy"], lab["dz"]], axis=1).astype(np.float32)
        print("[labels] using (dx,dy,dz) as per-sample position vector")
    else:
        raise ValueError(f"labels.npz missing needed keys. Found: {list(lab.keys())[:30]}")

    N, D = Z.shape
    print("[data] Z:", Z.shape, "POS:", POS.shape)

    # split indices for samples (not pairs)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_val = max(4000, int(0.10 * N))
    val_idx = idx[:n_val]
    tr_idx  = idx[n_val:]

    # z-score normalize latents using train subset (recommended)
    if args.zscore:
        mu = Z[tr_idx].mean(axis=0, keepdims=True)
        sd = Z[tr_idx].std(axis=0, keepdims=True) + 1e-6
        ZN = (Z - mu) / sd
        np.savez(os.path.join(args.outdir, "zscore_latent_stats.npz"), mu=mu.astype(np.float32), sd=sd.astype(np.float32))
        print("[zscore] enabled, wrote zscore_latent_stats.npz")
    else:
        ZN = Z

    # Build random pairs
    def sample_pairs(idxs, M):
        a = rng.choice(idxs, size=M, replace=True)
        b = rng.choice(idxs, size=M, replace=True)
        dz = ZN[b] - ZN[a]           # [M,D]
        dpos = POS[b] - POS[a]       # [M,3]
        return dz.astype(np.float32), dpos.astype(np.float32)

    # Pre-sample train/val pairs once (stable eval)
    Mtr = int(args.pairs)
    Mva = max(50000, int(0.25 * Mtr))
    dz_tr, dpos_tr = sample_pairs(tr_idx, Mtr)
    dz_va, dpos_va = sample_pairs(val_idx, Mva)
    print("[pairs] train:", dz_tr.shape, "val:", dz_va.shape)

    # -------- linear baseline (closed form ridge-ish via lstsq) --------
    # Solve dz @ W ≈ dpos
    # Add tiny Tikhonov for stability: solve (X^T X + lam I)W = X^T Y
    lam = 1e-3
    X = dz_tr
    Y = dpos_tr
    XtX = X.T @ X
    XtY = X.T @ Y
    W = np.linalg.solve(XtX + lam*np.eye(D, dtype=np.float32), XtY).astype(np.float32)  # [D,3]
    b = Y.mean(axis=0, keepdims=True) - X.mean(axis=0, keepdims=True) @ W
    np.savez(os.path.join(args.outdir, "linear_delta_fit.npz"), W=W, b=b.astype(np.float32), lam=np.array(lam, np.float32))
    print("[linear] wrote linear_delta_fit.npz")

    def eval_linear(dz, dpos):
        pred = dz @ W + b
        mae = np.mean(np.abs(pred - dpos), axis=0)
        r2d = r2_perdim(dpos, pred)
        r2t = r2_total(dpos, pred)
        return mae, r2d, r2t

    lin_mae, lin_r2d, lin_r2t = eval_linear(dz_va, dpos_va)
    print(f"[linear] VAL MAE={tuple(float(f'{m:.4f}') for m in lin_mae)} R2_dim={tuple(float(f'{x:.3f}') for x in lin_r2d)} R2_total={lin_r2t:.3f}")

    # -------- MLP operator --------
    model = DeltaMLP(D, hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # normalize targets for training stability (but we eval in real units)
    t_mu = dpos_tr.mean(axis=0, keepdims=True)
    t_sd = dpos_tr.std(axis=0, keepdims=True) + 1e-6

    def batches(X, Y, bs):
        for i in range(0, X.shape[0], bs):
            yield X[i:i+bs], Y[i:i+bs]

    best = None
    for ep in range(1, args.epochs+1):
        model.train()
        total = 0.0
        seen = 0
        # shuffle pair rows
        perm = rng.permutation(dz_tr.shape[0])
        Xs = dz_tr[perm]
        Ys = dpos_tr[perm]

        for xb, yb in batches(Xs, Ys, args.batch):
            xbt = torch.from_numpy(xb).to(device)
            ynt = torch.from_numpy(((yb - t_mu) / t_sd).astype(np.float32)).to(device)

            pred_n = model(xbt)
            loss = F.mse_loss(pred_n, ynt)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += float(loss.item()) * xbt.shape[0]
            seen += int(xbt.shape[0])

        # eval
        model.eval()
        with torch.no_grad():
            xvt = torch.from_numpy(dz_va).to(device)
            pred_n = model(xvt).cpu().numpy()
        pred = pred_n * t_sd + t_mu  # back to REAL units

        mae = np.mean(np.abs(pred - dpos_va), axis=0)
        r2d = r2_perdim(dpos_va, pred)
        r2t = r2_total(dpos_va, pred)

        msg = f"[mlp ep {ep:02d}] train_mse={total/max(1,seen):.4f} VAL MAE={tuple(float(f'{m:.4f}') for m in mae)} R2_dim={tuple(float(f'{x:.3f}') for x in r2d)} R2_total={r2t:.3f}"
        print(msg)

        score = -r2t + 0.02*float(mae.mean())
        if best is None or score < best:
            best = score
            torch.save({
                "D": D,
                "hidden": args.hidden,
                "state_dict": model.state_dict(),
                "t_mu": t_mu.astype(np.float32),
                "t_sd": t_sd.astype(np.float32),
                "zscore": bool(args.zscore),
            }, os.path.join(args.outdir, "delta_mlp.pt"))
            print("[save] delta_mlp.pt (best so far)")

    print("[ok] done. outputs in:", args.outdir)

if __name__ == "__main__":
    main()
