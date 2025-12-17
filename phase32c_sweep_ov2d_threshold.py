#!/usr/bin/env python3
import os, argparse, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F

# must match Probe architecture
class Probe(nn.Module):
    def __init__(self, D, hidden=256, drop=0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(D, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(drop),
        )
        self.reg = nn.Linear(hidden, 4)
        self.cls = nn.Linear(hidden, 4)
    def forward(self, z):
        h = self.trunk(z)
        return self.reg(h), self.cls(h)

def prf(y_true, y_prob, thr):
    y_hat = (y_prob >= thr).astype(np.int64)
    tp = int(((y_hat==1)&(y_true==1)).sum())
    fp = int(((y_hat==1)&(y_true==0)).sum())
    fn = int(((y_hat==0)&(y_true==1)).sum())
    prec = tp / max(1, tp+fp)
    rec  = tp / max(1, tp+fn)
    f1   = (2*prec*rec) / max(1e-12, prec+rec)
    return prec, rec, f1, tp, fp, fn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--probe", required=True)   # probe.pt
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    Z = np.load(args.latents).astype(np.float32)
    lab = np.load(args.labels)
    y = np.stack([lab["left"], lab["above"], lab["front"], lab["ov2d"]], axis=1).astype(np.int64)
    N,D = Z.shape

    # same split rule as training script
    idx = np.arange(N); rng.shuffle(idx)
    n_val = max(2000, int(0.10*N))
    val_idx = idx[:n_val]

    ck = torch.load(args.probe, map_location="cpu", weights_only=False)
    model = Probe(D, hidden=int(ck["hidden"])).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    with torch.no_grad():
        zt = torch.from_numpy(Z[val_idx]).to(device)
        _, logits = model(zt)
        p = torch.sigmoid(logits).cpu().numpy()  # [Nv,4]

    ov_prob = p[:,3]
    ov_true = y[val_idx,3]

    best = None
    for thr in np.linspace(0.02, 0.98, 49):
        prec, rec, f1, tp, fp, fn = prf(ov_true, ov_prob, float(thr))
        score = f1
        if best is None or score > best[0]:
            best = (score, thr, prec, rec, tp, fp, fn)

    score, thr, prec, rec, tp, fp, fn = best
    print(f"[best ov2d] thr={thr:.3f} F1={score:.3f} P={prec:.3f} R={rec:.3f} tp/fp/fn={tp}/{fp}/{fn}")

if __name__ == "__main__":
    main()
