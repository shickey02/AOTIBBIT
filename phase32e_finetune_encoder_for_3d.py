#!/usr/bin/env python3
# phase32e_finetune_encoder_for_3d.py
#
# Fine-tune the existing Phase7 ConvAEHeads encoder to support true 3D supervision.
# - Loads your existing model ckpt (phase7) and uses its encoder to produce z.
# - Trains a small head on z for:
#     regression: dx,dy,dz,dist
#     classification: left,above,front,ov2d  (BCEWithLogits, with pos_weight for ov2d)
# - Optionally uses a small reconstruction loss to preserve the original manifold.
#
# Inputs:
#   --images images.npy  [N,3,64,64] float32
#   --labels labels.npz  with keys dx,dy,dz,dist,left,above,front,ov2d
#   --encoder_py path to file that defines ConvAEHeads (phase7)
#   --encoder_class typically ConvAEHeads
#   --ckpt path to scene_model_edges_relternary256_phase7.pt
#
# Outputs:
#   outdir/
#     finetuned_model.pt
#     head.pt
#     metrics.txt
#
import os, argparse, importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def load_class_from_py(py_path, class_name):
    spec = importlib.util.spec_from_file_location("encmod", py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = getattr(mod, class_name)
    return cls

class Head3D(nn.Module):
    def __init__(self, D, hidden=256, drop=0.10):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(D, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(drop),
        )
        self.reg = nn.Linear(hidden, 4)  # dx,dy,dz,dist
        self.cls = nn.Linear(hidden, 4)  # left,above,front,ov2d logits

    def forward(self, z):
        h = self.trunk(z)
        return self.reg(h), self.cls(h)

def try_encode(model, x):
    # Handle different model APIs gracefully
    if hasattr(model, "encode"):
        return model.encode(x)
    if hasattr(model, "encoder"):
        return model.encoder(x)
    # last resort: forward may return (recon, heads, z, ...)
    out = model(x)
    if isinstance(out, (tuple, list)):
        # try to find a 2D tensor [B,D]
        for t in out[::-1]:
            if torch.is_tensor(t) and t.dim() == 2:
                return t
    raise RuntimeError("Could not find encoder output z. Add an encode() method or expose model.encoder(x).")

def try_decode(model, z):
    if hasattr(model, "decode"):
        return model.decode(z)
    if hasattr(model, "decoder"):
        return model.decoder(z)
    # if model forward expects x, we can't decode here
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--encoder_py", required=True)
    ap.add_argument("--encoder_class", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)

    ap.add_argument("--lr_head", type=float, default=3e-4)
    ap.add_argument("--lr_enc", type=float, default=1e-5)
    ap.add_argument("--hidden", type=int, default=256)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_frac", type=float, default=0.10)

    ap.add_argument("--recon_weight", type=float, default=0.05, help="0 disables recon preservation")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    print("[device]", device)

    X = np.load(args.images).astype(np.float32)  # [N,3,64,64]
    lab = np.load(args.labels)

    y_reg = np.stack([lab["dx"], lab["dy"], lab["dz"], lab["dist"]], axis=1).astype(np.float32)
    y_cls = np.stack([lab["left"], lab["above"], lab["front"], lab["ov2d"]], axis=1).astype(np.float32)

    N = X.shape[0]
    idx = np.arange(N)
    rng.shuffle(idx)
    n_val = max(2000, int(args.val_frac * N))
    val_idx = idx[:n_val]
    tr_idx  = idx[n_val:]

    # load model
    EncCls = load_class_from_py(args.encoder_py, args.encoder_class)
    model = EncCls().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("[load] missing:", len(missing), "unexpected:", len(unexpected))

    # infer latent dim D
    with torch.no_grad():
        xb = torch.from_numpy(X[tr_idx[:8]]).to(device)
        z = try_encode(model, xb)
        D = int(z.shape[1])
    print("[enc] latent D =", D)

    head = Head3D(D, hidden=args.hidden).to(device)

    # normalize regression targets
    reg_mean = y_reg[tr_idx].mean(axis=0, keepdims=True)
    reg_std  = y_reg[tr_idx].std(axis=0, keepdims=True) + 1e-6

    # pos_weight for ov2d (and optionally others, but yours are balanced)
    # pos_weight = (neg/pos)
    pos = y_cls[tr_idx].sum(axis=0)
    neg = len(tr_idx) - pos
    pos_weight = (neg / (pos + 1e-6)).astype(np.float32)
    # Only ov2d is heavily imbalanced; clamp others near 1
    pos_weight[:3] = 1.0
    pos_weight = torch.from_numpy(pos_weight).to(device)
    print("[pos_weight] left/above/front/ov2d =", pos_weight.detach().cpu().numpy().round(3).tolist())

    # optim: separate lrs (encoder tiny, head normal)
    enc_params = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            enc_params.append(p)

    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": args.lr_enc},
        {"params": head.parameters(), "lr": args.lr_head},
    ], weight_decay=1e-2)

    def batches(idxs, bs):
        for i in range(0, len(idxs), bs):
            j = idxs[i:i+bs]
            yield j

    def eval_split(idxs):
        model.eval(); head.eval()
        mse_sum = 0.0
        cls_correct = np.zeros(4, dtype=np.int64)
        cls_total = 0

        # also report ov2d balanced accuracy-ish via thresholding with pos_weighted training
        with torch.no_grad():
            for j in batches(idxs, args.batch):
                xb = torch.from_numpy(X[j]).to(device)
                yr = torch.from_numpy(((y_reg[j] - reg_mean) / reg_std).astype(np.float32)).to(device)
                yc = torch.from_numpy(y_cls[j].astype(np.float32)).to(device)

                z = try_encode(model, xb)
                pr, pc = head(z)

                mse_sum += float(F.mse_loss(pr, yr, reduction="sum").item())

                pred = (torch.sigmoid(pc) > 0.5).float()
                for k in range(4):
                    cls_correct[k] += int((pred[:,k] == yc[:,k]).sum().item())
                cls_total += int(yc.shape[0])

        mse = mse_sum / max(1, len(idxs))
        acc = (cls_correct / max(1, cls_total)).tolist()
        return mse, acc

    best = None
    lines = []

    for ep in range(1, args.epochs+1):
        model.train(); head.train()
        total = 0.0
        seen = 0

        for j in batches(tr_idx, args.batch):
            xb = torch.from_numpy(X[j]).to(device)
            yr = torch.from_numpy(((y_reg[j] - reg_mean) / reg_std).astype(np.float32)).to(device)
            yc = torch.from_numpy(y_cls[j].astype(np.float32)).to(device)

            z = try_encode(model, xb)
            pr, pc = head(z)

            loss_reg = F.mse_loss(pr, yr)
            loss_cls = F.binary_cross_entropy_with_logits(pc, yc, pos_weight=pos_weight)

            loss = loss_reg + 0.6*loss_cls

            # optional recon preservation (if decoder available)
            if args.recon_weight > 0.0:
                xrec = try_decode(model, z)
                if xrec is not None and torch.is_tensor(xrec):
                    loss_recon = F.mse_loss(xrec, xb)
                    loss = loss + args.recon_weight * loss_recon

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(head.parameters()) + enc_params, 1.0)
            opt.step()

            total += float(loss.item()) * xb.shape[0]
            seen += int(xb.shape[0])

        tr_loss = total / max(1, seen)
        val_mse, val_acc = eval_split(val_idx)

        msg = f"[ep {ep:02d}] train_loss={tr_loss:.4f} val_reg_mse={val_mse:.4f} acc(left,above,front,ov2d)={tuple(round(a,3) for a in val_acc)}"
        print(msg)
        lines.append(msg)

        score = val_mse - 0.05*sum(val_acc)
        if best is None or score < best:
            best = score
            torch.save({"model_state_dict": model.state_dict()}, os.path.join(args.outdir, "finetuned_model.pt"))
            torch.save({
                "D": D,
                "hidden": args.hidden,
                "state_dict": head.state_dict(),
                "reg_mean": reg_mean.astype(np.float32),
                "reg_std": reg_std.astype(np.float32),
            }, os.path.join(args.outdir, "head.pt"))
            print("[save] finetuned_model.pt + head.pt (best so far)")

    with open(os.path.join(args.outdir, "metrics.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("[ok] wrote metrics.txt, finetuned_model.pt, head.pt")

if __name__ == "__main__":
    main()
