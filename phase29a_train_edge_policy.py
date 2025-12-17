#!/usr/bin/env python3
# phase29a_train_edge_policy.py
#
# Train an edge-scoring policy from rollout CSVs.
#
# Supports normal mode and flip-only mode.
#
# NEW (A2/A3):
#   --w_scale FLOAT              (scales w feature: w_used = w * w_scale)
#   --report_dataset             (prints class counts + w stats, esp for flip-only)
#   --balance_flip_classes       (flip-only: downsample majority class to match minority)

import os, re, csv, json, argparse, random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Rollout discovery
# -------------------------

_W_RE = re.compile(r"rollouts_wN=([0-9]*\.?[0-9]+)\.csv$", re.IGNORECASE)

def find_rollout_files(indir: str) -> Dict[float, str]:
    files = {}
    if not os.path.isdir(indir):
        return files
    for fn in os.listdir(indir):
        if not fn.lower().endswith(".csv"):
            continue
        if "rollouts_wn=" not in fn.lower():
            continue
        m = _W_RE.search(fn)
        if not m:
            continue
        w = float(m.group(1))
        files[w] = os.path.join(indir, fn)
    return files

def read_csv(path: str) -> List[dict]:
    out = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out.append(r)
    return out

def parse_path_nodes(row: dict) -> List[str]:
    p = (row.get("path") or "").strip()
    if not p:
        return []
    return [x.strip() for x in p.split("->") if x.strip()]


# -------------------------
# Cache -> derived edges
# -------------------------

def _split_pair_key(k: str) -> Tuple[str, str]:
    if "__to__" not in k:
        return ("", "")
    a, b = k.split("__to__", 1)
    return a, b

def derive_edges_from_transport_maps(cache_path: str, planes: List[str], min_edge_sim: float):
    data = json.load(open(cache_path, "r"))
    tm = data.get("transport_maps", {})
    edges = []
    for pair_key, entry in tm.items():
        src, dst = _split_pair_key(pair_key)
        if not src or not dst:
            continue

        planes_block = entry.get("planes", entry)
        if not isinstance(planes_block, dict):
            continue

        for pl in planes:
            pl_block = planes_block.get(pl)
            if not isinstance(pl_block, dict):
                continue
            sim = pl_block.get("similarity", None)
            if sim is None:
                continue
            try:
                sim = float(sim)
            except Exception:
                continue
            if sim < min_edge_sim:
                continue

            edges.append({"src": src, "dst": dst, "plane": pl, "sim": sim})

    edges_by_src = defaultdict(list)
    for e in edges:
        edges_by_src[e["src"]].append(e)

    for src in list(edges_by_src.keys()):
        edges_by_src[src].sort(key=lambda x: (-x["sim"], x["plane"], x["dst"]))

    return edges, edges_by_src


# -------------------------
# Dataset building
# -------------------------

def onehot_plane(plane: str, planes: List[str]) -> List[float]:
    v = [0.0] * len(planes)
    if plane in planes:
        v[planes.index(plane)] = 1.0
    return v

def build_examples(
    rollouts_by_w: Dict[float, List[dict]],
    edges_by_src: Dict[str, List[dict]],
    planes: List[str],
    topk_per_node: int,
    flip_only: bool,
    flip_node: str,
    flip_a: str,
    flip_b: str,
    w_scale: float,
):
    """
    Features per candidate edge:
      [sim, w_used] + plane_onehot
    where w_used = w * w_scale
    """
    X_list, y_list, meta = [], [], []

    def candidates(node: str):
        cand = edges_by_src.get(node, [])
        if topk_per_node and topk_per_node > 0:
            return cand[:topk_per_node]
        return cand

    for w, rows in rollouts_by_w.items():
        w_used = float(w) * float(w_scale)

        for r in rows:
            nodes = parse_path_nodes(r)
            if len(nodes) < 2:
                continue

            for i in range(len(nodes) - 1):
                cur = nodes[i]
                nxt = nodes[i + 1]

                cand = candidates(cur)
                if not cand:
                    continue

                if flip_only:
                    if cur != flip_node:
                        continue
                    if nxt not in (flip_a, flip_b):
                        continue

                    cand = [e for e in cand if e["dst"] in (flip_a, flip_b)]
                    if len({e["dst"] for e in cand}) < 2:
                        continue

                chosen_idx = None
                for j, e in enumerate(cand):
                    if e["dst"] == nxt:
                        chosen_idx = j
                        break
                if chosen_idx is None:
                    continue

                feats = []
                for e in cand:
                    f = [float(e["sim"]), float(w_used)]
                    f.extend(onehot_plane(e["plane"], planes))
                    feats.append(f)

                X = np.asarray(feats, dtype=np.float32)
                y = int(chosen_idx)

                X_list.append(X)
                y_list.append(y)

                meta.append({
                    "w": float(w),
                    "w_used": float(w_used),
                    "cur": cur,
                    "nxt": nxt,
                    "label_dst": cand[chosen_idx]["dst"],
                    "n_cand": X.shape[0],
                })

    if not X_list:
        return None, None, None, None

    maxC = max(x.shape[0] for x in X_list)
    D = X_list[0].shape[1]
    N = len(X_list)

    Xp = np.zeros((N, maxC, D), dtype=np.float32)
    Mp = np.zeros((N, maxC), dtype=np.float32)

    for i, x in enumerate(X_list):
        c = x.shape[0]
        Xp[i, :c, :] = x
        Mp[i, :c] = 1.0

    y = np.asarray(y_list, dtype=np.int64)

    return torch.from_numpy(Xp), torch.from_numpy(Mp), torch.from_numpy(y), meta


def report_flip_dataset(meta: List[dict], flip_a: str, flip_b: str):
    # Count labels by destination + show w stats
    a_ws, b_ws = [], []
    for m in meta:
        if m.get("label_dst") == flip_a:
            a_ws.append(m.get("w", 0.0))
        elif m.get("label_dst") == flip_b:
            b_ws.append(m.get("w", 0.0))

    def _stat(xs):
        if not xs:
            return "none"
        xs = np.asarray(xs, dtype=float)
        return f"n={len(xs)}  w[min/mean/max]=[{xs.min():.3f}/{xs.mean():.3f}/{xs.max():.3f}]"

    print("[dataset] flip-only label distribution:")
    print(f"  A={flip_a:>24s}  {_stat(a_ws)}")
    print(f"  B={flip_b:>24s}  {_stat(b_ws)}")

def balance_flip_only(X, M, y, meta, flip_a: str, flip_b: str, seed: int = 0):
    # Downsample majority class to match minority, based on meta['label_dst']
    idx_a = [i for i,m in enumerate(meta) if m.get("label_dst") == flip_a]
    idx_b = [i for i,m in enumerate(meta) if m.get("label_dst") == flip_b]
    if not idx_a or not idx_b:
        print("[balance] cannot balance: one class has zero samples.")
        return X, M, y, meta

    rng = np.random.RandomState(seed)
    if len(idx_a) > len(idx_b):
        keep_a = rng.choice(idx_a, size=len(idx_b), replace=False).tolist()
        keep = sorted(keep_a + idx_b)
    else:
        keep_b = rng.choice(idx_b, size=len(idx_a), replace=False).tolist()
        keep = sorted(idx_a + keep_b)

    X2 = X[keep]
    M2 = M[keep]
    y2 = y[keep]
    meta2 = [meta[i] for i in keep]
    print(f"[balance] kept {len(keep)} samples (A={sum(m['label_dst']==flip_a for m in meta2)}, "
          f"B={sum(m['label_dst']==flip_b for m in meta2)})")
    return X2, M2, y2, meta2


# -------------------------
# Model
# -------------------------

class EdgeScorer(nn.Module):
    def __init__(self, d_in: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, X, mask):
        h = self.net(X).squeeze(-1)
        h = h.masked_fill(mask <= 0, float("-inf"))
        return h

def masked_accuracy(logits, y):
    pred = torch.argmax(logits, dim=1)
    return (pred == y).float().mean().item()


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--rollout_dir", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--planes", nargs="+", required=True)
    ap.add_argument("--min_edge_sim", type=float, default=0.05)
    ap.add_argument("--topk_per_node", type=int, default=12)

    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--flip_only", action="store_true")
    ap.add_argument("--flip_node", type=str, default="between_deep_clear")
    ap.add_argument("--flip_a", type=str, default="left_clean")
    ap.add_argument("--flip_b", type=str, default="overlap_boundary_only")

    ap.add_argument("--w_scale", type=float, default=1.0)

    ap.add_argument("--train_frac", type=float, default=0.9)
    ap.add_argument("--report_dataset", action="store_true")
    ap.add_argument("--balance_flip_classes", action="store_true")

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.outdir, exist_ok=True)

    edges, edges_by_src = derive_edges_from_transport_maps(args.cache, args.planes, args.min_edge_sim)
    print(f"[info] derived_edges={len(edges)} nodes_with_outgoing={len(edges_by_src)}")

    files_by_w = find_rollout_files(args.rollout_dir)
    if not files_by_w:
        raise SystemExit(f"[error] no rollout CSVs found in {args.rollout_dir} containing 'rollouts_wN='")

    rollouts_by_w = {w: read_csv(fp) for w, fp in sorted(files_by_w.items())}
    print(f"[info] loaded rollouts weights: {sorted(rollouts_by_w.keys())}")
    print(f"[info] w_scale = {args.w_scale}")

    X, mask, y, meta = build_examples(
        rollouts_by_w=rollouts_by_w,
        edges_by_src=edges_by_src,
        planes=args.planes,
        topk_per_node=args.topk_per_node,
        flip_only=args.flip_only,
        flip_node=args.flip_node,
        flip_a=args.flip_a,
        flip_b=args.flip_b,
        w_scale=args.w_scale,
    )
    if X is None:
        mode = "flip-only" if args.flip_only else "full"
        raise SystemExit(f"[error] no training examples produced ({mode}). Try increasing topk_per_node or lowering min_edge_sim.")

    print(f"[info] examples: {X.shape[0]}  d_in={X.shape[2]}  maxC={X.shape[1]}")
    if args.flip_only:
        print(f"[info] flip-only: node={args.flip_node} targets=({args.flip_a},{args.flip_b})")
        if args.report_dataset:
            report_flip_dataset(meta, args.flip_a, args.flip_b)

        if args.balance_flip_classes:
            X, mask, y, meta = balance_flip_only(X, mask, y, meta, args.flip_a, args.flip_b, seed=args.seed)

    # split train/val
    N = X.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)
    ntr = int(args.train_frac * N)
    tr = idx[:ntr]
    va = idx[ntr:] if ntr < N else idx[:0]

    Xtr, Mtr, ytr = X[tr], mask[tr], y[tr]
    Xva, Mva, yva = (X[va], mask[va], y[va]) if len(va) else (None, None, None)

    d_in = X.shape[2]
    model = EdgeScorer(d_in=d_in, hidden=64)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    log_path = os.path.join(args.outdir, "phase29a_train_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "acc_train", "acc_val", "N_train", "N_val"])

    for ep in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(Xtr, Mtr)
        loss = F.cross_entropy(logits, ytr)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            acc_tr = masked_accuracy(model(Xtr, Mtr), ytr)
            acc_va = None
            if Xva is not None and len(va) > 0:
                acc_va = masked_accuracy(model(Xva, Mva), yva)

        if ep == 1 or ep % 20 == 0 or ep == args.epochs:
            if acc_va is None:
                print(f"[ep {ep:4d}] loss={loss.item():.4f} acc={acc_tr:.3f}")
            else:
                print(f"[ep {ep:4d}] loss={loss.item():.4f} acc_tr={acc_tr:.3f} acc_va={acc_va:.3f}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, f"{loss.item():.6f}", f"{acc_tr:.6f}",
                                    "" if acc_va is None else f"{acc_va:.6f}",
                                    len(tr), len(va)])

    out_pt = os.path.join(args.outdir, "phase29a_policy.pt")
    payload = {
        "state_dict": model.state_dict(),
        "planes": args.planes,
        "min_edge_sim": args.min_edge_sim,
        "topk_per_node": args.topk_per_node,
        "flip_only": bool(args.flip_only),
        "flip_node": args.flip_node,
        "flip_a": args.flip_a,
        "flip_b": args.flip_b,
        "w_scale": float(args.w_scale),
        "d_in": d_in,
        "hidden": 64,
    }
    torch.save(payload, out_pt)
    print(f"[ok] wrote: {out_pt}")
    print(f"[ok] wrote: {log_path}")

if __name__ == "__main__":
    main()
