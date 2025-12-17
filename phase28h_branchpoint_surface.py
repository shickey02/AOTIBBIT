#!/usr/bin/env python3
# phase28h_branchpoint_surface.py (A2.1)
#
# A2.1 change:
#   * compare A vs B by taking MAX logit over all edge-variants whose dst==A/B
#     (this matches rollout greedy behavior when decision is by dst).
#
# Still writes:
#   - phase28h_branch_surface.csv
#   - phase28h_branch_surface.png (delta curve)
#   - phase28h_branch_surface_logits.png (A/B max-logit curves)

import os, json, csv, argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


def _split_pair_key(k: str):
    if "__to__" not in k:
        return ("", "")
    a, b = k.split("__to__", 1)
    return a, b


def derive_edges_from_transport_maps(cache_path, planes, min_edge_sim):
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
            blk = planes_block.get(pl)
            if not isinstance(blk, dict):
                continue
            sim = blk.get("similarity", None)
            if sim is None:
                continue
            sim = float(sim)
            if sim < min_edge_sim:
                continue
            edges.append({"src": src, "dst": dst, "plane": pl, "sim": sim})

    edges_by_src = defaultdict(list)
    for e in edges:
        edges_by_src[e["src"]].append(e)

    for s in list(edges_by_src.keys()):
        edges_by_src[s].sort(key=lambda x: (-x["sim"], x["plane"], x["dst"]))
    return edges_by_src


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
        return h.masked_fill(mask <= 0, float("-inf"))


def onehot_plane(plane, planes):
    v = [0.0] * len(planes)
    if plane in planes:
        v[planes.index(plane)] = 1.0
    return v


def build_X(cands, planes, w_used):
    feats = []
    for e in cands:
        f = [float(e["sim"]), float(w_used)]
        f.extend(onehot_plane(e["plane"], planes))
        feats.append(f)
    X = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)  # [1,C,D]
    M = torch.ones((1, len(cands)), dtype=torch.float32)
    return X, M


def select_candidates(node, edges_by_src, topk, include_all_planes_for_pair, a, b):
    allc = list(edges_by_src.get(node, []))
    if not allc:
        return []
    cands = allc[:topk] if topk > 0 else allc

    if include_all_planes_for_pair:
        add = [e for e in allc if e["dst"] in (a, b)]
        seen = set((e["dst"], e["plane"]) for e in cands)
        for e in add:
            key = (e["dst"], e["plane"])
            if key not in seen:
                cands.append(e)
                seen.add(key)

    cands.sort(key=lambda x: (-x["sim"], x["plane"], x["dst"]))
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--planes", nargs="+", default=None)
    ap.add_argument("--min_edge_sim", type=float, default=0.05)
    ap.add_argument("--topk_per_node", type=int, default=12)

    ap.add_argument("--node", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)

    ap.add_argument("--w_min", type=float, default=0.0)
    ap.add_argument("--w_max", type=float, default=1.0)
    ap.add_argument("--w_step", type=float, default=0.01)

    ap.add_argument("--include_all_planes_for_pair", action="store_true")
    ap.add_argument("--title", type=str, default="28h: flip surface (A2.1 max-over-variants)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    ckpt = torch.load(args.policy, map_location="cpu")
    planes = args.planes if args.planes else ckpt.get("planes", None)
    if not planes:
        raise RuntimeError("No planes resolved. Pass --planes bo lt bo_lr or ensure policy contains planes.")
    w_scale = float(ckpt.get("w_scale", 1.0))
    d_in = int(ckpt.get("d_in", 2 + len(planes)))
    hidden = int(ckpt.get("hidden", 64))

    print(f"[info] policy planes = {planes}")
    print(f"[info] policy w_scale = {w_scale}")

    model = EdgeScorer(d_in=d_in, hidden=hidden)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    edges_by_src = derive_edges_from_transport_maps(args.cache, planes, args.min_edge_sim)
    cands = select_candidates(args.node, edges_by_src, args.topk_per_node,
                              args.include_all_planes_for_pair, args.a, args.b)
    if not cands:
        raise RuntimeError(f"No candidates for node={args.node} with min_edge_sim={args.min_edge_sim}")

    idxA = [i for i,e in enumerate(cands) if e["dst"] == args.a]
    idxB = [i for i,e in enumerate(cands) if e["dst"] == args.b]
    if not idxA or not idxB:
        raise RuntimeError("A or B not present in candidates. Use --include_all_planes_for_pair and/or raise topk.")

    w_vals = np.arange(args.w_min, args.w_max + 1e-12, args.w_step, dtype=float)

    rows = []
    delta = []
    logitA = []
    logitB = []

    for w in w_vals:
        w_used = float(w) * w_scale
        X, M = build_X(cands, planes, w_used)
        with torch.no_grad():
            logits = model(X, M).squeeze(0).cpu().numpy()

        la = float(np.max(logits[idxA]))
        lb = float(np.max(logits[idxB]))
        d  = lb - la

        rows.append([w, w_used, la, lb, d])
        logitA.append(la)
        logitB.append(lb)
        delta.append(d)

    out_csv = os.path.join(args.outdir, "phase28h_branch_surface.csv")
    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["w_novelty","w_used","logitA_max","logitB_max","deltaB_minus_A"])
        for r in rows:
            wr.writerow([f"{r[0]:.3f}", f"{r[1]:.3f}", f"{r[2]:.6f}", f"{r[3]:.6f}", f"{r[4]:.6f}"])

    out_png = os.path.join(args.outdir, "phase28h_branch_surface.png")
    plt.figure(figsize=(9,4.2))
    plt.plot(w_vals, delta, marker="o", markersize=3, linewidth=1)
    plt.axhline(0.0, linewidth=1)
    plt.grid(True, alpha=0.25)
    plt.xlabel("w_novelty")
    plt.ylabel("Δ = max_logit(B) - max_logit(A)")
    plt.title(f"{args.title} (w_scale={w_scale})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    out_png2 = os.path.join(args.outdir, "phase28h_branch_surface_logits.png")
    plt.figure(figsize=(9,4.2))
    plt.plot(w_vals, logitA, marker="o", markersize=3, linewidth=1, label=f"max_logit(A={args.a})")
    plt.plot(w_vals, logitB, marker="o", markersize=3, linewidth=1, label=f"max_logit(B={args.b})")
    plt.grid(True, alpha=0.25)
    plt.xlabel("w_novelty")
    plt.ylabel("logit")
    plt.title(f"Max logits vs w_novelty @ node={args.node} (w_scale={w_scale})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png2, dpi=200)
    plt.close()

    flip_bracket = None
    for i in range(1, len(w_vals)):
        if delta[i-1] < 0 and delta[i] > 0:
            flip_bracket = (w_vals[i-1], w_vals[i], delta[i-1], delta[i])
            break

    if flip_bracket:
        print(f"[flip] bracket [{flip_bracket[0]:.3f},{flip_bracket[1]:.3f}] (delta {flip_bracket[2]:.4f}->{flip_bracket[3]:.4f})")
    else:
        print("[flip] none")

    print(f"[ok] wrote: {out_csv}")
    print(f"[ok] wrote: {out_png}")
    print(f"[ok] wrote: {out_png2}")


if __name__ == "__main__":
    main()
