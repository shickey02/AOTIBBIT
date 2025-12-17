#!/usr/bin/env python3
# phase29b_policy_rollout.py
#
# Roll out a learned edge policy across novelty weights.
#
# FIXES:
# - If --planes not provided, try policy payload 'planes', else infer from cache transport_maps.
# - Robustly loads policy file saved as:
#     A) {"state_dict": ..., "planes": ..., ...}  (trainer v2)
#     B) raw state_dict
# - Output files: phase29b_rollouts_wN=0.350.csv etc (so phase28g visuals finds them).
#
# Requires:
# - cache JSON with 'transport_maps'
# - policy.pt produced by phase29a_train_edge_policy.py

import os, re, csv, json, argparse, math
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Helpers: cache -> edges
# -------------------------

def _split_pair_key(k: str) -> Tuple[str, str]:
    if "__to__" not in k:
        return ("", "")
    a, b = k.split("__to__", 1)
    return a, b

def infer_planes_from_cache(cache_path: str) -> List[str]:
    """
    Looks at the first transport_maps entry and tries to read plane keys.
    """
    data = json.load(open(cache_path, "r"))
    tm = data.get("transport_maps", {})
    if not tm:
        return []
    k = next(iter(tm.keys()))
    entry = tm[k]
    planes_block = entry.get("planes", entry)
    if isinstance(planes_block, dict):
        # likely { 'bo': {...}, 'lt': {...}, ... }
        # keep only keys whose value is dict w/ similarity
        out = []
        for pl, blk in planes_block.items():
            if isinstance(blk, dict) and ("similarity" in blk):
                out.append(pl)
        return sorted(out)
    return []

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
            blk = planes_block.get(pl)
            if not isinstance(blk, dict):
                continue
            sim = blk.get("similarity", None)
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
# Model (must match trainer)
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
        h = self.net(X).squeeze(-1)  # [B,C]
        h = h.masked_fill(mask <= 0, float("-inf"))
        return h

def onehot_plane(plane: str, planes: List[str]) -> List[float]:
    v = [0.0] * len(planes)
    if plane in planes:
        v[planes.index(plane)] = 1.0
    return v

def build_candidate_tensor(cands: List[dict], planes: List[str], w_novelty: float):
    """
    Feature per candidate:
      [sim, w_novelty] + plane_onehot
    Returns X [1,C,D], mask [1,C]
    """
    feats = []
    for e in cands:
        f = [float(e["sim"]), float(w_novelty)]
        f.extend(onehot_plane(e["plane"], planes))
        feats.append(f)
    X = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)  # [1,C,D]
    M = torch.ones((1, len(cands)), dtype=torch.float32)
    return X, M


# -------------------------
# Rollout
# -------------------------

def rollout_one(start: str, edges_by_src: Dict[str, List[dict]], model: nn.Module,
               planes: List[str], w_novelty: float,
               topk_per_node: int, max_steps: int, no_backtrack: bool):
    """
    Returns dict row with path, steps, status, cycle info.
    """
    path = [start]
    prev = None
    seen_at = {}  # node -> first index in path
    status = "halt"

    for t in range(max_steps):
        cur = path[-1]
        cands = edges_by_src.get(cur, [])
        if not cands:
            status = "deadend"
            break

        if topk_per_node and topk_per_node > 0:
            cands = cands[:topk_per_node]

        if no_backtrack and prev is not None:
            cands2 = [e for e in cands if e["dst"] != prev]
            if cands2:
                cands = cands2

        # policy choose
        X, M = build_candidate_tensor(cands, planes, w_novelty)
        with torch.no_grad():
            logits = model(X, M).squeeze(0)  # [C]
            j = int(torch.argmax(logits).item())
        nxt = cands[j]["dst"]

        prev = cur
        path.append(nxt)

        # cycle detect
        if nxt in seen_at:
            status = "cycle"
            break
        seen_at[cur] = len(path) - 2  # index of cur in path

    # compute cycle_key if cycle
    cycle_key = ""
    cycle_len = ""
    if status == "cycle":
        # cycle starts at first occurrence of nxt in path[:-1]
        first = path.index(path[-1])
        cyc_nodes = path[first:-1]  # exclude repeated last
        cycle_len = str(len(cyc_nodes))
        cycle_key = "|".join(sorted(set(cyc_nodes)))

    return {
        "start": start,
        "status": status,
        "steps": str(len(path) - 1),
        "cycle_len": str(cycle_len),
        "cycle_key": cycle_key,
        "path": "->".join(path),
    }


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--planes", nargs="+", default=None)
    ap.add_argument("--min_edge_sim", type=float, default=0.05)
    ap.add_argument("--topk_per_node", type=int, default=12)

    ap.add_argument("--w_list", type=float, nargs="+", required=True)
    ap.add_argument("--max_steps", type=int, default=40)
    ap.add_argument("--no_backtrack", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # load policy checkpoint robustly
    ckpt = torch.load(args.policy, map_location="cpu")
    if isinstance(ckpt, dict) and ("state_dict" in ckpt):
        state_dict = ckpt["state_dict"]
        policy_planes = ckpt.get("planes", None)
        d_in = ckpt.get("d_in", None)
        hidden = ckpt.get("hidden", 64)
    elif isinstance(ckpt, dict):
        # assume raw state_dict
        state_dict = ckpt
        policy_planes = None
        d_in = None
        hidden = 64
    else:
        raise RuntimeError("Unrecognized policy checkpoint format.")

    # resolve planes: CLI -> policy -> cache
    planes = args.planes if args.planes else (policy_planes if policy_planes else infer_planes_from_cache(args.cache))
    if not planes:
        raise RuntimeError("No planes could be resolved (pass --planes bo lt bo_lr).")

    # infer d_in if missing
    if d_in is None:
        # feature = [sim, w] + plane_onehot
        d_in = 2 + len(planes)

    model = EdgeScorer(d_in=d_in, hidden=hidden)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # edges
    edges, edges_by_src = derive_edges_from_transport_maps(args.cache, planes, args.min_edge_sim)
    starts = sorted(edges_by_src.keys())
    print(f"[info] planes={planes} min_edge_sim={args.min_edge_sim}")
    print(f"[info] derived_edges={len(edges)} nodes_with_outgoing={len(edges_by_src)} starts={starts}")

    # rollouts per w
    for w in args.w_list:
        rows = []
        for s in starts:
            rows.append(rollout_one(
                start=s,
                edges_by_src=edges_by_src,
                model=model,
                planes=planes,
                w_novelty=float(w),
                topk_per_node=args.topk_per_node,
                max_steps=args.max_steps,
                no_backtrack=args.no_backtrack,
            ))

        out_csv = os.path.join(args.outdir, f"phase29b_rollouts_wN={float(w):.3f}.csv")
        with open(out_csv, "w", newline="") as f:
            cols = ["start", "status", "steps", "cycle_len", "cycle_key", "path"]
            wr = csv.DictWriter(f, fieldnames=cols)
            wr.writeheader()
            for r in rows:
                wr.writerow(r)

        print(f"[ok] wrote: {out_csv}")


if __name__ == "__main__":
    main()
