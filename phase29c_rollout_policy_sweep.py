#!/usr/bin/env python3
# phase29c_rollout_policy_sweep.py
#
# Roll out an edge-policy over the transport-cache graph while sweeping novelty weight w.
# Robust to phase15f cache schema:
#   - cache["bases"][node] may be list OR dict with nested list
#   - cache["transport_maps"][src__to__dst][plane] may be dict containing matrices
#
# Outputs:
#   - phase29c_choice_sweep.csv + .png
#   - if --detailed_nodes: per-w node-visit tables

import os, json, glob, math, argparse
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ------------------------
# Policy model (must match phase29a saved state_dict)
# ------------------------
class EdgePolicy(nn.Module):
    def __init__(self, d_in: int, hidden: int = 64):
        super().__init__()
        # Match the 3-layer MLP you used in 29a:
        # Linear(d_in->64) ReLU Linear(64->64) ReLU Linear(64->1)
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, X, M):
        # X: [B,C,D], M: [B,C]
        h = self.net(X).squeeze(-1)  # [B,C]
        h = h.masked_fill(M <= 0, float("-inf"))
        return h


# ------------------------
# Cache helpers (schema-robust)
# ------------------------
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def _try_array(x):
    """Return np.array(x) if it looks numeric; else None."""
    try:
        a = np.array(x)
        if a.dtype == object:
            return None
        if a.size == 0:
            return None
        # accept float/int
        if a.dtype.kind in ("f", "i", "u"):
            return a.astype(np.float32)
        return None
    except Exception:
        return None

def _find_numeric_lists(obj, min_len=8):
    """
    Recursively find numeric lists. Returns list of np arrays (1D or 2D).
    """
    found = []
    if isinstance(obj, list):
        a = _try_array(obj)
        if a is not None and a.size >= min_len:
            found.append(a)
        else:
            for v in obj:
                found.extend(_find_numeric_lists(v, min_len=min_len))
    elif isinstance(obj, dict):
        for v in obj.values():
            found.extend(_find_numeric_lists(v, min_len=min_len))
    return found

def get_base_vec(cache, node):
    """
    Return base vector for node as float32 shape [256].
    Phase15f bases are dicts; probe script showed they contain a list(len=256) somewhere.
    """
    b = cache["bases"][node]
    if isinstance(b, list):
        v = _try_array(b)
        if v is None:
            raise ValueError(f"bases[{node}] list but not numeric")
        v = v.reshape(-1).astype(np.float32)
        return v
    if isinstance(b, dict):
        cands = _find_numeric_lists(b, min_len=200)
        # pick first 256-length 1D
        for a in cands:
            if a.ndim == 1 and a.shape[0] == 256:
                return a.astype(np.float32)
            if a.ndim == 2 and 256 in a.shape:
                # sometimes stored as [1,256] or [256,1]
                aa = a.reshape(-1)
                if aa.shape[0] == 256:
                    return aa.astype(np.float32)
        raise ValueError(f"bases[{node}] dict but no numeric 256-vector found. keys={list(b.keys())[:10]}")
    raise ValueError(f"bases[{node}] unsupported type {type(b)}")

def transport_key(src, dst):
    return f"{src}__to__{dst}"

def get_transport_plane_obj(cache, src, dst, plane):
    tm = cache["transport_maps"].get(transport_key(src, dst))
    if tm is None:
        return None
    return tm.get(plane)

def _extract_plane_mats(plane_obj):
    """
    From a plane_obj (often dict), try to extract:
      - P: projector 256->2 (shape (2,256) or (256,2))
      - T: in-plane transform (shape (2,2)) OR fallback big A (256,256)
    Returns (P, T, A_big) where any can be None.
    """
    P = None
    T = None
    A_big = None

    if plane_obj is None:
        return None, None, None

    # If already numeric matrix
    A = _try_array(plane_obj)
    if A is not None and A.ndim == 2:
        if A.shape == (256, 256):
            A_big = A
            return None, None, A_big
        if A.shape in ((2,256),(256,2)):
            P = A
            return P, None, None
        if A.shape == (2,2):
            T = A
            return None, T, None

    if not isinstance(plane_obj, dict):
        return None, None, None

    # Search all numeric matrices inside dict
    mats = _find_numeric_lists(plane_obj, min_len=4)
    for m in mats:
        if m.ndim != 2:
            continue
        if m.shape == (256, 256) and A_big is None:
            A_big = m
        elif m.shape in ((2,256),(256,2)) and P is None:
            P = m
        elif m.shape == (2,2) and T is None:
            T = m

    return P, T, A_big

def proj_plane(bs, bd, plane_obj):
    """
    Return ||projection|| for this plane, robust to dict schemas.
    """
    d = (bd - bs).astype(np.float32)  # [256]

    P, T, A_big = _extract_plane_mats(plane_obj)

    # Best case: big linear map
    if A_big is not None:
        y = A_big @ d
        return float(np.linalg.norm(y))

    # Next: explicit projector 256->2
    if P is not None:
        if P.shape == (2,256):
            coords = P @ d
        else:  # (256,2)
            coords = P.T @ d

        if T is not None and T.shape == (2,2):
            coords = T @ coords
        return float(np.linalg.norm(coords))

    # Next: in-plane transform only (2x2)
    if T is not None and T.shape == (2,2):
        coords = T @ d[:2]
        return float(np.linalg.norm(coords))

    # Fallback: just use first 2 dims
    return float(np.linalg.norm(d[:2]))


# ------------------------
# Candidate selection / features
# ------------------------
def list_candidates(cache, src, topk):
    # Candidates are all dst with a transport map from src
    out = []
    prefix = f"{src}__to__"
    for k in cache["transport_maps"].keys():
        if k.startswith(prefix):
            dst = k.split("__to__")[1]
            out.append(dst)
    out = sorted(set(out))
    return out[:topk] if topk > 0 else out

def edge_features(cache, planes, src, dst, w_used):
    """
    Feature vector matches policy training expectation:
      [edge_sim, novelty] + per-plane projection magnitudes
    Where novelty is scaled externally by w_used.
    """
    bs = get_base_vec(cache, src)
    bd = get_base_vec(cache, dst)

    # "edge_sim": cosine similarity in base space (simple, stable)
    denom = (np.linalg.norm(bs) * np.linalg.norm(bd) + 1e-9)
    edge_sim = float(np.dot(bs, bd) / denom)

    # "novelty": encourage leaving familiar; we use (1 - edge_sim) scaled by w_used
    novelty = float((1.0 - edge_sim) * w_used)

    feats = [edge_sim, novelty]

    for p in planes:
        pobj = get_transport_plane_obj(cache, src, dst, p)
        feats.append(proj_plane(bs, bd, pobj))

    return np.array(feats, dtype=np.float32)


# ------------------------
# Rollout
# ------------------------
def rollout(cache, model, planes, start, node_of_interest, a_name, b_name,
            min_edge_sim, topk_per_node, w_used,
            max_steps, greedy, temperature,
            restrict_to_pair=False, detailed_nodes=False, rng=None):

    if rng is None:
        rng = np.random.default_rng(0)

    cur = start
    visited = [cur]
    counts = Counter()
    counts[cur] += 1

    # stop conditions bookkeeping
    status_deadend = 0
    status_cycle = 0
    status_max_steps = 0

    for t in range(max_steps):
        cands = list_candidates(cache, cur, topk_per_node)

        # filter by cosine sim if requested
        if min_edge_sim is not None:
            bs = get_base_vec(cache, cur)
            keep = []
            for dst in cands:
                bd = get_base_vec(cache, dst)
                denom = (np.linalg.norm(bs) * np.linalg.norm(bd) + 1e-9)
                sim = float(np.dot(bs, bd) / denom)
                if sim >= float(min_edge_sim):
                    keep.append(dst)
            cands = keep

        if restrict_to_pair:
            cands = [x for x in cands if x in (a_name, b_name)]
            if not cands:
                status_deadend = 1
                break

        if not cands:
            status_deadend = 1
            break

        # Build policy inputs
        X = np.stack([edge_features(cache, planes, cur, dst, w_used) for dst in cands], axis=0)  # [C,D]
        X_t = torch.from_numpy(X[None, :, :])  # [1,C,D]
        M_t = torch.ones((1, X.shape[0]), dtype=torch.float32)

        with torch.no_grad():
            logits = model(X_t, M_t).squeeze(0).cpu().numpy()  # [C]

        if greedy:
            j = int(np.argmax(logits))
        else:
            # softmax sampling
            z = logits / float(max(1e-6, temperature))
            z = z - np.max(z)
            p = np.exp(z)
            p = p / (np.sum(p) + 1e-9)
            j = int(rng.choice(len(cands), p=p))

        nxt = cands[j]
        visited.append(nxt)
        counts[nxt] += 1

        # If we returned to already-seen node -> cycle stop (like your earlier runs)
        if nxt in visited[:-1]:
            status_cycle = 1
            break

        cur = nxt

    else:
        status_max_steps = 1

    # Determine choice at node_of_interest: look at the FIRST step out of that node if start==node
    # If you later generalize start != node, you can search visited for first occurrence.
    chosen = visited[1] if len(visited) > 1 else None

    res = {
        "visited": visited,
        "counts": counts,
        "chosen_top1": chosen,
        "status_deadend": status_deadend,
        "status_cycle": status_cycle,
        "status_max_steps": status_max_steps,
    }
    return res


# ------------------------
# Policy loader
# ------------------------
def load_policy(policy_path):
    ckpt = torch.load(policy_path, map_location="cpu")

    # schema: {"state_dict":..., "planes":[...], "w_scale":...}
    planes = ckpt.get("planes", None)
    if planes is None:
        raise ValueError("policy checkpoint missing 'planes'")

    w_scale = float(ckpt.get("w_scale", 1.0))

    d_in = 2 + len(planes)
    model = EdgePolicy(d_in=d_in, hidden=64)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return model, planes, w_scale


# ------------------------
# Main
# ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--start", required=True)
    ap.add_argument("--node", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)

    ap.add_argument("--min_edge_sim", type=float, default=None)
    ap.add_argument("--topk_per_node", type=int, default=12)

    ap.add_argument("--include_all_planes_for_pair", action="store_true")  # kept for CLI compat; no-op here
    ap.add_argument("--restrict_to_pair", action="store_true")

    ap.add_argument("--w_min", type=float, default=0.0)
    ap.add_argument("--w_max", type=float, default=1.0)
    ap.add_argument("--w_step", type=float, default=0.01)

    ap.add_argument("--N", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=40)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)

    ap.add_argument("--detailed_nodes", action="store_true")
    ap.add_argument("--title", type=str, default="29c rollout sweep")

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    cache = load_json(args.cache)
    model, planes, w_scale = load_policy(args.policy)

    print(f"[info] planes={planes}")
    print(f"[info] w_scale={w_scale}  (w_used = w_novelty * w_scale)")
    print(f"[info] greedy={args.greedy} temp={args.temperature}")
    print(f"[info] restrict_to_pair={args.restrict_to_pair}")
    print(f"[info] detailed_nodes={args.detailed_nodes}")
    print(f"[info] start={args.start} node={args.node} A={args.a} B={args.b}")

    ws = []
    w = args.w_min
    while w <= args.w_max + 1e-12:
        ws.append(round(w, 10))
        w += args.w_step

    rows = []
    rng = np.random.default_rng(0)

    # aggregate node visits per w if requested
    detailed = {}

    for w_novelty in ws:
        w_used = float(w_novelty) * float(w_scale)

        N = int(args.N)
        nA = 0
        nB = 0
        nOther = 0

        dead = 0
        cyc = 0
        mx = 0

        chosen_mode_counter = Counter()
        node_visit_counter = Counter()

        for i in range(N):
            r = rollout(
                cache=cache,
                model=model,
                planes=planes,
                start=args.start,
                node_of_interest=args.node,
                a_name=args.a,
                b_name=args.b,
                min_edge_sim=args.min_edge_sim,
                topk_per_node=args.topk_per_node,
                w_used=w_used,
                max_steps=args.max_steps,
                greedy=args.greedy,
                temperature=args.temperature,
                restrict_to_pair=args.restrict_to_pair,
                detailed_nodes=args.detailed_nodes,
                rng=rng
            )

            chosen = r["chosen_top1"]
            if chosen == args.a:
                nA += 1
            elif chosen == args.b:
                nB += 1
            else:
                nOther += 1

            chosen_mode_counter[chosen] += 1

            dead += r["status_deadend"]
            cyc += r["status_cycle"]
            mx += r["status_max_steps"]

            if args.detailed_nodes:
                node_visit_counter.update(r["counts"])

        pB = (nB / float(N)) if N > 0 else 0.0
        mode = chosen_mode_counter.most_common(1)[0][0]

        print(f"[w={w_novelty:.3f}] N={N} A={nA} B={nB} other={nOther} pB={pB:.3f} mode={mode}")

        rows.append([
            w_novelty, w_used, N,
            nA, nB, nOther, pB,
            str(mode),
            dead, cyc, mx
        ])

        if args.detailed_nodes:
            detailed[w_novelty] = node_visit_counter

    # write CSV
    out_csv = os.path.join(args.outdir, "phase29c_choice_sweep.csv")
    with open(out_csv, "w") as f:
        f.write("w_novelty,w_used,N,N_A,N_B,N_other,pB,chosen_top1_mode,status_deadend,status_cycle,status_max_steps\n")
        for r in rows:
            f.write(",".join([f"{x:.6f}" if isinstance(x, float) else str(x) for x in r]) + "\n")
    print(f"[ok] wrote: {out_csv}")

    # plot
    out_png = os.path.join(args.outdir, "phase29c_choice_sweep.png")
    xs = [r[0] for r in rows]
    ys = [r[6] for r in rows]
    plt.figure()
    plt.plot(xs, ys)
    plt.title(args.title)
    plt.xlabel("w_novelty")
    plt.ylabel("p(B)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    print(f"[ok] wrote: {out_png}")

    # summary
    out_txt = os.path.join(args.outdir, "phase29c_choice_sweep_summary.txt")
    with open(out_txt, "w") as f:
        f.write(f"title: {args.title}\n")
        f.write(f"planes: {planes}\n")
        f.write(f"w_scale: {w_scale}\n")
        f.write(f"start={args.start} node={args.node} A={args.a} B={args.b}\n")
        f.write(f"min_edge_sim={args.min_edge_sim} topk={args.topk_per_node}\n")
        f.write(f"greedy={args.greedy} temp={args.temperature}\n")
        f.write(f"restrict_to_pair={args.restrict_to_pair}\n")
        f.write(f"N={args.N} max_steps={args.max_steps}\n")
    print(f"[ok] wrote: {out_txt}")

    # detailed node visits
    if args.detailed_nodes:
        out_nodes = os.path.join(args.outdir, "phase29c_node_visits_by_w.csv")
        # union of nodes
        all_nodes = set()
        for vc in detailed.values():
            all_nodes |= set(vc.keys())
        all_nodes = sorted(all_nodes)

        with open(out_nodes, "w") as f:
            f.write("w_novelty," + ",".join(all_nodes) + "\n")
            for w_novelty in ws:
                vc = detailed.get(w_novelty, Counter())
                f.write(f"{w_novelty:.6f}," + ",".join(str(vc.get(n, 0)) for n in all_nodes) + "\n")
        print(f"[ok] wrote: {out_nodes}")


if __name__ == "__main__":
    main()
