#!/usr/bin/env python3
# phase30i_promote_nodes_from_rollouts.py
#
# Promotes nodes visited in rollouts by attaching 256-D vectors from cache where possible.
# (Fixed parsing: splits "a->b->c" and "a|b|c" into true node tokens)

import os, json, glob, argparse
import numpy as np
import pandas as pd
from collections import Counter

LATENT_DIM = 256
BAD_TOKENS = {"", "nan", "none", "null", "cycle", "deadend"}

def _clean_token(x: str) -> str:
    return str(x).strip()

def _split_maybe_path(cell: str):
    s = _clean_token(cell)
    if s.lower() in BAD_TOKENS:
        return []
    if ("->" in s) or ("|" in s):
        s2 = s.replace("->", "|")
        parts = [p.strip() for p in s2.split("|")]
        parts = [p for p in parts if p and p.lower() not in BAD_TOKENS]
        return parts
    return [s] if s and s.lower() not in BAD_TOKENS else []

def is_numeric_list(x, n=None):
    if not isinstance(x, list):
        return False
    if n is not None and len(x) != n:
        return False
    try:
        float(x[0]); float(x[-1])
    except Exception:
        return False
    return True

def coerce_vec(x):
    v = np.array(x, dtype=np.float32)
    if v.ndim != 1 or v.shape[0] != LATENT_DIM:
        return None
    if not np.all(np.isfinite(v)):
        return None
    return v

def find_vec_for_node(cache, node_name):
    bases = cache.get("bases", {})
    if isinstance(bases, dict) and node_name in bases:
        b = bases[node_name]
        if is_numeric_list(b, LATENT_DIM):
            return coerce_vec(b)
        if isinstance(b, dict):
            for k in ["vec", "base", "v", "mu", "center", "z"]:
                if k in b and is_numeric_list(b[k], LATENT_DIM):
                    return coerce_vec(b[k])
            for _, vv in b.items():
                if is_numeric_list(vv, LATENT_DIM):
                    return coerce_vec(vv)

    anchors = cache.get("anchors", {})
    if isinstance(anchors, dict) and node_name in anchors:
        a = anchors[node_name]
        if isinstance(a, dict):
            for k in ["vec", "base", "v", "mu", "center", "z"]:
                if k in a and is_numeric_list(a[k], LATENT_DIM):
                    return coerce_vec(a[k])

    return None

def parse_rollout_csv(path):
    df = pd.read_csv(path)

    if {"src","dst"}.issubset(df.columns) or {"u","v"}.issubset(df.columns):
        s = "src" if "src" in df.columns else "u"
        t = "dst" if "dst" in df.columns else "v"
        paths = []
        for _, r in df.iterrows():
            a = _split_maybe_path(r[s])
            b = _split_maybe_path(r[t])
            if len(a) == 1 and len(b) == 1:
                paths.append([a[0], b[0]])
        return paths

    obj_cols = [c for c in df.columns if df[c].dtype == object]
    if not obj_cols:
        return []

    paths = []
    for _, r in df.iterrows():
        seq = []
        for c in obj_cols:
            if pd.isna(r[c]): continue
            parts = _split_maybe_path(r[c])
            if parts:
                seq.extend(parts)

        # drop consecutive repeats
        seq2 = []
        for x in seq:
            if not seq2 or seq2[-1] != x:
                seq2.append(x)

        if len(seq2) >= 2:
            paths.append(seq2)

    return paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--rollout_dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--min_visits", type=int, default=5)
    ap.add_argument("--max_new", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cache = json.load(open(args.cache, "r"))

    anchors = list(cache.get("anchors", {}).keys())
    anchor_set = set(anchors)

    csvs = sorted(glob.glob(os.path.join(args.rollout_dir, "*.csv")))
    if not csvs:
        raise SystemExit(f"[error] no csvs in {args.rollout_dir}")

    node_counts = Counter()
    edge_counts = Counter()

    for p in csvs:
        paths = parse_rollout_csv(p)
        for seq in paths:
            for n in seq:
                node_counts[n] += 1
            for i in range(len(seq)-1):
                edge_counts[(seq[i], seq[i+1])] += 1

    # candidates to promote
    non_anchor = [(n,c) for n,c in node_counts.items() if n not in anchor_set and c >= args.min_visits]
    non_anchor.sort(key=lambda x: x[1], reverse=True)
    non_anchor = non_anchor[:args.max_new]

    vecs = {}
    missing = []

    # anchors
    for a in anchors:
        v = find_vec_for_node(cache, a)
        if v is None:
            missing.append((a, node_counts.get(a, 0)))
        else:
            vecs[a] = v

    # new
    for n,c in non_anchor:
        v = find_vec_for_node(cache, n)
        if v is None:
            missing.append((n, c))
        else:
            vecs[n] = v

    kept = set(vecs.keys())

    nodes_csv = os.path.join(args.outdir, "phase30i_nodes.csv")
    edges_csv = os.path.join(args.outdir, "phase30i_edges.csv")
    rpt_txt   = os.path.join(args.outdir, "phase30i_report.txt")

    # nodes
    rows = []
    for n in sorted(kept):
        v = vecs[n]
        row = {"node": n}
        for i in range(LATENT_DIM):
            row[f"d{i}"] = float(v[i])
        rows.append(row)
    pd.DataFrame(rows).to_csv(nodes_csv, index=False)

    # edges
    ed = []
    for (s,t), cnt in edge_counts.items():
        if s in kept and t in kept and s != t:
            ed.append({"src": s, "dst": t, "count": int(cnt)})
    pd.DataFrame(ed).to_csv(edges_csv, index=False)

    with open(rpt_txt, "w") as f:
        f.write(f"[cache] anchors={len(anchors)} bases_keys={len(cache.get('bases', {}))}\n")
        f.write(f"[rollouts] csvs={len(csvs)} unique_nodes={len(node_counts)} unique_edges={len(edge_counts)}\n")
        f.write(f"[promote] requested_new={len(non_anchor)} kept_total={len(kept)} (anchors_kept={sum(1 for a in anchors if a in kept)})\n")
        f.write(f"[missing] count={len(missing)}\n")
        for n,c in missing[:200]:
            f.write(f"  - {n} (visits={c})\n")

    print(f"[ok] wrote: {nodes_csv}")
    print(f"[ok] wrote: {edges_csv}")
    print(f"[ok] wrote: {rpt_txt}")
    print(f"[info] kept_nodes={len(kept)} missing={len(missing)}")

if __name__ == "__main__":
    main()
