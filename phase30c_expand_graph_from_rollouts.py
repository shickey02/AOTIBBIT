#!/usr/bin/env python3
# phase30c_expand_graph_from_rollouts.py
#
# Step 3A (fixed):
# - Read rollout CSVs
# - Robustly parse node sequences even if stored as "a->b->c" or "a|b|c"
# - Build expanded nodes/edges CSVs
#
# Outputs:
#   phase30c_nodes.csv  (node, visits)
#   phase30c_edges.csv  (src, dst, count)

import os, json, glob, argparse
import pandas as pd
from collections import Counter

BAD_TOKENS = {"", "nan", "none", "null", "cycle", "deadend"}

def _clean_token(x: str) -> str:
    x = str(x).strip()
    return x

def _split_maybe_path(cell: str):
    """
    If the cell looks like a path encoding, split it.
    Supports:
      - "a->b->c"
      - "a|b|c"
      - "a->b|c" (we normalize by splitting on both)
    """
    s = _clean_token(cell)
    if s.lower() in BAD_TOKENS:
        return []

    # if it's a path-like string, split
    if ("->" in s) or ("|" in s):
        # normalize delimiters
        s2 = s.replace("->", "|")
        parts = [p.strip() for p in s2.split("|")]
        parts = [p for p in parts if p and p.lower() not in BAD_TOKENS]
        return parts

    return [s] if s and s.lower() not in BAD_TOKENS else []

def parse_rollout_csv(path):
    df = pd.read_csv(path)

    # Case 1: explicit edges format
    if {"src", "dst"}.issubset(df.columns) or {"u", "v"}.issubset(df.columns):
        s = "src" if "src" in df.columns else "u"
        t = "dst" if "dst" in df.columns else "v"
        paths = []
        for _, r in df.iterrows():
            a = _split_maybe_path(r[s])
            b = _split_maybe_path(r[t])
            if len(a) == 1 and len(b) == 1:
                paths.append([a[0], b[0]])
        return paths, "edges"

    # Case 2: path-in-columns (object dtype columns)
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    if not obj_cols:
        return [], "none"

    paths = []
    for _, r in df.iterrows():
        seq = []
        for c in obj_cols:
            if pd.isna(r[c]):
                continue
            parts = _split_maybe_path(r[c])
            if parts:
                # IMPORTANT: if a cell is itself a whole path, we EXTEND
                seq.extend(parts)

        # drop consecutive repeats
        seq2 = []
        for x in seq:
            if not seq2 or seq2[-1] != x:
                seq2.append(x)

        # require at least 2 nodes for edges
        if len(seq2) >= 2:
            paths.append(seq2)

    return paths, "path"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--rollout_dir", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.cache, "r") as f:
        cache = json.load(f)

    top_keys = list(cache.keys())
    print(f"[cache] top_keys={top_keys}")

    csvs = sorted(glob.glob(os.path.join(args.rollout_dir, "*.csv")))
    print(f"[scan] rollout_dir={args.rollout_dir}")
    print(f"[scan] files total={len(glob.glob(os.path.join(args.rollout_dir,'*')))} csv={len(csvs)}")

    node_counts = Counter()
    edge_counts = Counter()
    edges_total = 0

    for p in csvs:
        paths, mode = parse_rollout_csv(p)
        ecount = 0
        for seq in paths:
            for n in seq:
                node_counts[n] += 1
            for i in range(len(seq)-1):
                a, b = seq[i], seq[i+1]
                if a != b:
                    edge_counts[(a, b)] += 1
                    ecount += 1
        edges_total += ecount
        print(f"[csv] mode={mode:4s} rows={len(paths):4d} file={os.path.basename(p)} edges={ecount}")

    nodes_out = os.path.join(args.outdir, "phase30c_nodes.csv")
    edges_out = os.path.join(args.outdir, "phase30c_edges.csv")

    ndf = pd.DataFrame([{"node": n, "visits": int(c)} for n, c in node_counts.most_common()])
    edf = pd.DataFrame([{"src": s, "dst": t, "count": int(c)} for (s,t), c in edge_counts.most_common()])

    ndf.to_csv(nodes_out, index=False)
    edf.to_csv(edges_out, index=False)

    print(f"[graph] nodes={len(node_counts)} edges(unique)={len(edge_counts)} edges(total)={edges_total}")
    print(f"[out] {nodes_out}")
    print(f"[out] {edges_out}")

if __name__ == "__main__":
    main()
