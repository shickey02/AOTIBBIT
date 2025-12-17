#!/usr/bin/env python3
# phase28g_bifurcation_visuals.py
#
# Robust bifurcation visuals:
# - Accept rollout CSVs named like: phase26g_rollouts_wN=0.350.csv, phase29b_rollouts_wN=0.350.csv, etc.
# - Writes:
#   * phase28g_attractor_matrix.csv
#   * phase28g_bifurcation_scatter.png
#   * phase28g_flip_edges.csv
#   * phase28g_edge_heatmap.png        (skips safely if empty)
#   * phase28g_cycle_strip_<start>.png (per-start strip; optional focus_start)

import os, re, csv, argparse
from collections import defaultdict, Counter

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------
# File discovery
# ---------------------------

_W_RE = re.compile(r"rollouts_wN=([0-9]*\.?[0-9]+)\.csv$", re.IGNORECASE)

def find_rollout_files(indir: str):
    """
    Returns dict: w(float) -> fullpath
    Accepts any prefix as long as filename contains 'rollouts_wN=' and ends with '.csv'
    """
    files_by_w = {}
    if not os.path.isdir(indir):
        return files_by_w

    for fn in os.listdir(indir):
        if not fn.lower().endswith(".csv"):
            continue
        if "rollouts_wN=" not in fn:
            continue
        m = _W_RE.search(fn)
        if not m:
            continue
        w = float(m.group(1))
        files_by_w[w] = os.path.join(indir, fn)
    return files_by_w


# ---------------------------
# CSV parsing helpers
# ---------------------------

def read_rollout_csv(path: str):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows

def get_cycle_key(row: dict):
    return (row.get("cycle_key")
            or row.get("attractor_key")
            or row.get("attractor_sig")
            or row.get("attractor")
            or "")

def get_path_nodes(row: dict):
    p = (row.get("path") or "").strip()
    if not p:
        return []
    return [x.strip() for x in p.split("->") if x.strip()]

def divergence_edge(base_nodes, nov_nodes):
    """
    Returns (div_index, from_node, base_next, nov_next) for first divergence.
    If identical, returns None.
    """
    n = min(len(base_nodes), len(nov_nodes))
    div = None
    for i in range(n):
        if base_nodes[i] != nov_nodes[i]:
            div = i
            break
    if div is None:
        if len(base_nodes) == len(nov_nodes):
            return None
        div = n

    if div <= 0:
        return (div, "(start)",
                base_nodes[0] if base_nodes else "(empty)",
                nov_nodes[0] if nov_nodes else "(empty)")

    from_node = base_nodes[div - 1]
    base_next = base_nodes[div] if div < len(base_nodes) else "(end)"
    nov_next  = nov_nodes[div] if div < len(nov_nodes) else "(end)"
    return (div, from_node, base_next, nov_next)

def edge_label(from_node, base_next, nov_next):
    return f"{from_node}: {base_next} -> {nov_next}"


# ---------------------------
# Plot helpers
# ---------------------------

def plot_cycle_strip(out_png, title, w_vals, cycle_keys, max_label=70):
    """
    Simple per-start strip:
      x-axis: w_novelty
      y-axis: discrete cycle id (integer)
      annotations: cycle key (truncated)
    """
    # map cycle key -> integer id (stable order by first appearance in w order)
    key_to_id = {}
    y = []
    for k in cycle_keys:
        if k not in key_to_id:
            key_to_id[k] = len(key_to_id)
        y.append(key_to_id[k])

    plt.figure(figsize=(10, 2.6))
    plt.plot(w_vals, y, marker="o")
    plt.yticks(list(key_to_id.values()),
               [ (k if len(k) <= max_label else k[:max_label-3] + "...") for k in key_to_id.keys() ])
    plt.xlabel("w_novelty")
    plt.title(title)
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--baseline", type=float, required=True)
    ap.add_argument("--w_list", type=float, nargs="+", required=True)
    ap.add_argument("--top_edges", type=int, default=15)
    ap.add_argument("--title", type=str, default="Bifurcation Visuals")
    ap.add_argument("--focus_start", type=str, default="", help="If set, only generate cycle strip for this start")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    files_by_w = find_rollout_files(args.indir)
    if not files_by_w:
        raise SystemExit(f"[error] No rollout CSVs found in {args.indir} containing 'rollouts_wN='")

    # resolve floats robustly (0.1 vs 0.10)
    def resolve_w(w):
        if w in files_by_w:
            return w
        for ww in files_by_w.keys():
            if abs(ww - w) < 1e-9:
                return ww
        return None

    w_resolved = []
    for w in args.w_list:
        rw = resolve_w(w)
        if rw is None:
            print(f"[warn] missing rollout for w={w} (skipping)")
            continue
        w_resolved.append(rw)
    w_resolved = sorted(set(w_resolved))

    base_w = resolve_w(args.baseline)
    if base_w is None:
        raise SystemExit(f"[error] baseline w={args.baseline} missing from indir")
    if base_w not in w_resolved:
        w_resolved = sorted(set(w_resolved + [base_w]))

    # load rows
    rollouts_by_w = {w: read_rollout_csv(files_by_w[w]) for w in w_resolved}

    # common starts (intersection so comparisons are apples-to-apples)
    start_sets = []
    for w in w_resolved:
        start_sets.append(set((r.get("start") or "") for r in rollouts_by_w[w] if (r.get("start") or "")))
    starts = sorted(set.intersection(*start_sets)) if start_sets else []
    if not starts:
        raise SystemExit("[error] no common starts across provided CSVs")

    if args.focus_start:
        if args.focus_start not in starts:
            raise SystemExit(f"[error] focus_start='{args.focus_start}' not in common starts: {starts}")
        starts_for_strip = [args.focus_start]
    else:
        starts_for_strip = starts

    # index by (w,start)
    row_by_ws = {}
    for w in w_resolved:
        for r in rollouts_by_w[w]:
            s = r.get("start") or ""
            if s in starts:
                row_by_ws[(w, s)] = r

    # 1) attractor matrix
    mat_csv = os.path.join(args.outdir, "phase28g_attractor_matrix.csv")
    with open(mat_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["start"] + [f"{w:.3f}" for w in w_resolved])
        for s in starts:
            wr.writerow([s] + [get_cycle_key(row_by_ws[(w, s)]) for w in w_resolved])
    print(f"[ok] wrote: {mat_csv}")

    # 1b) per-start cycle strip(s) — THIS is the key new visual
    for s in starts_for_strip:
        out_png = os.path.join(args.outdir, f"phase28g_cycle_strip_{s}.png")
        cycles = [get_cycle_key(row_by_ws[(w, s)]) for w in w_resolved]
        plot_cycle_strip(
            out_png,
            title=f"{args.title} — cycle strip (start={s})",
            w_vals=w_resolved,
            cycle_keys=cycles,
        )
        print(f"[ok] wrote: {out_png}")

    # 2) agreement scatter vs baseline
    agree = {}
    for w in w_resolved:
        if w == base_w:
            agree[w] = 1.0
            continue
        good = 0
        for s in starts:
            if get_cycle_key(row_by_ws[(w, s)]) == get_cycle_key(row_by_ws[(base_w, s)]):
                good += 1
        agree[w] = good / max(1, len(starts))

    scatter_png = os.path.join(args.outdir, "phase28g_bifurcation_scatter.png")
    plt.figure()
    xs = w_resolved
    ys = [agree[w] for w in xs]
    plt.plot(xs, ys, marker="o")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("w_novelty")
    plt.ylabel("agreement vs baseline")
    plt.title(args.title)
    plt.tight_layout()
    plt.savefig(scatter_png, dpi=200)
    plt.close()
    print(f"[ok] wrote: {scatter_png}")

    # 3) flip edges csv + heatmap counts
    flip_rows = []
    edge_counts_by_w = defaultdict(Counter)
    total_edge_counts = Counter()

    for w in w_resolved:
        if w == base_w:
            continue
        for s in starts:
            rb = row_by_ws[(base_w, s)]
            rn = row_by_ws[(w, s)]
            bnodes = get_path_nodes(rb)
            nnodes = get_path_nodes(rn)
            div = divergence_edge(bnodes, nnodes)
            if div is None:
                continue
            div_i, from_node, base_next, nov_next = div
            if base_next == nov_next:
                continue
            lab = edge_label(from_node, base_next, nov_next)
            edge_counts_by_w[w][lab] += 1
            total_edge_counts[lab] += 1
            flip_rows.append({
                "w_novelty": f"{w:.3f}",
                "start": s,
                "div_index": str(div_i),
                "edge": lab,
                "base_cycle": get_cycle_key(rb),
                "nov_cycle": get_cycle_key(rn),
                "base_path": "->".join(bnodes),
                "nov_path": "->".join(nnodes),
            })

    flip_csv = os.path.join(args.outdir, "phase28g_flip_edges.csv")
    with open(flip_csv, "w", newline="") as f:
        cols = ["w_novelty","start","div_index","edge","base_cycle","nov_cycle","base_path","nov_path"]
        dw = csv.DictWriter(f, fieldnames=cols)
        dw.writeheader()
        for r in flip_rows:
            dw.writerow(r)
    print(f"[ok] wrote: {flip_csv}")

    # 4) heatmap (NONZERO edges only, SAFE when empty)
    # Filter to edges that actually occurred at least once.
    nonzero_edges = [e for e,c in total_edge_counts.items() if c > 0]
    if not nonzero_edges:
        print("[warn] no flip edges to plot; skipping heatmap")
        return

    top_edges = [e for e,_ in total_edge_counts.most_common(args.top_edges)]
    w_nonbase = [w for w in w_resolved if w != base_w]

    if not top_edges or not w_nonbase:
        print("[warn] no flip edges to plot; skipping heatmap")
        return

    heat = np.zeros((len(top_edges), len(w_nonbase)), dtype=float)
    for j, w in enumerate(w_nonbase):
        ctr = edge_counts_by_w.get(w, Counter())
        for i, e in enumerate(top_edges):
            heat[i, j] = ctr.get(e, 0)

    if heat.size == 0 or np.max(heat) == 0:
        print("[warn] heatmap has no nonzero counts; skipping heatmap")
        return

    heat_png = os.path.join(args.outdir, "phase28g_edge_heatmap.png")
    plt.figure(figsize=(12, max(2, 0.35 * heat.shape[0])))
    plt.imshow(heat, aspect="auto")
    plt.colorbar(label="count")
    plt.yticks(range(len(top_edges)), top_edges, fontsize=8)
    plt.xticks(range(len(w_nonbase)), [f"{w:.2f}" for w in w_nonbase], rotation=45, ha="right")
    plt.title(args.title + " — flip edges heatmap")
    plt.tight_layout()
    plt.savefig(heat_png, dpi=200)
    plt.close()
    print(f"[ok] wrote: {heat_png}")


if __name__ == "__main__":
    main()
