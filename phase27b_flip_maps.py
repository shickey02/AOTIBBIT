#!/usr/bin/env python3
# phase27b_flip_maps.py
#
# Visual + report tooling for Phase26g novelty bifurcations.
# Reads phase26g_rollouts_wN=*.csv, compares baseline vs novelty rollouts,
# finds first divergence per start, and exports flip summaries + DOT graph.
#
# Assumes rollouts CSV columns include:
#   start, status, steps, cycle_len, cycle_key, path
# where path is like: a->b->c->...
#
# Usage:
#   python bbit_geomlang/phase27b_flip_maps.py ^
#     --indir outputs_edges_relternary256_phase15/phase26g_bifurcate_v2 ^
#     --outdir outputs_edges_relternary256_phase15/phase27b_flip_maps ^
#     --novelty 0.35

import os, csv, argparse, math
from collections import defaultdict, Counter

def parse_path(path_str):
    if not path_str:
        return []
    # robust split
    parts = [p.strip() for p in path_str.split("->")]
    return [p for p in parts if p]

def load_rollouts_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def parse_w_from_filename(fn):
    # expects phase26g_rollouts_wN=0.350.csv
    s = fn.split("wN=", 1)[1].rsplit(".csv", 1)[0]
    return float(s)

def find_rollout_file(indir, w_target, tol=1e-9):
    # choose exact match if possible; else nearest within small tol
    candidates = []
    for fn in os.listdir(indir):
        if fn.startswith("phase26g_rollouts_wN=") and fn.endswith(".csv"):
            w = parse_w_from_filename(fn)
            candidates.append((w, fn))
    if not candidates:
        return None

    # exact match
    for w, fn in candidates:
        if abs(w - w_target) <= tol:
            return os.path.join(indir, fn)

    # nearest
    candidates.sort(key=lambda t: abs(t[0] - w_target))
    w_best, fn_best = candidates[0]
    return os.path.join(indir, fn_best)

def index_rows_by_start(rows):
    # There may be multiple rows per start; we pick the first "cycle" row if present,
    # else the first row.
    by = defaultdict(list)
    for r in rows:
        by[r.get("start","")].append(r)

    pick = {}
    for s, lst in by.items():
        # prefer status=cycle, else any
        lst_cycle = [r for r in lst if (r.get("status","").lower() == "cycle")]
        chosen = (lst_cycle[0] if lst_cycle else lst[0])
        pick[s] = chosen
    return pick

def first_divergence(pathA, pathB):
    n = min(len(pathA), len(pathB))
    for i in range(n):
        if pathA[i] != pathB[i]:
            return i
    if len(pathA) != len(pathB):
        return n
    return None

def dot_escape(s):
    return s.replace('"', r'\"')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="Phase26g output dir containing rollouts CSVs")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--baseline", type=float, default=0.0, help="baseline novelty weight (default 0.0)")
    ap.add_argument("--novelty", type=float, default=0.35, help="novelty weight to compare against baseline")
    ap.add_argument("--emit_dot", action="store_true", help="also emit Graphviz .dot file")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    f_base = find_rollout_file(args.indir, args.baseline)
    f_nov  = find_rollout_file(args.indir, args.novelty)

    if not f_base or not os.path.exists(f_base):
        raise FileNotFoundError(f"Could not find baseline rollout file for wN={args.baseline}")
    if not f_nov or not os.path.exists(f_nov):
        raise FileNotFoundError(f"Could not find novelty rollout file for wN={args.novelty}")

    base_rows = load_rollouts_csv(f_base)
    nov_rows  = load_rollouts_csv(f_nov)

    base_by_start = index_rows_by_start(base_rows)
    nov_by_start  = index_rows_by_start(nov_rows)

    starts = sorted(set(base_by_start.keys()) | set(nov_by_start.keys()))
    if not starts:
        raise RuntimeError("No starts found in rollouts files.")

    divergences = []
    edge_flips = Counter()

    report_lines = []
    report_lines.append(f"[files] baseline={os.path.basename(f_base)} novelty={os.path.basename(f_nov)}")
    report_lines.append(f"[starts] {len(starts)} total")
    report_lines.append("")

    for s in starts:
        rb = base_by_start.get(s)
        rn = nov_by_start.get(s)
        if rb is None or rn is None:
            continue

        pb = parse_path(rb.get("path",""))
        pn = parse_path(rn.get("path",""))

        div_i = first_divergence(pb, pn)
        if div_i is None:
            # identical
            continue

        # divergence at node index div_i; edge decision made at step div_i-1 (from pb[div_i-1] -> pb[div_i])
        # if div_i == 0, it means starting nodes differ (should not happen)
        from_node = pb[div_i-1] if div_i > 0 and div_i-1 < len(pb) else "(none)"
        base_next = pb[div_i] if div_i < len(pb) else "(end)"
        nov_next  = pn[div_i] if div_i < len(pn) else "(end)"

        base_edge = f"{from_node} -> {base_next}"
        nov_edge  = f"{from_node} -> {nov_next}"

        edge_flips[(from_node, base_next, nov_next)] += 1

        divergences.append({
            "start": s,
            "div_index": div_i,
            "from_node": from_node,
            "base_next": base_next,
            "nov_next": nov_next,
            "base_cycle": rb.get("cycle_key",""),
            "nov_cycle": rn.get("cycle_key",""),
            "base_path": "->".join(pb),
            "nov_path":  "->".join(pn),
        })

    # Write divergences CSV
    out_div = os.path.join(args.outdir, "phase27b_divergences.csv")
    with open(out_div, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "start","div_index","from_node","base_next","nov_next",
            "base_cycle","nov_cycle","base_path","nov_path"
        ])
        w.writeheader()
        for r in divergences:
            w.writerow(r)

    # Edge flip summary
    out_edge = os.path.join(args.outdir, "phase27b_edge_flips.csv")
    with open(out_edge, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["from_node","base_next","nov_next","count"])
        w.writeheader()
        for (frm, bnx, nnx), cnt in edge_flips.most_common():
            w.writerow({"from_node": frm, "base_next": bnx, "nov_next": nnx, "count": cnt})

    # Text report
    report_lines.append(f"[divergences] {len(divergences)} starts diverged (baseline vs novelty)")
    report_lines.append("")
    if divergences:
        report_lines.append("Top flip edges (from_node: base_next -> nov_next):")
        for (frm, bnx, nnx), cnt in edge_flips.most_common(25):
            report_lines.append(f"  count={cnt:3d}  {frm}: {bnx}  ->  {nnx}")
        report_lines.append("")
        report_lines.append("Per-start divergence details:")
        for r in divergences:
            report_lines.append(f"== start: {r['start']} ==")
            report_lines.append(f"div_index: {r['div_index']}  (decision from {r['from_node']})")
            report_lines.append(f"base_next: {r['base_next']}")
            report_lines.append(f"nov_next : {r['nov_next']}")
            report_lines.append(f"base_cycle: {r['base_cycle']}")
            report_lines.append(f"nov_cycle : {r['nov_cycle']}")
            report_lines.append(f"base_path : {r['base_path']}")
            report_lines.append(f"nov_path  : {r['nov_path']}")
            report_lines.append("")
    else:
        report_lines.append("No divergences detected (paths identical).")

    out_txt = os.path.join(args.outdir, "phase27b_flip_report.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # Optional DOT graph for the flip edges
    # Nodes = all nodes appearing in any divergence paths. Edges labeled base vs novelty.
    if args.emit_dot:
        nodes = set()
        for r in divergences:
            for n in r["base_path"].split("->"):
                if n: nodes.add(n)
            for n in r["nov_path"].split("->"):
                if n: nodes.add(n)

        dot_lines = []
        dot_lines.append("digraph flipnet {")
        dot_lines.append('  rankdir=LR;')
        dot_lines.append('  node [shape=ellipse, fontsize=10];')
        for n in sorted(nodes):
            dot_lines.append(f'  "{dot_escape(n)}";')

        # base edges in red, novelty edges in blue; flip edges emphasized
        for r in divergences:
            pb = r["base_path"].split("->")
            pn = r["nov_path"].split("->")
            # draw full base path
            for i in range(len(pb)-1):
                a, b = pb[i], pb[i+1]
                dot_lines.append(f'  "{dot_escape(a)}" -> "{dot_escape(b)}" [color="red", penwidth=1.2, label="base"];')
            # draw full novelty path
            for i in range(len(pn)-1):
                a, b = pn[i], pn[i+1]
                dot_lines.append(f'  "{dot_escape(a)}" -> "{dot_escape(b)}" [color="blue", penwidth=1.2, label="nov"];')

            # emphasize the flip edge
            frm = r["from_node"]
            dot_lines.append(f'  "{dot_escape(frm)}" -> "{dot_escape(r["base_next"])}" [color="red", penwidth=3.0];')
            dot_lines.append(f'  "{dot_escape(frm)}" -> "{dot_escape(r["nov_next"])}"  [color="blue", penwidth=3.0];')

        dot_lines.append("}")
        out_dot = os.path.join(args.outdir, "phase27b_flip_network.dot")
        with open(out_dot, "w", encoding="utf-8") as f:
            f.write("\n".join(dot_lines))

    print("[ok] wrote:")
    print(" ", out_div)
    print(" ", out_edge)
    print(" ", out_txt)
    if args.emit_dot:
        print(" ", os.path.join(args.outdir, "phase27b_flip_network.dot"))

if __name__ == "__main__":
    main()
