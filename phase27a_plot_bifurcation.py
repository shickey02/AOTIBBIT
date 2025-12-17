#!/usr/bin/env python3
# phase27a_plot_bifurcation.py
#
# Reads phase26g outputs and makes quick visuals:
#  - bifurcation diagram: novelty weight vs attractor counts
#  - attractor frequency bars per weight
#  - prints/exports emergence report (new cycles appearing as wN increases)
#
# Usage:
#   python bbit_geomlang/phase27a_plot_bifurcation.py ^
#     --indir outputs_edges_relternary256_phase15/phase26g_bifurcate_v2 ^
#     --outdir outputs_edges_relternary256_phase15/phase27a_bifurcation_viz

import os, csv, argparse
from collections import Counter, defaultdict

import matplotlib.pyplot as plt

def read_sweep(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "w_novelty": float(r["w_novelty"]),
                "agree": float(r["agree_frac_vs_base"]),
                "n_attr": int(r["n_attractors"]),
                "sig": r.get("signature","")
            })
    rows.sort(key=lambda x: x["w_novelty"])
    return rows

def read_rollouts(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def parse_w_from_filename(fn):
    # expects phase26g_rollouts_wN=0.350.csv
    s = fn.split("wN=", 1)[1].rsplit(".csv", 1)[0]
    return float(s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    sweep_csv = os.path.join(args.indir, "phase26g_bifurcation_sweep.csv")
    if not os.path.exists(sweep_csv):
        raise FileNotFoundError(f"Missing: {sweep_csv}")

    sweep = read_sweep(sweep_csv)

    # Collect rollout files
    rollout_files = []
    for fn in os.listdir(args.indir):
        if fn.startswith("phase26g_rollouts_wN=") and fn.endswith(".csv"):
            rollout_files.append(fn)
    if not rollout_files:
        raise RuntimeError("No rollout CSVs found in indir")

    rollout_files.sort(key=lambda fn: parse_w_from_filename(fn))

    # Map w -> attractor counts (cycle_key)
    w_to_counts = {}
    w_to_total = {}
    for fn in rollout_files:
        wN = parse_w_from_filename(fn)
        rows = read_rollouts(os.path.join(args.indir, fn))
        c = Counter()
        for r in rows:
            ck = r.get("cycle_key", "NONE") or "NONE"
            c[ck] += 1
        w_to_counts[wN] = c
        w_to_total[wN] = sum(c.values())

    # 1) Bifurcation diagram: n_attractors vs wN + agree curve
    xs = [r["w_novelty"] for r in sweep]
    nattrs = [r["n_attr"] for r in sweep]
    agrees = [r["agree"] for r in sweep]

    plt.figure()
    plt.plot(xs, nattrs, marker="o")
    plt.xlabel("w_novelty")
    plt.ylabel("n_attractors")
    plt.title("Bifurcation: number of attractors vs novelty weight")
    out1 = os.path.join(args.outdir, "phase27a_bifurcation_n_attractors.png")
    plt.savefig(out1, dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(xs, agrees, marker="o")
    plt.xlabel("w_novelty")
    plt.ylabel("agree_frac_vs_baseline")
    plt.title("Stability: agreement vs baseline")
    out2 = os.path.join(args.outdir, "phase27a_agreement_vs_baseline.png")
    plt.savefig(out2, dpi=180, bbox_inches="tight")
    plt.close()

    # 2) Attractor frequency bars per weight (top K)
    TOPK = 8
    # determine global top cycles across all weights (excluding NONE)
    global_counts = Counter()
    for wN, c in w_to_counts.items():
        for k, v in c.items():
            if k != "NONE":
                global_counts[k] += v
    top_cycles = [k for k,_ in global_counts.most_common(TOPK)]

    # stacked bar: frequency of each top cycle across wN
    plt.figure()
    bottoms = [0]*len(rollout_files)
    w_list = [parse_w_from_filename(fn) for fn in rollout_files]

    for cyc in top_cycles:
        vals = []
        for wN in w_list:
            vals.append(w_to_counts[wN].get(cyc, 0))
        plt.bar(w_list, vals, bottom=bottoms, label=cyc)
        bottoms = [b+v for b,v in zip(bottoms, vals)]

    plt.xlabel("w_novelty")
    plt.ylabel("count (out of starts)")
    plt.title("Attractor frequencies vs novelty weight (stacked)")
    plt.legend(fontsize=7, loc="best")
    out3 = os.path.join(args.outdir, "phase27a_attractor_frequencies_stacked.png")
    plt.savefig(out3, dpi=180, bbox_inches="tight")
    plt.close()

    # 3) Emergence report: what new cycles appear as w increases
    seen = set()
    report_lines = []
    for wN in w_list:
        c = w_to_counts[wN]
        present = set(k for k,v in c.items() if v > 0 and k != "NONE")
        new = sorted(list(present - seen))
        report_lines.append(f"wN={wN:.3f} present={len(present)} new={len(new)}")
        for k in new:
            report_lines.append(f"  + {k}  (count={c[k]})")
        seen |= present

    report_path = os.path.join(args.outdir, "phase27a_emergence_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # Print quick summary
    print("[ok] wrote:")
    print(" ", out1)
    print(" ", out2)
    print(" ", out3)
    print(" ", report_path)

if __name__ == "__main__":
    main()
