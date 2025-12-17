#!/usr/bin/env python3
# phase17a_holonomy_viz.py
#
# Phase 17a: Visualize holonomy loop results from Phase 16a.
#
# Supports BOTH input JSON formats:
#   (A) Old format:
#       { "plane": "...", "top": [<loopdict>...], "all_count": <int> }
#
#   (B) New threshold format (from your updated 16a):
#       {
#         "plane": "...",
#         "total_valid": <int>,
#         "threshold_results": {
#            "0.30": {"min_sim":0.3,"kept_count":20,"top":[...loops...]},
#            "0.35": {"min_sim":0.35,"kept_count":10,"top":[...loops...]}
#         }
#       }
#
# Usage:
#   python phase17a_holonomy_viz.py --in_json outputs.../phase16a_holonomy_bo_lr.json --outdir outputs.../phase17a_bo_lr
#
# Optional:
#   --choose_threshold 0.35     # pick that bucket (if available)
#   --pick max|min|first        # default=max (strictest threshold with non-empty top)
#   --num 25                    # limit number of loops to visualize (default: all in chosen bucket)
#
# Output:
#   - summary.json (like the one you showed)
#   - plots:
#       holonomy_hist.png
#       sim_vs_holonomy.png
#       angle_vs_holonomy.png
#       per-loop:
#           loop_###_R_loop.png
#           loop_###_eigvals.png
#   - report.txt

import os, json, argparse, math
import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless safe
import matplotlib.pyplot as plt

# ------------------------- helpers -------------------------

def _ensure_dir(d):
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _as_float(x, default=float("nan")):
    try:
        return float(x)
    except Exception:
        return default

def _string_loop(loop_field):
    # loop may be list of node names OR already a string
    if isinstance(loop_field, list):
        if not loop_field:
            return ""
        return " -> ".join(loop_field + [loop_field[0]])
    if isinstance(loop_field, str):
        return loop_field
    return str(loop_field)

def _extract_angle_deg(loopdict):
    # Phase16a stores: trace_angle: { angle_deg_proxy: ... }
    ta = loopdict.get("trace_angle", {})
    if isinstance(ta, dict) and "angle_deg_proxy" in ta:
        return _as_float(ta.get("angle_deg_proxy"))
    # Some downstream summaries might store angle directly
    if "angle_deg_proxy" in loopdict:
        return _as_float(loopdict["angle_deg_proxy"])
    return float("nan")

def _extract_R(loopdict):
    R = loopdict.get("R_loop", None)
    if R is None:
        return None
    try:
        return np.array(R, dtype=np.float64)
    except Exception:
        return None

# ------------------------- loader -------------------------

def load_phase16a(path, choose_threshold=None, pick="max"):
    """
    Returns: (plane, loops, all_count, meta)
      - plane: str
      - loops: list[dict]  (each should include R_loop, fro_norm_R_minus_I, edge_sim_mean, etc.)
      - all_count: int (total loops considered/kept depending on format)
      - meta: dict describing selection
    """
    with open(path, "r") as f:
        data = json.load(f)

    plane = data.get("plane", "unknown")

    # --- New format: threshold_results ---
    if isinstance(data, dict) and "threshold_results" in data and isinstance(data["threshold_results"], dict):
        thr = data["threshold_results"]

        # Build list of candidate buckets with numeric threshold keys
        buckets = []
        for k_str, bucket in thr.items():
            try:
                k_val = float(k_str)
            except Exception:
                # fallback to bucket["min_sim"] if key isn't numeric
                k_val = _as_float(bucket.get("min_sim", float("nan")))
            top_list = bucket.get("top", [])
            kept = int(bucket.get("kept_count", len(top_list)))
            buckets.append((k_val, k_str, bucket, kept, len(top_list)))

        # Choose bucket
        chosen = None
        if choose_threshold is not None:
            # pick closest matching threshold key (exact match preferred)
            target = float(choose_threshold)
            # exact match by numeric
            exact = [b for b in buckets if abs(b[0] - target) < 1e-12]
            if exact:
                chosen = sorted(exact, key=lambda x: x[0])[0]
            else:
                # choose closest
                chosen = min(buckets, key=lambda x: abs(x[0] - target)) if buckets else None
        else:
            # default: pick strictest non-empty bucket (max threshold with top entries)
            nonempty = [b for b in buckets if b[4] > 0]
            if not nonempty:
                nonempty = buckets
            if not nonempty:
                chosen = None
            else:
                if pick == "min":
                    chosen = sorted(nonempty, key=lambda x: x[0])[0]
                elif pick == "first":
                    chosen = sorted(nonempty, key=lambda x: str(x[1]))[0]
                else:
                    # max (strictest)
                    chosen = sorted(nonempty, key=lambda x: x[0], reverse=True)[0]

        if chosen is None:
            return plane, [], 0, {"format": "threshold_results", "error": "no buckets"}

        k_val, k_str, bucket, kept, top_len = chosen
        loops = bucket.get("top", [])
        total_valid = int(data.get("total_valid", data.get("all_count", kept)))

        meta = {
            "format": "threshold_results",
            "chosen_threshold": float(bucket.get("min_sim", k_val)),
            "chosen_threshold_key": k_str,
            "kept_count": kept,
            "total_valid": total_valid,
            "min_sim_thresholds": sorted([float(b[0]) for b in buckets if not math.isnan(b[0])])
        }

        # plane could be missing at top-level in some variants; infer from first loop if needed
        if plane == "unknown" and loops:
            plane = loops[0].get("plane", "unknown")

        return plane, loops, kept, meta

    # --- Old format ---
    loops = data.get("top", [])
    all_count = int(data.get("all_count", len(loops)))
    if not isinstance(loops, list):
        loops = []
    meta = {"format": "old_top_list"}
    if plane == "unknown" and loops:
        plane = loops[0].get("plane", "unknown")
    return plane, loops, all_count, meta

# ------------------------- plotting -------------------------

def plot_hist(values, outpath, title, xlabel):
    vals = [v for v in values if np.isfinite(v)]
    plt.figure()
    if len(vals) == 0:
        plt.title(title + " (no finite values)")
        plt.savefig(outpath, dpi=160, bbox_inches="tight")
        plt.close()
        return
    plt.hist(vals, bins=min(30, max(5, int(np.sqrt(len(vals))))))  # simple rule of thumb
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close()

def plot_scatter(x, y, outpath, title, xlabel, ylabel):
    xs = []
    ys = []
    for a, b in zip(x, y):
        if np.isfinite(a) and np.isfinite(b):
            xs.append(a); ys.append(b)
    plt.figure()
    if len(xs) == 0:
        plt.title(title + " (no finite points)")
        plt.savefig(outpath, dpi=160, bbox_inches="tight")
        plt.close()
        return
    plt.scatter(xs, ys, s=18)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close()

def plot_R_heatmap(R, outpath, title):
    plt.figure()
    plt.imshow(R, aspect="equal")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close()

def plot_eigs(eigs, outpath, title):
    plt.figure()
    re = [np.real(z) for z in eigs]
    im = [np.imag(z) for z in eigs]
    # unit circle for reference
    t = np.linspace(0, 2*np.pi, 400)
    plt.plot(np.cos(t), np.sin(t), linewidth=1)
    plt.scatter(re, im, s=25)
    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(title)
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close()

# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True, help="Phase16a output JSON (old or threshold format)")
    ap.add_argument("--outdir", required=True, help="Output directory for plots + summary")
    ap.add_argument("--choose_threshold", type=float, default=None, help="If input has threshold_results, pick this threshold (closest match if not exact).")
    ap.add_argument("--pick", choices=["max", "min", "first"], default="max", help="If choose_threshold is not provided, how to pick a threshold bucket.")
    ap.add_argument("--num", type=int, default=0, help="Limit number of loops visualized (0 = all in chosen bucket).")
    args = ap.parse_args()

    _ensure_dir(args.outdir)

    plane, loops, all_count, meta = load_phase16a(
        args.in_json,
        choose_threshold=args.choose_threshold,
        pick=args.pick
    )

    if not loops:
        raise SystemExit(f"[error] No loops found in {args.in_json}. meta={meta}")

    # limit
    if args.num and args.num > 0:
        loops_viz = loops[:args.num]
    else:
        loops_viz = loops

    # Extract arrays
    hol = [ _as_float(ld.get("fro_norm_R_minus_I", float("nan"))) for ld in loops_viz ]
    sim_mean = [ _as_float(ld.get("edge_sim_mean", float("nan"))) for ld in loops_viz ]
    sim_min  = [ _as_float(ld.get("edge_sim_min", float("nan"))) for ld in loops_viz ]
    angle = [ _extract_angle_deg(ld) for ld in loops_viz ]
    score = [ _as_float(ld.get("rank_score", float("nan"))) for ld in loops_viz ]
    k_list = [ int(ld.get("k", -1)) if str(ld.get("k","")).isdigit() else int(ld.get("k", -1)) for ld in loops_viz ]

    # Global plots
    plot_hist(hol, os.path.join(args.outdir, "holonomy_hist.png"),
              title=f"Holonomy magnitude ||R-I||_F ({plane})",
              xlabel="||R - I||_F")

    plot_scatter(sim_mean, hol, os.path.join(args.outdir, "sim_vs_holonomy.png"),
                 title=f"Similarity(mean) vs Holonomy ({plane})",
                 xlabel="edge_sim_mean", ylabel="||R-I||_F")

    plot_scatter(angle, hol, os.path.join(args.outdir, "angle_vs_holonomy.png"),
                 title=f"Angle proxy vs Holonomy ({plane})",
                 xlabel="angle_deg_proxy", ylabel="||R-I||_F")

    # Per-loop visuals + report rows
    rows = []
    report_lines = []
    report_lines.append(f"Phase17a Holonomy Viz")
    report_lines.append(f"input: {args.in_json}")
    report_lines.append(f"outdir: {args.outdir}")
    report_lines.append(f"plane: {plane}")
    report_lines.append(f"meta: {json.dumps(meta)}")
    report_lines.append(f"loops_visualized: {len(loops_viz)}")
    report_lines.append("")

    for idx, ld in enumerate(loops_viz, 1):
        loop_str = _string_loop(ld.get("loop", ""))
        R = _extract_R(ld)
        ang = _extract_angle_deg(ld)
        row = {
            "idx": idx,
            "plane": plane,
            "k": int(ld.get("k", -1)) if ld.get("k", None) is not None else None,
            "rank_score": _as_float(ld.get("rank_score", float("nan"))),
            "fro_norm_R_minus_I": _as_float(ld.get("fro_norm_R_minus_I", float("nan"))),
            "det_R_loop": _as_float(ld.get("det_R_loop", float("nan"))),
            "angle_deg_proxy": ang,
            "edge_sim_min": _as_float(ld.get("edge_sim_min", float("nan"))),
            "edge_sim_mean": _as_float(ld.get("edge_sim_mean", float("nan"))),
            "edge_sim_max": _as_float(ld.get("edge_sim_max", float("nan"))),
            "loop": loop_str,
        }
        rows.append(row)

        report_lines.append(
            f"{idx:03d} score={row['rank_score']:.6f}  ||R-I||_F={row['fro_norm_R_minus_I']:.6f}  "
            f"det={row['det_R_loop']:.6f}  ang={row['angle_deg_proxy']:.3f}  "
            f"sim_min={row['edge_sim_min']:.4f}  sim_mean={row['edge_sim_mean']:.4f}  loop: {loop_str}"
        )

        # Save R heatmap + eig plot if available
        if R is not None and R.ndim == 2 and R.shape[0] == R.shape[1]:
            outR = os.path.join(args.outdir, f"loop_{idx:03d}_R_loop.png")
            plot_R_heatmap(R, outR, title=f"R_loop (idx={idx})")

            try:
                eigs = np.linalg.eigvals(R)
                outE = os.path.join(args.outdir, f"loop_{idx:03d}_eigvals.png")
                plot_eigs(eigs, outE, title=f"eigvals(R_loop) (idx={idx})")
            except Exception:
                pass

    # Write report
    with open(os.path.join(args.outdir, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    # Write summary.json (matches your shown structure)
    summary = {
        "plane": plane,
        "format": meta.get("format", "unknown"),
        "chosen_threshold": meta.get("chosen_threshold", None),
        "all_count": int(all_count),
        "num_visualized": int(len(loops_viz)),
        "source_json": os.path.abspath(args.in_json),
        "meta": meta,
        "rows": rows
    }
    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[ok] plane={plane} loops_visualized={len(loops_viz)}")
    print(f"[saved] {os.path.join(args.outdir, 'summary.json')}")
    print(f"[saved] {os.path.join(args.outdir, 'report.txt')}")
    print(f"[saved] plots in {args.outdir}")

if __name__ == "__main__":
    main()
