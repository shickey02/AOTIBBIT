#!/usr/bin/env python3
"""
Phase 26DS — strict-line rescue CUDA/staged search for Basic32.

What changed from 26DR-FAST
---------------------------
26DR-FAST made the strict-envelope audit runnable, but its scalar DR score still
let high-score cases survive with bad full-envelope tangent tails. 26DS is a
rescue pass: it ranks by strict-line viability first, then score. It searches
near the best DP/DQ/DR/DR-FAST candidates, then applies directed micro-sweeps
that try to push the Pareto knee downward in tangent/radial drift without losing
capture.

Outputs are written to E:/BBIT/outputs_basic32:
  phase26ds_screen_results.csv
  phase26ds_screen_envelope_results.csv
  phase26ds_case_results.csv
  phase26ds_envelope_results.csv
  phase26ds_pareto_front.csv
  phase26ds_summary.json
  phase26ds_screen_top_scores.png
  phase26ds_top_scores.png
  phase26ds_capture_vs_tangent_pareto.png

Recommended first run:
  python bbit_geomlang/geomlang_phase26ds_strict_line_rescue_cuda_basic32_E_drive.py --mode turbo --device cuda

Wider run if turbo is promising:
  python bbit_geomlang/geomlang_phase26ds_strict_line_rescue_cuda_basic32_E_drive.py --mode balanced --device cuda
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PHASE = "26DS"
TITLE = "strict-line rescue CUDA/staged low-tangent search"
ROOT = Path(r"E:/BBIT/outputs_basic32")
PARAMS = [
    "BOWL_RADIUS_FRAC",
    "BOWL_SEAT_AXIS_GAIN",
    "BOWL_DIRECTIONAL_BLEND",
    "BOWL_NORM_CAP_MULT",
    "BOWL_SHELL_RADIAL_GAIN",
    "BOWL_TANGENT_KILL",
]
BOUNDS = {
    "BOWL_RADIUS_FRAC": (0.56, 0.82),
    "BOWL_SEAT_AXIS_GAIN": (0.26, 0.56),
    "BOWL_DIRECTIONAL_BLEND": (0.92, 1.18),
    "BOWL_NORM_CAP_MULT": (0.96, 1.20),
    "BOWL_SHELL_RADIAL_GAIN": (0.08, 0.18),
    "BOWL_TANGENT_KILL": (0.68, 1.02),
}
DEFAULT_TARGET_WORST_TAN = 2.10
DEFAULT_TARGET_WORST_CAP = 0.285
DEFAULT_TARGET_Q20_DP = 2.50


def import_dr_fast() -> Any:
    try:
        return importlib.import_module("geomlang_phase26dr_fast_staged_cuda_basic32_E_drive")
    except ModuleNotFoundError:
        import sys
        here = Path(__file__).resolve().parent
        if str(here) not in sys.path:
            sys.path.insert(0, str(here))
        return importlib.import_module("geomlang_phase26dr_fast_staged_cuda_basic32_E_drive")


def fnum(x: Any, default: float = float("nan")) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def clamp_param(k: str, v: float) -> float:
    lo, hi = BOUNDS[k]
    return float(min(hi, max(lo, v)))


def clamp_ov(ov: Dict[str, float]) -> Dict[str, float]:
    return {k: clamp_param(k, fnum(ov.get(k), 0.0)) for k in PARAMS}


def row_key(ov: Dict[str, float]) -> Tuple[float, ...]:
    return tuple(round(float(ov[k]), 5) for k in PARAMS)


def add_unique(out: List[Tuple[str, Dict[str, float]]], seen: set, name: str, ov: Dict[str, float]) -> None:
    cov = clamp_ov(ov)
    key = row_key(cov)
    if key not in seen:
        out.append((name, cov))
        seen.add(key)


def parse_overrides_from_row(row: pd.Series) -> Dict[str, float] | None:
    if all(k in row and pd.notna(row[k]) for k in PARAMS):
        return clamp_ov({k: float(row[k]) for k in PARAMS})
    for col in ("overrides", "overrides_json"):
        if col in row and isinstance(row[col], str) and row[col].strip():
            try:
                obj = json.loads(row[col])
                if all(k in obj for k in PARAMS):
                    return clamp_ov({k: float(obj[k]) for k in PARAMS})
            except Exception:
                pass
    return None


def read_prior_csv_candidates() -> List[Tuple[str, Dict[str, float]]]:
    """Collect the best prior points from DP/DQ/DR-fast CSV outputs if present."""
    files = [
        "phase26dr_fast_case_results.csv",
        "phase26dr_fast_screen_results.csv",
        "phase26dq_case_results.csv",
        "phase26dp_case_results.csv",
        "phase26do_case_results.csv",
        "phase26dn_base_results.csv",
        "phase26dm_case_results.csv",
    ]
    out: List[Tuple[str, Dict[str, float]]] = []
    seen: set = set()
    for fname in files:
        p = ROOT / fname
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        # Keep several views: high score, low tangent, high capture, and strong q20/worst DP when available.
        views = []
        if "dr_score" in df.columns:
            views.append(df.sort_values("dr_score", ascending=False).head(18))
        if "dq_score" in df.columns:
            views.append(df.sort_values("dq_score", ascending=False).head(18))
        if "dp_score" in df.columns:
            views.append(df.sort_values("dp_score", ascending=False).head(18))
        if "worst_tangent_ratio" in df.columns:
            views.append(df.sort_values("worst_tangent_ratio", ascending=True).head(18))
        if "tangent_ratio" in df.columns:
            views.append(df.sort_values("tangent_ratio", ascending=True).head(18))
        if "worst_capture_rate" in df.columns:
            views.append(df.sort_values("worst_capture_rate", ascending=False).head(18))
        if "capture_rate" in df.columns:
            views.append(df.sort_values("capture_rate", ascending=False).head(18))
        if not views:
            views = [df.head(24)]
        for view_i, vdf in enumerate(views):
            for j, row in vdf.iterrows():
                ov = parse_overrides_from_row(row)
                if ov is None:
                    continue
                case = str(row.get("case", f"row{j}"))[:72]
                add_unique(out, seen, f"prior_csv_{Path(fname).stem}_{view_i:02d}_{case}", ov)
    return out


def micro_variants(base_name: str, base: Dict[str, float], rng: random.Random, n_random: int) -> List[Tuple[str, Dict[str, float]]]:
    """Directed variants: lower tangent tail first, then recover capture."""
    out: List[Tuple[str, Dict[str, float]]] = []
    seen: set = set()
    add_unique(out, seen, f"{base_name}_base", base)

    # Single-axis rescue sweeps around the base point.
    for r in [0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74]:
        ov = dict(base); ov["BOWL_RADIUS_FRAC"] = r
        add_unique(out, seen, f"{base_name}_r{r:.2f}", ov)
    for s in [0.30, 0.34, 0.38, 0.42, 0.46, 0.50]:
        ov = dict(base); ov["BOWL_SEAT_AXIS_GAIN"] = s
        add_unique(out, seen, f"{base_name}_seat{s:.2f}", ov)
    for b in [0.98, 1.00, 1.04, 1.08, 1.12, 1.16]:
        ov = dict(base); ov["BOWL_DIRECTIONAL_BLEND"] = b
        add_unique(out, seen, f"{base_name}_blend{b:.2f}", ov)
    for c in [1.00, 1.04, 1.08, 1.12, 1.16]:
        ov = dict(base); ov["BOWL_NORM_CAP_MULT"] = c
        add_unique(out, seen, f"{base_name}_cap{c:.2f}", ov)
    for sh in [0.10, 0.12, 0.14, 0.16]:
        ov = dict(base); ov["BOWL_SHELL_RADIAL_GAIN"] = sh
        add_unique(out, seen, f"{base_name}_shell{sh:.2f}", ov)
    for tk in [0.72, 0.76, 0.80, 0.84, 0.88, 0.92, 0.96, 1.00]:
        ov = dict(base); ov["BOWL_TANGENT_KILL"] = tk
        add_unique(out, seen, f"{base_name}_tk{tk:.2f}", ov)

    # Coupled moves: radius down + seat/cap adjusted + tangent kill sweep.
    for r in [0.62, 0.64, 0.66, 0.68, 0.70]:
        for s in [0.34, 0.38, 0.42, 0.46]:
            for tk in [0.76, 0.84, 0.92]:
                ov = dict(base)
                ov["BOWL_RADIUS_FRAC"] = r
                ov["BOWL_SEAT_AXIS_GAIN"] = s
                ov["BOWL_TANGENT_KILL"] = tk
                ov["BOWL_NORM_CAP_MULT"] = clamp_param("BOWL_NORM_CAP_MULT", base["BOWL_NORM_CAP_MULT"] + (0.02 if r < 0.66 else 0.0))
                add_unique(out, seen, f"{base_name}_coupled_r{r:.2f}_s{s:.2f}_tk{tk:.2f}", ov)

    # Random local cloud, biased toward lower tangent tail rather than raw score.
    sig = {
        "BOWL_RADIUS_FRAC": 0.028,
        "BOWL_SEAT_AXIS_GAIN": 0.045,
        "BOWL_DIRECTIONAL_BLEND": 0.030,
        "BOWL_NORM_CAP_MULT": 0.025,
        "BOWL_SHELL_RADIAL_GAIN": 0.018,
        "BOWL_TANGENT_KILL": 0.050,
    }
    for i in range(n_random):
        ov = {}
        for k in PARAMS:
            ov[k] = base[k] + rng.gauss(0.0, sig[k])
        # every third sample explicitly nudges toward the observed knee band
        if i % 3 == 0:
            ov["BOWL_RADIUS_FRAC"] = 0.64 + rng.gauss(0.0, 0.035)
            ov["BOWL_TANGENT_KILL"] = 0.84 + rng.gauss(0.0, 0.065)
        add_unique(out, seen, f"{base_name}_cloud_{i:03d}", ov)
    return out


def global_knee_grid() -> List[Tuple[str, Dict[str, float]]]:
    out: List[Tuple[str, Dict[str, float]]] = []
    seen: set = set()
    for r in [0.62, 0.64, 0.66, 0.68, 0.70]:
        for s in [0.34, 0.38, 0.42, 0.46]:
            for b in [1.00, 1.04, 1.08, 1.12]:
                for c in [1.04, 1.08, 1.12]:
                    for sh in [0.10, 0.12, 0.14, 0.16]:
                        for tk in [0.76, 0.84, 0.92]:
                            ov = {
                                "BOWL_RADIUS_FRAC": r,
                                "BOWL_SEAT_AXIS_GAIN": s,
                                "BOWL_DIRECTIONAL_BLEND": b,
                                "BOWL_NORM_CAP_MULT": c,
                                "BOWL_SHELL_RADIAL_GAIN": sh,
                                "BOWL_TANGENT_KILL": tk,
                            }
                            add_unique(out, seen, f"ds_grid_r{r:.2f}_s{s:.2f}_b{b:.2f}_c{c:.2f}_sh{sh:.2f}_tk{tk:.2f}", ov)
    return out


def build_candidates(drfast: Any, mode: str, max_candidates: int, seed: int) -> List[Tuple[str, Dict[str, float]]]:
    rng = random.Random(seed)
    priors = read_prior_csv_candidates()
    # Add inherited DR priors/hardening pool, but only a diverse head because DS is focused.
    try:
        dr = drfast.import_phase26dr()
        inherited = dr.hardening_candidates(dr.read_prior_candidates())
        for name, ov in inherited[:80]:
            priors.append((f"inherited_{name}", ov))
    except Exception:
        pass

    # Pick seed bases from prior pool. If no files exist, fall back to current constants.
    seen_bases = set()
    bases: List[Tuple[str, Dict[str, float]]] = []
    for name, ov in priors:
        cov = clamp_ov(ov)
        key = row_key(cov)
        if key not in seen_bases:
            bases.append((name[:80], cov)); seen_bases.add(key)
    if not bases:
        bases = [("ds_current_constants", {
            "BOWL_RADIUS_FRAC": 0.66,
            "BOWL_SEAT_AXIS_GAIN": 0.35,
            "BOWL_DIRECTIONAL_BLEND": 1.00,
            "BOWL_NORM_CAP_MULT": 1.00,
            "BOWL_SHELL_RADIAL_GAIN": 0.12,
            "BOWL_TANGENT_KILL": 0.90,
        })]

    n_seed_bases = {"turbo": 6, "balanced": 12, "careful": 20}[mode]
    n_random = {"turbo": 14, "balanced": 26, "careful": 42}[mode]
    out: List[Tuple[str, Dict[str, float]]] = []
    seen: set = set()
    for bi, (name, base) in enumerate(bases[:n_seed_bases]):
        for vname, ov in micro_variants(f"ds_seed{bi:02d}_{name}", base, rng, n_random=n_random):
            add_unique(out, seen, vname, ov)

    # Add a diverse slice of the global knee grid; not the whole Cartesian product in turbo.
    grid = global_knee_grid()
    rng.shuffle(grid)
    grid_keep = {"turbo": 130, "balanced": 360, "careful": 720}[mode]
    for name, ov in grid[:grid_keep]:
        add_unique(out, seen, name, ov)

    if len(out) > max_candidates:
        # Keep the front loaded prior variants plus a spread through the rest.
        head_n = min(max_candidates // 2, len(out))
        head = out[:head_n]
        rest = out[head_n:]
        idxs = np.linspace(0, max(0, len(rest) - 1), max_candidates - head_n).round().astype(int).tolist() if rest else []
        out = head + [rest[i] for i in idxs]
    return out[:max_candidates]


def rescue_score(row: Dict[str, Any], target_cap: float, target_tan: float, target_q20: float) -> float:
    cap = fnum(row.get("worst_capture_rate"), 0.0)
    tan = fnum(row.get("worst_tangent_ratio"), 999.0)
    q20 = fnum(row.get("q20_dp_score"), -999.0)
    worst_dp = fnum(row.get("worst_dp_score"), -999.0)
    dr_score = fnum(row.get("dr_score"), 0.0)
    strict_surv = fnum(row.get("strict_survival_rate"), 0.0)
    # Hard objective: first get below tangent line, then raise capture, then preserve DP tail.
    tan_bonus = max(0.0, target_tan - tan) * 2.2
    tan_penalty = max(0.0, tan - target_tan) * 1.25
    cap_bonus = (cap - target_cap) * 10.0
    q20_bonus = 0.35 * (q20 - target_q20)
    dp_bonus = 0.25 * (worst_dp - 2.0)
    return float(dr_score + tan_bonus - tan_penalty + cap_bonus + q20_bonus + dp_bonus + 0.8 * strict_surv)


def promote_stage2(screen_df: pd.DataFrame, stage2_top: int, target_cap: float, target_tan: float) -> List[str]:
    if screen_df.empty:
        return []
    views = []
    views.append(screen_df.sort_values("ds_rescue_score", ascending=False).head(stage2_top))
    views.append(screen_df.sort_values(["worst_tangent_ratio", "worst_capture_rate"], ascending=[True, False]).head(max(6, stage2_top // 3)))
    views.append(screen_df.sort_values(["worst_capture_rate", "worst_tangent_ratio"], ascending=[False, True]).head(max(6, stage2_top // 3)))
    views.append(screen_df[(screen_df["worst_tangent_ratio"] <= target_tan * 1.45)].sort_values("ds_rescue_score", ascending=False).head(stage2_top))
    views.append(screen_df[(screen_df["worst_capture_rate"] >= target_cap * 0.88)].sort_values("ds_rescue_score", ascending=False).head(stage2_top))
    names = pd.concat(views, ignore_index=True).drop_duplicates("case")["case"].tolist()
    return names[: max(stage2_top, min(len(names), stage2_top * 2))]


def plot_bar(df: pd.DataFrame, out_path: Path, score_col: str, title: str) -> None:
    if df.empty or score_col not in df.columns:
        return
    top = df.sort_values(score_col, ascending=False).head(30).iloc[::-1]
    plt.figure(figsize=(15, max(8, 0.48 * len(top))))
    plt.barh(top["case"], top[score_col])
    plt.xlabel(score_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_scatter(df: pd.DataFrame, pf: pd.DataFrame, out_path: Path, target_cap: float, target_tan: float) -> None:
    if df.empty:
        return
    plt.figure(figsize=(10, 7))
    sc = plt.scatter(df["worst_capture_rate"], df["worst_tangent_ratio"], c=df["ds_rescue_score"], s=74, alpha=0.78)
    if not pf.empty:
        plt.scatter(pf["worst_capture_rate"], pf["worst_tangent_ratio"], facecolors="none", edgecolors="black", s=170, linewidths=1.7, label="Pareto front")
        plt.legend()
    plt.axvline(target_cap, linewidth=1.0, alpha=0.45)
    plt.axhline(target_tan, linewidth=1.0, alpha=0.45)
    plt.colorbar(sc, label="DS rescue score")
    plt.xlabel("worst envelope capture_rate")
    plt.ylabel("worst envelope tangent/radial ratio")
    plt.title("26DS strict-line rescue: capture vs tangent drift")
    plt.grid(True, alpha=0.28)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run(args: argparse.Namespace) -> Dict[str, Any]:
    t0 = time.time()
    ROOT.mkdir(parents=True, exist_ok=True)
    drfast = import_dr_fast()
    torch_info = drfast.setup_torch(args.device)
    dr = drfast.import_phase26dr()

    target_tan = float(getattr(dr, "TARGET_WORST_TAN", DEFAULT_TARGET_WORST_TAN))
    target_cap = float(getattr(dr, "TARGET_WORST_CAP", DEFAULT_TARGET_WORST_CAP))
    target_q20 = float(getattr(dr, "TARGET_Q20_DP", DEFAULT_TARGET_Q20_DP))

    cfg = {
        "turbo": {"max_candidates": 320, "stage2_top": 22, "print_every": 12},
        "balanced": {"max_candidates": 720, "stage2_top": 42, "print_every": 20},
        "careful": {"max_candidates": 1400, "stage2_top": 70, "print_every": 30},
    }[args.mode]
    max_candidates = args.max_candidates or cfg["max_candidates"]
    stage2_top = args.stage2_top or cfg["stage2_top"]
    print_every = args.print_every or cfg["print_every"]

    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] mode={args.mode} device={args.device} torch={torch_info}")
    print(f"[{PHASE}] targets: worstTan<={target_tan:.2f}, worstCap>={target_cap:.3f}, q20DP>={target_q20:.2f}")

    eval_helper = getattr(dr, "EVAL_HELPER", None)
    if eval_helper is None or not hasattr(eval_helper, "build_basins"):
        eval_helper = dr.previous_eval_helper()
    z2, rel_ids, basins = eval_helper.build_basins()
    print(f"[{PHASE}] Built basins: {len(basins)}")

    candidates = build_candidates(drfast, args.mode, max_candidates=max_candidates, seed=args.seed)
    name_to_ov = {name: ov for name, ov in candidates}
    print(f"[{PHASE}] Stage-1 rescue candidates: {len(candidates)}")

    cache: Dict[Tuple[str, Tuple[float, ...]], Dict[str, Any]] = {}
    screen_rows: List[Dict[str, Any]] = []
    screen_env_rows: List[Dict[str, Any]] = []
    for i, (case, ov) in enumerate(candidates, 1):
        # Use the cheap DR screen, plus more directed stress cases in non-turbo modes.
        env = drfast.make_screen_envelope(dr, ov, seed=args.seed + 10000 + i, mode=args.mode)
        agg, rows = drfast.evaluate_envelope(dr, case, ov, basins, seed=args.seed + 10000 + i, envelope=env, cache=cache, phase_label="screen")
        agg["ds_rescue_score"] = rescue_score(agg, target_cap, target_tan, target_q20)
        screen_rows.append(agg)
        screen_env_rows.extend(rows)
        if i <= 5 or i % print_every == 0:
            print(
                f"[{PHASE}] screen {i:04d}/{len(candidates)} {case[:64]:64s} "
                f"DS={agg['ds_rescue_score']:.3f} DR={agg['dr_score']:.3f} "
                f"q20={agg['q20_dp_score']:.3f} worstDP={agg['worst_dp_score']:.3f} "
                f"worstCap={agg['worst_capture_rate']:.3f} worstTan={agg['worst_tangent_ratio']:.3f}"
            )

    screen_df = pd.DataFrame(screen_rows).sort_values("ds_rescue_score", ascending=False)
    screen_env = pd.DataFrame(screen_env_rows)
    screen_df.to_csv(ROOT / "phase26ds_screen_results.csv", index=False)
    screen_env.to_csv(ROOT / "phase26ds_screen_envelope_results.csv", index=False)

    stage2_names = promote_stage2(screen_df, stage2_top, target_cap, target_tan)
    stage2_candidates = [(name, name_to_ov[name]) for name in stage2_names if name in name_to_ov]
    print(f"[{PHASE}] Stage-2 full-envelope candidates: {len(stage2_candidates)}")

    full_rows: List[Dict[str, Any]] = []
    full_env_rows: List[Dict[str, Any]] = []
    for i, (case, ov) in enumerate(stage2_candidates, 1):
        env = dr.envelope_cases(ov, seed=args.seed + 20000 + i)
        agg, rows = drfast.evaluate_envelope(dr, case, ov, basins, seed=args.seed + 20000 + i, envelope=env, cache=cache, phase_label="full")
        agg["ds_rescue_score"] = rescue_score(agg, target_cap, target_tan, target_q20)
        full_rows.append(agg)
        full_env_rows.extend(rows)
        print(
            f"[{PHASE}] full   {i:03d}/{len(stage2_candidates)} {case[:64]:64s} "
            f"DS={agg['ds_rescue_score']:.3f} DR={agg['dr_score']:.3f} "
            f"q20={agg['q20_dp_score']:.3f} worstDP={agg['worst_dp_score']:.3f} "
            f"worstCap={agg['worst_capture_rate']:.3f} worstTan={agg['worst_tangent_ratio']:.3f} "
            f"strict={agg['strict_survival_rate']:.2f} soft={agg['survival_rate']:.2f}"
        )

    full_df = pd.DataFrame(full_rows).sort_values("ds_rescue_score", ascending=False) if full_rows else pd.DataFrame()
    full_env = pd.DataFrame(full_env_rows)
    full_df.to_csv(ROOT / "phase26ds_case_results.csv", index=False)
    full_env.to_csv(ROOT / "phase26ds_envelope_results.csv", index=False)

    try:
        pf = dr.pareto_front(full_df) if not full_df.empty else full_df.copy()
    except Exception:
        pf = full_df.sort_values("ds_rescue_score", ascending=False).head(20).copy() if not full_df.empty else full_df.copy()
    pf.to_csv(ROOT / "phase26ds_pareto_front.csv", index=False)

    plot_bar(screen_df, ROOT / "phase26ds_screen_top_scores.png", "ds_rescue_score", "26DS screen rescue scores")
    plot_bar(full_df, ROOT / "phase26ds_top_scores.png", "ds_rescue_score", "26DS full-envelope rescue scores")
    plot_scatter(full_df, pf, ROOT / "phase26ds_capture_vs_tangent_pareto.png", target_cap, target_tan)

    strict_df = full_df[(full_df["worst_capture_rate"] >= target_cap) & (full_df["worst_tangent_ratio"] <= target_tan)].copy() if not full_df.empty else pd.DataFrame()
    best = full_df.iloc[0] if len(full_df) else (screen_df.iloc[0] if len(screen_df) else pd.Series(dtype=float))
    best_strict = strict_df.sort_values("ds_rescue_score", ascending=False).iloc[0] if len(strict_df) else pd.Series(dtype=float)

    plot_errors: List[str] = []
    if args.inherited_plots and len(best):
        try:
            plot_errors = dr.inherited_plots(z2, rel_ids, basins, best)
        except Exception as e:
            plot_errors = [repr(e)]

    def row_overrides(row: pd.Series) -> Dict[str, float] | None:
        if not len(row):
            return None
        return {k: fnum(row[k]) for k in PARAMS if k in row and pd.notna(row[k])}

    summary = {
        "phase": PHASE,
        "title": TITLE,
        "mode": args.mode,
        "torch_info": torch_info,
        "targets": {
            "target_worst_tangent_ratio": target_tan,
            "target_worst_capture_rate": target_cap,
            "target_q20_dp_score": target_q20,
        },
        "num_candidates": int(len(candidates)),
        "num_screen_envelope_evals": int(len(screen_env)),
        "num_full_candidates": int(len(stage2_candidates)),
        "num_full_envelope_evals": int(len(full_env)),
        "num_cached_eval_entries": int(len(cache)),
        "num_pareto": int(len(pf)),
        "num_strict_line_pass": int(len(strict_df)),
        "elapsed_sec": float(time.time() - t0),
        "best_case": None if not len(best) else str(best.get("case")),
        "best_ds_rescue_score": None if not len(best) else fnum(best.get("ds_rescue_score")),
        "best_dr_score": None if not len(best) else fnum(best.get("dr_score")),
        "best_q20_dp_score": None if not len(best) else fnum(best.get("q20_dp_score")),
        "best_worst_dp_score": None if not len(best) else fnum(best.get("worst_dp_score")),
        "best_worst_capture_rate": None if not len(best) else fnum(best.get("worst_capture_rate")),
        "best_worst_tangent_ratio": None if not len(best) else fnum(best.get("worst_tangent_ratio")),
        "best_overrides": row_overrides(best),
        "best_strict_case": None if not len(best_strict) else str(best_strict.get("case")),
        "best_strict_ds_rescue_score": None if not len(best_strict) else fnum(best_strict.get("ds_rescue_score")),
        "best_strict_overrides": row_overrides(best_strict),
        "screen_top10": screen_df.head(10).to_dict(orient="records") if len(screen_df) else [],
        "full_top10": full_df.head(10).to_dict(orient="records") if len(full_df) else [],
        "pareto_top10": pf.head(10).to_dict(orient="records") if len(pf) else [],
        "plot_errors": plot_errors,
        "outputs": [
            "phase26ds_screen_results.csv",
            "phase26ds_screen_envelope_results.csv",
            "phase26ds_case_results.csv",
            "phase26ds_envelope_results.csv",
            "phase26ds_pareto_front.csv",
            "phase26ds_summary.json",
            "phase26ds_screen_top_scores.png",
            "phase26ds_top_scores.png",
            "phase26ds_capture_vs_tangent_pareto.png",
        ],
    }
    with open(ROOT / "phase26ds_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[{PHASE}] Done in {summary['elapsed_sec']:.1f}s")
    if len(best):
        print(
            f"[{PHASE}] BEST: {summary['best_case']} | "
            f"DS={summary['best_ds_rescue_score']:.4f} DR={summary['best_dr_score']:.4f} "
            f"q20DP={summary['best_q20_dp_score']:.4f} worstDP={summary['best_worst_dp_score']:.4f} "
            f"worstCap={summary['best_worst_capture_rate']:.3f} worstTan={summary['best_worst_tangent_ratio']:.3f}"
        )
        print(f"[{PHASE}] BEST overrides: {summary['best_overrides']}")
    if len(best_strict):
        print(f"[{PHASE}] BEST STRICT-LINE PASS: {summary['best_strict_case']} | DS={summary['best_strict_ds_rescue_score']:.4f}")
        print(f"[{PHASE}] BEST STRICT overrides: {summary['best_strict_overrides']}")
    else:
        print(f"[{PHASE}] No full-envelope candidate passed strict line worstCap>={target_cap:.3f} and worstTan<={target_tan:.2f}.")
    print(f"[{PHASE}] Wrote summary: {ROOT / 'phase26ds_summary.json'}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=TITLE)
    p.add_argument("--mode", choices=["turbo", "balanced", "careful"], default="turbo")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--max-candidates", type=int, default=None)
    p.add_argument("--stage2-top", type=int, default=None)
    p.add_argument("--print-every", type=int, default=None)
    p.add_argument("--seed", type=int, default=26019)
    p.add_argument("--inherited-plots", action="store_true", help="Generate inherited AP/DQ best-case plots; slower and off by default.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
