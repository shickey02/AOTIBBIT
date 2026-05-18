#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 26DR-FAST — staged strict confidence-envelope hardening pass.

Why this exists:
    The original 26DR strict-envelope script evaluates every candidate against a
    large confidence envelope. That is scientifically useful, but it is slow:
    candidates × envelope cases × AP/DQ inherited rollout work.

This version keeps the same scoring logic and parameter family, but makes the
search staged:
    1. Cheap screen over many candidates with a small envelope.
    2. Full strict envelope only on the survivors/top candidates.
    3. Optional final inherited plots only for the best result.

CUDA note:
    This script enables CUDA-friendly torch settings and reports GPU status. The
    actual speedup depends on the inherited build_basins/eval_base_case stack.
    If those routines already use torch tensors on cuda, this will use it. If the
    bottleneck is Python loops / pandas / matplotlib / CPU-side rollout logic,
    CUDA will not magically accelerate that part, so the staged evaluation is the
    main speed win.

Recommended first run:
    python bbit_geomlang/geomlang_phase26dr_fast_staged_cuda_basic32_E_drive.py --mode turbo --device cuda

More complete run:
    python bbit_geomlang/geomlang_phase26dr_fast_staged_cuda_basic32_E_drive.py --mode balanced --device cuda

Fuller validation of the top few:
    python bbit_geomlang/geomlang_phase26dr_fast_staged_cuda_basic32_E_drive.py --mode careful --device cuda
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHASE = "26DR_FAST"
TITLE = "Staged CUDA-aware strict confidence-envelope low-tangent hardening"

ROOT = Path(r"E:\BBIT\outputs_basic32")
LOCAL_ROOT = Path(__file__).resolve().parent
DR_FILENAME = "geomlang_phase26dr_strict_confidence_envelope_low_tangent_hardening_basic32_E_drive.py"

# Same strict-line targets as 26DR. These are read from the DR module when
# present, but kept here as defaults so the script remains robust.
DEFAULT_TARGET_WORST_TAN = 2.10
DEFAULT_TARGET_WORST_CAP = 0.285
DEFAULT_TARGET_Q20_DP = 2.30
DEFAULT_TARGET_WORST_DP = 1.95

PARAMS = [
    "BOWL_RADIUS_FRAC",
    "BOWL_SEAT_AXIS_GAIN",
    "BOWL_DIRECTIONAL_BLEND",
    "BOWL_NORM_CAP_MULT",
    "BOWL_SHELL_RADIAL_GAIN",
    "BOWL_TANGENT_KILL",
]


def import_phase26dr() -> Any:
    r"""Import the original 26DR file from the current folder or E:\BBIT path."""
    candidates = [
        LOCAL_ROOT / DR_FILENAME,
        Path(r"E:\BBIT\bbit_geomlang") / DR_FILENAME,
        ROOT.parent / "bbit_geomlang" / DR_FILENAME,
    ]
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location("phase26dr_original", str(path))
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules["phase26dr_original"] = mod
            spec.loader.exec_module(mod)
            print(f"[{PHASE}] Loaded 26DR base module: {path}")
            return mod
    raise FileNotFoundError(
        "Could not find the original 26DR script. Put this file in the same "
        f"folder as {DR_FILENAME}, or keep it at E:\\BBIT\\bbit_geomlang."
    )


def setup_torch(device: str) -> Dict[str, Any]:
    """Enable low-risk torch/CUDA speed settings if torch is available."""
    info: Dict[str, Any] = {"torch_available": False, "cuda_available": False, "selected_device": "unknown"}
    try:
        import torch
        info["torch_available"] = True
        info["cuda_available"] = bool(torch.cuda.is_available())
        if device == "auto":
            selected = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            selected = "cpu"
            print(f"[{PHASE}] Requested CUDA, but torch.cuda.is_available() is False. Falling back to CPU.")
        else:
            selected = device
        info["selected_device"] = selected
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
        # Do NOT call torch.set_default_device("cuda") here. Some inherited code
        # creates CPU numpy views / CPU tensors intentionally; forcing a global
        # default device can break that. The inherited modules should move their
        # model/data to CUDA themselves when supported.
    except Exception as e:
        info["torch_error"] = repr(e)
    print(f"[{PHASE}] Torch/CUDA: {info}")
    return info


def fnum(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def row_key(ov: Dict[str, float]) -> Tuple[float, ...]:
    return tuple(round(float(ov[k]), 5) for k in PARAMS)


def short_directed_stress_variants(dr: Any, base: Dict[str, float]) -> List[Tuple[str, Dict[str, float]]]:
    """A small but representative directed-stress subset for screening."""
    specs = [
        ("stress_cap_tk", {"BOWL_NORM_CAP_MULT": -0.035, "BOWL_TANGENT_KILL": +0.055}),
        ("stress_shell_blend", {"BOWL_SHELL_RADIAL_GAIN": -0.014, "BOWL_DIRECTIONAL_BLEND": -0.035}),
        ("stress_radius_in", {"BOWL_RADIUS_FRAC": -0.026}),
        ("stress_seat_down", {"BOWL_SEAT_AXIS_GAIN": -0.040}),
        ("stress_tk_drop", {"BOWL_TANGENT_KILL": -0.060, "BOWL_SHELL_RADIAL_GAIN": +0.012}),
        ("stress_cap_blend_drop", {"BOWL_NORM_CAP_MULT": -0.045, "BOWL_DIRECTIONAL_BLEND": -0.035}),
    ]
    out: List[Tuple[str, Dict[str, float]]] = []
    for label, delta in specs:
        ov = dict(base)
        for k, d in delta.items():
            ov[k] = dr.clamp_param(k, ov[k] + float(d))
        out.append((label, dr.full_params(ov)))
    return out


def make_screen_envelope(dr: Any, base: Dict[str, float], seed: int, mode: str) -> List[Tuple[str, Dict[str, float], str]]:
    """Cheap envelope for Stage 1. Keeps the same kinds, fewer cases."""
    rng = random.Random(seed)
    out: List[Tuple[str, Dict[str, float], str]] = [("base", dr.full_params(base), "base")]

    if mode == "turbo":
        small_n, medium_n, large_n = 1, 1, 0
        directed = short_directed_stress_variants(dr, base)[:4]
    elif mode == "balanced":
        small_n, medium_n, large_n = 2, 2, 1
        directed = short_directed_stress_variants(dr, base)
    else:  # careful screen, still smaller than full 26DR
        small_n, medium_n, large_n = 3, 3, 2
        directed = short_directed_stress_variants(dr, base)

    for i in range(small_n):
        out.append((f"screen_small_{i:02d}", dr.jitter_variant(base, rng, scale=1.0), "jitter_small"))
    for i in range(medium_n):
        out.append((f"screen_medium_{i:02d}", dr.jitter_variant(base, rng, scale=1.65), "jitter_medium"))
    for i in range(large_n):
        out.append((f"screen_large_{i:02d}", dr.jitter_variant(base, rng, scale=2.35), "jitter_large"))
    for label, ov in directed:
        out.append((label, ov, "directed_stress"))
    return out


def evaluate_envelope(
    dr: Any,
    case: str,
    ov: Dict[str, float],
    basins: Any,
    seed: int,
    envelope: Sequence[Tuple[str, Dict[str, float], str]],
    cache: Dict[Tuple[str, Tuple[float, ...]], Dict[str, Any]],
    phase_label: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    for label, eov, kind in envelope:
        name = f"{case}__{phase_label}__{label}"
        ck = (kind + ":" + label, row_key(eov))
        if ck in cache:
            r = dict(cache[ck])
            r["case"] = name
        else:
            try:
                r = dr.eval_base_case(name, eov, basins)
            except Exception as e:
                r = {"case": name, "error": repr(e), "dp_score": float("nan"), **eov}
            cache[ck] = dict(r)
        r["parent_case"] = case
        r["envelope_label"] = label
        r["envelope_kind"] = kind
        r["stage"] = phase_label
        rows.append(r)

    agg = dr.strict_score_from_envelope(rows)
    base_row = next((r for r in rows if r.get("envelope_label") == "base"), rows[0])
    out: Dict[str, Any] = {
        "case": case,
        "stage": phase_label,
        **{k: float(ov[k]) for k in PARAMS},
        **agg,
        "base_dp_score": fnum(base_row.get("dp_score")),
        "base_capture_rate": fnum(base_row.get("capture_rate")),
        "base_tangent_ratio": fnum(base_row.get("tangent_ratio")),
        "base_distance_progress": fnum(base_row.get("distance_progress")),
        "base_signed_alignment": fnum(base_row.get("signed_alignment")),
        "overrides": json.dumps({k: float(ov[k]) for k in PARAMS}, sort_keys=True),
        "num_envelope_cases": int(len(rows)),
    }
    return out, rows


def default_mode_config(mode: str) -> Dict[str, int]:
    if mode == "turbo":
        return {"max_candidates": 80, "stage2_top": 10, "final_top": 3, "print_every": 5}
    if mode == "balanced":
        return {"max_candidates": 160, "stage2_top": 18, "final_top": 5, "print_every": 8}
    return {"max_candidates": 260, "stage2_top": 30, "final_top": 8, "print_every": 10}


def limit_candidates(candidates: List[Tuple[str, Dict[str, float]]], max_candidates: int) -> List[Tuple[str, Dict[str, float]]]:
    """
    Keep diverse candidates while limiting work. Original candidate order is not
    random: it contains priors, hardening packets, random cloud, then microgrid.
    We take a head slice plus evenly spaced coverage across the remainder.
    """
    if max_candidates <= 0 or len(candidates) <= max_candidates:
        return candidates
    head_n = min(max_candidates // 2, 70, len(candidates))
    out = list(candidates[:head_n])
    remain = candidates[head_n:]
    slots = max_candidates - len(out)
    if slots > 0 and remain:
        idxs = np.linspace(0, len(remain) - 1, slots).round().astype(int).tolist()
        seen = {name for name, _ in out}
        for idx in idxs:
            name, ov = remain[idx]
            if name not in seen:
                out.append((name, ov))
                seen.add(name)
    return out[:max_candidates]


def plot_fast_top(df: pd.DataFrame, out_path: Path, score_col: str, title: str) -> None:
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


def plot_fast_scatter(df: pd.DataFrame, pf: pd.DataFrame, out_path: Path, target_cap: float, target_tan: float) -> None:
    if df.empty:
        return
    plt.figure(figsize=(10, 7))
    sc = plt.scatter(
        df["worst_capture_rate"],
        df["worst_tangent_ratio"],
        c=df["dr_score"],
        s=72,
        alpha=0.76,
    )
    if not pf.empty:
        plt.scatter(
            pf["worst_capture_rate"], pf["worst_tangent_ratio"],
            facecolors="none", edgecolors="black", s=170, linewidths=1.7,
            label="Pareto front",
        )
        plt.legend()
    plt.axvline(target_cap, linewidth=1.0, alpha=0.45)
    plt.axhline(target_tan, linewidth=1.0, alpha=0.45)
    plt.colorbar(sc, label="DR score")
    plt.xlabel("worst envelope capture_rate")
    plt.ylabel("worst envelope tangent/radial ratio")
    plt.title("26DR-FAST strict envelope capture vs tangent drift")
    plt.grid(True, alpha=0.28)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def safe_pareto(dr: Any, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    try:
        return dr.pareto_front(df)
    except Exception:
        return df.sort_values("dr_score", ascending=False).head(20).copy()


def run(args: argparse.Namespace) -> Dict[str, Any]:
    t0 = time.time()
    ROOT.mkdir(parents=True, exist_ok=True)
    torch_info = setup_torch(args.device)
    dr = import_phase26dr()

    target_tan = float(getattr(dr, "TARGET_WORST_TAN", DEFAULT_TARGET_WORST_TAN))
    target_cap = float(getattr(dr, "TARGET_WORST_CAP", DEFAULT_TARGET_WORST_CAP))
    target_q20 = float(getattr(dr, "TARGET_Q20_DP", DEFAULT_TARGET_Q20_DP))

    cfg = default_mode_config(args.mode)
    max_candidates = args.max_candidates if args.max_candidates is not None else cfg["max_candidates"]
    stage2_top = args.stage2_top if args.stage2_top is not None else cfg["stage2_top"]
    final_top = args.final_top if args.final_top is not None else cfg["final_top"]
    print_every = args.print_every if args.print_every is not None else cfg["print_every"]

    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] mode={args.mode} max_candidates={max_candidates} stage2_top={stage2_top} final_top={final_top}")
    print(f"[{PHASE}] strict targets: worstTan<={target_tan:.2f}, worstCap>={target_cap:.3f}, q20DP>={target_q20:.2f}")

    # Same basin construction path as 26DR.
    eval_helper = getattr(dr, "EVAL_HELPER", None)
    if eval_helper is None or not hasattr(eval_helper, "build_basins"):
        eval_helper = dr.previous_eval_helper()
    z2, rel_ids, basins = eval_helper.build_basins()
    print(f"[{PHASE}] Built basins: {len(basins)}")

    priors = dr.read_prior_candidates()
    all_candidates = dr.hardening_candidates(priors)
    candidates = limit_candidates(all_candidates, max_candidates)
    print(f"[{PHASE}] Prior candidates: {len(priors)}")
    print(f"[{PHASE}] Original DR candidates: {len(all_candidates)}")
    print(f"[{PHASE}] Stage-1 candidates after limit/diversity: {len(candidates)}")

    cache: Dict[Tuple[str, Tuple[float, ...]], Dict[str, Any]] = {}
    screen_rows: List[Dict[str, Any]] = []
    screen_env_rows: List[Dict[str, Any]] = []

    # Stage 1: cheap screen.
    for i, (case, ov) in enumerate(candidates, 1):
        env = make_screen_envelope(dr, ov, seed=9000 + i, mode=args.mode)
        agg, rows = evaluate_envelope(dr, case, ov, basins, seed=9000 + i, envelope=env, cache=cache, phase_label="screen")
        screen_rows.append(agg)
        screen_env_rows.extend(rows)
        if i <= 5 or i % print_every == 0:
            print(
                f"[{PHASE}] screen {i:03d}/{len(candidates)} {case[:68]:68s} "
                f"DR={agg['dr_score']:.3f} q20={agg['q20_dp_score']:.3f} "
                f"worstDP={agg['worst_dp_score']:.3f} worstCap={agg['worst_capture_rate']:.3f} "
                f"worstTan={agg['worst_tangent_ratio']:.3f} env={agg['num_envelope_cases']}"
            )

    screen_df = pd.DataFrame(screen_rows).sort_values("dr_score", ascending=False)
    screen_env = pd.DataFrame(screen_env_rows)
    screen_df.to_csv(ROOT / "phase26dr_fast_screen_results.csv", index=False)
    screen_env.to_csv(ROOT / "phase26dr_fast_screen_envelope_results.csv", index=False)

    # Preserve a few different kinds of promising candidate, not just the raw top.
    top_by_score = screen_df.head(stage2_top)
    top_by_capture = screen_df.sort_values(["worst_capture_rate", "dr_score"], ascending=False).head(max(4, stage2_top // 4))
    top_by_tangent = screen_df.sort_values(["worst_tangent_ratio", "dr_score"], ascending=[True, False]).head(max(4, stage2_top // 4))
    strictish = screen_df[(screen_df["worst_capture_rate"] >= target_cap * 0.92) & (screen_df["worst_tangent_ratio"] <= target_tan * 1.50)].head(stage2_top)
    stage2_names = pd.concat([top_by_score, top_by_capture, top_by_tangent, strictish], ignore_index=True).drop_duplicates("case")["case"].tolist()
    stage2_names = stage2_names[: max(stage2_top, len(stage2_names))]
    name_to_ov = {name: ov for name, ov in candidates}
    stage2_candidates = [(name, name_to_ov[name]) for name in stage2_names if name in name_to_ov]
    print(f"[{PHASE}] Stage-2 full-envelope candidates: {len(stage2_candidates)}")

    # Stage 2: original full DR envelope only for the survivors.
    full_rows: List[Dict[str, Any]] = []
    full_env_rows: List[Dict[str, Any]] = []
    for i, (case, ov) in enumerate(stage2_candidates, 1):
        env = dr.envelope_cases(ov, seed=11000 + i)
        agg, rows = evaluate_envelope(dr, case, ov, basins, seed=11000 + i, envelope=env, cache=cache, phase_label="full")
        full_rows.append(agg)
        full_env_rows.extend(rows)
        print(
            f"[{PHASE}] full   {i:03d}/{len(stage2_candidates)} {case[:68]:68s} "
            f"DR={agg['dr_score']:.3f} q20={agg['q20_dp_score']:.3f} "
            f"worstDP={agg['worst_dp_score']:.3f} worstCap={agg['worst_capture_rate']:.3f} "
            f"worstTan={agg['worst_tangent_ratio']:.3f} strict={agg['strict_survival_rate']:.2f} soft={agg['survival_rate']:.2f}"
        )

    full_df = pd.DataFrame(full_rows).sort_values("dr_score", ascending=False) if full_rows else pd.DataFrame()
    full_env = pd.DataFrame(full_env_rows)
    full_df.to_csv(ROOT / "phase26dr_fast_case_results.csv", index=False)
    full_env.to_csv(ROOT / "phase26dr_fast_envelope_results.csv", index=False)

    pf = safe_pareto(dr, full_df)
    pf.to_csv(ROOT / "phase26dr_fast_pareto_front.csv", index=False)

    plot_fast_top(screen_df, ROOT / "phase26dr_fast_screen_top_scores.png", "dr_score", "26DR-FAST screen scores")
    plot_fast_top(full_df, ROOT / "phase26dr_fast_top_scores.png", "dr_score", "26DR-FAST full-envelope scores")
    plot_fast_scatter(full_df, pf, ROOT / "phase26dr_fast_capture_vs_tangent_pareto.png", target_cap, target_tan)

    best = full_df.iloc[0] if len(full_df) else (screen_df.iloc[0] if len(screen_df) else pd.Series(dtype=float))
    strict_df = full_df[(full_df["worst_capture_rate"] >= target_cap) & (full_df["worst_tangent_ratio"] <= target_tan)].copy() if len(full_df) else pd.DataFrame()
    best_strict = strict_df.iloc[0] if len(strict_df) else pd.Series(dtype=float)

    plot_errors: List[str] = []
    if args.inherited_plots and len(best):
        try:
            plot_errors = dr.inherited_plots(z2, rel_ids, basins, best)
        except Exception as e:
            plot_errors = [repr(e)]

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
        "num_priors": int(len(priors)),
        "num_original_candidates": int(len(all_candidates)),
        "num_screen_candidates": int(len(candidates)),
        "num_stage2_full_candidates": int(len(stage2_candidates)),
        "num_screen_envelope_evals": int(len(screen_env)),
        "num_full_envelope_evals": int(len(full_env)),
        "num_cached_eval_entries": int(len(cache)),
        "num_pareto": int(len(pf)),
        "num_strict_line_pass": int(len(strict_df)),
        "elapsed_sec": float(time.time() - t0),
        "best_case": None if not len(best) else str(best.get("case")),
        "best_dr_score": None if not len(best) else fnum(best.get("dr_score")),
        "best_q20_dp_score": None if not len(best) else fnum(best.get("q20_dp_score")),
        "best_worst_dp_score": None if not len(best) else fnum(best.get("worst_dp_score")),
        "best_worst_capture_rate": None if not len(best) else fnum(best.get("worst_capture_rate")),
        "best_worst_tangent_ratio": None if not len(best) else fnum(best.get("worst_tangent_ratio")),
        "best_survival_rate": None if not len(best) else fnum(best.get("survival_rate")),
        "best_strict_survival_rate": None if not len(best) else fnum(best.get("strict_survival_rate")),
        "best_overrides": None if not len(best) else {k: fnum(best[k]) for k in PARAMS if k in best and pd.notna(best[k])},
        "best_strict_case": None if not len(best_strict) else str(best_strict.get("case")),
        "best_strict_dr_score": None if not len(best_strict) else fnum(best_strict.get("dr_score")),
        "best_strict_overrides": None if not len(best_strict) else {k: fnum(best_strict[k]) for k in PARAMS if k in best_strict and pd.notna(best_strict[k])},
        "screen_top10": screen_df.head(10).to_dict(orient="records") if len(screen_df) else [],
        "full_top10": full_df.head(10).to_dict(orient="records") if len(full_df) else [],
        "pareto_top10": pf.head(10).to_dict(orient="records") if len(pf) else [],
        "plot_errors": plot_errors,
        "outputs": [
            "phase26dr_fast_screen_results.csv",
            "phase26dr_fast_screen_envelope_results.csv",
            "phase26dr_fast_case_results.csv",
            "phase26dr_fast_envelope_results.csv",
            "phase26dr_fast_pareto_front.csv",
            "phase26dr_fast_summary.json",
            "phase26dr_fast_screen_top_scores.png",
            "phase26dr_fast_top_scores.png",
            "phase26dr_fast_capture_vs_tangent_pareto.png",
        ],
    }
    with open(ROOT / "phase26dr_fast_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[{PHASE}] Done in {summary['elapsed_sec']:.1f}s")
    if len(best):
        print(
            f"[{PHASE}] BEST: {summary['best_case']} | "
            f"DR={summary['best_dr_score']:.4f} q20DP={summary['best_q20_dp_score']:.4f} "
            f"worstDP={summary['best_worst_dp_score']:.4f} worstCap={summary['best_worst_capture_rate']:.3f} "
            f"worstTan={summary['best_worst_tangent_ratio']:.3f}"
        )
        print(f"[{PHASE}] BEST overrides: {summary['best_overrides']}")
    if len(best_strict):
        print(f"[{PHASE}] BEST STRICT-LINE PASS: {summary['best_strict_case']} | DR={summary['best_strict_dr_score']:.4f}")
        print(f"[{PHASE}] BEST STRICT overrides: {summary['best_strict_overrides']}")
    else:
        print(f"[{PHASE}] No full-envelope candidate passed strict line worstCap>={target_cap:.3f} and worstTan<={target_tan:.2f}.")
    print(f"[{PHASE}] Wrote summary: {ROOT / 'phase26dr_fast_summary.json'}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=TITLE)
    p.add_argument("--mode", choices=["turbo", "balanced", "careful"], default="turbo",
                   help="turbo is fastest; balanced is a good normal run; careful screens more candidates.")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                   help="CUDA-aware setting. Actual speedup depends on inherited torch code.")
    p.add_argument("--max-candidates", type=int, default=None,
                   help="Override number of candidates in the cheap screen.")
    p.add_argument("--stage2-top", type=int, default=None,
                   help="Override number of candidates promoted to full envelope.")
    p.add_argument("--final-top", type=int, default=None,
                   help="Reserved for future deeper validation; currently included in summary config.")
    p.add_argument("--print-every", type=int, default=None)
    p.add_argument("--inherited-plots", action="store_true",
                   help="Generate inherited AP/DQ rollout/vector plots for the best case. Slower; off by default.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
