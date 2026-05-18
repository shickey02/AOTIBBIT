# geomlang_phase26dq_confidence_envelope_lockdown_basic32_E_drive.py
# Phase 26DQ — Confidence-envelope lockdown audit for the DP Pareto-knee winners.
#
# What this phase does:
#   1) Imports Phase 26DP and reuses the same basin/eval machinery.
#   2) Pulls the best DP / Pareto / stability candidates from previous CSVs.
#   3) Re-evaluates each candidate through a compact confidence envelope:
#        - exact/base case
#        - small jitter shell
#        - medium jitter shell
#        - directed stress nudges that should expose overfit candidates
#   4) Scores candidates by worst-case and lower-quantile behavior, not only peak score.
#
# Interpretation:
#   DP found the high knee. DQ asks which knee candidate survives perturbation.
#   A good DQ winner should have respectable capture, controlled tangent drift, and
#   not collapse under small constant changes.

from __future__ import annotations

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
import matplotlib.pyplot as plt

PHASE = "26DQ"
TITLE = "Confidence-envelope lockdown audit for DP Pareto-knee candidates"

ROOT = Path(r"E:\BBIT\outputs_basic32")
if not ROOT.exists():
    ROOT = Path.cwd()

HERE = Path(__file__).resolve().parent
PREV_PATHS = [
    Path(r"E:\BBIT\bbit_geomlang\geomlang_phase26dp_pareto_knee_lockin_low_tangent_basic32_E_drive.py"),
    HERE / "geomlang_phase26dp_pareto_knee_lockin_low_tangent_basic32_E_drive.py",
    ROOT / "geomlang_phase26dp_pareto_knee_lockin_low_tangent_basic32_E_drive.py",
]
PREV_PATH = next((p for p in PREV_PATHS if p.exists()), PREV_PATHS[0])


def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing required previous phase script: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


PREV = load_module(PREV_PATH, "phase26dp_prev")
PARAMS: List[str] = list(getattr(PREV, "PARAMS"))


def previous_eval_helper() -> Any:
    q: List[Any] = [PREV]
    seen: set = set()
    while q:
        obj = q.pop(0)
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        if hasattr(obj, "build_basins") and hasattr(obj, "eval_case"):
            return obj
        for attr in ["PREV", "DM"]:
            child = getattr(obj, attr, None)
            if child is not None:
                q.append(child)
    raise AttributeError("Could not find a previous helper exposing build_basins and eval_case.")


EVAL_HELPER = previous_eval_helper()


def full_params(ov: Dict[str, Any]) -> Dict[str, float]:
    return {k: float(v) for k, v in PREV.full_params(ov).items() if k in PARAMS}


def clamp_param(k: str, v: float) -> float:
    return float(PREV.clamp_param(k, float(v)))


def clean_name(name: str, n: int = 120) -> str:
    s = str(name).replace("\n", " ").replace("\r", " ")
    return s[:n]


def metric(summary: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        if key in summary and summary[key] is not None:
            v = float(summary[key])
            if math.isfinite(v):
                return v
    except Exception:
        pass
    return default


def eval_base_case(name: str, ov: Dict[str, float], basins: Any) -> Dict[str, Any]:
    res = EVAL_HELPER.eval_case(name, ov, basins)
    return PREV.compact_result(name, ov, res)


def dq_score_from_envelope(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate an envelope into a confidence score.

    DQ deliberately downranks fragile high-peak candidates. The most important
    terms are the 20th-percentile DP score, worst DP score, worst tangent, and
    worst capture. This is the opposite of the previous phases' pure search.
    """
    good = [r for r in rows if math.isfinite(float(r.get("dp_score", float("nan"))))]
    if not good:
        return {
            "dq_score": float("nan"),
            "mean_dp_score": float("nan"),
            "q20_dp_score": float("nan"),
            "worst_dp_score": float("nan"),
            "std_dp_score": float("nan"),
            "mean_capture_rate": float("nan"),
            "worst_capture_rate": float("nan"),
            "mean_tangent_ratio": float("nan"),
            "worst_tangent_ratio": float("nan"),
            "survival_rate": 0.0,
        }

    dp = np.array([float(r["dp_score"]) for r in good], dtype=float)
    cap = np.array([float(r.get("capture_rate", 0.0)) for r in good], dtype=float)
    tan = np.array([float(r.get("tangent_ratio", 999.0)) for r in good], dtype=float)
    prog = np.array([float(r.get("distance_progress", 0.0)) for r in good], dtype=float)
    align = np.array([float(r.get("signed_alignment", 0.0)) for r in good], dtype=float)

    q20 = float(np.quantile(dp, 0.20))
    q35 = float(np.quantile(dp, 0.35))
    worst = float(np.min(dp))
    mean = float(np.mean(dp))
    std = float(np.std(dp))
    worst_cap = float(np.min(cap))
    mean_cap = float(np.mean(cap))
    worst_tan = float(np.max(tan))
    mean_tan = float(np.mean(tan))
    mean_prog = float(np.mean(prog))
    mean_align = float(np.mean(align))

    # Survives if it stays in the usable knee band, not necessarily perfect.
    survive = np.logical_and(cap >= 0.285, tan <= 2.10)
    survival_rate = float(np.mean(survive))

    # Strongly prefer candidates that remain low-tangent across the envelope.
    tangent_credit = 1.25 / (1.0 + max(0.0, mean_tan - 1.00) ** 1.45)
    worst_tangent_penalty = 0.34 * max(0.0, worst_tan - 1.75) + 0.75 * max(0.0, worst_tan - 2.75)
    worst_capture_penalty = 5.50 * max(0.0, 0.285 - worst_cap)
    volatility_penalty = 0.38 * std

    dq = (
        0.42 * q20
        + 0.25 * q35
        + 0.18 * worst
        + 0.15 * mean
        + 0.52 * survival_rate
        + tangent_credit
        + 0.65 * mean_prog
        + 0.22 * mean_align
        - worst_tangent_penalty
        - worst_capture_penalty
        - volatility_penalty
    )

    return {
        "dq_score": float(dq),
        "mean_dp_score": mean,
        "q20_dp_score": q20,
        "q35_dp_score": q35,
        "worst_dp_score": worst,
        "std_dp_score": std,
        "mean_capture_rate": mean_cap,
        "worst_capture_rate": worst_cap,
        "mean_tangent_ratio": mean_tan,
        "worst_tangent_ratio": worst_tan,
        "mean_distance_progress": mean_prog,
        "mean_signed_alignment": mean_align,
        "survival_rate": survival_rate,
    }


def parse_overrides_from_row(row: pd.Series) -> Optional[Dict[str, float]]:
    if hasattr(PREV, "parse_overrides_from_row"):
        try:
            ov = PREV.parse_overrides_from_row(row)
            if ov:
                return full_params(ov)
        except Exception:
            pass
    vals: Dict[str, float] = {}
    for k in PARAMS:
        if k in row and pd.notna(row[k]):
            vals[k] = clamp_param(k, float(row[k]))
    if vals:
        return full_params(vals)
    if "overrides" in row and isinstance(row["overrides"], str):
        try:
            raw = json.loads(row["overrides"])
            if isinstance(raw, dict):
                return full_params({k: float(v) for k, v in raw.items() if k in PARAMS})
        except Exception:
            pass
    return None


def add_candidate(acc: List[Tuple[str, Dict[str, float]]], seen: set, name: str, ov: Dict[str, Any]) -> None:
    clean = full_params({k: clamp_param(k, float(v)) for k, v in ov.items() if k in PARAMS})
    key = tuple(round(clean[k], 5) for k in PARAMS)
    if key in seen:
        return
    seen.add(key)
    acc.append((clean_name(name), clean))


def read_prior_candidates() -> List[Tuple[str, Dict[str, float]]]:
    files = [
        ROOT / "phase26dp_stability_summary.csv",
        ROOT / "phase26dp_pareto_front.csv",
        ROOT / "phase26dp_case_results.csv",
        ROOT / "phase26do_stability_summary.csv",
        ROOT / "phase26do_pareto_front.csv",
        ROOT / "phase26dn_stability_summary.csv",
    ]
    out: List[Tuple[str, Dict[str, float]]] = []
    seen: set = set()

    for path in files:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        score_order = [
            "dq_score", "stability_score", "dp_score", "worst_jitter_dp_score",
            "do_score", "dn_score", "dl_scalar_score", "score",
        ]
        sort_cols = [c for c in score_order if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols[0], ascending=False)

        limit = 18 if "case_results" in path.name else 12
        for i, row in df.head(limit).iterrows():
            ov = parse_overrides_from_row(row)
            if not ov:
                continue
            stem = path.stem.replace("phase26", "p26")
            nm = str(row.get("case", f"row_{i}"))
            add_candidate(out, seen, f"dq_prior_{stem}_{nm}", ov)

    # Explicitly include DP's own candidates if available.
    if hasattr(PREV, "KNEE_CENTERS"):
        for name, ov in getattr(PREV, "KNEE_CENTERS"):
            add_candidate(out, seen, f"dq_dp_center_{name}", ov)

    return out


def local_lockdown_candidates(priors: List[Tuple[str, Dict[str, float]]]) -> List[Tuple[str, Dict[str, float]]]:
    out: List[Tuple[str, Dict[str, float]]] = []
    seen: set = set()
    for name, ov in priors:
        add_candidate(out, seen, name, ov)

    # Use a compact center: the best prior available, else the current constants.
    centers = [ov for _, ov in priors[:8]] or [full_params({})]

    # Directed micro adjustments: bias toward lower tangent without giving up the knee.
    for ci, base in enumerate(centers[:6]):
        for cap_delta, tk_delta, shell_delta, blend_delta in [
            (0.00, 0.00, 0.00, 0.00),
            (-0.02, +0.03, -0.01, -0.02),
            (+0.02, +0.02, 0.00, -0.01),
            (+0.04, -0.02, +0.01, +0.01),
            (-0.04, +0.05, -0.02, -0.03),
        ]:
            ov = dict(base)
            ov["BOWL_NORM_CAP_MULT"] = clamp_param("BOWL_NORM_CAP_MULT", ov["BOWL_NORM_CAP_MULT"] + cap_delta)
            ov["BOWL_TANGENT_KILL"] = clamp_param("BOWL_TANGENT_KILL", ov["BOWL_TANGENT_KILL"] + tk_delta)
            ov["BOWL_SHELL_RADIAL_GAIN"] = clamp_param("BOWL_SHELL_RADIAL_GAIN", ov["BOWL_SHELL_RADIAL_GAIN"] + shell_delta)
            ov["BOWL_DIRECTIONAL_BLEND"] = clamp_param("BOWL_DIRECTIONAL_BLEND", ov["BOWL_DIRECTIONAL_BLEND"] + blend_delta)
            add_candidate(out, seen, f"dq_lockdown_c{ci:02d}_cap{cap_delta:+.2f}_tk{tk_delta:+.2f}_sh{shell_delta:+.2f}_bl{blend_delta:+.2f}", ov)

    # Small random cloud around the survivability band; deterministic seed.
    rng = random.Random(2626)
    for j in range(32):
        base = dict(rng.choice(centers))
        ov = {
            "BOWL_RADIUS_FRAC": base["BOWL_RADIUS_FRAC"] + rng.uniform(-0.016, 0.016),
            "BOWL_SEAT_AXIS_GAIN": base["BOWL_SEAT_AXIS_GAIN"] + rng.uniform(-0.022, 0.022),
            "BOWL_DIRECTIONAL_BLEND": base["BOWL_DIRECTIONAL_BLEND"] + rng.uniform(-0.022, 0.022),
            "BOWL_NORM_CAP_MULT": base["BOWL_NORM_CAP_MULT"] + rng.uniform(-0.026, 0.026),
            "BOWL_SHELL_RADIAL_GAIN": base["BOWL_SHELL_RADIAL_GAIN"] + rng.uniform(-0.010, 0.010),
            "BOWL_TANGENT_KILL": base["BOWL_TANGENT_KILL"] + rng.uniform(-0.035, 0.035),
        }
        add_candidate(out, seen, f"dq_confidence_cloud_{j:02d}", ov)

    return out


def jitter_variant(base: Dict[str, float], rng: random.Random, scale: float) -> Dict[str, float]:
    return full_params({
        "BOWL_RADIUS_FRAC": base["BOWL_RADIUS_FRAC"] + rng.uniform(-0.012, 0.012) * scale,
        "BOWL_SEAT_AXIS_GAIN": base["BOWL_SEAT_AXIS_GAIN"] + rng.uniform(-0.018, 0.018) * scale,
        "BOWL_DIRECTIONAL_BLEND": base["BOWL_DIRECTIONAL_BLEND"] + rng.uniform(-0.018, 0.018) * scale,
        "BOWL_NORM_CAP_MULT": base["BOWL_NORM_CAP_MULT"] + rng.uniform(-0.018, 0.018) * scale,
        "BOWL_SHELL_RADIAL_GAIN": base["BOWL_SHELL_RADIAL_GAIN"] + rng.uniform(-0.008, 0.008) * scale,
        "BOWL_TANGENT_KILL": base["BOWL_TANGENT_KILL"] + rng.uniform(-0.025, 0.025) * scale,
    })


def directed_stress_variants(base: Dict[str, float]) -> List[Tuple[str, Dict[str, float]]]:
    specs = [
        ("stress_low_cap_high_tk", {"BOWL_NORM_CAP_MULT": -0.030, "BOWL_TANGENT_KILL": +0.040}),
        ("stress_high_cap_low_tk", {"BOWL_NORM_CAP_MULT": +0.035, "BOWL_TANGENT_KILL": -0.040}),
        ("stress_low_shell_low_blend", {"BOWL_SHELL_RADIAL_GAIN": -0.014, "BOWL_DIRECTIONAL_BLEND": -0.026}),
        ("stress_high_shell_high_blend", {"BOWL_SHELL_RADIAL_GAIN": +0.014, "BOWL_DIRECTIONAL_BLEND": +0.026}),
        ("stress_radius_in", {"BOWL_RADIUS_FRAC": -0.018}),
        ("stress_radius_out", {"BOWL_RADIUS_FRAC": +0.018}),
        ("stress_seat_down", {"BOWL_SEAT_AXIS_GAIN": -0.026}),
        ("stress_seat_up", {"BOWL_SEAT_AXIS_GAIN": +0.026}),
    ]
    out: List[Tuple[str, Dict[str, float]]] = []
    for label, delta in specs:
        ov = dict(base)
        for k, d in delta.items():
            ov[k] = clamp_param(k, ov[k] + float(d))
        out.append((label, full_params(ov)))
    return out


def envelope_cases(base: Dict[str, float], seed: int) -> List[Tuple[str, Dict[str, float], str]]:
    rng = random.Random(seed)
    out: List[Tuple[str, Dict[str, float], str]] = [("base", full_params(base), "base")]
    for i in range(4):
        out.append((f"jitter_small_{i:02d}", jitter_variant(base, rng, scale=1.0), "jitter_small"))
    for i in range(4):
        out.append((f"jitter_medium_{i:02d}", jitter_variant(base, rng, scale=1.75), "jitter_medium"))
    for label, ov in directed_stress_variants(base):
        out.append((label, ov, "directed_stress"))
    return out


def evaluate_candidate_envelope(case: str, ov: Dict[str, float], basins: Any, seed: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    for label, eov, kind in envelope_cases(ov, seed=seed):
        name = f"{case}__{label}"
        try:
            r = eval_base_case(name, eov, basins)
        except Exception as e:
            r = {"case": name, "error": repr(e), "dp_score": float("nan"), **eov}
        r["parent_case"] = case
        r["envelope_label"] = label
        r["envelope_kind"] = kind
        rows.append(r)

    agg = dq_score_from_envelope(rows)
    base_row = next((r for r in rows if r.get("envelope_label") == "base"), rows[0])
    out: Dict[str, Any] = {
        "case": case,
        **{k: float(ov[k]) for k in PARAMS},
        **agg,
        "base_dp_score": float(base_row.get("dp_score", float("nan"))),
        "base_capture_rate": float(base_row.get("capture_rate", float("nan"))),
        "base_tangent_ratio": float(base_row.get("tangent_ratio", float("nan"))),
        "base_distance_progress": float(base_row.get("distance_progress", float("nan"))),
        "base_signed_alignment": float(base_row.get("signed_alignment", float("nan"))),
        "overrides": json.dumps({k: float(ov[k]) for k in PARAMS}, sort_keys=True),
    }
    return out, rows


def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    work = df.copy()
    for c, default in [
        ("dq_score", 0.0), ("q20_dp_score", 0.0), ("worst_dp_score", 0.0),
        ("survival_rate", 0.0), ("worst_capture_rate", 0.0), ("worst_tangent_ratio", 999.0),
    ]:
        if c not in work:
            work[c] = default
    vals = work[["dq_score", "q20_dp_score", "worst_dp_score", "survival_rate", "worst_capture_rate", "worst_tangent_ratio"]].to_numpy(float)
    keep: List[int] = []
    for i, a in enumerate(vals):
        dominated = False
        for j, b in enumerate(vals):
            if i == j:
                continue
            ge = (b[0] >= a[0] and b[1] >= a[1] and b[2] >= a[2] and b[3] >= a[3] and b[4] >= a[4] and b[5] <= a[5])
            gt = (b[0] > a[0] or b[1] > a[1] or b[2] > a[2] or b[3] > a[3] or b[4] > a[4] or b[5] < a[5])
            if ge and gt:
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return work.iloc[keep].sort_values("dq_score", ascending=False)


def plot_top_scores(df: pd.DataFrame) -> None:
    if df.empty:
        return
    top = df.sort_values("dq_score", ascending=False).head(32).iloc[::-1]
    plt.figure(figsize=(15, max(8, 0.52 * len(top))))
    plt.barh(top["case"], top["dq_score"])
    plt.xlabel("DQ confidence-envelope score")
    plt.title("26DQ confidence-envelope lockdown scores")
    plt.tight_layout()
    plt.savefig(ROOT / "phase26dq_top_scores.png", dpi=150)
    plt.close()


def plot_capture_tangent(df: pd.DataFrame, pf: pd.DataFrame) -> None:
    if df.empty:
        return
    plt.figure(figsize=(10, 7))
    sc = plt.scatter(
        df["worst_capture_rate"],
        df["worst_tangent_ratio"],
        c=df["dq_score"],
        s=68,
        alpha=0.76,
    )
    if not pf.empty:
        plt.scatter(
            pf["worst_capture_rate"], pf["worst_tangent_ratio"],
            facecolors="none", edgecolors="black", s=165, linewidths=1.7,
            label="DQ Pareto front",
        )
        plt.legend()
    plt.axvline(0.285, linewidth=1.0, alpha=0.45)
    plt.axhline(2.10, linewidth=1.0, alpha=0.45)
    plt.colorbar(sc, label="DQ score")
    plt.xlabel("worst envelope capture_rate")
    plt.ylabel("worst envelope tangent/radial ratio")
    plt.title("26DQ worst-envelope capture vs tangent drift")
    plt.grid(True, alpha=0.28)
    plt.tight_layout()
    plt.savefig(ROOT / "phase26dq_capture_vs_tangent_pareto.png", dpi=150)
    plt.close()


def plot_envelope_bars(df: pd.DataFrame) -> None:
    if df.empty:
        return
    top = df.sort_values("dq_score", ascending=False).head(12).iloc[::-1]
    y = np.arange(len(top))
    plt.figure(figsize=(14, max(6, 0.56 * len(top))))
    plt.barh(y - 0.22, top["dq_score"], height=0.28, label="DQ score")
    plt.barh(y + 0.08, top["q20_dp_score"], height=0.28, label="q20 DP")
    plt.barh(y + 0.38, top["worst_dp_score"], height=0.28, label="worst DP")
    plt.yticks(y, top["case"])
    plt.xlabel("score")
    plt.title("26DQ confidence envelope: score vs lower-tail DP")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "phase26dq_envelope_scores.png", dpi=150)
    plt.close()


def main() -> Dict[str, Any]:
    t0 = time.time()
    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] Loaded previous phase: {PREV_PATH}")
    print(f"[{PHASE}] Output root: {ROOT}")

    z2, rel_ids, basins = EVAL_HELPER.build_basins()
    print(f"[{PHASE}] Built basins: {len(basins)}")

    priors = read_prior_candidates()
    candidates = local_lockdown_candidates(priors)
    print(f"[{PHASE}] Prior candidates: {len(priors)}")
    print(f"[{PHASE}] DQ candidates: {len(candidates)}")

    case_rows: List[Dict[str, Any]] = []
    env_rows: List[Dict[str, Any]] = []

    for i, (case, ov) in enumerate(candidates, 1):
        try:
            agg, rows = evaluate_candidate_envelope(case, ov, basins, seed=3000 + i)
            case_rows.append(agg)
            env_rows.extend(rows)
            if i <= 5 or i % 8 == 0:
                print(
                    f"[{PHASE}] {i:03d}/{len(candidates)} {case[:76]:76s} "
                    f"DQ={agg['dq_score']:.4f} q20={agg['q20_dp_score']:.4f} "
                    f"worstDP={agg['worst_dp_score']:.4f} worstCap={agg['worst_capture_rate']:.3f} "
                    f"worstTan={agg['worst_tangent_ratio']:.3f} survival={agg['survival_rate']:.2f}"
                )
        except Exception as e:
            print(f"[{PHASE}] ERROR {case}: {e}")
            case_rows.append({"case": case, "error": repr(e), "dq_score": float("nan"), **ov})

    df = pd.DataFrame(case_rows)
    env = pd.DataFrame(env_rows)
    df.to_csv(ROOT / "phase26dq_case_results.csv", index=False)
    env.to_csv(ROOT / "phase26dq_envelope_results.csv", index=False)

    df_ok = df[pd.to_numeric(df.get("dq_score", pd.Series(dtype=float)), errors="coerce").notna()].copy()
    df_ok = df_ok.sort_values("dq_score", ascending=False)
    pf = pareto_front(df_ok)
    pf.to_csv(ROOT / "phase26dq_pareto_front.csv", index=False)

    plot_top_scores(df_ok)
    plot_capture_tangent(df_ok, pf)
    plot_envelope_bars(df_ok)

    best = df_ok.iloc[0] if len(df_ok) else pd.Series(dtype=float)

    # Emit inherited/AP-style plots for the best DQ survivor where available.
    plot_errors: List[str] = []
    if len(best) and hasattr(PREV, "inherited_plots"):
        try:
            plot_errors = PREV.inherited_plots(z2, rel_ids, basins, best)
        except Exception as e:
            plot_errors.append(repr(e))
            print(f"[{PHASE}] inherited plot warning: {e}")
    elif len(best) and hasattr(PREV, "maybe_inherited_plots"):
        try:
            plot_errors = PREV.maybe_inherited_plots(z2, rel_ids, basins, best)
        except Exception as e:
            plot_errors.append(repr(e))
            print(f"[{PHASE}] inherited plot warning: {e}")

    summary = {
        "phase": PHASE,
        "title": TITLE,
        "prev_path": str(PREV_PATH),
        "root": str(ROOT),
        "num_priors": int(len(priors)),
        "num_candidates": int(len(candidates)),
        "num_success": int(len(df_ok)),
        "num_envelope_evals": int(len(env)),
        "num_pareto": int(len(pf)),
        "elapsed_sec": float(time.time() - t0),
        "best_case": None if not len(best) else str(best.get("case")),
        "best_dq_score": None if not len(best) else float(best.get("dq_score", float("nan"))),
        "best_q20_dp_score": None if not len(best) else float(best.get("q20_dp_score", float("nan"))),
        "best_worst_dp_score": None if not len(best) else float(best.get("worst_dp_score", float("nan"))),
        "best_worst_capture_rate": None if not len(best) else float(best.get("worst_capture_rate", float("nan"))),
        "best_worst_tangent_ratio": None if not len(best) else float(best.get("worst_tangent_ratio", float("nan"))),
        "best_survival_rate": None if not len(best) else float(best.get("survival_rate", float("nan"))),
        "best_overrides": None if not len(best) else {k: float(best[k]) for k in PARAMS if k in best and pd.notna(best[k])},
        "top8": df_ok.head(8).to_dict(orient="records") if len(df_ok) else [],
        "pareto_top8": pf.head(8).to_dict(orient="records") if len(pf) else [],
        "plot_errors": plot_errors,
        "outputs": [
            "phase26dq_case_results.csv",
            "phase26dq_envelope_results.csv",
            "phase26dq_pareto_front.csv",
            "phase26dq_summary.json",
            "phase26dq_top_scores.png",
            "phase26dq_capture_vs_tangent_pareto.png",
            "phase26dq_envelope_scores.png",
        ],
    }
    with open(ROOT / "phase26dq_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[{PHASE}] Done in {summary['elapsed_sec']:.1f}s")
    if len(best):
        print(
            f"[{PHASE}] BEST: {summary['best_case']} | "
            f"DQ={summary['best_dq_score']:.4f} q20DP={summary['best_q20_dp_score']:.4f} "
            f"worstDP={summary['best_worst_dp_score']:.4f} worstCap={summary['best_worst_capture_rate']:.3f} "
            f"worstTan={summary['best_worst_tangent_ratio']:.3f} survival={summary['best_survival_rate']:.2f}"
        )
        print(f"[{PHASE}] BEST overrides: {summary['best_overrides']}")
    print(f"[{PHASE}] Wrote summary: {ROOT / 'phase26dq_summary.json'}")
    return summary


if __name__ == "__main__":
    main()
