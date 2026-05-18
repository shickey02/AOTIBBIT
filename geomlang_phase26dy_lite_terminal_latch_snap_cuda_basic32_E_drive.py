# geomlang_phase26dy_lite_terminal_latch_snap_cuda_basic32_E_drive.py
# Phase 26DY-LITE — terminal latch / last-inch capture snap after DX.
#
# Why this exists:
#   26DW-LITE showed that the DV near-miss is not a simple "more capture" problem.
#   It exposed two competing regimes:
#       1) low-capture / clean-tangent cases, especially screen_small_00
#       2) high-capture / tangent-explosion cases, especially base and stress_cap_tk
#
#   DY-LITE therefore does not globally lift the bowl. It searches a narrow band and
#   scores candidates by named culprit gates:
#       - screen_small_00 capture must cross the capture target
#       - base tangent must remain controlled
#       - stress_cap_tk tangent must remain controlled
#       - stress_shell_blend and screen_medium_00 must not become hidden tails
#       - q20 DP must remain healthy
#
# Recommended first run:
#   python bbit_geomlang/geomlang_phase26dy_lite_terminal_latch_snap_cuda_basic32_E_drive.py --device cuda
#
# Fast smoke:
#   python bbit_geomlang/geomlang_phase26dy_lite_terminal_latch_snap_cuda_basic32_E_drive.py --device cuda --max-variants 8 --seed-count 2
#
# Wider controlled:
#   python bbit_geomlang/geomlang_phase26dy_lite_terminal_latch_snap_cuda_basic32_E_drive.py --device cuda --max-variants 24 --seed-count 3 --audit-repeats 1
#
# Suspicious audit:
#   python bbit_geomlang/geomlang_phase26dy_lite_terminal_latch_snap_cuda_basic32_E_drive.py --device cuda --max-variants 24 --seed-count 3 --audit-repeats 2 --sample-full-tail

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PHASE = "26DY-LITE"
TITLE = "terminal latch / last-inch capture snap after DX"
ROOT = Path(r"E:\BBIT\outputs_basic32")
HERE = Path(__file__).resolve().parent

DS_LITE_FILE = HERE / "geomlang_phase26ds_lite_strict_line_rescue_cuda_basic32_E_drive.py"
DS_FULL_FILE = HERE / "geomlang_phase26ds_strict_line_rescue_cuda_basic32_E_drive.py"
DR_FAST_FILE = HERE / "geomlang_phase26dr_fast_staged_cuda_basic32_E_drive.py"
DV_SUMMARY = ROOT / "phase26dv_lite_summary.json"
DV_VARIANTS = ROOT / "phase26dv_lite_variant_summary.csv"
DX_SUMMARY = ROOT / "phase26dx_lite_summary.json"
DX_RUNS = ROOT / "phase26dx_lite_culprit_gated_results.csv"
DX_CASES = ROOT / "phase26dx_lite_culprit_gated_cases.csv"

PARAMS = [
    "BOWL_RADIUS_FRAC",
    "BOWL_SEAT_AXIS_GAIN",
    "BOWL_DIRECTIONAL_BLEND",
    "BOWL_NORM_CAP_MULT",
    "BOWL_SHELL_RADIAL_GAIN",
    "BOWL_TANGENT_KILL",
]

CRITICAL_LABELS = [
    "base",
    "screen_small_00",
    "screen_medium_00",
    "stress_cap_tk",
    "stress_shell_blend",
    "stress_radius_in",
    "stress_seat_down",
]

# Narrow DY clamp: avoid the old capture-heavy / tangent-chaotic corner.
BOUNDS = {
    # DY keeps the basin close to DX's best candidate, but allows a slightly higher
    # tangent-kill ceiling and a slightly lower shell/radius floor to test the
    # "last-inch latch without waking the side-slip tail" hypothesis.
    "BOWL_RADIUS_FRAC": (0.695, 0.742),
    "BOWL_SEAT_AXIS_GAIN": (0.385, 0.445),
    "BOWL_DIRECTIONAL_BLEND": (0.955, 1.030),
    "BOWL_NORM_CAP_MULT": (1.075, 1.195),
    "BOWL_SHELL_RADIAL_GAIN": (0.128, 0.174),
    "BOWL_TANGENT_KILL": (0.940, 1.085),
}

# DX/DW/DV anchors.
FALLBACK_DV_BEST = {
    "BOWL_RADIUS_FRAC": 0.72,
    "BOWL_SEAT_AXIS_GAIN": 0.42,
    "BOWL_DIRECTIONAL_BLEND": 1.02,
    "BOWL_NORM_CAP_MULT": 1.16,
    "BOWL_SHELL_RADIAL_GAIN": 0.165,
    "BOWL_TANGENT_KILL": 1.00,
}

# DW showed this lifted variant had good base/stress behavior but some medium/small stress tails.
FALLBACK_DW_BALANCED = {
    "BOWL_RADIUS_FRAC": 0.6997443058798729,
    "BOWL_SEAT_AXIS_GAIN": 0.42,
    "BOWL_DIRECTIONAL_BLEND": 0.9711619081247089,
    "BOWL_NORM_CAP_MULT": 1.10,
    "BOWL_SHELL_RADIAL_GAIN": 0.14705073683834083,
    "BOWL_TANGENT_KILL": 0.94,
}

# DW low-capture clean case. This is an envelope-local override, but it is useful as a candidate seed.
FALLBACK_LOW_CAPTURE_CLEAN = {
    "BOWL_RADIUS_FRAC": 0.7291,
    "BOWL_SEAT_AXIS_GAIN": 0.4031,
    "BOWL_DIRECTIONAL_BLEND": 0.9981,
    "BOWL_NORM_CAP_MULT": 1.1654,
    "BOWL_SHELL_RADIAL_GAIN": 0.1648,
    "BOWL_TANGENT_KILL": 0.9718,
}

# DX best compromise from the run immediately before DY. It was one capture quantum
# short but had useful named-guard behavior, so DY centers here instead of the older
# DV/DU anchors.
FALLBACK_DX_BEST = {
    "BOWL_RADIUS_FRAC": 0.7123398273572922,
    "BOWL_SEAT_AXIS_GAIN": 0.435,
    "BOWL_DIRECTIONAL_BLEND": 1.025,
    "BOWL_NORM_CAP_MULT": 1.123734619299985,
    "BOWL_SHELL_RADIAL_GAIN": 0.172,
    "BOWL_TANGENT_KILL": 0.9811105492968176,
}


def import_by_path(path: Path, module_name: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing dependency: {path}")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def fnum(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or pd.isna(x):
            return default
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def clamp_param(k: str, v: float) -> float:
    lo, hi = BOUNDS[k]
    return float(min(hi, max(lo, v)))


def clamp_ov(ov: Dict[str, Any], fallback: Dict[str, float] | None = None) -> Dict[str, float]:
    fb = fallback or FALLBACK_DW_BALANCED
    return {k: clamp_param(k, fnum(ov.get(k), fb[k])) for k in PARAMS}


def row_key(ov: Dict[str, Any], ndigits: int = 6) -> Tuple[float, ...]:
    cov = clamp_ov(ov)
    return tuple(round(float(cov[k]), ndigits) for k in PARAMS)


def add_unique(out: List[Tuple[str, Dict[str, float], str]], seen: set, name: str, ov: Dict[str, Any], source: str) -> None:
    cov = clamp_ov(ov)
    key = row_key(cov)
    if key not in seen:
        out.append((name[:118], cov, source))
        seen.add(key)


def parse_json_cell(x: Any) -> Dict[str, Any] | None:
    if not isinstance(x, str) or not x.strip():
        return None
    try:
        obj = json.loads(x)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def parse_overrides_from_row(row: pd.Series) -> Dict[str, float] | None:
    if all(k in row.index and pd.notna(row[k]) for k in PARAMS):
        return clamp_ov({k: row[k] for k in PARAMS})
    for col in ("overrides", "overrides_json", "best_overrides"):
        if col in row.index:
            obj = parse_json_cell(row[col])
            if obj and all(k in obj for k in PARAMS):
                return clamp_ov(obj)
    return None


def load_json_overrides(path: Path, key: str = "best_overrides") -> Tuple[str, Dict[str, float], str] | None:
    if not path.exists():
        return None
    try:
        js = json.loads(path.read_text(encoding="utf-8"))
        ov = js.get(key)
        name = str(js.get("best_variant") or path.stem)
        if isinstance(ov, dict) and all(k in ov for k in PARAMS):
            return name, clamp_ov(ov), str(path)
    except Exception as e:
        print(f"[{PHASE}] WARNING: could not read {path.name}: {e!r}")
    return None


def load_dw_guided_candidates(limit_each: int = 10) -> List[Tuple[str, Dict[str, float], str]]:
    out: List[Tuple[str, Dict[str, float], str]] = []
    seen: set = set()

    add_unique(out, seen, "dy_anchor_dw_balanced_lift", FALLBACK_DW_BALANCED, "fallback_dw_balanced")
    add_unique(out, seen, "dy_anchor_dv_best_grid", FALLBACK_DV_BEST, "fallback_dv_best")
    add_unique(out, seen, "dy_anchor_low_capture_clean", FALLBACK_LOW_CAPTURE_CLEAN, "fallback_low_capture_clean")

    for p in (DX_SUMMARY, DV_SUMMARY):
        loaded = load_json_overrides(p)
        if loaded:
            name, ov, src = loaded
            add_unique(out, seen, f"prior_{name}", ov, src)

    # DW run summaries: prefer candidates with base/stress tangent under control and capture near target.
    if DX_RUNS.exists():
        try:
            df = pd.read_csv(DX_RUNS)
            if not df.empty:
                rank = (
                    20.0 * df.get("worst_capture_rate", 0.0).astype(float)
                    + 2.0 * df.get("q20_dp_score", 0.0).astype(float)
                    - 3.0 * np.maximum(0.0, df.get("worst_tangent_ratio", 999.0).astype(float) - 2.30)
                    - 1.6 * np.maximum(0.0, df.get("base_tangent_ratio", 999.0).astype(float) - 2.30)
                )
                df = df.assign(dy_rank=rank).sort_values("dy_rank", ascending=False)
                for _, row in df.head(limit_each).iterrows():
                    ov = parse_overrides_from_row(row)
                    if ov is not None:
                        add_unique(out, seen, f"prior_dw_run_{row.get('audit_variant', 'row')}", ov, str(DX_RUNS))
        except Exception as e:
            print(f"[{PHASE}] WARNING: could not read DW runs: {e!r}")

    # DW case-level rows: exploit clean low-capture local overrides and avoid tangent-chaotic ones.
    if DX_CASES.exists():
        try:
            df = pd.read_csv(DX_CASES)
            if not df.empty:
                safe_low_capture = df[
                    (df.get("capture_rate", 1.0).astype(float) < 0.285)
                    & (df.get("tangent_ratio", 999.0).astype(float) <= 2.10)
                    & (df.get("dp_score", -999.0).astype(float) >= 2.25)
                ].copy()
                if not safe_low_capture.empty:
                    safe_low_capture["dy_case_rank"] = (
                        20.0 * safe_low_capture["capture_rate"].astype(float)
                        + 2.0 * safe_low_capture["dp_score"].astype(float)
                        - safe_low_capture["tangent_ratio"].astype(float)
                    )
                    for _, row in safe_low_capture.sort_values("dy_case_rank", ascending=False).head(limit_each).iterrows():
                        ov = parse_overrides_from_row(row)
                        if ov is not None:
                            add_unique(out, seen, f"prior_clean_lowcap_{row.get('envelope_label', 'case')}", ov, str(DX_CASES))

                clean_pass = df[
                    (df.get("capture_rate", 0.0).astype(float) >= 0.285)
                    & (df.get("tangent_ratio", 999.0).astype(float) <= 2.10)
                    & (df.get("dp_score", -999.0).astype(float) >= 2.25)
                ].copy()
                if not clean_pass.empty:
                    clean_pass["dy_case_rank"] = (
                        10.0 * clean_pass["capture_rate"].astype(float)
                        + clean_pass["dp_score"].astype(float)
                        - 0.5 * clean_pass["tangent_ratio"].astype(float)
                    )
                    for _, row in clean_pass.sort_values("dy_case_rank", ascending=False).head(limit_each).iterrows():
                        ov = parse_overrides_from_row(row)
                        if ov is not None:
                            add_unique(out, seen, f"prior_clean_pass_{row.get('envelope_label', 'case')}", ov, str(DX_CASES))
        except Exception as e:
            print(f"[{PHASE}] WARNING: could not read DW cases: {e!r}")

    # DV variant summary: keep the best near-capture family and the low-tangent alternatives.
    if DV_VARIANTS.exists():
        try:
            df = pd.read_csv(DV_VARIANTS)
            if not df.empty:
                rank = (
                    2.0 * df.get("near_capture_rate", 0.0).astype(float)
                    + 20.0 * df.get("min_worst_capture_rate", 0.0).astype(float)
                    + 0.4 * df.get("dv_family_score", 0.0).astype(float)
                    - 2.0 * np.maximum(0.0, df.get("max_worst_tangent_ratio", 999.0).astype(float) - 2.30)
                )
                df = df.assign(dy_dv_rank=rank).sort_values("dy_dv_rank", ascending=False)
                for _, row in df.head(limit_each).iterrows():
                    ov = parse_overrides_from_row(row)
                    if ov is not None:
                        add_unique(out, seen, f"prior_dv_variant_{row.get('variant', 'row')}", ov, str(DV_VARIANTS))
        except Exception as e:
            print(f"[{PHASE}] WARNING: could not read DV variants: {e!r}")

    return out


def set_ov(base: Dict[str, float], **kwargs: float) -> Dict[str, float]:
    ov = dict(base)
    ov.update(kwargs)
    return clamp_ov(ov)


def generate_dy_variants(seed: int, max_variants: int) -> List[Tuple[str, Dict[str, float], str]]:
    """
    DY is a late-stage, last-inch search. The underlying DR/DS evaluator only accepts
    the existing BOWL_* override surface, so this script implements the terminal-latch
    idea as a family of *micro-proxy* mutations:

      A) close-radius snap: slightly expand the radius/cap while keeping seat modest
      B) low-tangent snap: raise tangent kill and reduce shell/blend a hair
      C) DP-positive snap: preserve the high-DP basin and only nudge cap
      D) screen-small snap: target the one missing capture quantum with radius/cap
      E) tangent-governed cap boost: pair every cap lift with tangent-kill lift

    If later DR exposes true terminal-step latch hooks, this is the phase to wire them
    into. For now, the goal is to see whether a normal-parameter proxy can flip the
    0.28125 -> 0.296875 capture quantum without reviving the 3-5x tangent tail.
    """
    rng = random.Random(seed)
    out: List[Tuple[str, Dict[str, float], str]] = []
    seen: set = set()

    priors = load_dw_guided_candidates(limit_each=10)
    # Promote DX best first, then the best prior rows found from DX/DW/DV CSVs.
    add_unique(out, seen, "dy_anchor_dx_best_last_inch", FALLBACK_DX_BEST, "fallback_dx_best")
    add_unique(out, seen, "dy_anchor_dv_safe_grid", FALLBACK_DV_BEST, "fallback_dv_best")
    add_unique(out, seen, "dy_anchor_low_capture_clean", FALLBACK_LOW_CAPTURE_CLEAN, "fallback_low_capture_clean")
    add_unique(out, seen, "dy_anchor_dw_balanced", FALLBACK_DW_BALANCED, "fallback_dw_balanced")
    for name, ov, src in priors[:8]:
        add_unique(out, seen, f"prior_{name}", ov, src)

    centers = [FALLBACK_DX_BEST, FALLBACK_LOW_CAPTURE_CLEAN, FALLBACK_DW_BALANCED]

    # A. close-radius snap only: radius/cap up, seat not raised, shell trimmed if needed.
    for ci, center in enumerate(centers):
        for r in [0.716, 0.724, 0.732, 0.740]:
            for cap in [1.125, 1.145, 1.165, 1.185]:
                ov = set_ov(center,
                    BOWL_RADIUS_FRAC=r,
                    BOWL_SEAT_AXIS_GAIN=min(center["BOWL_SEAT_AXIS_GAIN"], 0.415),
                    BOWL_DIRECTIONAL_BLEND=min(center["BOWL_DIRECTIONAL_BLEND"], 1.005),
                    BOWL_NORM_CAP_MULT=cap,
                    BOWL_SHELL_RADIAL_GAIN=min(center["BOWL_SHELL_RADIAL_GAIN"], 0.162),
                    BOWL_TANGENT_KILL=max(center["BOWL_TANGENT_KILL"], 1.000),
                )
                add_unique(out, seen, f"dy_close_radius_c{ci}_r{r:.3f}_cap{cap:.3f}", ov, "A_close_radius_snap")

    # B. low-tangent snap: sacrifice a little shell/blend and spend the budget on tangent kill.
    for ci, center in enumerate(centers):
        for tk in [1.025, 1.045, 1.065, 1.080]:
            for sh in [0.136, 0.145, 0.154, 0.162]:
                ov = set_ov(center,
                    BOWL_RADIUS_FRAC=0.724,
                    BOWL_SEAT_AXIS_GAIN=0.405,
                    BOWL_DIRECTIONAL_BLEND=0.982,
                    BOWL_NORM_CAP_MULT=1.155,
                    BOWL_SHELL_RADIAL_GAIN=sh,
                    BOWL_TANGENT_KILL=tk,
                )
                add_unique(out, seen, f"dy_low_tangent_c{ci}_sh{sh:.3f}_tk{tk:.3f}", ov, "B_low_tangent_snap")

    # C. DP-positive snap: preserve DU/DV's high q20 style; only tiny cap/radius nudges.
    for cap in [1.110, 1.125, 1.140, 1.155, 1.170]:
        for tk in [0.990, 1.010, 1.030, 1.050]:
            ov = set_ov(FALLBACK_DW_BALANCED,
                BOWL_RADIUS_FRAC=0.710,
                BOWL_SEAT_AXIS_GAIN=0.405,
                BOWL_DIRECTIONAL_BLEND=0.975,
                BOWL_NORM_CAP_MULT=cap,
                BOWL_SHELL_RADIAL_GAIN=0.148,
                BOWL_TANGENT_KILL=tk,
            )
            add_unique(out, seen, f"dy_dp_positive_cap{cap:.3f}_tk{tk:.3f}", ov, "C_dp_positive_snap")

    # D. screen-small-specific proxy: the closest thing available to a conditional latch.
    # Keep it narrow: the prior culprit was screen_small_00 at cap≈0.28125 with clean tangent.
    for r in [0.722, 0.728, 0.734, 0.740]:
        for cap in [1.150, 1.165, 1.180, 1.195]:
            for tk in [1.015, 1.040, 1.065]:
                ov = set_ov(FALLBACK_LOW_CAPTURE_CLEAN,
                    BOWL_RADIUS_FRAC=r,
                    BOWL_SEAT_AXIS_GAIN=0.398,
                    BOWL_DIRECTIONAL_BLEND=0.990,
                    BOWL_NORM_CAP_MULT=cap,
                    BOWL_SHELL_RADIAL_GAIN=0.158,
                    BOWL_TANGENT_KILL=tk,
                )
                add_unique(out, seen, f"dy_screen_small_r{r:.3f}_cap{cap:.3f}_tk{tk:.3f}", ov, "D_screen_small_snap")

    # E. tangent-governed cap boost: every cap lift must buy tangent kill and lower shell.
    for cap in [1.13, 1.15, 1.17, 1.19]:
        for tk in [1.00, 1.025, 1.05, 1.075]:
            sh = 0.166 - 0.18 * max(0.0, cap - 1.13)  # cap up => shell down
            ov = set_ov(FALLBACK_DX_BEST,
                BOWL_RADIUS_FRAC=0.718 + 0.18 * max(0.0, cap - 1.13),
                BOWL_SEAT_AXIS_GAIN=0.402,
                BOWL_DIRECTIONAL_BLEND=0.992,
                BOWL_NORM_CAP_MULT=cap,
                BOWL_SHELL_RADIAL_GAIN=sh,
                BOWL_TANGENT_KILL=tk,
            )
            add_unique(out, seen, f"dy_tangent_governed_cap{cap:.2f}_tk{tk:.3f}", ov, "E_tangent_governed_cap")

    # Small random cloud around the best compromise. Reject the old dangerous corner.
    for i in range(max(32, max_variants * 3)):
        ov = {
            "BOWL_RADIUS_FRAC": rng.gauss(0.724, 0.009),
            "BOWL_SEAT_AXIS_GAIN": rng.gauss(0.405, 0.010),
            "BOWL_DIRECTIONAL_BLEND": rng.gauss(0.992, 0.012),
            "BOWL_NORM_CAP_MULT": rng.gauss(1.160, 0.018),
            "BOWL_SHELL_RADIAL_GAIN": rng.gauss(0.154, 0.008),
            "BOWL_TANGENT_KILL": rng.gauss(1.040, 0.025),
        }
        cov = clamp_ov(ov)
        if cov["BOWL_SEAT_AXIS_GAIN"] > 0.425 and cov["BOWL_SHELL_RADIAL_GAIN"] > 0.162:
            continue
        if cov["BOWL_DIRECTIONAL_BLEND"] > 1.015 and cov["BOWL_SHELL_RADIAL_GAIN"] > 0.162:
            continue
        add_unique(out, seen, f"dy_terminal_cloud_{i:03d}", cov, "dy_terminal_cloud")

    if len(out) > max_variants:
        # Keep anchors and line-search families, then evenly sample the tail.
        anchors = out[: min(7, len(out))]
        rest = out[len(anchors):]
        keep = max_variants - len(anchors)
        if keep > 0 and rest:
            idxs = np.linspace(0, len(rest) - 1, keep).round().astype(int).tolist()
            out = anchors + [rest[i] for i in idxs]
        else:
            out = anchors
    return out[:max_variants]


def pass_flags(row: Dict[str, Any], target_cap: float, target_tan: float, target_q20: float, relaxed_tan: float) -> Dict[str, bool]:
    cap = fnum(row.get("worst_capture_rate"), 0.0)
    tan = fnum(row.get("worst_tangent_ratio"), 999.0)
    q20 = fnum(row.get("q20_dp_score"), -999.0)
    return {
        "pass_cap": cap >= target_cap,
        "pass_tan_strict": tan <= target_tan,
        "pass_tan_relaxed": tan <= relaxed_tan,
        "pass_q20": q20 >= target_q20,
        "pass_strict_line": cap >= target_cap and tan <= target_tan and q20 >= target_q20,
        "pass_relaxed_line": cap >= target_cap and tan <= relaxed_tan and q20 >= target_q20,
    }


def add_case_diagnostics(env_df: pd.DataFrame, target_cap: float, target_tan: float, relaxed_tan: float, target_q20: float) -> pd.DataFrame:
    if env_df.empty:
        return env_df
    df = env_df.copy()
    df["capture_deficit"] = np.maximum(0.0, target_cap - df["capture_rate"].astype(float))
    df["tangent_excess_strict"] = np.maximum(0.0, df["tangent_ratio"].astype(float) - target_tan)
    df["tangent_excess_relaxed"] = np.maximum(0.0, df["tangent_ratio"].astype(float) - relaxed_tan)
    df["dp_deficit"] = np.maximum(0.0, target_q20 - df["dp_score"].astype(float))
    df["case_pass_capture"] = df["capture_rate"].astype(float) >= target_cap
    df["case_pass_tangent_strict"] = df["tangent_ratio"].astype(float) <= target_tan
    df["case_pass_tangent_relaxed"] = df["tangent_ratio"].astype(float) <= relaxed_tan
    df["case_pass_dp"] = df["dp_score"].astype(float) >= target_q20
    df["case_pass_strict"] = df["case_pass_capture"] & df["case_pass_tangent_strict"] & df["case_pass_dp"]
    df["case_pass_relaxed"] = df["case_pass_capture"] & df["case_pass_tangent_relaxed"] & df["case_pass_dp"]
    return df


def label_metric(case_df: pd.DataFrame, parent_case: str, label: str, col: str, agg: str, default: float) -> float:
    if case_df.empty or "parent_case" not in case_df or "envelope_label" not in case_df:
        return default
    g = case_df[(case_df["parent_case"] == parent_case) & (case_df["envelope_label"] == label)]
    if g.empty or col not in g:
        return default
    vals = g[col].astype(float)
    if agg == "min":
        return float(vals.min())
    if agg == "max":
        return float(vals.max())
    return float(vals.mean())


def add_culprit_gates(results_df: pd.DataFrame, case_df: pd.DataFrame, target_cap: float, target_tan: float, relaxed_tan: float, target_q20: float) -> pd.DataFrame:
    if results_df.empty:
        return results_df
    rows = []
    for _, row in results_df.iterrows():
        d = row.to_dict()
        parent = str(d.get("case"))
        # Label-specific gates. Defaults are harsh so missing labels are not silently promoted.
        screen_small_cap = label_metric(case_df, parent, "screen_small_00", "capture_rate", "min", 0.0)
        screen_small_tan = label_metric(case_df, parent, "screen_small_00", "tangent_ratio", "max", 999.0)
        screen_medium_cap = label_metric(case_df, parent, "screen_medium_00", "capture_rate", "min", 0.0)
        screen_medium_tan = label_metric(case_df, parent, "screen_medium_00", "tangent_ratio", "max", 999.0)
        base_tan = label_metric(case_df, parent, "base", "tangent_ratio", "max", 999.0)
        base_dp = label_metric(case_df, parent, "base", "dp_score", "min", -999.0)
        stress_cap_tk_tan = label_metric(case_df, parent, "stress_cap_tk", "tangent_ratio", "max", 999.0)
        stress_shell_tan = label_metric(case_df, parent, "stress_shell_blend", "tangent_ratio", "max", 999.0)
        stress_radius_tan = label_metric(case_df, parent, "stress_radius_in", "tangent_ratio", "max", 999.0)
        stress_seat_tan = label_metric(case_df, parent, "stress_seat_down", "tangent_ratio", "max", 999.0)
        label_min_cap = min(screen_small_cap, screen_medium_cap)
        label_max_guard_tan = max(base_tan, stress_cap_tk_tan, stress_shell_tan, screen_medium_tan, stress_radius_tan, stress_seat_tan)

        d.update({
            "screen_small_00_min_capture": screen_small_cap,
            "screen_small_00_max_tangent": screen_small_tan,
            "screen_medium_00_min_capture": screen_medium_cap,
            "screen_medium_00_max_tangent": screen_medium_tan,
            "base_max_tangent": base_tan,
            "base_min_dp": base_dp,
            "stress_cap_tk_max_tangent": stress_cap_tk_tan,
            "stress_shell_blend_max_tangent": stress_shell_tan,
            "stress_radius_in_max_tangent": stress_radius_tan,
            "stress_seat_down_max_tangent": stress_seat_tan,
            "label_min_screen_capture": label_min_cap,
            "label_max_guard_tangent": label_max_guard_tan,
        })
        d["gate_screen_small_capture"] = screen_small_cap >= target_cap
        d["gate_screen_medium_capture"] = screen_medium_cap >= target_cap
        d["gate_base_tangent_strict"] = base_tan <= target_tan
        d["gate_base_tangent_relaxed"] = base_tan <= relaxed_tan
        d["gate_stress_cap_tk_tangent_strict"] = stress_cap_tk_tan <= target_tan
        d["gate_stress_cap_tk_tangent_relaxed"] = stress_cap_tk_tan <= relaxed_tan
        d["gate_stress_shell_tangent_relaxed"] = stress_shell_tan <= relaxed_tan
        d["gate_all_named_strict"] = (
            d["gate_screen_small_capture"]
            and d["gate_screen_medium_capture"]
            and base_tan <= target_tan
            and stress_cap_tk_tan <= target_tan
            and stress_shell_tan <= target_tan
            and screen_medium_tan <= target_tan
            and fnum(d.get("q20_dp_score"), -999.0) >= target_q20
        )
        d["gate_all_named_relaxed"] = (
            d["gate_screen_small_capture"]
            and d["gate_screen_medium_capture"]
            and base_tan <= relaxed_tan
            and stress_cap_tk_tan <= relaxed_tan
            and stress_shell_tan <= relaxed_tan
            and screen_medium_tan <= relaxed_tan
            and fnum(d.get("q20_dp_score"), -999.0) >= target_q20
        )
        # DY score directly encodes the DW diagnosis.
        global_score = fnum(d.get("base_rescue_score"), 0.0)
        cap_floor = min(fnum(d.get("worst_capture_rate"), 0.0), label_min_cap)
        q20 = fnum(d.get("q20_dp_score"), -999.0)
        dy_score = (
            global_score
            + 90.0 * (cap_floor - target_cap)
            + 35.0 * max(0.0, screen_small_cap - target_cap)
            + 8.0 * max(0.0, q20 - target_q20)
            - 4.0 * max(0.0, label_max_guard_tan - relaxed_tan)
            - 3.0 * max(0.0, label_max_guard_tan - target_tan)
            - 3.0 * max(0.0, base_tan - relaxed_tan)
            - 4.5 * max(0.0, stress_cap_tk_tan - relaxed_tan)
            - 2.0 * max(0.0, stress_shell_tan - relaxed_tan)
            + 4.0 * float(d["gate_all_named_relaxed"])
            + 6.0 * float(d["gate_all_named_strict"])
        )
        d["dy_terminal_latch_score_run"] = float(dy_score)
        rows.append(d)
    return pd.DataFrame(rows)


def summarize(results_df: pd.DataFrame, target_cap: float, target_tan: float, relaxed_tan: float, target_q20: float) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()
    rows = []
    for variant, g in results_df.groupby("variant", sort=False):
        strict_rate = float(g["pass_strict_line"].mean())
        relaxed_rate = float(g["pass_relaxed_line"].mean())
        gate_strict_rate = float(g["gate_all_named_strict"].mean())
        gate_relaxed_rate = float(g["gate_all_named_relaxed"].mean())
        min_cap = float(g["worst_capture_rate"].min())
        min_label_cap = float(g["label_min_screen_capture"].min())
        max_tan = float(g["worst_tangent_ratio"].max())
        max_guard_tan = float(g["label_max_guard_tangent"].max())
        min_q20 = float(g["q20_dp_score"].min())
        mean_score = float(g["dy_terminal_latch_score_run"].mean())
        tan_std = float(g["worst_tangent_ratio"].std(ddof=0)) if len(g) > 1 else 0.0
        cap_std = float(g["worst_capture_rate"].std(ddof=0)) if len(g) > 1 else 0.0
        family_score = (
            8.0 * gate_relaxed_rate
            + 10.0 * gate_strict_rate
            + 5.0 * relaxed_rate
            + 6.0 * strict_rate
            + 0.45 * mean_score
            + 110.0 * (min(min_cap, min_label_cap) - target_cap)
            + 2.0 * max(0.0, min_q20 - target_q20)
            - 3.0 * max(0.0, max_guard_tan - relaxed_tan)
            - 2.0 * max(0.0, max_guard_tan - target_tan)
            - 0.5 * tan_std
            - 18.0 * cap_std
        )
        rows.append({
            "variant": variant,
            "n": int(len(g)),
            "strict_pass_rate": strict_rate,
            "relaxed_pass_rate": relaxed_rate,
            "named_gate_strict_rate": gate_strict_rate,
            "named_gate_relaxed_rate": gate_relaxed_rate,
            "stable_strict_all_seeds": bool(g["pass_strict_line"].all()),
            "stable_relaxed_all_seeds": bool(g["pass_relaxed_line"].all()),
            "stable_named_gate_strict_all_seeds": bool(g["gate_all_named_strict"].all()),
            "stable_named_gate_relaxed_all_seeds": bool(g["gate_all_named_relaxed"].all()),
            "dy_family_score": float(family_score),
            "mean_dy_run_score": mean_score,
            "mean_dr_score": float(g["dr_score"].mean()),
            "min_q20_dp_score": min_q20,
            "min_worst_dp_score": float(g["worst_dp_score"].min()),
            "min_worst_capture_rate": min_cap,
            "min_label_screen_capture": min_label_cap,
            "max_worst_tangent_ratio": max_tan,
            "max_label_guard_tangent": max_guard_tan,
            "min_screen_small_00_capture": float(g["screen_small_00_min_capture"].min()),
            "max_base_tangent": float(g["base_max_tangent"].max()),
            "max_stress_cap_tk_tangent": float(g["stress_cap_tk_max_tangent"].max()),
            "max_stress_shell_blend_tangent": float(g["stress_shell_blend_max_tangent"].max()),
            "std_worst_tangent_ratio": tan_std,
            "std_worst_capture_rate": cap_std,
            **{k: float(g[k].iloc[0]) for k in PARAMS if k in g.columns},
            "variant_source": str(g.get("variant_source", pd.Series([""])).iloc[0]) if "variant_source" in g else "",
        })
    return pd.DataFrame(rows).sort_values(
        ["stable_named_gate_strict_all_seeds", "stable_named_gate_relaxed_all_seeds", "named_gate_relaxed_rate", "relaxed_pass_rate", "dy_family_score"],
        ascending=[False, False, False, False, False],
    )


def plot_family_scores(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    top = df.sort_values("dy_family_score", ascending=False).head(25).iloc[::-1]
    plt.figure(figsize=(14, max(6, 0.46 * len(top))))
    plt.barh(top["variant"], top["dy_family_score"])
    plt.xlabel("DY terminal-latch family score")
    plt.title("26DY-LITE terminal-latch family scores")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_pass_rates(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    top = df.sort_values(["named_gate_relaxed_rate", "relaxed_pass_rate", "dy_family_score"], ascending=[False, False, False]).head(25).iloc[::-1]
    y = np.arange(len(top))
    plt.figure(figsize=(14, max(6, 0.46 * len(top))))
    plt.barh(y - 0.27, top["named_gate_relaxed_rate"], height=0.22, label="named relaxed gate")
    plt.barh(y - 0.04, top["named_gate_strict_rate"], height=0.22, label="named strict gate")
    plt.barh(y + 0.19, top["relaxed_pass_rate"], height=0.22, label="global relaxed")
    plt.yticks(y, top["variant"])
    plt.xlim(0, 1.05)
    plt.xlabel("pass rate across seeds")
    plt.title("26DY-LITE pass/gate rates")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_capture_tangent(df: pd.DataFrame, out_path: Path, target_cap: float, target_tan: float, relaxed_tan: float) -> None:
    if df.empty:
        return
    plt.figure(figsize=(10, 7))
    sc = plt.scatter(df["label_min_screen_capture"], df["label_max_guard_tangent"], c=df["dy_terminal_latch_score_run"], s=76, alpha=0.80)
    plt.axvline(target_cap, linewidth=1.0, alpha=0.45, label="capture target")
    plt.axhline(target_tan, linewidth=1.0, alpha=0.45, label="strict tangent")
    plt.axhline(relaxed_tan, linewidth=1.0, alpha=0.25, linestyle="--", label="relaxed tangent")
    plt.colorbar(sc, label="DY gated run score")
    plt.xlabel("min named screen capture")
    plt.ylabel("max named guard tangent")
    plt.title("26DY-LITE named gates: capture vs tangent")
    plt.grid(True, alpha=0.28)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_label_culprits(case_df: pd.DataFrame, out_path: Path, target_cap: float, target_tan: float) -> None:
    if case_df.empty:
        return
    g = case_df.groupby("envelope_label", sort=False).agg(
        min_capture=("capture_rate", "min"),
        max_tangent=("tangent_ratio", "max"),
        mean_capture_deficit=("capture_deficit", "mean"),
        mean_tangent_excess=("tangent_excess_strict", "mean"),
        n=("capture_rate", "size"),
    ).reset_index()
    g["label_culprit_score"] = 100.0 * g["mean_capture_deficit"] + 3.0 * g["mean_tangent_excess"]
    g = g.sort_values("label_culprit_score", ascending=True)
    plt.figure(figsize=(12, max(5, 0.42 * len(g))))
    plt.barh(g["envelope_label"], g["label_culprit_score"])
    plt.xlabel("mean label culprit score")
    plt.title("26DY-LITE remaining culprit labels")
    for i, r in enumerate(g.itertuples(index=False)):
        plt.text(float(r.label_culprit_score), i, f" cap={r.min_capture:.3f} tan={r.max_tangent:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def run(args: argparse.Namespace) -> Dict[str, Any]:
    t0 = time.time()
    ROOT.mkdir(parents=True, exist_ok=True)

    ds_lite = import_by_path(DS_LITE_FILE, "phase26ds_lite")
    ds_full = import_by_path(DS_FULL_FILE, "phase26ds_full")
    drfast = import_by_path(DR_FAST_FILE, "phase26dr_fast")

    torch_info = drfast.setup_torch(args.device)
    dr = drfast.import_phase26dr()

    target_tan = float(getattr(dr, "TARGET_WORST_TAN", getattr(ds_full, "DEFAULT_TARGET_WORST_TAN", 2.10)))
    target_cap = float(getattr(dr, "TARGET_WORST_CAP", getattr(ds_full, "DEFAULT_TARGET_WORST_CAP", 0.285)))
    target_q20 = float(getattr(dr, "TARGET_Q20_DP", getattr(ds_full, "DEFAULT_TARGET_Q20_DP", 2.25)))
    relaxed_tan = float(args.relaxed_tan)

    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] device={args.device} torch={torch_info}")
    print(f"[{PHASE}] targets: strictTan<={target_tan:.2f}, relaxedTan<={relaxed_tan:.2f}, worstCap>={target_cap:.3f}, q20DP>={target_q20:.2f}")
    print(f"[{PHASE}] max_variants={args.max_variants}, seed_count={args.seed_count}, audit_repeats={args.audit_repeats}, sampled_full_tail={args.sample_full_tail}")
    print(f"[{PHASE}] gates: screen_small_00 capture, screen_medium_00 capture, base tangent, stress_cap_tk tangent, stress_shell_blend tangent")

    eval_helper = getattr(dr, "EVAL_HELPER", None)
    if eval_helper is None or not hasattr(eval_helper, "build_basins"):
        eval_helper = dr.previous_eval_helper()
    z2, rel_ids, basins = eval_helper.build_basins()
    print(f"[{PHASE}] Built basins: {len(basins)}")

    variants = generate_dy_variants(args.seed, args.max_variants)
    print(f"[{PHASE}] DY variants: {len(variants)}")
    for name, ov, src in variants[: min(8, len(variants))]:
        print(f"[{PHASE}]   candidate {name} source={src} overrides={ov}")

    cache: Dict[Tuple[str, Tuple[float, ...]], Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    env_rows: List[Dict[str, Any]] = []

    total = len(variants) * args.seed_count
    run_i = 0
    for vi, (variant, ov, src) in enumerate(variants, 1):
        for si in range(args.seed_count):
            run_i += 1
            seed = args.seed + 290000 + vi * 1009 + si * 7919
            case = f"{variant}_seed{si+1:02d}"
            env = ds_lite.micro_envelope(
                drfast,
                dr,
                ov,
                seed=seed,
                repeats=args.audit_repeats,
                use_full_tail=args.sample_full_tail,
            )
            agg, erows = drfast.evaluate_envelope(
                dr,
                case,
                ov,
                basins,
                seed=seed,
                envelope=env,
                cache=cache,
                phase_label="dy_terminal_latch",
            )
            agg["variant"] = variant
            agg["variant_source"] = src
            agg["seed_index"] = si + 1
            agg["dy_seed"] = seed
            base_score = ds_full.rescue_score(agg, target_cap, target_tan, target_q20)
            agg["base_rescue_score"] = base_score
            agg.update(pass_flags(agg, target_cap, target_tan, target_q20, relaxed_tan))
            for k, v in ov.items():
                agg[k] = v
            rows.append(agg)

            for er in erows:
                er = dict(er)
                er["variant"] = variant
                er["variant_source"] = src
                er["seed_index"] = si + 1
                er["dy_seed"] = seed
                env_rows.append(er)

            if run_i <= 5 or run_i % args.print_every == 0 or si == args.seed_count - 1:
                print(
                    f"[{PHASE}] run {run_i:03d}/{total:03d} {variant[:58]:58s} s{si+1:02d} "
                    f"GLOB_RELAX={int(agg['pass_relaxed_line'])} GLOB_STRICT={int(agg['pass_strict_line'])} "
                    f"DR={agg['dr_score']:.3f} q20={agg['q20_dp_score']:.3f} "
                    f"cap={agg['worst_capture_rate']:.3f} tan={agg['worst_tangent_ratio']:.3f}"
                )

    raw_results_df = pd.DataFrame(rows)
    raw_env_df = pd.DataFrame(env_rows)
    case_df = add_case_diagnostics(raw_env_df, target_cap, target_tan, relaxed_tan, target_q20)
    results_df = add_culprit_gates(raw_results_df, case_df, target_cap, target_tan, relaxed_tan, target_q20)
    summary_df = summarize(results_df, target_cap, target_tan, relaxed_tan, target_q20)

    results_df.to_csv(ROOT / "phase26dy_lite_terminal_latch_results.csv", index=False)
    case_df.to_csv(ROOT / "phase26dy_lite_terminal_latch_cases.csv", index=False)
    summary_df.to_csv(ROOT / "phase26dy_lite_variant_summary.csv", index=False)

    plot_family_scores(summary_df, ROOT / "phase26dy_lite_family_scores.png")
    plot_pass_rates(summary_df, ROOT / "phase26dy_lite_pass_rates.png")
    plot_capture_tangent(results_df, ROOT / "phase26dy_lite_named_capture_vs_tangent.png", target_cap, target_tan, relaxed_tan)
    plot_label_culprits(case_df, ROOT / "phase26dy_lite_remaining_culprit_labels.png", target_cap, target_tan)

    stable_named_strict = summary_df[summary_df["stable_named_gate_strict_all_seeds"] == True] if not summary_df.empty else pd.DataFrame()
    stable_named_relaxed = summary_df[summary_df["stable_named_gate_relaxed_all_seeds"] == True] if not summary_df.empty else pd.DataFrame()
    stable_global_strict = summary_df[summary_df["stable_strict_all_seeds"] == True] if not summary_df.empty else pd.DataFrame()
    stable_global_relaxed = summary_df[summary_df["stable_relaxed_all_seeds"] == True] if not summary_df.empty else pd.DataFrame()
    best = summary_df.iloc[0] if len(summary_df) else pd.Series(dtype=float)

    if len(stable_named_strict) and len(stable_global_strict):
        verdict = "strict_terminal_latch_family_found"
    elif len(stable_named_relaxed) and len(stable_global_relaxed):
        verdict = "relaxed_terminal_latch_family_found"
    elif len(summary_df) and fnum(best.get("named_gate_relaxed_rate"), 0.0) >= args.min_promote_rate:
        verdict = "partial_terminal_latch_candidate_found"
    elif len(summary_df) and fnum(best.get("min_label_screen_capture"), 0.0) >= target_cap and fnum(best.get("max_label_guard_tangent"), 999.0) <= relaxed_tan:
        verdict = "single_family_culprit_gates_near_stable"
    else:
        verdict = "no_terminal_latch_family_found"

    def row_to_overrides(row: pd.Series) -> Dict[str, float] | None:
        if not len(row):
            return None
        return {k: fnum(row.get(k), FALLBACK_DW_BALANCED[k]) for k in PARAMS}

    label_summary = pd.DataFrame()
    if not case_df.empty:
        label_summary = case_df.groupby("envelope_label", sort=False).agg(
            min_capture=("capture_rate", "min"),
            max_tangent=("tangent_ratio", "max"),
            min_dp=("dp_score", "min"),
            capture_fail_rate=("case_pass_capture", lambda s: float((~s).mean())),
            relaxed_tangent_fail_rate=("case_pass_tangent_relaxed", lambda s: float((~s).mean())),
            n=("capture_rate", "size"),
        ).reset_index()
        label_summary.to_csv(ROOT / "phase26dy_lite_label_summary.csv", index=False)

    summary = {
        "phase": PHASE,
        "title": TITLE,
        "torch_info": torch_info,
        "targets": {
            "target_worst_tangent_ratio_strict": target_tan,
            "target_worst_tangent_ratio_relaxed": relaxed_tan,
            "target_worst_capture_rate": target_cap,
            "target_q20_dp_score": target_q20,
            "min_promote_rate": args.min_promote_rate,
        },
        "seed_count": int(args.seed_count),
        "max_variants": int(args.max_variants),
        "audit_repeats": int(args.audit_repeats),
        "sample_full_tail": bool(args.sample_full_tail),
        "num_variants": int(len(variants)),
        "num_runs": int(len(results_df)),
        "num_case_evals": int(len(case_df)),
        "num_cached_eval_entries": int(len(cache)),
        "num_stable_named_strict_families": int(len(stable_named_strict)),
        "num_stable_named_relaxed_families": int(len(stable_named_relaxed)),
        "num_stable_global_strict_families": int(len(stable_global_strict)),
        "num_stable_global_relaxed_families": int(len(stable_global_relaxed)),
        "elapsed_sec": float(time.time() - t0),
        "verdict": verdict,
        "best_variant": None if not len(best) else str(best.get("variant")),
        "best_dy_family_score": None if not len(best) else fnum(best.get("dy_family_score")),
        "best_named_gate_strict_rate": None if not len(best) else fnum(best.get("named_gate_strict_rate")),
        "best_named_gate_relaxed_rate": None if not len(best) else fnum(best.get("named_gate_relaxed_rate")),
        "best_global_strict_rate": None if not len(best) else fnum(best.get("strict_pass_rate")),
        "best_global_relaxed_rate": None if not len(best) else fnum(best.get("relaxed_pass_rate")),
        "best_min_cap": None if not len(best) else fnum(best.get("min_worst_capture_rate")),
        "best_min_label_screen_capture": None if not len(best) else fnum(best.get("min_label_screen_capture")),
        "best_max_tan": None if not len(best) else fnum(best.get("max_worst_tangent_ratio")),
        "best_max_label_guard_tangent": None if not len(best) else fnum(best.get("max_label_guard_tangent")),
        "best_min_q20": None if not len(best) else fnum(best.get("min_q20_dp_score")),
        "best_overrides": row_to_overrides(best),
        "variant_top10": summary_df.head(10).to_dict(orient="records") if len(summary_df) else [],
        "label_summary": label_summary.to_dict(orient="records") if not label_summary.empty else [],
        "outputs": [
            "phase26dy_lite_terminal_latch_results.csv",
            "phase26dy_lite_terminal_latch_cases.csv",
            "phase26dy_lite_variant_summary.csv",
            "phase26dy_lite_label_summary.csv",
            "phase26dy_lite_summary.json",
            "phase26dy_lite_family_scores.png",
            "phase26dy_lite_pass_rates.png",
            "phase26dy_lite_named_capture_vs_tangent.png",
            "phase26dy_lite_remaining_culprit_labels.png",
        ],
    }
    (ROOT / "phase26dy_lite_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[{PHASE}] Done in {summary['elapsed_sec']:.1f}s")
    print(f"[{PHASE}] VERDICT: {verdict}")
    if len(best):
        print(
            f"[{PHASE}] BEST VARIANT: {best['variant']} | family={best['dy_family_score']:.3f} "
            f"namedStrict={best['named_gate_strict_rate']:.2f} namedRelax={best['named_gate_relaxed_rate']:.2f} "
            f"globalStrict={best['strict_pass_rate']:.2f} globalRelax={best['relaxed_pass_rate']:.2f} "
            f"minCap={best['min_worst_capture_rate']:.3f} labelCap={best['min_label_screen_capture']:.3f} "
            f"maxTan={best['max_worst_tangent_ratio']:.3f} guardTan={best['max_label_guard_tangent']:.3f} "
            f"minQ20={best['min_q20_dp_score']:.3f}"
        )
        print(f"[{PHASE}] BEST overrides: {row_to_overrides(best)}")
    print(f"[{PHASE}] Wrote summary: {ROOT / 'phase26dy_lite_summary.json'}")
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=f"{PHASE}: {TITLE}")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="torch device preference")
    ap.add_argument("--seed", type=int, default=29000)
    ap.add_argument("--max-variants", type=int, default=14, help="candidate families to test")
    ap.add_argument("--seed-count", type=int, default=2, help="seeds per variant; raise to 3 after a promising run")
    ap.add_argument("--audit-repeats", type=int, default=1, help="micro-envelope repeats per seed")
    ap.add_argument("--sample-full-tail", action="store_true", help="include a small sampled full-envelope tail; slower")
    ap.add_argument("--relaxed-tan", type=float, default=2.30, help="early relaxed tangent threshold")
    ap.add_argument("--min-promote-rate", type=float, default=0.67, help="promote threshold for partial stability")
    ap.add_argument("--print-every", type=int, default=6)
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
