#!/usr/bin/env python3
"""
Phase 26DR — strict confidence-envelope low-tangent hardening probe.

Run from E:\BBIT, same as the previous phases:

    python bbit_geomlang/geomlang_phase26dr_strict_confidence_envelope_low_tangent_hardening_basic32_E_drive.py

What this phase is trying to do
-------------------------------
26DQ found very high confidence-envelope scores, but the worst-envelope tangent/radial
ratio was still often above the soft lockdown line. 26DR keeps the same inherited
basin evaluator, but changes the search target:

    * not just high DP/DQ scalar score;
    * not just high capture;
    * specifically: preserve a good lower-tail DP score while forcing the worst
      stress-envelope tangent/radial ratio down toward <= ~2.10.

So this is a "strict survivor" phase. It builds candidates from the best DQ/DP/DO/DN
results, then tests each candidate under a heavier confidence envelope with larger
jitter and several directed adversarial perturbations.

Outputs
geomlang_phase26dr_strict_confidence_envelope_low_tangent_hardening_basic32_E_drive-------
    phase26dr_case_results.csv
    phase26dr_envelope_results.csv
    phase26dr_pareto_front.csv
    phase26dr_summary.json
    phase26dr_top_scores.png
    phase26dr_capture_vs_tangent_pareto.png
    phase26dr_envelope_scores.png
    phase26dr_strict_survivors.png
"""

from __future__ import annotations

import importlib.util
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PHASE = "26DR"
TITLE = "Strict confidence-envelope low-tangent hardening"

ROOT = Path(r"E:\BBIT\outputs_basic32")
THIS_DIR = Path(__file__).resolve().parent
PREV_PATH = THIS_DIR / "geomlang_phase26dq_confidence_envelope_lockdown_basic32_E_drive.py"

if not PREV_PATH.exists():
    raise FileNotFoundError(
        f"Cannot find previous phase file: {PREV_PATH}\n"
        "Put this script in E:\\BBIT\\bbit_geomlang next to the 26DQ script."
    )

spec = importlib.util.spec_from_file_location("phase26dq", str(PREV_PATH))
PREV = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(PREV)  # type: ignore[arg-type]

PARAMS = list(PREV.PARAMS)
full_params = PREV.full_params
clamp_param = PREV.clamp_param
eval_base_case = PREV.eval_base_case

# Pull the current known baseline from DQ.
BASE = full_params({})

# Strict-envelope targets. These are intentionally hard but not impossible.
TARGET_WORST_TAN = 2.10
TARGET_WORST_CAP = 0.285
TARGET_Q20_DP = 2.25
TARGET_WORST_DP = 1.95


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def param_key(ov: Dict[str, float], ndigits: int = 5) -> Tuple[float, ...]:
    return tuple(round(float(ov[k]), ndigits) for k in PARAMS)


def add_candidate(out: List[Tuple[str, Dict[str, float]]], seen: set, name: str, ov: Dict[str, float]) -> None:
    fp = full_params(ov)
    key = param_key(fp)
    if key in seen:
        return
    seen.add(key)
    safe_name = (
        name.replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("+", "p")
        .replace("-", "m")
        .replace(".", "p")
    )
    out.append((safe_name[:150], fp))


def read_csv_candidates(path: Path, score_col: str, top_n: int, prefix: str) -> List[Tuple[str, Dict[str, float], Dict[str, Any]]]:
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[{PHASE}] Warning: could not read {path.name}: {e}")
        return []
    for p in PARAMS:
        if p not in df.columns:
            return []
    if score_col not in df.columns:
        # Fall back to the first score-looking column.
        for c in ["dq_score", "dp_score", "do_score", "dn_score", "stability_score"]:
            if c in df.columns:
                score_col = c
                break
    if score_col in df.columns:
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
        df = df.sort_values(score_col, ascending=False)
    rows: List[Tuple[str, Dict[str, float], Dict[str, Any]]] = []
    for idx, r in df.head(top_n).iterrows():
        ov = {p: safe_float(r[p], BASE[p]) for p in PARAMS}
        case = str(r.get("case", f"row_{idx}"))
        rows.append((f"{prefix}_{idx:03d}_{case}", full_params(ov), r.to_dict()))
    return rows


def read_prior_candidates() -> List[Tuple[str, Dict[str, float], Dict[str, Any]]]:
    """Collect the best candidates from DQ plus nearby earlier phases."""
    specs = [
        (ROOT / "phase26dq_case_results.csv", "dq_score", 34, "dq_case"),
        (ROOT / "phase26dq_pareto_front.csv", "dq_score", 28, "dq_pareto"),
        (ROOT / "phase26dp_case_results.csv", "dp_score", 20, "dp_case"),
        (ROOT / "phase26do_case_results.csv", "do_score", 20, "do_case"),
        (ROOT / "phase26dn_base_results.csv", "dn_score", 16, "dn_base"),
    ]
    out: List[Tuple[str, Dict[str, float], Dict[str, Any]]] = []
    for path, score, n, prefix in specs:
        out.extend(read_csv_candidates(path, score, n, prefix))
    return out


def strict_score_from_envelope(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate stress-envelope rows into a DR score.

    This deliberately puts a much harsher penalty on the *worst* tangent/radial
    ratio than DQ did. A high score now means the candidate is not just good on
    average; it is resistant to the adversarial envelope.
    """
    vals: List[Dict[str, float]] = []
    for r in rows:
        vals.append({
            "dp_score": safe_float(r.get("dp_score"), -999.0),
            "capture_rate": safe_float(r.get("capture_rate"), 0.0),
            "tangent_ratio": safe_float(r.get("tangent_ratio"), 999.0),
            "distance_progress": safe_float(r.get("distance_progress"), -999.0),
            "signed_alignment": safe_float(r.get("signed_alignment"), -999.0),
        })

    if not vals:
        return {
            "dr_score": -999.0,
            "q20_dp_score": -999.0,
            "worst_dp_score": -999.0,
            "mean_dp_score": -999.0,
            "worst_capture_rate": 0.0,
            "mean_capture_rate": 0.0,
            "worst_tangent_ratio": 999.0,
            "median_tangent_ratio": 999.0,
            "survival_rate": 0.0,
            "strict_survival_rate": 0.0,
            "num_envelope_cases": 0,
        }

    dp = np.array([v["dp_score"] for v in vals], dtype=float)
    cap = np.array([v["capture_rate"] for v in vals], dtype=float)
    tan = np.array([v["tangent_ratio"] for v in vals], dtype=float)
    prog = np.array([v["distance_progress"] for v in vals], dtype=float)
    align = np.array([v["signed_alignment"] for v in vals], dtype=float)

    q20_dp = float(np.quantile(dp, 0.20))
    q10_dp = float(np.quantile(dp, 0.10))
    worst_dp = float(np.min(dp))
    mean_dp = float(np.mean(dp))
    worst_cap = float(np.min(cap))
    q20_cap = float(np.quantile(cap, 0.20))
    mean_cap = float(np.mean(cap))
    worst_tan = float(np.max(tan))
    q80_tan = float(np.quantile(tan, 0.80))
    med_tan = float(np.median(tan))
    worst_prog = float(np.min(prog))
    worst_align = float(np.min(align))

    soft_survive = (dp >= 1.65) & (cap >= 0.235) & (tan <= 3.00) & (prog > 0.0)
    strict_survive = (dp >= TARGET_WORST_DP) & (cap >= TARGET_WORST_CAP) & (tan <= TARGET_WORST_TAN) & (prog > 0.0)
    survival_rate = float(np.mean(soft_survive))
    strict_survival_rate = float(np.mean(strict_survive))

    # Smooth reward terms.
    cap_term = 5.0 * (worst_cap - 0.240) + 2.5 * (q20_cap - 0.260)
    tail_term = 0.90 * q20_dp + 0.55 * q10_dp + 0.45 * worst_dp + 0.22 * mean_dp
    survival_term = 1.40 * survival_rate + 1.65 * strict_survival_rate
    alignment_term = 0.45 * max(0.0, worst_align) + 0.35 * max(0.0, worst_prog)

    # Hard penalties: these dominate the score when the worst envelope explodes tangentially.
    tangent_penalty = 0.0
    tangent_penalty += 1.25 * max(0.0, worst_tan - TARGET_WORST_TAN)
    tangent_penalty += 0.55 * max(0.0, q80_tan - 1.65)
    tangent_penalty += 0.30 * max(0.0, med_tan - 1.20)
    cap_penalty = 5.0 * max(0.0, TARGET_WORST_CAP - worst_cap)
    tail_penalty = 0.85 * max(0.0, TARGET_Q20_DP - q20_dp) + 0.75 * max(0.0, TARGET_WORST_DP - worst_dp)

    dr_score = tail_term + cap_term + survival_term + alignment_term - tangent_penalty - cap_penalty - tail_penalty

    return {
        "dr_score": float(dr_score),
        "q20_dp_score": q20_dp,
        "q10_dp_score": q10_dp,
        "worst_dp_score": worst_dp,
        "mean_dp_score": mean_dp,
        "worst_capture_rate": worst_cap,
        "q20_capture_rate": q20_cap,
        "mean_capture_rate": mean_cap,
        "worst_tangent_ratio": worst_tan,
        "q80_tangent_ratio": q80_tan,
        "median_tangent_ratio": med_tan,
        "worst_distance_progress": worst_prog,
        "worst_signed_alignment": worst_align,
        "survival_rate": survival_rate,
        "strict_survival_rate": strict_survival_rate,
        "num_envelope_cases": int(len(vals)),
    }


def hardening_candidates(priors: List[Tuple[str, Dict[str, float], Dict[str, Any]]]) -> List[Tuple[str, Dict[str, float]]]:
    out: List[Tuple[str, Dict[str, float]]] = []
    seen: set = set()

    add_candidate(out, seen, "dr_base_current_constants", BASE)

    # Prior survivors: use all top DQ/DP/DO/DN points as anchors.
    for i, (name, ov, meta) in enumerate(priors[:72]):
        add_candidate(out, seen, f"dr_prior_{i:02d}_{name}", ov)

    anchors = [ov for _, ov, _ in priors[:42]] or [BASE]

    # Direct low-tangent hardening moves.  The important axis is not simply
    # "more tangent kill"; DQ/DP showed a knee, so we vary tangent kill with
    # cap/shell/blend in coupled packets.
    packets = [
        # conservative inward shell, more tangent kill
        (-0.020, -0.020, -0.010, +0.020, -0.006, 0.000),
        (-0.030, -0.030, -0.014, +0.040, -0.010, -0.010),
        (-0.040, -0.035, -0.018, +0.060, -0.014, -0.018),
        # keep cap while adding tangent kill
        (0.000, -0.016, -0.008, +0.030, -0.006, 0.000),
        (+0.010, -0.014, -0.006, +0.045, -0.006, +0.004),
        # radius/seat low-tangent micro-knee from DP
        (-0.020, -0.040, -0.020, +0.020, +0.000, -0.008),
        (-0.040, -0.050, -0.025, +0.030, +0.000, -0.012),
        # cap recovery packets if capture collapses under strict stress
        (+0.020, +0.000, +0.010, +0.020, +0.004, +0.004),
        (+0.035, +0.005, +0.014, +0.015, +0.006, +0.008),
        # small tangent-kill reduction branch in case high kill creates bounce
        (+0.000, +0.006, +0.006, -0.020, +0.004, +0.006),
    ]
    # tuple order: radius, seat, blend, tk, shell, cap
    for ai, base in enumerate(anchors[:30]):
        for pi, (dr, ds, db, dtk, dsh, dc) in enumerate(packets):
            ov = dict(base)
            ov["BOWL_RADIUS_FRAC"] = clamp_param("BOWL_RADIUS_FRAC", ov["BOWL_RADIUS_FRAC"] + dr)
            ov["BOWL_SEAT_AXIS_GAIN"] = clamp_param("BOWL_SEAT_AXIS_GAIN", ov["BOWL_SEAT_AXIS_GAIN"] + ds)
            ov["BOWL_DIRECTIONAL_BLEND"] = clamp_param("BOWL_DIRECTIONAL_BLEND", ov["BOWL_DIRECTIONAL_BLEND"] + db)
            ov["BOWL_TANGENT_KILL"] = clamp_param("BOWL_TANGENT_KILL", ov["BOWL_TANGENT_KILL"] + dtk)
            ov["BOWL_SHELL_RADIAL_GAIN"] = clamp_param("BOWL_SHELL_RADIAL_GAIN", ov["BOWL_SHELL_RADIAL_GAIN"] + dsh)
            ov["BOWL_NORM_CAP_MULT"] = clamp_param("BOWL_NORM_CAP_MULT", ov["BOWL_NORM_CAP_MULT"] + dc)
            add_candidate(out, seen, f"dr_harden_a{ai:02d}_p{pi:02d}", ov)

    # Deterministic random cloud with a bias toward the strict knee region.
    rng = random.Random(2637)
    for j in range(70):
        base = dict(rng.choice(anchors))
        # Push distributions slightly toward: lower radius/seat/blend, higher TK, modest cap.
        ov = {
            "BOWL_RADIUS_FRAC": base["BOWL_RADIUS_FRAC"] + rng.uniform(-0.045, 0.018),
            "BOWL_SEAT_AXIS_GAIN": base["BOWL_SEAT_AXIS_GAIN"] + rng.uniform(-0.055, 0.015),
            "BOWL_DIRECTIONAL_BLEND": base["BOWL_DIRECTIONAL_BLEND"] + rng.uniform(-0.040, 0.016),
            "BOWL_NORM_CAP_MULT": base["BOWL_NORM_CAP_MULT"] + rng.uniform(-0.012, 0.026),
            "BOWL_SHELL_RADIAL_GAIN": base["BOWL_SHELL_RADIAL_GAIN"] + rng.uniform(-0.018, 0.008),
            "BOWL_TANGENT_KILL": base["BOWL_TANGENT_KILL"] + rng.uniform(-0.006, 0.072),
        }
        add_candidate(out, seen, f"dr_strict_cloud_{j:02d}", ov)

    # Explicit micro-grid centered on the best DQ region in case prior CSV names differ.
    # Values are clamped, so this remains safe across slightly different earlier constants.
    for r in [0.62, 0.64, 0.66, 0.68]:
        for seat in [0.30, 0.34, 0.38, 0.42]:
            for tk in [0.88, 0.92, 0.96, 1.00]:
                ov = full_params({
                    "BOWL_RADIUS_FRAC": r,
                    "BOWL_SEAT_AXIS_GAIN": seat,
                    "BOWL_DIRECTIONAL_BLEND": 1.00,
                    "BOWL_NORM_CAP_MULT": 1.06,
                    "BOWL_SHELL_RADIAL_GAIN": 0.12,
                    "BOWL_TANGENT_KILL": tk,
                })
                add_candidate(out, seen, f"dr_microgrid_r{r:.2f}_s{seat:.2f}_tk{tk:.2f}", ov)

    return out


def jitter_variant(base: Dict[str, float], rng: random.Random, scale: float) -> Dict[str, float]:
    return full_params({
        "BOWL_RADIUS_FRAC": base["BOWL_RADIUS_FRAC"] + rng.uniform(-0.016, 0.016) * scale,
        "BOWL_SEAT_AXIS_GAIN": base["BOWL_SEAT_AXIS_GAIN"] + rng.uniform(-0.026, 0.026) * scale,
        "BOWL_DIRECTIONAL_BLEND": base["BOWL_DIRECTIONAL_BLEND"] + rng.uniform(-0.026, 0.026) * scale,
        "BOWL_NORM_CAP_MULT": base["BOWL_NORM_CAP_MULT"] + rng.uniform(-0.026, 0.026) * scale,
        "BOWL_SHELL_RADIAL_GAIN": base["BOWL_SHELL_RADIAL_GAIN"] + rng.uniform(-0.012, 0.012) * scale,
        "BOWL_TANGENT_KILL": base["BOWL_TANGENT_KILL"] + rng.uniform(-0.040, 0.040) * scale,
    })


def directed_stress_variants(base: Dict[str, float]) -> List[Tuple[str, Dict[str, float]]]:
    specs = [
        ("stress_low_cap_high_tk", {"BOWL_NORM_CAP_MULT": -0.040, "BOWL_TANGENT_KILL": +0.055}),
        ("stress_high_cap_low_tk", {"BOWL_NORM_CAP_MULT": +0.045, "BOWL_TANGENT_KILL": -0.055}),
        ("stress_low_shell_low_blend", {"BOWL_SHELL_RADIAL_GAIN": -0.018, "BOWL_DIRECTIONAL_BLEND": -0.035}),
        ("stress_high_shell_high_blend", {"BOWL_SHELL_RADIAL_GAIN": +0.018, "BOWL_DIRECTIONAL_BLEND": +0.035}),
        ("stress_radius_in", {"BOWL_RADIUS_FRAC": -0.026}),
        ("stress_radius_out", {"BOWL_RADIUS_FRAC": +0.026}),
        ("stress_seat_down", {"BOWL_SEAT_AXIS_GAIN": -0.040}),
        ("stress_seat_up", {"BOWL_SEAT_AXIS_GAIN": +0.040}),
        ("stress_tk_drop_shell_up", {"BOWL_TANGENT_KILL": -0.070, "BOWL_SHELL_RADIAL_GAIN": +0.014}),
        ("stress_tk_raise_shell_down", {"BOWL_TANGENT_KILL": +0.070, "BOWL_SHELL_RADIAL_GAIN": -0.014}),
        ("stress_cap_drop_blend_drop", {"BOWL_NORM_CAP_MULT": -0.050, "BOWL_DIRECTIONAL_BLEND": -0.040}),
        ("stress_cap_raise_blend_raise", {"BOWL_NORM_CAP_MULT": +0.050, "BOWL_DIRECTIONAL_BLEND": +0.040}),
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
    for i in range(5):
        out.append((f"jitter_small_{i:02d}", jitter_variant(base, rng, scale=1.0), "jitter_small"))
    for i in range(5):
        out.append((f"jitter_medium_{i:02d}", jitter_variant(base, rng, scale=1.65), "jitter_medium"))
    for i in range(4):
        out.append((f"jitter_large_{i:02d}", jitter_variant(base, rng, scale=2.35), "jitter_large"))
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

    agg = strict_score_from_envelope(rows)
    base_row = next((r for r in rows if r.get("envelope_label") == "base"), rows[0])
    out: Dict[str, Any] = {
        "case": case,
        **{k: float(ov[k]) for k in PARAMS},
        **agg,
        "base_dp_score": safe_float(base_row.get("dp_score")),
        "base_capture_rate": safe_float(base_row.get("capture_rate")),
        "base_tangent_ratio": safe_float(base_row.get("tangent_ratio")),
        "base_distance_progress": safe_float(base_row.get("distance_progress")),
        "base_signed_alignment": safe_float(base_row.get("signed_alignment")),
        "overrides": json.dumps({k: float(ov[k]) for k in PARAMS}, sort_keys=True),
    }
    return out, rows


def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    work = df.copy()
    cols_defaults = {
        "dr_score": 0.0,
        "q20_dp_score": 0.0,
        "worst_dp_score": 0.0,
        "strict_survival_rate": 0.0,
        "survival_rate": 0.0,
        "worst_capture_rate": 0.0,
        "worst_tangent_ratio": 999.0,
    }
    for c, default in cols_defaults.items():
        if c not in work:
            work[c] = default
        work[c] = pd.to_numeric(work[c], errors="coerce").fillna(default)

    vals = work[[
        "dr_score", "q20_dp_score", "worst_dp_score", "strict_survival_rate",
        "survival_rate", "worst_capture_rate", "worst_tangent_ratio",
    ]].to_numpy(float)
    keep: List[int] = []
    for i, a in enumerate(vals):
        dominated = False
        for j, b in enumerate(vals):
            if i == j:
                continue
            ge = (
                b[0] >= a[0] and b[1] >= a[1] and b[2] >= a[2] and
                b[3] >= a[3] and b[4] >= a[4] and b[5] >= a[5] and b[6] <= a[6]
            )
            gt = (
                b[0] > a[0] or b[1] > a[1] or b[2] > a[2] or
                b[3] > a[3] or b[4] > a[4] or b[5] > a[5] or b[6] < a[6]
            )
            if ge and gt:
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return work.iloc[keep].sort_values("dr_score", ascending=False)


def plot_top_scores(df: pd.DataFrame) -> None:
    if df.empty:
        return
    top = df.sort_values("dr_score", ascending=False).head(34).iloc[::-1]
    plt.figure(figsize=(15, max(8, 0.50 * len(top))))
    plt.barh(top["case"], top["dr_score"])
    plt.xlabel("DR strict-envelope score")
    plt.title("26DR strict confidence-envelope low-tangent scores")
    plt.tight_layout()
    plt.savefig(ROOT / "phase26dr_top_scores.png", dpi=150)
    plt.close()


def plot_capture_tangent(df: pd.DataFrame, pf: pd.DataFrame) -> None:
    if df.empty:
        return
    plt.figure(figsize=(10, 7))
    sc = plt.scatter(
        df["worst_capture_rate"],
        df["worst_tangent_ratio"],
        c=df["dr_score"],
        s=70,
        alpha=0.76,
    )
    if not pf.empty:
        plt.scatter(
            pf["worst_capture_rate"], pf["worst_tangent_ratio"],
            facecolors="none", edgecolors="black", s=170, linewidths=1.7,
            label="DR Pareto front",
        )
        plt.legend()
    plt.axvline(TARGET_WORST_CAP, linewidth=1.0, alpha=0.45)
    plt.axhline(TARGET_WORST_TAN, linewidth=1.0, alpha=0.45)
    plt.colorbar(sc, label="DR score")
    plt.xlabel("worst envelope capture_rate")
    plt.ylabel("worst envelope tangent/radial ratio")
    plt.title("26DR strict worst-envelope capture vs tangent drift")
    plt.grid(True, alpha=0.28)
    plt.tight_layout()
    plt.savefig(ROOT / "phase26dr_capture_vs_tangent_pareto.png", dpi=150)
    plt.close()


def plot_envelope_bars(df: pd.DataFrame) -> None:
    if df.empty:
        return
    top = df.sort_values("dr_score", ascending=False).head(12).iloc[::-1]
    y = np.arange(len(top))
    plt.figure(figsize=(14, max(6, 0.56 * len(top))))
    plt.barh(y - 0.25, top["dr_score"], height=0.22, label="DR score")
    plt.barh(y + 0.00, top["q20_dp_score"], height=0.22, label="q20 DP")
    plt.barh(y + 0.25, top["worst_dp_score"], height=0.22, label="worst DP")
    plt.barh(y + 0.50, top["worst_tangent_ratio"], height=0.22, label="worst tangent")
    plt.yticks(y, top["case"])
    plt.xlabel("score / ratio")
    plt.title("26DR strict envelope: score vs lower-tail DP and tangent")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "phase26dr_envelope_scores.png", dpi=150)
    plt.close()


def plot_strict_survivors(df: pd.DataFrame) -> None:
    if df.empty:
        return
    work = df.copy()
    work["passes_strict_line"] = (work["worst_capture_rate"] >= TARGET_WORST_CAP) & (work["worst_tangent_ratio"] <= TARGET_WORST_TAN)
    top = work.sort_values(["passes_strict_line", "strict_survival_rate", "dr_score"], ascending=False).head(18).iloc[::-1]
    y = np.arange(len(top))
    plt.figure(figsize=(13, max(6, 0.55 * len(top))))
    plt.barh(y - 0.18, top["strict_survival_rate"], height=0.30, label="strict survival")
    plt.barh(y + 0.18, top["survival_rate"], height=0.30, label="soft survival")
    plt.yticks(y, top["case"])
    plt.xlabel("fraction of envelope cases surviving")
    plt.title("26DR strict-envelope survivor rates")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "phase26dr_strict_survivors.png", dpi=150)
    plt.close()


def inherited_plots(z2: Any, rel_ids: Any, basins: Any, best: pd.Series) -> List[str]:
    errors: List[str] = []
    # DQ inherited plotting already knows how to reach the older AP plotting stack.
    for fn_name in ["inherited_plots", "maybe_inherited_plots"]:
        fn = getattr(PREV, fn_name, None)
        if fn is None:
            continue
        try:
            result = fn(z2, rel_ids, basins, best)
            if isinstance(result, list):
                errors.extend([str(x) for x in result])
            return errors
        except Exception as e:
            errors.append(repr(e))
    return errors


def main() -> Dict[str, Any]:
    t0 = time.time()
    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] Loaded previous phase: {PREV_PATH}")
    print(f"[{PHASE}] Output root: {ROOT}")
    print(f"[{PHASE}] Strict targets: worst_tan<={TARGET_WORST_TAN:.2f}, worst_cap>={TARGET_WORST_CAP:.3f}, q20DP>={TARGET_Q20_DP:.2f}")

    build_basins = PREV.PREV.build_basins if hasattr(PREV, "PREV") else PREV.build_basins
    z2, rel_ids, basins = build_basins()
    print(f"[{PHASE}] Built basins: {len(basins)}")

    priors = read_prior_candidates()
    candidates = hardening_candidates(priors)
    print(f"[{PHASE}] Prior candidates: {len(priors)}")
    print(f"[{PHASE}] DR candidates: {len(candidates)}")

    case_rows: List[Dict[str, Any]] = []
    env_rows: List[Dict[str, Any]] = []

    for i, (case, ov) in enumerate(candidates, 1):
        try:
            agg, rows = evaluate_candidate_envelope(case, ov, basins, seed=4000 + i)
            case_rows.append(agg)
            env_rows.extend(rows)
            if i <= 5 or i % 10 == 0:
                print(
                    f"[{PHASE}] {i:03d}/{len(candidates)} {case[:74]:74s} "
                    f"DR={agg['dr_score']:.4f} q20={agg['q20_dp_score']:.4f} "
                    f"worstDP={agg['worst_dp_score']:.4f} worstCap={agg['worst_capture_rate']:.3f} "
                    f"worstTan={agg['worst_tangent_ratio']:.3f} strict={agg['strict_survival_rate']:.2f} soft={agg['survival_rate']:.2f}"
                )
        except Exception as e:
            print(f"[{PHASE}] ERROR {case}: {e}")
            case_rows.append({"case": case, "error": repr(e), "dr_score": float("nan"), **ov})

    df = pd.DataFrame(case_rows)
    env = pd.DataFrame(env_rows)
    df.to_csv(ROOT / "phase26dr_case_results.csv", index=False)
    env.to_csv(ROOT / "phase26dr_envelope_results.csv", index=False)

    df_ok = df[pd.to_numeric(df.get("dr_score", pd.Series(dtype=float)), errors="coerce").notna()].copy()
    df_ok = df_ok.sort_values("dr_score", ascending=False)
    pf = pareto_front(df_ok)
    pf.to_csv(ROOT / "phase26dr_pareto_front.csv", index=False)

    plot_top_scores(df_ok)
    plot_capture_tangent(df_ok, pf)
    plot_envelope_bars(df_ok)
    plot_strict_survivors(df_ok)

    best = df_ok.iloc[0] if len(df_ok) else pd.Series(dtype=float)
    strict_df = df_ok[(df_ok["worst_capture_rate"] >= TARGET_WORST_CAP) & (df_ok["worst_tangent_ratio"] <= TARGET_WORST_TAN)].copy() if len(df_ok) else pd.DataFrame()
    best_strict = strict_df.iloc[0] if len(strict_df) else pd.Series(dtype=float)

    plot_errors: List[str] = []
    if len(best):
        plot_errors = inherited_plots(z2, rel_ids, basins, best)

    summary = {
        "phase": PHASE,
        "title": TITLE,
        "prev_path": str(PREV_PATH),
        "root": str(ROOT),
        "targets": {
            "target_worst_tangent_ratio": TARGET_WORST_TAN,
            "target_worst_capture_rate": TARGET_WORST_CAP,
            "target_q20_dp_score": TARGET_Q20_DP,
            "target_worst_dp_score": TARGET_WORST_DP,
        },
        "num_priors": int(len(priors)),
        "num_candidates": int(len(candidates)),
        "num_success": int(len(df_ok)),
        "num_envelope_evals": int(len(env)),
        "num_pareto": int(len(pf)),
        "num_strict_line_pass": int(len(strict_df)),
        "elapsed_sec": float(time.time() - t0),
        "best_case": None if not len(best) else str(best.get("case")),
        "best_dr_score": None if not len(best) else float(best.get("dr_score", float("nan"))),
        "best_q20_dp_score": None if not len(best) else float(best.get("q20_dp_score", float("nan"))),
        "best_worst_dp_score": None if not len(best) else float(best.get("worst_dp_score", float("nan"))),
        "best_worst_capture_rate": None if not len(best) else float(best.get("worst_capture_rate", float("nan"))),
        "best_worst_tangent_ratio": None if not len(best) else float(best.get("worst_tangent_ratio", float("nan"))),
        "best_survival_rate": None if not len(best) else float(best.get("survival_rate", float("nan"))),
        "best_strict_survival_rate": None if not len(best) else float(best.get("strict_survival_rate", float("nan"))),
        "best_overrides": None if not len(best) else {k: float(best[k]) for k in PARAMS if k in best and pd.notna(best[k])},
        "best_strict_case": None if not len(best_strict) else str(best_strict.get("case")),
        "best_strict_dr_score": None if not len(best_strict) else float(best_strict.get("dr_score", float("nan"))),
        "best_strict_overrides": None if not len(best_strict) else {k: float(best_strict[k]) for k in PARAMS if k in best_strict and pd.notna(best_strict[k])},
        "top10": df_ok.head(10).to_dict(orient="records") if len(df_ok) else [],
        "strict_top10": strict_df.head(10).to_dict(orient="records") if len(strict_df) else [],
        "pareto_top10": pf.head(10).to_dict(orient="records") if len(pf) else [],
        "plot_errors": plot_errors,
        "outputs": [
            "phase26dr_case_results.csv",
            "phase26dr_envelope_results.csv",
            "phase26dr_pareto_front.csv",
            "phase26dr_summary.json",
            "phase26dr_top_scores.png",
            "phase26dr_capture_vs_tangent_pareto.png",
            "phase26dr_envelope_scores.png",
            "phase26dr_strict_survivors.png",
        ],
    }
    with open(ROOT / "phase26dr_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[{PHASE}] Done in {summary['elapsed_sec']:.1f}s")
    if len(best):
        print(
            f"[{PHASE}] BEST: {summary['best_case']} | "
            f"DR={summary['best_dr_score']:.4f} q20DP={summary['best_q20_dp_score']:.4f} "
            f"worstDP={summary['best_worst_dp_score']:.4f} worstCap={summary['best_worst_capture_rate']:.3f} "
            f"worstTan={summary['best_worst_tangent_ratio']:.3f} "
            f"strict={summary['best_strict_survival_rate']:.2f} soft={summary['best_survival_rate']:.2f}"
        )
        print(f"[{PHASE}] BEST overrides: {summary['best_overrides']}")
    if len(best_strict):
        print(f"[{PHASE}] BEST STRICT-LINE PASS: {summary['best_strict_case']} | DR={summary['best_strict_dr_score']:.4f}")
        print(f"[{PHASE}] BEST STRICT overrides: {summary['best_strict_overrides']}")
    else:
        print(f"[{PHASE}] No candidate passed the strict line worstCap>={TARGET_WORST_CAP:.3f} and worstTan<={TARGET_WORST_TAN:.2f}.")
    print(f"[{PHASE}] Wrote summary: {ROOT / 'phase26dr_summary.json'}")
    return summary


if __name__ == "__main__":
    main()
