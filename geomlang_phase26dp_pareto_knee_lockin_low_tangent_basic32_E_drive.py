# geomlang_phase26dp_pareto_knee_lockin_low_tangent_basic32_E_drive.py
# Phase 26DP — Pareto-knee lock-in low-tangent audit/refine
#
# Purpose:
#   26DO showed that we can now push the capture/tangent frontier upward, but the raw
#   DO score can still favor high-capture candidates that pay for capture with too much
#   sideways/tangential drift. 26DP is a narrower, more diagnostic phase:
#
#   1. Re-import the 26DO/DN/DM winners and Pareto-front points.
#   2. Build a compact candidate cloud around the Pareto knees, not the whole grid.
#   3. Score with a stricter knee-lock objective: keep capture, punish tangent drift,
#      reward stability under small parameter jitter, and identify deployable constants.
#
# Expected outputs:
#   phase26dp_case_results.csv
#   phase26dp_pareto_front.csv
#   phase26dp_stability_summary.csv
#   phase26dp_jitter_results.csv
#   phase26dp_summary.json
#   phase26dp_top_scores.png
#   phase26dp_capture_vs_tangent_pareto.png
#   phase26dp_stability_scores.png
#   plus inherited AP-style vector/rollout/heatmap plots for the best row if available.

from __future__ import annotations

import importlib.util
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PHASE = "26DP"
TITLE = "Pareto-knee lock-in low-tangent refine"

ROOT = Path(r"E:\BBIT") if Path(r"E:\BBIT").exists() else Path.cwd()
PREV_PATH = ROOT / "bbit_geomlang" / "geomlang_phase26do_pareto_low_tangent_refine_basic32_E_drive.py"
if not PREV_PATH.exists():
    # Useful when this file is tested from /mnt/data, but normal use is E:\BBIT.
    PREV_PATH = Path(__file__).with_name("geomlang_phase26do_pareto_low_tangent_refine_basic32_E_drive.py")

spec = importlib.util.spec_from_file_location("phase26do", str(PREV_PATH))
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not import previous phase from {PREV_PATH}")
PREV = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PREV)  # type: ignore[union-attr]
print(f"[{PHASE}] Loaded Phase 26DO helpers from: {PREV_PATH}")

PARAMS = [
    "BOWL_RADIUS_FRAC",
    "BOWL_SEAT_AXIS_GAIN",
    "BOWL_DIRECTIONAL_BLEND",
    "BOWL_NORM_CAP_MULT",
    "BOWL_SHELL_RADIAL_GAIN",
    "BOWL_TANGENT_KILL",
]

# Current stable constants inherited from the DO/DN lineage.
BASE = {
    "BOWL_RADIUS_FRAC": 0.66,
    "BOWL_SEAT_AXIS_GAIN": 0.35,
    "BOWL_DIRECTIONAL_BLEND": 1.00,
    "BOWL_NORM_CAP_MULT": 1.00,
    "BOWL_SHELL_RADIAL_GAIN": 0.10,
    "BOWL_TANGENT_KILL": 0.90,
}

# 26DP is intentionally narrower than 26DO. These are knee neighborhoods suggested by
# DO/DN/DM: moderate-to-high capture, tangent often around 1.0-2.0, not the high tangent
# capture-at-any-cost region.
KNEE_CENTERS = [
    ("dp_base_current_constants", BASE),
    ("dp_low_tangent_nominal", {
        "BOWL_RADIUS_FRAC": 0.66,
        "BOWL_SEAT_AXIS_GAIN": 0.35,
        "BOWL_DIRECTIONAL_BLEND": 1.04,
        "BOWL_NORM_CAP_MULT": 1.04,
        "BOWL_SHELL_RADIAL_GAIN": 0.12,
        "BOWL_TANGENT_KILL": 0.92,
    }),
    ("dp_dn_knee_shell014_tk080", {
        "BOWL_RADIUS_FRAC": 0.66,
        "BOWL_SEAT_AXIS_GAIN": 0.35,
        "BOWL_DIRECTIONAL_BLEND": 1.04,
        "BOWL_NORM_CAP_MULT": 1.04,
        "BOWL_SHELL_RADIAL_GAIN": 0.14,
        "BOWL_TANGENT_KILL": 0.80,
    }),
    ("dp_do_cloud_center", {
        "BOWL_RADIUS_FRAC": 0.70,
        "BOWL_SEAT_AXIS_GAIN": 0.40,
        "BOWL_DIRECTIONAL_BLEND": 1.08,
        "BOWL_NORM_CAP_MULT": 1.12,
        "BOWL_SHELL_RADIAL_GAIN": 0.14,
        "BOWL_TANGENT_KILL": 0.80,
    }),
    ("dp_high_capture_controlled_tangent", {
        "BOWL_RADIUS_FRAC": 0.74,
        "BOWL_SEAT_AXIS_GAIN": 0.45,
        "BOWL_DIRECTIONAL_BLEND": 1.08,
        "BOWL_NORM_CAP_MULT": 1.12,
        "BOWL_SHELL_RADIAL_GAIN": 0.14,
        "BOWL_TANGENT_KILL": 0.76,
    }),
]

CLAMP_RANGES = {
    "BOWL_RADIUS_FRAC": (0.60, 0.82),
    "BOWL_SEAT_AXIS_GAIN": (0.22, 0.55),
    "BOWL_DIRECTIONAL_BLEND": (0.92, 1.16),
    "BOWL_NORM_CAP_MULT": (0.96, 1.20),
    "BOWL_SHELL_RADIAL_GAIN": (0.08, 0.18),
    "BOWL_TANGENT_KILL": (0.72, 0.98),
}


def clamp_param(k: str, v: float) -> float:
    lo, hi = CLAMP_RANGES[k]
    return float(min(hi, max(lo, v)))


def clean_overrides(ov: Dict[str, Any]) -> Dict[str, float]:
    if hasattr(PREV, "clean_overrides"):
        return PREV.clean_overrides(ov)
    return {k: float(v) for k, v in ov.items() if k in PARAMS}


def full_params(ov: Dict[str, Any]) -> Dict[str, float]:
    if hasattr(PREV, "full_params"):
        return PREV.full_params(ov)
    out = dict(BASE)
    out.update(clean_overrides(ov))
    return out


def metric(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    if hasattr(PREV, "metric"):
        return PREV.metric(row, key, default)
    aliases = {
        "capture_rate": ["capture_rate", "capture", "cap"],
        "distance_progress": ["distance_progress", "radial_progress", "mean_radial_progress", "progress"],
        "signed_alignment": ["signed_alignment", "alignment", "mean_alignment"],
        "tangent_ratio": ["tangent_ratio", "mean_tangent_ratio", "tangent_radial_ratio"],
        "mean_endpoint_dist": ["mean_endpoint_dist", "endpoint_dist", "mean_endpoint_distance"],
    }
    for k in aliases.get(key, [key]):
        if k in row:
            try:
                v = float(row[k])
                if math.isfinite(v):
                    return v
            except Exception:
                pass
    return default


def dp_score(summary: Dict[str, Any]) -> float:
    """Stricter knee-lock score than DO.

    We want a deployable basin field, not just a large scalar. The intended sweet spot is:
    capture roughly >= .30, tangent/radial roughly <= 1.35-1.75, positive progress, and
    decent alignment. High capture with tangent blow-up is deliberately clipped.
    """
    cap = metric(summary, "capture_rate")
    prog = metric(summary, "distance_progress")
    align = metric(summary, "signed_alignment")
    tan = max(0.0, metric(summary, "tangent_ratio"))
    end = max(0.0, metric(summary, "mean_endpoint_dist"))

    # Smooth low-tangent reward: near 1.0 is strongly rewarded, 2.0 is tolerated,
    # 4.0+ is treated as unstable even if capture is high.
    low_tan = 1.0 / (1.0 + max(0.0, tan - 0.85) ** 1.35)
    cap_gate = 1.0 / (1.0 + math.exp(-18.0 * (cap - 0.30)))

    score = (
        2.90 * cap
        + 2.15 * prog
        + 0.85 * align
        + 1.45 * low_tan
        + 0.55 * cap_gate
        - 0.030 * end
        - 0.17 * math.log1p(tan)
    )

    # Harder penalties for two failure modes: low capture or tangent escape.
    if cap < 0.27:
        score -= 2.25 * (0.27 - cap)
    if tan > 2.50:
        score -= 0.45 * (tan - 2.50)
    if tan > 4.50:
        score -= 0.75 * (tan - 4.50)
    return float(score)


def compact_result(case: str, overrides: Dict[str, float], res: Dict[str, Any]) -> Dict[str, Any]:
    summary = dict(res.get("best_summary", {}))
    for k, v in res.get("best", {}).items():
        summary.setdefault(k, v)

    out: Dict[str, Any] = {
        "case": case,
        "prev_score": float(res.get("score", 0.0)),
        "dp_score": dp_score(summary),
        "elapsed_sec": float(res.get("elapsed_sec", 0.0)),
    }
    out.update({k: float(v) for k, v in full_params(overrides).items()})
    for k in [
        "sigma_mult", "strength", "capture_rate", "distance_progress",
        "signed_alignment", "tangent_ratio", "mean_endpoint_dist", "endpoint_dist",
        "composite_score", "score", "dl_scalar_score", "dn_score", "do_score",
    ]:
        if k in summary:
            try:
                out[k] = float(summary[k])
            except Exception:
                out[k] = summary[k]
    out["overrides"] = json.dumps(clean_overrides(overrides), sort_keys=True)
    return out


def parse_overrides_from_row(row: pd.Series) -> Optional[Dict[str, float]]:
    if "overrides" in row and isinstance(row["overrides"], str) and row["overrides"].strip():
        try:
            val = json.loads(row["overrides"])
            if isinstance(val, dict):
                return {k: clamp_param(k, float(val[k])) for k in PARAMS if k in val}
        except Exception:
            pass
    vals: Dict[str, float] = {}
    for k in PARAMS:
        if k in row and pd.notna(row[k]):
            try:
                vals[k] = clamp_param(k, float(row[k]))
            except Exception:
                pass
    return vals if vals else None


def add_candidate(acc: List[Tuple[str, Dict[str, float]]], seen: set, name: str, ov: Dict[str, Any]) -> None:
    clean = {k: clamp_param(k, float(v)) for k, v in full_params(ov).items() if k in PARAMS}
    key = tuple(round(clean[k], 4) for k in PARAMS)
    if key in seen:
        return
    seen.add(key)
    acc.append((name[:130], clean))


def read_csv_candidates() -> List[Tuple[str, Dict[str, float]]]:
    sources = [
        ROOT / "phase26do_case_results.csv",
        ROOT / "phase26do_pareto_front.csv",
        ROOT / "phase26dn_base_results.csv",
        ROOT / "phase26dm_case_results.csv",
        ROOT / "phase26dl_case_results.csv",
    ]
    out: List[Tuple[str, Dict[str, float]]] = []
    seen = set()
    for path in sources:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        score_cols = [c for c in ["dp_score", "do_score", "dn_score", "dl_scalar_score", "score", "prev_score"] if c in df.columns]
        if score_cols:
            df = df.sort_values(score_cols[0], ascending=False)
        for i, row in df.head(24).iterrows():
            ov = parse_overrides_from_row(row)
            if ov is None:
                continue
            src = path.stem.replace("phase26", "p26")
            nm = str(row.get("case", f"row_{i}"))
            add_candidate(out, seen, f"prior_{src}_{nm}", ov)
    return out


def build_candidates() -> List[Tuple[str, Dict[str, float]]]:
    candidates: List[Tuple[str, Dict[str, float]]] = []
    seen: set = set()
    for name, ov in KNEE_CENTERS:
        add_candidate(candidates, seen, name, ov)

    # Import winners/pareto points, then generate structured local probes.
    priors = read_csv_candidates()
    for name, ov in priors[:32]:
        add_candidate(candidates, seen, name, ov)

    # One-factor micro-sweeps around the best inferred knee zone.
    center = full_params(KNEE_CENTERS[1][1])
    for r in [0.64, 0.66, 0.68, 0.70, 0.72, 0.74]:
        ov = dict(center); ov["BOWL_RADIUS_FRAC"] = r
        add_candidate(candidates, seen, f"dp_radius_micro_{r:.2f}", ov)
    for seat in [0.30, 0.35, 0.38, 0.40, 0.45, 0.50]:
        ov = dict(center); ov["BOWL_SEAT_AXIS_GAIN"] = seat
        add_candidate(candidates, seen, f"dp_seat_micro_{seat:.2f}", ov)
    for blend in [1.00, 1.02, 1.04, 1.06, 1.08, 1.12]:
        ov = dict(center); ov["BOWL_DIRECTIONAL_BLEND"] = blend
        add_candidate(candidates, seen, f"dp_blend_micro_{blend:.2f}", ov)
    for cap in [1.00, 1.02, 1.04, 1.06, 1.08, 1.12, 1.16]:
        ov = dict(center); ov["BOWL_NORM_CAP_MULT"] = cap
        add_candidate(candidates, seen, f"dp_cap_micro_{cap:.2f}", ov)
    for shell in [0.10, 0.12, 0.14, 0.16]:
        ov = dict(center); ov["BOWL_SHELL_RADIAL_GAIN"] = shell
        add_candidate(candidates, seen, f"dp_shell_micro_{shell:.2f}", ov)
    for tk in [0.76, 0.80, 0.84, 0.88, 0.92, 0.96]:
        ov = dict(center); ov["BOWL_TANGENT_KILL"] = tk
        add_candidate(candidates, seen, f"dp_tk_micro_{tk:.2f}", ov)

    # Controlled coupled candidates: not a large grid, just plausible knees.
    for r in [0.66, 0.70, 0.74]:
        for seat in [0.35, 0.40, 0.45]:
            for blend, cap, shell, tk in [
                (1.04, 1.04, 0.12, 0.88),
                (1.04, 1.08, 0.12, 0.84),
                (1.08, 1.08, 0.14, 0.80),
                (1.12, 1.12, 0.14, 0.76),
            ]:
                ov = {
                    "BOWL_RADIUS_FRAC": r,
                    "BOWL_SEAT_AXIS_GAIN": seat,
                    "BOWL_DIRECTIONAL_BLEND": blend,
                    "BOWL_NORM_CAP_MULT": cap,
                    "BOWL_SHELL_RADIAL_GAIN": shell,
                    "BOWL_TANGENT_KILL": tk,
                }
                add_candidate(candidates, seen, f"dp_coupled_r{r:.2f}_s{seat:.2f}_b{blend:.2f}_c{cap:.2f}_sh{shell:.2f}_tk{tk:.2f}", ov)

    # Small random cloud around imported Pareto/winner candidates. This is where 26DO
    # found several unexpectedly good low-tangent cases, but keep it compact.
    rng = random.Random(2604)
    cloud_centers = [ov for _, ov in (priors[:10] + KNEE_CENTERS)]
    for j in range(36):
        c = dict(rng.choice(cloud_centers))
        ov = {
            "BOWL_RADIUS_FRAC": c.get("BOWL_RADIUS_FRAC", 0.66) + rng.uniform(-0.025, 0.025),
            "BOWL_SEAT_AXIS_GAIN": c.get("BOWL_SEAT_AXIS_GAIN", 0.35) + rng.uniform(-0.035, 0.035),
            "BOWL_DIRECTIONAL_BLEND": c.get("BOWL_DIRECTIONAL_BLEND", 1.04) + rng.uniform(-0.030, 0.030),
            "BOWL_NORM_CAP_MULT": c.get("BOWL_NORM_CAP_MULT", 1.04) + rng.uniform(-0.035, 0.035),
            "BOWL_SHELL_RADIAL_GAIN": c.get("BOWL_SHELL_RADIAL_GAIN", 0.12) + rng.uniform(-0.015, 0.015),
            "BOWL_TANGENT_KILL": c.get("BOWL_TANGENT_KILL", 0.88) + rng.uniform(-0.045, 0.045),
        }
        add_candidate(candidates, seen, f"dp_random_knee_cloud_{j:02d}", ov)

    return candidates


def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    # Maximize capture/progress/alignment/dp_score. Minimize tangent/end.
    work = df.copy()
    for c in ["capture_rate", "distance_progress", "signed_alignment", "dp_score"]:
        if c not in work.columns:
            work[c] = 0.0
    if "tangent_ratio" not in work.columns:
        work["tangent_ratio"] = 999.0
    if "mean_endpoint_dist" not in work.columns:
        work["mean_endpoint_dist"] = 0.0
    vals = work[["capture_rate", "distance_progress", "signed_alignment", "dp_score", "tangent_ratio", "mean_endpoint_dist"]].to_numpy(float)
    keep: List[int] = []
    for i, a in enumerate(vals):
        dominated = False
        for j, b in enumerate(vals):
            if i == j:
                continue
            better_or_equal = (b[0] >= a[0] and b[1] >= a[1] and b[2] >= a[2] and b[3] >= a[3] and b[4] <= a[4] and b[5] <= a[5])
            strictly = (b[0] > a[0] or b[1] > a[1] or b[2] > a[2] or b[3] > a[3] or b[4] < a[4] or b[5] < a[5])
            if better_or_equal and strictly:
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return work.iloc[keep].sort_values("dp_score", ascending=False)


def jitter_variants(base: Dict[str, float], n: int = 5, seed: int = 2616) -> List[Dict[str, float]]:
    rng = random.Random(seed)
    variants: List[Dict[str, float]] = []
    for _ in range(n):
        variants.append({
            "BOWL_RADIUS_FRAC": clamp_param("BOWL_RADIUS_FRAC", base["BOWL_RADIUS_FRAC"] + rng.uniform(-0.012, 0.012)),
            "BOWL_SEAT_AXIS_GAIN": clamp_param("BOWL_SEAT_AXIS_GAIN", base["BOWL_SEAT_AXIS_GAIN"] + rng.uniform(-0.018, 0.018)),
            "BOWL_DIRECTIONAL_BLEND": clamp_param("BOWL_DIRECTIONAL_BLEND", base["BOWL_DIRECTIONAL_BLEND"] + rng.uniform(-0.018, 0.018)),
            "BOWL_NORM_CAP_MULT": clamp_param("BOWL_NORM_CAP_MULT", base["BOWL_NORM_CAP_MULT"] + rng.uniform(-0.018, 0.018)),
            "BOWL_SHELL_RADIAL_GAIN": clamp_param("BOWL_SHELL_RADIAL_GAIN", base["BOWL_SHELL_RADIAL_GAIN"] + rng.uniform(-0.008, 0.008)),
            "BOWL_TANGENT_KILL": clamp_param("BOWL_TANGENT_KILL", base["BOWL_TANGENT_KILL"] + rng.uniform(-0.025, 0.025)),
        })
    return variants


def stability_audit(df: pd.DataFrame, basins: Any, top_n: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    jit_rows: List[Dict[str, Any]] = []
    top = df.sort_values("dp_score", ascending=False).head(top_n)
    for idx, row in top.iterrows():
        base_ov = {k: float(row[k]) for k in PARAMS if k in row and pd.notna(row[k])}
        scores: List[float] = []
        captures: List[float] = []
        tangents: List[float] = []
        for j, ov in enumerate(jitter_variants(base_ov, n=5, seed=2700 + int(idx))):
            nm = f"jitter_{j:02d}_{str(row['case'])[:70]}"
            try:
                res = PREV.PREV.eval_case(nm, ov, basins) if hasattr(PREV, "PREV") else PREV.eval_case(nm, ov, basins)
                cr = compact_result(nm, ov, res)
            except Exception as e:
                cr = {"case": nm, "dp_score": float("nan"), "error": repr(e), **ov}
            cr["parent_case"] = row["case"]
            jit_rows.append(cr)
            if math.isfinite(float(cr.get("dp_score", float("nan")))):
                scores.append(float(cr["dp_score"]))
                captures.append(float(cr.get("capture_rate", 0.0)))
                tangents.append(float(cr.get("tangent_ratio", 999.0)))
        if scores:
            worst = min(scores)
            mean = float(np.mean(scores))
            std = float(np.std(scores))
            # Stability intentionally values worst-case more than average.
            stability = 0.66 * worst + 0.24 * mean - 0.10 * std
            rows.append({
                "case": row["case"],
                "base_dp_score": float(row["dp_score"]),
                "stability_score": float(stability),
                "worst_jitter_dp_score": float(worst),
                "mean_jitter_dp_score": mean,
                "std_jitter_dp_score": std,
                "worst_capture_rate": float(min(captures)) if captures else float("nan"),
                "worst_tangent_ratio": float(max(tangents)) if tangents else float("nan"),
                **{k: float(row[k]) for k in PARAMS if k in row},
            })
    return pd.DataFrame(rows).sort_values("stability_score", ascending=False), pd.DataFrame(jit_rows)


def plot_top(df: pd.DataFrame) -> None:
    top = df.sort_values("dp_score", ascending=False).head(34).iloc[::-1]
    plt.figure(figsize=(14, max(8, 0.48 * len(top))))
    plt.barh(top["case"], top["dp_score"])
    plt.xlabel("DP knee-lock score")
    plt.title("26DP Pareto-knee lock-in scores")
    plt.tight_layout()
    plt.savefig(ROOT / "phase26dp_top_scores.png", dpi=150)
    plt.close()


def plot_pareto(df: pd.DataFrame, pf: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 7))
    x = df.get("capture_rate", pd.Series(np.zeros(len(df))))
    y = df.get("tangent_ratio", pd.Series(np.zeros(len(df))))
    c = df.get("dp_score", pd.Series(np.zeros(len(df))))
    sc = plt.scatter(x, y, c=c, s=58, alpha=0.76)
    if len(pf):
        plt.scatter(pf.get("capture_rate", []), pf.get("tangent_ratio", []), facecolors="none", edgecolors="black", s=150, linewidths=1.6, label="Pareto front")
        plt.legend()
    plt.axhline(1.50, linewidth=1.0, alpha=0.45)
    plt.axvline(0.30, linewidth=1.0, alpha=0.45)
    plt.colorbar(sc, label="DP score")
    plt.xlabel("capture_rate")
    plt.ylabel("tangent/radial ratio")
    plt.title("26DP capture vs tangent drift")
    plt.grid(True, alpha=0.28)
    plt.tight_layout()
    plt.savefig(ROOT / "phase26dp_capture_vs_tangent_pareto.png", dpi=150)
    plt.close()


def plot_stability(stab: pd.DataFrame) -> None:
    if stab.empty:
        return
    top = stab.sort_values("stability_score", ascending=False).head(12).iloc[::-1]
    y = np.arange(len(top))
    plt.figure(figsize=(13, max(6, 0.56 * len(top))))
    plt.barh(y - 0.18, top["stability_score"], height=0.34, label="stability_score")
    plt.barh(y + 0.18, top["worst_jitter_dp_score"], height=0.34, label="worst_jitter_DP")
    plt.yticks(y, top["case"])
    plt.xlabel("score")
    plt.title("26DP jitter stability audit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "phase26dp_stability_scores.png", dpi=150)
    plt.close()


def inherited_plots(z2: Any, rel_ids: Any, basins: Any, best_row: pd.Series) -> List[str]:
    errors: List[str] = []
    # DO exposes maybe_inherited_plots, which calls DN/DM inherited AP plot emitters.
    if hasattr(PREV, "maybe_inherited_plots"):
        try:
            errors = PREV.maybe_inherited_plots(z2, rel_ids, basins, best_row)
        except Exception as e:
            errors.append(repr(e))
            print(f"[{PHASE}] inherited plot warning: {e}")
    return errors


def main() -> Dict[str, Any]:
    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] Output root: {ROOT}")

    z2, rel_ids, basins = PREV.PREV.build_basins() if hasattr(PREV, "PREV") else PREV.build_basins()
    print(f"[{PHASE}] Built basins: {len(basins)}")

    candidates = build_candidates()
    print(f"[{PHASE}] Candidates: {len(candidates)}")

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    eval_case = PREV.PREV.eval_case if hasattr(PREV, "PREV") else PREV.eval_case
    for i, (name, ov) in enumerate(candidates, 1):
        try:
            res = eval_case(name, ov, basins)
            cr = compact_result(name, ov, res)
            rows.append(cr)
            if i <= 5 or i % 12 == 0:
                print(
                    f"[{PHASE}] {i:03d}/{len(candidates)} {name[:70]:70s} "
                    f"DP={cr['dp_score']:.4f} cap={cr.get('capture_rate', float('nan')):.3f} "
                    f"tan={cr.get('tangent_ratio', float('nan')):.3f}"
                )
        except Exception as e:
            print(f"[{PHASE}] ERROR {name}: {e}")
            rows.append({"case": name, "error": repr(e), "dp_score": float("nan"), **full_params(ov)})

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "phase26dp_case_results.csv", index=False)
    df_ok = df[pd.to_numeric(df.get("dp_score", pd.Series(dtype=float)), errors="coerce").notna()].copy()
    df_ok = df_ok.sort_values("dp_score", ascending=False)

    pf = pareto_front(df_ok) if len(df_ok) else pd.DataFrame()
    pf.to_csv(ROOT / "phase26dp_pareto_front.csv", index=False)

    stab, jit = stability_audit(df_ok, basins, top_n=9) if len(df_ok) else (pd.DataFrame(), pd.DataFrame())
    stab.to_csv(ROOT / "phase26dp_stability_summary.csv", index=False)
    jit.to_csv(ROOT / "phase26dp_jitter_results.csv", index=False)

    if len(df_ok):
        plot_top(df_ok)
        plot_pareto(df_ok, pf)
    plot_stability(stab)

    # Prefer the most stable deployable row if available; otherwise use best DP score.
    best_source = "dp_score"
    if not stab.empty:
        best_case = stab.iloc[0]["case"]
        best_row = df_ok[df_ok["case"] == best_case].iloc[0]
        best_source = "stability_score"
    elif len(df_ok):
        best_row = df_ok.iloc[0]
    else:
        best_row = pd.Series(dtype=float)

    plot_errors: List[str] = []
    if len(best_row):
        plot_errors = inherited_plots(z2, rel_ids, basins, best_row)

    summary = {
        "phase": PHASE,
        "title": TITLE,
        "prev_path": str(PREV_PATH),
        "root": str(ROOT),
        "num_candidates": int(len(candidates)),
        "num_success": int(len(df_ok)),
        "num_pareto": int(len(pf)),
        "elapsed_sec": float(time.time() - t0),
        "best_source": best_source,
        "best_case": None if not len(best_row) else str(best_row.get("case")),
        "best_dp_score": None if not len(best_row) else float(best_row.get("dp_score", float("nan"))),
        "best_capture_rate": None if not len(best_row) else float(best_row.get("capture_rate", float("nan"))),
        "best_tangent_ratio": None if not len(best_row) else float(best_row.get("tangent_ratio", float("nan"))),
        "best_overrides": None if not len(best_row) else {k: float(best_row[k]) for k in PARAMS if k in best_row and pd.notna(best_row[k])},
        "top5": df_ok.head(5).to_dict(orient="records") if len(df_ok) else [],
        "stability_top5": stab.head(5).to_dict(orient="records") if not stab.empty else [],
        "plot_errors": plot_errors,
        "outputs": [
            "phase26dp_case_results.csv",
            "phase26dp_pareto_front.csv",
            "phase26dp_stability_summary.csv",
            "phase26dp_jitter_results.csv",
            "phase26dp_summary.json",
            "phase26dp_top_scores.png",
            "phase26dp_capture_vs_tangent_pareto.png",
            "phase26dp_stability_scores.png",
        ],
    }
    with open(ROOT / "phase26dp_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[{PHASE}] Done in {summary['elapsed_sec']:.1f}s")
    if len(best_row):
        print(
            f"[{PHASE}] BEST ({best_source}): {summary['best_case']} | "
            f"DP={summary['best_dp_score']:.4f} cap={summary['best_capture_rate']:.3f} "
            f"tan={summary['best_tangent_ratio']:.3f}"
        )
        print(f"[{PHASE}] BEST overrides: {summary['best_overrides']}")
    print(f"[{PHASE}] Wrote summary: {ROOT / 'phase26dp_summary.json'}")
    return summary


if __name__ == "__main__":
    main()
