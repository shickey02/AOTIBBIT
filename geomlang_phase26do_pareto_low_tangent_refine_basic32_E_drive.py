#!/usr/bin/env python3
r"""
Phase 26DO — Pareto low-tangent refinement audit for the 26D* shell-bridge basin field.

Goal
----
DN showed that the highest scalar winners are fairly stable, but many still buy their
capture rate with excess tangential drift. 26DO changes the search target:

  * keep capture/radial progress alive,
  * punish tangent/radial drift much harder,
  * report the Pareto front instead of only a single scalar champion,
  * jitter-audit the best low-tangent candidates.

This script is designed to be pasted/run from E:\BBIT. It reuses the previous 26DN
helpers when available, which in turn reuse the 26DM/26DL/26AP evaluation stack.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHASE = "26DO"
TITLE = "Pareto low-tangent refinement audit"
ROOT = Path(r"E:\BBIT\outputs_basic32")
if not ROOT.exists():
    ROOT = Path.cwd()

PREV_FILENAMES = [
    "geomlang_phase26dn_elite_stability_low_tangent_audit_basic32_E_drive.py",
    "geomlang_phase26dm_dl_ridge_extrapolation_low_tangent_probe_basic32_E_drive.py",
]

PARAMS = [
    "BOWL_RADIUS_FRAC",
    "BOWL_SEAT_AXIS_GAIN",
    "BOWL_DIRECTIONAL_BLEND",
    "BOWL_NORM_CAP_MULT",
    "BOWL_SHELL_RADIAL_GAIN",
    "BOWL_TANGENT_KILL",
]

BASELINE: Dict[str, float] = {
    "BOWL_RADIUS_FRAC": 0.70,
    "BOWL_SEAT_AXIS_GAIN": 0.40,
    "BOWL_DIRECTIONAL_BLEND": 1.04,
    "BOWL_NORM_CAP_MULT": 1.08,
    "BOWL_SHELL_RADIAL_GAIN": 0.12,
    "BOWL_TANGENT_KILL": 0.80,
}

# A compact hand-written set based on the 26DM/26DN charts. These are not assumed to
# be final winners; they are anchors for a new low-tangent objective.
ANCHORS: List[Tuple[str, Dict[str, float]]] = [
    ("do_base_current_constants", {}),
    ("do_dn_middle_stable", {"BOWL_RADIUS_FRAC": 0.70, "BOWL_SEAT_AXIS_GAIN": 0.40, "BOWL_DIRECTIONAL_BLEND": 1.04, "BOWL_NORM_CAP_MULT": 1.08, "BOWL_SHELL_RADIAL_GAIN": 0.12, "BOWL_TANGENT_KILL": 0.80}),
    ("do_dn_middle_shell", {"BOWL_RADIUS_FRAC": 0.70, "BOWL_SEAT_AXIS_GAIN": 0.40, "BOWL_DIRECTIONAL_BLEND": 1.04, "BOWL_NORM_CAP_MULT": 1.08, "BOWL_SHELL_RADIAL_GAIN": 0.14, "BOWL_TANGENT_KILL": 0.80}),
    ("do_low_tangent_tk092", {"BOWL_TANGENT_KILL": 0.92}),
    ("do_low_tangent_tk094", {"BOWL_TANGENT_KILL": 0.94}),
    ("do_low_tangent_shell016", {"BOWL_SHELL_RADIAL_GAIN": 0.16}),
    ("do_low_tangent_cap104", {"BOWL_NORM_CAP_MULT": 1.04}),
    ("do_low_tangent_blend104", {"BOWL_DIRECTIONAL_BLEND": 1.04}),
    ("do_dm_top_tk072", {"BOWL_RADIUS_FRAC": 0.78, "BOWL_SEAT_AXIS_GAIN": 0.50, "BOWL_DIRECTIONAL_BLEND": 1.12, "BOWL_NORM_CAP_MULT": 1.16, "BOWL_SHELL_RADIAL_GAIN": 0.16, "BOWL_TANGENT_KILL": 0.72}),
    ("do_dm_top_tk076", {"BOWL_RADIUS_FRAC": 0.78, "BOWL_SEAT_AXIS_GAIN": 0.50, "BOWL_DIRECTIONAL_BLEND": 1.12, "BOWL_NORM_CAP_MULT": 1.16, "BOWL_SHELL_RADIAL_GAIN": 0.16, "BOWL_TANGENT_KILL": 0.76}),
    ("do_dm_balanced_r066", {"BOWL_RADIUS_FRAC": 0.66, "BOWL_SEAT_AXIS_GAIN": 0.35, "BOWL_DIRECTIONAL_BLEND": 1.04, "BOWL_NORM_CAP_MULT": 1.00, "BOWL_SHELL_RADIAL_GAIN": 0.10, "BOWL_TANGENT_KILL": 0.90}),
    ("do_dm_stable_r066_shell10", {"BOWL_RADIUS_FRAC": 0.66, "BOWL_SEAT_AXIS_GAIN": 0.35, "BOWL_DIRECTIONAL_BLEND": 1.00, "BOWL_NORM_CAP_MULT": 1.00, "BOWL_SHELL_RADIAL_GAIN": 0.10, "BOWL_TANGENT_KILL": 0.92}),
]


def import_previous() -> Any:
    roots = [Path.cwd(), Path(__file__).resolve().parent, Path(r"E:\BBIT\bbit_geomlang"), Path(r"E:\BBIT")]
    tried: List[str] = []
    for root in roots:
        for fn in PREV_FILENAMES:
            p = root / fn
            tried.append(str(p))
            if p.exists():
                spec = importlib.util.spec_from_file_location("phase26_prev", str(p))
                if spec is None or spec.loader is None:
                    continue
                mod = importlib.util.module_from_spec(spec)
                import sys
                sys.modules[spec.name] = mod
                spec.loader.exec_module(mod)  # type: ignore[attr-defined]
                helper = mod
                if not (hasattr(helper, "build_basins") and hasattr(helper, "eval_case")):
                    helper = getattr(mod, "DM", helper)
                if hasattr(helper, "build_basins") and hasattr(helper, "eval_case"):
                    print(f"[{PHASE}] Loaded previous helpers from: {p}")
                    if helper is not mod:
                        print(f"[{PHASE}] Using nested helper API from: {type(helper).__name__}")
                    return helper
                print(f"[{PHASE}] Skipping helper without build_basins/eval_case: {p}")
    raise FileNotFoundError("Could not find a previous 26DN/26DM helper. Tried:\n" + "\n".join(tried))


PREV = import_previous()


def clean_overrides(d: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in d.items():
        if k in PARAMS:
            try:
                fv = float(v)
                if math.isfinite(fv):
                    out[k] = fv
            except Exception:
                pass
    return out


def full_params(overrides: Dict[str, Any]) -> Dict[str, float]:
    out = dict(BASELINE)
    # Pull real current constants from prior module where possible.
    for k in PARAMS:
        for obj in [PREV, getattr(PREV, "BASE", None), getattr(PREV, "PREV", None)]:
            if obj is not None and hasattr(obj, k):
                try:
                    out[k] = float(getattr(obj, k))
                    break
                except Exception:
                    pass
    out.update(clean_overrides(overrides))
    return out


def parse_overrides_cell(x: Any) -> Dict[str, float]:
    if isinstance(x, dict):
        return clean_overrides(x)
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return {}
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return {}
    try:
        return clean_overrides(json.loads(s))
    except Exception:
        pass
    try:
        return clean_overrides(ast.literal_eval(s))
    except Exception:
        return {}


def read_prior_candidates(limit_each: int = 16) -> List[Tuple[str, Dict[str, float]]]:
    out: List[Tuple[str, Dict[str, float]]] = []
    for stem in [
        "phase26dn_base_results.csv",
        "phase26dn_stability_summary.csv",
        "phase26dm_case_results.csv",
        "phase26dl_case_results.csv",
        "phase26dk_case_results.csv",
    ]:
        for p in [ROOT / stem, Path.cwd() / stem]:
            if not p.exists():
                continue
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            sort_col = None
            for c in ["dn_score", "stability_score", "score", "dl_scalar_score", "composite_score"]:
                if c in df.columns:
                    sort_col = c
                    break
            if sort_col is not None:
                df = df.sort_values(sort_col, ascending=False)
            n = 0
            for _, row in df.head(limit_each).iterrows():
                ov: Dict[str, float] = {}
                if "overrides" in row:
                    ov.update(parse_overrides_cell(row.get("overrides")))
                # Some prior CSVs expanded parameters into columns.
                for k in PARAMS:
                    if k in row and not pd.isna(row[k]):
                        try:
                            ov[k] = float(row[k])
                        except Exception:
                            pass
                if not ov:
                    continue
                name = str(row.get("case", row.get("name", stem.replace(".csv", ""))))
                out.append((f"prior_{stem.replace('.csv','')}_{n:02d}_{name}", clean_overrides(ov)))
                n += 1
            break
    return out


def mutate_candidates(seeds: Sequence[Tuple[str, Dict[str, float]]]) -> List[Tuple[str, Dict[str, float]]]:
    candidates: List[Tuple[str, Dict[str, float]]] = []
    seen = set()

    def add(name: str, ov: Dict[str, Any]) -> None:
        params = full_params(ov)
        # Clamp to the sane local region explored in 26DM/26DN.
        params["BOWL_RADIUS_FRAC"] = min(0.82, max(0.62, params["BOWL_RADIUS_FRAC"]))
        params["BOWL_SEAT_AXIS_GAIN"] = min(0.55, max(0.32, params["BOWL_SEAT_AXIS_GAIN"]))
        params["BOWL_DIRECTIONAL_BLEND"] = min(1.16, max(0.92, params["BOWL_DIRECTIONAL_BLEND"]))
        params["BOWL_NORM_CAP_MULT"] = min(1.20, max(0.96, params["BOWL_NORM_CAP_MULT"]))
        params["BOWL_SHELL_RADIAL_GAIN"] = min(0.18, max(0.08, params["BOWL_SHELL_RADIAL_GAIN"]))
        params["BOWL_TANGENT_KILL"] = min(0.98, max(0.70, params["BOWL_TANGENT_KILL"]))
        key = tuple(round(params[k], 4) for k in PARAMS)
        if key not in seen:
            seen.add(key)
            candidates.append((name[:140], params))

    for name, ov in seeds:
        add(name, ov)

    # Targeted low-tangent perturbations: increase tangent-kill, but also soften cap/shell
    # so the field does not become a sideways slingshot.
    for i, (name, ov) in enumerate(seeds[:24]):
        base = full_params(ov)
        for tk in [0.84, 0.88, 0.92, 0.96]:
            add(f"do_tk_sweep_{i:02d}_{tk:.2f}_{name}", {**base, "BOWL_TANGENT_KILL": tk})
        for cap in [1.00, 1.04, 1.08, 1.12]:
            add(f"do_cap_sweep_{i:02d}_{cap:.2f}_{name}", {**base, "BOWL_NORM_CAP_MULT": cap})
        for sh in [0.10, 0.12, 0.14, 0.16]:
            add(f"do_shell_sweep_{i:02d}_{sh:.2f}_{name}", {**base, "BOWL_SHELL_RADIAL_GAIN": sh})
        # Coupled moves most likely to lower tangential drift without destroying capture.
        for tk, cap, sh, blend in [
            (0.88, 1.04, 0.12, 1.00),
            (0.92, 1.04, 0.12, 1.00),
            (0.92, 1.00, 0.10, 1.04),
            (0.96, 1.00, 0.10, 1.04),
            (0.88, 1.08, 0.14, 1.04),
        ]:
            add(f"do_coupled_{i:02d}_tk{tk:.2f}_cap{cap:.2f}_sh{sh:.2f}_b{blend:.2f}", {
                **base,
                "BOWL_TANGENT_KILL": tk,
                "BOWL_NORM_CAP_MULT": cap,
                "BOWL_SHELL_RADIAL_GAIN": sh,
                "BOWL_DIRECTIONAL_BLEND": blend,
            })

    # Small deterministic random cloud around the best-looking DN/DM zones.
    rng = random.Random(2604)
    centers = [full_params(ov) for _, ov in seeds[:10]] + [full_params(dict(ANCHORS[1][1]))]
    for j in range(36):
        c = rng.choice(centers)
        ov = {
            "BOWL_RADIUS_FRAC": c["BOWL_RADIUS_FRAC"] + rng.choice([-1, 0, 1]) * rng.uniform(0.00, 0.025),
            "BOWL_SEAT_AXIS_GAIN": c["BOWL_SEAT_AXIS_GAIN"] + rng.choice([-1, 0, 1]) * rng.uniform(0.00, 0.035),
            "BOWL_DIRECTIONAL_BLEND": c["BOWL_DIRECTIONAL_BLEND"] + rng.choice([-1, 0, 1]) * rng.uniform(0.00, 0.035),
            "BOWL_NORM_CAP_MULT": c["BOWL_NORM_CAP_MULT"] + rng.choice([-1, 0, 1]) * rng.uniform(0.00, 0.035),
            "BOWL_SHELL_RADIAL_GAIN": c["BOWL_SHELL_RADIAL_GAIN"] + rng.choice([-1, 0, 1]) * rng.uniform(0.00, 0.018),
            "BOWL_TANGENT_KILL": c["BOWL_TANGENT_KILL"] + rng.choice([-1, 0, 1]) * rng.uniform(0.00, 0.055),
        }
        add(f"do_random_low_tangent_cloud_{j:02d}", ov)

    return candidates


def metric(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    # Accept either compacted keys or raw prior keys.
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


def do_score(summary: Dict[str, Any]) -> float:
    cap = metric(summary, "capture_rate")
    prog = metric(summary, "distance_progress")
    align = metric(summary, "signed_alignment")
    tan = max(0.0, metric(summary, "tangent_ratio"))
    end = max(0.0, metric(summary, "mean_endpoint_dist"))

    # Important: tangent is log-penalized plus gated. This prevents the old scalar score
    # from declaring a high-capture/high-sideways solution the best solution.
    score = 4.25 * cap + 2.25 * prog + 0.80 * align - 0.34 * math.log1p(tan) - 0.025 * end
    if cap < 0.22:
        score -= (0.22 - cap) * 2.0
    if tan > 4.0:
        score -= 0.18 * (tan - 4.0)
    if tan > 8.0:
        score -= 0.25 * (tan - 8.0)
    return float(score)


def compact_result(case: str, overrides: Dict[str, float], res: Dict[str, Any]) -> Dict[str, Any]:
    summary = dict(res.get("best_summary", {}))
    # If previous compact did not include these, try raw best.
    for k, v in res.get("best", {}).items():
        summary.setdefault(k, v)
    out: Dict[str, Any] = {
        "case": case,
        "prev_score": float(res.get("score", 0.0)),
        "do_score": do_score(summary),
        "elapsed_sec": float(res.get("elapsed_sec", 0.0)),
    }
    out.update({k: float(v) for k, v in full_params(overrides).items()})
    for k in ["sigma_mult", "strength", "capture_rate", "distance_progress", "signed_alignment", "tangent_ratio", "mean_endpoint_dist", "endpoint_dist", "composite_score", "score", "dl_scalar_score", "dn_score"]:
        if k in summary:
            try:
                out[k] = float(summary[k])
            except Exception:
                out[k] = summary[k]
    out["overrides"] = json.dumps(clean_overrides(overrides), sort_keys=True)
    return out


def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    # Maximize capture, progress, alignment, do_score. Minimize tangent and endpoint.
    cols = set(df.columns)
    work = df.copy()
    for c in ["capture_rate", "distance_progress", "signed_alignment", "do_score"]:
        if c not in cols:
            work[c] = 0.0
    if "tangent_ratio" not in cols:
        work["tangent_ratio"] = 999.0
    if "mean_endpoint_dist" not in cols:
        work["mean_endpoint_dist"] = 0.0
    values = work[["capture_rate", "distance_progress", "signed_alignment", "do_score", "tangent_ratio", "mean_endpoint_dist"]].to_numpy(float)
    keep = []
    for i, a in enumerate(values):
        dominated = False
        for j, b in enumerate(values):
            if i == j:
                continue
            better_or_equal = (
                b[0] >= a[0] and b[1] >= a[1] and b[2] >= a[2] and b[3] >= a[3]
                and b[4] <= a[4] and b[5] <= a[5]
            )
            strictly_better = (
                b[0] > a[0] or b[1] > a[1] or b[2] > a[2] or b[3] > a[3]
                or b[4] < a[4] or b[5] < a[5]
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        keep.append(not dominated)
    return work.loc[keep].sort_values("do_score", ascending=False)


def jitter_audit(top: pd.DataFrame, basins: Any, n_jitter: int = 5) -> pd.DataFrame:
    rng = random.Random(2615)
    rows: List[Dict[str, Any]] = []
    for _, r in top.iterrows():
        base_ov = {k: float(r[k]) for k in PARAMS if k in r and pd.notna(r[k])}
        scores: List[float] = []
        tangents: List[float] = []
        captures: List[float] = []
        for j in range(n_jitter):
            ov = dict(base_ov)
            ov["BOWL_RADIUS_FRAC"] += rng.uniform(-0.012, 0.012)
            ov["BOWL_SEAT_AXIS_GAIN"] += rng.uniform(-0.018, 0.018)
            ov["BOWL_DIRECTIONAL_BLEND"] += rng.uniform(-0.020, 0.020)
            ov["BOWL_NORM_CAP_MULT"] += rng.uniform(-0.020, 0.020)
            ov["BOWL_SHELL_RADIAL_GAIN"] += rng.uniform(-0.010, 0.010)
            ov["BOWL_TANGENT_KILL"] += rng.uniform(-0.025, 0.025)
            name = f"jitter_{j}_{str(r['case'])[:90]}"
            try:
                res = PREV.eval_case(name, ov, basins)
                cr = compact_result(name, ov, res)
                scores.append(float(cr["do_score"]))
                tangents.append(float(cr.get("tangent_ratio", 999.0)))
                captures.append(float(cr.get("capture_rate", 0.0)))
            except Exception as e:
                print(f"[{PHASE}] jitter warning for {r['case']}: {e}")
        if scores:
            base_score = float(r["do_score"])
            worst = min(scores)
            rows.append({
                "case": r["case"],
                "base_do_score": base_score,
                "jitter_mean_do_score": float(np.mean(scores)),
                "jitter_std_do_score": float(np.std(scores)),
                "worst_jitter_do_score": float(worst),
                "mean_jitter_capture": float(np.mean(captures)),
                "mean_jitter_tangent": float(np.mean(tangents)),
                "max_drop": float(base_score - worst),
                "stability_score": float(np.mean(scores) - 0.50 * np.std(scores) - 0.75 * max(0.0, base_score - worst)),
            })
    return pd.DataFrame(rows).sort_values("stability_score", ascending=False) if rows else pd.DataFrame()


def plot_scoreboard(df: pd.DataFrame) -> None:
    top = df.sort_values("do_score", ascending=False).head(30).iloc[::-1]
    plt.figure(figsize=(13, max(8, 0.38 * len(top))))
    plt.barh(top["case"], top["do_score"])
    plt.xlabel("DO low-tangent score")
    plt.title("26DO low-tangent Pareto-refine scores")
    plt.tight_layout()
    plt.savefig(ROOT / "phase26do_top_scores.png", dpi=150)
    plt.close()


def plot_pareto(df: pd.DataFrame, pf: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 7))
    x = df.get("capture_rate", pd.Series(np.zeros(len(df))))
    y = df.get("tangent_ratio", pd.Series(np.zeros(len(df))))
    c = df.get("do_score", pd.Series(np.zeros(len(df))))
    sc = plt.scatter(x, y, c=c, s=60, alpha=0.78)
    if len(pf):
        plt.scatter(pf.get("capture_rate", []), pf.get("tangent_ratio", []), facecolors="none", edgecolors="black", s=140, linewidths=1.5, label="Pareto front")
        plt.legend()
    plt.colorbar(sc, label="DO score")
    plt.xlabel("capture_rate")
    plt.ylabel("tangent/radial ratio")
    plt.title("26DO capture vs tangent drift")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROOT / "phase26do_capture_vs_tangent_pareto.png", dpi=150)
    plt.close()


def plot_stability(stab: pd.DataFrame) -> None:
    if stab.empty:
        return
    top = stab.sort_values("stability_score", ascending=False).head(12).iloc[::-1]
    y = np.arange(len(top))
    plt.figure(figsize=(13, max(6, 0.50 * len(top))))
    plt.barh(y - 0.18, top["stability_score"], height=0.34, label="stability_score")
    plt.barh(y + 0.18, top["worst_jitter_do_score"], height=0.34, label="worst_jitter_DO")
    plt.yticks(y, top["case"])
    plt.xlabel("score")
    plt.title("26DO jitter stability audit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "phase26do_stability_scores.png", dpi=150)
    plt.close()


def maybe_inherited_plots(z2: Any, rel_ids: Any, basins: Any, best_row: pd.Series) -> List[str]:
    errors: List[str] = []
    if not hasattr(PREV, "emit_inherited_plots"):
        return errors
    try:
        ov = {k: float(best_row[k]) for k in PARAMS if k in best_row and pd.notna(best_row[k])}
        fake = {"overrides": ov}
        errors = PREV.emit_inherited_plots(z2, rel_ids, basins, fake)
    except Exception as e:
        errors.append(repr(e))
        print(f"[{PHASE}] inherited plot warning: {e}")
    return errors


def main() -> Dict[str, Any]:
    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] Output root: {ROOT}")

    z2, rel_ids, basins = PREV.build_basins()
    print(f"[{PHASE}] Built basins: {len(basins)}")

    seeds = ANCHORS + read_prior_candidates(limit_each=14)
    candidates = mutate_candidates(seeds)
    print(f"[{PHASE}] Seeds: {len(seeds)} | candidates after mutation/dedupe: {len(candidates)}")

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, (name, ov) in enumerate(candidates, 1):
        try:
            res = PREV.eval_case(name, ov, basins)
            cr = compact_result(name, ov, res)
            rows.append(cr)
            if i <= 5 or i % 10 == 0:
                print(
                    f"[{PHASE}] {i:03d}/{len(candidates)} {name[:68]:68s} "
                    f"DO={cr['do_score']:.4f} cap={cr.get('capture_rate', float('nan')):.3f} "
                    f"tan={cr.get('tangent_ratio', float('nan')):.3f} prev={cr.get('prev_score', float('nan')):.4f}"
                )
        except Exception as e:
            print(f"[{PHASE}] eval warning for {name}: {e}")

    if not rows:
        raise RuntimeError("No successful 26DO candidate evaluations.")

    df = pd.DataFrame(rows).sort_values("do_score", ascending=False)
    pf = pareto_front(df)
    top_for_stability = pd.concat([df.head(8), pf.head(8)]).drop_duplicates(subset=["case"]).head(10)
    stab = jitter_audit(top_for_stability, basins, n_jitter=5)

    df.to_csv(ROOT / "phase26do_case_results.csv", index=False)
    pf.to_csv(ROOT / "phase26do_pareto_front.csv", index=False)
    if not stab.empty:
        stab.to_csv(ROOT / "phase26do_stability_summary.csv", index=False)

    plot_scoreboard(df)
    plot_pareto(df, pf)
    plot_stability(stab)

    best = df.iloc[0].to_dict()
    plot_errors = maybe_inherited_plots(z2, rel_ids, basins, df.iloc[0])

    summary = {
        "phase": PHASE,
        "title": TITLE,
        "elapsed_sec": time.time() - t0,
        "n_candidates": int(len(candidates)),
        "n_success": int(len(df)),
        "n_pareto": int(len(pf)),
        "best_case": str(best.get("case")),
        "best_do_score": float(best.get("do_score", 0.0)),
        "best_capture_rate": float(best.get("capture_rate", 0.0)),
        "best_tangent_ratio": float(best.get("tangent_ratio", 0.0)),
        "best_params": {k: float(best[k]) for k in PARAMS if k in best},
        "best_pareto_cases": pf.head(12)[["case", "do_score", "capture_rate", "tangent_ratio"]].to_dict(orient="records") if len(pf) else [],
        "stability_leaders": stab.head(8).to_dict(orient="records") if not stab.empty else [],
        "plot_errors": plot_errors,
        "outputs": [
            str(ROOT / "phase26do_case_results.csv"),
            str(ROOT / "phase26do_pareto_front.csv"),
            str(ROOT / "phase26do_stability_summary.csv"),
            str(ROOT / "phase26do_top_scores.png"),
            str(ROOT / "phase26do_capture_vs_tangent_pareto.png"),
            str(ROOT / "phase26do_stability_scores.png"),
        ],
    }
    with open(ROOT / "phase26do_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[{PHASE}] BEST: {summary['best_case']}")
    print(f"[{PHASE}] DO score={summary['best_do_score']:.4f} capture={summary['best_capture_rate']:.4f} tangent={summary['best_tangent_ratio']:.4f}")
    print(f"[{PHASE}] Params: {json.dumps(summary['best_params'], indent=2)}")
    print(f"[{PHASE}] Saved summary: {ROOT / 'phase26do_summary.json'}")
    return summary


if __name__ == "__main__":
    main()
