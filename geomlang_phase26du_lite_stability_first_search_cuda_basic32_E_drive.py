# geomlang_phase26du_lite_stability_first_search_cuda_basic32_E_drive.py
# Phase 26DU-LITE — stability-first rescue search after DS-LITE/DT-LITE.
#
# Why this exists:
#   26DS-LITE found a strict-line pass, but 26DT-LITE showed the exact winner
#   failed retesting across seeds. DU-LITE changes the search order:
#
#       old: one seed screen -> promote exciting candidate -> verify later
#       new: tiny candidate set -> evaluate each candidate across seeds immediately
#
#   The goal is not the highest single-run score. The goal is a family whose
#   tangent tail stays controlled under seed variation while preserving capture
#   and q20 DP.
#
# Recommended first run:
#   python bbit_geomlang/geomlang_phase26du_lite_stability_first_search_cuda_basic32_E_drive.py --device cuda
#
# Faster smoke:
#   python bbit_geomlang/geomlang_phase26du_lite_stability_first_search_cuda_basic32_E_drive.py --device cuda --max-variants 8 --seed-count 2
#
# Wider but still controlled:
#   python bbit_geomlang/geomlang_phase26du_lite_stability_first_search_cuda_basic32_E_drive.py --device cuda --max-variants 24 --seed-count 3 --audit-repeats 2

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

PHASE = "26DU-LITE"
TITLE = "stability-first strict-line rescue search"
ROOT = Path(r"E:\BBIT\outputs_basic32")
HERE = Path(__file__).resolve().parent

DS_LITE_FILE = HERE / "geomlang_phase26ds_lite_strict_line_rescue_cuda_basic32_E_drive.py"
DS_FULL_FILE = HERE / "geomlang_phase26ds_strict_line_rescue_cuda_basic32_E_drive.py"
DR_FAST_FILE = HERE / "geomlang_phase26dr_fast_staged_cuda_basic32_E_drive.py"
DT_SUMMARY = ROOT / "phase26dt_lite_summary.json"
DS_SUMMARY = ROOT / "phase26ds_lite_summary.json"

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

# The DS-LITE candidate that passed once, then failed DT repeatability.
FALLBACK_CENTER = {
    "BOWL_RADIUS_FRAC": 0.6720384774396677,
    "BOWL_SEAT_AXIS_GAIN": 0.36230771528999023,
    "BOWL_DIRECTIONAL_BLEND": 0.9793576654648143,
    "BOWL_NORM_CAP_MULT": 1.0146810931258021,
    "BOWL_SHELL_RADIAL_GAIN": 0.12301295634097414,
    "BOWL_TANGENT_KILL": 0.8263745410885555,
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


def clamp_ov(ov: Dict[str, Any]) -> Dict[str, float]:
    return {k: clamp_param(k, fnum(ov.get(k), FALLBACK_CENTER[k])) for k in PARAMS}


def row_key(ov: Dict[str, Any], ndigits: int = 6) -> Tuple[float, ...]:
    cov = clamp_ov(ov)
    return tuple(round(float(cov[k]), ndigits) for k in PARAMS)


def add_unique(out: List[Tuple[str, Dict[str, float]]], seen: set, name: str, ov: Dict[str, Any]) -> None:
    cov = clamp_ov(ov)
    key = row_key(cov)
    if key not in seen:
        out.append((name[:110], cov))
        seen.add(key)


def parse_overrides_cell(x: Any) -> Dict[str, float] | None:
    if not isinstance(x, str) or not x.strip():
        return None
    try:
        obj = json.loads(x)
        if isinstance(obj, dict) and all(k in obj for k in PARAMS):
            return clamp_ov(obj)
    except Exception:
        return None
    return None


def parse_overrides_from_row(row: pd.Series) -> Dict[str, float] | None:
    if all(k in row.index and pd.notna(row[k]) for k in PARAMS):
        return clamp_ov({k: row[k] for k in PARAMS})
    for col in ("overrides", "overrides_json"):
        if col in row.index:
            ov = parse_overrides_cell(row[col])
            if ov is not None:
                return ov
    return None


def load_center() -> Tuple[str, Dict[str, float], str]:
    if DS_SUMMARY.exists():
        try:
            js = json.loads(DS_SUMMARY.read_text(encoding="utf-8"))
            ov = js.get("best_strict_overrides") or js.get("best_overrides")
            name = js.get("best_strict_case") or js.get("best_case") or "ds_lite_center"
            if isinstance(ov, dict) and all(k in ov for k in PARAMS):
                return str(name), clamp_ov(ov), str(DS_SUMMARY)
        except Exception as e:
            print(f"[{PHASE}] WARNING: could not read DS summary: {e!r}")
    return "fallback_ds_lite_center", clamp_ov(FALLBACK_CENTER), "hardcoded_fallback"


def read_prior_candidates(limit_each: int = 10) -> List[Tuple[str, Dict[str, float]]]:
    """Read the best recent candidate families. Prefer DS-LITE screen/audit and DT summaries."""
    out: List[Tuple[str, Dict[str, float]]] = []
    seen: set = set()

    center_name, center, _ = load_center()
    add_unique(out, seen, f"du_center_{center_name}", center)

    # DT exact winner failed, but keeping it as a named anchor helps compare against new variants.
    if DT_SUMMARY.exists():
        try:
            js = json.loads(DT_SUMMARY.read_text(encoding="utf-8"))
            ov = js.get("winner_overrides") or js.get("exact_winner_summary")
            if isinstance(ov, dict) and all(k in ov for k in PARAMS):
                add_unique(out, seen, "du_dt_failed_exact_anchor", ov)
        except Exception:
            pass

    csvs = [
        ROOT / "phase26ds_lite_case_results.csv",
        ROOT / "phase26ds_lite_screen_results.csv",
        ROOT / "phase26dr_fast_case_results.csv",
        ROOT / "phase26dq_case_results.csv",
        ROOT / "phase26dp_case_results.csv",
    ]
    for p in csvs:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        views: List[pd.DataFrame] = []
        for col, asc in [
            ("ds_lite_score", False),
            ("dr_score", False),
            ("q20_dp_score", False),
            ("worst_tangent_ratio", True),
            ("worst_capture_rate", False),
        ]:
            if col in df.columns:
                views.append(df.sort_values(col, ascending=asc).head(limit_each))
        if not views:
            views = [df.head(limit_each)]
        for vi, vdf in enumerate(views):
            for j, row in vdf.iterrows():
                ov = parse_overrides_from_row(row)
                if ov is None:
                    continue
                case = str(row.get("case", f"row{j}"))
                add_unique(out, seen, f"prior_{p.stem}_{vi:02d}_{case}", ov)
    return out


def stability_variants(seed: int, max_variants: int) -> List[Tuple[str, Dict[str, float]]]:
    """Build a small, biased pool. Focus on moderate radius + stronger tangent kill."""
    rng = random.Random(seed)
    out: List[Tuple[str, Dict[str, float]]] = []
    seen: set = set()

    priors = read_prior_candidates(limit_each=8)
    for name, ov in priors[:12]:
        add_unique(out, seen, name, ov)

    _, center, _ = load_center()

    # Directed children around the failed DS winner: increase tangent kill and seat,
    # keep radius moderate, avoid the old capture-heavy 0.78/0.50/0.76 corner.
    for r in [0.64, 0.66, 0.68, 0.70, 0.72]:
        for tk in [0.86, 0.90, 0.94, 0.98]:
            ov = dict(center)
            ov["BOWL_RADIUS_FRAC"] = r
            ov["BOWL_TANGENT_KILL"] = tk
            ov["BOWL_SEAT_AXIS_GAIN"] = max(center["BOWL_SEAT_AXIS_GAIN"], 0.38)
            ov["BOWL_NORM_CAP_MULT"] = min(max(center["BOWL_NORM_CAP_MULT"], 1.04), 1.10)
            add_unique(out, seen, f"du_center_r{r:.2f}_tk{tk:.2f}", ov)

    # Compact hand grid around the suspected stability band.
    hand_grid = []
    for r in [0.66, 0.68, 0.70]:
        for s in [0.38, 0.42, 0.46]:
            for b in [0.98, 1.02, 1.06]:
                for c in [1.04, 1.08, 1.12]:
                    for sh in [0.12, 0.14, 0.16]:
                        for tk in [0.88, 0.92, 0.96]:
                            hand_grid.append((f"du_grid_r{r:.2f}_s{s:.2f}_b{b:.2f}_c{c:.2f}_sh{sh:.2f}_tk{tk:.2f}", {
                                "BOWL_RADIUS_FRAC": r,
                                "BOWL_SEAT_AXIS_GAIN": s,
                                "BOWL_DIRECTIONAL_BLEND": b,
                                "BOWL_NORM_CAP_MULT": c,
                                "BOWL_SHELL_RADIAL_GAIN": sh,
                                "BOWL_TANGENT_KILL": tk,
                            }))
    rng.shuffle(hand_grid)
    for name, ov in hand_grid[:max(12, max_variants)]:
        add_unique(out, seen, name, ov)

    # Random cloud biased toward stability rather than peak capture.
    for i in range(max(8, max_variants)):
        ov = {
            "BOWL_RADIUS_FRAC": rng.gauss(0.68, 0.025),
            "BOWL_SEAT_AXIS_GAIN": rng.gauss(0.42, 0.035),
            "BOWL_DIRECTIONAL_BLEND": rng.gauss(1.02, 0.035),
            "BOWL_NORM_CAP_MULT": rng.gauss(1.07, 0.030),
            "BOWL_SHELL_RADIAL_GAIN": rng.gauss(0.14, 0.018),
            "BOWL_TANGENT_KILL": rng.gauss(0.92, 0.055),
        }
        add_unique(out, seen, f"du_stability_cloud_{i:03d}", ov)

    # Selection: keep anchors plus an even spread. This avoids only testing near-duplicates.
    if len(out) > max_variants:
        anchors = out[: min(4, len(out))]
        rest = out[len(anchors):]
        keep_rest = max_variants - len(anchors)
        if keep_rest > 0 and rest:
            idxs = np.linspace(0, len(rest) - 1, keep_rest).round().astype(int).tolist()
            out = anchors + [rest[i] for i in idxs]
        else:
            out = anchors
    return out[:max_variants]


def pass_flags(row: Dict[str, Any], target_cap: float, target_tan: float, target_q20: float, relaxed_tan: float) -> Dict[str, bool]:
    cap = fnum(row.get("worst_capture_rate"), 0.0)
    tan = fnum(row.get("worst_tangent_ratio"), 999.0)
    q20 = fnum(row.get("q20_dp_score"), -999.0)
    strict = (cap >= target_cap and tan <= target_tan and q20 >= target_q20)
    relaxed = (cap >= target_cap and tan <= relaxed_tan and q20 >= target_q20)
    return {
        "pass_cap": cap >= target_cap,
        "pass_tan_strict": tan <= target_tan,
        "pass_tan_relaxed": tan <= relaxed_tan,
        "pass_q20": q20 >= target_q20,
        "pass_strict_line": strict,
        "pass_relaxed_line": relaxed,
    }


def summarize(results_df: pd.DataFrame, relaxed_tan: float) -> pd.DataFrame:
    rows = []
    if results_df.empty:
        return pd.DataFrame()
    for variant, g in results_df.groupby("variant", sort=False):
        max_tan = float(g["worst_tangent_ratio"].max())
        min_cap = float(g["worst_capture_rate"].min())
        min_q20 = float(g["q20_dp_score"].min())
        strict_rate = float(g["pass_strict_line"].mean())
        relaxed_rate = float(g["pass_relaxed_line"].mean())
        mean_score = float(g["du_stability_score_run"].mean())
        # stability score favors repeatability first, then score, then low variance/tangent tail.
        tan_var = float(g["worst_tangent_ratio"].std(ddof=0)) if len(g) > 1 else 0.0
        cap_var = float(g["worst_capture_rate"].std(ddof=0)) if len(g) > 1 else 0.0
        family_score = (
            6.0 * relaxed_rate
            + 4.0 * strict_rate
            + 0.35 * mean_score
            - 1.20 * max(0.0, max_tan - relaxed_tan)
            - 0.70 * tan_var
            + 6.0 * max(0.0, min_cap - 0.285)
            + 0.45 * max(0.0, min_q20 - 2.25)
        )
        rows.append({
            "variant": variant,
            "n": int(len(g)),
            "strict_pass_count": int(g["pass_strict_line"].sum()),
            "relaxed_pass_count": int(g["pass_relaxed_line"].sum()),
            "strict_pass_rate": strict_rate,
            "relaxed_pass_rate": relaxed_rate,
            "stable_strict_all_seeds": bool(g["pass_strict_line"].all()),
            "stable_relaxed_all_seeds": bool(g["pass_relaxed_line"].all()),
            "du_family_score": float(family_score),
            "mean_du_run_score": mean_score,
            "mean_dr_score": float(g["dr_score"].mean()),
            "min_q20_dp_score": min_q20,
            "min_worst_dp_score": float(g["worst_dp_score"].min()),
            "min_worst_capture_rate": min_cap,
            "max_worst_tangent_ratio": max_tan,
            "std_worst_tangent_ratio": tan_var,
            "std_worst_capture_rate": cap_var,
            "mean_strict_survival_rate": float(g["strict_survival_rate"].mean()) if "strict_survival_rate" in g else 0.0,
            **{k: float(g[k].iloc[0]) for k in PARAMS if k in g.columns},
        })
    return pd.DataFrame(rows).sort_values(
        ["stable_strict_all_seeds", "stable_relaxed_all_seeds", "relaxed_pass_rate", "du_family_score"],
        ascending=[False, False, False, False],
    )


def plot_family_scores(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    top = df.sort_values("du_family_score", ascending=False).head(25).iloc[::-1]
    plt.figure(figsize=(14, max(6, 0.46 * len(top))))
    plt.barh(top["variant"], top["du_family_score"])
    plt.xlabel("DU family stability score")
    plt.title("26DU-LITE stability-first family scores")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_pass_rates(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    top = df.sort_values(["relaxed_pass_rate", "du_family_score"], ascending=[False, False]).head(25).iloc[::-1]
    y = np.arange(len(top))
    plt.figure(figsize=(14, max(6, 0.46 * len(top))))
    plt.barh(y - 0.18, top["relaxed_pass_rate"], height=0.34, label="relaxed")
    plt.barh(y + 0.18, top["strict_pass_rate"], height=0.34, label="strict")
    plt.yticks(y, top["variant"])
    plt.xlim(0, 1.05)
    plt.xlabel("pass rate across seeds")
    plt.title("26DU-LITE pass rates")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_capture_tangent(df: pd.DataFrame, out_path: Path, target_cap: float, target_tan: float, relaxed_tan: float) -> None:
    if df.empty:
        return
    plt.figure(figsize=(10, 7))
    sc = plt.scatter(df["worst_capture_rate"], df["worst_tangent_ratio"], c=df["du_stability_score_run"], s=74, alpha=0.78)
    plt.axvline(target_cap, linewidth=1.0, alpha=0.45, label="capture target")
    plt.axhline(target_tan, linewidth=1.0, alpha=0.45, label="strict tangent")
    plt.axhline(relaxed_tan, linewidth=1.0, alpha=0.25, linestyle="--", label="relaxed tangent")
    plt.colorbar(sc, label="DU run score")
    plt.xlabel("worst envelope capture_rate")
    plt.ylabel("worst envelope tangent/radial ratio")
    plt.title("26DU-LITE stability runs: capture vs tangent")
    plt.grid(True, alpha=0.28)
    plt.legend()
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

    center_name, center_ov, center_source = load_center()

    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] device={args.device} torch={torch_info}")
    print(f"[{PHASE}] targets: strictTan<={target_tan:.2f}, relaxedTan<={relaxed_tan:.2f}, worstCap>={target_cap:.3f}, q20DP>={target_q20:.2f}")
    print(f"[{PHASE}] center_source={center_source}")
    print(f"[{PHASE}] center_name={center_name}")
    print(f"[{PHASE}] seed_count={args.seed_count}, max_variants={args.max_variants}, audit_repeats={args.audit_repeats}, sampled_full_tail={args.sample_full_tail}")

    eval_helper = getattr(dr, "EVAL_HELPER", None)
    if eval_helper is None or not hasattr(eval_helper, "build_basins"):
        eval_helper = dr.previous_eval_helper()
    z2, rel_ids, basins = eval_helper.build_basins()
    print(f"[{PHASE}] Built basins: {len(basins)}")

    variants = stability_variants(args.seed, args.max_variants)
    print(f"[{PHASE}] Stability-first variants: {len(variants)}")

    cache: Dict[Tuple[str, Tuple[float, ...]], Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    env_rows: List[Dict[str, Any]] = []

    total = len(variants) * args.seed_count
    run_i = 0
    for vi, (variant, ov) in enumerate(variants, 1):
        for si in range(args.seed_count):
            run_i += 1
            seed = args.seed + 260000 + vi * 1009 + si * 7919
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
                phase_label="du_stability",
            )
            agg["variant"] = variant
            agg["seed_index"] = si + 1
            agg["du_seed"] = seed
            agg["du_stability_score_run"] = ds_full.rescue_score(agg, target_cap, target_tan, target_q20)
            agg.update(pass_flags(agg, target_cap, target_tan, target_q20, relaxed_tan))
            rows.append(agg)
            env_rows.extend(erows)

            if run_i <= 5 or run_i % args.print_every == 0 or si == args.seed_count - 1:
                print(
                    f"[{PHASE}] run {run_i:03d}/{total:03d} {variant[:58]:58s} s{si+1:02d} "
                    f"RELAX={int(agg['pass_relaxed_line'])} STRICT={int(agg['pass_strict_line'])} "
                    f"DU={agg['du_stability_score_run']:.3f} DR={agg['dr_score']:.3f} "
                    f"q20={agg['q20_dp_score']:.3f} worstDP={agg['worst_dp_score']:.3f} "
                    f"cap={agg['worst_capture_rate']:.3f} tan={agg['worst_tangent_ratio']:.3f} "
                    f"strictSurv={agg.get('strict_survival_rate', 0.0):.2f}"
                )

    results_df = pd.DataFrame(rows)
    env_df = pd.DataFrame(env_rows)
    summary_df = summarize(results_df, relaxed_tan)

    results_df.to_csv(ROOT / "phase26du_lite_stability_results.csv", index=False)
    env_df.to_csv(ROOT / "phase26du_lite_envelope_results.csv", index=False)
    summary_df.to_csv(ROOT / "phase26du_lite_variant_summary.csv", index=False)

    plot_family_scores(summary_df, ROOT / "phase26du_lite_family_scores.png")
    plot_pass_rates(summary_df, ROOT / "phase26du_lite_pass_rates.png")
    plot_capture_tangent(results_df, ROOT / "phase26du_lite_capture_vs_tangent.png", target_cap, target_tan, relaxed_tan)

    stable_strict = summary_df[summary_df["stable_strict_all_seeds"] == True] if not summary_df.empty else pd.DataFrame()
    stable_relaxed = summary_df[summary_df["stable_relaxed_all_seeds"] == True] if not summary_df.empty else pd.DataFrame()
    best = summary_df.iloc[0] if len(summary_df) else pd.Series(dtype=float)

    if len(stable_strict):
        verdict = "strict_stable_family_found"
    elif len(stable_relaxed):
        verdict = "relaxed_stable_family_found"
    elif len(summary_df) and fnum(best.get("relaxed_pass_rate"), 0.0) >= args.min_promote_rate:
        verdict = "partial_stability_candidate_found"
    else:
        verdict = "no_stable_family_found"

    def row_to_overrides(row: pd.Series) -> Dict[str, float] | None:
        if not len(row):
            return None
        return {k: fnum(row.get(k), FALLBACK_CENTER[k]) for k in PARAMS}

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
        "center_source": center_source,
        "center_name": center_name,
        "center_overrides": center_ov,
        "seed_count": int(args.seed_count),
        "max_variants": int(args.max_variants),
        "audit_repeats": int(args.audit_repeats),
        "sample_full_tail": bool(args.sample_full_tail),
        "num_variants": int(len(variants)),
        "num_runs": int(len(results_df)),
        "num_envelope_evals": int(len(env_df)),
        "num_cached_eval_entries": int(len(cache)),
        "num_stable_strict_families": int(len(stable_strict)),
        "num_stable_relaxed_families": int(len(stable_relaxed)),
        "elapsed_sec": float(time.time() - t0),
        "verdict": verdict,
        "best_variant": None if not len(best) else str(best.get("variant")),
        "best_du_family_score": None if not len(best) else fnum(best.get("du_family_score")),
        "best_strict_pass_rate": None if not len(best) else fnum(best.get("strict_pass_rate")),
        "best_relaxed_pass_rate": None if not len(best) else fnum(best.get("relaxed_pass_rate")),
        "best_min_cap": None if not len(best) else fnum(best.get("min_worst_capture_rate")),
        "best_max_tan": None if not len(best) else fnum(best.get("max_worst_tangent_ratio")),
        "best_min_q20": None if not len(best) else fnum(best.get("min_q20_dp_score")),
        "best_overrides": row_to_overrides(best),
        "variant_top10": summary_df.head(10).to_dict(orient="records") if len(summary_df) else [],
        "outputs": [
            "phase26du_lite_stability_results.csv",
            "phase26du_lite_envelope_results.csv",
            "phase26du_lite_variant_summary.csv",
            "phase26du_lite_summary.json",
            "phase26du_lite_family_scores.png",
            "phase26du_lite_pass_rates.png",
            "phase26du_lite_capture_vs_tangent.png",
        ],
    }
    (ROOT / "phase26du_lite_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[{PHASE}] Done in {summary['elapsed_sec']:.1f}s")
    print(f"[{PHASE}] VERDICT: {verdict}")
    if len(best):
        print(
            f"[{PHASE}] BEST VARIANT: {best['variant']} | family={best['du_family_score']:.3f} "
            f"strictRate={best['strict_pass_rate']:.2f} relaxedRate={best['relaxed_pass_rate']:.2f} "
            f"minCap={best['min_worst_capture_rate']:.3f} maxTan={best['max_worst_tangent_ratio']:.3f} "
            f"minQ20={best['min_q20_dp_score']:.3f}"
        )
        print(f"[{PHASE}] BEST overrides: {row_to_overrides(best)}")
    print(f"[{PHASE}] Wrote summary: {ROOT / 'phase26du_lite_summary.json'}")
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=f"{PHASE}: {TITLE}")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="torch device preference")
    ap.add_argument("--seed", type=int, default=26000)
    ap.add_argument("--max-variants", type=int, default=12, help="candidate families to test")
    ap.add_argument("--seed-count", type=int, default=2, help="seeds per variant; raise to 3 after a promising run")
    ap.add_argument("--audit-repeats", type=int, default=1, help="micro-envelope repeats per seed")
    ap.add_argument("--sample-full-tail", action="store_true", help="include a small sampled full-envelope tail; slower")
    ap.add_argument("--relaxed-tan", type=float, default=2.30, help="early relaxed tangent threshold")
    ap.add_argument("--min-promote-rate", type=float, default=0.67, help="promote threshold for partial stability")
    ap.add_argument("--print-every", type=int, default=6)
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
