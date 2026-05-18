# geomlang_phase26dt_lite_winner_lockin_verifier_cuda_basic32_E_drive.py
# Phase 26DT-LITE — winner lock-in verifier for the DS-LITE strict-line pass.
#
# Purpose:
#   26DS-LITE found one candidate that passed the strict line, but the run was still
#   slow because the inherited evaluator is expensive. 26DT-LITE stops broad search
#   and only verifies the DS-LITE winner: exact repeatability first, optional tiny
#   local jitter second.
#
# Typical first run, fastest useful confirmation:
#   python bbit_geomlang/geomlang_phase26dt_lite_winner_lockin_verifier_cuda_basic32_E_drive.py --device cuda
#
# Tiny local stability cloud around the winner:
#   python bbit_geomlang/geomlang_phase26dt_lite_winner_lockin_verifier_cuda_basic32_E_drive.py --device cuda --variant-mode tiny --jitter-count 8 --seed-count 3
#
# Slower, more suspicious audit:
#   python bbit_geomlang/geomlang_phase26dt_lite_winner_lockin_verifier_cuda_basic32_E_drive.py --device cuda --variant-mode tiny --jitter-count 12 --seed-count 4 --audit-repeats 2 --sample-full-tail

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

PHASE = "26DT-LITE"
TITLE = "DS-LITE winner lock-in verifier"
ROOT = Path(r"E:\BBIT\outputs_basic32")
HERE = Path(__file__).resolve().parent

DS_LITE_FILE = HERE / "geomlang_phase26ds_lite_strict_line_rescue_cuda_basic32_E_drive.py"
DS_FULL_FILE = HERE / "geomlang_phase26ds_strict_line_rescue_cuda_basic32_E_drive.py"
DR_FAST_FILE = HERE / "geomlang_phase26dr_fast_staged_cuda_basic32_E_drive.py"

PARAMS = [
    "BOWL_RADIUS_FRAC",
    "BOWL_SEAT_AXIS_GAIN",
    "BOWL_DIRECTIONAL_BLEND",
    "BOWL_NORM_CAP_MULT",
    "BOWL_SHELL_RADIAL_GAIN",
    "BOWL_TANGENT_KILL",
]

# Bounds copied from 26DS so local jitter cannot walk into nonsense territory.
BOUNDS = {
    "BOWL_RADIUS_FRAC": (0.56, 0.82),
    "BOWL_SEAT_AXIS_GAIN": (0.26, 0.56),
    "BOWL_DIRECTIONAL_BLEND": (0.92, 1.18),
    "BOWL_NORM_CAP_MULT": (0.96, 1.20),
    "BOWL_SHELL_RADIAL_GAIN": (0.08, 0.18),
    "BOWL_TANGENT_KILL": (0.68, 1.02),
}

# DS-LITE winner from the 26DS-LITE run.
FALLBACK_WINNER_OVERRIDES = {
    "BOWL_RADIUS_FRAC": 0.6720384774396677,
    "BOWL_SEAT_AXIS_GAIN": 0.36230771528999023,
    "BOWL_DIRECTIONAL_BLEND": 0.9793576654648143,
    "BOWL_NORM_CAP_MULT": 1.0146810931258021,
    "BOWL_SHELL_RADIAL_GAIN": 0.12301295634097414,
    "BOWL_TANGENT_KILL": 0.8263745410885555,
}

FALLBACK_WINNER_NAME = "ds_lite_winner_dq_case_048_confidence_cloud_003"


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


def clamp_ov(ov: Dict[str, float]) -> Dict[str, float]:
    return {k: clamp_param(k, fnum(ov.get(k), FALLBACK_WINNER_OVERRIDES[k])) for k in PARAMS}


def row_key(ov: Dict[str, float], ndigits: int = 6) -> Tuple[float, ...]:
    cov = clamp_ov(ov)
    return tuple(round(float(cov[k]), ndigits) for k in PARAMS)


def load_winner_from_summary() -> Tuple[str, Dict[str, float], str]:
    """Prefer the actual DS-LITE summary if present; otherwise use the known winner."""
    summary_path = ROOT / "phase26ds_lite_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                js = json.load(f)
            ov = js.get("best_strict_overrides") or js.get("best_overrides")
            name = js.get("best_strict_case") or js.get("best_case") or FALLBACK_WINNER_NAME
            if isinstance(ov, dict) and all(k in ov for k in PARAMS):
                return str(name), clamp_ov({k: float(ov[k]) for k in PARAMS}), str(summary_path)
        except Exception as e:
            print(f"[{PHASE}] WARNING: could not read DS-LITE summary: {e!r}")
    return FALLBACK_WINNER_NAME, clamp_ov(FALLBACK_WINNER_OVERRIDES), "hardcoded_fallback"


def add_unique(out: List[Tuple[str, Dict[str, float]]], seen: set, name: str, ov: Dict[str, float]) -> None:
    cov = clamp_ov(ov)
    key = row_key(cov)
    if key not in seen:
        out.append((name, cov))
        seen.add(key)


def build_variants(base_name: str, base: Dict[str, float], mode: str, jitter_count: int, seed: int) -> List[Tuple[str, Dict[str, float]]]:
    """
    Keep this deliberately tiny. 26DT is a verifier, not a search phase.
    exact: one candidate only.
    axis: exact + one-parameter nudges.
    tiny: axis + small random local cloud.
    """
    rng = random.Random(seed)
    out: List[Tuple[str, Dict[str, float]]] = []
    seen: set = set()
    safe_base = clamp_ov(base)

    add_unique(out, seen, f"{base_name}_exact", safe_base)
    if mode == "exact":
        return out

    # Very small nudges: preserve the discovered basin while checking whether it is knife-edge.
    axis_steps = {
        "BOWL_RADIUS_FRAC": 0.010,
        "BOWL_SEAT_AXIS_GAIN": 0.015,
        "BOWL_DIRECTIONAL_BLEND": 0.012,
        "BOWL_NORM_CAP_MULT": 0.012,
        "BOWL_SHELL_RADIAL_GAIN": 0.006,
        "BOWL_TANGENT_KILL": 0.018,
    }
    for k, step in axis_steps.items():
        for sign, tag in [(-1.0, "minus"), (1.0, "plus")]:
            ov = dict(safe_base)
            ov[k] = ov[k] + sign * step
            add_unique(out, seen, f"{base_name}_axis_{k}_{tag}", ov)

    if mode == "axis":
        return out

    sig = {
        "BOWL_RADIUS_FRAC": 0.010,
        "BOWL_SEAT_AXIS_GAIN": 0.014,
        "BOWL_DIRECTIONAL_BLEND": 0.012,
        "BOWL_NORM_CAP_MULT": 0.012,
        "BOWL_SHELL_RADIAL_GAIN": 0.006,
        "BOWL_TANGENT_KILL": 0.018,
    }
    for i in range(max(0, jitter_count)):
        ov = {k: safe_base[k] + rng.gauss(0.0, sig[k]) for k in PARAMS}
        add_unique(out, seen, f"{base_name}_tiny_cloud_{i:03d}", ov)

    return out


def pass_flags(row: Dict[str, Any], target_cap: float, target_tan: float, target_q20: float) -> Dict[str, bool]:
    cap = fnum(row.get("worst_capture_rate"), 0.0)
    tan = fnum(row.get("worst_tangent_ratio"), 999.0)
    q20 = fnum(row.get("q20_dp_score"), -999.0)
    return {
        "pass_cap": cap >= target_cap,
        "pass_tan": tan <= target_tan,
        "pass_q20": q20 >= target_q20,
        "pass_strict_line": (cap >= target_cap and tan <= target_tan and q20 >= target_q20),
    }


def plot_variant_scores(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    top = df.sort_values("mean_ds_lite_score", ascending=False).head(25).iloc[::-1]
    plt.figure(figsize=(14, max(6, 0.46 * len(top))))
    plt.barh(top["variant"], top["mean_ds_lite_score"])
    plt.xlabel("mean DS-LITE score across seeds")
    plt.title("26DT-LITE winner lock-in: variant score")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_capture_tangent(df: pd.DataFrame, out_path: Path, target_cap: float, target_tan: float) -> None:
    if df.empty:
        return
    plt.figure(figsize=(10, 7))
    c = df["ds_lite_score"] if "ds_lite_score" in df.columns else None
    sc = plt.scatter(df["worst_capture_rate"], df["worst_tangent_ratio"], c=c, s=78, alpha=0.78)
    plt.axvline(target_cap, linewidth=1.0, alpha=0.45)
    plt.axhline(target_tan, linewidth=1.0, alpha=0.45)
    if c is not None:
        plt.colorbar(sc, label="DS-LITE score")
    plt.xlabel("worst envelope capture_rate")
    plt.ylabel("worst envelope tangent/radial ratio")
    plt.title("26DT-LITE verification runs: capture vs tangent")
    plt.grid(True, alpha=0.28)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_pass_rates(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty or "pass_rate" not in df.columns:
        return
    top = df.sort_values(["pass_rate", "mean_ds_lite_score"], ascending=[False, False]).head(25).iloc[::-1]
    plt.figure(figsize=(14, max(6, 0.46 * len(top))))
    plt.barh(top["variant"], top["pass_rate"])
    plt.xlim(0, 1.05)
    plt.xlabel("strict-line pass rate across tested seeds")
    plt.title("26DT-LITE stability pass rates")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def summarize_variants(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()
    groups = []
    for variant, g in results_df.groupby("variant", sort=False):
        groups.append({
            "variant": variant,
            "n": int(len(g)),
            "pass_count": int(g["pass_strict_line"].sum()),
            "pass_rate": float(g["pass_strict_line"].mean()),
            "mean_ds_lite_score": float(g["ds_lite_score"].mean()),
            "min_ds_lite_score": float(g["ds_lite_score"].min()),
            "mean_dr_score": float(g["dr_score"].mean()),
            "min_q20_dp_score": float(g["q20_dp_score"].min()),
            "min_worst_dp_score": float(g["worst_dp_score"].min()),
            "min_worst_capture_rate": float(g["worst_capture_rate"].min()),
            "max_worst_tangent_ratio": float(g["worst_tangent_ratio"].max()),
            "mean_strict_survival_rate": float(g["strict_survival_rate"].mean()) if "strict_survival_rate" in g.columns else 0.0,
            "stable_all_seeds": bool(g["pass_strict_line"].all()),
            **{k: float(g[k].iloc[0]) for k in PARAMS if k in g.columns},
        })
    return pd.DataFrame(groups).sort_values(
        ["stable_all_seeds", "pass_rate", "mean_ds_lite_score"],
        ascending=[False, False, False],
    )


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

    winner_name, winner_ov, winner_source = load_winner_from_summary() if not args.use_hardcoded_winner else (FALLBACK_WINNER_NAME, clamp_ov(FALLBACK_WINNER_OVERRIDES), "hardcoded_forced")

    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] device={args.device} torch={torch_info}")
    print(f"[{PHASE}] targets: worstTan<={target_tan:.2f}, worstCap>={target_cap:.3f}, q20DP>={target_q20:.2f}")
    print(f"[{PHASE}] winner_source={winner_source}")
    print(f"[{PHASE}] winner_name={winner_name}")
    print(f"[{PHASE}] winner_overrides={winner_ov}")
    print(f"[{PHASE}] variant_mode={args.variant_mode}, jitter_count={args.jitter_count}, seed_count={args.seed_count}, audit_repeats={args.audit_repeats}, sampled_full_tail={args.sample_full_tail}")

    eval_helper = getattr(dr, "EVAL_HELPER", None)
    if eval_helper is None or not hasattr(eval_helper, "build_basins"):
        eval_helper = dr.previous_eval_helper()
    z2, rel_ids, basins = eval_helper.build_basins()
    print(f"[{PHASE}] Built basins: {len(basins)}")

    variants = build_variants("dt_winner", winner_ov, args.variant_mode, args.jitter_count, args.seed)
    if args.max_variants is not None and args.max_variants > 0:
        variants = variants[: args.max_variants]
    print(f"[{PHASE}] Verification variants: {len(variants)}")

    cache: Dict[Tuple[str, Tuple[float, ...]], Dict[str, Any]] = {}
    result_rows: List[Dict[str, Any]] = []
    env_rows: List[Dict[str, Any]] = []

    seeds = [args.seed + 100000 + 4099 * i for i in range(max(1, args.seed_count))]
    total = len(variants) * len(seeds)
    done = 0

    for vi, (variant_name, ov) in enumerate(variants, 1):
        for si, seed in enumerate(seeds, 1):
            done += 1
            case = f"{variant_name}_seed{si:02d}"
            env = ds_lite.micro_envelope(
                drfast,
                dr,
                ov,
                seed=seed + vi * 31,
                repeats=args.audit_repeats,
                use_full_tail=args.sample_full_tail,
            )
            agg, rows = drfast.evaluate_envelope(
                dr,
                case,
                ov,
                basins,
                seed=seed + vi * 31,
                envelope=env,
                cache=cache,
                phase_label="dt_verify",
            )
            agg["variant"] = variant_name
            agg["seed_index"] = si
            agg["dt_seed"] = seed
            agg["ds_lite_score"] = ds_full.rescue_score(agg, target_cap, target_tan, target_q20)
            flags = pass_flags(agg, target_cap, target_tan, target_q20)
            agg.update(flags)
            result_rows.append(agg)
            env_rows.extend(rows)

            print(
                f"[{PHASE}] verify {done:03d}/{total:03d} {case[:62]:62s} "
                f"PASS={int(flags['pass_strict_line'])} LITE={agg['ds_lite_score']:.3f} DR={agg['dr_score']:.3f} "
                f"q20={agg['q20_dp_score']:.3f} worstDP={agg['worst_dp_score']:.3f} "
                f"worstCap={agg['worst_capture_rate']:.3f} worstTan={agg['worst_tangent_ratio']:.3f} "
                f"strict={agg.get('strict_survival_rate', 0.0):.2f}"
            )

    results_df = pd.DataFrame(result_rows)
    if not results_df.empty:
        results_df = results_df.sort_values(["pass_strict_line", "ds_lite_score"], ascending=[False, False])
    env_df = pd.DataFrame(env_rows)
    variant_df = summarize_variants(results_df)

    results_df.to_csv(ROOT / "phase26dt_lite_verification_results.csv", index=False)
    env_df.to_csv(ROOT / "phase26dt_lite_envelope_results.csv", index=False)
    variant_df.to_csv(ROOT / "phase26dt_lite_variant_summary.csv", index=False)

    plot_variant_scores(variant_df, ROOT / "phase26dt_lite_variant_scores.png")
    plot_capture_tangent(results_df, ROOT / "phase26dt_lite_capture_vs_tangent.png", target_cap, target_tan)
    plot_pass_rates(variant_df, ROOT / "phase26dt_lite_pass_rates.png")

    best_run = results_df.iloc[0] if len(results_df) else pd.Series(dtype=float)
    best_variant = variant_df.iloc[0] if len(variant_df) else pd.Series(dtype=float)
    exact_summary = variant_df[variant_df["variant"] == "dt_winner_exact"].iloc[0].to_dict() if len(variant_df) and "dt_winner_exact" in set(variant_df["variant"]) else None

    stable_variants = variant_df[variant_df["stable_all_seeds"] == True] if len(variant_df) else pd.DataFrame()
    verdict = "no_results"
    if exact_summary:
        if bool(exact_summary.get("stable_all_seeds")):
            verdict = "exact_winner_stable_all_tested_seeds"
        elif float(exact_summary.get("pass_rate", 0.0)) > 0.0:
            verdict = "exact_winner_partially_stable_needs_more_audit"
        else:
            verdict = "exact_winner_failed_retest"

    summary = {
        "phase": PHASE,
        "title": TITLE,
        "torch_info": torch_info,
        "targets": {
            "target_worst_tangent_ratio": target_tan,
            "target_worst_capture_rate": target_cap,
            "target_q20_dp_score": target_q20,
        },
        "winner_source": winner_source,
        "winner_name": winner_name,
        "winner_overrides": winner_ov,
        "variant_mode": args.variant_mode,
        "jitter_count": args.jitter_count,
        "seed_count": args.seed_count,
        "audit_repeats": args.audit_repeats,
        "sample_full_tail": bool(args.sample_full_tail),
        "num_variants": int(len(variants)),
        "num_verification_runs": int(len(results_df)),
        "num_envelope_evals": int(len(env_df)),
        "num_cached_eval_entries": int(len(cache)),
        "num_stable_variants_all_seeds": int(len(stable_variants)),
        "elapsed_sec": float(time.time() - t0),
        "verdict": verdict,
        "exact_winner_summary": exact_summary,
        "best_run": None if not len(best_run) else best_run.to_dict(),
        "best_variant_summary": None if not len(best_variant) else best_variant.to_dict(),
        "stable_variants_top10": stable_variants.head(10).to_dict(orient="records") if len(stable_variants) else [],
        "outputs": [
            "phase26dt_lite_verification_results.csv",
            "phase26dt_lite_envelope_results.csv",
            "phase26dt_lite_variant_summary.csv",
            "phase26dt_lite_summary.json",
            "phase26dt_lite_variant_scores.png",
            "phase26dt_lite_capture_vs_tangent.png",
            "phase26dt_lite_pass_rates.png",
        ],
    }

    with open(ROOT / "phase26dt_lite_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[{PHASE}] Done in {summary['elapsed_sec']:.1f}s")
    print(f"[{PHASE}] VERDICT: {verdict}")
    if exact_summary:
        print(
            f"[{PHASE}] EXACT WINNER: pass_rate={exact_summary['pass_rate']:.2f} "
            f"minCap={exact_summary['min_worst_capture_rate']:.3f} "
            f"maxTan={exact_summary['max_worst_tangent_ratio']:.3f} "
            f"minQ20={exact_summary['min_q20_dp_score']:.3f} "
            f"meanLITE={exact_summary['mean_ds_lite_score']:.3f}"
        )
    if len(best_variant):
        print(
            f"[{PHASE}] BEST VARIANT: {best_variant['variant']} | "
            f"pass_rate={best_variant['pass_rate']:.2f} "
            f"meanLITE={best_variant['mean_ds_lite_score']:.3f} "
            f"minCap={best_variant['min_worst_capture_rate']:.3f} "
            f"maxTan={best_variant['max_worst_tangent_ratio']:.3f}"
        )
    print(f"[{PHASE}] Wrote summary: {ROOT / 'phase26dt_lite_summary.json'}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=TITLE)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--variant-mode", choices=["exact", "axis", "tiny"], default="exact", help="exact is fastest; tiny adds a small local jitter cloud.")
    p.add_argument("--jitter-count", type=int, default=8, help="Only used with --variant-mode tiny.")
    p.add_argument("--max-variants", type=int, default=None, help="Optional hard cap after variant construction.")
    p.add_argument("--seed-count", type=int, default=5, help="Number of independent verification seeds per variant.")
    p.add_argument("--audit-repeats", type=int, default=1, help="Cheap screen-envelope repeats per verification run.")
    p.add_argument("--sample-full-tail", action="store_true", help="Add up to 8 sampled full-envelope cases per run. Slower.")
    p.add_argument("--use-hardcoded-winner", action="store_true", help="Ignore phase26ds_lite_summary.json and use the built-in winner constants.")
    p.add_argument("--seed", type=int, default=26030)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
