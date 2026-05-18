# geomlang_phase26dw_lite_culprit_case_audit_cuda_basic32_E_drive.py
# Phase 26DW-LITE — culprit-case audit for the DV near-capture family.
#
# Why this exists:
#   26DV-LITE found a very useful near miss:
#
#       dv_grid_r0.72_s0.42_b1.02_c1.16_sh0.165_tk1.00
#       minCap ~= 0.265625, maxTan ~= 2.033, minQ20 ~= 2.502
#
#   This means tangent and q20 are now basically solved for that family, but the
#   worst capture floor remains below target. DW-LITE is NOT another broad search.
#   It audits the individual envelope cases to identify the actual failure source:
#       - one recurring low-capture envelope case?
#       - a deterministic tangent culprit?
#       - an envelope/sample artifact?
#       - a tradeoff between capture lift and tangent discipline?
#
# Recommended first run:
#   python bbit_geomlang/geomlang_phase26dw_lite_culprit_case_audit_cuda_basic32_E_drive.py --device cuda
#
# Focus on only the DV best:
#   python bbit_geomlang/geomlang_phase26dw_lite_culprit_case_audit_cuda_basic32_E_drive.py --device cuda --top-variants 1 --seed-count 2
#
# Wider diagnostic:
#   python bbit_geomlang/geomlang_phase26dw_lite_culprit_case_audit_cuda_basic32_E_drive.py --device cuda --top-variants 4 --seed-count 3 --audit-repeats 1

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PHASE = "26DW-LITE"
TITLE = "culprit-case audit for DV near-capture strict-line rescue"
ROOT = Path(r"E:\BBIT\outputs_basic32")
HERE = Path(__file__).resolve().parent

DS_LITE_FILE = HERE / "geomlang_phase26ds_lite_strict_line_rescue_cuda_basic32_E_drive.py"
DS_FULL_FILE = HERE / "geomlang_phase26ds_strict_line_rescue_cuda_basic32_E_drive.py"
DR_FAST_FILE = HERE / "geomlang_phase26dr_fast_staged_cuda_basic32_E_drive.py"
DV_SUMMARY = ROOT / "phase26dv_lite_summary.json"
DV_VARIANTS = ROOT / "phase26dv_lite_variant_summary.csv"
DV_RESULTS = ROOT / "phase26dv_lite_capture_lift_results.csv"
DV_ENVELOPE = ROOT / "phase26dv_lite_envelope_results.csv"

PARAMS = [
    "BOWL_RADIUS_FRAC",
    "BOWL_SEAT_AXIS_GAIN",
    "BOWL_DIRECTIONAL_BLEND",
    "BOWL_NORM_CAP_MULT",
    "BOWL_SHELL_RADIAL_GAIN",
    "BOWL_TANGENT_KILL",
]

FALLBACK_DV_BEST = {
    "BOWL_RADIUS_FRAC": 0.72,
    "BOWL_SEAT_AXIS_GAIN": 0.42,
    "BOWL_DIRECTIONAL_BLEND": 1.02,
    "BOWL_NORM_CAP_MULT": 1.16,
    "BOWL_SHELL_RADIAL_GAIN": 0.165,
    "BOWL_TANGENT_KILL": 1.00,
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
        return {k: fnum(row[k], FALLBACK_DV_BEST[k]) for k in PARAMS}
    for col in ("overrides", "overrides_json"):
        if col in row.index:
            obj = parse_json_cell(row[col])
            if obj and all(k in obj for k in PARAMS):
                return {k: fnum(obj[k], FALLBACK_DV_BEST[k]) for k in PARAMS}
    return None


def load_dv_best() -> Tuple[str, Dict[str, float], str]:
    if DV_SUMMARY.exists():
        try:
            js = json.loads(DV_SUMMARY.read_text(encoding="utf-8"))
            ov = js.get("best_overrides")
            name = js.get("best_variant") or "dv_best"
            if isinstance(ov, dict) and all(k in ov for k in PARAMS):
                return str(name), {k: fnum(ov[k], FALLBACK_DV_BEST[k]) for k in PARAMS}, str(DV_SUMMARY)
        except Exception as e:
            print(f"[{PHASE}] WARNING: could not read DV summary: {e!r}")
    return "fallback_dv_grid_r0.72_s0.42_b1.02_c1.16_sh0.165_tk1.00", dict(FALLBACK_DV_BEST), "hardcoded_fallback"


def dedupe_key(ov: Dict[str, float]) -> Tuple[float, ...]:
    return tuple(round(float(ov[k]), 6) for k in PARAMS)


def load_audit_variants(top_variants: int) -> List[Tuple[str, Dict[str, float], str]]:
    out: List[Tuple[str, Dict[str, float], str]] = []
    seen: set = set()

    best_name, best_ov, best_src = load_dv_best()
    out.append((f"dw_best_{best_name}"[:118], best_ov, best_src))
    seen.add(dedupe_key(best_ov))

    if top_variants <= 1:
        return out[:top_variants]

    # Add top DV variant summaries, especially near-capture or tangent-disciplined cases.
    if DV_VARIANTS.exists():
        try:
            df = pd.read_csv(DV_VARIANTS)
            if not df.empty:
                # Be robust to missing columns. Prefer high family score, low max tangent, and min cap near target.
                rank = pd.Series(0.0, index=df.index)
                if "dv_family_score" in df:
                    rank += df["dv_family_score"].astype(float)
                if "near_capture_rate" in df:
                    rank += 4.0 * df["near_capture_rate"].astype(float)
                if "min_worst_capture_rate" in df:
                    rank += 40.0 * df["min_worst_capture_rate"].astype(float)
                if "max_worst_tangent_ratio" in df:
                    rank -= 2.0 * np.maximum(0.0, df["max_worst_tangent_ratio"].astype(float) - 2.10)
                df = df.assign(dw_rank=rank).sort_values("dw_rank", ascending=False)
                for _, row in df.head(max(12, top_variants * 4)).iterrows():
                    ov = parse_overrides_from_row(row)
                    if ov is None:
                        continue
                    key = dedupe_key(ov)
                    if key in seen:
                        continue
                    out.append((f"dw_top_{row.get('variant', 'variant')}"[:118], ov, str(DV_VARIANTS)))
                    seen.add(key)
                    if len(out) >= top_variants:
                        return out
        except Exception as e:
            print(f"[{PHASE}] WARNING: could not read DV variants: {e!r}")

    return out[:top_variants]


def add_case_diagnostics(df: pd.DataFrame, target_cap: float, target_tan: float, relaxed_tan: float, target_q20: float) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["capture_deficit"] = np.maximum(0.0, target_cap - out["capture_rate"].astype(float))
    out["capture_margin"] = out["capture_rate"].astype(float) - target_cap
    out["tangent_excess_strict"] = np.maximum(0.0, out["tangent_ratio"].astype(float) - target_tan)
    out["tangent_excess_relaxed"] = np.maximum(0.0, out["tangent_ratio"].astype(float) - relaxed_tan)
    out["dp_deficit_vs_q20_target"] = np.maximum(0.0, target_q20 - out["dp_score"].astype(float))
    out["distance_progress_badness"] = np.maximum(0.0, 0.0025 - out["distance_progress"].astype(float))
    out["case_pass_capture"] = out["capture_rate"].astype(float) >= target_cap
    out["case_pass_tangent_strict"] = out["tangent_ratio"].astype(float) <= target_tan
    out["case_pass_tangent_relaxed"] = out["tangent_ratio"].astype(float) <= relaxed_tan
    out["case_pass_dp_local"] = out["dp_score"].astype(float) >= target_q20
    out["case_strict_line_local"] = out["case_pass_capture"] & out["case_pass_tangent_strict"] & out["case_pass_dp_local"]
    out["capture_culprit_score"] = (
        100.0 * out["capture_deficit"]
        + 2.0 * out["tangent_excess_strict"]
        + 0.6 * out["dp_deficit_vs_q20_target"]
        + 30.0 * out["distance_progress_badness"]
    )
    out["tangent_culprit_score"] = (
        3.0 * out["tangent_excess_strict"]
        + 1.5 * out["tangent_excess_relaxed"]
        + 25.0 * out["capture_deficit"]
        + 0.3 * out["dp_deficit_vs_q20_target"]
    )
    out["overall_culprit_score"] = (
        out["capture_culprit_score"]
        + out["tangent_culprit_score"]
        + 0.5 * out["dp_deficit_vs_q20_target"]
    )
    out["culprit_kind"] = np.select(
        [
            (out["capture_deficit"] > 0) & (out["tangent_excess_strict"] > 0),
            out["capture_deficit"] > 0,
            out["tangent_excess_strict"] > 0,
            out["dp_deficit_vs_q20_target"] > 0,
        ],
        ["capture+tangent", "capture", "tangent", "dp"],
        default="pass_or_minor",
    )
    return out


def summarize_by_label(case_df: pd.DataFrame) -> pd.DataFrame:
    if case_df.empty:
        return pd.DataFrame()
    group_cols = ["audit_variant", "envelope_label", "envelope_kind"]
    rows = []
    for keys, g in case_df.groupby(group_cols, sort=False):
        audit_variant, label, kind = keys
        rows.append({
            "audit_variant": audit_variant,
            "envelope_label": label,
            "envelope_kind": kind,
            "n": int(len(g)),
            "mean_capture_rate": float(g["capture_rate"].mean()),
            "min_capture_rate": float(g["capture_rate"].min()),
            "max_tangent_ratio": float(g["tangent_ratio"].max()),
            "mean_tangent_ratio": float(g["tangent_ratio"].mean()),
            "min_dp_score": float(g["dp_score"].min()),
            "mean_dp_score": float(g["dp_score"].mean()),
            "mean_distance_progress": float(g["distance_progress"].mean()),
            "capture_fail_rate": float((~g["case_pass_capture"]).mean()),
            "strict_tangent_fail_rate": float((~g["case_pass_tangent_strict"]).mean()),
            "local_strict_pass_rate": float(g["case_strict_line_local"].mean()),
            "mean_capture_culprit_score": float(g["capture_culprit_score"].mean()),
            "mean_tangent_culprit_score": float(g["tangent_culprit_score"].mean()),
            "mean_overall_culprit_score": float(g["overall_culprit_score"].mean()),
        })
    return pd.DataFrame(rows).sort_values("mean_overall_culprit_score", ascending=False)


def summarize_by_variant(run_df: pd.DataFrame, case_df: pd.DataFrame, target_cap: float, target_tan: float, relaxed_tan: float, target_q20: float) -> pd.DataFrame:
    if run_df.empty:
        return pd.DataFrame()
    rows = []
    for variant, g in run_df.groupby("audit_variant", sort=False):
        cases = case_df[case_df["audit_variant"] == variant] if not case_df.empty else pd.DataFrame()
        rows.append({
            "audit_variant": variant,
            "n_runs": int(len(g)),
            "min_worst_capture_rate": float(g["worst_capture_rate"].min()),
            "max_worst_tangent_ratio": float(g["worst_tangent_ratio"].max()),
            "min_q20_dp_score": float(g["q20_dp_score"].min()),
            "mean_dr_score": float(g["dr_score"].mean()),
            "strict_run_pass_rate": float(((g["worst_capture_rate"] >= target_cap) & (g["worst_tangent_ratio"] <= target_tan) & (g["q20_dp_score"] >= target_q20)).mean()),
            "relaxed_run_pass_rate": float(((g["worst_capture_rate"] >= target_cap) & (g["worst_tangent_ratio"] <= relaxed_tan) & (g["q20_dp_score"] >= target_q20)).mean()),
            "case_count": int(len(cases)),
            "worst_case_label_by_capture": None if cases.empty else str(cases.sort_values(["capture_rate", "tangent_ratio"], ascending=[True, False]).iloc[0].get("envelope_label")),
            "worst_case_label_by_tangent": None if cases.empty else str(cases.sort_values("tangent_ratio", ascending=False).iloc[0].get("envelope_label")),
            "mean_capture_culprit_score": None if cases.empty else float(cases["capture_culprit_score"].mean()),
            "mean_tangent_culprit_score": None if cases.empty else float(cases["tangent_culprit_score"].mean()),
            "max_overall_culprit_score": None if cases.empty else float(cases["overall_culprit_score"].max()),
            **{k: float(g[k].iloc[0]) for k in PARAMS if k in g.columns},
        })
    return pd.DataFrame(rows).sort_values(["relaxed_run_pass_rate", "min_worst_capture_rate", "max_worst_tangent_ratio"], ascending=[False, False, True])


def plot_case_rank(df: pd.DataFrame, score_col: str, value_col: str, title: str, out_path: Path, top_n: int = 20) -> None:
    if df.empty or score_col not in df:
        return
    top = df.sort_values(score_col, ascending=False).head(top_n).iloc[::-1].copy()
    labels = top.apply(lambda r: f"{r.get('audit_variant','')[:26]} | {r.get('envelope_label','')}", axis=1)
    plt.figure(figsize=(14, max(6, 0.48 * len(top))))
    plt.barh(labels, top[score_col])
    plt.xlabel(score_col)
    plt.title(title)
    for i, v in enumerate(top[value_col].astype(float).values):
        plt.text(float(top[score_col].iloc[i]), i, f"  {value_col}={v:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_capture_tangent_cases(df: pd.DataFrame, out_path: Path, target_cap: float, target_tan: float, relaxed_tan: float) -> None:
    if df.empty:
        return
    plt.figure(figsize=(10, 7))
    sc = plt.scatter(
        df["capture_rate"],
        df["tangent_ratio"],
        c=df["overall_culprit_score"],
        s=70,
        alpha=0.78,
    )
    plt.axvline(target_cap, linewidth=1.0, alpha=0.45, label="capture target")
    plt.axhline(target_tan, linewidth=1.0, alpha=0.45, label="strict tangent")
    plt.axhline(relaxed_tan, linewidth=1.0, alpha=0.25, linestyle="--", label="relaxed tangent")
    plt.colorbar(sc, label="overall culprit score")
    plt.xlabel("per-case capture_rate")
    plt.ylabel("per-case tangent/radial ratio")
    plt.title("26DW-LITE culprit audit: per-case capture vs tangent")
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

    variants = load_audit_variants(args.top_variants)

    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] device={args.device} torch={torch_info}")
    print(f"[{PHASE}] targets: strictTan<={target_tan:.2f}, relaxedTan<={relaxed_tan:.2f}, worstCap>={target_cap:.3f}, q20DP>={target_q20:.2f}")
    print(f"[{PHASE}] top_variants={args.top_variants}, seed_count={args.seed_count}, audit_repeats={args.audit_repeats}, sampled_full_tail={args.sample_full_tail}")
    print(f"[{PHASE}] audit variants:")
    for name, ov, src in variants:
        print(f"[{PHASE}]   {name} source={src} overrides={ov}")

    eval_helper = getattr(dr, "EVAL_HELPER", None)
    if eval_helper is None or not hasattr(eval_helper, "build_basins"):
        eval_helper = dr.previous_eval_helper()
    z2, rel_ids, basins = eval_helper.build_basins()
    print(f"[{PHASE}] Built basins: {len(basins)}")

    cache: Dict[Tuple[str, Tuple[float, ...]], Dict[str, Any]] = {}
    run_rows: List[Dict[str, Any]] = []
    case_rows: List[Dict[str, Any]] = []

    total = len(variants) * args.seed_count
    run_i = 0
    for vi, (variant, ov, src) in enumerate(variants, 1):
        for si in range(args.seed_count):
            run_i += 1
            seed = args.seed + 280000 + vi * 1009 + si * 7919
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
                phase_label="dw_culprit_audit",
            )
            agg["audit_variant"] = variant
            agg["seed_index"] = si + 1
            agg["dw_seed"] = seed
            agg["variant_source"] = src
            agg["base_rescue_score"] = ds_full.rescue_score(agg, target_cap, target_tan, target_q20)
            for k, v in ov.items():
                agg[k] = v
            run_rows.append(agg)

            for er in erows:
                er = dict(er)
                er["audit_variant"] = variant
                er["seed_index"] = si + 1
                er["dw_seed"] = seed
                er["variant_source"] = src
                case_rows.append(er)

            print(
                f"[{PHASE}] run {run_i:03d}/{total:03d} {variant[:58]:58s} s{si+1:02d} "
                f"DR={agg['dr_score']:.3f} q20={agg['q20_dp_score']:.3f} "
                f"worstDP={agg['worst_dp_score']:.3f} cap={agg['worst_capture_rate']:.3f} "
                f"tan={agg['worst_tangent_ratio']:.3f} strictSurv={agg.get('strict_survival_rate', 0.0):.2f}"
            )

    run_df = pd.DataFrame(run_rows)
    raw_case_df = pd.DataFrame(case_rows)
    case_df = add_case_diagnostics(raw_case_df, target_cap, target_tan, relaxed_tan, target_q20)
    label_df = summarize_by_label(case_df)
    variant_df = summarize_by_variant(run_df, case_df, target_cap, target_tan, relaxed_tan, target_q20)

    run_out = ROOT / "phase26dw_lite_run_summary.csv"
    case_out = ROOT / "phase26dw_lite_culprit_cases.csv"
    label_out = ROOT / "phase26dw_lite_culprit_by_label.csv"
    variant_out = ROOT / "phase26dw_lite_culprit_by_variant.csv"
    summary_out = ROOT / "phase26dw_lite_summary.json"

    run_df.to_csv(run_out, index=False)
    case_df.to_csv(case_out, index=False)
    label_df.to_csv(label_out, index=False)
    variant_df.to_csv(variant_out, index=False)

    plot_case_rank(case_df, "capture_culprit_score", "capture_rate", "26DW-LITE worst capture culprit cases", ROOT / "phase26dw_lite_case_rank_capture.png", args.plot_top)
    plot_case_rank(case_df, "tangent_culprit_score", "tangent_ratio", "26DW-LITE worst tangent culprit cases", ROOT / "phase26dw_lite_case_rank_tangent.png", args.plot_top)
    plot_case_rank(case_df, "overall_culprit_score", "capture_rate", "26DW-LITE overall culprit cases", ROOT / "phase26dw_lite_case_rank_overall.png", args.plot_top)
    plot_capture_tangent_cases(case_df, ROOT / "phase26dw_lite_case_capture_vs_tangent.png", target_cap, target_tan, relaxed_tan)

    top_capture = case_df.sort_values("capture_culprit_score", ascending=False).head(10).to_dict(orient="records") if not case_df.empty else []
    top_tangent = case_df.sort_values("tangent_culprit_score", ascending=False).head(10).to_dict(orient="records") if not case_df.empty else []
    top_overall = case_df.sort_values("overall_culprit_score", ascending=False).head(10).to_dict(orient="records") if not case_df.empty else []
    label_top = label_df.head(10).to_dict(orient="records") if not label_df.empty else []

    # A compact diagnosis for the next phase.
    if not case_df.empty:
        worst_cap_case = case_df.sort_values(["capture_rate", "tangent_ratio"], ascending=[True, False]).iloc[0]
        worst_tan_case = case_df.sort_values("tangent_ratio", ascending=False).iloc[0]
        capture_fail_rate = float((~case_df["case_pass_capture"]).mean())
        tangent_fail_rate = float((~case_df["case_pass_tangent_strict"]).mean())
        repeated_worst_label = str(worst_cap_case.get("envelope_label"))
        same_label_count = int((case_df["envelope_label"] == repeated_worst_label).sum()) if "envelope_label" in case_df else 0
    else:
        worst_cap_case = pd.Series(dtype=float)
        worst_tan_case = pd.Series(dtype=float)
        capture_fail_rate = 0.0
        tangent_fail_rate = 0.0
        same_label_count = 0

    if capture_fail_rate > 0.50 and tangent_fail_rate <= 0.25:
        diagnosis = "capture_floor_general_with_tangent_controlled"
    elif capture_fail_rate > 0.0 and same_label_count >= max(2, args.seed_count):
        diagnosis = "recurring_low_capture_culprit_label"
    elif tangent_fail_rate > 0.25:
        diagnosis = "tangent_tail_still_active"
    elif capture_fail_rate > 0.0:
        diagnosis = "localized_low_capture_culprit"
    else:
        diagnosis = "no_major_case_culprit_detected"

    summary = {
        "phase": PHASE,
        "title": TITLE,
        "torch_info": torch_info,
        "targets": {
            "target_worst_tangent_ratio_strict": target_tan,
            "target_worst_tangent_ratio_relaxed": relaxed_tan,
            "target_worst_capture_rate": target_cap,
            "target_q20_dp_score": target_q20,
        },
        "seed_count": int(args.seed_count),
        "top_variants": int(args.top_variants),
        "audit_repeats": int(args.audit_repeats),
        "sample_full_tail": bool(args.sample_full_tail),
        "num_variants": int(len(variants)),
        "num_runs": int(len(run_df)),
        "num_case_evals": int(len(case_df)),
        "num_cached_eval_entries": int(len(cache)),
        "elapsed_sec": float(time.time() - t0),
        "diagnosis": diagnosis,
        "capture_case_fail_rate": capture_fail_rate,
        "strict_tangent_case_fail_rate": tangent_fail_rate,
        "worst_capture_case": worst_cap_case.to_dict() if len(worst_cap_case) else None,
        "worst_tangent_case": worst_tan_case.to_dict() if len(worst_tan_case) else None,
        "top_capture_culprits": top_capture,
        "top_tangent_culprits": top_tangent,
        "top_overall_culprits": top_overall,
        "top_label_culprits": label_top,
        "variant_summary_top": variant_df.head(10).to_dict(orient="records") if not variant_df.empty else [],
        "outputs": [
            "phase26dw_lite_run_summary.csv",
            "phase26dw_lite_culprit_cases.csv",
            "phase26dw_lite_culprit_by_label.csv",
            "phase26dw_lite_culprit_by_variant.csv",
            "phase26dw_lite_summary.json",
            "phase26dw_lite_case_rank_capture.png",
            "phase26dw_lite_case_rank_tangent.png",
            "phase26dw_lite_case_rank_overall.png",
            "phase26dw_lite_case_capture_vs_tangent.png",
        ],
    }
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[{PHASE}] Done in {summary['elapsed_sec']:.1f}s")
    print(f"[{PHASE}] DIAGNOSIS: {diagnosis}")
    if len(worst_cap_case):
        print(
            f"[{PHASE}] WORST CAPTURE: variant={worst_cap_case.get('audit_variant')} label={worst_cap_case.get('envelope_label')} "
            f"kind={worst_cap_case.get('envelope_kind')} cap={float(worst_cap_case.get('capture_rate')):.3f} "
            f"tan={float(worst_cap_case.get('tangent_ratio')):.3f} dp={float(worst_cap_case.get('dp_score')):.3f}"
        )
    if len(worst_tan_case):
        print(
            f"[{PHASE}] WORST TANGENT: variant={worst_tan_case.get('audit_variant')} label={worst_tan_case.get('envelope_label')} "
            f"kind={worst_tan_case.get('envelope_kind')} cap={float(worst_tan_case.get('capture_rate')):.3f} "
            f"tan={float(worst_tan_case.get('tangent_ratio')):.3f} dp={float(worst_tan_case.get('dp_score')):.3f}"
        )
    print(f"[{PHASE}] Wrote summary: {summary_out}")
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=f"{PHASE}: {TITLE}")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="torch device preference")
    ap.add_argument("--seed", type=int, default=28000)
    ap.add_argument("--top-variants", type=int, default=3, help="number of DV variants to audit, starting with DV best")
    ap.add_argument("--seed-count", type=int, default=2, help="audit seeds per variant")
    ap.add_argument("--audit-repeats", type=int, default=1, help="micro-envelope repeats per seed")
    ap.add_argument("--sample-full-tail", action="store_true", help="include a small sampled full-envelope tail; slower")
    ap.add_argument("--relaxed-tan", type=float, default=2.30, help="relaxed tangent threshold")
    ap.add_argument("--plot-top", type=int, default=24, help="number of culprit cases to show in rank plots")
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
