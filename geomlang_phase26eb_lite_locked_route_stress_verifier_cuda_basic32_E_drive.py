#!/usr/bin/env python3
"""
Phase 26EB-LITE — locked route stress verifier / ablation audit.

Purpose
-------
26EA found that seed-consensus label routing can produce full pass-rate routes.
26EB stops searching and treats those EA routes as hypotheses to verify:

1. Rebuild the same DZ/EA variant pool.
2. Load the best EA route maps from phase26ea_lite_route_summary.csv.
3. Verify them across a wider seed battery.
4. Build a majority-consensus lock from the top EA maps.
5. Run a label-ablation audit on the best EB route to see which gates are doing
   real work and which ones are accidental noise.

This is intentionally a verification phase, not another broad optimizer.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHASE = "26EB-LITE"
TITLE = "locked EA-route stress verifier and ablation audit"

ROOT = Path(r"E:\BBIT\outputs_basic32")
EA_SCRIPT = Path(r"E:\BBIT\bbit_geomlang\geomlang_phase26ea_lite_seed_consensus_label_router_cuda_basic32_E_drive.py")
if not EA_SCRIPT.exists():
    EA_SCRIPT = Path(__file__).with_name("geomlang_phase26ea_lite_seed_consensus_label_router_cuda_basic32_E_drive.py")


def import_by_path(path: Path, module_name: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Could not find required script: {path}")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def fnum(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return default
    except Exception:
        return default


def safe_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float, np.integer, np.floating)):
        return bool(x)
    s = str(x).strip().lower()
    return s in {"true", "1", "yes", "y", "t"}


def parse_route_map(x: Any) -> Dict[str, str]:
    if isinstance(x, dict):
        return {str(k): str(v) for k, v in x.items()}
    try:
        obj = json.loads(str(x))
        if isinstance(obj, dict):
            return {str(k): str(v) for k, v in obj.items()}
    except Exception:
        pass
    return {}


def route_short_name(name: str, limit: int = 58) -> str:
    name = str(name)
    return name if len(name) <= limit else name[: limit - 3] + "..."


def load_ea_routes(args: argparse.Namespace) -> pd.DataFrame:
    path = Path(args.ea_route_summary) if args.ea_route_summary else ROOT / "phase26ea_lite_route_summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing EA route summary: {path}\n"
            "Run 26EA first, or pass --ea-route-summary PATH."
        )
    df = pd.read_csv(path)
    if "route_map_json" not in df.columns:
        raise ValueError(f"EA route summary lacks route_map_json: {path}")
    if "route_name" not in df.columns:
        df["route_name"] = [f"ea_route_{i:03d}" for i in range(len(df))]

    # Prefer genuine EA beam routes, but keep controls if the file does not contain beams.
    beam = df[df["route_name"].astype(str).str.contains("ea_beam_consensus", regex=False, na=False)].copy()
    if beam.empty:
        beam = df.copy()

    for col in ["stable_strict_all_seeds", "stable_relaxed_all_seeds"]:
        if col not in beam.columns:
            beam[col] = False
        beam[col] = beam[col].map(safe_bool)
    for col in ["strict_pass_rate", "relaxed_pass_rate", "ea_route_family_score", "min_worst_capture_rate", "max_worst_tangent_ratio"]:
        if col not in beam.columns:
            beam[col] = 0.0
        beam[col] = pd.to_numeric(beam[col], errors="coerce").fillna(0.0)

    beam = beam.sort_values(
        ["stable_strict_all_seeds", "stable_relaxed_all_seeds", "strict_pass_rate", "relaxed_pass_rate", "ea_route_family_score"],
        ascending=[False, False, False, False, False],
    )
    return beam.head(max(1, int(args.locked_route_count))).copy()


def majority_consensus_route(ea_top: pd.DataFrame, dz: Any) -> Dict[str, str]:
    """Weighted majority vote over top EA route maps."""
    weights: List[Tuple[float, Dict[str, str]]] = []
    for rank, r in enumerate(ea_top.itertuples(), 1):
        route = parse_route_map(getattr(r, "route_map_json", "{}"))
        if not route:
            continue
        w = (
            50.0 * float(getattr(r, "stable_strict_all_seeds", False))
            + 35.0 * float(getattr(r, "stable_relaxed_all_seeds", False))
            + 12.0 * fnum(getattr(r, "strict_pass_rate", 0.0), 0.0)
            + 8.0 * fnum(getattr(r, "relaxed_pass_rate", 0.0), 0.0)
            + 0.25 * fnum(getattr(r, "ea_route_family_score", 0.0), 0.0)
            + max(0.0, 8.0 - rank)
        )
        weights.append((w, route))
    if not weights:
        return {}

    all_labels = set(["*"])
    try:
        all_labels.update([str(x) for x in getattr(dz, "CRITICAL_LABELS", [])])
    except Exception:
        pass
    for _, route in weights:
        all_labels.update(route.keys())

    out: Dict[str, str] = {}
    for lab in sorted(all_labels, key=lambda x: (x != "*", x)):
        votes: Dict[str, float] = {}
        for w, route in weights:
            if lab in route:
                votes[route[lab]] = votes.get(route[lab], 0.0) + w
            elif lab != "*" and "*" in route:
                # a default route implicitly votes for the default on labels it does not override
                votes[route["*"]] = votes.get(route["*"], 0.0) + 0.45 * w
        if votes:
            out[lab] = max(votes.items(), key=lambda kv: kv[1])[0]
    # Remove explicit labels that equal the default to keep route JSON clean.
    default = out.get("*")
    if default:
        out = {k: v for k, v in out.items() if k == "*" or v != default}
    return out


def build_locked_routes(ea_top: pd.DataFrame, dz: Any, args: argparse.Namespace) -> List[Tuple[str, Dict[str, str], str]]:
    routes: List[Tuple[str, Dict[str, str], str]] = []
    seen = set()
    for i, r in enumerate(ea_top.itertuples(), 0):
        route = parse_route_map(getattr(r, "route_map_json", "{}"))
        if not route:
            continue
        key = tuple(sorted(route.items()))
        if key in seen:
            continue
        seen.add(key)
        old_name = str(getattr(r, "route_name", f"ea_route_{i:03d}"))
        src = (
            f"locked_from={old_name}; "
            f"ea_strict={fnum(getattr(r, 'strict_pass_rate', 0.0), 0.0):.3f}; "
            f"ea_relaxed={fnum(getattr(r, 'relaxed_pass_rate', 0.0), 0.0):.3f}; "
            f"ea_score={fnum(getattr(r, 'ea_route_family_score', 0.0), 0.0):.3f}"
        )
        routes.append((f"eb_locked_ea_{i:03d}", route, src))

    maj = majority_consensus_route(ea_top, dz)
    if maj:
        key = tuple(sorted(maj.items()))
        if key not in seen:
            routes.insert(0, ("eb_majority_consensus_lock", maj, "weighted_majority_of_top_EA_locked_routes"))
            seen.add(key)

    # Also add a compact default-only control from the best route's default variant.
    if routes:
        default = routes[0][1].get("*")
        if default:
            r = {"*": default}
            key = tuple(sorted(r.items()))
            if key not in seen and args.include_default_control:
                routes.append(("eb_default_only_control", r, "default_only_from_best_locked_route"))

    return routes[: max(1, int(args.max_routes))]


def summarize_eb(dz: Any, results_df: pd.DataFrame, target_cap: float, target_tan: float, relaxed_tan: float, target_q20: float) -> pd.DataFrame:
    out = dz.summarize_routes(results_df, target_cap, target_tan, relaxed_tan, target_q20)
    if out.empty:
        return out
    if "dz_route_family_score" in out.columns:
        out = out.rename(columns={"dz_route_family_score": "eb_route_family_score"})
    elif "ea_route_family_score" in out.columns:
        out = out.rename(columns={"ea_route_family_score": "eb_route_family_score"})
    if "eb_route_family_score" not in out.columns:
        out["eb_route_family_score"] = 0.0
    return out.sort_values(
        ["stable_strict_all_seeds", "stable_relaxed_all_seeds", "strict_pass_rate", "relaxed_pass_rate", "eb_route_family_score"],
        ascending=[False, False, False, False, False],
    )


def plot_route_scores(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    top = df.sort_values("eb_route_family_score", ascending=False).head(25).iloc[::-1]
    plt.figure(figsize=(14, max(6, 0.5 * len(top))))
    plt.barh(top["route_name"], top["eb_route_family_score"])
    plt.xlabel("EB locked-route family score")
    plt.title("26EB-LITE locked EA-route stress scores")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_pass_rates(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    top = df.sort_values(["relaxed_pass_rate", "strict_pass_rate", "eb_route_family_score"], ascending=[False, False, False]).head(25).iloc[::-1]
    y = np.arange(len(top))
    plt.figure(figsize=(14, max(6, 0.5 * len(top))))
    plt.barh(y - 0.13, top["relaxed_pass_rate"], height=0.25, label="relaxed")
    plt.barh(y + 0.13, top["strict_pass_rate"], height=0.25, label="strict")
    plt.yticks(y, top["route_name"])
    plt.xlabel("pass rate across verification seeds")
    plt.title("26EB-LITE locked-route pass rates")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_capture_tangent(df: pd.DataFrame, out_path: Path, target_cap: float, target_tan: float, relaxed_tan: float) -> None:
    if df.empty:
        return
    c = df.get("dz_route_score_run", pd.Series(np.zeros(len(df))))
    plt.figure(figsize=(10, 8))
    plt.scatter(df["worst_capture_rate"], df["worst_tangent_ratio"], c=c, s=85, alpha=0.85)
    plt.axvline(target_cap, linewidth=0.8, label="capture target")
    plt.axhline(target_tan, linewidth=0.8, label="strict tangent")
    plt.axhline(relaxed_tan, linewidth=0.8, linestyle="--", label="relaxed tangent")
    plt.xlabel("verified worst capture_rate")
    plt.ylabel("verified worst tangent/radial ratio")
    plt.title("26EB-LITE locked routes: capture vs tangent")
    plt.colorbar(label="EB run score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_ablation(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    top = df.sort_values("delta_family_score", ascending=True).head(30).iloc[::-1]
    plt.figure(figsize=(14, max(6, 0.5 * len(top))))
    plt.barh(top["ablation_name"], top["delta_family_score"])
    plt.axvline(0.0, linewidth=0.8)
    plt.xlabel("family score delta vs locked route")
    plt.title("26EB-LITE label ablation impact")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_label_failures(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty or "envelope_label" not in df.columns:
        return
    rows = []
    for lab, g in df.groupby("envelope_label", sort=False):
        rows.append({
            "envelope_label": str(lab),
            "case_n": int(len(g)),
            "min_capture": fnum(g.get("capture_rate", pd.Series([np.nan])).min()),
            "max_tangent": fnum(g.get("tangent_ratio", pd.Series([np.nan])).max()),
            "mean_case_score": fnum(g.get("case_score", pd.Series([np.nan])).mean(), 0.0),
        })
    lab_df = pd.DataFrame(rows)
    if lab_df.empty:
        return
    # high tangent + low capture labels are the remaining suspicious labels.
    lab_df["risk"] = (2.30 - lab_df["min_capture"]).clip(lower=0) + lab_df["max_tangent"].clip(lower=0)
    top = lab_df.sort_values(["max_tangent", "min_capture"], ascending=[False, True]).head(20).iloc[::-1]
    plt.figure(figsize=(14, max(6, 0.5 * len(top))))
    plt.barh(top["envelope_label"], top["max_tangent"])
    for i, r in enumerate(top.itertuples()):
        plt.text(float(r.max_tangent), i, f" cap={r.min_capture:.3f}", va="center", fontsize=8)
    plt.xlabel("max tangent/radial ratio in best locked route")
    plt.title("26EB-LITE remaining label stress inside best route")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def evaluate_routes(dz: Any, routes: List[Tuple[str, Dict[str, str], str]], pool_seed_cases: Dict[int, Dict[str, pd.DataFrame]], target_cap: float, target_tan: float, relaxed_tan: float, target_q20: float, print_every: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    route_rows: List[Dict[str, Any]] = []
    routed_case_rows: List[pd.DataFrame] = []
    seeds = sorted(pool_seed_cases)
    for route_i, (route_name, route_map, route_src) in enumerate(routes, 1):
        for si, seed in enumerate(seeds, 1):
            metrics, cases = dz.compose_route(route_name, route_map, pool_seed_cases[seed], target_cap, target_tan, relaxed_tan, target_q20)
            metrics["route_source"] = route_src
            metrics["seed_index"] = si
            metrics["dz_seed"] = seed
            metrics["route_map_json"] = json.dumps(route_map, sort_keys=True)
            route_rows.append(metrics)
            if not cases.empty:
                cases = cases.copy()
                cases["route_source"] = route_src
                cases["seed_index"] = si
                cases["dz_seed"] = seed
                cases["route_map_json"] = json.dumps(route_map, sort_keys=True)
                routed_case_rows.append(cases)
            if route_i <= 5 or route_i % print_every == 0 or si == len(seeds):
                print(
                    f"[{PHASE}] route {route_i:03d}/{len(routes):03d} {route_short_name(route_name):58s} s{si:02d} "
                    f"RELAX={int(metrics.get('pass_relaxed_line', False))} STRICT={int(metrics.get('pass_strict_line', False))} "
                    f"cap={fnum(metrics.get('worst_capture_rate')):.3f} tan={fnum(metrics.get('worst_tangent_ratio')):.3f} "
                    f"q20={fnum(metrics.get('q20_dp_score')):.3f} score={fnum(metrics.get('dz_route_score_run')):.3f}"
                )
    return pd.DataFrame(route_rows), pd.concat(routed_case_rows, ignore_index=True) if routed_case_rows else pd.DataFrame()


def build_ablation_routes(best_name: str, best_route: Dict[str, str], max_labels: int = 24) -> List[Tuple[str, Dict[str, str], str]]:
    routes = [("ablate_NONE_locked_baseline", dict(best_route), f"baseline={best_name}")]
    explicit_labels = [k for k in best_route.keys() if k != "*"][:max_labels]
    for lab in explicit_labels:
        r = dict(best_route)
        removed = r.pop(lab, None)
        routes.append((f"ablate_drop_{lab}", r, f"drop_label={lab}; removed_variant={removed}; baseline={best_name}"))
    if "*" in best_route:
        default_only = {"*": best_route["*"]}
        routes.append(("ablate_default_only", default_only, f"default_only_from={best_name}"))
    return routes


def run(args: argparse.Namespace) -> None:
    t0 = time.time()
    ROOT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    ea = import_by_path(EA_SCRIPT, "phase26ea_lite")
    dz = ea.load_dz()

    if args.device:
        sys.argv = [sys.argv[0], "--device", args.device]

    ds_lite = dz.import_by_path(dz.DS_LITE_FILE, "phase26ds_lite")
    drfast = dz.import_by_path(dz.DR_FAST_FILE, "phase26dr_fast")
    dr = drfast.import_phase26dr()

    torch_info = drfast.setup_torch(args.device)
    eval_helper = getattr(dr, "EVAL_HELPER", None)
    if eval_helper is None or not hasattr(eval_helper, "build_basins"):
        eval_helper = dr.previous_eval_helper()
    _z2, _rel_ids, basins = eval_helper.build_basins()

    target_cap = float(args.target_capture)
    target_tan = float(args.target_tangent)
    relaxed_tan = float(args.relaxed_tangent)
    target_q20 = float(args.target_q20)

    ea_top = load_ea_routes(args)
    routes = build_locked_routes(ea_top, dz, args)
    if not routes:
        raise RuntimeError("No valid locked routes could be built from EA summary.")

    print(f"[{PHASE}] imported EA from: {EA_SCRIPT}")
    print(f"[{PHASE}] loaded EA locked route hypotheses={len(routes)}")
    for name, route, src in routes[:8]:
        print(f"[{PHASE}] route-hypothesis {name}: labels={len(route)} source={src[:140]}")

    base_pool = dz.variant_pool(args.pool_size)
    pool = ea.jitter_variants(dz, base_pool, args.extra_jitters, rng)[: max(4, args.pool_size + args.extra_jitters)]
    print(f"[{PHASE}] rebuilt pool variants={len(pool)} base={len(base_pool)} extra={max(0, len(pool)-len(base_pool))}")

    seeds = [args.seed + i for i in range(args.verify_seed_count)]
    cache: Dict[Any, Any] = {}
    pool_seed_cases: Dict[int, Dict[str, pd.DataFrame]] = {}
    all_pool_case_rows: List[pd.DataFrame] = []

    for seed in seeds:
        print(f"[{PHASE}] evaluating verification pool for seed={seed}")
        seed_cases = dz.evaluate_pool_for_seed(ds_lite, drfast, dr, pool, basins, seed, args.audit_repeats, args.sample_full_tail, cache)
        pool_seed_cases[seed] = seed_cases
        for df in seed_cases.values():
            if not df.empty:
                all_pool_case_rows.append(df)

    route_results_df, routed_cases_df = evaluate_routes(
        dz, routes, pool_seed_cases, target_cap, target_tan, relaxed_tan, target_q20, args.print_every
    )
    route_summary_df = summarize_eb(dz, route_results_df, target_cap, target_tan, relaxed_tan, target_q20)
    pool_cases_df = pd.concat(all_pool_case_rows, ignore_index=True) if all_pool_case_rows else pd.DataFrame()

    # Best-route ablation audit.
    ablation_results_df = pd.DataFrame()
    ablation_summary_df = pd.DataFrame()
    best_route_map: Dict[str, str] = {}
    best_route_name = None
    if not route_summary_df.empty:
        best = route_summary_df.iloc[0]
        best_route_name = str(best.get("route_name"))
        best_route_map = parse_route_map(best.get("route_map_json", "{}"))
        ablation_routes = build_ablation_routes(best_route_name, best_route_map, args.max_ablation_labels)
        print(f"[{PHASE}] running ablation audit on {best_route_name}: explicit_labels={max(0, len(best_route_map)-1)}")
        ablation_results_df, _ = evaluate_routes(
            dz, ablation_routes, pool_seed_cases, target_cap, target_tan, relaxed_tan, target_q20, args.print_every
        )
        ablation_summary_df = summarize_eb(dz, ablation_results_df, target_cap, target_tan, relaxed_tan, target_q20)
        if not ablation_summary_df.empty:
            baseline = ablation_summary_df[ablation_summary_df["route_name"] == "ablate_NONE_locked_baseline"]
            base_score = fnum(baseline.iloc[0].get("eb_route_family_score"), 0.0) if len(baseline) else fnum(ablation_summary_df.iloc[0].get("eb_route_family_score"), 0.0)
            ablation_summary_df["delta_family_score"] = ablation_summary_df["eb_route_family_score"] - base_score
            ablation_summary_df["ablation_name"] = ablation_summary_df["route_name"]

    # Outputs.
    pool_cases_df.to_csv(ROOT / "phase26eb_lite_pool_case_results.csv", index=False)
    routed_cases_df.to_csv(ROOT / "phase26eb_lite_route_cases.csv", index=False)
    route_results_df.to_csv(ROOT / "phase26eb_lite_route_results.csv", index=False)
    route_summary_df.to_csv(ROOT / "phase26eb_lite_route_summary.csv", index=False)
    ea_top.to_csv(ROOT / "phase26eb_lite_imported_ea_routes.csv", index=False)
    ablation_results_df.to_csv(ROOT / "phase26eb_lite_ablation_results.csv", index=False)
    ablation_summary_df.to_csv(ROOT / "phase26eb_lite_ablation_summary.csv", index=False)

    plot_route_scores(route_summary_df, ROOT / "phase26eb_lite_route_scores.png")
    plot_pass_rates(route_summary_df, ROOT / "phase26eb_lite_pass_rates.png")
    plot_capture_tangent(route_results_df, ROOT / "phase26eb_lite_route_capture_vs_tangent.png", target_cap, target_tan, relaxed_tan)
    plot_ablation(ablation_summary_df, ROOT / "phase26eb_lite_ablation_impact.png")
    if best_route_name and not routed_cases_df.empty:
        best_cases = routed_cases_df[routed_cases_df["route_name"] == best_route_name] if "route_name" in routed_cases_df.columns else routed_cases_df
        plot_label_failures(best_cases, ROOT / "phase26eb_lite_best_route_label_stress.png")

    best = route_summary_df.iloc[0] if len(route_summary_df) else pd.Series(dtype=object)
    stable_strict = route_summary_df[route_summary_df["stable_strict_all_seeds"] == True] if not route_summary_df.empty else pd.DataFrame()
    stable_relaxed = route_summary_df[route_summary_df["stable_relaxed_all_seeds"] == True] if not route_summary_df.empty else pd.DataFrame()

    if len(stable_strict):
        verdict = "locked_route_survives_strict_stress_verification"
    elif len(stable_relaxed):
        verdict = "locked_route_survives_relaxed_stress_verification"
    elif len(best) and fnum(best.get("relaxed_pass_rate"), 0.0) >= args.min_promote_rate:
        verdict = "locked_route_partially_survives_stress_verification"
    else:
        verdict = "locked_route_fails_stress_verification"

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
        "seed_start": int(args.seed),
        "verify_seed_count": int(args.verify_seed_count),
        "audit_repeats": int(args.audit_repeats),
        "sample_full_tail": bool(args.sample_full_tail),
        "pool_size": int(len(pool)),
        "base_pool_size": int(len(base_pool)),
        "extra_jitters": int(max(0, len(pool) - len(base_pool))),
        "locked_route_count": int(len(routes)),
        "num_stable_strict_routes": int(len(stable_strict)),
        "num_stable_relaxed_routes": int(len(stable_relaxed)),
        "num_cached_eval_entries": int(len(cache)),
        "elapsed_sec": float(time.time() - t0),
        "verdict": verdict,
        "best_route": None if not len(best) else str(best.get("route_name")),
        "best_route_source": None if not len(best) else str(best.get("route_source", "")),
        "best_route_family_score": None if not len(best) else fnum(best.get("eb_route_family_score")),
        "best_strict_pass_rate": None if not len(best) else fnum(best.get("strict_pass_rate")),
        "best_relaxed_pass_rate": None if not len(best) else fnum(best.get("relaxed_pass_rate")),
        "best_min_worst_capture_rate": None if not len(best) else fnum(best.get("min_worst_capture_rate")),
        "best_max_worst_tangent_ratio": None if not len(best) else fnum(best.get("max_worst_tangent_ratio")),
        "best_min_q20_dp_score": None if not len(best) else fnum(best.get("min_q20_dp_score")),
        "best_route_map": best_route_map,
        "route_top10": route_summary_df.head(10).to_dict(orient="records") if len(route_summary_df) else [],
        "ablation_top10": ablation_summary_df.head(10).to_dict(orient="records") if len(ablation_summary_df) else [],
        "outputs": [
            "phase26eb_lite_pool_case_results.csv",
            "phase26eb_lite_route_cases.csv",
            "phase26eb_lite_route_results.csv",
            "phase26eb_lite_route_summary.csv",
            "phase26eb_lite_imported_ea_routes.csv",
            "phase26eb_lite_ablation_results.csv",
            "phase26eb_lite_ablation_summary.csv",
            "phase26eb_lite_summary.json",
            "phase26eb_lite_route_scores.png",
            "phase26eb_lite_pass_rates.png",
            "phase26eb_lite_route_capture_vs_tangent.png",
            "phase26eb_lite_ablation_impact.png",
            "phase26eb_lite_best_route_label_stress.png",
        ],
    }
    (ROOT / "phase26eb_lite_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[{PHASE}] verdict={verdict}")
    print(
        f"[{PHASE}] best_route={summary['best_route']} strict={summary['best_strict_pass_rate']} "
        f"relaxed={summary['best_relaxed_pass_rate']} cap={summary['best_min_worst_capture_rate']} "
        f"tan={summary['best_max_worst_tangent_ratio']}"
    )
    print(f"[{PHASE}] wrote outputs to {ROOT}")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=f"{PHASE}: {TITLE}")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"], help="Torch device for imported CUDA-capable phases")
    ap.add_argument("--seed", type=int, default=20, help="First verification seed. Default avoids reusing EA seeds 0..2.")
    ap.add_argument("--verify-seed-count", type=int, default=2, help="Number of verification seeds for stress test")
    ap.add_argument("--pool-size", type=int, default=8, help="Base DZ pool size before EA jitters")
    ap.add_argument("--extra-jitters", type=int, default=4, help="Additional narrow EA local probes to rebuild EA pool")
    ap.add_argument("--audit-repeats", type=int, default=1)
    ap.add_argument("--sample-full-tail", action="store_true")
    ap.add_argument("--ea-route-summary", default="", help="Optional path to phase26ea_lite_route_summary.csv")
    ap.add_argument("--locked-route-count", type=int, default=6, help="Top EA routes to import")
    ap.add_argument("--max-routes", type=int, default=8, help="Max locked routes to verify after adding majority/control")
    ap.add_argument("--include-default-control", action="store_true", help="Also test default-only route from best lock")
    ap.add_argument("--max-ablation-labels", type=int, default=18)
    ap.add_argument("--print-every", type=int, default=10)
    ap.add_argument("--target-capture", type=float, default=0.285)
    ap.add_argument("--target-tangent", type=float, default=2.10)
    ap.add_argument("--relaxed-tangent", type=float, default=2.30)
    ap.add_argument("--target-q20", type=float, default=2.25)
    ap.add_argument("--min-promote-rate", type=float, default=0.80)
    return ap


if __name__ == "__main__":
    run(build_argparser().parse_args())
