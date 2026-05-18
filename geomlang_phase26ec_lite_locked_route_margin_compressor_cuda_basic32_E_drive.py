#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 26EC-LITE: Locked-route margin compressor / verifier for Basic32 geomlang.

Why EC exists after EB
----------------------
EB found the important thing: the EA-derived locked route can survive strict
verification.  But EB also showed two remaining risks:

  1. The route is almost certainly over-specified.  Some labels can be removed
     from the explicit route without hurting the locked result, while a few
     labels are essential.
  2. The strict pass is close enough to the tangent line that we should not only
     ask "does it pass?"; we should ask "how much margin does it have, and can a
     smaller locked route keep the pass?"

EC therefore performs a pragmatic lock-in pass:

  - imports the EB best route and the EB ablation summary if present;
  - rebuilds the same pool and verification seeds through the DZ/EA machinery;
  - creates compressed route hypotheses by dropping low-impact labels first;
  - adds per-label swap hypotheses for weak labels using the measured pool;
  - scores every candidate with stricter margin metrics;
  - emits a final locked-route candidate plus a label-retention audit.

This is still "lite": it reuses the CUDA pool evaluator from DS-LITE through DZ,
keeps the candidate count bounded, and focuses on route-map structure rather
than doing a full new global search.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
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

PHASE = "26EC-LITE"
TITLE = "Locked-route margin compressor and strict-stability verifier"

ROOT = Path(r"E:\BBIT\outputs_basic32")
EB_SCRIPT = Path(r"E:\BBIT\bbit_geomlang\geomlang_phase26eb_lite_locked_route_stress_verifier_cuda_basic32_E_drive.py")
EA_SCRIPT = Path(r"E:\BBIT\bbit_geomlang\geomlang_phase26ea_lite_seed_consensus_label_router_cuda_basic32_E_drive.py")

# Notebook/container fallback so the file can be inspected outside the E: drive.
if not EB_SCRIPT.exists():
    EB_SCRIPT = Path(__file__).with_name("geomlang_phase26eb_lite_locked_route_stress_verifier_cuda_basic32_E_drive.py")
if not EA_SCRIPT.exists():
    EA_SCRIPT = Path(__file__).with_name("geomlang_phase26ea_lite_seed_consensus_label_router_cuda_basic32_E_drive.py")
if not ROOT.exists() and Path("/mnt/data").exists():
    ROOT = Path("/mnt/data")


def import_by_path(path: Path, name: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing import target: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def fnum(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def bool_rate(s: pd.Series) -> float:
    if s is None or len(s) == 0:
        return 0.0
    return float(pd.Series(s).fillna(False).astype(bool).mean())


def parse_route_map(x: Any) -> Dict[str, str]:
    if isinstance(x, dict):
        return {str(k): str(v) for k, v in x.items()}
    if x is None:
        return {}
    try:
        obj = json.loads(str(x))
        if isinstance(obj, dict):
            return {str(k): str(v) for k, v in obj.items()}
    except Exception:
        pass
    return {}


def route_short_name(name: str, max_len: int = 72) -> str:
    name = str(name)
    return name if len(name) <= max_len else name[: max_len - 3] + "..."


def read_json(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def read_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def load_eb_best_route() -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    summary = read_json(ROOT / "phase26eb_lite_summary.json")
    route_name = str(summary.get("best_route") or "eb_imported_locked_route")
    route_map = parse_route_map(summary.get("best_route_map"))
    if not route_map:
        rs = read_csv(ROOT / "phase26eb_lite_route_summary.csv")
        if not rs.empty:
            row = rs.iloc[0]
            route_name = str(row.get("route_name", route_name))
            route_map = parse_route_map(row.get("route_map_json", "{}"))
    if not route_map:
        raise RuntimeError("Could not load EB best route map. Run 26EB first or keep phase26eb_lite_summary.json in outputs_basic32.")
    return route_name, route_map, summary


def load_low_impact_labels(args: argparse.Namespace, best_route: Dict[str, str]) -> List[str]:
    ab = read_csv(ROOT / "phase26eb_lite_ablation_summary.csv")
    explicit = [k for k in best_route.keys() if k != "*"]
    if ab.empty or "route_name" not in ab.columns:
        return []
    rows = []
    for _, r in ab.iterrows():
        rn = str(r.get("route_name", ""))
        if not rn.startswith("ablate_drop_"):
            continue
        lab = rn[len("ablate_drop_"):]
        if lab not in explicit:
            continue
        delta = fnum(r.get("delta_family_score"), float("nan"))
        strict = fnum(r.get("strict_pass_rate"), 0.0)
        relaxed = fnum(r.get("relaxed_pass_rate"), 0.0)
        max_tan = fnum(r.get("max_worst_tangent_ratio"), 999.0)
        min_cap = fnum(r.get("min_worst_capture_rate"), -999.0)
        # Drop labels that EB already suggested are non-essential: small score
        # damage, or full strict still holds after removal.
        keep_drop = (delta >= -abs(args.drop_delta_tolerance)) or (strict >= args.min_strict_rate_for_drop and relaxed >= 1.0)
        if keep_drop and min_cap >= args.drop_min_capture and max_tan <= args.drop_max_tangent:
            rows.append((lab, delta, strict, relaxed, min_cap, max_tan))
    rows.sort(key=lambda x: (x[1], x[2], -x[5]), reverse=True)
    return [r[0] for r in rows[: args.max_drop_labels]]


def build_compressed_routes(best_name: str, best_route: Dict[str, str], low_labels: Sequence[str]) -> List[Tuple[str, Dict[str, str], str]]:
    routes: List[Tuple[str, Dict[str, str], str]] = []
    routes.append(("ec_locked_full_eb_route", dict(best_route), f"full_import_from={best_name}"))
    if not low_labels:
        return routes

    # Individual drops.
    for lab in low_labels:
        if lab in best_route:
            r = dict(best_route)
            old = r.pop(lab)
            routes.append((f"ec_drop_{lab}", r, f"drop_one={lab}; removed={old}"))

    # Greedy cumulative drops in EB-ablation order.
    cumulative = dict(best_route)
    removed = []
    for lab in low_labels:
        if lab in cumulative:
            removed.append(f"{lab}:{cumulative.pop(lab)}")
            routes.append((f"ec_compress_drop_{len(removed):02d}", dict(cumulative), "cumulative_drop=" + ";".join(removed)))

    # Default-only plus explicit essentials.  This tests whether the route is
    # really a small set of exceptions over a default carrier.
    if "*" in best_route:
        essential = {k: v for k, v in best_route.items() if k == "*" or k not in set(low_labels)}
        routes.append(("ec_essential_labels_only", essential, f"explicit_kept={len(essential)-1}; dropped={len(low_labels)}"))
    return routes


def label_metrics_from_variant(dz: Any, df: pd.DataFrame, target_cap: float, target_tan: float, relaxed_tan: float, target_q20: float) -> Dict[str, float]:
    metrics, _ = dz.score_cases(df, target_cap, target_tan, relaxed_tan, target_q20)
    return {k: fnum(metrics.get(k)) for k in ["worst_capture_rate", "worst_tangent_ratio", "q20_dp_score", "dz_route_score_run"]}


def build_swap_routes(dz: Any, base_route: Dict[str, str], pool_seed_cases: Dict[int, Dict[str, pd.DataFrame]], args: argparse.Namespace) -> List[Tuple[str, Dict[str, str], str]]:
    """Build bounded per-label swap hypotheses from measured pool cases.

    For weak labels, choose variants that are not necessarily global winners but
    are label-local winners under the current thresholds.  This is the route-map
    equivalent of moving one stone at a time instead of redesigning the bridge.
    """
    if not pool_seed_cases:
        return []
    first_seed = sorted(pool_seed_cases)[0]
    cases_by_variant = pool_seed_cases[first_seed]
    default_variant = base_route.get("*") or next(iter(cases_by_variant.keys()))

    # Determine weak labels from current locked route on first seed.
    _, base_cases = dz.compose_route("ec_swap_probe", base_route, cases_by_variant, args.target_capture, args.target_tangent, args.relaxed_tangent, args.target_q20)
    if base_cases.empty or "envelope_label" not in base_cases.columns:
        return []
    weak = []
    for lab, g in base_cases.groupby(base_cases["envelope_label"].astype(str)):
        cap = fnum(g.get("capture_rate", pd.Series([np.nan])).min())
        tan = fnum(g.get("tangent_ratio", pd.Series([np.nan])).max())
        if cap < args.swap_capture_floor or tan > args.swap_tangent_ceiling:
            weak.append((lab, cap, tan))
    weak.sort(key=lambda x: (x[2], -x[1]), reverse=True)
    weak = weak[: args.max_swap_labels]

    out: List[Tuple[str, Dict[str, str], str]] = []
    for lab, old_cap, old_tan in weak:
        rows = []
        for vname, df in cases_by_variant.items():
            if df.empty or "envelope_label" not in df:
                continue
            sub = df[df["envelope_label"].astype(str) == lab]
            if sub.empty:
                continue
            m = label_metrics_from_variant(dz, sub, args.target_capture, args.target_tangent, args.relaxed_tangent, args.target_q20)
            cap = fnum(m.get("worst_capture_rate"))
            tan = fnum(m.get("worst_tangent_ratio"))
            q20 = fnum(m.get("q20_dp_score"), 0.0)
            score = fnum(m.get("dz_route_score_run"), 0.0)
            if cap >= args.swap_min_candidate_capture and tan <= args.swap_max_candidate_tangent:
                rows.append((vname, cap, tan, q20, score))
        rows.sort(key=lambda x: (x[4], x[1], -x[2], x[3]), reverse=True)
        for rank, (vname, cap, tan, q20, score) in enumerate(rows[: args.max_swaps_per_label], 1):
            old = base_route.get(lab, default_variant)
            if vname == old:
                continue
            r = dict(base_route)
            r[lab] = vname
            out.append((
                f"ec_swap_{lab}_{rank:02d}",
                r,
                f"swap_label={lab}; old={old}; new={vname}; old_cap={old_cap:.3f}; old_tan={old_tan:.3f}; new_cap={cap:.3f}; new_tan={tan:.3f}; new_score={score:.3f}",
            ))
    return out


def summarize_ec(route_results: pd.DataFrame, target_cap: float, target_tan: float, relaxed_tan: float, margin_tan: float, margin_cap: float) -> pd.DataFrame:
    if route_results.empty:
        return pd.DataFrame()
    rows = []
    for rn, g in route_results.groupby("route_name", dropna=False):
        min_cap = fnum(g.get("worst_capture_rate", pd.Series([np.nan])).min())
        max_tan = fnum(g.get("worst_tangent_ratio", pd.Series([np.nan])).max())
        min_q20 = fnum(g.get("q20_dp_score", pd.Series([np.nan])).min())
        strict_rate = bool_rate(g.get("pass_strict_line", pd.Series(dtype=bool)))
        relaxed_rate = bool_rate(g.get("pass_relaxed_line", pd.Series(dtype=bool)))
        route_map_json = str(g["route_map_json"].iloc[0]) if "route_map_json" in g and len(g) else "{}"
        route = parse_route_map(route_map_json)
        explicit = max(0, len([k for k in route if k != "*"]))
        # Margin is not the same as score.  We reward strict stability first,
        # then capture/tangent margin, then smaller route maps.
        cap_margin = min_cap - target_cap
        tan_margin = target_tan - max_tan
        relaxed_margin = relaxed_tan - max_tan
        margin_ok = (min_cap >= target_cap + margin_cap) and (max_tan <= target_tan - margin_tan)
        ec_score = (
            100.0 * strict_rate
            + 20.0 * relaxed_rate
            + 40.0 * min(0.05, max(-0.05, cap_margin))
            + 20.0 * min(0.40, max(-0.40, tan_margin))
            + 3.0 * min_q20
            - 0.12 * explicit
        )
        rows.append({
            "route_name": rn,
            "route_source": str(g["route_source"].iloc[0]) if "route_source" in g and len(g) else "",
            "route_map_json": route_map_json,
            "explicit_label_count": explicit,
            "num_seeds": int(len(g)),
            "strict_pass_rate": strict_rate,
            "relaxed_pass_rate": relaxed_rate,
            "stable_strict_all_seeds": bool(strict_rate >= 1.0),
            "stable_relaxed_all_seeds": bool(relaxed_rate >= 1.0),
            "strict_margin_all_seeds": bool(margin_ok),
            "min_worst_capture_rate": min_cap,
            "max_worst_tangent_ratio": max_tan,
            "min_q20_dp_score": min_q20,
            "capture_margin_vs_target": cap_margin,
            "tangent_margin_vs_strict": tan_margin,
            "tangent_margin_vs_relaxed": relaxed_margin,
            "ec_margin_score": ec_score,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["stable_strict_all_seeds", "strict_margin_all_seeds", "ec_margin_score", "explicit_label_count"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def plot_capture_tangent(df: pd.DataFrame, out_path: Path, target_cap: float, target_tan: float, relaxed_tan: float) -> None:
    if df.empty:
        return
    plt.figure(figsize=(10, 7))
    x = df.get("worst_capture_rate", pd.Series(dtype=float)).astype(float)
    y = df.get("worst_tangent_ratio", pd.Series(dtype=float)).astype(float)
    c = df.get("ec_margin_score", df.get("dz_route_score_run", pd.Series(np.zeros(len(df))))).astype(float)
    plt.scatter(x, y, c=c, s=70, alpha=0.85)
    plt.axvline(target_cap, linewidth=1, label="capture target")
    plt.axhline(target_tan, linewidth=1, label="strict tangent")
    plt.axhline(relaxed_tan, linewidth=1, linestyle="--", label="relaxed tangent")
    plt.xlabel("verified worst capture_rate")
    plt.ylabel("verified worst tangent/radial ratio")
    plt.title("26EC-LITE compressed/locked routes: capture vs tangent")
    plt.colorbar(label="EC run score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_route_scores(summary: pd.DataFrame, out_path: Path, title: str = "26EC-LITE route margin scores") -> None:
    if summary.empty:
        return
    top = summary.head(18).iloc[::-1]
    plt.figure(figsize=(14, max(6, 0.5 * len(top))))
    plt.barh(top["route_name"], top["ec_margin_score"])
    for i, r in enumerate(top.itertuples()):
        plt.text(float(r.ec_margin_score), i, f" cap={r.min_worst_capture_rate:.3f} tan={r.max_worst_tangent_ratio:.3f} labels={int(r.explicit_label_count)}", va="center", fontsize=8)
    plt.xlabel("EC margin/compression score")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_pass_rates(summary: pd.DataFrame, out_path: Path) -> None:
    if summary.empty:
        return
    top = summary.head(18).iloc[::-1]
    y = np.arange(len(top))
    plt.figure(figsize=(14, max(6, 0.5 * len(top))))
    plt.barh(y - 0.18, top["relaxed_pass_rate"].astype(float), height=0.36, label="relaxed")
    plt.barh(y + 0.18, top["strict_pass_rate"].astype(float), height=0.36, label="strict")
    plt.yticks(y, top["route_name"])
    plt.xlim(0, 1.05)
    plt.xlabel("pass rate across verification seeds")
    plt.title("26EC-LITE locked/compressed route pass rates")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_label_stress(cases: pd.DataFrame, out_path: Path) -> None:
    if cases.empty or "envelope_label" not in cases:
        return
    rows = []
    for lab, g in cases.groupby(cases["envelope_label"].astype(str)):
        rows.append({
            "label": lab,
            "min_cap": fnum(g.get("capture_rate", pd.Series([np.nan])).min()),
            "max_tan": fnum(g.get("tangent_ratio", pd.Series([np.nan])).max()),
            "n": int(len(g)),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return
    df["stress"] = df["max_tan"] - 2.10 + (0.285 - df["min_cap"]).clip(lower=0) * 10
    top = df.sort_values(["max_tan", "min_cap"], ascending=[False, True]).head(24).iloc[::-1]
    plt.figure(figsize=(14, max(6, 0.5 * len(top))))
    plt.barh(top["label"], top["max_tan"])
    for i, r in enumerate(top.itertuples()):
        plt.text(float(r.max_tan), i, f" cap={r.min_cap:.3f}", va="center", fontsize=8)
    plt.xlabel("max tangent/radial ratio in selected EC route")
    plt.title("26EC-LITE remaining label stress inside selected route")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def run(args: argparse.Namespace) -> None:
    t0 = time.time()
    ROOT.mkdir(parents=True, exist_ok=True)

    eb = import_by_path(EB_SCRIPT, "phase26eb_lite")
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
    rng = random.Random(args.seed)

    best_name, eb_best_route, eb_summary = load_eb_best_route()
    low_labels = load_low_impact_labels(args, eb_best_route)
    routes = build_compressed_routes(best_name, eb_best_route, low_labels)

    print(f"[{PHASE}] imported EB from: {EB_SCRIPT}")
    print(f"[{PHASE}] EB best route={best_name} explicit_labels={max(0, len(eb_best_route)-1)}")
    print(f"[{PHASE}] low-impact drop candidates={low_labels}")

    base_pool = dz.variant_pool(args.pool_size)
    pool = ea.jitter_variants(dz, base_pool, args.extra_jitters, rng)[: max(4, args.pool_size + args.extra_jitters)]
    seeds = [args.seed + i for i in range(args.verify_seed_count)]
    print(f"[{PHASE}] rebuilt pool variants={len(pool)} seeds={seeds} repeats={args.audit_repeats}")

    cache: Dict[Any, Any] = {}
    pool_seed_cases: Dict[int, Dict[str, pd.DataFrame]] = {}
    all_pool_case_rows: List[pd.DataFrame] = []
    for seed in seeds:
        print(f"[{PHASE}] evaluating verification pool for seed={seed}")
        seed_cases = dz.evaluate_pool_for_seed(ds_lite, drfast, dr, pool, basins, seed, args.audit_repeats, args.sample_full_tail, cache)
        pool_seed_cases[seed] = seed_cases
        for name, df in seed_cases.items():
            if not df.empty:
                all_pool_case_rows.append(df)

    # Add label-local swaps after the pool is measured.
    swap_routes = build_swap_routes(dz, eb_best_route, pool_seed_cases, args)
    routes.extend(swap_routes[: args.max_swap_routes])

    # De-duplicate route maps.
    dedup: List[Tuple[str, Dict[str, str], str]] = []
    seen = set()
    for name, rmap, src in routes:
        key = json.dumps(rmap, sort_keys=True)
        if key not in seen:
            seen.add(key)
            dedup.append((name, rmap, src))
    routes = dedup
    print(f"[{PHASE}] candidate routes after compression/swap/dedup={len(routes)}")

    route_results, route_cases = eb.evaluate_routes(
        dz, routes, pool_seed_cases,
        args.target_capture, args.target_tangent, args.relaxed_tangent, args.target_q20,
        args.print_every,
    )
    route_summary = summarize_ec(route_results, args.target_capture, args.target_tangent, args.relaxed_tangent, args.margin_tangent, args.margin_capture)
    if not route_summary.empty:
        score_lookup = route_summary.set_index("route_name")["ec_margin_score"].to_dict()
        route_results["ec_margin_score"] = route_results["route_name"].map(score_lookup).fillna(route_results.get("dz_route_score_run", 0.0))

    pool_cases = pd.concat(all_pool_case_rows, ignore_index=True) if all_pool_case_rows else pd.DataFrame()

    # Best selected route cases.
    best = route_summary.iloc[0] if len(route_summary) else pd.Series(dtype=object)
    best_route_name = str(best.get("route_name", "")) if len(best) else ""
    best_cases = route_cases[route_cases["route_name"].astype(str) == best_route_name].copy() if best_route_name and not route_cases.empty else pd.DataFrame()
    best_route_map = parse_route_map(best.get("route_map_json", "{}")) if len(best) else {}

    # Output artifacts.
    pool_cases.to_csv(ROOT / "phase26ec_lite_pool_case_results.csv", index=False)
    route_cases.to_csv(ROOT / "phase26ec_lite_route_cases.csv", index=False)
    route_results.to_csv(ROOT / "phase26ec_lite_route_results.csv", index=False)
    route_summary.to_csv(ROOT / "phase26ec_lite_route_summary.csv", index=False)
    if best_route_map:
        (ROOT / "phase26ec_lite_locked_route_map.json").write_text(json.dumps(best_route_map, indent=2, sort_keys=True), encoding="utf-8")

    plot_route_scores(route_summary, ROOT / "phase26ec_lite_route_scores.png")
    plot_pass_rates(route_summary, ROOT / "phase26ec_lite_pass_rates.png")
    plot_capture_tangent(route_results, ROOT / "phase26ec_lite_route_capture_vs_tangent.png", args.target_capture, args.target_tangent, args.relaxed_tangent)
    plot_label_stress(best_cases, ROOT / "phase26ec_lite_best_route_label_stress.png")

    stable_strict = route_summary[route_summary["stable_strict_all_seeds"] == True] if not route_summary.empty else pd.DataFrame()
    margin_strict = route_summary[route_summary["strict_margin_all_seeds"] == True] if not route_summary.empty else pd.DataFrame()

    if len(margin_strict):
        verdict = "compressed_or_locked_route_survives_strict_margin_verification"
    elif len(stable_strict):
        verdict = "locked_route_survives_strict_verification_without_requested_margin"
    elif len(route_summary) and fnum(route_summary.iloc[0].get("relaxed_pass_rate"), 0.0) >= args.min_promote_rate:
        verdict = "route_survives_relaxed_only_or_partial_strict"
    else:
        verdict = "route_fails_ec_verification"

    summary = {
        "phase": PHASE,
        "title": TITLE,
        "torch_info": torch_info,
        "seed_start": int(args.seed),
        "verify_seed_count": int(args.verify_seed_count),
        "audit_repeats": int(args.audit_repeats),
        "sample_full_tail": bool(args.sample_full_tail),
        "pool_size": int(len(pool)),
        "base_pool_size": int(len(base_pool)),
        "extra_jitters": int(max(0, len(pool) - len(base_pool))),
        "targets": {
            "target_capture": float(args.target_capture),
            "target_tangent_strict": float(args.target_tangent),
            "target_tangent_relaxed": float(args.relaxed_tangent),
            "target_q20": float(args.target_q20),
            "requested_capture_margin": float(args.margin_capture),
            "requested_tangent_margin": float(args.margin_tangent),
        },
        "imported_eb_best_route": best_name,
        "imported_eb_best_route_map": eb_best_route,
        "low_impact_drop_labels_from_eb_ablation": list(low_labels),
        "candidate_route_count": int(len(routes)),
        "swap_route_count": int(len(swap_routes)),
        "num_stable_strict_routes": int(len(stable_strict)),
        "num_strict_margin_routes": int(len(margin_strict)),
        "num_cached_eval_entries": int(len(cache)),
        "elapsed_sec": float(time.time() - t0),
        "verdict": verdict,
        "best_route": None if not len(best) else str(best.get("route_name")),
        "best_route_source": None if not len(best) else str(best.get("route_source", "")),
        "best_ec_margin_score": None if not len(best) else fnum(best.get("ec_margin_score")),
        "best_strict_pass_rate": None if not len(best) else fnum(best.get("strict_pass_rate")),
        "best_relaxed_pass_rate": None if not len(best) else fnum(best.get("relaxed_pass_rate")),
        "best_min_worst_capture_rate": None if not len(best) else fnum(best.get("min_worst_capture_rate")),
        "best_max_worst_tangent_ratio": None if not len(best) else fnum(best.get("max_worst_tangent_ratio")),
        "best_explicit_label_count": None if not len(best) else int(best.get("explicit_label_count", 0)),
        "best_route_map": best_route_map,
        "route_top10": route_summary.head(10).to_dict(orient="records") if len(route_summary) else [],
        "outputs": [
            "phase26ec_lite_pool_case_results.csv",
            "phase26ec_lite_route_cases.csv",
            "phase26ec_lite_route_results.csv",
            "phase26ec_lite_route_summary.csv",
            "phase26ec_lite_locked_route_map.json",
            "phase26ec_lite_summary.json",
            "phase26ec_lite_route_scores.png",
            "phase26ec_lite_pass_rates.png",
            "phase26ec_lite_route_capture_vs_tangent.png",
            "phase26ec_lite_best_route_label_stress.png",
        ],
    }
    (ROOT / "phase26ec_lite_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({
        "phase": PHASE,
        "verdict": verdict,
        "best_route": summary["best_route"],
        "best_strict_pass_rate": summary["best_strict_pass_rate"],
        "best_relaxed_pass_rate": summary["best_relaxed_pass_rate"],
        "best_min_worst_capture_rate": summary["best_min_worst_capture_rate"],
        "best_max_worst_tangent_ratio": summary["best_max_worst_tangent_ratio"],
        "best_explicit_label_count": summary["best_explicit_label_count"],
        "candidate_route_count": summary["candidate_route_count"],
        "swap_route_count": summary["swap_route_count"],
        "elapsed_sec": summary["elapsed_sec"],
    }, indent=2))


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=TITLE)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"], help="Torch device for inherited DS-LITE evaluator.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify-seed-count", type=int, default=2)
    ap.add_argument("--audit-repeats", type=int, default=1)
    ap.add_argument("--pool-size", type=int, default=8)
    ap.add_argument("--extra-jitters", type=int, default=4)
    ap.add_argument("--sample-full-tail", action="store_true")
    ap.add_argument("--print-every", type=int, default=8)

    # Same line targets as EB unless changed from CLI.
    ap.add_argument("--target-capture", type=float, default=0.285)
    ap.add_argument("--target-tangent", type=float, default=2.10)
    ap.add_argument("--relaxed-tangent", type=float, default=2.30)
    ap.add_argument("--target-q20", type=float, default=2.25)
    ap.add_argument("--min-promote-rate", type=float, default=0.75)

    # EC-specific compression/margin controls.
    ap.add_argument("--margin-capture", type=float, default=0.010, help="Desired extra capture margin over target.")
    ap.add_argument("--margin-tangent", type=float, default=0.050, help="Desired strict tangent margin under target.")
    ap.add_argument("--drop-delta-tolerance", type=float, default=0.75, help="Labels whose EB ablation score delta is within this loss are tested for compression.")
    ap.add_argument("--min-strict-rate-for-drop", type=float, default=1.0)
    ap.add_argument("--drop-min-capture", type=float, default=0.285)
    ap.add_argument("--drop-max-tangent", type=float, default=2.30)
    ap.add_argument("--max-drop-labels", type=int, default=8)

    # Swap search controls.
    ap.add_argument("--swap-capture-floor", type=float, default=0.335, help="Current-label capture below this is considered weak.")
    ap.add_argument("--swap-tangent-ceiling", type=float, default=2.00, help="Current-label tangent above this is considered weak.")
    ap.add_argument("--swap-min-candidate-capture", type=float, default=0.300)
    ap.add_argument("--swap-max-candidate-tangent", type=float, default=2.50)
    ap.add_argument("--max-swap-labels", type=int, default=4)
    ap.add_argument("--max-swaps-per-label", type=int, default=2)
    ap.add_argument("--max-swap-routes", type=int, default=8)
    return ap


def main() -> None:
    run(build_argparser().parse_args())


if __name__ == "__main__":
    main()
