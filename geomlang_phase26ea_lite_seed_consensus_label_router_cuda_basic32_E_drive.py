#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 26EA-LITE — seed-consensus label router for strict-line rescue (CUDA-capable)

EA takes the lesson from DZ:
    - one routed cloud can partially pass,
    - but the label route is still seed-fragile,
    - and the remaining failures are label-specific rather than global.

Instead of inventing another single global bowl, EA does a cheap second-order pass:
    1. import/reuse 26DZ helpers and its already-known pool machinery,
    2. add small deterministic jitter families around the best anchors,
    3. evaluate the pool across a few seeds,
    4. build a per-label, across-seed score matrix,
    5. beam-search route maps using the measured label matrix only.

This keeps runtime controlled because the expensive envelope/case audit happens once per
variant/seed; thousands of route combinations are then scored by recombining measured
case rows, not by re-running the decoder audits.

Typical:
  python bbit_geomlang/geomlang_phase26ea_lite_seed_consensus_label_router_cuda_basic32_E_drive.py --device cuda

Fast smoke:
  python bbit_geomlang/geomlang_phase26ea_lite_seed_consensus_label_router_cuda_basic32_E_drive.py --device cuda --pool-size 12 --seed-count 2 --audit-repeats 2 --beam-width 24

Stronger:
  python bbit_geomlang/geomlang_phase26ea_lite_seed_consensus_label_router_cuda_basic32_E_drive.py --device cuda --pool-size 28 --seed-count 4 --audit-repeats 4 --beam-width 96 --top-per-label 5
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
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PHASE = "26EA-LITE"
TITLE = "Seed-consensus label router"

ROOT = Path(r"E:\BBIT\outputs_basic32")
SCRIPT_DIR = Path(r"E:\BBIT\bbit_geomlang")
DZ_SCRIPT = SCRIPT_DIR / "geomlang_phase26dz_lite_label_routed_envelope_verifier_cuda_basic32_E_drive.py"

# Fallback for local review in this ChatGPT container; harmless on the user's E: drive.
if not DZ_SCRIPT.exists():
    local = Path(__file__).with_name("geomlang_phase26dz_lite_label_routed_envelope_verifier_cuda_basic32_E_drive.py")
    if local.exists():
        DZ_SCRIPT = local


def import_by_path(path: Path, module_name: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing dependency: {path}")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
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


def load_dz() -> Any:
    return import_by_path(DZ_SCRIPT, "phase26dz_lite_helpers")


def row_key(ov: Dict[str, float], params: List[str], ndigits: int = 6) -> Tuple[float, ...]:
    return tuple(round(float(ov[k]), ndigits) for k in params)


def add_variant(out: List[Tuple[str, Dict[str, float], str]], seen: set, name: str, ov: Dict[str, Any], src: str, params: List[str]) -> None:
    if not isinstance(ov, dict) or any(k not in ov for k in params):
        return
    cov = {k: float(ov[k]) for k in params}
    key = row_key(cov, params)
    if key not in seen:
        out.append((name[:118], cov, src))
        seen.add(key)


def jitter_variants(dz: Any, base_pool: List[Tuple[str, Dict[str, float], str]], max_extra: int, rng: random.Random) -> List[Tuple[str, Dict[str, float], str]]:
    """Add tiny local probes around the best historical anchors.  Not a wide search."""
    params = list(dz.PARAMS)
    out: List[Tuple[str, Dict[str, float], str]] = []
    seen: set = set()
    for n, ov, src in base_pool:
        add_variant(out, seen, n, ov, src, params)

    anchors: List[Tuple[str, Dict[str, float]]] = []
    for name, ov_name in [
        ("ea_anchor_dx", "FALLBACK_DX_BEST"),
        ("ea_anchor_dw_balanced", "FALLBACK_DW_BALANCED"),
        ("ea_anchor_low_capture", "FALLBACK_LOW_CAPTURE_CLEAN"),
        ("ea_anchor_dv_best", "FALLBACK_DV_BEST"),
    ]:
        if hasattr(dz, ov_name):
            anchors.append((name, getattr(dz, ov_name)))

    # Deterministic, narrow repairs aimed at remaining DZ/DY/DX culprits:
    # - slightly higher cap for capture-starved labels,
    # - slightly higher tangent kill for stress_cap_tk / seat_down,
    # - lower shell gain variants to see whether the tail tangent is being injected by the shell.
    cap_mults = [0.985, 1.000, 1.015, 1.030]
    tk_mults = [0.970, 1.000, 1.030, 1.060]
    shell_mults = [0.880, 0.940, 1.000, 1.060]
    seat_mults = [0.940, 1.000, 1.060]

    for aname, base in anchors:
        for i, (cm, tm, sm, sem) in enumerate([(cm, tm, sm, sem) for cm in cap_mults for tm in tk_mults for sm in shell_mults for sem in seat_mults]):
            if len(out) >= len(base_pool) + max_extra:
                return out
            ov = dict(base)
            ov["BOWL_NORM_CAP_MULT"] = float(np.clip(ov["BOWL_NORM_CAP_MULT"] * cm, 0.92, 1.24))
            ov["BOWL_TANGENT_KILL"] = float(np.clip(ov["BOWL_TANGENT_KILL"] * tm, 0.86, 1.12))
            ov["BOWL_SHELL_RADIAL_GAIN"] = float(np.clip(ov["BOWL_SHELL_RADIAL_GAIN"] * sm, 0.08, 0.23))
            ov["BOWL_SEAT_AXIS_GAIN"] = float(np.clip(ov["BOWL_SEAT_AXIS_GAIN"] * sem, 0.32, 0.52))
            add_variant(out, seen, f"ea_jitter_{aname}_{i:03d}", ov, "ea_local_jitter", params)

    # If room remains, add a few random micro-jitters around the first anchors.
    while len(out) < len(base_pool) + max_extra and anchors:
        aname, base = rng.choice(anchors)
        ov = dict(base)
        ov["BOWL_RADIUS_FRAC"] = float(np.clip(ov["BOWL_RADIUS_FRAC"] + rng.uniform(-0.018, 0.018), 0.66, 0.78))
        ov["BOWL_SEAT_AXIS_GAIN"] = float(np.clip(ov["BOWL_SEAT_AXIS_GAIN"] + rng.uniform(-0.035, 0.035), 0.32, 0.52))
        ov["BOWL_DIRECTIONAL_BLEND"] = float(np.clip(ov["BOWL_DIRECTIONAL_BLEND"] + rng.uniform(-0.055, 0.055), 0.90, 1.22))
        ov["BOWL_NORM_CAP_MULT"] = float(np.clip(ov["BOWL_NORM_CAP_MULT"] + rng.uniform(-0.055, 0.075), 0.92, 1.24))
        ov["BOWL_SHELL_RADIAL_GAIN"] = float(np.clip(ov["BOWL_SHELL_RADIAL_GAIN"] + rng.uniform(-0.035, 0.035), 0.08, 0.23))
        ov["BOWL_TANGENT_KILL"] = float(np.clip(ov["BOWL_TANGENT_KILL"] + rng.uniform(-0.055, 0.065), 0.86, 1.12))
        add_variant(out, seen, f"ea_random_micro_{aname}_{len(out):03d}", ov, "ea_random_micro", params)

    return out


def label_matrix_across_seeds(dz: Any, pool_seed_cases: Dict[int, Dict[str, pd.DataFrame]], target_cap: float, target_tan: float, relaxed_tan: float, target_q20: float) -> pd.DataFrame:
    rows = []
    for seed, by_variant in pool_seed_cases.items():
        for variant, df in by_variant.items():
            if df.empty or "envelope_label" not in df:
                continue
            for label, g in df.groupby("envelope_label", sort=False):
                m, _ = dz.score_cases(g, target_cap, target_tan, relaxed_tan, target_q20)
                rows.append({
                    "dz_seed": seed,
                    "source_variant": variant,
                    "envelope_label": str(label),
                    "label_score": fnum(m.get("dz_route_score_run")),
                    "capture": fnum(m.get("worst_capture_rate")),
                    "tangent": fnum(m.get("worst_tangent_ratio"), 999.0),
                    "q20_dp": fnum(m.get("q20_dp_score"), -999.0),
                    "pass_strict": bool(m.get("pass_strict_line", False)),
                    "pass_relaxed": bool(m.get("pass_relaxed_line", False)),
                })
    if not rows:
        return pd.DataFrame()
    raw = pd.DataFrame(rows)
    agg_rows = []
    for (variant, label), g in raw.groupby(["source_variant", "envelope_label"], sort=False):
        pass_rel = float(g["pass_relaxed"].mean())
        pass_str = float(g["pass_strict"].mean())
        min_cap = float(g["capture"].min())
        max_tan = float(g["tangent"].max())
        mean_score = float(g["label_score"].mean())
        std_score = float(g["label_score"].std(ddof=0)) if len(g) > 1 else 0.0
        consensus_score = (
            mean_score
            + 11.0 * pass_str
            + 8.0 * pass_rel
            + 160.0 * (min_cap - target_cap)
            - 4.0 * max(0.0, max_tan - relaxed_tan)
            - 1.2 * std_score
        )
        agg_rows.append({
            "source_variant": variant,
            "envelope_label": label,
            "seed_n": int(len(g)),
            "label_consensus_score": float(consensus_score),
            "mean_label_score": mean_score,
            "std_label_score": std_score,
            "strict_rate": pass_str,
            "relaxed_rate": pass_rel,
            "min_capture": min_cap,
            "max_tangent": max_tan,
            "min_q20_dp": float(g["q20_dp"].min()),
        })
    return pd.DataFrame(agg_rows).sort_values(["envelope_label", "label_consensus_score"], ascending=[True, False])


def route_score_quick(dz: Any, route: Dict[str, str], pool_seed_cases: Dict[int, Dict[str, pd.DataFrame]], target_cap: float, target_tan: float, relaxed_tan: float, target_q20: float) -> float:
    rows = []
    for seed in sorted(pool_seed_cases):
        m, _ = dz.compose_route("ea_beam_tmp", route, pool_seed_cases[seed], target_cap, target_tan, relaxed_tan, target_q20)
        rows.append(m)
    if not rows:
        return -1e9
    df = pd.DataFrame(rows)
    strict_rate = float(df["pass_strict_line"].mean())
    relaxed_rate = float(df["pass_relaxed_line"].mean())
    min_cap = float(df["worst_capture_rate"].min())
    max_tan = float(df["worst_tangent_ratio"].max())
    mean = float(df["dz_route_score_run"].mean())
    cap_std = float(df["worst_capture_rate"].std(ddof=0)) if len(df) > 1 else 0.0
    tan_std = float(df["worst_tangent_ratio"].std(ddof=0)) if len(df) > 1 else 0.0
    return (
        22.0 * strict_rate
        + 16.0 * relaxed_rate
        + 0.70 * mean
        + 180.0 * (min_cap - target_cap)
        - 5.0 * max(0.0, max_tan - relaxed_tan)
        - 2.0 * max(0.0, max_tan - target_tan)
        - 22.0 * cap_std
        - 1.1 * tan_std
    )


def beam_consensus_routes(dz: Any, label_matrix: pd.DataFrame, pool_seed_cases: Dict[int, Dict[str, pd.DataFrame]], target_cap: float, target_tan: float, relaxed_tan: float, target_q20: float, beam_width: int, top_per_label: int, route_count: int) -> List[Tuple[str, Dict[str, str], str]]:
    if label_matrix.empty or not pool_seed_cases:
        return []
    first_seed_cases = pool_seed_cases[sorted(pool_seed_cases)[0]]
    names = list(first_seed_cases.keys())
    labels = [x for x in getattr(dz, "CRITICAL_LABELS", []) if x in set(label_matrix["envelope_label"].astype(str))]
    if not labels:
        labels = sorted(label_matrix["envelope_label"].astype(str).unique().tolist())

    # Defaults: choose variants with good whole-envelope behavior.
    default_scores = []
    for n in names:
        r = {"*": n}
        default_scores.append((route_score_quick(dz, r, pool_seed_cases, target_cap, target_tan, relaxed_tan, target_q20), n))
    defaults = [n for _, n in sorted(default_scores, reverse=True)[: min(6, len(default_scores))]]

    choices: Dict[str, List[str]] = {}
    for lab in labels:
        sub = label_matrix[label_matrix["envelope_label"].astype(str) == lab].sort_values("label_consensus_score", ascending=False)
        top = sub["source_variant"].astype(str).head(max(1, top_per_label)).tolist()
        # Always include the two best capture and two best tangent-clean candidates.
        cap = sub.sort_values(["min_capture", "max_tangent"], ascending=[False, True])["source_variant"].astype(str).head(2).tolist()
        clean = sub.sort_values(["max_tangent", "min_capture"], ascending=[True, False])["source_variant"].astype(str).head(2).tolist()
        merged = []
        for x in top + cap + clean:
            if x in names and x not in merged:
                merged.append(x)
        choices[lab] = merged[: max(top_per_label, 3)]

    beam: List[Tuple[float, Dict[str, str]]] = []
    for d in defaults:
        r = {"*": d}
        beam.append((route_score_quick(dz, r, pool_seed_cases, target_cap, target_tan, relaxed_tan, target_q20), r))
    beam = sorted(beam, key=lambda x: x[0], reverse=True)[:beam_width]

    for lab in labels:
        nxt: List[Tuple[float, Dict[str, str]]] = []
        for _, route in beam:
            # option A: leave on default; option B: route label explicitly.
            candidates = [route.get("*")] + choices.get(lab, [])
            seen_c = set()
            for v in candidates:
                if not v or v in seen_c:
                    continue
                seen_c.add(v)
                nr = dict(route)
                if v == nr.get("*"):
                    nr.pop(lab, None)
                else:
                    nr[lab] = v
                sc = route_score_quick(dz, nr, pool_seed_cases, target_cap, target_tan, relaxed_tan, target_q20)
                nxt.append((sc, nr))
        # dedupe
        tmp: Dict[Tuple[Tuple[str, str], ...], Tuple[float, Dict[str, str]]] = {}
        for sc, r in nxt:
            key = tuple(sorted(r.items()))
            if key not in tmp or sc > tmp[key][0]:
                tmp[key] = (sc, r)
        beam = sorted(tmp.values(), key=lambda x: x[0], reverse=True)[:beam_width]
        print(f"[{PHASE}] beam after label={lab:18s}: best={beam[0][0]:.3f} width={len(beam)}")

    routes: List[Tuple[str, Dict[str, str], str]] = []
    for i, (sc, r) in enumerate(beam[:route_count]):
        routes.append((f"ea_beam_consensus_{i:03d}", r, f"beam_consensus_score={sc:.6f}"))
    return routes


def summarize_routes_ea(dz: Any, results_df: pd.DataFrame, target_cap: float, target_tan: float, relaxed_tan: float, target_q20: float) -> pd.DataFrame:
    out = dz.summarize_routes(results_df, target_cap, target_tan, relaxed_tan, target_q20)
    if out.empty:
        return out
    out = out.rename(columns={"dz_route_family_score": "ea_route_family_score"})
    return out.sort_values(["stable_strict_all_seeds", "stable_relaxed_all_seeds", "relaxed_pass_rate", "ea_route_family_score"], ascending=[False, False, False, False])


def plot_route_scores(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    top = df.sort_values("ea_route_family_score", ascending=False).head(25).iloc[::-1]
    plt.figure(figsize=(14, max(6, 0.48 * len(top))))
    plt.barh(top["route_name"], top["ea_route_family_score"])
    plt.xlabel("EA seed-consensus family score")
    plt.title("26EA-LITE seed-consensus routed-envelope family scores")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_pass_rates(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    top = df.sort_values(["relaxed_pass_rate", "strict_pass_rate", "ea_route_family_score"], ascending=[False, False, False]).head(25).iloc[::-1]
    y = np.arange(len(top))
    plt.figure(figsize=(14, max(6, 0.48 * len(top))))
    plt.barh(y - 0.13, top["relaxed_pass_rate"], height=0.25, label="relaxed")
    plt.barh(y + 0.13, top["strict_pass_rate"], height=0.25, label="strict")
    plt.yticks(y, top["route_name"])
    plt.xlabel("pass rate across seeds")
    plt.title("26EA-LITE routed-envelope pass rates")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_capture_tangent(df: pd.DataFrame, out_path: Path, target_cap: float, target_tan: float, relaxed_tan: float) -> None:
    if df.empty:
        return
    c = df.get("dz_route_score_run", pd.Series(np.zeros(len(df))))
    plt.figure(figsize=(10, 8))
    plt.scatter(df["worst_capture_rate"], df["worst_tangent_ratio"], c=c, s=80, alpha=0.85)
    plt.axvline(target_cap, linewidth=0.8, label="capture target")
    plt.axhline(target_tan, linewidth=0.8, label="strict tangent")
    plt.axhline(relaxed_tan, linewidth=0.8, linestyle="--", label="relaxed tangent")
    plt.xlabel("routed-envelope worst capture_rate")
    plt.ylabel("routed-envelope worst tangent/radial ratio")
    plt.title("26EA-LITE seed-consensus routes: capture vs tangent")
    plt.colorbar(label="EA run score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_label_matrix(label_df: pd.DataFrame, out_path: Path) -> None:
    if label_df.empty:
        return
    top = label_df.sort_values("label_consensus_score", ascending=False).groupby("envelope_label", as_index=False).head(1)
    top = top.sort_values("label_consensus_score").tail(20)
    plt.figure(figsize=(14, max(6, 0.55 * len(top))))
    labels = [f"{r.envelope_label} ← {str(r.source_variant)[:48]}" for r in top.itertuples()]
    plt.barh(labels, top["label_consensus_score"])
    for i, r in enumerate(top.itertuples()):
        plt.text(float(r.label_consensus_score), i, f" cap={r.min_capture:.3f} tan={r.max_tangent:.3f}", va="center", fontsize=8)
    plt.xlabel("best per-label consensus score")
    plt.title("26EA-LITE best consensus label assignments")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def run(args: argparse.Namespace) -> None:
    t0 = time.time()
    ROOT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    dz = load_dz()

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

    base_pool = dz.variant_pool(args.pool_size)
    pool = jitter_variants(dz, base_pool, args.extra_jitters, rng)[: max(4, args.pool_size + args.extra_jitters)]
    print(f"[{PHASE}] imported DZ from: {DZ_SCRIPT}")
    print(f"[{PHASE}] pool variants={len(pool)} base={len(base_pool)} extra={max(0, len(pool)-len(base_pool))}")

    approx_micro_audits = len(pool) * int(args.seed_count) * int(args.audit_repeats)
    print(
        f"[{PHASE}] seeds={args.seed_count} audit_repeats={args.audit_repeats} "
        f"approx_micro_audits={approx_micro_audits}"
    )

    seeds = [args.seed + i for i in range(args.seed_count)]
    cache: Dict[Any, Any] = {}
    pool_seed_cases: Dict[int, Dict[str, pd.DataFrame]] = {}
    all_pool_case_rows: List[pd.DataFrame] = []

    for seed in seeds:
        print(f"[{PHASE}] evaluating pool for seed={seed}")
        seed_cases = dz.evaluate_pool_for_seed(ds_lite, drfast, dr, pool, basins, seed, args.audit_repeats, args.sample_full_tail, cache)
        pool_seed_cases[seed] = seed_cases
        for df in seed_cases.values():
            if not df.empty:
                all_pool_case_rows.append(df)

    label_df = label_matrix_across_seeds(dz, pool_seed_cases, target_cap, target_tan, relaxed_tan, target_q20)
    routes = beam_consensus_routes(dz, label_df, pool_seed_cases, target_cap, target_tan, relaxed_tan, target_q20, args.beam_width, args.top_per_label, args.route_count)

    # Preserve a few DZ-style controls for comparison.
    hist = dz.load_historical_label_scores(pool, target_cap, target_tan, relaxed_tan, target_q20)
    controls = dz.build_routes(pool, hist, min(args.control_routes, 8), rng)
    routes = controls + routes
    print(f"[{PHASE}] total routes including controls={len(routes)}")

    route_rows: List[Dict[str, Any]] = []
    routed_case_rows: List[pd.DataFrame] = []
    for route_i, (route_name, route_map, route_src) in enumerate(routes, 1):
        for si, seed in enumerate(seeds, 1):
            metrics, cases = dz.compose_route(route_name, route_map, pool_seed_cases[seed], target_cap, target_tan, relaxed_tan, target_q20)
            metrics["route_source"] = route_src
            metrics["seed_index"] = si
            metrics["dz_seed"] = seed
            route_rows.append(metrics)
            if not cases.empty:
                cases["route_source"] = route_src
                cases["seed_index"] = si
                routed_case_rows.append(cases)
            if route_i <= 5 or route_i % args.print_every == 0 or si == args.seed_count:
                print(
                    f"[{PHASE}] route {route_i:03d}/{len(routes):03d} {route_name[:58]:58s} s{si:02d} "
                    f"RELAX={int(metrics.get('pass_relaxed_line', False))} STRICT={int(metrics.get('pass_strict_line', False))} "
                    f"cap={fnum(metrics.get('worst_capture_rate')):.3f} tan={fnum(metrics.get('worst_tangent_ratio')):.3f} "
                    f"q20={fnum(metrics.get('q20_dp_score')):.3f} score={fnum(metrics.get('dz_route_score_run')):.3f}"
                )

    pool_cases_df = pd.concat(all_pool_case_rows, ignore_index=True) if all_pool_case_rows else pd.DataFrame()
    routed_cases_df = pd.concat(routed_case_rows, ignore_index=True) if routed_case_rows else pd.DataFrame()
    route_results_df = pd.DataFrame(route_rows)
    route_summary_df = summarize_routes_ea(dz, route_results_df, target_cap, target_tan, relaxed_tan, target_q20)

    pool_cases_df.to_csv(ROOT / "phase26ea_lite_pool_case_results.csv", index=False)
    routed_cases_df.to_csv(ROOT / "phase26ea_lite_route_cases.csv", index=False)
    route_results_df.to_csv(ROOT / "phase26ea_lite_route_results.csv", index=False)
    route_summary_df.to_csv(ROOT / "phase26ea_lite_route_summary.csv", index=False)
    label_df.to_csv(ROOT / "phase26ea_lite_label_consensus_matrix.csv", index=False)

    plot_route_scores(route_summary_df, ROOT / "phase26ea_lite_route_scores.png")
    plot_pass_rates(route_summary_df, ROOT / "phase26ea_lite_pass_rates.png")
    plot_capture_tangent(route_results_df, ROOT / "phase26ea_lite_route_capture_vs_tangent.png", target_cap, target_tan, relaxed_tan)
    plot_label_matrix(label_df, ROOT / "phase26ea_lite_best_label_consensus.png")

    best = route_summary_df.iloc[0] if len(route_summary_df) else pd.Series(dtype=object)
    best_route_map: Dict[str, str] = {}
    if len(best):
        try:
            best_route_map = json.loads(str(best.get("route_map_json", "{}")))
        except Exception:
            best_route_map = {}

    stable_strict = route_summary_df[route_summary_df["stable_strict_all_seeds"] == True] if not route_summary_df.empty else pd.DataFrame()
    stable_relaxed = route_summary_df[route_summary_df["stable_relaxed_all_seeds"] == True] if not route_summary_df.empty else pd.DataFrame()

    if len(stable_strict):
        verdict = "strict_seed_consensus_route_found"
    elif len(stable_relaxed):
        verdict = "relaxed_seed_consensus_route_found"
    elif len(best) and fnum(best.get("relaxed_pass_rate"), 0.0) >= args.min_promote_rate:
        verdict = "partial_seed_consensus_advantage_found"
    else:
        verdict = "no_seed_consensus_pass_found"

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
        "pool_size": int(len(pool)),
        "base_pool_size": int(len(base_pool)),
        "extra_jitters": int(max(0, len(pool) - len(base_pool))),
        "seed_count": int(args.seed_count),
        "audit_repeats": int(args.audit_repeats),
        "sample_full_tail": bool(args.sample_full_tail),
        "beam_width": int(args.beam_width),
        "top_per_label": int(args.top_per_label),
        "route_count": int(len(routes)),
        "num_cached_eval_entries": int(len(cache)),
        "num_stable_strict_routes": int(len(stable_strict)),
        "num_stable_relaxed_routes": int(len(stable_relaxed)),
        "elapsed_sec": float(time.time() - t0),
        "verdict": verdict,
        "best_route": None if not len(best) else str(best.get("route_name")),
        "best_route_source": None if not len(best) else str(best.get("route_source", "")),
        "best_route_family_score": None if not len(best) else fnum(best.get("ea_route_family_score")),
        "best_strict_pass_rate": None if not len(best) else fnum(best.get("strict_pass_rate")),
        "best_relaxed_pass_rate": None if not len(best) else fnum(best.get("relaxed_pass_rate")),
        "best_min_worst_capture_rate": None if not len(best) else fnum(best.get("min_worst_capture_rate")),
        "best_max_worst_tangent_ratio": None if not len(best) else fnum(best.get("max_worst_tangent_ratio")),
        "best_min_q20_dp_score": None if not len(best) else fnum(best.get("min_q20_dp_score")),
        "best_route_map": best_route_map,
        "route_top10": route_summary_df.head(10).to_dict(orient="records") if len(route_summary_df) else [],
        "outputs": [
            "phase26ea_lite_pool_case_results.csv",
            "phase26ea_lite_route_cases.csv",
            "phase26ea_lite_route_results.csv",
            "phase26ea_lite_route_summary.csv",
            "phase26ea_lite_label_consensus_matrix.csv",
            "phase26ea_lite_summary.json",
            "phase26ea_lite_route_scores.png",
            "phase26ea_lite_pass_rates.png",
            "phase26ea_lite_route_capture_vs_tangent.png",
            "phase26ea_lite_best_label_consensus.png",
        ],
    }
    (ROOT / "phase26ea_lite_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[{PHASE}] verdict={verdict}")
    print(f"[{PHASE}] best_route={summary['best_route']} strict={summary['best_strict_pass_rate']} relaxed={summary['best_relaxed_pass_rate']} cap={summary['best_min_worst_capture_rate']} tan={summary['best_max_worst_tangent_ratio']}")
    print(f"[{PHASE}] wrote outputs to {ROOT}")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=f"{PHASE}: {TITLE}")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"], help="Torch device for imported CUDA-capable phases")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-count", type=int, default=2)
    ap.add_argument("--pool-size", type=int, default=8, help="Base DZ pool size before EA jitters")
    ap.add_argument("--extra-jitters", type=int, default=4, help="Additional narrow EA local probes")
    ap.add_argument("--audit-repeats", type=int, default=1)
    ap.add_argument("--sample-full-tail", action="store_true")
    ap.add_argument("--beam-width", type=int, default=24)
    ap.add_argument("--top-per-label", type=int, default=3)
    ap.add_argument("--route-count", type=int, default=12, help="Number of EA beam routes to verify")
    ap.add_argument("--control-routes", type=int, default=4, help="DZ-style global/hand controls to include")
    ap.add_argument("--print-every", type=int, default=5)
    ap.add_argument("--target-capture", type=float, default=0.285)
    ap.add_argument("--target-tangent", type=float, default=2.10)
    ap.add_argument("--relaxed-tangent", type=float, default=2.30)
    ap.add_argument("--target-q20", type=float, default=2.25)
    ap.add_argument("--min-promote-rate", type=float, default=0.67)
    return ap


if __name__ == "__main__":
    run(build_argparser().parse_args())
