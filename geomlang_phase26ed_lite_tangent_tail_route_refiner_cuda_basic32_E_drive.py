#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Phase 26ED-LITE — tangent-tail route refiner / strict-margin verifier.

What ED does after EC:
  - EC proved the locked EA/EB route can pass strict verification, but the
    selected route was still riding a tangent tail near ~2.02.
  - ED keeps the successful locked-route idea, then does a bounded per-label
    tangent-tail search: it swaps only the labels that still create the route's
    worst tangent/radial spikes.
  - It explicitly rewards tangent margin below a stricter target while refusing
    to trade away the capture floor.

Run from E:\BBIT, for example:

  py -3.12 bbit_geomlang/geomlang_phase26ed_lite_tangent_tail_route_refiner_cuda_basic32_E_drive.py --device cuda

Optional faster / deeper examples:

  py -3.12 bbit_geomlang/geomlang_phase26ed_lite_tangent_tail_route_refiner_cuda_basic32_E_drive.py --device cuda --max-candidates 80 --max-tail-labels 5 --max-swaps-per-label 5

  py -3.12 bbit_geomlang/geomlang_phase26ed_lite_tangent_tail_route_refiner_cuda_basic32_E_drive.py --device cuda --strict-tangent-target 2.00 --relaxed-tangent 2.10 --seeds 12,13,14,15,16,17
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(r"E:\BBIT\outputs_basic32") if Path(r"E:\BBIT").exists() else Path("/mnt/data")
SCRIPT_DIR = Path(r"E:\BBIT\bbit_geomlang") if Path(r"E:\BBIT\bbit_geomlang").exists() else Path("/mnt/data")
EC_SCRIPT = SCRIPT_DIR / "geomlang_phase26ec_lite_locked_route_margin_compressor_cuda_basic32_E_drive.py"
EB_SCRIPT = SCRIPT_DIR / "geomlang_phase26eb_lite_locked_route_stress_verifier_cuda_basic32_E_drive.py"
EA_SCRIPT = SCRIPT_DIR / "geomlang_phase26ea_lite_seed_consensus_label_router_cuda_basic32_E_drive.py"

PHASE = "26ED-LITE"
TITLE = "tangent-tail route refiner / strict-margin verifier"


def import_by_path(path: Path, module_name: str) -> Any:
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
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


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


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


def load_ec_route() -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    summary = read_json(ROOT / "phase26ec_lite_summary.json")
    name = str(summary.get("best_route") or "ec_locked_full_eb_route")
    route = parse_route_map(summary.get("best_route_map"))
    if not route:
        rs = read_csv(ROOT / "phase26ec_lite_route_summary.csv")
        if not rs.empty:
            row = rs.iloc[0]
            name = str(row.get("route_name", name))
            route = parse_route_map(row.get("route_map_json", "{}"))
    if not route:
        # Fall back to EB if EC summary is unavailable.
        eb = read_json(ROOT / "phase26eb_lite_summary.json")
        name = str(eb.get("best_route") or "eb_locked_route")
        route = parse_route_map(eb.get("best_route_map"))
    if not route:
        raise RuntimeError("Could not load EC/EB locked route map. Run 26EC first and keep its summary in outputs_basic32.")
    return name, route, summary


def build_candidate_pool(
    dz: Any,
    ea: Any,
    max_candidates: int,
    required_names: Sequence[str],
    rng: random.Random,
) -> List[Tuple[str, Dict[str, float], str]]:
    """Rebuild the current DZ/EA variant pool and keep locked-route members."""
    required = [str(n).strip() for n in required_names if str(n).strip()]
    required_set = set(required)

    base_size = max(4, min(8, max_candidates))
    base_pool = dz.variant_pool(base_size)
    extra = max(max_candidates, len(required) + 4)
    if hasattr(ea, "jitter_variants"):
        pool = ea.jitter_variants(dz, base_pool, extra, rng)
    else:
        pool = list(base_pool)

    by_name: Dict[str, Tuple[str, Dict[str, float], str]] = {}
    for name, ov, src in pool:
        sname = str(name).strip()
        if sname and sname not in by_name:
            by_name[sname] = (sname, ov, src)

    selected: List[Tuple[str, Dict[str, float], str]] = []
    for name, item in by_name.items():
        if name not in required_set and len(selected) < max_candidates:
            selected.append(item)
    for name in required:
        item = by_name.get(name)
        if item is not None and all(existing[0] != name for existing in selected):
            selected.append(item)

    missing = [name for name in required if name not in by_name]
    if missing:
        print(f"[26ED] WARN locked route references variants not in rebuilt pool: {missing[:8]}")
    return selected


def route_explicit_count(route: Dict[str, str]) -> int:
    return int(len([k for k in route if k != "*"]))


def safe_name(s: str, n: int = 80) -> str:
    s = str(s).replace(" ", "_").replace("/", "_").replace("\\", "_")
    return s[:n]


def summarize_routes(route_results: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
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
        explicit = route_explicit_count(route)
        cap_margin = min_cap - args.target_capture
        tan_margin = args.strict_tangent_target - max_tan
        relaxed_margin = args.relaxed_tangent - max_tan
        # ED score is intentionally tangent-biased.  By this point we already
        # know strict pass is possible; now we want more air under the tail.
        tail_penalty = max(0.0, max_tan - args.strict_tangent_target)
        cap_penalty = max(0.0, args.target_capture - min_cap)
        ed_score = (
            120.0 * strict_rate
            + 25.0 * relaxed_rate
            + 60.0 * min(0.08, max(-0.08, cap_margin))
            + 45.0 * min(0.50, max(-0.50, tan_margin))
            + 4.0 * min_q20
            - 80.0 * tail_penalty
            - 140.0 * cap_penalty
            - 0.10 * explicit
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
            "min_worst_capture_rate": min_cap,
            "max_worst_tangent_ratio": max_tan,
            "min_q20_dp_score": min_q20,
            "capture_margin_vs_target": cap_margin,
            "tangent_margin_vs_strict_target": tan_margin,
            "tangent_margin_vs_relaxed": relaxed_margin,
            "ed_tail_score": ed_score,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["stable_strict_all_seeds", "ed_tail_score", "max_worst_tangent_ratio", "min_worst_capture_rate"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)


def label_stress_from_cases(cases: pd.DataFrame) -> pd.DataFrame:
    if cases.empty or "envelope_label" not in cases:
        return pd.DataFrame()
    rows = []
    for lab, g in cases.groupby(cases["envelope_label"].astype(str)):
        rows.append({
            "label": lab,
            "min_capture_rate": fnum(g.get("capture_rate", pd.Series([np.nan])).min()),
            "max_tangent_ratio": fnum(g.get("tangent_ratio", pd.Series([np.nan])).max()),
            "mean_tangent_ratio": fnum(g.get("tangent_ratio", pd.Series([np.nan])).mean()),
            "num_cases": int(len(g)),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["tail_excess_vs_2p0"] = (out["max_tangent_ratio"] - 2.0).clip(lower=0)
    out["capture_gap_vs_0p285"] = (0.285 - out["min_capture_rate"]).clip(lower=0)
    out["label_tail_stress"] = out["tail_excess_vs_2p0"] * 10.0 + out["capture_gap_vs_0p285"] * 6.0
    return out.sort_values(["max_tangent_ratio", "min_capture_rate"], ascending=[False, True]).reset_index(drop=True)


def score_variant_label(dz: Any, df: pd.DataFrame, lab: str, args: argparse.Namespace) -> Dict[str, float]:
    sub = df[df["envelope_label"].astype(str) == lab] if "envelope_label" in df else pd.DataFrame()
    if sub.empty:
        return {"cap": -999.0, "tan": 999.0, "q20": -999.0, "score": -9999.0}
    metrics, _ = dz.score_cases(sub, args.target_capture, args.strict_tangent_target, args.relaxed_tangent, args.target_q20)
    cap = fnum(metrics.get("worst_capture_rate"), -999.0)
    tan = fnum(metrics.get("worst_tangent_ratio"), 999.0)
    q20 = fnum(metrics.get("q20_dp_score"), -999.0)
    # Local score: first suppress tangent tail, then preserve capture.
    score = 80.0 * min(0.08, cap - args.target_capture) + 45.0 * min(0.60, args.strict_tangent_target - tan) + 2.0 * q20
    if cap < args.label_min_capture:
        score -= 200.0 * (args.label_min_capture - cap)
    if tan > args.label_max_tangent:
        score -= 90.0 * (tan - args.label_max_tangent)
    return {"cap": cap, "tan": tan, "q20": q20, "score": score}


def build_tail_routes(dz: Any, base_route: Dict[str, str], pool_seed_cases: Dict[int, Dict[str, pd.DataFrame]], args: argparse.Namespace) -> Tuple[List[Tuple[str, Dict[str, str], str]], pd.DataFrame]:
    if not pool_seed_cases:
        return [], pd.DataFrame()
    first_seed = sorted(pool_seed_cases)[0]
    cases_by_variant = pool_seed_cases[first_seed]
    _, base_cases = dz.compose_route("ed_tail_probe", base_route, cases_by_variant, args.target_capture, args.strict_tangent_target, args.relaxed_tangent, args.target_q20)
    stress = label_stress_from_cases(base_cases)
    if stress.empty:
        return [], stress
    weak_labels = stress[(stress["max_tangent_ratio"] >= args.tail_label_tangent_floor) | (stress["min_capture_rate"] < args.tail_label_capture_floor)]["label"].astype(str).tolist()
    weak_labels = weak_labels[: args.max_tail_labels]

    routes: List[Tuple[str, Dict[str, str], str]] = []
    routes.append(("ed_locked_ec_baseline", dict(base_route), "baseline=ec_best_route"))

    # Individual tail swaps.
    best_per_label: Dict[str, List[Tuple[str, float, float, float, float]]] = {}
    for lab in weak_labels:
        rows = []
        for vname, df in cases_by_variant.items():
            m = score_variant_label(dz, df, lab, args)
            cap, tan, q20, score = m["cap"], m["tan"], m["q20"], m["score"]
            if cap >= args.candidate_min_capture and tan <= args.candidate_max_tangent:
                rows.append((vname, cap, tan, q20, score))
        rows.sort(key=lambda x: (x[4], x[1], -x[2], x[3]), reverse=True)
        best_per_label[lab] = rows[: args.max_swaps_per_label]
        for i, (vname, cap, tan, q20, score) in enumerate(best_per_label[lab], 1):
            old = base_route.get(lab, base_route.get("*", ""))
            if vname == old:
                continue
            r = dict(base_route)
            r[lab] = vname
            routes.append((
                f"ed_tail_swap_{safe_name(lab, 36)}_{i:02d}",
                r,
                f"single_tail_swap label={lab}; old={old}; new={vname}; cap={cap:.3f}; tan={tan:.3f}; local_score={score:.3f}",
            ))

    # Greedy cumulative route: add the best improving swap for the worst labels.
    greedy = dict(base_route)
    edits = []
    for lab in weak_labels:
        choices = best_per_label.get(lab) or []
        if not choices:
            continue
        vname, cap, tan, q20, score = choices[0]
        old = greedy.get(lab, greedy.get("*", ""))
        if vname != old:
            greedy[lab] = vname
            edits.append(f"{lab}:{old}->{vname}(cap={cap:.3f},tan={tan:.3f})")
            routes.append((f"ed_greedy_tail_{len(edits):02d}", dict(greedy), "greedy_edits=" + ";".join(edits)))

    # Pairwise swaps for the top few labels: catches cases where screen labels
    # and stress labels only improve when moved together.
    pair_labels = weak_labels[: args.max_pair_labels]
    for i, a in enumerate(pair_labels):
        for b in pair_labels[i + 1:]:
            for ai, ca in enumerate(best_per_label.get(a, [])[: args.max_pair_choices], 1):
                for bi, cb in enumerate(best_per_label.get(b, [])[: args.max_pair_choices], 1):
                    va = ca[0]; vb = cb[0]
                    r = dict(base_route)
                    r[a] = va
                    r[b] = vb
                    routes.append((f"ed_pair_{safe_name(a,16)}_{ai:02d}_{safe_name(b,16)}_{bi:02d}", r, f"pair_tail_swap {a}->{va}; {b}->{vb}"))

    # Deduplicate identical route maps.
    seen = set()
    dedup = []
    for name, r, src in routes:
        key = json.dumps(r, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        dedup.append((name, r, src))
    return dedup[: args.max_routes], stress


def plot_label_stress(stress: pd.DataFrame, out_path: Path) -> None:
    if stress.empty:
        return
    top = stress.head(24).iloc[::-1]
    plt.figure(figsize=(14, max(6, 0.5 * len(top))))
    plt.barh(top["label"], top["max_tangent_ratio"])
    for i, r in enumerate(top.itertuples()):
        plt.text(float(r.max_tangent_ratio), i, f" cap={r.min_capture_rate:.3f}", va="center", fontsize=8)
    plt.xlabel("max tangent/radial ratio in selected ED route")
    plt.title("26ED-LITE remaining label tangent tail inside selected route")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_capture_tangent(df: pd.DataFrame, out_path: Path, args: argparse.Namespace) -> None:
    if df.empty:
        return
    plt.figure(figsize=(10, 7))
    x = df.get("worst_capture_rate", pd.Series(dtype=float)).astype(float)
    y = df.get("worst_tangent_ratio", pd.Series(dtype=float)).astype(float)
    c = df.get("ed_tail_score", pd.Series(np.zeros(len(df)))).astype(float)
    plt.scatter(x, y, c=c, s=72, alpha=0.88)
    plt.axvline(args.target_capture, linewidth=1, label="capture target")
    plt.axhline(args.strict_tangent_target, linewidth=1, label="strict tangent target")
    plt.axhline(args.relaxed_tangent, linewidth=1, linestyle="--", label="relaxed tangent")
    plt.xlabel("verified worst capture_rate")
    plt.ylabel("verified worst tangent/radial ratio")
    plt.title("26ED-LITE tangent-tail refined routes: capture vs tangent")
    plt.colorbar(label="ED tail score")
    plt.legend()
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
    plt.title("26ED-LITE tangent-tail route pass rates")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_route_scores(summary: pd.DataFrame, out_path: Path) -> None:
    if summary.empty:
        return
    top = summary.head(18).iloc[::-1]
    plt.figure(figsize=(14, max(6, 0.5 * len(top))))
    plt.barh(top["route_name"], top["ed_tail_score"])
    for i, r in enumerate(top.itertuples()):
        plt.text(float(r.ed_tail_score), i, f" cap={r.min_worst_capture_rate:.3f} tan={r.max_worst_tangent_ratio:.3f} labels={int(r.explicit_label_count)}", va="center", fontsize=8)
    plt.xlabel("ED tangent-tail score")
    plt.title("26ED-LITE tangent-tail route scores")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def run(args: argparse.Namespace) -> None:
    t0 = time.time()
    ROOT.mkdir(parents=True, exist_ok=True)

    ec = import_by_path(EC_SCRIPT, "phase26ec_lite")
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

    ec_name, ec_route, ec_summary = load_ec_route()

    route_names_needed = set(ec_route.values())
    route_names_needed.update([ec_route.get("*", "")])
    pool = build_candidate_pool(dz, ea, args.max_candidates, sorted(n for n in route_names_needed if n), rng)
    pool_sources = {name: src for name, _ov, src in pool}

    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
    pool_seed_cases: Dict[int, Dict[str, pd.DataFrame]] = {}
    pool_rows = []

    print(json.dumps({
        "phase": PHASE,
        "title": TITLE,
        "root": str(ROOT),
        "device": args.device,
        "torch": torch_info,
        "source_route": ec_name,
        "source_explicit_labels": route_explicit_count(ec_route),
        "target_capture": args.target_capture,
        "strict_tangent_target": args.strict_tangent_target,
        "relaxed_tangent": args.relaxed_tangent,
        "seeds": seeds,
        "max_candidates": args.max_candidates,
        "candidate_variants": len(pool),
        "required_route_variants": len([n for n in route_names_needed if n]),
    }, indent=2))

    cache: Dict[Any, Any] = {}
    for seed in seeds:
        cases_by_variant = dz.evaluate_pool_for_seed(
            ds_lite, drfast, dr, pool, basins, seed, 1, False, cache
        )
        for name, cases in cases_by_variant.items():
            try:
                if cases is not None and not cases.empty:
                    cases = cases.copy()
                    cases["variant"] = name
                    cases["seed"] = seed
                    cases_by_variant[name] = cases
                    metrics, _ = dz.score_cases(
                        cases,
                        args.target_capture,
                        args.strict_tangent_target,
                        args.relaxed_tangent,
                        args.target_q20,
                    )
                    metrics = dict(metrics)
                    metrics.update({"variant": name, "seed": seed, "variant_source": pool_sources.get(name, "")})
                    pool_rows.append(metrics)
            except Exception as e:
                print(f"[26ED] WARN variant failed seed={seed} name={name}: {e}")
        pool_seed_cases[seed] = cases_by_variant
        print(f"[26ED] seed={seed} evaluated_variants={len(cases_by_variant)}")

    routes, initial_stress = build_tail_routes(dz, ec_route, pool_seed_cases, args)
    print(f"[26ED] candidate_routes={len(routes)} weak_labels={(initial_stress.head(args.max_tail_labels)['label'].tolist() if not initial_stress.empty else [])}")

    route_rows = []
    route_case_rows = []
    for route_name, route_map, source in routes:
        for seed in seeds:
            cases_by_variant = pool_seed_cases.get(seed, {})
            if not cases_by_variant:
                continue
            try:
                metrics, rcases = dz.compose_route(route_name, route_map, cases_by_variant, args.target_capture, args.strict_tangent_target, args.relaxed_tangent, args.target_q20)
                metrics = dict(metrics)
                metrics.update({
                    "phase": PHASE,
                    "route_name": route_name,
                    "route_source": source,
                    "seed": seed,
                    "route_map_json": json.dumps(route_map, sort_keys=True),
                    "explicit_label_count": route_explicit_count(route_map),
                })
                route_rows.append(metrics)
                if rcases is not None and not rcases.empty:
                    rc = rcases.copy()
                    rc["route_name"] = route_name
                    rc["seed"] = seed
                    route_case_rows.append(rc)
            except Exception as e:
                print(f"[26ED] WARN route failed seed={seed} route={route_name}: {e}")

    pool_df = pd.DataFrame(pool_rows)
    route_results = pd.DataFrame(route_rows)
    route_cases = pd.concat(route_case_rows, ignore_index=True) if route_case_rows else pd.DataFrame()
    summary = summarize_routes(route_results, args)

    best_route_name = str(summary.iloc[0]["route_name"]) if not summary.empty else ""
    best_route_map = parse_route_map(summary.iloc[0]["route_map_json"]) if not summary.empty else {}
    best_cases = route_cases[route_cases["route_name"].astype(str) == best_route_name].copy() if not route_cases.empty and best_route_name else pd.DataFrame()
    best_stress = label_stress_from_cases(best_cases)

    pool_df.to_csv(ROOT / "phase26ed_lite_pool_case_results.csv", index=False)
    route_results.to_csv(ROOT / "phase26ed_lite_route_results.csv", index=False)
    route_cases.to_csv(ROOT / "phase26ed_lite_route_cases.csv", index=False)
    summary.to_csv(ROOT / "phase26ed_lite_route_summary.csv", index=False)
    initial_stress.to_csv(ROOT / "phase26ed_lite_initial_label_stress.csv", index=False)
    best_stress.to_csv(ROOT / "phase26ed_lite_best_label_stress.csv", index=False)

    plot_label_stress(best_stress, ROOT / "phase26ed_lite_best_route_label_stress.png")
    plot_capture_tangent(summary.rename(columns={"min_worst_capture_rate":"worst_capture_rate", "max_worst_tangent_ratio":"worst_tangent_ratio"}), ROOT / "phase26ed_lite_route_capture_vs_tangent.png", args)
    plot_pass_rates(summary, ROOT / "phase26ed_lite_pass_rates.png")
    plot_route_scores(summary, ROOT / "phase26ed_lite_route_scores.png")

    verdict = "NO_PASS"
    if not summary.empty:
        b = summary.iloc[0]
        if bool(b.get("stable_strict_all_seeds")):
            verdict = "STRICT_PASS_TANGENT_REFINED"
        elif bool(b.get("stable_relaxed_all_seeds")):
            verdict = "RELAXED_PASS_ONLY"

    out_json = {
        "phase": PHASE,
        "title": TITLE,
        "verdict": verdict,
        "source_route": ec_name,
        "best_route": best_route_name,
        "best_route_map": best_route_map,
        "best_explicit_label_count": route_explicit_count(best_route_map) if best_route_map else None,
        "best_min_worst_capture_rate": fnum(summary.iloc[0]["min_worst_capture_rate"]) if not summary.empty else None,
        "best_max_worst_tangent_ratio": fnum(summary.iloc[0]["max_worst_tangent_ratio"]) if not summary.empty else None,
        "best_strict_pass_rate": fnum(summary.iloc[0]["strict_pass_rate"]) if not summary.empty else None,
        "best_relaxed_pass_rate": fnum(summary.iloc[0]["relaxed_pass_rate"]) if not summary.empty else None,
        "target_capture": args.target_capture,
        "strict_tangent_target": args.strict_tangent_target,
        "relaxed_tangent": args.relaxed_tangent,
        "num_candidate_routes": int(len(routes)),
        "num_route_rows": int(len(route_results)),
        "num_pool_rows": int(len(pool_df)),
        "runtime_sec": time.time() - t0,
        "outputs": {
            "route_summary_csv": str(ROOT / "phase26ed_lite_route_summary.csv"),
            "route_results_csv": str(ROOT / "phase26ed_lite_route_results.csv"),
            "route_cases_csv": str(ROOT / "phase26ed_lite_route_cases.csv"),
            "best_label_stress_csv": str(ROOT / "phase26ed_lite_best_label_stress.csv"),
            "best_route_label_stress_png": str(ROOT / "phase26ed_lite_best_route_label_stress.png"),
            "route_capture_vs_tangent_png": str(ROOT / "phase26ed_lite_route_capture_vs_tangent.png"),
            "pass_rates_png": str(ROOT / "phase26ed_lite_pass_rates.png"),
            "route_scores_png": str(ROOT / "phase26ed_lite_route_scores.png"),
        },
    }
    (ROOT / "phase26ed_lite_summary.json").write_text(json.dumps(out_json, indent=2), encoding="utf-8")
    print(json.dumps(out_json, indent=2))


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=f"{PHASE}: {TITLE}")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--seeds", default="12,13")
    ap.add_argument("--max-candidates", type=int, default=24)
    ap.add_argument("--max-routes", type=int, default=24)

    ap.add_argument("--steps", type=int, default=160)
    ap.add_argument("--dt", type=float, default=0.035)
    ap.add_argument("--num-trajs", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.22)

    ap.add_argument("--target-capture", type=float, default=0.285)
    ap.add_argument("--strict-tangent-target", type=float, default=2.00)
    ap.add_argument("--relaxed-tangent", type=float, default=2.10)
    ap.add_argument("--target-q20", type=float, default=2.25)

    ap.add_argument("--tail-label-tangent-floor", type=float, default=1.96)
    ap.add_argument("--tail-label-capture-floor", type=float, default=0.315)
    ap.add_argument("--max-tail-labels", type=int, default=4)
    ap.add_argument("--max-swaps-per-label", type=int, default=3)
    ap.add_argument("--max-pair-labels", type=int, default=2)
    ap.add_argument("--max-pair-choices", type=int, default=2)
    ap.add_argument("--label-min-capture", type=float, default=0.285)
    ap.add_argument("--label-max-tangent", type=float, default=2.10)
    ap.add_argument("--candidate-min-capture", type=float, default=0.280)
    ap.add_argument("--candidate-max-tangent", type=float, default=2.18)
    return ap


if __name__ == "__main__":
    run(build_argparser().parse_args())
