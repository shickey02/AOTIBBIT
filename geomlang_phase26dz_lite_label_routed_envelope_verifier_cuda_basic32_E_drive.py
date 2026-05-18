# geomlang_phase26dz_lite_label_routed_envelope_verifier_cuda_basic32_E_drive.py
# Phase 26DZ-LITE — label-routed envelope verifier after DY.
#
# Why this exists:
#   DX/DY kept returning the same shape: a global bowl can be clean or it can be
#   capture-heavy, but the same global override does not reliably satisfy all named
#   culprit labels at once. DY's best result was still a last-inch miss: clean tangent,
#   but screen capture stuck at the 0.28125 quantum.
#
#   DZ-LITE tests the next hypothesis directly:
#       "Is the remaining failure caused by using one global override for labels that
#        need different local behavior?"
#
#   DZ does not pretend this is already a production field. It builds *virtual routed
#   envelopes*: evaluate a small pool of known good global candidates, then compose a
#   routed envelope by taking label A from variant X, label B from variant Y, etc.
#   If a routed envelope passes where every global candidate fails, the next real phase
#   should implement an actual conditional gate/latch inside the force law. If routed
#   envelopes also fail, the culprit is deeper than global-vs-local routing.
#
# Recommended first run:
#   python bbit_geomlang/geomlang_phase26dz_lite_label_routed_envelope_verifier_cuda_basic32_E_drive.py --device cuda
#
# Fast smoke:
#   python bbit_geomlang/geomlang_phase26dz_lite_label_routed_envelope_verifier_cuda_basic32_E_drive.py --device cuda --pool-size 6 --seed-count 2
#
# Wider controlled:
#   python bbit_geomlang/geomlang_phase26dz_lite_label_routed_envelope_verifier_cuda_basic32_E_drive.py --device cuda --pool-size 10 --route-count 18 --seed-count 3 --audit-repeats 1
#
# Suspicious/full-tail audit:
#   python bbit_geomlang/geomlang_phase26dz_lite_label_routed_envelope_verifier_cuda_basic32_E_drive.py --device cuda --pool-size 10 --route-count 18 --seed-count 3 --audit-repeats 2 --sample-full-tail

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

PHASE = "26DZ-LITE"
TITLE = "label-routed envelope verifier after DY"
ROOT = Path(r"E:\BBIT\outputs_basic32")
HERE = Path(__file__).resolve().parent

DY_FILE = HERE / "geomlang_phase26dy_lite_terminal_latch_snap_cuda_basic32_E_drive.py"
DS_LITE_FILE = HERE / "geomlang_phase26ds_lite_strict_line_rescue_cuda_basic32_E_drive.py"
DS_FULL_FILE = HERE / "geomlang_phase26ds_strict_line_rescue_cuda_basic32_E_drive.py"
DR_FAST_FILE = HERE / "geomlang_phase26dr_fast_staged_cuda_basic32_E_drive.py"

DY_SUMMARY = ROOT / "phase26dy_lite_summary.json"
DY_VARIANTS = ROOT / "phase26dy_lite_variant_summary.csv"
DY_CASES = ROOT / "phase26dy_lite_terminal_latch_cases.csv"
DX_SUMMARY = ROOT / "phase26dx_lite_summary.json"
DX_VARIANTS = ROOT / "phase26dx_lite_variant_summary.csv"
DX_CASES = ROOT / "phase26dx_lite_culprit_gated_cases.csv"
DV_SUMMARY = ROOT / "phase26dv_lite_summary.json"
DV_VARIANTS = ROOT / "phase26dv_lite_variant_summary.csv"
DS_SUMMARY = ROOT / "phase26ds_lite_summary.json"
DS_CASES = ROOT / "phase26ds_lite_case_results.csv"

PARAMS = [
    "BOWL_RADIUS_FRAC",
    "BOWL_SEAT_AXIS_GAIN",
    "BOWL_DIRECTIONAL_BLEND",
    "BOWL_NORM_CAP_MULT",
    "BOWL_SHELL_RADIAL_GAIN",
    "BOWL_TANGENT_KILL",
]

CRITICAL_LABELS = [
    "base",
    "screen_small_00",
    "screen_medium_00",
    "stress_cap_tk",
    "stress_shell_blend",
    "stress_radius_in",
    "stress_seat_down",
]

# Labels not seen in prior runs are routed through the default/global route member.
FALLBACK_DX_BEST = {
    "BOWL_RADIUS_FRAC": 0.7123398273572922,
    "BOWL_SEAT_AXIS_GAIN": 0.435,
    "BOWL_DIRECTIONAL_BLEND": 1.025,
    "BOWL_NORM_CAP_MULT": 1.123734619299985,
    "BOWL_SHELL_RADIAL_GAIN": 0.172,
    "BOWL_TANGENT_KILL": 0.9811105492968176,
}
FALLBACK_DW_BALANCED = {
    "BOWL_RADIUS_FRAC": 0.6997443058798729,
    "BOWL_SEAT_AXIS_GAIN": 0.42,
    "BOWL_DIRECTIONAL_BLEND": 0.9711619081247089,
    "BOWL_NORM_CAP_MULT": 1.10,
    "BOWL_SHELL_RADIAL_GAIN": 0.14705073683834083,
    "BOWL_TANGENT_KILL": 0.94,
}
FALLBACK_LOW_CAPTURE_CLEAN = {
    "BOWL_RADIUS_FRAC": 0.7291,
    "BOWL_SEAT_AXIS_GAIN": 0.4031,
    "BOWL_DIRECTIONAL_BLEND": 0.9981,
    "BOWL_NORM_CAP_MULT": 1.1654,
    "BOWL_SHELL_RADIAL_GAIN": 0.1648,
    "BOWL_TANGENT_KILL": 0.9718,
}
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


def valid_ov(ov: Dict[str, Any] | None) -> bool:
    return isinstance(ov, dict) and all(k in ov for k in PARAMS)


def clean_ov(ov: Dict[str, Any]) -> Dict[str, float]:
    return {k: fnum(ov.get(k), FALLBACK_DW_BALANCED[k]) for k in PARAMS}


def row_key(ov: Dict[str, float], ndigits: int = 6) -> Tuple[float, ...]:
    return tuple(round(float(ov[k]), ndigits) for k in PARAMS)


def add_unique(out: List[Tuple[str, Dict[str, float], str]], seen: set, name: str, ov: Dict[str, Any], source: str) -> None:
    if not valid_ov(ov):
        return
    cov = clean_ov(ov)
    key = row_key(cov)
    if key not in seen:
        out.append((name[:118], cov, source))
        seen.add(key)


def overrides_from_row(row: pd.Series) -> Dict[str, float] | None:
    if all(k in row.index and pd.notna(row[k]) for k in PARAMS):
        return clean_ov({k: row[k] for k in PARAMS})
    for col in ("overrides", "overrides_json", "best_overrides"):
        if col in row.index:
            obj = parse_json_cell(row[col])
            if valid_ov(obj):
                return clean_ov(obj)  # type: ignore[arg-type]
    return None


def load_json_override(path: Path, key: str = "best_overrides") -> Tuple[str, Dict[str, float], str] | None:
    if not path.exists():
        return None
    try:
        js = json.loads(path.read_text(encoding="utf-8"))
        ov = js.get(key)
        name = str(js.get("best_variant") or path.stem)
        if valid_ov(ov):
            return name, clean_ov(ov), str(path)
    except Exception as e:
        print(f"[{PHASE}] WARNING: could not read {path.name}: {e!r}")
    return None


def variant_pool(pool_size: int) -> List[Tuple[str, Dict[str, float], str]]:
    """Build a compact variant pool from DY/DX/DV plus hand anchors."""
    out: List[Tuple[str, Dict[str, float], str]] = []
    seen: set = set()
    add_unique(out, seen, "dz_anchor_dy_best_dw_balanced", FALLBACK_DW_BALANCED, "fallback_dw_balanced")
    add_unique(out, seen, "dz_anchor_dx_best", FALLBACK_DX_BEST, "fallback_dx_best")
    add_unique(out, seen, "dz_anchor_low_capture_clean", FALLBACK_LOW_CAPTURE_CLEAN, "fallback_low_capture_clean")
    add_unique(out, seen, "dz_anchor_dv_capture_grid", FALLBACK_DV_BEST, "fallback_dv_best")

    for p in (DY_SUMMARY, DX_SUMMARY, DV_SUMMARY, DS_SUMMARY):
        loaded = load_json_override(p)
        if loaded:
            name, ov, src = loaded
            add_unique(out, seen, f"prior_{name}", ov, src)

    for p, score_cols in [
        (DY_VARIANTS, ["stable_named_gate_relaxed_all_seeds", "named_gate_relaxed_rate", "dy_family_score"]),
        (DX_VARIANTS, ["stable_named_gate_relaxed_all_seeds", "named_gate_relaxed_rate", "dx_family_score"]),
        (DV_VARIANTS, ["near_capture_rate", "dv_family_score", "min_worst_capture_rate"]),
    ]:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
            if df.empty:
                continue
            rank = pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
            for c in score_cols:
                if c in df:
                    if df[c].dtype == bool:
                        rank += df[c].astype(float) * 5.0
                    else:
                        rank += pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            # also favor near-threshold capture and controlled tangent when columns exist
            if "min_label_screen_capture" in df:
                rank += 60.0 * pd.to_numeric(df["min_label_screen_capture"], errors="coerce").fillna(0.0)
            if "min_worst_capture_rate" in df:
                rank += 60.0 * pd.to_numeric(df["min_worst_capture_rate"], errors="coerce").fillna(0.0)
            if "max_label_guard_tangent" in df:
                rank -= 1.5 * pd.to_numeric(df["max_label_guard_tangent"], errors="coerce").fillna(99.0)
            elif "max_worst_tangent_ratio" in df:
                rank -= 1.5 * pd.to_numeric(df["max_worst_tangent_ratio"], errors="coerce").fillna(99.0)
            df = df.assign(dz_rank=rank).sort_values("dz_rank", ascending=False)
            for _, row in df.head(max(pool_size, 6)).iterrows():
                ov = overrides_from_row(row)
                if ov is not None:
                    add_unique(out, seen, f"prior_{p.stem}_{row.get('variant', row.get('case', 'row'))}", ov, str(p))
        except Exception as e:
            print(f"[{PHASE}] WARNING: could not read {p.name}: {e!r}")

    return out[: max(4, pool_size)]


def load_historical_label_scores(pool: List[Tuple[str, Dict[str, float], str]], target_cap: float, target_tan: float, relaxed_tan: float, target_q20: float) -> pd.DataFrame:
    """Use prior case CSVs to estimate which prior variant was best for each label."""
    frames = []
    for p, vcol in [(DY_CASES, "variant"), (DX_CASES, "variant"), (DS_CASES, "case")]:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
            if df.empty or "envelope_label" not in df:
                continue
            if vcol not in df:
                # DW/DX sometimes used audit_variant instead of variant.
                if "audit_variant" in df:
                    vcol = "audit_variant"
                elif "parent_case" in df:
                    vcol = "parent_case"
                else:
                    continue
            df = df.copy()
            df["hist_source"] = p.name
            df["hist_variant"] = df[vcol].astype(str)
            frames.append(df)
        except Exception as e:
            print(f"[{PHASE}] WARNING: could not load historical cases {p.name}: {e!r}")
    if not frames:
        return pd.DataFrame()
    cases = pd.concat(frames, ignore_index=True)
    for c in ["capture_rate", "tangent_ratio", "dp_score"]:
        if c not in cases:
            cases[c] = 0.0
    cases["label_score"] = (
        100.0 * (pd.to_numeric(cases["capture_rate"], errors="coerce").fillna(0.0) - target_cap)
        - 2.5 * np.maximum(0.0, pd.to_numeric(cases["tangent_ratio"], errors="coerce").fillna(999.0) - relaxed_tan)
        - 1.5 * np.maximum(0.0, pd.to_numeric(cases["tangent_ratio"], errors="coerce").fillna(999.0) - target_tan)
        + 3.0 * np.maximum(0.0, pd.to_numeric(cases["dp_score"], errors="coerce").fillna(-999.0) - target_q20)
    )
    label_df = cases.groupby(["envelope_label", "hist_variant"], sort=False).agg(
        mean_label_score=("label_score", "mean"),
        min_capture=("capture_rate", "min"),
        max_tangent=("tangent_ratio", "max"),
        min_dp=("dp_score", "min"),
        n=("capture_rate", "size"),
    ).reset_index()
    return label_df.sort_values(["envelope_label", "mean_label_score"], ascending=[True, False])


def pass_flags_from_metrics(cap: float, tan: float, q20: float, target_cap: float, target_tan: float, relaxed_tan: float, target_q20: float) -> Dict[str, bool]:
    return {
        "pass_cap": cap >= target_cap,
        "pass_tan_strict": tan <= target_tan,
        "pass_tan_relaxed": tan <= relaxed_tan,
        "pass_q20": q20 >= target_q20,
        "pass_strict_line": cap >= target_cap and tan <= target_tan and q20 >= target_q20,
        "pass_relaxed_line": cap >= target_cap and tan <= relaxed_tan and q20 >= target_q20,
    }


def score_cases(df: pd.DataFrame, target_cap: float, target_tan: float, relaxed_tan: float, target_q20: float) -> Tuple[Dict[str, Any], pd.DataFrame]:
    if df.empty:
        return {}, df
    out = df.copy()
    for c in ["capture_rate", "tangent_ratio", "dp_score"]:
        if c not in out:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    out["capture_deficit"] = np.maximum(0.0, target_cap - out["capture_rate"])
    out["tangent_excess_strict"] = np.maximum(0.0, out["tangent_ratio"] - target_tan)
    out["tangent_excess_relaxed"] = np.maximum(0.0, out["tangent_ratio"] - relaxed_tan)
    out["dp_deficit"] = np.maximum(0.0, target_q20 - out["dp_score"])
    out["case_pass_capture"] = out["capture_rate"] >= target_cap
    out["case_pass_tangent_strict"] = out["tangent_ratio"] <= target_tan
    out["case_pass_tangent_relaxed"] = out["tangent_ratio"] <= relaxed_tan
    out["case_pass_dp"] = out["dp_score"] >= target_q20
    out["case_pass_strict"] = out["case_pass_capture"] & out["case_pass_tangent_strict"] & out["case_pass_dp"]
    out["case_pass_relaxed"] = out["case_pass_capture"] & out["case_pass_tangent_relaxed"] & out["case_pass_dp"]

    cap = float(out["capture_rate"].min())
    tan = float(out["tangent_ratio"].max())
    worst_dp = float(out["dp_score"].min())
    q20 = float(out["dp_score"].quantile(0.20)) if len(out) else worst_dp
    mean_dp = float(out["dp_score"].mean()) if len(out) else worst_dp
    base_score = (
        125.0 * (cap - target_cap)
        - 4.0 * max(0.0, tan - relaxed_tan)
        - 2.0 * max(0.0, tan - target_tan)
        + 3.0 * max(0.0, q20 - target_q20)
        + 5.0 * float(cap >= target_cap)
        + 4.0 * float(tan <= relaxed_tan)
        + 6.0 * float(tan <= target_tan)
        + 2.0 * float(q20 >= target_q20)
    )
    metrics = {
        "worst_capture_rate": cap,
        "worst_tangent_ratio": tan,
        "worst_dp_score": worst_dp,
        "q20_dp_score": q20,
        "mean_dp_score": mean_dp,
        "dz_route_score_run": float(base_score),
        **pass_flags_from_metrics(cap, tan, q20, target_cap, target_tan, relaxed_tan, target_q20),
    }
    return metrics, out


def build_routes(pool: List[Tuple[str, Dict[str, float], str]], historical: pd.DataFrame, route_count: int, rng: random.Random) -> List[Tuple[str, Dict[str, str], str]]:
    """Return route maps: envelope_label -> variant_name, with key '*' as default."""
    names = [x[0] for x in pool]
    default = names[0]
    dx_best = next((n for n in names if "dx_best" in n), names[min(1, len(names)-1)])
    low_clean = next((n for n in names if "low_capture" in n), names[min(2, len(names)-1)])
    dv_cap = next((n for n in names if "dv" in n or "capture_grid" in n), names[-1])

    routes: List[Tuple[str, Dict[str, str], str]] = []
    seen_route = set()

    def add_route(name: str, route: Dict[str, str], source: str) -> None:
        route = {k: v for k, v in route.items() if v in names or k == "*"}
        route.setdefault("*", default)
        key = tuple(sorted(route.items()))
        if key not in seen_route:
            routes.append((name[:118], route, source))
            seen_route.add(key)

    # Baseline: no routing, useful control.
    for n in names[: min(5, len(names))]:
        add_route(f"dz_global_{n[:70]}", {"*": n}, "global_control")

    # Hand-built hypotheses from DW/DX/DY diagnosis.
    add_route("dz_route_screen_lift_guard_stress", {
        "*": default,
        "screen_small_00": dv_cap,
        "screen_medium_00": dv_cap,
        "base": low_clean,
        "stress_cap_tk": low_clean,
        "stress_seat_down": low_clean,
        "stress_shell_blend": dx_best,
        "stress_radius_in": dx_best,
    }, "hand_screen_lift_guard_stress")

    add_route("dz_route_lowcap_for_screens_balanced_for_stress", {
        "*": default,
        "screen_small_00": low_clean,
        "screen_medium_00": low_clean,
        "base": dx_best,
        "stress_cap_tk": dx_best,
        "stress_shell_blend": dx_best,
        "stress_radius_in": dx_best,
        "stress_seat_down": dx_best,
    }, "hand_lowcap_screens")

    add_route("dz_route_capture_grid_screens_clean_tangent_tail", {
        "*": low_clean,
        "screen_small_00": dv_cap,
        "screen_medium_00": dv_cap,
        "base": low_clean,
        "stress_cap_tk": low_clean,
        "stress_shell_blend": low_clean,
        "stress_radius_in": low_clean,
        "stress_seat_down": low_clean,
    }, "hand_capture_screens_clean_tail")

    # Historical greedy: choose best observed variant name by label when it exists in pool.
    if not historical.empty:
        route: Dict[str, str] = {"*": default}
        # Loose name matching, because prior rows may include prefixes/suffixes around the same variant.
        for label in CRITICAL_LABELS:
            sub = historical[historical["envelope_label"] == label]
            chosen = None
            for _, r in sub.iterrows():
                hv = str(r["hist_variant"])
                for n in names:
                    if hv in n or n in hv or hv.split("_seed")[0] in n:
                        chosen = n
                        break
                if chosen:
                    break
            if chosen:
                route[label] = chosen
        add_route("dz_route_historical_greedy_label_best", route, "historical_label_scores")

    # Measured greedy among pool will be added after evaluation in run(); placeholder routes are only random/hand now.
    # Random route cloud: small sample, not exhaustive.
    for i in range(max(0, route_count - len(routes))):
        route = {"*": rng.choice(names[: min(4, len(names))])}
        for lab in CRITICAL_LABELS:
            # Bias screens toward capture candidates, stress/base toward clean candidates.
            if lab.startswith("screen"):
                choices = [dv_cap, low_clean, dx_best] + names[: min(4, len(names))]
            elif lab in ("base", "stress_cap_tk", "stress_seat_down"):
                choices = [low_clean, dx_best, default] + names[: min(4, len(names))]
            else:
                choices = [dx_best, low_clean, default] + names[: min(4, len(names))]
            route[lab] = rng.choice([c for c in choices if c in names])
        add_route(f"dz_route_cloud_{i:03d}", route, "route_cloud")

    return routes[:route_count]


def evaluate_pool_for_seed(ds_lite: Any, drfast: Any, dr: Any, pool: List[Tuple[str, Dict[str, float], str]], basins: Any, seed: int, audit_repeats: int, sample_full_tail: bool, cache: Dict[Any, Any]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for name, ov, src in pool:
        env = ds_lite.micro_envelope(drfast, dr, ov, seed=seed, repeats=audit_repeats, use_full_tail=sample_full_tail)
        case = f"dz_pool_{name}_seed{seed}"
        agg, erows = drfast.evaluate_envelope(dr, case, ov, basins, seed=seed, envelope=env, cache=cache, phase_label="dz_label_route_pool")
        rows = []
        for er in erows:
            d = dict(er)
            d["source_variant"] = name
            d["source_variant_source"] = src
            d["dz_seed"] = seed
            for k, v in ov.items():
                d[k] = v
            rows.append(d)
        df = pd.DataFrame(rows)
        if not df.empty and "envelope_label" in df:
            out[name] = df
        else:
            out[name] = pd.DataFrame()
        print(
            f"[{PHASE}] pool {name[:58]:58s} seed={seed} "
            f"cap={fnum(agg.get('worst_capture_rate')):.3f} tan={fnum(agg.get('worst_tangent_ratio')):.3f} q20={fnum(agg.get('q20_dp_score')):.3f}"
        )
    return out


def compose_route(route_name: str, route: Dict[str, str], seed_cases: Dict[str, pd.DataFrame], target_cap: float, target_tan: float, relaxed_tan: float, target_q20: float) -> Tuple[Dict[str, Any], pd.DataFrame]:
    parts = []
    default_name = route.get("*") or next(iter(seed_cases.keys()))
    all_labels = sorted(set().union(*[set(df.get("envelope_label", pd.Series(dtype=str)).astype(str).tolist()) for df in seed_cases.values() if not df.empty]))
    for lab in all_labels:
        src_name = route.get(lab, default_name)
        df = seed_cases.get(src_name, pd.DataFrame())
        if df.empty or "envelope_label" not in df:
            continue
        sub = df[df["envelope_label"].astype(str) == lab].copy()
        if sub.empty and src_name != default_name:
            sub = seed_cases.get(default_name, pd.DataFrame())
            if not sub.empty and "envelope_label" in sub:
                sub = sub[sub["envelope_label"].astype(str) == lab].copy()
        if sub.empty:
            continue
        sub["route_name"] = route_name
        sub["routed_from_variant"] = src_name
        sub["route_default_variant"] = default_name
        parts.append(sub)
    routed_cases = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    metrics, scored = score_cases(routed_cases, target_cap, target_tan, relaxed_tan, target_q20)
    metrics.update({
        "route_name": route_name,
        "route_map_json": json.dumps(route, sort_keys=True),
        "route_default_variant": default_name,
        "num_labels": int(scored["envelope_label"].nunique()) if not scored.empty and "envelope_label" in scored else 0,
        "num_case_rows": int(len(scored)),
    })
    return metrics, scored


def measured_best_routes(pool_seed_cases: Dict[int, Dict[str, pd.DataFrame]], routes: List[Tuple[str, Dict[str, str], str]], target_cap: float, target_tan: float, relaxed_tan: float, target_q20: float, max_extra: int = 3) -> List[Tuple[str, Dict[str, str], str]]:
    """Create routes from actual pool measurements on the first seed."""
    if not pool_seed_cases:
        return []
    first_seed = sorted(pool_seed_cases.keys())[0]
    cases_by_variant = pool_seed_cases[first_seed]
    label_best: Dict[str, str] = {}
    label_best_clean: Dict[str, str] = {}
    label_best_capture: Dict[str, str] = {}
    for lab in CRITICAL_LABELS:
        rows = []
        for name, df in cases_by_variant.items():
            if df.empty or "envelope_label" not in df:
                continue
            sub = df[df["envelope_label"].astype(str) == lab]
            if sub.empty:
                continue
            metrics, _ = score_cases(sub, target_cap, target_tan, relaxed_tan, target_q20)
            rows.append((name, metrics["worst_capture_rate"], metrics["worst_tangent_ratio"], metrics["q20_dp_score"], metrics["dz_route_score_run"]))
        if not rows:
            continue
        best = sorted(rows, key=lambda x: (x[4], x[1], -x[2]), reverse=True)[0]
        best_clean = sorted(rows, key=lambda x: (x[2], -x[1]))[0]
        best_cap = sorted(rows, key=lambda x: (x[1], -x[2]), reverse=True)[0]
        label_best[lab] = best[0]
        label_best_clean[lab] = best_clean[0]
        label_best_capture[lab] = best_cap[0]
    if not label_best:
        return []
    default = next(iter(cases_by_variant.keys()))
    out = []
    route = {"*": default, **label_best}
    out.append(("dz_route_measured_best_per_label", route, "measured_first_seed"))
    route2 = {"*": default, **label_best_clean}
    for lab in ["screen_small_00", "screen_medium_00"]:
        if lab in label_best_capture:
            route2[lab] = label_best_capture[lab]
    out.append(("dz_route_measured_clean_tail_capture_screens", route2, "measured_first_seed"))
    route3 = {"*": default, **label_best_capture}
    for lab in ["base", "stress_cap_tk", "stress_seat_down", "stress_shell_blend", "stress_radius_in"]:
        if lab in label_best_clean:
            route3[lab] = label_best_clean[lab]
    out.append(("dz_route_measured_capture_default_clean_stress", route3, "measured_first_seed"))
    return out[:max_extra]


def summarize_routes(results_df: pd.DataFrame, target_cap: float, target_tan: float, relaxed_tan: float, target_q20: float) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()
    rows = []
    for route_name, g in results_df.groupby("route_name", sort=False):
        strict_rate = float(g["pass_strict_line"].mean())
        relaxed_rate = float(g["pass_relaxed_line"].mean())
        min_cap = float(g["worst_capture_rate"].min())
        max_tan = float(g["worst_tangent_ratio"].max())
        min_q20 = float(g["q20_dp_score"].min())
        mean_score = float(g["dz_route_score_run"].mean())
        cap_std = float(g["worst_capture_rate"].std(ddof=0)) if len(g) > 1 else 0.0
        tan_std = float(g["worst_tangent_ratio"].std(ddof=0)) if len(g) > 1 else 0.0
        family_score = (
            12.0 * strict_rate
            + 9.0 * relaxed_rate
            + 0.50 * mean_score
            + 140.0 * (min_cap - target_cap)
            + 4.0 * max(0.0, min_q20 - target_q20)
            - 3.0 * max(0.0, max_tan - relaxed_tan)
            - 1.5 * max(0.0, max_tan - target_tan)
            - 18.0 * cap_std
            - 0.7 * tan_std
        )
        rows.append({
            "route_name": route_name,
            "n": int(len(g)),
            "strict_pass_rate": strict_rate,
            "relaxed_pass_rate": relaxed_rate,
            "stable_strict_all_seeds": bool(g["pass_strict_line"].all()),
            "stable_relaxed_all_seeds": bool(g["pass_relaxed_line"].all()),
            "dz_route_family_score": float(family_score),
            "mean_route_score": mean_score,
            "min_worst_capture_rate": min_cap,
            "max_worst_tangent_ratio": max_tan,
            "min_q20_dp_score": min_q20,
            "std_worst_capture_rate": cap_std,
            "std_worst_tangent_ratio": tan_std,
            "route_default_variant": str(g["route_default_variant"].iloc[0]),
            "route_map_json": str(g["route_map_json"].iloc[0]),
            "num_labels_min": int(g["num_labels"].min()),
            "num_case_rows_min": int(g["num_case_rows"].min()),
        })
    return pd.DataFrame(rows).sort_values(
        ["stable_strict_all_seeds", "stable_relaxed_all_seeds", "relaxed_pass_rate", "dz_route_family_score"],
        ascending=[False, False, False, False],
    )


def summarize_variant_label_matrix(case_rows: pd.DataFrame, target_cap: float, target_tan: float, relaxed_tan: float, target_q20: float) -> pd.DataFrame:
    if case_rows.empty or "source_variant" not in case_rows or "envelope_label" not in case_rows:
        return pd.DataFrame()
    rows = []
    for (v, lab), g in case_rows.groupby(["source_variant", "envelope_label"], sort=False):
        metrics, _ = score_cases(g, target_cap, target_tan, relaxed_tan, target_q20)
        rows.append({
            "source_variant": v,
            "envelope_label": lab,
            "label_score": metrics.get("dz_route_score_run", 0.0),
            "min_capture": metrics.get("worst_capture_rate", 0.0),
            "max_tangent": metrics.get("worst_tangent_ratio", 999.0),
            "q20_dp": metrics.get("q20_dp_score", -999.0),
            "passes_strict": metrics.get("pass_strict_line", False),
            "passes_relaxed": metrics.get("pass_relaxed_line", False),
            "n": int(len(g)),
        })
    return pd.DataFrame(rows).sort_values(["envelope_label", "label_score"], ascending=[True, False])


def plot_route_scores(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    top = df.sort_values("dz_route_family_score", ascending=False).head(25).iloc[::-1]
    plt.figure(figsize=(14, max(6, 0.48 * len(top))))
    plt.barh(top["route_name"], top["dz_route_family_score"])
    plt.xlabel("DZ routed-envelope family score")
    plt.title("26DZ-LITE routed-envelope family scores")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_pass_rates(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    top = df.sort_values(["relaxed_pass_rate", "strict_pass_rate", "dz_route_family_score"], ascending=[False, False, False]).head(25).iloc[::-1]
    y = np.arange(len(top))
    plt.figure(figsize=(14, max(6, 0.48 * len(top))))
    plt.barh(y - 0.13, top["relaxed_pass_rate"], height=0.25, label="relaxed")
    plt.barh(y + 0.13, top["strict_pass_rate"], height=0.25, label="strict")
    plt.yticks(y, top["route_name"])
    plt.xlim(0, 1.05)
    plt.xlabel("pass rate across seeds")
    plt.title("26DZ-LITE routed-envelope pass rates")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_capture_tangent(df: pd.DataFrame, out_path: Path, target_cap: float, target_tan: float, relaxed_tan: float) -> None:
    if df.empty:
        return
    plt.figure(figsize=(10, 7))
    sc = plt.scatter(df["worst_capture_rate"], df["worst_tangent_ratio"], c=df["dz_route_score_run"], s=72, alpha=0.82)
    plt.axvline(target_cap, linewidth=1.0, alpha=0.45, label="capture target")
    plt.axhline(target_tan, linewidth=1.0, alpha=0.45, label="strict tangent")
    plt.axhline(relaxed_tan, linewidth=1.0, alpha=0.25, linestyle="--", label="relaxed tangent")
    plt.colorbar(sc, label="DZ route run score")
    plt.xlabel("routed-envelope worst capture_rate")
    plt.ylabel("routed-envelope worst tangent/radial ratio")
    plt.title("26DZ-LITE routed envelope: capture vs tangent")
    plt.grid(True, alpha=0.28)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_label_assignments(best_route: Dict[str, str], matrix: pd.DataFrame, out_path: Path) -> None:
    if matrix.empty or not best_route:
        return
    rows = []
    for lab in CRITICAL_LABELS:
        v = best_route.get(lab, best_route.get("*", ""))
        sub = matrix[(matrix["source_variant"] == v) & (matrix["envelope_label"] == lab)]
        if sub.empty:
            rows.append({"label": lab, "variant": v, "label_score": 0.0, "min_capture": 0.0, "max_tangent": 0.0})
        else:
            r = sub.iloc[0]
            rows.append({"label": lab, "variant": v, "label_score": fnum(r.get("label_score")), "min_capture": fnum(r.get("min_capture")), "max_tangent": fnum(r.get("max_tangent"))})
    df = pd.DataFrame(rows).sort_values("label_score", ascending=True)
    plt.figure(figsize=(13, max(5, 0.52 * len(df))))
    plt.barh(df["label"] + " ← " + df["variant"].str.slice(0, 42), df["label_score"])
    plt.xlabel("selected per-label score")
    plt.title("26DZ-LITE best route label assignments")
    for i, r in enumerate(df.itertuples(index=False)):
        plt.text(float(r.label_score), i, f" cap={r.min_capture:.3f} tan={r.max_tangent:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def run(args: argparse.Namespace) -> Dict[str, Any]:
    t0 = time.time()
    ROOT.mkdir(parents=True, exist_ok=True)

    dy = import_by_path(DY_FILE, "phase26dy_lite")
    ds_lite = import_by_path(DS_LITE_FILE, "phase26ds_lite")
    ds_full = import_by_path(DS_FULL_FILE, "phase26ds_full")
    drfast = import_by_path(DR_FAST_FILE, "phase26dr_fast")

    torch_info = drfast.setup_torch(args.device)
    dr = drfast.import_phase26dr()
    target_tan = float(getattr(dr, "TARGET_WORST_TAN", getattr(ds_full, "DEFAULT_TARGET_WORST_TAN", 2.10)))
    target_cap = float(getattr(dr, "TARGET_WORST_CAP", getattr(ds_full, "DEFAULT_TARGET_WORST_CAP", 0.285)))
    target_q20 = float(getattr(dr, "TARGET_Q20_DP", getattr(ds_full, "DEFAULT_TARGET_Q20_DP", 2.25)))
    relaxed_tan = float(args.relaxed_tan)

    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] device={args.device} torch={torch_info}")
    print(f"[{PHASE}] targets: strictTan<={target_tan:.2f}, relaxedTan<={relaxed_tan:.2f}, worstCap>={target_cap:.3f}, q20DP>={target_q20:.2f}")
    print(f"[{PHASE}] pool_size={args.pool_size}, route_count={args.route_count}, seed_count={args.seed_count}, audit_repeats={args.audit_repeats}, full_tail={args.sample_full_tail}")

    eval_helper = getattr(dr, "EVAL_HELPER", None)
    if eval_helper is None or not hasattr(eval_helper, "build_basins"):
        eval_helper = dr.previous_eval_helper()
    z2, rel_ids, basins = eval_helper.build_basins()
    print(f"[{PHASE}] Built basins: {len(basins)}")

    rng = random.Random(args.seed)
    pool = variant_pool(args.pool_size)
    hist = load_historical_label_scores(pool, target_cap, target_tan, relaxed_tan, target_q20)
    routes = build_routes(pool, hist, args.route_count, rng)

    print(f"[{PHASE}] variant pool:")
    for name, ov, src in pool:
        print(f"[{PHASE}]   {name} source={Path(src).name if src else src} ov={ov}")
    print(f"[{PHASE}] initial routes={len(routes)}")

    cache: Dict[Any, Any] = {}
    pool_seed_cases: Dict[int, Dict[str, pd.DataFrame]] = {}
    all_pool_case_rows: List[pd.DataFrame] = []
    route_rows: List[Dict[str, Any]] = []
    routed_case_rows: List[pd.DataFrame] = []

    seeds = [args.seed + 300000 + i * 7919 for i in range(args.seed_count)]

    # First evaluate all pool variants on all seeds once. Routing then becomes cheap.
    for seed in seeds:
        print(f"[{PHASE}] evaluating pool for seed={seed}")
        seed_cases = evaluate_pool_for_seed(ds_lite, drfast, dr, pool, basins, seed, args.audit_repeats, args.sample_full_tail, cache)
        pool_seed_cases[seed] = seed_cases
        for df in seed_cases.values():
            if not df.empty:
                all_pool_case_rows.append(df)

    # Add measured routes after seeing the actual first-seed label matrix.
    measured = measured_best_routes(pool_seed_cases, routes, target_cap, target_tan, relaxed_tan, target_q20, max_extra=args.measured_routes)
    for mr in measured:
        if len(routes) < args.route_count + args.measured_routes:
            routes.append(mr)
    print(f"[{PHASE}] total routes after measured additions={len(routes)}")

    for route_i, (route_name, route_map, route_src) in enumerate(routes, 1):
        for si, seed in enumerate(seeds, 1):
            metrics, cases = compose_route(route_name, route_map, pool_seed_cases[seed], target_cap, target_tan, relaxed_tan, target_q20)
            metrics["route_source"] = route_src
            metrics["seed_index"] = si
            metrics["dz_seed"] = seed
            route_rows.append(metrics)
            if not cases.empty:
                cases["route_source"] = route_src
                cases["seed_index"] = si
                routed_case_rows.append(cases)
            if route_i <= 4 or route_i % args.print_every == 0 or si == args.seed_count:
                print(
                    f"[{PHASE}] route {route_i:03d}/{len(routes):03d} {route_name[:58]:58s} s{si:02d} "
                    f"RELAX={int(metrics.get('pass_relaxed_line', False))} STRICT={int(metrics.get('pass_strict_line', False))} "
                    f"cap={fnum(metrics.get('worst_capture_rate')):.3f} tan={fnum(metrics.get('worst_tangent_ratio')):.3f} "
                    f"q20={fnum(metrics.get('q20_dp_score')):.3f} score={fnum(metrics.get('dz_route_score_run')):.3f}"
                )

    pool_cases_df = pd.concat(all_pool_case_rows, ignore_index=True) if all_pool_case_rows else pd.DataFrame()
    routed_cases_df = pd.concat(routed_case_rows, ignore_index=True) if routed_case_rows else pd.DataFrame()
    route_results_df = pd.DataFrame(route_rows)
    route_summary_df = summarize_routes(route_results_df, target_cap, target_tan, relaxed_tan, target_q20)
    matrix_df = summarize_variant_label_matrix(pool_cases_df, target_cap, target_tan, relaxed_tan, target_q20)

    route_results_df.to_csv(ROOT / "phase26dz_lite_route_results.csv", index=False)
    routed_cases_df.to_csv(ROOT / "phase26dz_lite_route_cases.csv", index=False)
    pool_cases_df.to_csv(ROOT / "phase26dz_lite_pool_case_results.csv", index=False)
    route_summary_df.to_csv(ROOT / "phase26dz_lite_route_summary.csv", index=False)
    matrix_df.to_csv(ROOT / "phase26dz_lite_variant_label_matrix.csv", index=False)
    hist.to_csv(ROOT / "phase26dz_lite_historical_label_scores.csv", index=False)

    plot_route_scores(route_summary_df, ROOT / "phase26dz_lite_route_scores.png")
    plot_pass_rates(route_summary_df, ROOT / "phase26dz_lite_pass_rates.png")
    plot_capture_tangent(route_results_df, ROOT / "phase26dz_lite_route_capture_vs_tangent.png", target_cap, target_tan, relaxed_tan)

    best = route_summary_df.iloc[0] if len(route_summary_df) else pd.Series(dtype=object)
    best_route_map = {}
    if len(best):
        try:
            best_route_map = json.loads(str(best.get("route_map_json", "{}")))
        except Exception:
            best_route_map = {}
    plot_label_assignments(best_route_map, matrix_df, ROOT / "phase26dz_lite_best_route_label_assignments.png")

    stable_strict = route_summary_df[route_summary_df["stable_strict_all_seeds"] == True] if not route_summary_df.empty else pd.DataFrame()
    stable_relaxed = route_summary_df[route_summary_df["stable_relaxed_all_seeds"] == True] if not route_summary_df.empty else pd.DataFrame()
    best_is_routed = bool(len(best) and not str(best.get("route_name", "")).startswith("dz_global_"))
    best_relaxed_rate = fnum(best.get("relaxed_pass_rate"), 0.0) if len(best) else 0.0

    if len(stable_strict):
        verdict = "strict_routed_envelope_pass_found"
    elif len(stable_relaxed):
        verdict = "relaxed_routed_envelope_pass_found"
    elif best_is_routed and best_relaxed_rate >= args.min_promote_rate:
        verdict = "partial_routed_advantage_found"
    elif best_is_routed and len(route_summary_df):
        # If the best route beats the best global route, routing is at least causally implicated.
        best_global = route_summary_df[route_summary_df["route_name"].astype(str).str.startswith("dz_global_")]
        if not best_global.empty and fnum(best.get("dz_route_family_score"), -999.0) > float(best_global["dz_route_family_score"].max()) + args.route_advantage_margin:
            verdict = "routed_family_beats_global_but_not_pass"
        else:
            verdict = "no_routed_advantage_found"
    else:
        verdict = "no_routed_advantage_found"

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
            "route_advantage_margin": args.route_advantage_margin,
        },
        "pool_size": int(len(pool)),
        "route_count": int(len(routes)),
        "seed_count": int(args.seed_count),
        "audit_repeats": int(args.audit_repeats),
        "sample_full_tail": bool(args.sample_full_tail),
        "num_pool_case_rows": int(len(pool_cases_df)),
        "num_routed_case_rows": int(len(routed_cases_df)),
        "num_route_runs": int(len(route_results_df)),
        "num_cached_eval_entries": int(len(cache)),
        "num_stable_strict_routes": int(len(stable_strict)),
        "num_stable_relaxed_routes": int(len(stable_relaxed)),
        "elapsed_sec": float(time.time() - t0),
        "verdict": verdict,
        "best_route": None if not len(best) else str(best.get("route_name")),
        "best_route_source": None if not len(best) else str(best.get("route_source", "")),
        "best_route_family_score": None if not len(best) else fnum(best.get("dz_route_family_score")),
        "best_strict_pass_rate": None if not len(best) else fnum(best.get("strict_pass_rate")),
        "best_relaxed_pass_rate": None if not len(best) else fnum(best.get("relaxed_pass_rate")),
        "best_min_worst_capture_rate": None if not len(best) else fnum(best.get("min_worst_capture_rate")),
        "best_max_worst_tangent_ratio": None if not len(best) else fnum(best.get("max_worst_tangent_ratio")),
        "best_min_q20_dp_score": None if not len(best) else fnum(best.get("min_q20_dp_score")),
        "best_route_map": best_route_map,
        "route_top10": route_summary_df.head(10).to_dict(orient="records") if len(route_summary_df) else [],
        "pool_variants": [{"name": n, "source": src, "overrides": ov} for n, ov, src in pool],
        "outputs": [
            "phase26dz_lite_route_results.csv",
            "phase26dz_lite_route_cases.csv",
            "phase26dz_lite_pool_case_results.csv",
            "phase26dz_lite_route_summary.csv",
            "phase26dz_lite_variant_label_matrix.csv",
            "phase26dz_lite_historical_label_scores.csv",
            "phase26dz_lite_summary.json",
            "phase26dz_lite_route_scores.png",
            "phase26dz_lite_pass_rates.png",
            "phase26dz_lite_route_capture_vs_tangent.png",
            "phase26dz_lite_best_route_label_assignments.png",
        ],
    }
    (ROOT / "phase26dz_lite_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[{PHASE}] Done in {summary['elapsed_sec']:.1f}s")
    print(f"[{PHASE}] VERDICT: {verdict}")
    if len(best):
        print(
            f"[{PHASE}] BEST ROUTE: {best['route_name']} | family={best['dz_route_family_score']:.3f} "
            f"strict={best['strict_pass_rate']:.2f} relaxed={best['relaxed_pass_rate']:.2f} "
            f"minCap={best['min_worst_capture_rate']:.3f} maxTan={best['max_worst_tangent_ratio']:.3f} "
            f"minQ20={best['min_q20_dp_score']:.3f}"
        )
        print(f"[{PHASE}] BEST route map: {best_route_map}")
    print(f"[{PHASE}] Wrote summary: {ROOT / 'phase26dz_lite_summary.json'}")
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=f"{PHASE}: {TITLE}")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="torch device preference")
    ap.add_argument("--seed", type=int, default=30000)
    ap.add_argument("--pool-size", type=int, default=8, help="number of global variants to evaluate once per seed")
    ap.add_argument("--route-count", type=int, default=12, help="number of routed compositions to test before measured-route additions")
    ap.add_argument("--measured-routes", type=int, default=3, help="extra first-seed measured routes to append")
    ap.add_argument("--seed-count", type=int, default=2, help="seeds per route")
    ap.add_argument("--audit-repeats", type=int, default=1, help="micro-envelope repeats per seed")
    ap.add_argument("--sample-full-tail", action="store_true", help="include a small sampled full-envelope tail; slower")
    ap.add_argument("--relaxed-tan", type=float, default=2.30, help="early relaxed tangent threshold")
    ap.add_argument("--min-promote-rate", type=float, default=0.67, help="promote threshold for partial routed stability")
    ap.add_argument("--route-advantage-margin", type=float, default=0.50, help="minimum family-score advantage over global control")
    ap.add_argument("--print-every", type=int, default=4)
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
