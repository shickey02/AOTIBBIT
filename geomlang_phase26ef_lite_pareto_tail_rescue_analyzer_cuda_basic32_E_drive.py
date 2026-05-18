#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 26EF-LITE — Pareto tail-rescue analyzer / route builder
==============================================================

What EF is for
--------------
EE showed the important thing: the old EC/EB/EE route can pass when locked,
but it is NOT robust. The worst remaining problem is not the whole route;
it is a small set of labels whose tangent tail explodes when we try to keep
capture high.

EF does not blindly launch another expensive search. It reads the already
measured pool-case CSVs from EE/EC/EB/DZ, collapses each candidate into a
seed-stable per-label score, builds Pareto fronts, then constructs several
new route maps using explicit tradeoff rules:

    1. tangent-first rescue
    2. capture-floor then tangent-min rescue
    3. balanced Pareto rescue
    4. strict-tail-only rescue
    5. relaxed-tail-only rescue
    6. labelwise oracle upper bound

The goal is to answer:
    "Is there an existing candidate combination that satisfies both capture
     and tangent, or are the remaining labels genuinely incompatible?"

Outputs
-------
    phase26ef_lite_label_variant_collapsed.csv
    phase26ef_lite_label_pareto_front.csv
    phase26ef_lite_route_map.json
    phase26ef_lite_route_results.csv
    phase26ef_lite_route_summary.csv
    phase26ef_lite_label_rescue_choices.csv
    phase26ef_lite_summary.json

Plots:
    phase26ef_lite_route_capture_vs_tangent.png
    phase26ef_lite_route_scores.png
    phase26ef_lite_label_front_sizes.png
    phase26ef_lite_label_best_tradeoffs.png

Run from E:/BBIT:
    py -3.12 bbit_geomlang/geomlang_phase26ef_lite_pareto_tail_rescue_analyzer_cuda_basic32_E_drive.py

Optional:
    py -3.12 bbit_geomlang/geomlang_phase26ef_lite_pareto_tail_rescue_analyzer_cuda_basic32_E_drive.py --capture-target 0.285 --strict-tangent 2.10 --relaxed-tangent 2.30
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHASE = "26EF-LITE"
TITLE = "Pareto tail-rescue analyzer / route builder"

LABELS_DEFAULT = [
    "base",
    "screen_small_00",
    "screen_medium_00",
    "stress_cap_tk",
    "stress_shell_blend",
    "stress_radius_in",
    "stress_seat_down",
]

SOURCE_PRIORITY = [
    "phase26ee_lite_pool_case_results.csv",
    "phase26ed_lite_pool_case_results.csv",
    "phase26ec_lite_pool_case_results.csv",
    "phase26eb_lite_pool_case_results.csv",
    "phase26ea_lite_pool_case_results.csv",
    "phase26dz_lite_pool_case_results.csv",
]


def root_dir() -> Path:
    # Designed for E:\BBIT, but also works when run from the script folder.
    cwd = Path.cwd()
    if (cwd / "bbit_geomlang").exists():
        return cwd
    if cwd.name.lower() == "bbit_geomlang":
        return cwd.parent
    # Container/local fallback.
    return Path("/mnt/data") if Path("/mnt/data").exists() else cwd


def out_path(name: str) -> Path:
    return output_dir() / name


def output_dir() -> Path:
    out = root_dir() / "outputs_basic32"
    out.mkdir(parents=True, exist_ok=True)
    return out


def find_first_existing(names: List[str]) -> Path:
    r = root_dir()
    candidates = []
    for name in names:
        candidates += [r / "outputs_basic32" / name, r / name, r / "bbit_geomlang" / name, Path(name)]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find any pool-case CSV. Looked for: "
        + ", ".join(str(p) for p in candidates)
    )


def load_pool() -> Tuple[pd.DataFrame, str]:
    p = find_first_existing(SOURCE_PRIORITY)
    df = pd.read_csv(p)
    required = {"envelope_label", "source_variant", "dz_seed", "capture_rate", "tangent_ratio", "dp_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{p.name} is missing required columns: {sorted(missing)}")
    df = df.copy()
    for c in ["capture_rate", "tangent_ratio", "dp_score"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["envelope_label", "source_variant", "dz_seed", "capture_rate", "tangent_ratio"])
    df["envelope_label"] = df["envelope_label"].astype(str)
    df["source_variant"] = df["source_variant"].astype(str)
    return df, p.name


def collapse_pool(pool: pd.DataFrame) -> pd.DataFrame:
    # Multiple cases can exist for the same seed/label/source_variant. Collapse them
    # pessimistically for capture/tangent, then collapse across seeds.
    seed = (
        pool.groupby(["envelope_label", "source_variant", "dz_seed"], as_index=False)
        .agg(
            seed_worst_capture=("capture_rate", "min"),
            seed_worst_tangent=("tangent_ratio", "max"),
            seed_mean_tangent=("tangent_ratio", "mean"),
            seed_mean_dp=("dp_score", "mean"),
            seed_min_dp=("dp_score", "min"),
            n_cases=("capture_rate", "size"),
        )
    )
    collapsed = (
        seed.groupby(["envelope_label", "source_variant"], as_index=False)
        .agg(
            worst_capture=("seed_worst_capture", "min"),
            mean_capture=("seed_worst_capture", "mean"),
            q20_capture=("seed_worst_capture", lambda s: float(np.quantile(s, 0.20))),
            worst_tangent=("seed_worst_tangent", "max"),
            mean_tangent=("seed_worst_tangent", "mean"),
            q80_tangent=("seed_worst_tangent", lambda s: float(np.quantile(s, 0.80))),
            mean_dp=("seed_mean_dp", "mean"),
            min_dp=("seed_min_dp", "min"),
            seed_count=("dz_seed", "nunique"),
            total_cases=("n_cases", "sum"),
        )
    )
    collapsed["capture_shortfall"] = np.maximum(0.0, 0.285 - collapsed["worst_capture"])
    collapsed["strict_tangent_excess"] = np.maximum(0.0, collapsed["worst_tangent"] - 2.10)
    collapsed["relaxed_tangent_excess"] = np.maximum(0.0, collapsed["worst_tangent"] - 2.30)
    collapsed["balanced_score"] = (
        90.0 * collapsed["worst_capture"]
        - 8.0 * np.maximum(0.0, collapsed["worst_tangent"] - 2.10)
        - 1.5 * collapsed["mean_tangent"]
        + 2.0 * collapsed["mean_dp"]
    )
    collapsed["tail_rescue_score"] = (
        120.0 * collapsed["worst_capture"]
        - 12.0 * collapsed["worst_tangent"]
        + 2.0 * collapsed["mean_dp"]
    )
    return collapsed.sort_values(["envelope_label", "balanced_score"], ascending=[True, False])


def pareto_front(g: pd.DataFrame) -> pd.DataFrame:
    # Maximize worst_capture, minimize worst_tangent. Keep non-dominated rows.
    rows = []
    gg = g.sort_values(["worst_capture", "worst_tangent"], ascending=[False, True]).reset_index(drop=True)
    best_tan = math.inf
    for _, r in gg.iterrows():
        t = float(r["worst_tangent"])
        if t < best_tan - 1e-12:
            rows.append(r)
            best_tan = t
    if not rows:
        return gg.head(0)
    return pd.DataFrame(rows)


def build_fronts(collapsed: pd.DataFrame) -> pd.DataFrame:
    fronts = []
    for label, g in collapsed.groupby("envelope_label"):
        f = pareto_front(g).copy()
        f["front_rank"] = np.arange(len(f))
        fronts.append(f)
    if not fronts:
        return collapsed.head(0)
    return pd.concat(fronts, ignore_index=True)


def choose_variant(g: pd.DataFrame, mode: str, cap_target: float, strict_tan: float, relaxed_tan: float) -> pd.Series:
    gg = g.copy()
    if mode == "tangent_first":
        return gg.sort_values(["worst_tangent", "worst_capture", "mean_dp"], ascending=[True, False, False]).iloc[0]
    if mode == "cap_floor_then_tangent":
        ok = gg[gg["worst_capture"] >= cap_target]
        if len(ok) == 0:
            ok = gg[gg["worst_capture"] >= max(0.0, cap_target - 0.02)]
        if len(ok) == 0:
            ok = gg
        return ok.sort_values(["worst_tangent", "worst_capture", "mean_dp"], ascending=[True, False, False]).iloc[0]
    if mode == "balanced_pareto":
        return gg.sort_values(["balanced_score", "worst_capture", "worst_tangent"], ascending=[False, False, True]).iloc[0]
    if mode == "strict_tail_only":
        ok = gg[gg["worst_tangent"] <= strict_tan]
        if len(ok) == 0:
            ok = gg.sort_values("worst_tangent").head(max(1, min(8, len(gg))))
        return ok.sort_values(["worst_capture", "mean_dp", "worst_tangent"], ascending=[False, False, True]).iloc[0]
    if mode == "relaxed_tail_only":
        ok = gg[gg["worst_tangent"] <= relaxed_tan]
        if len(ok) == 0:
            ok = gg.sort_values("worst_tangent").head(max(1, min(12, len(gg))))
        return ok.sort_values(["worst_capture", "mean_dp", "worst_tangent"], ascending=[False, False, True]).iloc[0]
    if mode == "oracle_capture":
        return gg.sort_values(["worst_capture", "worst_tangent", "mean_dp"], ascending=[False, True, False]).iloc[0]
    if mode == "oracle_score":
        return gg.sort_values(["tail_rescue_score", "worst_capture", "worst_tangent"], ascending=[False, False, True]).iloc[0]
    raise ValueError(mode)


def make_route_choices(collapsed: pd.DataFrame, cap_target: float, strict_tan: float, relaxed_tan: float) -> Tuple[Dict[str, Dict[str, str]], pd.DataFrame]:
    modes = [
        "tangent_first",
        "cap_floor_then_tangent",
        "balanced_pareto",
        "strict_tail_only",
        "relaxed_tail_only",
        "oracle_capture",
        "oracle_score",
    ]
    route_map: Dict[str, Dict[str, str]] = {}
    choice_rows = []
    labels = [x for x in LABELS_DEFAULT if x in set(collapsed["envelope_label"])]
    labels += sorted(set(collapsed["envelope_label"]) - set(labels))
    for mode in modes:
        route_name = f"ef_{mode}"
        route_map[route_name] = {}
        for label in labels:
            g = collapsed[collapsed["envelope_label"] == label]
            if len(g) == 0:
                continue
            r = choose_variant(g, mode, cap_target, strict_tan, relaxed_tan)
            route_map[route_name][label] = str(r["source_variant"])
            d = r.to_dict()
            d.update({"route": route_name, "selection_mode": mode})
            choice_rows.append(d)
    return route_map, pd.DataFrame(choice_rows)


def evaluate_routes(pool: pd.DataFrame, route_map: Dict[str, Dict[str, str]], cap_target: float, strict_tan: float, relaxed_tan: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    labels = sorted(pool["envelope_label"].unique())
    seeds = sorted(pool["dz_seed"].unique())
    # Collapse rows per seed/label/source first, pessimistically.
    seed_source = (
        pool.groupby(["dz_seed", "envelope_label", "source_variant"], as_index=False)
        .agg(
            capture=("capture_rate", "min"),
            tangent=("tangent_ratio", "max"),
            dp=("dp_score", "mean"),
        )
    )
    lookup = {
        (int(r.dz_seed), str(r.envelope_label), str(r.source_variant)): r
        for r in seed_source.itertuples(index=False)
    }
    for route, mapping in route_map.items():
        for seed in seeds:
            caps, tans, dps, missing = [], [], [], []
            for label in labels:
                src = mapping.get(label)
                if src is None:
                    missing.append(label)
                    continue
                rec = lookup.get((int(seed), label, src))
                if rec is None:
                    missing.append(label)
                    continue
                caps.append(float(rec.capture))
                tans.append(float(rec.tangent))
                dps.append(float(rec.dp))
            if caps:
                worst_cap = min(caps)
                worst_tan = max(tans)
                mean_dp = float(np.mean(dps)) if dps else np.nan
                score = 140.0 * worst_cap - 18.0 * max(0.0, worst_tan - strict_tan) - 5.0 * max(0.0, cap_target - worst_cap) + 3.0 * mean_dp
            else:
                worst_cap = np.nan
                worst_tan = np.nan
                mean_dp = np.nan
                score = -999.0
            rows.append({
                "route": route,
                "dz_seed": seed,
                "worst_capture": worst_cap,
                "worst_tangent": worst_tan,
                "mean_dp": mean_dp,
                "score": score,
                "pass_strict": bool(worst_cap >= cap_target and worst_tan <= strict_tan) if caps else False,
                "pass_relaxed": bool(worst_cap >= cap_target and worst_tan <= relaxed_tan) if caps else False,
                "missing_labels": ";".join(missing),
            })
    results = pd.DataFrame(rows)
    summary = (
        results.groupby("route", as_index=False)
        .agg(
            verified_worst_capture=("worst_capture", "min"),
            verified_mean_capture=("worst_capture", "mean"),
            verified_worst_tangent=("worst_tangent", "max"),
            verified_mean_tangent=("worst_tangent", "mean"),
            mean_score=("score", "mean"),
            min_score=("score", "min"),
            strict_pass_rate=("pass_strict", "mean"),
            relaxed_pass_rate=("pass_relaxed", "mean"),
            seeds=("dz_seed", "nunique"),
        )
    )
    summary["ef_score"] = (
        150.0 * summary["verified_worst_capture"]
        - 20.0 * np.maximum(0.0, summary["verified_worst_tangent"] - strict_tan)
        + 4.0 * summary["mean_score"]
        + 25.0 * summary["strict_pass_rate"]
        + 10.0 * summary["relaxed_pass_rate"]
    )
    summary = summary.sort_values("ef_score", ascending=False)
    return results, summary


def save_plots(collapsed: pd.DataFrame, fronts: pd.DataFrame, route_summary: pd.DataFrame, route_results: pd.DataFrame, cap_target: float, strict_tan: float, relaxed_tan: float) -> None:
    # Route capture vs tangent
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(route_summary["verified_worst_capture"], route_summary["verified_worst_tangent"], c=route_summary["ef_score"], s=120)
    ax.axvline(cap_target, label="capture target")
    ax.axhline(strict_tan, label="strict tangent")
    ax.axhline(relaxed_tan, linestyle="--", label="relaxed tangent")
    for _, r in route_summary.iterrows():
        ax.annotate(str(r["route"]).replace("ef_", ""), (r["verified_worst_capture"], r["verified_worst_tangent"]), fontsize=8, xytext=(4,4), textcoords="offset points")
    ax.set_title(f"{PHASE} route rescue: capture vs tangent")
    ax.set_xlabel("verified worst capture_rate")
    ax.set_ylabel("verified worst tangent/radial ratio")
    ax.legend()
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("EF score")
    fig.tight_layout()
    fig.savefig(out_path("phase26ef_lite_route_capture_vs_tangent.png"), dpi=160)
    plt.close(fig)

    # Route scores
    fig, ax = plt.subplots(figsize=(13, 7))
    rs = route_summary.sort_values("ef_score")
    ax.barh(rs["route"], rs["ef_score"])
    ax.set_title(f"{PHASE} route scores")
    ax.set_xlabel("EF route score")
    fig.tight_layout()
    fig.savefig(out_path("phase26ef_lite_route_scores.png"), dpi=160)
    plt.close(fig)

    # Pareto front sizes
    fs = fronts.groupby("envelope_label").size().sort_values()
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(fs.index, fs.values)
    ax.set_title(f"{PHASE} label Pareto-front sizes")
    ax.set_xlabel("non-dominated candidates per label")
    fig.tight_layout()
    fig.savefig(out_path("phase26ef_lite_label_front_sizes.png"), dpi=160)
    plt.close(fig)

    # Best tradeoffs by label
    best = collapsed.sort_values(["envelope_label", "balanced_score"], ascending=[True, False]).groupby("envelope_label", as_index=False).head(1)
    fig, ax = plt.subplots(figsize=(11, 7))
    sc = ax.scatter(best["worst_capture"], best["worst_tangent"], c=best["balanced_score"], s=120)
    ax.axvline(cap_target, label="capture target")
    ax.axhline(strict_tan, label="strict tangent")
    ax.axhline(relaxed_tan, linestyle="--", label="relaxed tangent")
    for _, r in best.iterrows():
        ax.annotate(str(r["envelope_label"]), (r["worst_capture"], r["worst_tangent"]), fontsize=8, xytext=(4,4), textcoords="offset points")
    ax.set_title(f"{PHASE} best single-label tradeoffs")
    ax.set_xlabel("label worst capture")
    ax.set_ylabel("label worst tangent")
    ax.legend()
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("label balanced score")
    fig.tight_layout()
    fig.savefig(out_path("phase26ef_lite_label_best_tradeoffs.png"), dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture-target", type=float, default=0.285)
    ap.add_argument("--strict-tangent", type=float, default=2.10)
    ap.add_argument("--relaxed-tangent", type=float, default=2.30)
    args = ap.parse_args()

    print(f"[{PHASE}] {TITLE}")
    pool, source_name = load_pool()
    print(f"[{PHASE}] loaded pool: {source_name} rows={len(pool)} labels={pool['envelope_label'].nunique()} variants={pool['source_variant'].nunique()} seeds={pool['dz_seed'].nunique()}")

    collapsed = collapse_pool(pool)
    fronts = build_fronts(collapsed)
    route_map, choices = make_route_choices(collapsed, args.capture_target, args.strict_tangent, args.relaxed_tangent)
    route_results, route_summary = evaluate_routes(pool, route_map, args.capture_target, args.strict_tangent, args.relaxed_tangent)

    collapsed.to_csv(out_path("phase26ef_lite_label_variant_collapsed.csv"), index=False)
    fronts.to_csv(out_path("phase26ef_lite_label_pareto_front.csv"), index=False)
    choices.to_csv(out_path("phase26ef_lite_label_rescue_choices.csv"), index=False)
    route_results.to_csv(out_path("phase26ef_lite_route_results.csv"), index=False)
    route_summary.to_csv(out_path("phase26ef_lite_route_summary.csv"), index=False)
    with open(out_path("phase26ef_lite_route_map.json"), "w", encoding="utf-8") as f:
        json.dump(route_map, f, indent=2)

    save_plots(collapsed, fronts, route_summary, route_results, args.capture_target, args.strict_tangent, args.relaxed_tangent)

    best_route = route_summary.iloc[0].to_dict() if len(route_summary) else {}
    # Label blockers: best available per label under balanced score and under tangent-first.
    label_best_bal = collapsed.sort_values(["envelope_label", "balanced_score"], ascending=[True, False]).groupby("envelope_label", as_index=False).head(1)
    blockers = label_best_bal.sort_values(["worst_tangent", "worst_capture"], ascending=[False, True]).head(7)

    summary: Dict[str, Any] = {
        "phase": PHASE,
        "title": TITLE,
        "source_pool": source_name,
        "rows_loaded": int(len(pool)),
        "labels": sorted(map(str, pool["envelope_label"].unique())),
        "capture_target": args.capture_target,
        "strict_tangent": args.strict_tangent,
        "relaxed_tangent": args.relaxed_tangent,
        "best_route": best_route,
        "route_count": int(len(route_summary)),
        "strict_routes_passing_all_seeds": route_summary.loc[route_summary["strict_pass_rate"] >= 1.0, "route"].tolist(),
        "relaxed_routes_passing_all_seeds": route_summary.loc[route_summary["relaxed_pass_rate"] >= 1.0, "route"].tolist(),
        "top_label_blockers_balanced": blockers[["envelope_label", "source_variant", "worst_capture", "worst_tangent", "balanced_score"]].to_dict(orient="records"),
        "outputs": [
            "phase26ef_lite_label_variant_collapsed.csv",
            "phase26ef_lite_label_pareto_front.csv",
            "phase26ef_lite_label_rescue_choices.csv",
            "phase26ef_lite_route_map.json",
            "phase26ef_lite_route_results.csv",
            "phase26ef_lite_route_summary.csv",
            "phase26ef_lite_summary.json",
            "phase26ef_lite_route_capture_vs_tangent.png",
            "phase26ef_lite_route_scores.png",
            "phase26ef_lite_label_front_sizes.png",
            "phase26ef_lite_label_best_tradeoffs.png",
        ],
    }
    with open(out_path("phase26ef_lite_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[{PHASE}] best route:")
    print(route_summary.head(10).to_string(index=False))
    print(f"[{PHASE}] likely label blockers:")
    print(blockers[["envelope_label", "source_variant", "worst_capture", "worst_tangent", "balanced_score"]].to_string(index=False))
    print(f"[{PHASE}] wrote outputs to: {output_dir()}")


if __name__ == "__main__":
    main()
