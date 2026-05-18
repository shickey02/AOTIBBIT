#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 26EG-LITE — Exhaustive Pareto feasibility router / blocker surgeon

Why this exists after 26EF:
  EF showed a strong route-score improvement, but still no strict/relaxed pass.
  The remaining failure is no longer just "find a better greedy route"; it is a
  feasibility question: does the current label pool contain a combination that
  can hit BOTH capture and tangent targets at the same time?

What EG does:
  1. Loads the existing pooled case results, preferably from 26EE/26EF outputs.
  2. Collapses rows to one metric row per (label, source_variant, seed).
  3. Builds per-label Pareto fronts over worst capture / worst tangent.
  4. Exhaustively searches the product of those label Pareto fronts when the
     product size is reasonable; otherwise it falls back to a bounded beam search.
  5. Produces hard feasibility diagnostics for each label:
       - best capture available under strict/relaxed tangent limits
       - best tangent available under the capture target
       - whether the label is a true pool blocker
  6. Writes route maps, CSVs, plots, and a concise JSON summary.

This is deliberately an analyzer/router, not another blind parameter jitter.
If EG says the pool ceiling is below target for screen labels, the next phase
should generate NEW screen-specific candidate physics rather than keep rerouting
old candidates.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

PHASE = "26EG-LITE"
DEFAULT_OUT = Path(r"E:\BBIT\outputs_basic32")
FALLBACK_OUT = Path("/mnt/data")

CAPTURE_TARGET = 0.285
STRICT_TANGENT_TARGET = 2.10
RELAXED_TANGENT_TARGET = 2.30

# Route scoring: capture floor matters most, then tangent, then mean quality.
def route_score(worst_capture: float, mean_capture: float, worst_tangent: float, mean_tangent: float) -> float:
    cap_def = max(0.0, CAPTURE_TARGET - worst_capture)
    tan_def = max(0.0, worst_tangent - STRICT_TANGENT_TARGET)
    relaxed_def = max(0.0, worst_tangent - RELAXED_TANGENT_TARGET)
    return (
        350.0 * worst_capture
        + 60.0 * mean_capture
        - 45.0 * tan_def
        - 35.0 * relaxed_def
        - 280.0 * cap_def
        - 6.0 * max(0.0, mean_tangent - STRICT_TANGENT_TARGET)
    )


def resolve_out_dir(cli_out: str | None) -> Path:
    if cli_out:
        p = Path(cli_out)
    elif DEFAULT_OUT.exists():
        p = DEFAULT_OUT
    else:
        p = FALLBACK_OUT
    p.mkdir(parents=True, exist_ok=True)
    return p


def find_first_existing(out_dir: Path, names: Iterable[str]) -> Path:
    roots = [out_dir, Path.cwd(), FALLBACK_OUT]
    for root in roots:
        for name in names:
            p = root / name
            if p.exists():
                return p
    raise FileNotFoundError("Could not find any of: " + ", ".join(names))


def load_pool(out_dir: Path, explicit: str | None = None) -> pd.DataFrame:
    if explicit:
        p = Path(explicit)
    else:
        p = find_first_existing(
            out_dir,
            [
                "phase26ee_lite_pool_case_results.csv",
                "phase26ef_lite_pool_case_results.csv",
                "phase26ec_lite_pool_case_results.csv",
                "phase26eb_lite_pool_case_results.csv",
            ],
        )
    df = pd.read_csv(p)
    required = {"envelope_label", "source_variant", "dz_seed", "capture_rate", "tangent_ratio"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Pool file {p} is missing columns: {missing}")
    print(f"[{PHASE}] loaded pool: {p.name} rows={len(df)} labels={df.envelope_label.nunique()} variants={df.source_variant.nunique()} seeds={df.dz_seed.nunique()}")
    return df


def collapse_seed_rows(pool: pd.DataFrame) -> pd.DataFrame:
    # Several upstream rows can share label/source/seed due to stage imports.
    # Keep the best dp_score when available, otherwise average numeric metrics.
    key = ["envelope_label", "source_variant", "dz_seed"]
    if "dp_score" in pool.columns:
        idx = pool.groupby(key)["dp_score"].idxmax()
        seed = pool.loc[idx].copy()
    else:
        seed = pool.groupby(key, as_index=False).agg(
            capture_rate=("capture_rate", "mean"),
            tangent_ratio=("tangent_ratio", "mean"),
        )
    seed["capture_rate"] = pd.to_numeric(seed["capture_rate"], errors="coerce")
    seed["tangent_ratio"] = pd.to_numeric(seed["tangent_ratio"], errors="coerce")
    seed = seed.dropna(subset=["capture_rate", "tangent_ratio"])
    return seed[key + ["capture_rate", "tangent_ratio"]]


def collapse_variants(seed_rows: pd.DataFrame) -> pd.DataFrame:
    g = seed_rows.groupby(["envelope_label", "source_variant"], as_index=False).agg(
        worst_capture=("capture_rate", "min"),
        mean_capture=("capture_rate", "mean"),
        worst_tangent=("tangent_ratio", "max"),
        mean_tangent=("tangent_ratio", "mean"),
        seeds=("dz_seed", "nunique"),
    )
    g["strict_pass_label"] = (g["worst_capture"] >= CAPTURE_TARGET) & (g["worst_tangent"] <= STRICT_TANGENT_TARGET)
    g["relaxed_pass_label"] = (g["worst_capture"] >= CAPTURE_TARGET) & (g["worst_tangent"] <= RELAXED_TANGENT_TARGET)
    g["capture_deficit"] = np.maximum(0.0, CAPTURE_TARGET - g["worst_capture"])
    g["strict_tangent_excess"] = np.maximum(0.0, g["worst_tangent"] - STRICT_TANGENT_TARGET)
    g["relaxed_tangent_excess"] = np.maximum(0.0, g["worst_tangent"] - RELAXED_TANGENT_TARGET)
    g["variant_score"] = [route_score(a, b, c, d) for a, b, c, d in zip(g.worst_capture, g.mean_capture, g.worst_tangent, g.mean_tangent)]
    return g.sort_values(["envelope_label", "variant_score"], ascending=[True, False]).reset_index(drop=True)


def is_dominated(row, others: pd.DataFrame) -> bool:
    # We maximize capture and minimize tangent. Mean terms are tiebreakers only.
    better_or_equal = (
        (others["worst_capture"] >= row["worst_capture"])
        & (others["worst_tangent"] <= row["worst_tangent"])
        & (others["mean_capture"] >= row["mean_capture"] - 1e-12)
    )
    strictly_better = (
        (others["worst_capture"] > row["worst_capture"] + 1e-12)
        | (others["worst_tangent"] < row["worst_tangent"] - 1e-12)
        | (others["mean_capture"] > row["mean_capture"] + 1e-12)
    )
    return bool((better_or_equal & strictly_better).any())


def pareto_fronts(collapsed: pd.DataFrame, max_per_label: int = 10) -> pd.DataFrame:
    fronts = []
    for lab, sub in collapsed.groupby("envelope_label"):
        keep = []
        for _, r in sub.iterrows():
            if not is_dominated(r, sub.drop(index=r.name)):
                keep.append(r)
        f = pd.DataFrame(keep)
        # Always include a few high-score fallbacks in case the front is tiny.
        f = pd.concat([f, sub.head(max_per_label)], ignore_index=True).drop_duplicates(["envelope_label", "source_variant"])
        f = f.sort_values("variant_score", ascending=False).head(max_per_label).copy()
        f["front_rank"] = range(len(f))
        fronts.append(f)
    return pd.concat(fronts, ignore_index=True)


def evaluate_route(route_name: str, route_map: Dict[str, str], seed_rows: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    pieces = []
    missing = []
    for lab, sv in route_map.items():
        sub = seed_rows[(seed_rows.envelope_label == lab) & (seed_rows.source_variant == sv)].copy()
        if sub.empty:
            missing.append(f"{lab}<-{sv}")
            continue
        pieces.append(sub)
    if not pieces:
        raise ValueError(f"route {route_name} has no valid label/source entries")
    route_df = pd.concat(pieces, ignore_index=True)
    rows = []
    for seed, sub in route_df.groupby("dz_seed"):
        wc = float(sub.capture_rate.min())
        wt = float(sub.tangent_ratio.max())
        mc = float(sub.capture_rate.mean())
        mt = float(sub.tangent_ratio.mean())
        rows.append({
            "route": route_name,
            "dz_seed": int(seed),
            "worst_capture": wc,
            "mean_capture": mc,
            "worst_tangent": wt,
            "mean_tangent": mt,
            "score": route_score(wc, mc, wt, mt),
            "pass_strict": bool((wc >= CAPTURE_TARGET) and (wt <= STRICT_TANGENT_TARGET)),
            "pass_relaxed": bool((wc >= CAPTURE_TARGET) and (wt <= RELAXED_TANGENT_TARGET)),
            "missing_labels": ";".join(missing) if missing else "",
        })
    per_seed = pd.DataFrame(rows)
    summary = {
        "route": route_name,
        "verified_worst_capture": float(per_seed.worst_capture.min()),
        "verified_mean_capture": float(per_seed.worst_capture.mean()),
        "verified_worst_tangent": float(per_seed.worst_tangent.max()),
        "verified_mean_tangent": float(per_seed.worst_tangent.mean()),
        "mean_score": float(per_seed.score.mean()),
        "min_score": float(per_seed.score.min()),
        "strict_pass_rate": float(per_seed.pass_strict.mean()),
        "relaxed_pass_rate": float(per_seed.pass_relaxed.mean()),
        "seeds": int(per_seed.dz_seed.nunique()),
        "eg_score": float(per_seed.score.mean() + 0.4 * per_seed.score.min()),
    }
    return per_seed, summary


def exhaustive_search(fronts: pd.DataFrame, seed_rows: pd.DataFrame, max_product: int = 750_000) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    labels = list(fronts.envelope_label.drop_duplicates())
    seeds = sorted(seed_rows.dz_seed.unique())
    seed_index = {s: i for i, s in enumerate(seeds)}

    # Precompute one compact metric vector for every label choice so route
    # evaluation is numpy min/max instead of repeated pandas filters.
    choices = []
    for lab in labels:
        lab_front = fronts[fronts.envelope_label == lab].copy()
        lab_choices = []
        for _, r in lab_front.iterrows():
            sv = r["source_variant"]
            sub = seed_rows[(seed_rows.envelope_label == lab) & (seed_rows.source_variant == sv)]
            cap = np.full(len(seeds), np.nan, dtype=float)
            tan = np.full(len(seeds), np.nan, dtype=float)
            for _, rr in sub.iterrows():
                j = seed_index[rr.dz_seed]
                cap[j] = float(rr.capture_rate)
                tan[j] = float(rr.tangent_ratio)
            if np.isnan(cap).any() or np.isnan(tan).any():
                continue
            lab_choices.append({"label": lab, "source_variant": sv, "cap": cap, "tan": tan, "variant_score": float(r.variant_score)})
        if not lab_choices:
            raise ValueError(f"No complete seed choices for label {lab}")
        choices.append(lab_choices)

    product_size = math.prod(len(c) for c in choices)
    print(f"[{PHASE}] Pareto product size={product_size:,} labels={len(labels)}")

    # If too large, keep the best few choices per label by local score.
    if product_size > max_product:
        trimmed = []
        for c in choices:
            c2 = sorted(c, key=lambda x: x["variant_score"], reverse=True)[:6]
            trimmed.append(c2)
        choices = trimmed
        product_size = math.prod(len(c) for c in choices)
        print(f"[{PHASE}] trimmed product size={product_size:,}")

    scored = []
    best_map: Dict[str, str] = {}
    best_per_seed = None
    best_score = -1e18

    for idx, combo in enumerate(itertools.product(*choices)):
        caps = np.vstack([c["cap"] for c in combo])
        tans = np.vstack([c["tan"] for c in combo])
        worst_capture_by_seed = caps.min(axis=0)
        worst_tangent_by_seed = tans.max(axis=0)
        mean_capture_by_seed = caps.mean(axis=0)
        mean_tangent_by_seed = tans.mean(axis=0)
        scores = np.array([route_score(wc, mc, wt, mt) for wc, mc, wt, mt in zip(worst_capture_by_seed, mean_capture_by_seed, worst_tangent_by_seed, mean_tangent_by_seed)])
        strict = (worst_capture_by_seed >= CAPTURE_TARGET) & (worst_tangent_by_seed <= STRICT_TANGENT_TARGET)
        relaxed = (worst_capture_by_seed >= CAPTURE_TARGET) & (worst_tangent_by_seed <= RELAXED_TANGENT_TARGET)
        route_name = f"eg_pareto_combo_{idx:06d}"
        summ = {
            "route": route_name,
            "verified_worst_capture": float(worst_capture_by_seed.min()),
            "verified_mean_capture": float(worst_capture_by_seed.mean()),
            "verified_worst_tangent": float(worst_tangent_by_seed.max()),
            "verified_mean_tangent": float(worst_tangent_by_seed.mean()),
            "mean_score": float(scores.mean()),
            "min_score": float(scores.min()),
            "strict_pass_rate": float(strict.mean()),
            "relaxed_pass_rate": float(relaxed.mean()),
            "seeds": int(len(seeds)),
            "eg_score": float(scores.mean() + 0.4 * scores.min()),
        }
        scored.append(summ)
        if summ["eg_score"] > best_score:
            best_score = summ["eg_score"]
            best_map = {c["label"]: c["source_variant"] for c in combo}
            best_per_seed = pd.DataFrame({
                "route": route_name,
                "dz_seed": seeds,
                "worst_capture": worst_capture_by_seed,
                "mean_capture": mean_capture_by_seed,
                "worst_tangent": worst_tangent_by_seed,
                "mean_tangent": mean_tangent_by_seed,
                "score": scores,
                "pass_strict": strict,
                "pass_relaxed": relaxed,
            })

    route_summaries = pd.DataFrame(scored).sort_values("eg_score", ascending=False).reset_index(drop=True)
    return route_summaries, best_per_seed if best_per_seed is not None else pd.DataFrame(), best_map

def label_feasibility(collapsed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lab, sub in collapsed.groupby("envelope_label"):
        strict_ok = sub[sub.worst_tangent <= STRICT_TANGENT_TARGET]
        relaxed_ok = sub[sub.worst_tangent <= RELAXED_TANGENT_TARGET]
        cap_ok = sub[sub.worst_capture >= CAPTURE_TARGET]
        best_any = sub.sort_values("variant_score", ascending=False).iloc[0]
        rows.append({
            "envelope_label": lab,
            "variants": int(len(sub)),
            "best_any_variant": best_any.source_variant,
            "best_any_capture": float(best_any.worst_capture),
            "best_any_tangent": float(best_any.worst_tangent),
            "best_capture_any": float(sub.worst_capture.max()),
            "best_tangent_any": float(sub.worst_tangent.min()),
            "best_capture_under_strict_tangent": float(strict_ok.worst_capture.max()) if len(strict_ok) else np.nan,
            "best_capture_under_relaxed_tangent": float(relaxed_ok.worst_capture.max()) if len(relaxed_ok) else np.nan,
            "best_tangent_at_capture_target": float(cap_ok.worst_tangent.min()) if len(cap_ok) else np.nan,
            "has_strict_label_pass": bool(((sub.worst_capture >= CAPTURE_TARGET) & (sub.worst_tangent <= STRICT_TANGENT_TARGET)).any()),
            "has_relaxed_label_pass": bool(((sub.worst_capture >= CAPTURE_TARGET) & (sub.worst_tangent <= RELAXED_TANGENT_TARGET)).any()),
        })
    out = pd.DataFrame(rows)
    out["pool_blocker"] = ~(out.has_relaxed_label_pass)
    return out.sort_values(["pool_blocker", "best_capture_under_relaxed_tangent"], ascending=[False, True])


def per_seed_oracle(seed_rows: pd.DataFrame) -> pd.DataFrame:
    # Theoretical ceiling if the route were allowed to choose source_variant per label per seed.
    rows = []
    for (seed, lab), sub in seed_rows.groupby(["dz_seed", "envelope_label"]):
        tmp = sub.copy()
        tmp["oracle_score"] = [route_score(a, a, b, b) for a, b in zip(tmp.capture_rate, tmp.tangent_ratio)]
        r = tmp.sort_values("oracle_score", ascending=False).iloc[0]
        rows.append({"dz_seed": seed, "envelope_label": lab, "source_variant": r.source_variant, "capture_rate": r.capture_rate, "tangent_ratio": r.tangent_ratio})
    oracle_rows = pd.DataFrame(rows)
    sums = []
    for seed, sub in oracle_rows.groupby("dz_seed"):
        wc = float(sub.capture_rate.min())
        wt = float(sub.tangent_ratio.max())
        sums.append({
            "route": "eg_seed_oracle_ceiling_not_deployable",
            "dz_seed": int(seed),
            "worst_capture": wc,
            "worst_tangent": wt,
            "pass_strict": bool(wc >= CAPTURE_TARGET and wt <= STRICT_TANGENT_TARGET),
            "pass_relaxed": bool(wc >= CAPTURE_TARGET and wt <= RELAXED_TANGENT_TARGET),
        })
    return pd.DataFrame(sums)


def plot_outputs(out_dir: Path, collapsed: pd.DataFrame, fronts: pd.DataFrame, route_summary: pd.DataFrame, feasibility: pd.DataFrame):
    if plt is None:
        return
    # Scatter all collapsed variants, front emphasized.
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.scatter(collapsed.worst_capture, collapsed.worst_tangent, s=28, alpha=0.25, label="pool collapsed")
    ax.scatter(fronts.worst_capture, fronts.worst_tangent, s=70, alpha=0.85, label="per-label Pareto/front")
    for _, r in feasibility.iterrows():
        ax.annotate(str(r.envelope_label), (r.best_any_capture, r.best_any_tangent), fontsize=8)
    ax.axvline(CAPTURE_TARGET, label="capture target")
    ax.axhline(STRICT_TANGENT_TARGET, label="strict tangent")
    ax.axhline(RELAXED_TANGENT_TARGET, linestyle="--", label="relaxed tangent")
    ax.set_xlabel("worst capture")
    ax.set_ylabel("worst tangent/radial")
    ax.set_title(f"{PHASE} pool feasibility: capture vs tangent")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "phase26eg_lite_pool_feasibility_scatter.png", dpi=150)
    plt.close(fig)

    top = route_summary.head(20).sort_values("eg_score")
    fig, ax = plt.subplots(figsize=(12, max(5, 0.34 * len(top))))
    ax.barh(top.route, top.eg_score)
    for i, (_, r) in enumerate(top.iterrows()):
        ax.text(r.eg_score, i, f" cap={r.verified_worst_capture:.3f} tan={r.verified_worst_tangent:.3f}", va="center", fontsize=8)
    ax.set_xlabel("EG route score")
    ax.set_title(f"{PHASE} best exhaustive/beam route scores")
    fig.tight_layout()
    fig.savefig(out_dir / "phase26eg_lite_route_scores.png", dpi=150)
    plt.close(fig)

    f = feasibility.sort_values("best_capture_under_relaxed_tangent")
    fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * len(f))))
    vals = f["best_capture_under_relaxed_tangent"].fillna(0.0)
    ax.barh(f.envelope_label, vals)
    ax.axvline(CAPTURE_TARGET)
    ax.set_xlabel("best capture available while tangent <= relaxed target")
    ax.set_title(f"{PHASE} label ceiling under relaxed tangent gate")
    fig.tight_layout()
    fig.savefig(out_dir / "phase26eg_lite_label_ceiling_under_relaxed_tangent.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--pool", default=None)
    ap.add_argument("--front-max", type=int, default=10)
    ap.add_argument("--max-product", type=int, default=750000)
    args = ap.parse_args()

    out_dir = resolve_out_dir(args.out_dir)
    print(f"[{PHASE}] Exhaustive Pareto feasibility router / blocker surgeon")

    pool = load_pool(out_dir, args.pool)
    seed_rows = collapse_seed_rows(pool)
    collapsed = collapse_variants(seed_rows)
    fronts = pareto_fronts(collapsed, max_per_label=args.front_max)
    feasibility = label_feasibility(collapsed)
    route_summary, best_seed_rows, best_map = exhaustive_search(fronts, seed_rows, max_product=args.max_product)
    oracle = per_seed_oracle(seed_rows)

    collapsed.to_csv(out_dir / "phase26eg_lite_label_variant_collapsed.csv", index=False)
    fronts.to_csv(out_dir / "phase26eg_lite_label_pareto_front.csv", index=False)
    feasibility.to_csv(out_dir / "phase26eg_lite_label_feasibility.csv", index=False)
    route_summary.to_csv(out_dir / "phase26eg_lite_route_summary.csv", index=False)
    best_seed_rows.to_csv(out_dir / "phase26eg_lite_best_route_results.csv", index=False)
    oracle.to_csv(out_dir / "phase26eg_lite_seed_oracle_ceiling.csv", index=False)
    with open(out_dir / "phase26eg_lite_route_map.json", "w", encoding="utf-8") as f:
        json.dump(best_map, f, indent=2, sort_keys=True)

    summary = {
        "phase": PHASE,
        "capture_target": CAPTURE_TARGET,
        "strict_tangent_target": STRICT_TANGENT_TARGET,
        "relaxed_tangent_target": RELAXED_TANGENT_TARGET,
        "pool_rows": int(len(pool)),
        "seed_rows": int(len(seed_rows)),
        "labels": int(collapsed.envelope_label.nunique()),
        "source_variants": int(collapsed.source_variant.nunique()),
        "best_route": route_summary.head(1).to_dict("records")[0] if len(route_summary) else {},
        "best_route_map": best_map,
        "relaxed_label_blockers": feasibility[feasibility.pool_blocker].envelope_label.tolist(),
        "seed_oracle_relaxed_pass_rate": float(oracle.pass_relaxed.mean()) if len(oracle) else 0.0,
        "seed_oracle_strict_pass_rate": float(oracle.pass_strict.mean()) if len(oracle) else 0.0,
    }
    with open(out_dir / "phase26eg_lite_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_outputs(out_dir, collapsed, fronts, route_summary, feasibility)

    print(f"[{PHASE}] best route:")
    cols = ["route", "verified_worst_capture", "verified_mean_capture", "verified_worst_tangent", "verified_mean_tangent", "strict_pass_rate", "relaxed_pass_rate", "eg_score"]
    print(route_summary.head(10)[cols].to_string(index=False))
    print(f"[{PHASE}] label feasibility / blockers:")
    show_cols = ["envelope_label", "best_capture_under_relaxed_tangent", "best_tangent_at_capture_target", "has_relaxed_label_pass", "pool_blocker"]
    print(feasibility[show_cols].to_string(index=False))
    print(f"[{PHASE}] seed oracle ceiling strict={summary['seed_oracle_strict_pass_rate']:.3f} relaxed={summary['seed_oracle_relaxed_pass_rate']:.3f}")
    print(f"[{PHASE}] wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
