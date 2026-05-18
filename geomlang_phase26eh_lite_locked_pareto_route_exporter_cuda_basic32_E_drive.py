#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 26EH-LITE — Locked Pareto route exporter / deployment confirmer

Why this exists after 26EG:
  26EG finally found a deployable route in the existing pool:
      worst_capture >= capture target
      worst_tangent <= strict tangent target
      strict_pass_rate == 1.0
      relaxed_pass_rate == 1.0

  That is the first phase where the problem stops being "discover a route" and
  becomes "freeze the route so the next real/geometric phase can use it without
  accidentally drifting back into exploratory mixtures."  EH is therefore a
  confirmer/exporter, not a new random search.

What EH does:
  1. Loads the 26EG route map, preferably phase26eg_lite_route_map.json.
  2. Loads the pooled verification rows, preferably phase26ee_lite_pool_case_results.csv.
  3. Re-evaluates the locked route across all seeds from the raw pool rows.
  4. Computes per-seed and per-label margins against the capture/tangent gates.
  5. Emits a compact deployable route package JSON that can be imported by the
     next phase.
  6. Writes plots showing route margins, per-label selected stress, and the
     seed-by-seed capture/tangent confirmation.

Interpretation:
  If EH reports strict_pass_rate=1.0, the label router is solved for this pool.
  The next phase should NOT keep jittering all labels blindly.  It should either:
    A) use the EH locked package as the stable route selector, or
    B) generate new physics only if you want a higher capture floor than 0.328.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

PHASE = "26EH-LITE"
DEFAULT_OUT = Path(r"E:\BBIT\outputs_basic32")
FALLBACK_OUT = Path("/mnt/data")

CAPTURE_TARGET = 0.285
STRICT_TANGENT_TARGET = 2.10
RELAXED_TANGENT_TARGET = 2.30


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


def load_route_map(out_dir: Path, explicit: str | None = None) -> Tuple[Path, Dict[str, str]]:
    if explicit:
        p = Path(explicit)
    else:
        p = find_first_existing(
            out_dir,
            [
                "phase26eg_lite_route_map.json",
                "phase26ef_lite_route_map.json",
                "phase26ee_lite_locked_route_map.json",
                "phase26ec_lite_locked_route_map.json",
            ],
        )
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Route map {p} is not a non-empty label->source_variant dict")
    return p, {str(k): str(v) for k, v in data.items()}


def load_pool(out_dir: Path, explicit: str | None = None) -> Tuple[Path, pd.DataFrame]:
    if explicit:
        p = Path(explicit)
    else:
        p = find_first_existing(
            out_dir,
            [
                "phase26ee_lite_pool_case_results.csv",
                "phase26ef_lite_pool_case_results.csv",
                "phase26eg_lite_pool_case_results.csv",
                "phase26ec_lite_pool_case_results.csv",
                "phase26eb_lite_pool_case_results.csv",
            ],
        )
    df = pd.read_csv(p)
    required = {"envelope_label", "source_variant", "dz_seed", "capture_rate", "tangent_ratio"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Pool file {p} is missing columns: {missing}")
    df["capture_rate"] = pd.to_numeric(df["capture_rate"], errors="coerce")
    df["tangent_ratio"] = pd.to_numeric(df["tangent_ratio"], errors="coerce")
    df = df.dropna(subset=["capture_rate", "tangent_ratio"])
    return p, df


def collapse_seed_rows(pool: pd.DataFrame) -> pd.DataFrame:
    key = ["envelope_label", "source_variant", "dz_seed"]
    if "dp_score" in pool.columns:
        # Upstream pools sometimes include duplicate label/source/seed rows.
        # Keep the row upstream scoring considered best for that exact case.
        idx = pool.groupby(key)["dp_score"].idxmax()
        seed = pool.loc[idx].copy()
    else:
        seed = pool.groupby(key, as_index=False).agg(
            capture_rate=("capture_rate", "mean"),
            tangent_ratio=("tangent_ratio", "mean"),
        )
    return seed[key + ["capture_rate", "tangent_ratio"]].reset_index(drop=True)


def route_fingerprint(route_map: Dict[str, str]) -> str:
    payload = json.dumps(route_map, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def evaluate_locked_route(route_map: Dict[str, str], seed_rows: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    pieces = []
    missing = []
    for lab, sv in route_map.items():
        sub = seed_rows[(seed_rows.envelope_label == lab) & (seed_rows.source_variant == sv)].copy()
        if sub.empty:
            missing.append({"envelope_label": lab, "source_variant": sv})
        else:
            pieces.append(sub)
    if missing:
        raise ValueError("Locked route references missing label/source rows: " + json.dumps(missing, indent=2))
    route_df = pd.concat(pieces, ignore_index=True)

    per_seed_rows = []
    for seed, sub in route_df.groupby("dz_seed"):
        wc = float(sub.capture_rate.min())
        wt = float(sub.tangent_ratio.max())
        mc = float(sub.capture_rate.mean())
        mt = float(sub.tangent_ratio.mean())
        blocker_cap = sub.loc[sub.capture_rate.idxmin(), "envelope_label"]
        blocker_tan = sub.loc[sub.tangent_ratio.idxmax(), "envelope_label"]
        per_seed_rows.append({
            "route": "eh_locked_pareto_route",
            "dz_seed": int(seed),
            "worst_capture": wc,
            "mean_capture": mc,
            "worst_tangent": wt,
            "mean_tangent": mt,
            "capture_margin": wc - CAPTURE_TARGET,
            "strict_tangent_margin": STRICT_TANGENT_TARGET - wt,
            "relaxed_tangent_margin": RELAXED_TANGENT_TARGET - wt,
            "capture_blocker_label": str(blocker_cap),
            "tangent_blocker_label": str(blocker_tan),
            "score": route_score(wc, mc, wt, mt),
            "pass_strict": bool((wc >= CAPTURE_TARGET) and (wt <= STRICT_TANGENT_TARGET)),
            "pass_relaxed": bool((wc >= CAPTURE_TARGET) and (wt <= RELAXED_TANGENT_TARGET)),
        })
    per_seed = pd.DataFrame(per_seed_rows).sort_values("dz_seed").reset_index(drop=True)

    label_rows = []
    for lab, sub in route_df.groupby("envelope_label"):
        sv = str(sub.source_variant.iloc[0])
        label_rows.append({
            "envelope_label": lab,
            "source_variant": sv,
            "worst_capture": float(sub.capture_rate.min()),
            "mean_capture": float(sub.capture_rate.mean()),
            "worst_tangent": float(sub.tangent_ratio.max()),
            "mean_tangent": float(sub.tangent_ratio.mean()),
            "capture_margin": float(sub.capture_rate.min() - CAPTURE_TARGET),
            "strict_tangent_margin": float(STRICT_TANGENT_TARGET - sub.tangent_ratio.max()),
            "relaxed_tangent_margin": float(RELAXED_TANGENT_TARGET - sub.tangent_ratio.max()),
            "seeds": int(sub.dz_seed.nunique()),
        })
    per_label = pd.DataFrame(label_rows).sort_values(["capture_margin", "strict_tangent_margin"], ascending=[True, True]).reset_index(drop=True)

    summary = {
        "phase": PHASE,
        "route": "eh_locked_pareto_route",
        "route_fingerprint": route_fingerprint(route_map),
        "capture_target": CAPTURE_TARGET,
        "strict_tangent_target": STRICT_TANGENT_TARGET,
        "relaxed_tangent_target": RELAXED_TANGENT_TARGET,
        "labels": int(len(route_map)),
        "seeds": int(per_seed.dz_seed.nunique()),
        "verified_worst_capture": float(per_seed.worst_capture.min()),
        "verified_mean_capture_floor": float(per_seed.worst_capture.mean()),
        "verified_worst_tangent": float(per_seed.worst_tangent.max()),
        "verified_mean_tangent_ceiling": float(per_seed.worst_tangent.mean()),
        "min_capture_margin": float(per_seed.capture_margin.min()),
        "min_strict_tangent_margin": float(per_seed.strict_tangent_margin.min()),
        "min_relaxed_tangent_margin": float(per_seed.relaxed_tangent_margin.min()),
        "strict_pass_rate": float(per_seed.pass_strict.mean()),
        "relaxed_pass_rate": float(per_seed.pass_relaxed.mean()),
        "mean_score": float(per_seed.score.mean()),
        "min_score": float(per_seed.score.min()),
        "eh_score": float(per_seed.score.mean() + 0.5 * per_seed.score.min() + 100.0 * per_seed.pass_strict.mean()),
        "route_map": route_map,
    }
    return per_seed, per_label, summary


def compare_with_eg(out_dir: Path, summary: dict) -> pd.DataFrame:
    try:
        p = find_first_existing(out_dir, ["phase26eg_lite_route_summary.csv"])
    except FileNotFoundError:
        return pd.DataFrame()
    eg = pd.read_csv(p)
    if eg.empty:
        return pd.DataFrame()
    cols = [c for c in ["route", "verified_worst_capture", "verified_worst_tangent", "strict_pass_rate", "relaxed_pass_rate", "eg_score"] if c in eg.columns]
    top = eg[cols].head(25).copy()
    locked = {
        "route": "eh_locked_pareto_route",
        "verified_worst_capture": summary["verified_worst_capture"],
        "verified_worst_tangent": summary["verified_worst_tangent"],
        "strict_pass_rate": summary["strict_pass_rate"],
        "relaxed_pass_rate": summary["relaxed_pass_rate"],
        "eg_score": np.nan,
    }
    return pd.concat([pd.DataFrame([locked]), top], ignore_index=True)


def plot_outputs(out_dir: Path, per_seed: pd.DataFrame, per_label: pd.DataFrame, comparison: pd.DataFrame):
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(per_seed.worst_capture, per_seed.worst_tangent, s=80)
    for _, r in per_seed.iterrows():
        ax.annotate(str(int(r.dz_seed)), (r.worst_capture, r.worst_tangent), fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.axvline(CAPTURE_TARGET, label="capture target")
    ax.axhline(STRICT_TANGENT_TARGET, label="strict tangent")
    ax.axhline(RELAXED_TANGENT_TARGET, linestyle="--", label="relaxed tangent")
    ax.set_xlabel("verified worst capture by seed")
    ax.set_ylabel("verified worst tangent/radial by seed")
    ax.set_title(f"{PHASE} locked route confirmation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "phase26eh_lite_locked_route_capture_vs_tangent.png", dpi=150)
    plt.close(fig)

    lp = per_label.sort_values("worst_tangent")
    fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * len(lp))))
    ax.barh(lp.envelope_label, lp.worst_tangent)
    ax.axvline(STRICT_TANGENT_TARGET, label="strict tangent")
    ax.axvline(RELAXED_TANGENT_TARGET, linestyle="--", label="relaxed tangent")
    for i, (_, r) in enumerate(lp.iterrows()):
        ax.text(r.worst_tangent, i, f" cap={r.worst_capture:.3f}", va="center", fontsize=8)
    ax.set_xlabel("selected label worst tangent/radial")
    ax.set_title(f"{PHASE} selected label tangent stress")
    fig.tight_layout()
    fig.savefig(out_dir / "phase26eh_lite_selected_label_tangent_stress.png", dpi=150)
    plt.close(fig)

    cm = per_label.sort_values("capture_margin")
    fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * len(cm))))
    ax.barh(cm.envelope_label, cm.capture_margin)
    ax.axvline(0.0)
    for i, (_, r) in enumerate(cm.iterrows()):
        ax.text(r.capture_margin, i, f" cap={r.worst_capture:.3f}", va="center", fontsize=8)
    ax.set_xlabel("capture margin above target")
    ax.set_title(f"{PHASE} selected label capture margins")
    fig.tight_layout()
    fig.savefig(out_dir / "phase26eh_lite_selected_label_capture_margins.png", dpi=150)
    plt.close(fig)

    if comparison is not None and not comparison.empty and "verified_worst_tangent" in comparison.columns:
        top = comparison.head(20).copy().iloc[::-1]
        fig, ax = plt.subplots(figsize=(12, max(5, 0.36 * len(top))))
        ax.barh(top.route, top.verified_worst_tangent)
        ax.axvline(STRICT_TANGENT_TARGET, label="strict tangent")
        ax.set_xlabel("verified worst tangent/radial")
        ax.set_title(f"{PHASE} locked route vs EG top routes")
        fig.tight_layout()
        fig.savefig(out_dir / "phase26eh_lite_locked_vs_eg_tangent.png", dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--pool", default=None)
    ap.add_argument("--route-map", default=None)
    args = ap.parse_args()

    out_dir = resolve_out_dir(args.out_dir)
    print(f"[{PHASE}] Locked Pareto route exporter / deployment confirmer")

    route_path, route_map = load_route_map(out_dir, args.route_map)
    pool_path, pool = load_pool(out_dir, args.pool)
    seed_rows = collapse_seed_rows(pool)

    per_seed, per_label, summary = evaluate_locked_route(route_map, seed_rows)
    comparison = compare_with_eg(out_dir, summary)

    per_seed.to_csv(out_dir / "phase26eh_lite_locked_route_seed_results.csv", index=False)
    per_label.to_csv(out_dir / "phase26eh_lite_locked_route_label_results.csv", index=False)
    if comparison is not None and not comparison.empty:
        comparison.to_csv(out_dir / "phase26eh_lite_locked_route_eg_comparison.csv", index=False)

    # Two JSONs: one full audit package, one small map for direct import.
    package = dict(summary)
    package["source_route_map_file"] = str(route_path)
    package["source_pool_file"] = str(pool_path)
    package["deploy_note"] = "Use route_map as the fixed label->source_variant selector for the next non-lite phase."
    with open(out_dir / "phase26eh_lite_locked_route_package.json", "w", encoding="utf-8") as f:
        json.dump(package, f, indent=2, sort_keys=True)
    with open(out_dir / "phase26eh_lite_deployable_route_map.json", "w", encoding="utf-8") as f:
        json.dump(route_map, f, indent=2, sort_keys=True)

    plot_outputs(out_dir, per_seed, per_label, comparison)

    print(f"[{PHASE}] loaded route map: {route_path.name} fingerprint={summary['route_fingerprint']}")
    print(f"[{PHASE}] loaded pool: {pool_path.name} rows={len(pool)} labels={pool.envelope_label.nunique()} variants={pool.source_variant.nunique()} seeds={pool.dz_seed.nunique()}")
    show = {
        "verified_worst_capture": summary["verified_worst_capture"],
        "verified_worst_tangent": summary["verified_worst_tangent"],
        "min_capture_margin": summary["min_capture_margin"],
        "min_strict_tangent_margin": summary["min_strict_tangent_margin"],
        "strict_pass_rate": summary["strict_pass_rate"],
        "relaxed_pass_rate": summary["relaxed_pass_rate"],
        "eh_score": summary["eh_score"],
    }
    print(f"[{PHASE}] locked route summary:")
    print(json.dumps(show, indent=2))
    print(f"[{PHASE}] selected labels:")
    print(per_label[["envelope_label", "source_variant", "worst_capture", "worst_tangent", "capture_margin", "strict_tangent_margin"]].to_string(index=False))
    print(f"[{PHASE}] wrote deployable package to: {out_dir / 'phase26eh_lite_locked_route_package.json'}")
    print(f"[{PHASE}] wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
