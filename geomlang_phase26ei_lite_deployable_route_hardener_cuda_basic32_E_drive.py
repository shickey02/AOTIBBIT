#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 26EI-LITE — Deployable locked-route hardener / transplant packager

Why this exists after 26EH:
  26EH confirmed that the EG Pareto route is no longer exploratory noise. It is a
  deployable label->source_variant selector:
      worst_capture >= 0.285
      worst_tangent <= 2.10
      strict_pass_rate == 1.0

  EI does not search for a new route. It hardens the locked route so it can be
  transplanted into the next real/non-lite phase without accidentally losing the
  solved label routing. It answers:
    1) What are the true weak margins of the locked route?
    2) How much capture loss / tangent inflation can it tolerate before failure?
    3) Which label blocks any future tightening?
    4) What exact minimal Python selector should the next phase import?

Inputs it prefers:
  E:\BBIT\outputs_basic32\phase26eh_lite_locked_route_package.json
  E:\BBIT\outputs_basic32\phase26ee_lite_pool_case_results.csv

Outputs:
  phase26ei_lite_summary.json
  phase26ei_lite_hardened_route_package.json
  phase26ei_lite_locked_route_selector.py
  phase26ei_lite_transplant_manifest.json
  phase26ei_lite_seed_label_matrix.csv
  phase26ei_lite_margin_table.csv
  phase26ei_lite_robustness_grid.csv
  phase26ei_lite_codex_handoff.txt
  plus diagnostic PNGs.

Interpretation:
  If the nominal route still passes and the robustness grid shows positive room,
  the next phase should import the EI selector and stop treating label routing as
  an open search variable. Future work should target only the weakest margins:
    - capture floor: usually screen_medium_00 / screen_small_00
    - tangent ceiling: usually base
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

PHASE = "26EI-LITE"
DEFAULT_OUT = Path(r"E:\BBIT\outputs_basic32")
FALLBACK_OUT = Path("/mnt/data")

CAPTURE_TARGET = 0.285
STRICT_TANGENT_TARGET = 2.10
RELAXED_TANGENT_TARGET = 2.30


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


def fingerprint(obj: object) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_locked_package(out_dir: Path, explicit: str | None = None) -> Tuple[Path, dict, Dict[str, str]]:
    if explicit:
        p = Path(explicit)
    else:
        p = find_first_existing(
            out_dir,
            [
                "phase26eh_lite_locked_route_package.json",
                "phase26eh_lite_deployable_route_map.json",
                "phase26eg_lite_route_map.json",
            ],
        )
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "route_map" in data:
        route_map = data["route_map"]
        package = data
    elif isinstance(data, dict):
        route_map = data
        package = {"route_map": data}
    else:
        raise ValueError(f"Locked package {p} is not a JSON object")
    if not isinstance(route_map, dict) or not route_map:
        raise ValueError(f"Locked package {p} does not contain a non-empty route_map")
    route_map = {str(k): str(v) for k, v in route_map.items()}
    package["route_map"] = route_map
    return p, package, route_map


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
            ],
        )
    df = pd.read_csv(p)
    required = {"envelope_label", "source_variant", "dz_seed", "capture_rate", "tangent_ratio"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Pool file {p} missing required columns: {missing}")
    df["capture_rate"] = pd.to_numeric(df["capture_rate"], errors="coerce")
    df["tangent_ratio"] = pd.to_numeric(df["tangent_ratio"], errors="coerce")
    df = df.dropna(subset=["capture_rate", "tangent_ratio"]).copy()
    df["dz_seed"] = pd.to_numeric(df["dz_seed"], errors="coerce").astype(int)
    return p, df


def collapse_seed_rows(pool: pd.DataFrame) -> pd.DataFrame:
    key = ["envelope_label", "source_variant", "dz_seed"]
    if "dp_score" in pool.columns:
        dp = pd.to_numeric(pool["dp_score"], errors="coerce")
        tmp = pool.copy()
        tmp["_dp"] = dp.fillna(-1e18)
        idx = tmp.groupby(key)["_dp"].idxmax()
        out = tmp.loc[idx, key + ["capture_rate", "tangent_ratio"]].copy()
    else:
        out = pool.groupby(key, as_index=False).agg(
            capture_rate=("capture_rate", "mean"),
            tangent_ratio=("tangent_ratio", "mean"),
        )
    return out.reset_index(drop=True)


def extract_route_rows(route_map: Dict[str, str], seed_rows: pd.DataFrame) -> pd.DataFrame:
    parts = []
    missing = []
    for lab, variant in route_map.items():
        sub = seed_rows[(seed_rows.envelope_label == lab) & (seed_rows.source_variant == variant)].copy()
        if sub.empty:
            missing.append({"envelope_label": lab, "source_variant": variant})
        else:
            parts.append(sub)
    if missing:
        raise ValueError("Locked route references missing rows:\n" + json.dumps(missing, indent=2))
    return pd.concat(parts, ignore_index=True)


def evaluate_route(route_rows: pd.DataFrame, cap_loss: float = 0.0, tan_inflate: float = 0.0) -> Tuple[pd.DataFrame, dict]:
    rows = []
    rr = route_rows.copy()
    rr["capture_stressed"] = rr["capture_rate"] - cap_loss
    rr["tangent_stressed"] = rr["tangent_ratio"] + tan_inflate
    for seed, sub in rr.groupby("dz_seed"):
        wc = float(sub.capture_stressed.min())
        wt = float(sub.tangent_stressed.max())
        rows.append({
            "dz_seed": int(seed),
            "worst_capture": wc,
            "mean_capture": float(sub.capture_stressed.mean()),
            "worst_tangent": wt,
            "mean_tangent": float(sub.tangent_stressed.mean()),
            "capture_margin": wc - CAPTURE_TARGET,
            "strict_tangent_margin": STRICT_TANGENT_TARGET - wt,
            "relaxed_tangent_margin": RELAXED_TANGENT_TARGET - wt,
            "capture_blocker_label": str(sub.loc[sub.capture_stressed.idxmin(), "envelope_label"]),
            "tangent_blocker_label": str(sub.loc[sub.tangent_stressed.idxmax(), "envelope_label"]),
            "pass_strict": bool((wc >= CAPTURE_TARGET) and (wt <= STRICT_TANGENT_TARGET)),
            "pass_relaxed": bool((wc >= CAPTURE_TARGET) and (wt <= RELAXED_TANGENT_TARGET)),
        })
    per_seed = pd.DataFrame(rows).sort_values("dz_seed").reset_index(drop=True)
    summary = {
        "verified_worst_capture": float(per_seed.worst_capture.min()),
        "verified_mean_capture_floor": float(per_seed.worst_capture.mean()),
        "verified_worst_tangent": float(per_seed.worst_tangent.max()),
        "verified_mean_tangent_ceiling": float(per_seed.worst_tangent.mean()),
        "min_capture_margin": float(per_seed.capture_margin.min()),
        "min_strict_tangent_margin": float(per_seed.strict_tangent_margin.min()),
        "min_relaxed_tangent_margin": float(per_seed.relaxed_tangent_margin.min()),
        "strict_pass_rate": float(per_seed.pass_strict.mean()),
        "relaxed_pass_rate": float(per_seed.pass_relaxed.mean()),
        "capture_blocker_mode": str(per_seed.capture_blocker_label.mode().iloc[0]),
        "tangent_blocker_mode": str(per_seed.tangent_blocker_label.mode().iloc[0]),
        "seeds": int(per_seed.dz_seed.nunique()),
    }
    return per_seed, summary


def build_margin_table(route_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lab, sub in route_rows.groupby("envelope_label"):
        wc = float(sub.capture_rate.min())
        wt = float(sub.tangent_ratio.max())
        rows.append({
            "envelope_label": lab,
            "source_variant": str(sub.source_variant.iloc[0]),
            "worst_capture": wc,
            "mean_capture": float(sub.capture_rate.mean()),
            "worst_tangent": wt,
            "mean_tangent": float(sub.tangent_ratio.mean()),
            "capture_margin": wc - CAPTURE_TARGET,
            "strict_tangent_margin": STRICT_TANGENT_TARGET - wt,
            "relaxed_tangent_margin": RELAXED_TANGENT_TARGET - wt,
            "weakness_score": min(wc - CAPTURE_TARGET, STRICT_TANGENT_TARGET - wt),
            "seeds": int(sub.dz_seed.nunique()),
        })
    return pd.DataFrame(rows).sort_values("weakness_score").reset_index(drop=True)


def build_seed_label_matrix(route_rows: pd.DataFrame) -> pd.DataFrame:
    wide_cap = route_rows.pivot_table(index="dz_seed", columns="envelope_label", values="capture_rate", aggfunc="min")
    wide_tan = route_rows.pivot_table(index="dz_seed", columns="envelope_label", values="tangent_ratio", aggfunc="max")
    out = []
    for seed in sorted(route_rows.dz_seed.unique()):
        for lab in sorted(route_rows.envelope_label.unique()):
            out.append({
                "dz_seed": int(seed),
                "envelope_label": lab,
                "capture_rate": float(wide_cap.loc[seed, lab]),
                "tangent_ratio": float(wide_tan.loc[seed, lab]),
                "capture_margin": float(wide_cap.loc[seed, lab] - CAPTURE_TARGET),
                "strict_tangent_margin": float(STRICT_TANGENT_TARGET - wide_tan.loc[seed, lab]),
            })
    return pd.DataFrame(out)


def robustness_grid(route_rows: pd.DataFrame, max_cap_loss: float, max_tan_inflate: float, steps: int) -> pd.DataFrame:
    cap_losses = np.linspace(0.0, max_cap_loss, steps)
    tan_inflates = np.linspace(0.0, max_tan_inflate, steps)
    rows = []
    for cl in cap_losses:
        for ti in tan_inflates:
            _, s = evaluate_route(route_rows, cap_loss=float(cl), tan_inflate=float(ti))
            rows.append({
                "capture_loss": float(cl),
                "tangent_inflate": float(ti),
                "verified_worst_capture": s["verified_worst_capture"],
                "verified_worst_tangent": s["verified_worst_tangent"],
                "min_capture_margin": s["min_capture_margin"],
                "min_strict_tangent_margin": s["min_strict_tangent_margin"],
                "strict_pass_rate": s["strict_pass_rate"],
                "relaxed_pass_rate": s["relaxed_pass_rate"],
                "passes_strict_all": bool(s["strict_pass_rate"] >= 1.0),
            })
    return pd.DataFrame(rows)


def write_selector(out_dir: Path, route_map: Dict[str, str], package_fp: str) -> Path:
    p = out_dir / "phase26ei_lite_locked_route_selector.py"
    text = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-generated by 26EI-LITE.
Import this in the next phase to freeze solved label routing.
Route fingerprint: {package_fp}
"""

LOCKED_ROUTE_FINGERPRINT = "{package_fp}"
CAPTURE_TARGET = {CAPTURE_TARGET!r}
STRICT_TANGENT_TARGET = {STRICT_TANGENT_TARGET!r}
RELAXED_TANGENT_TARGET = {RELAXED_TANGENT_TARGET!r}

LOCKED_ROUTE_MAP = {json.dumps(route_map, indent=4, sort_keys=True)}


def select_locked_source_variant(envelope_label: str, default: str | None = None) -> str:
    """Return the locked source_variant for an envelope label."""
    if envelope_label in LOCKED_ROUTE_MAP:
        return LOCKED_ROUTE_MAP[envelope_label]
    if default is not None:
        return default
    raise KeyError(f"No locked 26EI route variant for label: {{envelope_label}}")


def is_locked_label(envelope_label: str) -> bool:
    return envelope_label in LOCKED_ROUTE_MAP


def locked_labels() -> list[str]:
    return list(LOCKED_ROUTE_MAP.keys())
'''
    p.write_text(text, encoding="utf-8")
    return p


def write_codex_handoff(out_dir: Path, selector_path: Path, summary: dict, margin_table: pd.DataFrame) -> Path:
    weakest_cap = margin_table.sort_values("capture_margin").iloc[0]
    weakest_tan = margin_table.sort_values("strict_tangent_margin").iloc[0]
    p = out_dir / "phase26ei_lite_codex_handoff.txt"
    text = f"""26EI-LITE handoff for Codex / next phase

Current status:
- Locked route passes strict gates across all verification seeds.
- worst_capture={summary['verified_worst_capture']:.6f} target={CAPTURE_TARGET:.6f} margin={summary['min_capture_margin']:.6f}
- worst_tangent={summary['verified_worst_tangent']:.6f} target={STRICT_TANGENT_TARGET:.6f} margin={summary['min_strict_tangent_margin']:.6f}
- strict_pass_rate={summary['strict_pass_rate']:.3f}
- relaxed_pass_rate={summary['relaxed_pass_rate']:.3f}
- route_fingerprint={summary['route_fingerprint']}

Use this selector in the next phase:
{selector_path}

Integration rule:
- Do not re-search all label routing unless the locked route fails in a genuinely new physics test.
- Import LOCKED_ROUTE_MAP or select_locked_source_variant() and force each envelope_label to use its locked source_variant.
- New experiments should target margin improvement only, not random route churn.

Weakest capture label:
- {weakest_cap['envelope_label']} | cap={weakest_cap['worst_capture']:.6f} | margin={weakest_cap['capture_margin']:.6f} | variant={weakest_cap['source_variant']}

Weakest tangent label:
- {weakest_tan['envelope_label']} | tan={weakest_tan['worst_tangent']:.6f} | strict_margin={weakest_tan['strict_tangent_margin']:.6f} | variant={weakest_tan['source_variant']}

Suggested next goal:
- Preserve strict pass.
- Raise the capture floor from {summary['verified_worst_capture']:.6f} toward >=0.350, OR lower worst tangent from {summary['verified_worst_tangent']:.6f} toward <=1.90.
- Do this through targeted variants for the weak labels, especially the capture floor labels and the tangent blocker, rather than reopening solved labels.
"""
    p.write_text(text, encoding="utf-8")
    return p


def plot_outputs(out_dir: Path, per_seed: pd.DataFrame, margin: pd.DataFrame, grid: pd.DataFrame, matrix: pd.DataFrame):
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(per_seed.worst_capture, per_seed.worst_tangent, s=90)
    for _, r in per_seed.iterrows():
        ax.annotate(str(int(r.dz_seed)), (r.worst_capture, r.worst_tangent), fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.axvline(CAPTURE_TARGET, label="capture target")
    ax.axhline(STRICT_TANGENT_TARGET, label="strict tangent")
    ax.axhline(RELAXED_TANGENT_TARGET, linestyle="--", label="relaxed tangent")
    ax.set_xlabel("seed worst capture")
    ax.set_ylabel("seed worst tangent/radial")
    ax.set_title(f"{PHASE} locked-route seed confirmation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "phase26ei_lite_seed_confirmation.png", dpi=150)
    plt.close(fig)

    mt = margin.sort_values("weakness_score", ascending=True).iloc[::-1]
    fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * len(mt))))
    y = np.arange(len(mt))
    ax.barh(y - 0.18, mt.capture_margin, height=0.35, label="capture margin")
    ax.barh(y + 0.18, mt.strict_tangent_margin, height=0.35, label="strict tangent margin")
    ax.axvline(0.0)
    ax.set_yticks(y)
    ax.set_yticklabels(mt.envelope_label)
    ax.set_xlabel("positive margin is safe; negative margin fails")
    ax.set_title(f"{PHASE} per-label margin hardening")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "phase26ei_lite_label_margin_hardening.png", dpi=150)
    plt.close(fig)

    pivot = grid.pivot_table(index="tangent_inflate", columns="capture_loss", values="strict_pass_rate", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(pivot.values, origin="lower", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{x:.3f}" for x in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{y:.3f}" for y in pivot.index])
    ax.set_xlabel("artificial capture loss")
    ax.set_ylabel("artificial tangent inflation")
    ax.set_title(f"{PHASE} robustness grid: strict pass rate")
    fig.colorbar(im, ax=ax, label="strict pass rate")
    fig.tight_layout()
    fig.savefig(out_dir / "phase26ei_lite_robustness_grid.png", dpi=150)
    plt.close(fig)

    heat = matrix.pivot_table(index="envelope_label", columns="dz_seed", values="strict_tangent_margin", aggfunc="min")
    fig, ax = plt.subplots(figsize=(10, max(5, 0.4 * len(heat))))
    im = ax.imshow(heat.values, aspect="auto")
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels([str(x) for x in heat.columns])
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_xlabel("dz_seed")
    ax.set_title(f"{PHASE} strict tangent margin by label/seed")
    fig.colorbar(im, ax=ax, label="strict tangent margin")
    fig.tight_layout()
    fig.savefig(out_dir / "phase26ei_lite_seed_label_tangent_margin_heatmap.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--package", default=None, help="Path to phase26eh_lite_locked_route_package.json")
    ap.add_argument("--pool", default=None, help="Path to pooled case results CSV")
    ap.add_argument("--max-cap-loss", type=float, default=0.075)
    ap.add_argument("--max-tan-inflate", type=float, default=0.150)
    ap.add_argument("--grid-steps", type=int, default=16)
    ap.add_argument("--device", default="auto", help="Accepted for workflow compatibility; EI is pandas/NumPy-bound.")
    args = ap.parse_args()

    out_dir = resolve_out_dir(args.out_dir)
    print(f"[{PHASE}] Deployable locked-route hardener / transplant packager")

    package_path, upstream_package, route_map = load_locked_package(out_dir, args.package)
    pool_path, pool = load_pool(out_dir, args.pool)
    seed_rows = collapse_seed_rows(pool)
    route_rows = extract_route_rows(route_map, seed_rows)

    per_seed, nominal = evaluate_route(route_rows)
    margin = build_margin_table(route_rows)
    matrix = build_seed_label_matrix(route_rows)
    grid = robustness_grid(route_rows, args.max_cap_loss, args.max_tan_inflate, max(3, args.grid_steps))

    route_fp = fingerprint(route_map)
    nominal.update({
        "phase": PHASE,
        "route": "ei_hardened_locked_pareto_route",
        "route_fingerprint": route_fp,
        "source_package_file": str(package_path),
        "source_pool_file": str(pool_path),
        "labels": int(len(route_map)),
        "capture_target": CAPTURE_TARGET,
        "strict_tangent_target": STRICT_TANGENT_TARGET,
        "relaxed_tangent_target": RELAXED_TANGENT_TARGET,
        "route_map": route_map,
        "upstream_eh_summary": {k: upstream_package.get(k) for k in [
            "verified_worst_capture", "verified_worst_tangent", "min_capture_margin",
            "min_strict_tangent_margin", "strict_pass_rate", "relaxed_pass_rate", "route_fingerprint"
        ] if k in upstream_package},
    })

    # Robustness ceilings: largest single-axis perturbations that still keep all seeds strict-pass.
    cap_axis = grid[(grid.tangent_inflate == grid.tangent_inflate.min()) & (grid.passes_strict_all)]
    tan_axis = grid[(grid.capture_loss == grid.capture_loss.min()) & (grid.passes_strict_all)]
    nominal["max_tested_capture_loss_with_strict_pass"] = float(cap_axis.capture_loss.max()) if not cap_axis.empty else 0.0
    nominal["max_tested_tangent_inflate_with_strict_pass"] = float(tan_axis.tangent_inflate.max()) if not tan_axis.empty else 0.0
    both = grid[grid.passes_strict_all]
    nominal["robust_grid_strict_pass_cells"] = int(len(both))
    nominal["robust_grid_total_cells"] = int(len(grid))

    selector_path = write_selector(out_dir, route_map, route_fp)
    handoff_path = write_codex_handoff(out_dir, selector_path, nominal, margin)

    hardened = dict(nominal)
    hardened["selector_file"] = str(selector_path)
    hardened["codex_handoff_file"] = str(handoff_path)
    hardened["margin_table_file"] = str(out_dir / "phase26ei_lite_margin_table.csv")
    hardened["deployment_note"] = "Import phase26ei_lite_locked_route_selector.py in the next phase and keep route_map fixed unless a new non-lite verifier fails it."

    manifest = {
        "phase": PHASE,
        "route_fingerprint": route_fp,
        "selector_file": str(selector_path),
        "package_file": str(out_dir / "phase26ei_lite_hardened_route_package.json"),
        "route_map_file": str(out_dir / "phase26ei_lite_deployable_route_map.json"),
        "pool_file": str(pool_path),
        "strict_pass_rate": nominal["strict_pass_rate"],
        "verified_worst_capture": nominal["verified_worst_capture"],
        "verified_worst_tangent": nominal["verified_worst_tangent"],
        "weakest_capture_label": str(margin.sort_values("capture_margin").iloc[0].envelope_label),
        "weakest_tangent_label": str(margin.sort_values("strict_tangent_margin").iloc[0].envelope_label),
    }

    per_seed.to_csv(out_dir / "phase26ei_lite_seed_results.csv", index=False)
    route_rows.to_csv(out_dir / "phase26ei_lite_locked_route_rows.csv", index=False)
    margin.to_csv(out_dir / "phase26ei_lite_margin_table.csv", index=False)
    matrix.to_csv(out_dir / "phase26ei_lite_seed_label_matrix.csv", index=False)
    grid.to_csv(out_dir / "phase26ei_lite_robustness_grid.csv", index=False)
    with open(out_dir / "phase26ei_lite_summary.json", "w", encoding="utf-8") as f:
        json.dump(nominal, f, indent=2, sort_keys=True)
    with open(out_dir / "phase26ei_lite_hardened_route_package.json", "w", encoding="utf-8") as f:
        json.dump(hardened, f, indent=2, sort_keys=True)
    with open(out_dir / "phase26ei_lite_transplant_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    with open(out_dir / "phase26ei_lite_deployable_route_map.json", "w", encoding="utf-8") as f:
        json.dump(route_map, f, indent=2, sort_keys=True)

    plot_outputs(out_dir, per_seed, margin, grid, matrix)

    print(f"[{PHASE}] loaded EH package: {package_path.name}")
    print(f"[{PHASE}] loaded pool: {pool_path.name} rows={len(pool)} labels={pool.envelope_label.nunique()} variants={pool.source_variant.nunique()} seeds={pool.dz_seed.nunique()}")
    print(f"[{PHASE}] route fingerprint: {route_fp}")
    show = {
        "verified_worst_capture": nominal["verified_worst_capture"],
        "verified_worst_tangent": nominal["verified_worst_tangent"],
        "min_capture_margin": nominal["min_capture_margin"],
        "min_strict_tangent_margin": nominal["min_strict_tangent_margin"],
        "strict_pass_rate": nominal["strict_pass_rate"],
        "relaxed_pass_rate": nominal["relaxed_pass_rate"],
        "max_tested_capture_loss_with_strict_pass": nominal["max_tested_capture_loss_with_strict_pass"],
        "max_tested_tangent_inflate_with_strict_pass": nominal["max_tested_tangent_inflate_with_strict_pass"],
        "robust_grid_strict_pass_cells": nominal["robust_grid_strict_pass_cells"],
        "robust_grid_total_cells": nominal["robust_grid_total_cells"],
    }
    print(f"[{PHASE}] hardened route summary:")
    print(json.dumps(show, indent=2))
    print(f"[{PHASE}] weakest margins:")
    print(margin[["envelope_label", "source_variant", "worst_capture", "worst_tangent", "capture_margin", "strict_tangent_margin", "weakness_score"]].to_string(index=False))
    print(f"[{PHASE}] wrote selector: {selector_path}")
    print(f"[{PHASE}] wrote Codex handoff: {handoff_path}")
    print(f"[{PHASE}] wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
