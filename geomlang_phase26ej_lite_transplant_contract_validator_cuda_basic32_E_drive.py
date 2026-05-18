#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Phase 26EJ-LITE — Transplant contract validator / route integration sentinel

Why this exists after 26EI:
  26EH locked the EG Pareto route. 26EI hardened and packaged it. EJ is the
  final guardrail before the route is transplanted into a real/non-lite phase.
  It does not search for a better route. It verifies that the deployable
  selector, hardened route package, and pool rows all describe the same solved
  object, then writes a compact contract that future phases can import.

What EJ answers:
  1) Does phase26ei_lite_locked_route_selector.py return exactly the packaged
     label -> source_variant map?
  2) Does the locked map still pass the strict gate on the seed rows?
  3) What is the exact failure budget if a future phase loses capture or inflates
     tangent?
  4) Which labels must be protected during transplant?
  5) What one-file contract should Codex / future phases use as the invariant?

Inputs it prefers:
  E:\BBIT\outputs_basic32\phase26ei_lite_hardened_route_package.json
  E:\BBIT\outputs_basic32\phase26ei_lite_locked_route_selector.py
  E:\BBIT\outputs_basic32\phase26ee_lite_pool_case_results.csv

Outputs:
  phase26ej_lite_summary.json
  phase26ej_lite_integration_contract.json
  phase26ej_lite_transplant_manifest.json
  phase26ej_lite_selector_selftest.py
  phase26ej_lite_locked_route_contract.md
  phase26ej_lite_locked_route_rows.csv
  phase26ej_lite_seed_results.csv
  phase26ej_lite_failure_budget.csv
  phase26ej_lite_label_protection_table.csv
  plus diagnostic PNGs.

Interpretation:
  If EJ prints CONTRACT_PASS=True, the solved label routing is no longer an
  experimental variable. The next phase should import the selector/contract and
  fail fast if any label is missing, renamed, remapped, or moved outside:
      worst_capture >= 0.285
      worst_tangent <= 2.10

  The current solved route is strong but not infinitely padded:
      capture weak labels: screen_medium_00, screen_small_00
      tangent weak label: base
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple, Any

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

PHASE = "26EJ-LITE"
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


def stable_fingerprint(obj: object) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_json_object(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def normalize_route_map(data: dict) -> Dict[str, str]:
    if "route_map" in data:
        rm = data["route_map"]
    elif "deployable_route_map" in data:
        rm = data["deployable_route_map"]
    else:
        rm = data
    if not isinstance(rm, dict) or not rm:
        raise ValueError("Could not find non-empty route_map in package")
    return {str(k): str(v) for k, v in rm.items()}


def load_hardened_package(out_dir: Path, explicit: str | None) -> Tuple[Path, dict, Dict[str, str]]:
    if explicit:
        p = Path(explicit)
    else:
        p = find_first_existing(out_dir, [
            "phase26ei_lite_hardened_route_package.json",
            "phase26eh_lite_locked_route_package.json",
            "phase26ei_lite_deployable_route_map.json",
        ])
    package = load_json_object(p)
    route_map = normalize_route_map(package)
    package["route_map"] = route_map
    return p, package, route_map


def load_selector(selector_path: Path) -> Dict[str, str]:
    spec = importlib.util.spec_from_file_location("phase26ei_lite_locked_route_selector", str(selector_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import selector from {selector_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    candidates = [
        "get_locked_route_map",
        "get_route_map",
        "locked_route_map",
        "route_map",
        "select_locked_route",
    ]
    for name in candidates:
        if hasattr(mod, name):
            obj = getattr(mod, name)
            val = obj() if callable(obj) else obj
            if isinstance(val, dict) and val:
                return {str(k): str(v) for k, v in val.items()}

    # Fallback: collect uppercase dict constants that look like label maps.
    for name in dir(mod):
        if name.isupper():
            val = getattr(mod, name)
            if isinstance(val, dict) and val:
                if all(isinstance(k, str) and isinstance(v, str) for k, v in val.items()):
                    return {str(k): str(v) for k, v in val.items()}
    raise AttributeError(
        f"Selector {selector_path} does not expose a usable route-map function or constant"
    )


def load_pool(out_dir: Path, explicit: str | None) -> Tuple[Path, pd.DataFrame]:
    if explicit:
        p = Path(explicit)
    else:
        p = find_first_existing(out_dir, [
            "phase26ee_lite_pool_case_results.csv",
            "phase26ef_lite_pool_case_results.csv",
            "phase26ec_lite_pool_case_results.csv",
        ])
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
        tmp = pool.copy()
        tmp["_dp"] = pd.to_numeric(tmp["dp_score"], errors="coerce").fillna(-1e18)
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
        raise ValueError("Route references missing seed rows:\n" + json.dumps(missing, indent=2))
    return pd.concat(parts, ignore_index=True)


def evaluate_route(route_rows: pd.DataFrame, cap_loss: float = 0.0, tan_inflate: float = 0.0) -> Tuple[pd.DataFrame, dict]:
    rr = route_rows.copy()
    rr["capture_stressed"] = rr["capture_rate"] - cap_loss
    rr["tangent_stressed"] = rr["tangent_ratio"] + tan_inflate
    rows = []
    for seed, sub in rr.groupby("dz_seed"):
        cap_idx = sub["capture_stressed"].idxmin()
        tan_idx = sub["tangent_stressed"].idxmax()
        wc = float(sub.loc[cap_idx, "capture_stressed"])
        wt = float(sub.loc[tan_idx, "tangent_stressed"])
        rows.append({
            "dz_seed": int(seed),
            "worst_capture": wc,
            "mean_capture": float(sub["capture_stressed"].mean()),
            "worst_tangent": wt,
            "mean_tangent": float(sub["tangent_stressed"].mean()),
            "capture_margin": wc - CAPTURE_TARGET,
            "strict_tangent_margin": STRICT_TANGENT_TARGET - wt,
            "relaxed_tangent_margin": RELAXED_TANGENT_TARGET - wt,
            "capture_blocker_label": str(sub.loc[cap_idx, "envelope_label"]),
            "tangent_blocker_label": str(sub.loc[tan_idx, "envelope_label"]),
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


def label_protection_table(route_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lab, sub in route_rows.groupby("envelope_label"):
        wc = float(sub.capture_rate.min())
        wt = float(sub.tangent_ratio.max())
        rows.append({
            "envelope_label": lab,
            "source_variant": str(sub.source_variant.iloc[0]),
            "worst_capture": wc,
            "worst_tangent": wt,
            "capture_margin": wc - CAPTURE_TARGET,
            "strict_tangent_margin": STRICT_TANGENT_TARGET - wt,
            "relaxed_tangent_margin": RELAXED_TANGENT_TARGET - wt,
            "protect_reason": "capture_floor" if (wc - CAPTURE_TARGET) <= (STRICT_TANGENT_TARGET - wt) else "tangent_ceiling",
            "weakness_score": min(wc - CAPTURE_TARGET, STRICT_TANGENT_TARGET - wt),
        })
    return pd.DataFrame(rows).sort_values("weakness_score").reset_index(drop=True)


def failure_budget_grid(route_rows: pd.DataFrame, max_cap_loss: float, max_tan_inflate: float, steps: int) -> pd.DataFrame:
    rows = []
    cap_losses = np.linspace(0.0, max_cap_loss, steps)
    tan_inflates = np.linspace(0.0, max_tan_inflate, steps)
    for cl in cap_losses:
        for ti in tan_inflates:
            _, sm = evaluate_route(route_rows, float(cl), float(ti))
            rows.append({
                "capture_loss": float(cl),
                "tangent_inflate": float(ti),
                "strict_pass_rate": sm["strict_pass_rate"],
                "relaxed_pass_rate": sm["relaxed_pass_rate"],
                "min_capture_margin": sm["min_capture_margin"],
                "min_strict_tangent_margin": sm["min_strict_tangent_margin"],
                "contract_pass": bool(sm["strict_pass_rate"] == 1.0),
            })
    return pd.DataFrame(rows)


def write_selector_selftest(out_dir: Path, contract_name: str) -> Path:
    p = out_dir / "phase26ej_lite_selector_selftest.py"
    p.write_text(f'''#!/usr/bin/env python3
# Auto-generated by 26EJ-LITE. Run this after moving the selector/contract.
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONTRACT = json.loads((ROOT / "{contract_name}").read_text(encoding="utf-8"))

try:
    from phase26ei_lite_locked_route_selector import get_locked_route_map
except Exception as exc:
    raise SystemExit(f"FAIL: could not import get_locked_route_map: {{exc}}")

selector_map = {{str(k): str(v) for k, v in get_locked_route_map().items()}}
contract_map = {{str(k): str(v) for k, v in CONTRACT["route_map"].items()}}

if selector_map != contract_map:
    missing = sorted(set(contract_map) - set(selector_map))
    extra = sorted(set(selector_map) - set(contract_map))
    changed = sorted(k for k in contract_map if k in selector_map and contract_map[k] != selector_map[k])
    raise SystemExit(
        "FAIL: selector map mismatch\n"
        f"missing={{missing}}\nextra={{extra}}\nchanged={{changed}}"
    )

s = CONTRACT["contract_summary"]
if not (s["verified_worst_capture"] >= CONTRACT["targets"]["capture_target"]):
    raise SystemExit("FAIL: contract capture below target")
if not (s["verified_worst_tangent"] <= CONTRACT["targets"]["strict_tangent_target"]):
    raise SystemExit("FAIL: contract tangent above strict target")

print("PASS: selector map and contract gates match")
print(json.dumps(s, indent=2))
''', encoding="utf-8")
    return p


def write_contract_md(out_dir: Path, contract: dict, protect: pd.DataFrame) -> Path:
    p = out_dir / "phase26ej_lite_locked_route_contract.md"
    s = contract["contract_summary"]
    lines = []
    lines.append("# Phase 26EJ-LITE Locked Route Contract\n")
    lines.append("This route is now treated as a solved selector, not as an open search variable.\n")
    lines.append("## Gates\n")
    lines.append(f"- Capture target: `{CAPTURE_TARGET}`\n")
    lines.append(f"- Strict tangent target: `{STRICT_TANGENT_TARGET}`\n")
    lines.append(f"- Relaxed tangent target: `{RELAXED_TANGENT_TARGET}`\n")
    lines.append("## Confirmed route summary\n")
    lines.append(f"- Verified worst capture: `{s['verified_worst_capture']}`\n")
    lines.append(f"- Verified worst tangent: `{s['verified_worst_tangent']}`\n")
    lines.append(f"- Strict pass rate: `{s['strict_pass_rate']}`\n")
    lines.append(f"- Relaxed pass rate: `{s['relaxed_pass_rate']}`\n")
    lines.append(f"- Min capture margin: `{s['min_capture_margin']}`\n")
    lines.append(f"- Min strict tangent margin: `{s['min_strict_tangent_margin']}`\n")
    lines.append("## Labels to protect first\n")
    for _, r in protect.iterrows():
        lines.append(
            f"- `{r.envelope_label}` -> `{r.source_variant}`; "
            f"cap_margin=`{r.capture_margin:.6f}`, "
            f"strict_tangent_margin=`{r.strict_tangent_margin:.6f}`, "
            f"reason=`{r.protect_reason}`\n"
        )
    p.write_text("".join(lines), encoding="utf-8")
    return p


def plot_outputs(out_dir: Path, seed_results: pd.DataFrame, protect: pd.DataFrame, grid: pd.DataFrame) -> None:
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(seed_results["worst_capture"], seed_results["worst_tangent"])
    for _, r in seed_results.iterrows():
        ax.annotate(str(int(r.dz_seed)), (r.worst_capture, r.worst_tangent), fontsize=8)
    ax.axvline(CAPTURE_TARGET, linestyle="-", label="capture target")
    ax.axhline(STRICT_TANGENT_TARGET, linestyle="-", label="strict tangent")
    ax.axhline(RELAXED_TANGENT_TARGET, linestyle="--", label="relaxed tangent")
    ax.set_title("26EJ-LITE selector contract seed confirmation")
    ax.set_xlabel("seed worst capture")
    ax.set_ylabel("seed worst tangent/radial")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "phase26ej_lite_contract_seed_confirmation.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    p2 = protect.sort_values("weakness_score", ascending=False)
    y = np.arange(len(p2))
    ax.barh(y - 0.18, p2["capture_margin"], height=0.35, label="capture margin")
    ax.barh(y + 0.18, p2["strict_tangent_margin"], height=0.35, label="strict tangent margin")
    ax.set_yticks(y)
    ax.set_yticklabels(p2["envelope_label"])
    ax.axvline(0.0)
    ax.set_title("26EJ-LITE protected label margins")
    ax.set_xlabel("positive margin is safe; negative margin fails")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "phase26ej_lite_protected_label_margins.png", dpi=150)
    plt.close(fig)

    piv = grid.pivot_table(index="tangent_inflate", columns="capture_loss", values="strict_pass_rate")
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(piv.values, origin="lower", aspect="auto", vmin=0, vmax=1)
    ax.set_title("26EJ-LITE failure budget: strict pass rate")
    ax.set_xlabel("capture loss")
    ax.set_ylabel("tangent inflation")
    ax.set_xticks(np.arange(len(piv.columns))[::max(1, len(piv.columns)//10)])
    ax.set_xticklabels([f"{x:.3f}" for x in piv.columns[::max(1, len(piv.columns)//10)]], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(piv.index))[::max(1, len(piv.index)//10)])
    ax.set_yticklabels([f"{x:.3f}" for x in piv.index[::max(1, len(piv.index)//10)]])
    fig.colorbar(im, ax=ax, label="strict pass rate")
    fig.tight_layout()
    fig.savefig(out_dir / "phase26ej_lite_failure_budget_grid.png", dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"{PHASE} transplant contract validator")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--package", default=None, help="EI hardened route package JSON")
    ap.add_argument("--selector", default=None, help="EI locked route selector .py")
    ap.add_argument("--pool", default=None, help="pool case results CSV")
    ap.add_argument("--max-capture-loss", type=float, default=0.08)
    ap.add_argument("--max-tangent-inflate", type=float, default=0.16)
    ap.add_argument("--grid-steps", type=int, default=17)
    args = ap.parse_args(argv)

    out_dir = resolve_out_dir(args.out_dir)
    print(f"[{PHASE}] Transplant contract validator / integration sentinel")

    package_path, package, package_map = load_hardened_package(out_dir, args.package)
    selector_path = Path(args.selector) if args.selector else find_first_existing(
        out_dir, ["phase26ei_lite_locked_route_selector.py"]
    )
    selector_map = load_selector(selector_path)
    pool_path, pool = load_pool(out_dir, args.pool)
    seed_rows = collapse_seed_rows(pool)

    maps_match = package_map == selector_map
    if not maps_match:
        print(f"[{PHASE}] ERROR: selector map does not match package map")
        print("package-only labels:", sorted(set(package_map) - set(selector_map)))
        print("selector-only labels:", sorted(set(selector_map) - set(package_map)))
        changed = sorted(k for k in package_map if k in selector_map and package_map[k] != selector_map[k])
        print("changed labels:", changed)
        return 2

    route_rows = extract_route_rows(package_map, seed_rows)
    seed_results, summary = evaluate_route(route_rows)
    protect = label_protection_table(route_rows)
    grid = failure_budget_grid(route_rows, args.max_capture_loss, args.max_tangent_inflate, args.grid_steps)

    strict_cells = int(grid.contract_pass.sum())
    total_cells = int(len(grid))
    max_cap_loss_strict = float(grid.loc[grid.contract_pass, "capture_loss"].max()) if strict_cells else 0.0
    max_tan_inflate_strict = float(grid.loc[grid.contract_pass, "tangent_inflate"].max()) if strict_cells else 0.0

    contract_pass = bool(
        maps_match
        and summary["verified_worst_capture"] >= CAPTURE_TARGET
        and summary["verified_worst_tangent"] <= STRICT_TANGENT_TARGET
        and summary["strict_pass_rate"] == 1.0
    )

    contract = {
        "phase": PHASE,
        "contract_pass": contract_pass,
        "route_fingerprint": stable_fingerprint(package_map),
        "source_package": str(package_path),
        "source_selector": str(selector_path),
        "source_pool": str(pool_path),
        "targets": {
            "capture_target": CAPTURE_TARGET,
            "strict_tangent_target": STRICT_TANGENT_TARGET,
            "relaxed_tangent_target": RELAXED_TANGENT_TARGET,
        },
        "contract_summary": summary,
        "failure_budget": {
            "max_tested_capture_loss_with_strict_pass": max_cap_loss_strict,
            "max_tested_tangent_inflate_with_strict_pass": max_tan_inflate_strict,
            "strict_pass_cells": strict_cells,
            "total_cells": total_cells,
        },
        "protected_labels_by_weakness": protect.to_dict(orient="records"),
        "route_map": package_map,
    }

    manifest = {
        "phase": PHASE,
        "next_phase_instruction": "Import the EI selector or EJ contract. Do not reopen label routing search unless this contract fails.",
        "must_copy_files": [
            "phase26ei_lite_locked_route_selector.py",
            "phase26ej_lite_integration_contract.json",
            "phase26ej_lite_selector_selftest.py",
        ],
        "hard_fail_conditions": [
            "missing envelope_label",
            "missing source_variant",
            "selector route_map fingerprint mismatch",
            "verified_worst_capture < 0.285",
            "verified_worst_tangent > 2.10",
            "strict_pass_rate < 1.0",
        ],
        "soft_warning_conditions": [
            "screen labels lose any capture margin",
            "base tangent rises by more than 0.04",
            "stress_seat_down or stress_shell_blend tangent rises near 2.10",
        ],
    }

    # Write files.
    route_rows.to_csv(out_dir / "phase26ej_lite_locked_route_rows.csv", index=False)
    seed_results.to_csv(out_dir / "phase26ej_lite_seed_results.csv", index=False)
    protect.to_csv(out_dir / "phase26ej_lite_label_protection_table.csv", index=False)
    grid.to_csv(out_dir / "phase26ej_lite_failure_budget.csv", index=False)
    (out_dir / "phase26ej_lite_integration_contract.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )
    (out_dir / "phase26ej_lite_transplant_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (out_dir / "phase26ej_lite_summary.json").write_text(
        json.dumps({
            "phase": PHASE,
            "contract_pass": contract_pass,
            "route_fingerprint": contract["route_fingerprint"],
            **summary,
            "max_tested_capture_loss_with_strict_pass": max_cap_loss_strict,
            "max_tested_tangent_inflate_with_strict_pass": max_tan_inflate_strict,
            "strict_pass_cells": strict_cells,
            "total_cells": total_cells,
        }, indent=2),
        encoding="utf-8",
    )
    selftest = write_selector_selftest(out_dir, "phase26ej_lite_integration_contract.json")
    md = write_contract_md(out_dir, contract, protect)
    plot_outputs(out_dir, seed_results, protect, grid)

    print(f"[{PHASE}] package: {package_path.name}")
    print(f"[{PHASE}] selector: {selector_path.name}")
    print(f"[{PHASE}] route fingerprint: {contract['route_fingerprint']}")
    print(f"[{PHASE}] CONTRACT_PASS={contract_pass}")
    print(f"[{PHASE}] contract summary:")
    print(json.dumps(contract["contract_summary"], indent=2))
    print(f"[{PHASE}] failure budget:")
    print(json.dumps(contract["failure_budget"], indent=2))
    print(f"[{PHASE}] weakest protected labels:")
    print(protect[["envelope_label", "source_variant", "capture_margin", "strict_tangent_margin", "protect_reason"]].to_string(index=False))
    print(f"[{PHASE}] wrote selftest: {selftest}")
    print(f"[{PHASE}] wrote contract doc: {md}")
    print(f"[{PHASE}] wrote outputs to: {out_dir}")
    return 0 if contract_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
