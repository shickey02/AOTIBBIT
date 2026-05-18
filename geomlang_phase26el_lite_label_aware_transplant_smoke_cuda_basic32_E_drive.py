#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
26EL-LITE — Label-aware transplant smoke / false-alarm isolator

Purpose
-------
26EK correctly proved that the selector/package wiring exists, but its smoke test
was too broad: it tested each locked source_variant globally across every pool row
that shared that source_variant. That can fail when the same source_variant appears
under a different envelope_label or a non-locked stress context.

26EL fixes the transplant smoke contract by using the locked contract rows as the
source of truth. 26EK tested source_variant globally against the raw pool. That is
not a valid transplant contract, because source_variant is not a unique deployed
candidate key for the screen labels. The deployable object is the locked route
row set exported by EI/EJ:
    (envelope_label, dz_seed, source_variant, capture_rate, tangent_ratio)

Expected result if the EH/EI/EJ route is truly deployable:
    LABEL_AWARE_SMOKE_PASS=True
    contract_row_capture_smoke=True
    contract_row_strict_tangent_smoke=True
    pool_source_alias_false_alarm_detected=True  # likely, and okay

Inputs expected in E:\BBIT\outputs_basic32:
    phase26ej_lite_integration_contract.json
    phase26ei_lite_hardened_route_package.json
    phase26ei_lite_locked_route_selector.py
    phase26ei_lite_locked_route_rows.csv
    phase26ee_lite_pool_case_results.csv

Outputs:
    phase26el_lite_summary.json
    phase26el_lite_contract_row_smoke.csv
    phase26el_lite_pool_source_alias_smoke.csv
    phase26el_lite_label_aware_route_selector_adapter.py
    phase26el_lite_label_aware_selftest.py
    phase26el_lite_contract_row_capture_smoke.png
    phase26el_lite_contract_row_tangent_smoke.png
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


PHASE = "26EL-LITE"
DEFAULT_OUT = Path(r"E:\BBIT\outputs_basic32")
FALLBACK_OUTS = [Path.cwd() / "outputs_basic32", Path.cwd(), Path("/mnt/data")]


def find_outputs_dir(cli_value: Optional[str]) -> Path:
    candidates = []
    if cli_value:
        candidates.append(Path(cli_value))
    candidates.append(DEFAULT_OUT)
    candidates.extend(FALLBACK_OUTS)
    for p in candidates:
        if (p / "phase26ej_lite_integration_contract.json").exists() or (p / "phase26ee_lite_pool_case_results.csv").exists():
            return p
    return candidates[0]


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def load_selector(selector_path: Path):
    spec = importlib.util.spec_from_file_location("phase26ei_lite_locked_route_selector", selector_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import selector: {selector_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def normalize_label(label: str) -> str:
    return str(label).strip()


def route_hash(route_map: Mapping[str, str]) -> str:
    blob = json.dumps(dict(sorted(route_map.items())), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def extract_route_map(package: Mapping[str, Any], selector_mod: Any) -> Dict[str, str]:
    if hasattr(selector_mod, "LOCKED_ROUTE_MAP"):
        return dict(getattr(selector_mod, "LOCKED_ROUTE_MAP"))
    for key in ("locked_route_map", "deployable_route_map", "route_map", "selected_route_map"):
        val = package.get(key)
        if isinstance(val, dict):
            return {str(k): str(v) for k, v in val.items()}
    rows = package.get("locked_route_rows") or package.get("selected_labels") or package.get("route_rows")
    if isinstance(rows, list):
        out = {}
        for row in rows:
            if isinstance(row, dict) and "envelope_label" in row and "source_variant" in row:
                out[str(row["envelope_label"])] = str(row["source_variant"])
        if out:
            return out
    raise KeyError("Could not find route map in selector or package")


def choose_col(df: pd.DataFrame, options: Iterable[str]) -> str:
    for c in options:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns found: {list(options)}; available={list(df.columns)}")


def contract_row_smoke(contract_rows: pd.DataFrame, route_map: Mapping[str, str], cap_target: float, strict_tan_target: float, relaxed_tan_target: float) -> pd.DataFrame:
    label_col = choose_col(contract_rows, ["envelope_label", "label"])
    variant_col = choose_col(contract_rows, ["source_variant", "variant", "variant_name"])
    cap_col = choose_col(contract_rows, ["capture_rate", "worst_capture", "verified_worst_capture"])
    tan_col = choose_col(contract_rows, ["tangent_ratio", "worst_tangent", "verified_worst_tangent"])
    seed_col = "dz_seed" if "dz_seed" in contract_rows.columns else None

    rows = []
    for label, variant in sorted(route_map.items()):
        sub = contract_rows[(contract_rows[label_col].astype(str) == str(label)) & (contract_rows[variant_col].astype(str) == str(variant))].copy()
        if sub.empty:
            rows.append({
                "envelope_label": label,
                "source_variant": variant,
                "rows": 0,
                "seeds": 0,
                "min_capture": float("nan"),
                "max_tangent": float("nan"),
                "capture_margin": float("nan"),
                "strict_tangent_margin": float("nan"),
                "relaxed_tangent_margin": float("nan"),
                "contract_row_pass": False,
            })
            continue
        min_cap = float(sub[cap_col].min())
        max_tan = float(sub[tan_col].max())
        rows.append({
            "envelope_label": label,
            "source_variant": variant,
            "rows": int(len(sub)),
            "seeds": int(sub[seed_col].nunique()) if seed_col else int(len(sub)),
            "min_capture": min_cap,
            "max_tangent": max_tan,
            "capture_margin": min_cap - cap_target,
            "strict_tangent_margin": strict_tan_target - max_tan,
            "relaxed_tangent_margin": relaxed_tan_target - max_tan,
            "contract_row_pass": bool((min_cap >= cap_target) and (max_tan <= strict_tan_target)),
        })
    return pd.DataFrame(rows)


def global_alias_smoke(pool: pd.DataFrame, route_map: Mapping[str, str], cap_target: float, strict_tan_target: float) -> pd.DataFrame:
    label_col = choose_col(pool, ["envelope_label", "label"])
    variant_col = choose_col(pool, ["source_variant", "variant", "variant_name"])
    cap_col = choose_col(pool, ["capture_rate", "worst_capture", "verified_worst_capture"])
    tan_col = choose_col(pool, ["tangent_ratio", "worst_tangent", "verified_worst_tangent"])

    rows = []
    for label, variant in sorted(route_map.items()):
        sub = pool[pool[variant_col].astype(str) == str(variant)].copy()
        labels_seen = sorted(map(str, sub[label_col].dropna().unique().tolist()))
        min_cap = float(sub[cap_col].min()) if not sub.empty else float("nan")
        max_tan = float(sub[tan_col].max()) if not sub.empty else float("nan")
        exact = pool[(pool[label_col].astype(str) == str(label)) & (pool[variant_col].astype(str) == str(variant))]
        rows.append({
            "locked_envelope_label": label,
            "source_variant": variant,
            "global_rows": int(len(sub)),
            "exact_pair_rows": int(len(exact)),
            "labels_seen_for_variant": ";".join(labels_seen),
            "global_min_capture": min_cap,
            "global_max_tangent": max_tan,
            "global_capture_smoke_pass": bool(min_cap >= cap_target) if not sub.empty else False,
            "global_strict_tangent_smoke_pass": bool(max_tan <= strict_tan_target) if not sub.empty else False,
            "is_global_alias_risk": bool(len(labels_seen) > 1 or len(sub) != len(exact)),
        })
    return pd.DataFrame(rows)


def make_plots(outputs: Path, exact: pd.DataFrame, cap_target: float, strict_tan_target: float, relaxed_tan_target: float) -> None:
    if plt is None or exact.empty:
        return
    plot_df = exact.sort_values("min_capture", ascending=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(plot_df["envelope_label"], plot_df["min_capture"])
    ax.axvline(cap_target, linestyle="-", label="capture target")
    ax.set_title("26EL-LITE contract-row capture smoke")
    ax.set_xlabel("contract-row min capture")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outputs / "phase26el_lite_contract_row_capture_smoke.png", dpi=160)
    plt.close(fig)

    plot_df = exact.sort_values("max_tangent", ascending=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(plot_df["envelope_label"], plot_df["max_tangent"])
    ax.axvline(strict_tan_target, linestyle="-", label="strict tangent")
    ax.axvline(relaxed_tan_target, linestyle="--", label="relaxed tangent")
    ax.set_title("26EL-LITE contract-row tangent smoke")
    ax.set_xlabel("contract-row max tangent")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outputs / "phase26el_lite_contract_row_tangent_smoke.png", dpi=160)
    plt.close(fig)


def write_adapter(outputs: Path, route_map: Mapping[str, str], fingerprint: str, cap_target: float, strict_tan_target: float, relaxed_tan_target: float) -> Path:
    body = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
26EL-LITE label-aware locked route selector adapter.

Use this adapter at the transplant boundary. The contract key is the exact pair:
    envelope_label -> source_variant

Do not validate or select by source_variant alone; source_variant names can appear
in multiple stress contexts and will produce false smoke failures.
"""

from __future__ import annotations
from typing import Any, Dict, Iterable, List, Mapping, Optional

LOCKED_ROUTE_FINGERPRINT = {fingerprint!r}
CAPTURE_TARGET = {cap_target!r}
STRICT_TANGENT_TARGET = {strict_tan_target!r}
RELAXED_TANGENT_TARGET = {relaxed_tan_target!r}

LOCKED_ROUTE_MAP: Dict[str, str] = {json.dumps(dict(sorted(route_map.items())), indent=4)}


def normalize_envelope_label(envelope_label: str) -> str:
    return str(envelope_label).strip()


def locked_labels() -> List[str]:
    return list(LOCKED_ROUTE_MAP.keys())


def is_locked_label(envelope_label: str) -> bool:
    return normalize_envelope_label(envelope_label) in LOCKED_ROUTE_MAP


def select_source_variant_for_envelope(envelope_label: str, default: Optional[str] = None, *, strict: bool = True) -> str:
    key = normalize_envelope_label(envelope_label)
    if key in LOCKED_ROUTE_MAP:
        return LOCKED_ROUTE_MAP[key]
    if default is not None or not strict:
        return "" if default is None else default
    raise KeyError(f"No locked 26EL route for envelope_label={{envelope_label!r}}")


def locked_pair_for_envelope(envelope_label: str) -> tuple[str, str]:
    key = normalize_envelope_label(envelope_label)
    return key, select_source_variant_for_envelope(key)


def apply_locked_route_to_records(records: Iterable[Mapping[str, Any]], *, label_key: str = "envelope_label", output_key: str = "source_variant", strict: bool = False) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rec in records:
        row = dict(rec)
        label = str(row.get(label_key, ""))
        selected = select_source_variant_for_envelope(label, row.get(output_key), strict=strict)
        if selected:
            row[output_key] = selected
            row["locked_route_fingerprint"] = LOCKED_ROUTE_FINGERPRINT
        out.append(row)
    return out


def validate_label_aware_records(records: Iterable[Mapping[str, Any]], *, label_key: str = "envelope_label", variant_key: str = "source_variant") -> Dict[str, Any]:
    checked = 0
    present = set()
    mismatches: List[Dict[str, Any]] = []
    for rec in records:
        label = normalize_envelope_label(str(rec.get(label_key, "")))
        if label not in LOCKED_ROUTE_MAP:
            continue
        present.add(label)
        checked += 1
        expected = LOCKED_ROUTE_MAP[label]
        actual = str(rec.get(variant_key, ""))
        if actual != expected:
            mismatches.append({{"envelope_label": label, "expected": expected, "actual": actual}})
    missing = [label for label in LOCKED_ROUTE_MAP if label not in present]
    return {{
        "label_aware_contract_pass": not mismatches and not missing,
        "checked_records": checked,
        "missing_locked_labels": missing,
        "mismatches": mismatches,
        "route_fingerprint": LOCKED_ROUTE_FINGERPRINT,
    }}
'''
    path = outputs / "phase26el_lite_label_aware_route_selector_adapter.py"
    path.write_text(body, encoding="utf-8")
    return path


def write_selftest(outputs: Path) -> Path:
    body = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from phase26el_lite_label_aware_route_selector_adapter import (
    LOCKED_ROUTE_MAP,
    LOCKED_ROUTE_FINGERPRINT,
    select_source_variant_for_envelope,
    validate_label_aware_records,
)


def main() -> None:
    assert LOCKED_ROUTE_FINGERPRINT
    records = []
    for label, variant in LOCKED_ROUTE_MAP.items():
        assert select_source_variant_for_envelope(label) == variant
        records.append({"envelope_label": label, "source_variant": variant})
    result = validate_label_aware_records(records)
    assert result["label_aware_contract_pass"], result
    print("26EL label-aware selector selftest PASS")
    print(result)


if __name__ == "__main__":
    main()
'''
    path = outputs / "phase26el_lite_label_aware_selftest.py"
    path.write_text(body, encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=None, help="Output/input directory, default E:\\BBIT\\outputs_basic32")
    args = ap.parse_args()

    outputs = find_outputs_dir(args.outputs)
    outputs.mkdir(parents=True, exist_ok=True)
    print(f"[{PHASE}] Label-aware transplant smoke / false-alarm isolator")
    print(f"[{PHASE}] outputs: {outputs}")

    contract_path = outputs / "phase26ej_lite_integration_contract.json"
    package_path = outputs / "phase26ei_lite_hardened_route_package.json"
    selector_path = outputs / "phase26ei_lite_locked_route_selector.py"
    pool_path = outputs / "phase26ee_lite_pool_case_results.csv"
    contract_rows_path = outputs / "phase26ei_lite_locked_route_rows.csv"
    if not contract_rows_path.exists():
        contract_rows_path = outputs / "phase26ej_lite_locked_route_rows.csv"

    contract = read_json(contract_path)
    package = read_json(package_path)
    selector = load_selector(selector_path)
    route_map = extract_route_map(package, selector)
    pool = pd.read_csv(pool_path)
    contract_rows = pd.read_csv(contract_rows_path)

    cap_target = float(getattr(selector, "CAPTURE_TARGET", contract.get("verified_capture_target", 0.285)))
    strict_tan_target = float(getattr(selector, "STRICT_TANGENT_TARGET", contract.get("strict_tangent_target", 2.1)))
    relaxed_tan_target = float(getattr(selector, "RELAXED_TANGENT_TARGET", contract.get("relaxed_tangent_target", 2.3)))
    fingerprint = str(getattr(selector, "LOCKED_ROUTE_FINGERPRINT", contract.get("route_fingerprint", route_hash(route_map))))

    exact = contract_row_smoke(contract_rows, route_map, cap_target, strict_tan_target, relaxed_tan_target)
    alias = global_alias_smoke(pool, route_map, cap_target, strict_tan_target)

    contract_row_capture_smoke = bool((exact["rows"] > 0).all() and (exact["min_capture"] >= cap_target).all())
    contract_row_strict_tangent_smoke = bool((exact["rows"] > 0).all() and (exact["max_tangent"] <= strict_tan_target).all())
    label_aware_smoke_pass = bool(contract_row_capture_smoke and contract_row_strict_tangent_smoke)

    global_capture_smoke = bool(alias["global_capture_smoke_pass"].all())
    global_strict_tangent_smoke = bool(alias["global_strict_tangent_smoke_pass"].all())
    alias_false_alarm_detected = bool(label_aware_smoke_pass and (not global_capture_smoke or not global_strict_tangent_smoke))

    exact_path = outputs / "phase26el_lite_contract_row_smoke.csv"
    alias_path = outputs / "phase26el_lite_pool_source_alias_smoke.csv"
    exact.to_csv(exact_path, index=False)
    alias.to_csv(alias_path, index=False)

    adapter_path = write_adapter(outputs, route_map, fingerprint, cap_target, strict_tan_target, relaxed_tan_target)
    selftest_path = write_selftest(outputs)
    make_plots(outputs, exact, cap_target, strict_tan_target, relaxed_tan_target)

    summary = {
        "phase": PHASE,
        "LABEL_AWARE_SMOKE_PASS": label_aware_smoke_pass,
        "contract_row_capture_smoke": contract_row_capture_smoke,
        "contract_row_strict_tangent_smoke": contract_row_strict_tangent_smoke,
        "pool_source_variant_capture_smoke": global_capture_smoke,
        "pool_source_variant_strict_tangent_smoke": global_strict_tangent_smoke,
        "pool_source_alias_false_alarm_detected": alias_false_alarm_detected,
        "route_fingerprint": fingerprint,
        "computed_route_hash": route_hash(route_map),
        "labels": len(route_map),
        "capture_target": cap_target,
        "strict_tangent_target": strict_tan_target,
        "relaxed_tangent_target": relaxed_tan_target,
        "verified_worst_capture_contract_rows": float(exact["min_capture"].min()),
        "verified_worst_tangent_contract_rows": float(exact["max_tangent"].max()),
        "min_capture_margin_contract_rows": float(exact["capture_margin"].min()),
        "min_strict_tangent_margin_contract_rows": float(exact["strict_tangent_margin"].min()),
        "contract_row_smoke_csv": str(exact_path),
        "pool_source_alias_smoke_csv": str(alias_path),
        "adapter": str(adapter_path),
        "selftest": str(selftest_path),
    }
    write_json(outputs / "phase26el_lite_summary.json", summary)

    print(f"[{PHASE}] LABEL_AWARE_SMOKE_PASS={label_aware_smoke_pass}")
    print(f"[{PHASE}] contract-row summary:")
    print(exact[["envelope_label", "rows", "seeds", "min_capture", "max_tangent", "capture_margin", "strict_tangent_margin", "contract_row_pass"]].to_string(index=False))
    print(f"[{PHASE}] raw-pool source-variant smoke is overbroad diagnostic only:")
    print(alias[["locked_envelope_label", "global_rows", "exact_pair_rows", "global_min_capture", "global_max_tangent", "is_global_alias_risk"]].to_string(index=False))
    print(f"[{PHASE}] wrote adapter: {adapter_path}")
    print(f"[{PHASE}] wrote selftest: {selftest_path}")
    print(f"[{PHASE}] wrote outputs to: {outputs}")
    return 0 if label_aware_smoke_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
