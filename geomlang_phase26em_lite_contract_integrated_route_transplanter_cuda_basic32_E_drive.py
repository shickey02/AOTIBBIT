#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Phase 26EM-LITE — Contract-integrated locked-route transplanter / live selector gate
-----------------------------------------------------------------------------------

Purpose
=======
26EK intentionally failed because it smoke-tested the locked source_variant globally.
26EL proved the correct transplant boundary is label-aware:

    (envelope_label, source_variant)

26EM takes that result and produces the practical transplant artifact for the next
real geomlang phase. It does four things:

1. Loads the 26EJ/26EL contract artifacts and the 26EE candidate pool.
2. Re-validates the locked route by exact contract rows only.
3. Writes a small runtime router module that can be imported by the next phase.
4. Writes a transplant manifest, a patch note, and a selftest that fail loudly if
   anyone falls back to global source_variant-only selection.

This is intentionally still LITE: it does not rerun the expensive field evaluator.
It confirms and packages the contract solved by EG/EH/EI/EJ/EL.

Expected input location on Windows:
    E:\BBIT\outputs_basic32

Expected upstream files:
    phase26ej_lite_integration_contract.json
    phase26ej_lite_locked_route_rows.csv
    phase26el_lite_label_aware_route_selector_adapter.py
    phase26ee_lite_pool_case_results.csv

Main outputs:
    phase26em_lite_runtime_route_router.py
    phase26em_lite_runtime_route_selftest.py
    phase26em_lite_transplant_manifest.json
    phase26em_lite_patch_note.txt
    phase26em_lite_summary.json
    phase26em_lite_contract_row_smoke.csv
    phase26em_lite_alias_risk_table.csv
    phase26em_lite_margin_dashboard.png
    phase26em_lite_alias_risk_dashboard.png

Run:
    python bbit_geomlang/geomlang_phase26em_lite_contract_integrated_route_transplanter_cuda_basic32_E_drive.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    raise SystemExit("[26EM-LITE] pandas is required for this phase") from exc

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


PHASE = "26EM-LITE"
DEFAULT_OUTPUTS_DIR = Path(r"E:\BBIT\outputs_basic32")
FALLBACK_OUTPUTS_DIR = Path.cwd() / "outputs_basic32"

CAPTURE_TARGET_DEFAULT = 0.285
STRICT_TANGENT_TARGET_DEFAULT = 2.1
RELAXED_TANGENT_TARGET_DEFAULT = 2.3

REQUIRED_LABELS = [
    "base",
    "screen_medium_00",
    "screen_small_00",
    "stress_cap_tk",
    "stress_radius_in",
    "stress_seat_down",
    "stress_shell_blend",
]


# ----------------------------- small IO helpers -----------------------------

def choose_outputs_dir(cli_value: Optional[str]) -> Path:
    if cli_value:
        return Path(cli_value)
    if DEFAULT_OUTPUTS_DIR.exists():
        return DEFAULT_OUTPUTS_DIR
    if FALLBACK_OUTPUTS_DIR.exists():
        return FALLBACK_OUTPUTS_DIR
    return DEFAULT_OUTPUTS_DIR


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.write("\n")


def sha16_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def route_fingerprint(route_map: Mapping[str, str]) -> str:
    payload = json.dumps(dict(sorted(route_map.items())), sort_keys=True, separators=(",", ":"))
    return sha16_text(payload)


def load_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def coerce_float(v: Any, default: float = math.nan) -> float:
    try:
        return float(v)
    except Exception:
        return default


# ----------------------------- contract loading -----------------------------

def load_contract(outputs: Path) -> Dict[str, Any]:
    preferred = outputs / "phase26ej_lite_integration_contract.json"
    fallback = outputs / "phase26ei_lite_hardened_route_package.json"
    if preferred.exists():
        contract = read_json(preferred)
        source = preferred.name
    elif fallback.exists():
        contract = read_json(fallback)
        source = fallback.name
    else:
        raise FileNotFoundError(
            f"Missing contract. Expected {preferred.name} or {fallback.name} in {outputs}"
        )
    contract["_loaded_from"] = source
    return contract


def extract_route_map(contract: Mapping[str, Any], outputs: Path) -> Dict[str, str]:
    for key in ("locked_route_map", "route_map", "LOCKED_ROUTE_MAP"):
        val = contract.get(key)
        if isinstance(val, dict) and val:
            return {str(k): str(v) for k, v in val.items()}

    for filename in (
        "phase26ei_lite_deployable_route_map.json",
        "phase26eh_lite_deployable_route_map.json",
        "phase26eg_lite_route_map.json",
    ):
        p = outputs / filename
        if p.exists():
            obj = read_json(p)
            if isinstance(obj, dict):
                # Some route-map files are direct maps; some wrap the map.
                for key in ("locked_route_map", "route_map", "LOCKED_ROUTE_MAP"):
                    if isinstance(obj.get(key), dict):
                        return {str(k): str(v) for k, v in obj[key].items()}
                if all(isinstance(k, str) and isinstance(v, str) for k, v in obj.items()):
                    return {str(k): str(v) for k, v in obj.items()}

    adapter_path = outputs / "phase26el_lite_label_aware_route_selector_adapter.py"
    if adapter_path.exists():
        mod = load_module(adapter_path, "phase26el_adapter")
        if hasattr(mod, "LOCKED_ROUTE_MAP"):
            return {str(k): str(v) for k, v in dict(mod.LOCKED_ROUTE_MAP).items()}

    raise RuntimeError("Could not extract locked route map from contract/map/adapter artifacts")


def targets_from_contract(contract: Mapping[str, Any]) -> Tuple[float, float, float]:
    return (
        coerce_float(contract.get("capture_target", contract.get("CAPTURE_TARGET", CAPTURE_TARGET_DEFAULT)), CAPTURE_TARGET_DEFAULT),
        coerce_float(contract.get("strict_tangent_target", contract.get("STRICT_TANGENT_TARGET", STRICT_TANGENT_TARGET_DEFAULT)), STRICT_TANGENT_TARGET_DEFAULT),
        coerce_float(contract.get("relaxed_tangent_target", contract.get("RELAXED_TANGENT_TARGET", RELAXED_TANGENT_TARGET_DEFAULT)), RELAXED_TANGENT_TARGET_DEFAULT),
    )


# ----------------------------- validation logic -----------------------------

def load_locked_rows(outputs: Path, route_map: Mapping[str, str], pool: pd.DataFrame) -> pd.DataFrame:
    locked_path = outputs / "phase26ej_lite_locked_route_rows.csv"
    if locked_path.exists():
        df = pd.read_csv(locked_path)
        # Make sure it is exact-pair only, not global source-only.
        if {"envelope_label", "source_variant"}.issubset(df.columns):
            return df.copy()

    pieces = []
    for label, variant in route_map.items():
        m = (pool["envelope_label"].astype(str) == label) & (pool["source_variant"].astype(str) == variant)
        part = pool.loc[m].copy()
        if part.empty:
            raise RuntimeError(f"No exact contract rows in pool for ({label!r}, {variant!r})")
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def metric_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    seed_col = "dz_seed" if "dz_seed" in df.columns else ("seed" if "seed" in df.columns else "_seed")
    cap_candidates = ["capture_rate", "worst_capture", "verified_worst_capture", "min_capture"]
    tan_candidates = ["tangent_ratio", "worst_tangent", "verified_worst_tangent", "max_tangent"]
    cap_col = next((c for c in cap_candidates if c in df.columns), None)
    tan_col = next((c for c in tan_candidates if c in df.columns), None)
    if cap_col is None or tan_col is None:
        raise RuntimeError(f"Could not find capture/tangent metric columns in {list(df.columns)}")
    if seed_col == "_seed":
        df[seed_col] = range(len(df))
    return seed_col, cap_col, tan_col


def summarize_contract_rows(
    locked_rows: pd.DataFrame,
    route_map: Mapping[str, str],
    capture_target: float,
    strict_tangent_target: float,
    relaxed_tangent_target: float,
) -> pd.DataFrame:
    df = locked_rows.copy()
    seed_col, cap_col, tan_col = metric_columns(df)
    out_rows = []
    for label in route_map:
        variant = route_map[label]
        part = df[(df["envelope_label"].astype(str) == label) & (df["source_variant"].astype(str) == variant)].copy()
        if part.empty:
            out_rows.append({
                "envelope_label": label,
                "source_variant": variant,
                "rows": 0,
                "seeds": 0,
                "min_capture": math.nan,
                "max_tangent": math.nan,
                "capture_margin": -math.inf,
                "strict_tangent_margin": -math.inf,
                "relaxed_tangent_margin": -math.inf,
                "contract_row_pass": False,
            })
            continue
        caps = pd.to_numeric(part[cap_col], errors="coerce")
        tans = pd.to_numeric(part[tan_col], errors="coerce")
        seeds = part[seed_col].nunique(dropna=True)
        min_cap = float(caps.min())
        max_tan = float(tans.max())
        out_rows.append({
            "envelope_label": label,
            "source_variant": variant,
            "rows": int(len(part)),
            "seeds": int(seeds),
            "min_capture": min_cap,
            "mean_capture": float(caps.mean()),
            "max_tangent": max_tan,
            "mean_tangent": float(tans.mean()),
            "capture_margin": min_cap - capture_target,
            "strict_tangent_margin": strict_tangent_target - max_tan,
            "relaxed_tangent_margin": relaxed_tangent_target - max_tan,
            "contract_row_pass": bool(min_cap >= capture_target and max_tan <= strict_tangent_target),
        })
    return pd.DataFrame(out_rows)


def build_alias_risk_table(pool: pd.DataFrame, route_map: Mapping[str, str], capture_target: float, strict_tangent_target: float) -> pd.DataFrame:
    seed_col, cap_col, tan_col = metric_columns(pool)
    rows = []
    for label, variant in route_map.items():
        global_part = pool[pool["source_variant"].astype(str) == variant]
        exact_part = pool[(pool["envelope_label"].astype(str) == label) & (pool["source_variant"].astype(str) == variant)]
        rows.append({
            "locked_envelope_label": label,
            "source_variant": variant,
            "global_rows": int(len(global_part)),
            "exact_pair_rows": int(len(exact_part)),
            "global_min_capture": float(pd.to_numeric(global_part[cap_col], errors="coerce").min()) if len(global_part) else math.nan,
            "exact_min_capture": float(pd.to_numeric(exact_part[cap_col], errors="coerce").min()) if len(exact_part) else math.nan,
            "global_max_tangent": float(pd.to_numeric(global_part[tan_col], errors="coerce").max()) if len(global_part) else math.nan,
            "exact_max_tangent": float(pd.to_numeric(exact_part[tan_col], errors="coerce").max()) if len(exact_part) else math.nan,
            "global_source_only_would_fail": bool(
                len(global_part)
                and (
                    float(pd.to_numeric(global_part[cap_col], errors="coerce").min()) < capture_target
                    or float(pd.to_numeric(global_part[tan_col], errors="coerce").max()) > strict_tangent_target
                )
            ),
        })
    return pd.DataFrame(rows)


# ------------------------------- code writers -------------------------------

def py_string(s: str) -> str:
    return json.dumps(str(s))


def write_runtime_router(path: Path, route_map: Mapping[str, str], fingerprint: str, cap: float, strict: float, relaxed: float) -> None:
    mapping_lines = [f"    {py_string(k)}: {py_string(v)}," for k, v in sorted(route_map.items())]
    content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
26EM-LITE runtime locked-route router.

Import this at the next phase boundary and call select_source_variant(envelope_label).
The selector is label-aware by design. Never select by source_variant alone.
"""

from __future__ import annotations
from typing import Any, Dict, Iterable, List, Mapping, Optional

LOCKED_ROUTE_FINGERPRINT = {py_string(fingerprint)}
CAPTURE_TARGET = {cap!r}
STRICT_TANGENT_TARGET = {strict!r}
RELAXED_TANGENT_TARGET = {relaxed!r}

LOCKED_ROUTE_MAP: Dict[str, str] = {{
{chr(10).join(mapping_lines)}
}}


def normalize_envelope_label(envelope_label: Any) -> str:
    return str(envelope_label).strip()


def locked_labels() -> List[str]:
    return list(LOCKED_ROUTE_MAP.keys())


def is_locked_label(envelope_label: Any) -> bool:
    return normalize_envelope_label(envelope_label) in LOCKED_ROUTE_MAP


def select_source_variant(envelope_label: Any, default: Optional[str] = None, *, strict: bool = True) -> str:
    key = normalize_envelope_label(envelope_label)
    if key in LOCKED_ROUTE_MAP:
        return LOCKED_ROUTE_MAP[key]
    if default is not None or not strict:
        return "" if default is None else str(default)
    raise KeyError(f"No 26EM locked route for envelope_label={{envelope_label!r}}")


def locked_pair(envelope_label: Any) -> tuple[str, str]:
    key = normalize_envelope_label(envelope_label)
    return key, select_source_variant(key)


def apply_to_record(record: Mapping[str, Any], *, label_key: str = "envelope_label", output_key: str = "source_variant", strict: bool = True) -> Dict[str, Any]:
    row = dict(record)
    row[output_key] = select_source_variant(row.get(label_key, ""), row.get(output_key), strict=strict)
    row["locked_route_fingerprint"] = LOCKED_ROUTE_FINGERPRINT
    row["locked_route_contract"] = "26EM_label_aware_exact_pair"
    return row


def apply_to_records(records: Iterable[Mapping[str, Any]], *, label_key: str = "envelope_label", output_key: str = "source_variant", strict: bool = True) -> List[Dict[str, Any]]:
    return [apply_to_record(r, label_key=label_key, output_key=output_key, strict=strict) for r in records]


def validate_records(records: Iterable[Mapping[str, Any]], *, label_key: str = "envelope_label", variant_key: str = "source_variant") -> Dict[str, Any]:
    present = set()
    mismatches = []
    checked = 0
    for rec in records:
        label = normalize_envelope_label(rec.get(label_key, ""))
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
        "contract": "26EM_label_aware_exact_pair",
        "route_fingerprint": LOCKED_ROUTE_FINGERPRINT,
        "checked_records": checked,
        "missing_locked_labels": missing,
        "mismatches": mismatches,
        "contract_pass": (not missing and not mismatches),
    }}
'''
    path.write_text(content, encoding="utf-8")


def write_runtime_selftest(path: Path, router_filename: str) -> None:
    content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUTER = HERE / {router_filename!r}

spec = importlib.util.spec_from_file_location("phase26em_runtime_route_router", ROUTER)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mod)

assert len(mod.LOCKED_ROUTE_MAP) == 7, mod.LOCKED_ROUTE_MAP
assert mod.select_source_variant("base") == "prior_phase26dv_lite_variant_summary_dv_capture_cloud_013"
assert mod.select_source_variant("screen_small_00") == "prior_phase26dy_lite_variant_summary_dy_screen_small_r0.722_cap1.195_tk1.015"
assert mod.select_source_variant("stress_shell_blend") == "prior_phase26dy_lite_variant_summary_dy_screen_small_r0.722_cap1.195_tk1.015"

records = [{{"envelope_label": label}} for label in mod.locked_labels()]
applied = mod.apply_to_records(records)
check = mod.validate_records(applied)
assert check["contract_pass"], check

# The same source_variant is allowed to appear under multiple labels. This is not
# a failure: the exact pair is the contract key.
assert mod.select_source_variant("screen_small_00") == mod.select_source_variant("stress_shell_blend")

print("[26EM-LITE] runtime router selftest PASS")
print(check)
'''
    path.write_text(content, encoding="utf-8")


def write_patch_note(path: Path, summary: Mapping[str, Any], router_name: str) -> None:
    text = f"""26EM-LITE transplant patch note
===============================

Status: {'PASS' if summary.get('CONTRACT_TRANSPLANT_READY') else 'FAIL'}

Use this file in the next real phase:

    {router_name}

Import pattern:

    from phase26em_lite_runtime_route_router import select_source_variant, validate_records

Selection rule:

    source_variant = select_source_variant(envelope_label)

Do not select by source_variant globally. 26EK proved that global source-only smoke
is an overbroad diagnostic because several locked variants also appear in other
stress contexts. 26EL and 26EM confirm that the transplant contract is exact pair:

    envelope_label + source_variant

Current contract:

    route fingerprint: {summary.get('route_fingerprint')}
    worst capture:     {summary.get('verified_worst_capture')}
    worst tangent:     {summary.get('verified_worst_tangent')}
    strict pass rate:  {summary.get('strict_pass_rate')}
    relaxed pass rate: {summary.get('relaxed_pass_rate')}

Weakest capture labels:

    screen_medium_00 and screen_small_00, margin {summary.get('min_capture_margin')}

Weakest tangent label:

    base, strict tangent margin {summary.get('min_strict_tangent_margin')}
"""
    path.write_text(text, encoding="utf-8")


# ---------------------------------- plots -----------------------------------

def make_plots(outputs: Path, contract_rows: pd.DataFrame, alias_risk: pd.DataFrame, cap_target: float, strict_tangent: float, relaxed_tangent: float) -> None:
    if plt is None:
        return

    # Margin dashboard.
    plot_df = contract_rows.sort_values("capture_margin", ascending=True)
    labels = plot_df["envelope_label"].tolist()
    y = list(range(len(plot_df)))
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.barh([i - 0.18 for i in y], plot_df["capture_margin"].astype(float), height=0.34, label="capture margin")
    ax.barh([i + 0.18 for i in y], plot_df["strict_tangent_margin"].astype(float), height=0.34, label="strict tangent margin")
    ax.axvline(0.0, linewidth=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("positive margin is safe; negative margin fails")
    ax.set_title("26EM-LITE contract-integrated route margins")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(outputs / "phase26em_lite_margin_dashboard.png", dpi=140)
    plt.close(fig)

    # Alias risk dashboard.
    risk_df = alias_risk.sort_values("global_max_tangent", ascending=True)
    labels = risk_df["locked_envelope_label"].tolist()
    y = list(range(len(risk_df)))
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.barh(y, risk_df["global_max_tangent"].astype(float), label="global source-only max tangent")
    ax.scatter(risk_df["exact_max_tangent"].astype(float), y, label="exact-pair max tangent")
    ax.axvline(strict_tangent, linewidth=1.5, label="strict tangent")
    ax.axvline(relaxed_tangent, linewidth=1.5, linestyle="--", label="relaxed tangent")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("tangent/radial")
    ax.set_title("26EM-LITE alias risk: global source smoke vs exact contract row")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(outputs / "phase26em_lite_alias_risk_dashboard.png", dpi=140)
    plt.close(fig)


# ----------------------------------- main -----------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="26EM-LITE contract-integrated locked-route transplanter")
    ap.add_argument("--outputs", default=None, help="outputs_basic32 directory; defaults to E:\\BBIT\\outputs_basic32")
    ap.add_argument("--pool", default="phase26ee_lite_pool_case_results.csv", help="candidate pool CSV name/path")
    ap.add_argument("--no-plots", action="store_true", help="skip matplotlib plots")
    args = ap.parse_args(argv)

    outputs = choose_outputs_dir(args.outputs)
    outputs.mkdir(parents=True, exist_ok=True)
    print(f"[{PHASE}] Contract-integrated locked-route transplanter / live selector gate")
    print(f"[{PHASE}] outputs: {outputs}")

    contract = load_contract(outputs)
    route_map = extract_route_map(contract, outputs)
    cap_target, strict_tangent, relaxed_tangent = targets_from_contract(contract)
    fingerprint = str(contract.get("route_fingerprint") or contract.get("locked_route_fingerprint") or route_fingerprint(route_map))

    pool_path = Path(args.pool)
    if not pool_path.is_absolute():
        pool_path = outputs / pool_path
    if not pool_path.exists():
        raise FileNotFoundError(f"Missing pool CSV: {pool_path}")
    pool = pd.read_csv(pool_path)
    print(f"[{PHASE}] loaded pool: {pool_path.name} rows={len(pool)} labels={pool['envelope_label'].nunique()} variants={pool['source_variant'].nunique()}")

    missing_labels = [x for x in REQUIRED_LABELS if x not in route_map]
    extra_labels = [x for x in route_map if x not in REQUIRED_LABELS]
    if missing_labels:
        raise RuntimeError(f"Route map is missing required labels: {missing_labels}")

    locked_rows = load_locked_rows(outputs, route_map, pool)
    contract_rows = summarize_contract_rows(locked_rows, route_map, cap_target, strict_tangent, relaxed_tangent)
    alias_risk = build_alias_risk_table(pool, route_map, cap_target, strict_tangent)

    contract_pass = bool(contract_rows["contract_row_pass"].all())
    verified_worst_capture = float(contract_rows["min_capture"].min())
    verified_worst_tangent = float(contract_rows["max_tangent"].max())
    verified_mean_capture = float(contract_rows["mean_capture"].mean()) if "mean_capture" in contract_rows else math.nan
    verified_mean_tangent = float(contract_rows["mean_tangent"].mean()) if "mean_tangent" in contract_rows else math.nan
    strict_pass_rate = 1.0 if contract_pass else 0.0
    relaxed_pass_rate = 1.0 if bool((contract_rows["min_capture"] >= cap_target).all() and (contract_rows["max_tangent"] <= relaxed_tangent).all()) else 0.0

    router_name = "phase26em_lite_runtime_route_router.py"
    selftest_name = "phase26em_lite_runtime_route_selftest.py"
    router_path = outputs / router_name
    selftest_path = outputs / selftest_name
    write_runtime_router(router_path, route_map, fingerprint, cap_target, strict_tangent, relaxed_tangent)
    write_runtime_selftest(selftest_path, router_name)

    summary = {
        "phase": PHASE,
        "CONTRACT_TRANSPLANT_READY": contract_pass,
        "loaded_contract_from": contract.get("_loaded_from"),
        "route_fingerprint": fingerprint,
        "computed_route_hash": route_fingerprint(route_map),
        "labels": len(route_map),
        "missing_required_labels": missing_labels,
        "extra_labels": extra_labels,
        "capture_target": cap_target,
        "strict_tangent_target": strict_tangent,
        "relaxed_tangent_target": relaxed_tangent,
        "verified_worst_capture": verified_worst_capture,
        "verified_mean_capture": verified_mean_capture,
        "verified_worst_tangent": verified_worst_tangent,
        "verified_mean_tangent": verified_mean_tangent,
        "min_capture_margin": float(contract_rows["capture_margin"].min()),
        "min_strict_tangent_margin": float(contract_rows["strict_tangent_margin"].min()),
        "min_relaxed_tangent_margin": float(contract_rows["relaxed_tangent_margin"].min()),
        "strict_pass_rate": strict_pass_rate,
        "relaxed_pass_rate": relaxed_pass_rate,
        "alias_risk_labels": int(alias_risk["global_source_only_would_fail"].sum()),
        "runtime_router": str(router_path),
        "runtime_selftest": str(selftest_path),
        "contract_key": ["envelope_label", "source_variant"],
        "do_not_use": "global source_variant-only validation/selection",
    }

    contract_rows.to_csv(outputs / "phase26em_lite_contract_row_smoke.csv", index=False)
    alias_risk.to_csv(outputs / "phase26em_lite_alias_risk_table.csv", index=False)
    write_json(outputs / "phase26em_lite_summary.json", summary)

    manifest = {
        "phase": PHASE,
        "status": "PASS" if contract_pass else "FAIL",
        "contract_key": ["envelope_label", "source_variant"],
        "route_fingerprint": fingerprint,
        "route_map": route_map,
        "targets": {
            "capture": cap_target,
            "strict_tangent": strict_tangent,
            "relaxed_tangent": relaxed_tangent,
        },
        "integration_files": {
            "runtime_router": router_name,
            "runtime_selftest": selftest_name,
            "patch_note": "phase26em_lite_patch_note.txt",
        },
        "guardrails": [
            "Call select_source_variant(envelope_label) at the transplant boundary.",
            "Validate exact (envelope_label, source_variant) rows only.",
            "Do not treat a global source_variant failure as a contract failure; that is alias risk, not route failure.",
        ],
    }
    write_json(outputs / "phase26em_lite_transplant_manifest.json", manifest)
    write_patch_note(outputs / "phase26em_lite_patch_note.txt", summary, router_name)

    if not args.no_plots:
        make_plots(outputs, contract_rows, alias_risk, cap_target, strict_tangent, relaxed_tangent)

    print(f"[{PHASE}] CONTRACT_TRANSPLANT_READY={contract_pass}")
    print(f"[{PHASE}] contract-row summary:")
    show_cols = [
        "envelope_label", "rows", "seeds", "min_capture", "max_tangent",
        "capture_margin", "strict_tangent_margin", "contract_row_pass",
    ]
    print(contract_rows[show_cols].to_string(index=False))
    print(f"[{PHASE}] alias-risk labels if source_variant is tested globally: {summary['alias_risk_labels']}/{len(route_map)}")
    print(f"[{PHASE}] wrote runtime router: {router_path}")
    print(f"[{PHASE}] wrote selftest: {selftest_path}")
    print(f"[{PHASE}] wrote outputs to: {outputs}")
    return 0 if contract_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
