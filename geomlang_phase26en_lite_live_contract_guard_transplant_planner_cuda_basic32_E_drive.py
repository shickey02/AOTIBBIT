#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
26EN-LITE live contract guard / transplant planner.

Purpose
-------
26EM proved that the locked route is valid only as an exact pair contract:

    envelope_label + source_variant

26EK's failed smoke was useful because it showed the danger of testing a locked
source_variant globally. 26EL and 26EM isolated that as alias risk, not route
failure. 26EN turns that result into a live integration guard and a concrete
transplant plan for the next real phase.

This script does not mutate your parent geomlang files. It writes a small guard
module, a selftest, a callsite scan, and a Codex handoff note. The guard module
is the part to import at the real route-selection boundary.

Default paths assume:

    E:\BBIT\outputs_basic32
    E:\BBIT\bbit_geomlang

The script also works from another machine if --outputs and --source-root are
provided.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    raise SystemExit("26EN-LITE requires pandas. Install with: pip install pandas") from exc

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

PHASE = "26EN-LITE"
REQUIRED_LABELS = [
    "base",
    "screen_medium_00",
    "screen_small_00",
    "stress_cap_tk",
    "stress_radius_in",
    "stress_seat_down",
    "stress_shell_blend",
]

STRICT_TANGENT_FALLBACK = 2.10
RELAXED_TANGENT_FALLBACK = 2.30
CAPTURE_TARGET_FALLBACK = 0.285


# ------------------------------- path helpers ------------------------------

def default_outputs_dir() -> Path:
    e = Path(r"E:\BBIT\outputs_basic32")
    if e.exists() or os.name == "nt":
        return e
    return Path.cwd() / "outputs_basic32"


def default_source_root() -> Path:
    e = Path(r"E:\BBIT\bbit_geomlang")
    if e.exists() or os.name == "nt":
        return e
    return Path.cwd() / "bbit_geomlang"


def resolve_outputs(p: Optional[str]) -> Path:
    return Path(p).expanduser().resolve() if p else default_outputs_dir()


def resolve_source_root(p: Optional[str]) -> Path:
    return Path(p).expanduser().resolve() if p else default_source_root()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def sha256_text(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def route_fingerprint(route_map: Mapping[str, str]) -> str:
    blob = json.dumps({k: route_map[k] for k in sorted(route_map)}, sort_keys=True)
    return sha256_text(blob, 16)


# ------------------------------- loaders -----------------------------------

def load_em_artifacts(outputs: Path) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    summary_path = outputs / "phase26em_lite_summary.json"
    manifest_path = outputs / "phase26em_lite_transplant_manifest.json"
    contract_rows_path = outputs / "phase26em_lite_contract_row_smoke.csv"
    alias_path = outputs / "phase26em_lite_alias_risk_table.csv"

    missing = [str(p) for p in [summary_path, manifest_path, contract_rows_path, alias_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing 26EM artifact(s). Run 26EM first or pass --outputs to the directory containing them:\n  "
            + "\n  ".join(missing)
        )

    summary = read_json(summary_path)
    manifest = read_json(manifest_path)
    contract_rows = pd.read_csv(contract_rows_path)
    alias_risk = pd.read_csv(alias_path)
    return summary, manifest, contract_rows, alias_risk


def extract_route_map(summary: Mapping[str, Any], manifest: Mapping[str, Any]) -> Dict[str, str]:
    route_map = manifest.get("route_map") or summary.get("route_map")
    if not isinstance(route_map, dict):
        raise RuntimeError("Could not find route_map in 26EM summary/manifest.")
    return {str(k): str(v) for k, v in route_map.items()}


def targets(summary: Mapping[str, Any], manifest: Mapping[str, Any]) -> Tuple[float, float, float]:
    mt = manifest.get("targets") if isinstance(manifest.get("targets"), dict) else {}
    cap = float(summary.get("capture_target", mt.get("capture", CAPTURE_TARGET_FALLBACK)))
    strict = float(summary.get("strict_tangent_target", mt.get("strict_tangent", STRICT_TANGENT_FALLBACK)))
    relaxed = float(summary.get("relaxed_tangent_target", mt.get("relaxed_tangent", RELAXED_TANGENT_FALLBACK)))
    return cap, strict, relaxed


def import_module_from_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def run_python(path: Path) -> Tuple[int, str, str]:
    proc = subprocess.run([sys.executable, str(path)], capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


# ------------------------------- guard writer ------------------------------

def py_string(s: Any) -> str:
    return json.dumps(str(s))


def write_live_guard(
    path: Path,
    route_map: Mapping[str, str],
    fingerprint: str,
    capture_target: float,
    strict_tangent_target: float,
    relaxed_tangent_target: float,
) -> None:
    mapping_lines = [f"    {py_string(k)}: {py_string(v)}," for k, v in sorted(route_map.items())]
    content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
26EN-LITE live integration guard.

Import this in the first real phase that chooses or filters route variants.
The invariant is exact-pair routing:

    envelope_label -> locked source_variant

Never validate a locked source_variant globally without its envelope_label.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

LOCKED_ROUTE_FINGERPRINT = {py_string(fingerprint)}
CONTRACT_NAME = "26EN_label_aware_exact_pair_guard"
CAPTURE_TARGET = {capture_target!r}
STRICT_TANGENT_TARGET = {strict_tangent_target!r}
RELAXED_TANGENT_TARGET = {relaxed_tangent_target!r}

LOCKED_ROUTE_MAP: Dict[str, str] = {{
{chr(10).join(mapping_lines)}
}}


@dataclass(frozen=True)
class RouteDecision:
    envelope_label: str
    source_variant: str
    locked: bool
    route_fingerprint: str = LOCKED_ROUTE_FINGERPRINT
    contract: str = CONTRACT_NAME


def normalize_label(envelope_label: Any) -> str:
    return str(envelope_label).strip()


def locked_labels() -> List[str]:
    return list(LOCKED_ROUTE_MAP.keys())


def is_locked_label(envelope_label: Any) -> bool:
    return normalize_label(envelope_label) in LOCKED_ROUTE_MAP


def select_source_variant(envelope_label: Any, default: Optional[Any] = None, *, strict: bool = True) -> str:
    label = normalize_label(envelope_label)
    if label in LOCKED_ROUTE_MAP:
        return LOCKED_ROUTE_MAP[label]
    if default is not None:
        return str(default)
    if strict:
        raise KeyError(f"No locked 26EN route for envelope_label={{envelope_label!r}}")
    return ""


def decide(envelope_label: Any, default: Optional[Any] = None, *, strict: bool = True) -> RouteDecision:
    label = normalize_label(envelope_label)
    return RouteDecision(
        envelope_label=label,
        source_variant=select_source_variant(label, default=default, strict=strict),
        locked=(label in LOCKED_ROUTE_MAP),
    )


def apply_to_record(
    record: Mapping[str, Any],
    *,
    label_key: str = "envelope_label",
    variant_key: str = "source_variant",
    strict: bool = True,
) -> Dict[str, Any]:
    row = dict(record)
    label = normalize_label(row.get(label_key, ""))
    decision = decide(label, default=row.get(variant_key), strict=strict)
    row[label_key] = decision.envelope_label
    row[variant_key] = decision.source_variant
    row["locked_route_contract"] = decision.contract
    row["locked_route_fingerprint"] = decision.route_fingerprint
    row["locked_route_selected"] = decision.locked
    return row


def apply_to_records(
    records: Iterable[Mapping[str, Any]],
    *,
    label_key: str = "envelope_label",
    variant_key: str = "source_variant",
    strict: bool = True,
) -> List[Dict[str, Any]]:
    return [apply_to_record(r, label_key=label_key, variant_key=variant_key, strict=strict) for r in records]


def validate_exact_pair_records(
    records: Iterable[Mapping[str, Any]],
    *,
    label_key: str = "envelope_label",
    variant_key: str = "source_variant",
    require_all_labels: bool = True,
) -> Dict[str, Any]:
    present = set()
    mismatches: List[Dict[str, str]] = []
    checked = 0
    for rec in records:
        label = normalize_label(rec.get(label_key, ""))
        if label not in LOCKED_ROUTE_MAP:
            continue
        present.add(label)
        checked += 1
        expected = LOCKED_ROUTE_MAP[label]
        actual = str(rec.get(variant_key, ""))
        if actual != expected:
            mismatches.append({{"envelope_label": label, "expected": expected, "actual": actual}})
    missing = [label for label in LOCKED_ROUTE_MAP if label not in present] if require_all_labels else []
    return {{
        "contract": CONTRACT_NAME,
        "route_fingerprint": LOCKED_ROUTE_FINGERPRINT,
        "checked_records": checked,
        "missing_locked_labels": missing,
        "mismatches": mismatches,
        "contract_pass": (not missing and not mismatches),
    }}


def filter_dataframe_exact_pair(df: Any, *, label_col: str = "envelope_label", variant_col: str = "source_variant") -> Any:
    """Return only rows matching the exact locked (label, variant) route.

    This function is intentionally written to work with pandas without importing
    pandas at module import time. It expects df-like boolean indexing.
    """
    mask = None
    for label, variant in LOCKED_ROUTE_MAP.items():
        m = (df[label_col].astype(str) == label) & (df[variant_col].astype(str) == variant)
        mask = m if mask is None else (mask | m)
    return df[mask].copy()


def assert_contract_metrics(
    rows: Iterable[Mapping[str, Any]],
    *,
    label_key: str = "envelope_label",
    capture_key: str = "min_capture",
    tangent_key: str = "max_tangent",
) -> Dict[str, Any]:
    failures = []
    checked = 0
    for rec in rows:
        label = normalize_label(rec.get(label_key, ""))
        if label not in LOCKED_ROUTE_MAP:
            continue
        checked += 1
        cap = float(rec.get(capture_key))
        tan = float(rec.get(tangent_key))
        if cap < CAPTURE_TARGET or tan > STRICT_TANGENT_TARGET:
            failures.append({{"envelope_label": label, "capture": cap, "tangent": tan}})
    return {{
        "contract": CONTRACT_NAME,
        "checked_labels": checked,
        "metric_failures": failures,
        "metrics_pass": not failures,
        "capture_target": CAPTURE_TARGET,
        "strict_tangent_target": STRICT_TANGENT_TARGET,
    }}
'''
    path.write_text(content, encoding="utf-8")


def write_guard_selftest(path: Path, guard_filename: str, contract_csv_name: str) -> None:
    content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
GUARD_PATH = HERE / {guard_filename!r}
CONTRACT_ROWS = HERE / {contract_csv_name!r}

spec = importlib.util.spec_from_file_location("phase26en_live_guard", GUARD_PATH)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

assert len(mod.LOCKED_ROUTE_MAP) == 7
assert mod.select_source_variant("base") == "prior_phase26dv_lite_variant_summary_dv_capture_cloud_013"
assert mod.select_source_variant("screen_medium_00") == "prior_phase26dy_lite_variant_summary_dy_terminal_cloud_009"
assert mod.select_source_variant("screen_small_00") == "prior_phase26dy_lite_variant_summary_dy_screen_small_r0.722_cap1.195_tk1.015"
assert mod.select_source_variant("stress_shell_blend") == "prior_phase26dy_lite_variant_summary_dy_screen_small_r0.722_cap1.195_tk1.015"

records = [{{"envelope_label": label}} for label in mod.locked_labels()]
applied = mod.apply_to_records(records)
exact = mod.validate_exact_pair_records(applied)
assert exact["contract_pass"], exact

rows = pd.read_csv(CONTRACT_ROWS)
metric = mod.assert_contract_metrics(rows.to_dict("records"))
assert metric["metrics_pass"], metric

# Prove the alias edge case remains safe: same source_variant may appear under
# multiple labels, but selection is keyed by envelope_label.
assert mod.select_source_variant("screen_small_00") == mod.select_source_variant("stress_shell_blend")
assert mod.decide("screen_small_00").envelope_label != mod.decide("stress_shell_blend").envelope_label

print("[26EN-LITE] live guard selftest PASS")
print(exact)
print(metric)
'''
    path.write_text(content, encoding="utf-8")


# ------------------------------- source scanner ----------------------------

KEYWORDS = [
    "source_variant",
    "envelope_label",
    "select_source_variant",
    "variant_summary",
    "route_map",
    "evaluate_field",
    "BACKBONE_EVALUATE_FIELD",
    "NativeClutchIntegrator",
]


def scan_source_root(source_root: Path, max_files: int = 600) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if not source_root.exists():
        return pd.DataFrame(columns=["path", "score", "keyword_hits", "line_hits", "suggestion"])

    files = sorted(source_root.glob("geomlang_phase26*.py"), key=lambda p: p.stat().st_mtime, reverse=True)[:max_files]
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        hits = {kw: text.count(kw) for kw in KEYWORDS}
        score = (
            hits["source_variant"] * 6
            + hits["envelope_label"] * 6
            + hits["route_map"] * 5
            + hits["select_source_variant"] * 8
            + hits["evaluate_field"] * 2
            + hits["BACKBONE_EVALUATE_FIELD"] * 4
            + hits["NativeClutchIntegrator"] * 3
            + hits["variant_summary"] * 3
        )
        if score <= 0:
            continue
        line_hits = []
        for i, line in enumerate(text.splitlines(), start=1):
            if any(kw in line for kw in KEYWORDS[:5]):
                line_hits.append(i)
            if len(line_hits) >= 12:
                break
        suggestion = "inspect route/candidate selection boundary" if hits["source_variant"] or hits["route_map"] else "secondary context only"
        rows.append({
            "path": str(path),
            "file": path.name,
            "score": int(score),
            "source_variant_hits": int(hits["source_variant"]),
            "envelope_label_hits": int(hits["envelope_label"]),
            "route_map_hits": int(hits["route_map"]),
            "evaluate_field_hits": int(hits["evaluate_field"]),
            "line_hits": ";".join(map(str, line_hits)),
            "suggestion": suggestion,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["score", "source_variant_hits", "route_map_hits"], ascending=False)
    return df


# ------------------------------- docs/plots --------------------------------

def write_codex_handoff(
    path: Path,
    guard_name: str,
    selftest_name: str,
    summary: Mapping[str, Any],
    callsites: pd.DataFrame,
) -> None:
    top = []
    if not callsites.empty:
        for _, r in callsites.head(12).iterrows():
            top.append(f"- {r['file']} | score={r['score']} | lines={r['line_hits']} | {r['suggestion']}")
    else:
        top.append("- No source-root scan results were available. Use ripgrep manually for source_variant/envelope_label/route_map.")

    text = f"""26EN-LITE Codex transplant handoff
===================================

Goal
----
Integrate the locked 26EM/26EN exact-pair route into the next real geomlang phase
without reintroducing the 26EK false alarm.

The contract is:

    envelope_label + source_variant

not:

    source_variant alone

Files written by 26EN
---------------------

- {guard_name}
- {selftest_name}
- phase26en_lite_live_transplant_checklist.md
- phase26en_lite_callsite_candidates.csv
- phase26en_lite_summary.json

Required behavior
-----------------

1. Import the guard module near the phase's route/candidate selection boundary.
2. Replace any global/default source variant choice with:

       selected_variant = select_source_variant(envelope_label, default=old_variant, strict=False)

   or, for strict locked labels:

       selected_variant = select_source_variant(envelope_label)

3. Do not validate by source_variant alone. If a source_variant appears in another
   envelope label, that is alias risk and must not be treated as contract failure.
4. Exact rows must satisfy:

       min_capture >= {summary.get('capture_target')}
       max_tangent <= {summary.get('strict_tangent_target')}

5. Preserve these current locked-route floors:

       verified_worst_capture = {summary.get('verified_worst_capture')}
       verified_worst_tangent = {summary.get('verified_worst_tangent')}
       min_capture_margin = {summary.get('min_capture_margin')}
       min_strict_tangent_margin = {summary.get('min_strict_tangent_margin')}

Suggested callsites from scan
-----------------------------

{chr(10).join(top)}

Acceptance tests
----------------

Run:

    python E:\\BBIT\\outputs_basic32\\{selftest_name}

Then run the real next phase and verify that any exported row table contains the
locked_route_fingerprint and that exact-pair rows, not global source-only rows,
are used for contract validation.

Success means
-------------

- all seven locked labels are selected by envelope_label
- CONTRACT_PASS remains true
- strict pass rate remains 1.0
- no source_variant-only smoke is used as a blocking criterion
"""
    path.write_text(text, encoding="utf-8")


def write_checklist(path: Path, summary: Mapping[str, Any]) -> None:
    text = f"""# 26EN-LITE live transplant checklist

## Contract

- Key: `envelope_label + source_variant`
- Route fingerprint: `{summary.get('route_fingerprint')}`
- Capture target: `{summary.get('capture_target')}`
- Strict tangent target: `{summary.get('strict_tangent_target')}`
- Relaxed tangent target: `{summary.get('relaxed_tangent_target')}`

## Current locked-route result

- Worst capture: `{summary.get('verified_worst_capture')}`
- Worst tangent: `{summary.get('verified_worst_tangent')}`
- Strict pass rate: `{summary.get('strict_pass_rate')}`
- Weakest capture margin: `{summary.get('min_capture_margin')}`
- Weakest strict tangent margin: `{summary.get('min_strict_tangent_margin')}`

## Guardrail

Do **not** use global `source_variant` validation. 26EM reported alias risk on all
seven labels when the source variant is tested outside its label. That is exactly
why EN exports a label-aware guard.

## Integration point

Patch the first location where a phase converts an envelope label into a variant
or candidate source. The patch should call:

```python
from phase26en_lite_live_contract_guard import select_source_variant
source_variant = select_source_variant(envelope_label, default=source_variant, strict=False)
```

For a strict locked-only pass, use:

```python
source_variant = select_source_variant(envelope_label)
```

## Immediate selftest

```powershell
python E:\\BBIT\\outputs_basic32\\phase26en_lite_live_guard_selftest.py
```
"""
    path.write_text(text, encoding="utf-8")


def make_plots(outputs: Path, contract_rows: pd.DataFrame, alias_risk: pd.DataFrame, callsites: pd.DataFrame, strict_tangent: float, relaxed_tangent: float) -> None:
    if plt is None:
        return

    # Contract pass status dashboard.
    df = contract_rows.copy().sort_values("capture_margin", ascending=True)
    fig, ax = plt.subplots(figsize=(14, 7))
    y = list(range(len(df)))
    ax.barh([i - 0.18 for i in y], df["capture_margin"].astype(float), height=0.34, label="capture margin")
    ax.barh([i + 0.18 for i in y], df["strict_tangent_margin"].astype(float), height=0.34, label="strict tangent margin")
    ax.axvline(0.0, linewidth=1.4)
    ax.set_yticks(y)
    ax.set_yticklabels(df["envelope_label"].tolist())
    ax.set_xlabel("positive margin means contract-safe")
    ax.set_title("26EN-LITE exact-pair guard margins")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(outputs / "phase26en_lite_guard_margin_dashboard.png", dpi=140)
    plt.close(fig)

    # Alias risk isolation.
    adf = alias_risk.copy().sort_values("global_max_tangent", ascending=True)
    fig, ax = plt.subplots(figsize=(14, 7))
    y = list(range(len(adf)))
    ax.barh(y, adf["global_max_tangent"].astype(float), label="global source-only max tangent")
    ax.scatter(adf["exact_max_tangent"].astype(float), y, label="exact-pair max tangent")
    ax.axvline(strict_tangent, linewidth=1.4, label="strict tangent")
    ax.axvline(relaxed_tangent, linewidth=1.4, linestyle="--", label="relaxed tangent")
    ax.set_yticks(y)
    ax.set_yticklabels(adf["locked_envelope_label"].tolist())
    ax.set_xlabel("tangent/radial")
    ax.set_title("26EN-LITE alias risk isolated from exact-pair contract")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(outputs / "phase26en_lite_alias_isolation_dashboard.png", dpi=140)
    plt.close(fig)

    # Callsite candidates.
    if not callsites.empty:
        cdf = callsites.head(20).copy().sort_values("score", ascending=True)
        fig, ax = plt.subplots(figsize=(14, 8))
        y = list(range(len(cdf)))
        ax.barh(y, cdf["score"].astype(float))
        ax.set_yticks(y)
        ax.set_yticklabels(cdf["file"].tolist())
        ax.set_xlabel("route-selection relevance score")
        ax.set_title("26EN-LITE candidate transplant callsites")
        fig.tight_layout()
        fig.savefig(outputs / "phase26en_lite_callsite_candidates.png", dpi=140)
        plt.close(fig)


# ----------------------------------- main -----------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="26EN-LITE live contract guard / transplant planner")
    ap.add_argument("--outputs", default=None, help="outputs_basic32 directory")
    ap.add_argument("--source-root", default=None, help="bbit_geomlang source directory to scan")
    ap.add_argument("--max-files", type=int, default=600, help="max source files to scan")
    ap.add_argument("--no-plots", action="store_true", help="skip matplotlib plots")
    args = ap.parse_args(argv)

    outputs = resolve_outputs(args.outputs)
    source_root = resolve_source_root(args.source_root)
    outputs.mkdir(parents=True, exist_ok=True)

    print(f"[{PHASE}] live contract guard / transplant planner")
    print(f"[{PHASE}] outputs: {outputs}")
    print(f"[{PHASE}] source_root: {source_root}")

    em_summary, em_manifest, contract_rows, alias_risk = load_em_artifacts(outputs)
    route_map = extract_route_map(em_summary, em_manifest)
    cap_target, strict_tangent, relaxed_tangent = targets(em_summary, em_manifest)
    fingerprint = str(em_summary.get("route_fingerprint") or em_manifest.get("route_fingerprint") or route_fingerprint(route_map))

    missing_labels = [x for x in REQUIRED_LABELS if x not in route_map]
    extra_labels = [x for x in route_map if x not in REQUIRED_LABELS]

    contract_ready = bool(em_summary.get("CONTRACT_TRANSPLANT_READY"))
    exact_rows_pass = bool(contract_rows.get("contract_row_pass", pd.Series(dtype=bool)).all())
    alias_risk_count = int(alias_risk.get("global_source_only_would_fail", pd.Series(dtype=bool)).astype(bool).sum()) if not alias_risk.empty else 0

    guard_name = "phase26en_lite_live_contract_guard.py"
    selftest_name = "phase26en_lite_live_guard_selftest.py"
    guard_path = outputs / guard_name
    selftest_path = outputs / selftest_name

    write_live_guard(guard_path, route_map, fingerprint, cap_target, strict_tangent, relaxed_tangent)
    write_guard_selftest(selftest_path, guard_name, "phase26em_lite_contract_row_smoke.csv")

    selftest_rc, selftest_out, selftest_err = run_python(selftest_path)
    guard_selftest_pass = selftest_rc == 0

    callsites = scan_source_root(source_root, max_files=args.max_files)
    callsites_path = outputs / "phase26en_lite_callsite_candidates.csv"
    callsites.to_csv(callsites_path, index=False)

    summary = {
        "phase": PHASE,
        "LIVE_GUARD_READY": bool(contract_ready and exact_rows_pass and guard_selftest_pass and not missing_labels),
        "em_contract_ready": contract_ready,
        "exact_rows_pass": exact_rows_pass,
        "guard_selftest_pass": guard_selftest_pass,
        "guard_selftest_returncode": selftest_rc,
        "route_fingerprint": fingerprint,
        "computed_route_hash": route_fingerprint(route_map),
        "labels": len(route_map),
        "missing_required_labels": missing_labels,
        "extra_labels": extra_labels,
        "capture_target": cap_target,
        "strict_tangent_target": strict_tangent,
        "relaxed_tangent_target": relaxed_tangent,
        "verified_worst_capture": float(em_summary.get("verified_worst_capture", contract_rows["min_capture"].min())),
        "verified_worst_tangent": float(em_summary.get("verified_worst_tangent", contract_rows["max_tangent"].max())),
        "min_capture_margin": float(em_summary.get("min_capture_margin", contract_rows["capture_margin"].min())),
        "min_strict_tangent_margin": float(em_summary.get("min_strict_tangent_margin", contract_rows["strict_tangent_margin"].min())),
        "strict_pass_rate": float(em_summary.get("strict_pass_rate", 1.0 if exact_rows_pass else 0.0)),
        "relaxed_pass_rate": float(em_summary.get("relaxed_pass_rate", 1.0 if exact_rows_pass else 0.0)),
        "alias_risk_labels": alias_risk_count,
        "alias_risk_is_expected": alias_risk_count == len(route_map),
        "contract_key": ["envelope_label", "source_variant"],
        "guard_module": str(guard_path),
        "guard_selftest": str(selftest_path),
        "callsite_candidates": str(callsites_path),
        "top_callsite_candidates": callsites.head(10).to_dict("records") if not callsites.empty else [],
        "do_not_use": "source_variant-only global validation",
    }

    write_json(outputs / "phase26en_lite_summary.json", summary)
    write_json(outputs / "phase26en_lite_live_transplant_manifest.json", {
        "phase": PHASE,
        "status": "PASS" if summary["LIVE_GUARD_READY"] else "FAIL",
        "contract_key": ["envelope_label", "source_variant"],
        "route_fingerprint": fingerprint,
        "route_map": route_map,
        "files": {
            "guard": guard_name,
            "selftest": selftest_name,
            "codex_handoff": "phase26en_lite_codex_transplant_handoff.txt",
            "checklist": "phase26en_lite_live_transplant_checklist.md",
            "callsite_candidates": "phase26en_lite_callsite_candidates.csv",
        },
        "guardrails": [
            "Select by envelope_label, not global source_variant.",
            "Validate exact (envelope_label, source_variant) rows.",
            "Treat source-only smoke failures as alias diagnostics, not contract failures.",
        ],
    })

    write_codex_handoff(outputs / "phase26en_lite_codex_transplant_handoff.txt", guard_name, selftest_name, summary, callsites)
    write_checklist(outputs / "phase26en_lite_live_transplant_checklist.md", summary)

    if not args.no_plots:
        make_plots(outputs, contract_rows, alias_risk, callsites, strict_tangent, relaxed_tangent)

    print(f"[{PHASE}] LIVE_GUARD_READY={summary['LIVE_GUARD_READY']}")
    print(f"[{PHASE}] checks:")
    print(f"  - em_contract_ready: {contract_ready}")
    print(f"  - exact_rows_pass: {exact_rows_pass}")
    print(f"  - guard_selftest_pass: {guard_selftest_pass}")
    print(f"  - missing_required_labels: {missing_labels}")
    print(f"  - alias_risk_labels_expected: {alias_risk_count}/{len(route_map)}")
    print(f"[{PHASE}] locked route margins:")
    show_cols = ["envelope_label", "min_capture", "max_tangent", "capture_margin", "strict_tangent_margin", "contract_row_pass"]
    print(contract_rows[show_cols].to_string(index=False))
    if not callsites.empty:
        print(f"[{PHASE}] top source callsite candidates:")
        print(callsites[["file", "score", "line_hits", "suggestion"]].head(10).to_string(index=False))
    else:
        print(f"[{PHASE}] source callsite scan found no files or source_root did not exist")
    if not guard_selftest_pass:
        print(f"[{PHASE}] guard selftest stdout:\n{selftest_out}")
        print(f"[{PHASE}] guard selftest stderr:\n{selftest_err}")
    print(f"[{PHASE}] wrote guard: {guard_path}")
    print(f"[{PHASE}] wrote selftest: {selftest_path}")
    print(f"[{PHASE}] wrote Codex handoff: {outputs / 'phase26en_lite_codex_transplant_handoff.txt'}")
    print(f"[{PHASE}] wrote outputs to: {outputs}")
    return 0 if summary["LIVE_GUARD_READY"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
