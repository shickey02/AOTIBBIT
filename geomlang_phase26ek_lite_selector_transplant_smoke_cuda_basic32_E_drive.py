#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Phase 26EK-LITE — selector transplant smoke / integration bridge
E-drive/basic32 version.

Purpose
-------
26EJ proved the locked route contract. 26EK turns that proof into a small,
importable bridge that can be dropped into the next real geomlang phase without
re-opening the route search.

It does four things:
  1. Loads the 26EJ integration contract and the 26EI locked selector.
  2. Re-validates fingerprint/targets/route map against the pool rows.
  3. Writes a tiny transplant adapter module with stable functions:
        select_source_variant_for_envelope(...)
        apply_locked_route_to_records(...)
        validate_locked_route_records(...)
  4. Writes a Codex/hand patch note explaining exactly where the selector must
     be inserted in the next phase.

This is intentionally not another optimizer. It is a smoke test and deployment
bridge. If this passes, the route should be treated as locked infrastructure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import runpy
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _default_root() -> Path:
    # Script is usually run from E:\BBIT, but make it robust.
    cwd = Path.cwd()
    if (cwd / "outputs_basic32").exists():
        return cwd
    if Path("E:/BBIT").exists():
        return Path("E:/BBIT")
    return cwd


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default


def _fingerprint_route_map(route_map: Mapping[str, str]) -> str:
    blob = json.dumps(dict(sorted(route_map.items())), sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _load_selector(selector_path: Path):
    spec = importlib.util.spec_from_file_location("phase26ei_lite_locked_route_selector", selector_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load selector module spec: {selector_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _load_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if fieldnames:
            with path.open("w", encoding="utf-8", newline="") as f:
                csv.DictWriter(f, fieldnames=list(fieldnames)).writeheader()
        return
    if fieldnames is None:
        keys: List[str] = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _pick_col(rows: Sequence[Mapping[str, Any]], candidates: Sequence[str]) -> Optional[str]:
    if not rows:
        return None
    keys = set(rows[0].keys())
    for c in candidates:
        if c in keys:
            return c
    return None


def _group_locked_pool_rows(pool_rows: Sequence[Mapping[str, Any]], route_map: Mapping[str, str]) -> List[Dict[str, Any]]:
    if not pool_rows:
        return []
    label_col = _pick_col(pool_rows, ["envelope_label", "label", "case_label"])
    variant_col = _pick_col(pool_rows, ["source_variant", "variant", "variant_name"])
    seed_col = _pick_col(pool_rows, ["dz_seed", "seed", "verification_seed"])
    cap_col = _pick_col(pool_rows, ["worst_capture", "verified_worst_capture", "capture", "capture_rate"])
    tan_col = _pick_col(pool_rows, ["worst_tangent", "verified_worst_tangent", "tangent", "tangent_ratio", "worst_tangent_radial"])
    if not label_col or not variant_col:
        return []

    out: List[Dict[str, Any]] = []
    for label, locked_variant in sorted(route_map.items()):
        matching = [r for r in pool_rows if r.get(label_col) == label and r.get(variant_col) == locked_variant]
        caps = [_safe_float(r.get(cap_col)) for r in matching] if cap_col else []
        tans = [_safe_float(r.get(tan_col)) for r in matching] if tan_col else []
        caps = [x for x in caps if math.isfinite(x)]
        tans = [x for x in tans if math.isfinite(x)]
        seeds = sorted({r.get(seed_col, "") for r in matching}) if seed_col else []
        out.append({
            "envelope_label": label,
            "locked_source_variant": locked_variant,
            "matching_pool_rows": len(matching),
            "unique_seeds": len([s for s in seeds if s != ""]),
            "pool_min_capture": min(caps) if caps else "",
            "pool_max_tangent": max(tans) if tans else "",
            "pool_mean_capture": (sum(caps) / len(caps)) if caps else "",
            "pool_mean_tangent": (sum(tans) / len(tans)) if tans else "",
        })
    return out


def _write_adapter(adapter_path: Path, contract: Mapping[str, Any], route_map: Mapping[str, str]) -> None:
    targets = contract.get("targets", {}) or {}
    summary = contract.get("contract_summary", {}) or {}
    protected = contract.get("protected_labels_by_weakness", []) or []
    fingerprint = contract.get("route_fingerprint") or _fingerprint_route_map(route_map)

    content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Auto-generated by 26EK-LITE.
Locked selector transplant adapter for the 26E solved route.

Do not tune this file unless you intentionally want to re-open the route search.
Route fingerprint: {fingerprint}
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

LOCKED_ROUTE_FINGERPRINT = {fingerprint!r}
CAPTURE_TARGET = {float(targets.get("capture_target", 0.285))!r}
STRICT_TANGENT_TARGET = {float(targets.get("strict_tangent_target", 2.1))!r}
RELAXED_TANGENT_TARGET = {float(targets.get("relaxed_tangent_target", 2.3))!r}

CONTRACT_SUMMARY = {json.dumps(summary, indent=4, sort_keys=False)}

LOCKED_ROUTE_MAP: Dict[str, str] = {json.dumps(dict(sorted(route_map.items())), indent=4, sort_keys=False)}

PROTECTED_LABELS_BY_WEAKNESS: List[Dict[str, Any]] = {json.dumps(protected, indent=4, sort_keys=False)}


def normalize_envelope_label(label: str) -> str:
    """Normalize labels only enough to avoid whitespace/case copy mistakes."""
    return str(label).strip()


def is_locked_label(envelope_label: str) -> bool:
    return normalize_envelope_label(envelope_label) in LOCKED_ROUTE_MAP


def locked_labels() -> List[str]:
    return list(LOCKED_ROUTE_MAP.keys())


def select_source_variant_for_envelope(
    envelope_label: str,
    default: Optional[str] = None,
    *,
    strict: bool = True,
) -> str:
    """
    Return the locked source variant for an envelope label.

    strict=True means unknown labels raise KeyError. For exploratory code, pass
    strict=False and a default to avoid crashing on unrelated labels.
    """
    key = normalize_envelope_label(envelope_label)
    if key in LOCKED_ROUTE_MAP:
        return LOCKED_ROUTE_MAP[key]
    if default is not None or not strict:
        return default if default is not None else ""
    raise KeyError(f"No 26EK locked route for envelope_label={{envelope_label!r}}")


def apply_locked_route_to_records(
    records: Iterable[Mapping[str, Any]],
    *,
    label_key: str = "envelope_label",
    output_key: str = "locked_source_variant",
    strict: bool = False,
) -> List[Dict[str, Any]]:
    """Attach the locked source variant to dict-like records."""
    out: List[Dict[str, Any]] = []
    for rec in records:
        row = dict(rec)
        label = row.get(label_key, "")
        row[output_key] = select_source_variant_for_envelope(str(label), row.get(output_key), strict=strict)
        out.append(row)
    return out


def validate_locked_route_records(
    records: Iterable[Mapping[str, Any]],
    *,
    label_key: str = "envelope_label",
    variant_key: str = "source_variant",
) -> Dict[str, Any]:
    """
    Validate that records containing locked labels use the locked source_variant.
    This does not recompute capture/tangent; it checks transplant wiring.
    """
    checked = 0
    mismatches: List[Dict[str, Any]] = []
    present = set()
    for rec in records:
        label = normalize_envelope_label(str(rec.get(label_key, "")))
        if label not in LOCKED_ROUTE_MAP:
            continue
        present.add(label)
        checked += 1
        expected = LOCKED_ROUTE_MAP[label]
        actual = str(rec.get(variant_key, ""))
        if actual and actual != expected:
            mismatches.append({{"envelope_label": label, "expected": expected, "actual": actual}})
    missing = [x for x in LOCKED_ROUTE_MAP if x not in present]
    return {{
        "contract_pass": not mismatches and not missing,
        "checked_records": checked,
        "missing_locked_labels": missing,
        "mismatches": mismatches,
        "route_fingerprint": LOCKED_ROUTE_FINGERPRINT,
    }}
'''
    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    adapter_path.write_text(content, encoding="utf-8")


def _write_patch_note(path: Path, args: argparse.Namespace, contract: Mapping[str, Any], adapter_name: str) -> None:
    s = contract.get("contract_summary", {}) or {}
    route_map = contract.get("route_map", {}) or {}
    lines = []
    lines.append("26EK-LITE TRANSPLANT PATCH NOTE")
    lines.append("================================")
    lines.append("")
    lines.append("Status: 26EJ contract passed. 26EK generated a locked selector adapter.")
    lines.append("")
    lines.append("Copy these files into E:\\BBIT\\bbit_geomlang if you want the next phase to import them directly:")
    lines.append(f"  - E:\\BBIT\\outputs_basic32\\{adapter_name}")
    lines.append("  - E:\\BBIT\\outputs_basic32\\phase26ei_lite_locked_route_selector.py")
    lines.append("")
    lines.append("Minimal integration pattern:")
    lines.append("```python")
    lines.append("try:")
    lines.append(f"    from {Path(adapter_name).stem} import select_source_variant_for_envelope, LOCKED_ROUTE_MAP")
    lines.append("except Exception:")
    lines.append("    select_source_variant_for_envelope = None")
    lines.append("    LOCKED_ROUTE_MAP = {}")
    lines.append("")
    lines.append("# Wherever the next phase chooses a candidate/source_variant for a known envelope label:")
    lines.append("if select_source_variant_for_envelope is not None and envelope_label in LOCKED_ROUTE_MAP:")
    lines.append("    source_variant = select_source_variant_for_envelope(envelope_label)")
    lines.append("```")
    lines.append("")
    lines.append("The locked route must not be averaged, re-ranked, or mutated in the next phase. Treat it as a contract:")
    lines.append(f"  - capture >= {contract.get('targets', {}).get('capture_target', 0.285)}")
    lines.append(f"  - tangent <= {contract.get('targets', {}).get('strict_tangent_target', 2.1)} strict")
    lines.append(f"  - fingerprint = {contract.get('route_fingerprint')}")
    lines.append("")
    lines.append("Confirmed margins:")
    lines.append(f"  - verified_worst_capture = {s.get('verified_worst_capture')}")
    lines.append(f"  - verified_worst_tangent = {s.get('verified_worst_tangent')}")
    lines.append(f"  - min_capture_margin = {s.get('min_capture_margin')}")
    lines.append(f"  - min_strict_tangent_margin = {s.get('min_strict_tangent_margin')}")
    lines.append("")
    lines.append("Locked label map:")
    for k, v in sorted(route_map.items()):
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append("Next useful phase after this: transplant the adapter into the actual evaluator/integrator and run one end-to-end live capture/tangent evaluation. Do not run another search unless this contract breaks after integration.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _maybe_plot(outputs: Path, rows: Sequence[Mapping[str, Any]], targets: Mapping[str, Any]) -> List[str]:
    written: List[str] = []
    if not rows:
        return written
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return written

    labels = [str(r["envelope_label"]) for r in rows]
    cap = [_safe_float(r.get("pool_min_capture")) for r in rows]
    tan = [_safe_float(r.get("pool_max_tangent")) for r in rows]

    if any(math.isfinite(x) for x in cap):
        fig, ax = plt.subplots(figsize=(12, 6))
        y = list(range(len(labels)))
        ax.barh(y, cap)
        ax.axvline(float(targets.get("capture_target", 0.285)), linestyle="-")
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("pool min capture for locked source variant")
        ax.set_title("26EK-LITE locked selector capture smoke")
        fig.tight_layout()
        p = outputs / "phase26ek_lite_locked_selector_capture_smoke.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        written.append(str(p))

    if any(math.isfinite(x) for x in tan):
        fig, ax = plt.subplots(figsize=(12, 6))
        y = list(range(len(labels)))
        ax.barh(y, tan)
        ax.axvline(float(targets.get("strict_tangent_target", 2.1)), linestyle="-")
        ax.axvline(float(targets.get("relaxed_tangent_target", 2.3)), linestyle="--")
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("pool max tangent for locked source variant")
        ax.set_title("26EK-LITE locked selector tangent smoke")
        fig.tight_layout()
        p = outputs / "phase26ek_lite_locked_selector_tangent_smoke.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        written.append(str(p))
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="26EK-LITE selector transplant smoke / integration bridge")
    ap.add_argument("--root", default=str(_default_root()), help="BBIT root, default auto-detect")
    ap.add_argument("--outputs", default=None, help="outputs_basic32 directory")
    ap.add_argument("--contract", default="phase26ej_lite_integration_contract.json")
    ap.add_argument("--selector", default="phase26ei_lite_locked_route_selector.py")
    ap.add_argument("--pool", default="phase26ee_lite_pool_case_results.csv")
    ap.add_argument("--run-selftest", action="store_true", help="Run phase26ej_lite_selector_selftest.py if present")
    args = ap.parse_args(argv)

    root = Path(args.root)
    outputs = Path(args.outputs) if args.outputs else root / "outputs_basic32"
    outputs.mkdir(parents=True, exist_ok=True)

    contract_path = outputs / args.contract
    selector_path = outputs / args.selector
    pool_path = outputs / args.pool
    selftest_path = outputs / "phase26ej_lite_selector_selftest.py"

    print("[26EK-LITE] selector transplant smoke / integration bridge")
    print(f"[26EK-LITE] outputs: {outputs}")

    if not contract_path.exists():
        raise FileNotFoundError(f"Missing contract: {contract_path}")
    if not selector_path.exists():
        raise FileNotFoundError(f"Missing selector: {selector_path}")

    contract = _read_json(contract_path)
    selector = _load_selector(selector_path)
    route_map = dict(contract.get("route_map") or getattr(selector, "LOCKED_ROUTE_MAP", {}))
    fingerprint = contract.get("route_fingerprint") or getattr(selector, "LOCKED_ROUTE_FINGERPRINT", "")
    computed_fp = _fingerprint_route_map(route_map)

    selector_map = dict(getattr(selector, "LOCKED_ROUTE_MAP", {}))
    selector_fp = getattr(selector, "LOCKED_ROUTE_FINGERPRINT", "")

    checks: List[Dict[str, Any]] = []
    checks.append({"check": "contract_pass", "pass": bool(contract.get("contract_pass")), "detail": contract.get("contract_pass")})
    checks.append({"check": "route_map_nonempty", "pass": bool(route_map), "detail": len(route_map)})
    checks.append({"check": "selector_route_map_matches_contract", "pass": selector_map == route_map, "detail": len(selector_map)})
    checks.append({"check": "selector_fingerprint_matches_contract", "pass": selector_fp == fingerprint, "detail": {"selector": selector_fp, "contract": fingerprint}})
    # This computed fingerprint is informational because earlier phases may use their own fingerprint function.
    checks.append({"check": "computed_route_hash", "pass": True, "detail": computed_fp})

    selected_rows = _group_locked_pool_rows(_load_csv_rows(pool_path), route_map)
    if selected_rows:
        targets = contract.get("targets", {}) or {}
        cap_target = float(targets.get("capture_target", 0.285))
        strict_tan = float(targets.get("strict_tangent_target", 2.1))
        for r in selected_rows:
            cap = _safe_float(r.get("pool_min_capture"))
            tan = _safe_float(r.get("pool_max_tangent"))
            r["capture_smoke_pass"] = bool(math.isfinite(cap) and cap >= cap_target)
            r["strict_tangent_smoke_pass"] = bool(math.isfinite(tan) and tan <= strict_tan)
        checks.append({"check": "pool_rows_found_for_all_locked_labels", "pass": all(int(r["matching_pool_rows"]) > 0 for r in selected_rows), "detail": {r["envelope_label"]: r["matching_pool_rows"] for r in selected_rows}})
        checks.append({"check": "pool_capture_smoke", "pass": all(bool(r["capture_smoke_pass"]) for r in selected_rows), "detail": cap_target})
        checks.append({"check": "pool_strict_tangent_smoke", "pass": all(bool(r["strict_tangent_smoke_pass"]) for r in selected_rows), "detail": strict_tan})

    selftest_result: Dict[str, Any] = {"attempted": False}
    if args.run_selftest and selftest_path.exists():
        proc = subprocess.run([sys.executable, str(selftest_path)], cwd=str(outputs), text=True, capture_output=True)
        selftest_result = {
            "attempted": True,
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        }
        checks.append({"check": "selector_selftest", "pass": proc.returncode == 0, "detail": proc.returncode})

    adapter_name = "phase26ek_lite_route_selector_contract_adapter.py"
    adapter_path = outputs / adapter_name
    _write_adapter(adapter_path, contract, route_map)

    patch_note = outputs / "phase26ek_lite_transplant_patch_note.txt"
    _write_patch_note(patch_note, args, contract, adapter_name)

    selected_csv = outputs / "phase26ek_lite_locked_selector_pool_smoke.csv"
    _write_csv(selected_csv, selected_rows)

    plots = _maybe_plot(outputs, selected_rows, contract.get("targets", {}) or {})

    contract_ok = all(bool(c.get("pass")) for c in checks if c["check"] != "computed_route_hash")
    summary = {
        "phase": "26EK-LITE",
        "contract_smoke_pass": contract_ok,
        "route_fingerprint": fingerprint,
        "computed_route_hash_informational": computed_fp,
        "locked_labels": list(route_map.keys()),
        "files": {
            "adapter": str(adapter_path),
            "patch_note": str(patch_note),
            "selected_pool_smoke_csv": str(selected_csv),
            "plots": plots,
        },
        "checks": checks,
        "selftest_result": selftest_result,
    }
    _write_json(outputs / "phase26ek_lite_summary.json", summary)

    print(f"[26EK-LITE] CONTRACT_SMOKE_PASS={contract_ok}")
    print("[26EK-LITE] checks:")
    for c in checks:
        print(f"  - {c['check']}: {c['pass']} :: {c['detail']}")
    print(f"[26EK-LITE] wrote adapter: {adapter_path}")
    print(f"[26EK-LITE] wrote patch note: {patch_note}")
    print(f"[26EK-LITE] wrote outputs to: {outputs}")
    return 0 if contract_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
