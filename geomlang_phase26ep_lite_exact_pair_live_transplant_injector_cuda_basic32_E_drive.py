#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Phase 26EP-LITE — Exact-pair live transplant injector / shadow patch builder.

Purpose
-------
26EK proved that checking a locked `source_variant` globally is too broad.
26EL/EM/EN/EO proved the correct live contract key is the exact pair:

    (envelope_label, source_variant)

26EP does the next practical step: it prepares a guarded transplant into real
route/candidate-selection code while staying safe by default.

Default behavior is READ-ONLY:
  * loads the 26EO exact-pair helper
  * verifies the locked contract rows
  * scans E:\BBIT\bbit_geomlang for likely live selection boundaries
  * filters out known analysis/smoke/contract files unless --include-analysis is used
  * writes a ranked transplant report
  * writes a copy/paste patch kit
  * writes optional shadow-patched copies that DO NOT touch originals

Optional behavior:
  * --shadow-top N writes shadow copies of top N candidates with the EP guard shim inserted
  * --apply-to <filename.py> modifies one chosen file in-place, with a timestamped .bak backup

Recommended first run:

    python bbit_geomlang/geomlang_phase26ep_lite_exact_pair_live_transplant_injector_cuda_basic32_E_drive.py

Then inspect:

    E:\BBIT\outputs_basic32\phase26ep_lite_transplant_report.md
    E:\BBIT\outputs_basic32\phase26ep_lite_patch_kit.txt

If the chosen file is obvious, run a shadow copy first:

    python bbit_geomlang/geomlang_phase26ep_lite_exact_pair_live_transplant_injector_cuda_basic32_E_drive.py --shadow-top 5

Only after inspecting the shadow output, apply to one file:

    python bbit_geomlang/geomlang_phase26ep_lite_exact_pair_live_transplant_injector_cuda_basic32_E_drive.py --apply-to geomlang_phase26xx_some_real_phase.py
"""
from __future__ import annotations

import argparse
import ast
import csv
import datetime as _dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

PHASE = "26EP-LITE"
TITLE = "Exact-pair live transplant injector / shadow patch builder"

DEFAULT_ROOT = Path(r"E:\BBIT")
DEFAULT_SOURCE_ROOT = DEFAULT_ROOT / "bbit_geomlang"
DEFAULT_OUTPUTS = DEFAULT_ROOT / "outputs_basic32"

CAPTURE_TARGET = 0.285
STRICT_TANGENT_TARGET = 2.1
RELAXED_TANGENT_TARGET = 2.3
EXPECTED_FINGERPRINT = "98ebdcbb8e995bc1"

ANALYSIS_NAME_HINTS = (
    "transplant", "contract", "guard", "smoke", "planner", "patcher",
    "handoff", "validator", "hardener", "exporter", "confirm", "analyzer",
    "feasibility", "selftest", "summary", "report",
)

STRONG_BOUNDARY_PATTERNS = (
    "source_variant",
    "envelope_label",
    "route_map",
    "select_source_variant",
    "variant_summary",
    "variant_name",
    "locked_route",
    "label_router",
    "candidate",
)

ASSIGN_RE = re.compile(r"^(?P<indent>\s*)(?P<name>source_variant|variant_name|chosen_variant|selected_variant)\s*=\s*(?P<rhs>.+)$")
ROW_APPEND_RE = re.compile(r"^(?P<indent>\s*)(?P<name>rows|records|out_rows|result_rows)\.append\((?P<row>.+)\)\s*$")
DICT_SOURCE_RE = re.compile(r"(['\"]source_variant['\"]\s*:\s*)(?P<value>[^,}\n]+)")


def now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def stable_hash_text(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:n]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8", newline="\n")


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    keys: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def load_module_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def locate_outputs(user_outputs: Optional[str]) -> Path:
    if user_outputs:
        return Path(user_outputs)
    env = os.environ.get("BBIT_OUTPUTS_BASIC32")
    if env:
        return Path(env)
    return DEFAULT_OUTPUTS


def locate_source_root(user_source: Optional[str]) -> Path:
    if user_source:
        return Path(user_source)
    env = os.environ.get("BBIT_GEOMLANG_ROOT")
    if env:
        return Path(env)
    return DEFAULT_SOURCE_ROOT


def load_eo_helper(outputs: Path):
    helper_path = outputs / "phase26eo_lite_exact_pair_route_helper.py"
    if not helper_path.exists():
        fallback = outputs / "phase26em_lite_runtime_route_router.py"
        if fallback.exists():
            helper_path = fallback
        else:
            raise FileNotFoundError(
                f"Missing 26EO helper: {helper_path}. Run 26EO first or pass --outputs."
            )
    mod = load_module_from_path(helper_path, "phase26ep_loaded_exact_pair_helper")
    return helper_path, mod


def helper_contract_rows(helper: Any) -> List[Dict[str, Any]]:
    rows = []
    route_map = dict(getattr(helper, "LOCKED_ROUTE_MAP"))
    floors = dict(getattr(helper, "CONTRACT_ROW_FLOORS"))
    for label, variant in route_map.items():
        floor = floors.get(label, {})
        min_capture = float(floor.get("min_capture", 0.0))
        max_tangent = float(floor.get("max_tangent", 999.0))
        rows.append({
            "envelope_label": label,
            "source_variant": variant,
            "min_capture": min_capture,
            "max_tangent": max_tangent,
            "capture_margin": min_capture - CAPTURE_TARGET,
            "strict_tangent_margin": STRICT_TANGENT_TARGET - max_tangent,
            "relaxed_tangent_margin": RELAXED_TANGENT_TARGET - max_tangent,
            "contract_row_pass": bool(min_capture >= CAPTURE_TARGET and max_tangent <= STRICT_TANGENT_TARGET),
        })
    return rows


def summarize_contract(helper: Any) -> Dict[str, Any]:
    rows = helper_contract_rows(helper)
    route_map = dict(getattr(helper, "LOCKED_ROUTE_MAP"))
    fingerprint = str(getattr(helper, "LOCKED_ROUTE_FINGERPRINT", ""))
    failures = [r for r in rows if not r["contract_row_pass"]]
    return {
        "fingerprint": fingerprint,
        "fingerprint_expected": EXPECTED_FINGERPRINT,
        "fingerprint_pass": fingerprint == EXPECTED_FINGERPRINT,
        "labels": list(route_map.keys()),
        "label_count": len(route_map),
        "contract_rows_pass": not failures,
        "contract_failures": failures,
        "min_capture_margin": min(r["capture_margin"] for r in rows) if rows else None,
        "min_strict_tangent_margin": min(r["strict_tangent_margin"] for r in rows) if rows else None,
        "capture_blocker_mode": min(rows, key=lambda r: r["capture_margin"])["envelope_label"] if rows else None,
        "tangent_blocker_mode": min(rows, key=lambda r: r["strict_tangent_margin"])["envelope_label"] if rows else None,
        "ready": bool(fingerprint == EXPECTED_FINGERPRINT and not failures and len(route_map) == 7),
    }


def ast_signals(text: str) -> Dict[str, int]:
    signals = {
        "functions": 0,
        "classes": 0,
        "assign_source_variant": 0,
        "assign_envelope_label": 0,
        "dict_source_variant": 0,
        "dict_envelope_label": 0,
    }
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return signals
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            signals["functions"] += 1
        elif isinstance(node, ast.ClassDef):
            signals["classes"] += 1
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id == "source_variant":
                        signals["assign_source_variant"] += 1
                    if target.id == "envelope_label":
                        signals["assign_envelope_label"] += 1
        elif isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and key.value == "source_variant":
                    signals["dict_source_variant"] += 1
                if isinstance(key, ast.Constant) and key.value == "envelope_label":
                    signals["dict_envelope_label"] += 1
    return signals


def scan_file(path: Path, source_root: Path, include_analysis: bool = False) -> Dict[str, Any]:
    rel = str(path.relative_to(source_root)) if path.is_relative_to(source_root) else path.name
    name = path.name
    text = read_text(path)
    lower_name = name.lower()
    generated_analysis_like = any(h in lower_name for h in ANALYSIS_NAME_HINTS)
    already_guarded = (
        "phase26eo_select_source_variant" in text
        or "phase26ep_select_source_variant" in text
        or "phase26eo_lite_exact_pair_route_helper" in text
        or "locked_route_contract" in text and "source_variant" in text and "envelope_label" in text
    )
    line_hits: List[int] = []
    local_score = 0
    for i, line in enumerate(text.splitlines(), start=1):
        l = line.lower()
        hits = sum(1 for p in STRONG_BOUNDARY_PATTERNS if p in l)
        if hits:
            line_hits.append(i)
            local_score += hits * 8
        if ASSIGN_RE.match(line):
            local_score += 50
        if ROW_APPEND_RE.match(line):
            local_score += 15
        if "for" in l and "variant" in l and "label" in l:
            local_score += 20
        if "groupby" in l and "envelope_label" in l:
            local_score += 25
        if "sort_values" in l and "score" in l:
            local_score += 10
    sig = ast_signals(text)
    local_score += sig["assign_source_variant"] * 60
    local_score += sig["dict_source_variant"] * 30
    local_score += sig["dict_envelope_label"] * 30
    local_score += sig["assign_envelope_label"] * 25
    if already_guarded:
        local_score -= 125
    if generated_analysis_like and not include_analysis:
        local_score -= 250
    return {
        "file": name,
        "relative_path": rel,
        "path": str(path),
        "bytes": len(text.encode("utf-8", errors="replace")),
        "sha16": stable_hash_text(text),
        "generated_analysis_like": generated_analysis_like,
        "already_guarded": already_guarded,
        "line_hits": ";".join(map(str, line_hits[:25])),
        "line_hit_count": len(line_hits),
        "assign_source_variant": sig["assign_source_variant"],
        "dict_source_variant": sig["dict_source_variant"],
        "dict_envelope_label": sig["dict_envelope_label"],
        "score": local_score,
    }


def scan_source_root(source_root: Path, include_analysis: bool = False) -> List[Dict[str, Any]]:
    if not source_root.exists():
        raise FileNotFoundError(f"source_root does not exist: {source_root}")
    rows = []
    for path in sorted(source_root.glob("geomlang_phase*.py")):
        try:
            row = scan_file(path, source_root, include_analysis=include_analysis)
            if row["score"] > 0 or row["line_hit_count"] > 0:
                rows.append(row)
        except Exception as e:
            rows.append({"file": path.name, "path": str(path), "score": -999, "error": repr(e)})
    rows.sort(key=lambda r: (int(r.get("score", 0)), int(r.get("line_hit_count", 0))), reverse=True)
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
    return rows


def guard_import_block(outputs: Path) -> str:
    outputs_s = str(outputs).replace("\\", "\\\\")
    return f'''
# --- 26EP exact-pair locked-route guard BEGIN ---
# Contract key: (envelope_label, source_variant). Never validate source_variant globally.
import sys as _phase26ep_sys
from pathlib import Path as _phase26ep_Path
_phase26ep_outputs = _phase26ep_Path(r"{outputs_s}")
if str(_phase26ep_outputs) not in _phase26ep_sys.path:
    _phase26ep_sys.path.insert(0, str(_phase26ep_outputs))
try:
    from phase26eo_lite_exact_pair_route_helper import (
        LOCKED_ROUTE_FINGERPRINT as PHASE26EP_LOCKED_ROUTE_FINGERPRINT,
        CONTRACT_NAME as PHASE26EP_LOCKED_ROUTE_CONTRACT,
        select_source_variant as phase26ep_select_source_variant,
        apply_to_record as phase26ep_apply_to_record,
        validate_exact_pairs as phase26ep_validate_exact_pairs,
        validate_contract_rows as phase26ep_validate_contract_rows,
    )
except Exception as _phase26ep_import_error:
    raise RuntimeError(
        "26EP exact-pair route guard could not import phase26eo_lite_exact_pair_route_helper.py. "
        "Run 26EO first and keep outputs_basic32 on disk."
    ) from _phase26ep_import_error
# --- 26EP exact-pair locked-route guard END ---
'''.strip("\n")


def insertion_index(lines: List[str]) -> int:
    """Find a conservative place after shebang/encoding/docstring/from __future__."""
    idx = 0
    if lines and lines[0].startswith("#!"):
        idx = 1
    while idx < len(lines) and ("coding" in lines[idx] or lines[idx].strip() == ""):
        idx += 1
    # Skip module docstring if present.
    if idx < len(lines) and lines[idx].lstrip().startswith(("'''", '"""')):
        quote = "'''" if lines[idx].lstrip().startswith("'''") else '"""'
        if lines[idx].count(quote) >= 2 and len(lines[idx].strip()) > 3:
            idx += 1
        else:
            idx += 1
            while idx < len(lines):
                if quote in lines[idx]:
                    idx += 1
                    break
                idx += 1
    # Keep from __future__ imports first.
    while idx < len(lines) and (lines[idx].startswith("from __future__") or lines[idx].strip() == ""):
        idx += 1
    return idx


def guarded_source_text(original: str, outputs: Path, *, aggressive: bool = False) -> Tuple[str, Dict[str, Any]]:
    if "26EP exact-pair locked-route guard BEGIN" in original or "phase26ep_select_source_variant" in original:
        return original, {"changed": False, "reason": "already contains EP guard"}
    lines = original.splitlines()
    idx = insertion_index(lines)
    block = guard_import_block(outputs).splitlines()
    new_lines = lines[:idx] + [""] + block + [""] + lines[idx:]
    replacements = 0

    if aggressive:
        patched: List[str] = []
        for line in new_lines:
            m = ASSIGN_RE.match(line)
            if m and "phase26ep_select_source_variant" not in line:
                indent, name, rhs = m.group("indent"), m.group("name"), m.group("rhs")
                patched.append(line)
                patched.append(
                    f"{indent}{name} = phase26ep_select_source_variant(envelope_label, default={name}, strict=False)  # 26EP exact-pair guard"
                )
                replacements += 1
            else:
                patched.append(line)
        new_lines = patched

    return "\n".join(new_lines) + "\n", {"changed": True, "insert_index": idx + 1, "aggressive_replacements": replacements}


def write_shadow_patch(path: Path, source_root: Path, outputs: Path, shadow_dir: Path, aggressive: bool = False) -> Dict[str, Any]:
    text = read_text(path)
    patched, meta = guarded_source_text(text, outputs, aggressive=aggressive)
    rel = path.relative_to(source_root) if path.is_relative_to(source_root) else Path(path.name)
    out_path = shadow_dir / rel
    write_text(out_path, patched)
    meta.update({
        "source_file": str(path),
        "shadow_file": str(out_path),
        "source_sha16": stable_hash_text(text),
        "shadow_sha16": stable_hash_text(patched),
    })
    return meta


def apply_patch_in_place(path: Path, outputs: Path, aggressive: bool = False) -> Dict[str, Any]:
    text = read_text(path)
    patched, meta = guarded_source_text(text, outputs, aggressive=aggressive)
    if not meta.get("changed"):
        return {"file": str(path), **meta}
    backup = path.with_suffix(path.suffix + f".phase26ep_{now_stamp()}.bak")
    shutil.copy2(path, backup)
    write_text(path, patched)
    meta.update({"file": str(path), "backup": str(backup), "source_sha16": stable_hash_text(text), "patched_sha16": stable_hash_text(patched)})
    return meta


def build_patch_kit(outputs: Path, top_rows: Sequence[Dict[str, Any]]) -> str:
    block = guard_import_block(outputs)
    top_table = "\n".join(
        f"{r.get('rank', '')}. {r.get('file')}  score={r.get('score')}  hits={r.get('line_hits')}"
        for r in top_rows[:15]
    )
    return f"""26EP-LITE exact-pair transplant patch kit
============================================

The locked route is contract-safe only when selected by exact pair:

    envelope_label + source_variant

Do not validate a source variant globally. 26EK failed because it treated the
source as globally safe; 26EL/EM/EN/EO proved the exact contract row is safe.

1) Import guard block
---------------------
Paste this near the top of the target phase, after `from __future__` imports:

```python
{block}
```

2) Candidate-selection boundary
-------------------------------
At the point where a candidate/source variant is chosen for a label, force the
locked exact-pair route:

```python
source_variant = phase26ep_select_source_variant(
    envelope_label,
    default=source_variant,
    strict=False,
)
```

For a strict locked-route-only section:

```python
source_variant = phase26ep_select_source_variant(envelope_label, strict=True)
```

3) Exported rows
----------------
Before appending or writing result rows, stamp the row:

```python
row = phase26ep_apply_to_record(
    row,
    label_key="envelope_label",
    variant_key="source_variant",
    strict=False,
)
```

The resulting rows should contain:

```text
locked_route_selected
locked_route_fingerprint
locked_route_contract
locked_route_key
```

4) Top current callsite candidates
----------------------------------
{top_table}

5) Post-transplant acceptance
-----------------------------
Run the target phase, then verify:

```text
fingerprint == 98ebdcbb8e995bc1
min_capture >= 0.285
max_tangent <= 2.1
all 7 envelope labels present
source_variant is checked only with envelope_label
```
"""


def build_report(contract: Dict[str, Any], rows: Sequence[Dict[str, Any]], outputs: Path, source_root: Path) -> str:
    def yn(v: Any) -> str:
        return "PASS" if v else "FAIL"
    top = rows[:20]
    table = ["| rank | file | score | guarded | analysis-like | hits |", "|---:|---|---:|---:|---:|---|"]
    for r in top:
        table.append(
            f"| {r.get('rank')} | `{r.get('file')}` | {r.get('score')} | {r.get('already_guarded')} | {r.get('generated_analysis_like')} | `{r.get('line_hits')}` |"
        )
    return f"""# 26EP-LITE exact-pair live transplant report

## Contract status

- overall ready: **{yn(contract.get('ready'))}**
- fingerprint: `{contract.get('fingerprint')}`
- fingerprint pass: **{yn(contract.get('fingerprint_pass'))}**
- contract rows pass: **{yn(contract.get('contract_rows_pass'))}**
- labels: `{', '.join(contract.get('labels', []))}`
- weakest capture label: `{contract.get('capture_blocker_mode')}`
- weakest tangent label: `{contract.get('tangent_blocker_mode')}`
- min capture margin: `{contract.get('min_capture_margin')}`
- min strict tangent margin: `{contract.get('min_strict_tangent_margin')}`

## Source scan

- source root: `{source_root}`
- outputs: `{outputs}`
- total candidate files: `{len(rows)}`

## Ranked candidates

{chr(10).join(table)}

## Recommendation

First inspect the top-ranked file that is not merely an analysis/contract/smoke file.
Use `--shadow-top 5` to generate safe copies with the import guard inserted.
Use `--apply-to <filename.py>` only after inspecting the shadow copy.
"""


def plot_candidates(rows: Sequence[Dict[str, Any]], out_path: Path) -> bool:
    if plt is None or not rows:
        return False
    top = list(rows[:20])[::-1]
    labels = [r["file"] for r in top]
    scores = [float(r.get("score", 0)) for r in top]
    fig_h = max(6, 0.42 * len(top))
    plt.figure(figsize=(14, fig_h))
    plt.barh(labels, scores)
    plt.xlabel("route-boundary relevance score")
    plt.title("26EP-LITE live transplant callsite candidates")
    plt.tight_layout()
    ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"{PHASE}: {TITLE}")
    p.add_argument("--outputs", default=None, help="outputs_basic32 directory")
    p.add_argument("--source-root", default=None, help="bbit_geomlang source directory")
    p.add_argument("--include-analysis", action="store_true", help="do not penalize generated analyzer/smoke/contract files")
    p.add_argument("--shadow-top", type=int, default=0, help="write shadow-patched copies of top N candidates")
    p.add_argument("--apply-to", default=None, help="modify one filename/path in-place, with backup")
    p.add_argument("--aggressive", action="store_true", help="after insertion, also add guard lines after source_variant assignments")
    return p.parse_args(argv)


def resolve_apply_target(apply_to: str, source_root: Path) -> Path:
    p = Path(apply_to)
    if p.exists():
        return p
    p2 = source_root / apply_to
    if p2.exists():
        return p2
    matches = list(source_root.glob(f"**/{apply_to}"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Could not find --apply-to target: {apply_to}")
    raise RuntimeError(f"Ambiguous --apply-to target {apply_to}: {matches[:5]}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    outputs = locate_outputs(args.outputs)
    source_root = locate_source_root(args.source_root)
    ensure_dir(outputs)

    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] outputs: {outputs}")
    print(f"[{PHASE}] source_root: {source_root}")

    helper_path, helper = load_eo_helper(outputs)
    print(f"[{PHASE}] loaded helper: {helper_path.name}")

    contract = summarize_contract(helper)
    print(f"[{PHASE}] CONTRACT_READY={contract['ready']}")
    print(f"[{PHASE}] fingerprint={contract['fingerprint']} rows_pass={contract['contract_rows_pass']}")

    rows = scan_source_root(source_root, include_analysis=args.include_analysis)

    # Write contract rows and candidate outputs.
    contract_rows = helper_contract_rows(helper)
    write_csv(outputs / "phase26ep_lite_contract_rows.csv", contract_rows)
    write_csv(outputs / "phase26ep_lite_ranked_callsites.csv", rows)
    write_json(outputs / "phase26ep_lite_summary.json", {
        "phase": PHASE,
        "title": TITLE,
        "outputs": str(outputs),
        "source_root": str(source_root),
        "helper_path": str(helper_path),
        "contract": contract,
        "top_candidates": rows[:20],
        "include_analysis": args.include_analysis,
        "shadow_top": args.shadow_top,
        "apply_to": args.apply_to,
        "aggressive": args.aggressive,
    })

    report = build_report(contract, rows, outputs, source_root)
    write_text(outputs / "phase26ep_lite_transplant_report.md", report)
    patch_kit = build_patch_kit(outputs, rows[:20])
    write_text(outputs / "phase26ep_lite_patch_kit.txt", patch_kit)
    plot_candidates(rows, outputs / "phase26ep_lite_callsite_candidates.png")

    shadow_results: List[Dict[str, Any]] = []
    if args.shadow_top > 0:
        shadow_dir = outputs / "phase26ep_lite_shadow_patches"
        ensure_dir(shadow_dir)
        for r in rows[: args.shadow_top]:
            p = Path(str(r["path"]))
            if p.exists():
                shadow_results.append(write_shadow_patch(p, source_root, outputs, shadow_dir, aggressive=args.aggressive))
        write_json(outputs / "phase26ep_lite_shadow_patch_manifest.json", shadow_results)
        print(f"[{PHASE}] wrote shadow patches: {shadow_dir} count={len(shadow_results)}")

    apply_result: Optional[Dict[str, Any]] = None
    if args.apply_to:
        target = resolve_apply_target(args.apply_to, source_root)
        apply_result = apply_patch_in_place(target, outputs, aggressive=args.aggressive)
        write_json(outputs / "phase26ep_lite_apply_result.json", apply_result)
        print(f"[{PHASE}] applied patch target: {target}")
        if "backup" in apply_result:
            print(f"[{PHASE}] backup: {apply_result['backup']}")

    print(f"[{PHASE}] top live callsite candidates:")
    for r in rows[:10]:
        print(
            f"  {r['rank']:>2}. {r['file']} score={r['score']} "
            f"guarded={r['already_guarded']} analysis_like={r['generated_analysis_like']} hits={r['line_hits']}"
        )

    print(f"[{PHASE}] wrote report: {outputs / 'phase26ep_lite_transplant_report.md'}")
    print(f"[{PHASE}] wrote patch kit: {outputs / 'phase26ep_lite_patch_kit.txt'}")
    print(f"[{PHASE}] wrote outputs to: {outputs}")
    return 0 if contract["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
