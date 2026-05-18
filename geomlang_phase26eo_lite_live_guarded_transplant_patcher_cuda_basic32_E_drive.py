#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
26EO-LITE live guarded transplant patcher / dry-run injector.

Purpose
-------
26EN proved the important thing: the locked route is safe only as an exact pair:

    envelope_label + source_variant

26EK failed because it smoked source_variant globally. 26EL/26EM/26EN fixed that by
forcing label-aware exact-pair routing and exact-row validation.

26EO is the bridge from analysis files into a real next geomlang phase. It does NOT
blindly mutate your working phase by default. It:

1. Loads the 26EN live guard and verifies the guard selftest.
2. Loads the EN callsite scan and ranks candidate files.
3. Writes a transplant helper module that can be imported by the next real phase.
4. Writes a per-candidate patch plan with exact import/helper snippets.
5. Optionally applies a small safe scaffold to one target file when --apply is used.
6. Verifies that the scaffold still selects the seven locked labels by exact pair.

Default mode is dry-run.

Typical command
---------------
    python E:\BBIT\bbit_geomlang\geomlang_phase26eo_lite_live_guarded_transplant_patcher_cuda_basic32_E_drive.py

Optional targeted dry-run
-------------------------
    python E:\BBIT\bbit_geomlang\geomlang_phase26eo_lite_live_guarded_transplant_patcher_cuda_basic32_E_drive.py ^
      --target E:\BBIT\bbit_geomlang\geomlang_phase26dz_lite_label_routed_envelope_verifier_cuda_basic32_E_drive.py

Optional apply scaffold only
----------------------------
    python E:\BBIT\bbit_geomlang\geomlang_phase26eo_lite_live_guarded_transplant_patcher_cuda_basic32_E_drive.py ^
      --target E:\BBIT\bbit_geomlang\NEXT_REAL_PHASE.py --apply

The --apply operation only inserts an import/helper scaffold. It does not rewrite your
route logic automatically. After insertion, replace the old source-variant selection
boundary with:

    source_variant = phase26eo_select_source_variant(envelope_label, default=source_variant, strict=False)

or for locked-only sections:

    source_variant = phase26eo_select_source_variant(envelope_label, strict=True)
"""

from __future__ import annotations
# --- 26EQ exact-pair locked-route guard BEGIN ---
# Contract key: (envelope_label, source_variant). Never validate source_variant globally.
import sys as _phase26eq_sys
from pathlib import Path as _phase26eq_Path
_phase26eq_outputs = _phase26eq_Path(r"E:\BBIT\outputs_basic32")
if str(_phase26eq_outputs) not in _phase26eq_sys.path:
    _phase26eq_sys.path.insert(0, str(_phase26eq_outputs))
try:
    from phase26eo_lite_exact_pair_route_helper import (
        LOCKED_ROUTE_FINGERPRINT as PHASE26EQ_LOCKED_ROUTE_FINGERPRINT,
        CONTRACT_NAME as PHASE26EQ_LOCKED_ROUTE_CONTRACT,
        select_source_variant as phase26eq_select_source_variant,
        apply_to_record as phase26eq_apply_to_record,
        apply_to_records as phase26eq_apply_to_records,
        validate_exact_pairs as phase26eq_validate_exact_pairs,
        validate_contract_rows as phase26eq_validate_contract_rows,
        locked_labels as phase26eq_locked_labels,
        is_locked_label as phase26eq_is_locked_label,
    )
except Exception as _phase26eq_import_error:
    raise RuntimeError(
        "26EQ exact-pair route guard could not import phase26eo_lite_exact_pair_route_helper.py. "
        "Run 26EO/26EP first and keep outputs_basic32 on disk."
    ) from _phase26eq_import_error
# --- 26EQ exact-pair locked-route guard END ---

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PHASE = "26EO-LITE"
TITLE = "Live guarded exact-pair transplant patcher"

DEFAULT_SOURCE_ROOT = Path(r"E:\BBIT\bbit_geomlang")
DEFAULT_OUTPUTS = Path(r"E:\BBIT\outputs_basic32")

CAPTURE_TARGET = 0.285
STRICT_TANGENT_TARGET = 2.1
RELAXED_TANGENT_TARGET = 2.3
EXPECTED_ROUTE_FINGERPRINT = "98ebdcbb8e995bc1"

LOCKED_ROUTE_MAP: Dict[str, str] = {
    "base": "prior_phase26dv_lite_variant_summary_dv_capture_cloud_013",
    "screen_medium_00": "prior_phase26dy_lite_variant_summary_dy_terminal_cloud_009",
    "screen_small_00": "prior_phase26dy_lite_variant_summary_dy_screen_small_r0.722_cap1.195_tk1.015",
    "stress_cap_tk": "ea_jitter_ea_anchor_dx_009",
    "stress_radius_in": "ea_jitter_ea_anchor_dx_008",
    "stress_seat_down": "prior_phase26dx_lite_variant_summary_dx_grid_c0_r0.715_s0.415_b1.005_c1.16_sh0.168_tk1.000",
    "stress_shell_blend": "prior_phase26dy_lite_variant_summary_dy_screen_small_r0.722_cap1.195_tk1.015",
}

CONTRACT_ROW_FLOORS: Dict[str, Dict[str, float]] = {
    "base": {"min_capture": 0.468750, "max_tangent": 2.034203052520752},
    "screen_medium_00": {"min_capture": 0.328125, "max_tangent": 1.617377},
    "screen_small_00": {"min_capture": 0.328125, "max_tangent": 1.503680},
    "stress_cap_tk": {"min_capture": 0.421875, "max_tangent": 1.135642},
    "stress_radius_in": {"min_capture": 0.468750, "max_tangent": 1.699199},
    "stress_seat_down": {"min_capture": 0.390625, "max_tangent": 1.990724},
    "stress_shell_blend": {"min_capture": 0.390625, "max_tangent": 1.981300},
}

IMPORT_SENTINEL = "# --- 26EO-LITE EXACT-PAIR ROUTE GUARD: BEGIN ---"
IMPORT_SENTINEL_END = "# --- 26EO-LITE EXACT-PAIR ROUTE GUARD: END ---"

HELPER_MODULE_NAME = "phase26eo_lite_exact_pair_route_helper.py"
SELFTEST_NAME = "phase26eo_lite_transplant_selftest.py"
PATCH_PLAN_NAME = "phase26eo_lite_transplant_patch_plan.md"
PATCH_JSON_NAME = "phase26eo_lite_transplant_patch_plan.json"
SUMMARY_NAME = "phase26eo_lite_summary.json"
CALLSITE_OUT_NAME = "phase26eo_lite_ranked_callsites.csv"


@dataclass
class CandidateSite:
    file: str
    path: str
    score: int
    line_hits: str
    envelope_label_hits: int = 0
    source_variant_hits: int = 0
    route_map_hits: int = 0
    evaluate_field_hits: int = 0
    exists: bool = False
    already_guarded: bool = False
    likely_boundary_score: int = 0
    recommendation: str = "inspect route/candidate selection boundary"


@dataclass
class EOResult:
    phase: str
    title: str
    outputs: str
    source_root: str
    live_guard_ready: bool
    guard_selftest_pass: bool
    helper_written: str
    selftest_written: str
    patch_plan_written: str
    patch_json_written: str
    ranked_callsites_written: str
    target: Optional[str]
    apply: bool
    scaffold_applied: bool
    scaffold_backup: Optional[str]
    contract_exact_pair_pass: bool
    route_fingerprint: str
    top_candidates: List[Dict[str, Any]]
    next_action: str


def pwin(path: Path) -> str:
    return str(path)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8", newline="\n")


def sha16_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def import_module_from_path(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def run_selftest(path: Path) -> Tuple[bool, str]:
    if not path.exists():
        return False, f"missing selftest: {path}"
    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(path.parent),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=60,
        )
        return proc.returncode == 0, proc.stdout[-4000:]
    except Exception as exc:
        return False, f"selftest exception: {exc!r}"


def load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(read_text(path))
    except Exception:
        return {}


def resolve_outputs_and_root(args: argparse.Namespace) -> Tuple[Path, Path]:
    outputs = Path(args.outputs) if args.outputs else DEFAULT_OUTPUTS
    source_root = Path(args.source_root) if args.source_root else DEFAULT_SOURCE_ROOT
    return outputs, source_root


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def load_callsite_candidates(outputs: Path, source_root: Path) -> List[CandidateSite]:
    csv_path = outputs / "phase26en_lite_callsite_candidates.csv"
    candidates: List[CandidateSite] = []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_name = row.get("file", "")
                raw_path = row.get("path") or str(source_root / file_name)
                path = Path(raw_path)
                if not path.exists() and file_name:
                    path = source_root / file_name
                text = read_text(path) if path.exists() else ""
                already = IMPORT_SENTINEL in text or "phase26eo_select_source_variant" in text
                likely = score_likely_boundary(text, row.get("line_hits", "")) if text else 0
                candidates.append(
                    CandidateSite(
                        file=file_name,
                        path=str(path),
                        score=parse_int(row.get("score")),
                        line_hits=row.get("line_hits", ""),
                        envelope_label_hits=parse_int(row.get("envelope_label_hits")),
                        source_variant_hits=parse_int(row.get("source_variant_hits")),
                        route_map_hits=parse_int(row.get("route_map_hits")),
                        evaluate_field_hits=parse_int(row.get("evaluate_field_hits")),
                        exists=path.exists(),
                        already_guarded=already,
                        likely_boundary_score=likely,
                    )
                )
    else:
        # Fallback scan: useful if EN candidate CSV was not copied to outputs.
        patterns = ["envelope_label", "source_variant", "route_map", "select_source_variant"]
        for path in sorted(source_root.glob("geomlang_phase26*.py")):
            text = read_text(path)
            hits = {pat: len(re.findall(re.escape(pat), text)) for pat in patterns}
            if not any(hits.values()):
                continue
            score = hits["envelope_label"] * 8 + hits["source_variant"] * 7 + hits["route_map"] * 5
            candidates.append(
                CandidateSite(
                    file=path.name,
                    path=str(path),
                    score=score,
                    line_hits="",
                    envelope_label_hits=hits["envelope_label"],
                    source_variant_hits=hits["source_variant"],
                    route_map_hits=hits["route_map"],
                    exists=True,
                    already_guarded=(IMPORT_SENTINEL in text),
                    likely_boundary_score=score_likely_boundary(text, ""),
                )
            )
    candidates.sort(key=lambda c: (c.exists, c.score + c.likely_boundary_score, -int(c.already_guarded)), reverse=True)
    return candidates


def score_likely_boundary(text: str, line_hits: str) -> int:
    score = 0
    boundary_terms = [
        "candidate", "route", "select", "variant", "source_variant", "envelope_label",
        "best", "pareto", "locked", "contract", "validate", "filter",
    ]
    for term in boundary_terms:
        score += text.count(term) // 5
    for frag in ["for label", "for envelope", "groupby", "sort_values", "idxmax", "idxmin", "best_route"]:
        if frag in text:
            score += 10
    for line_s in str(line_hits).split(";"):
        try:
            idx = int(line_s) - 1
        except Exception:
            continue
        lines = text.splitlines()
        window = "\n".join(lines[max(0, idx - 3): idx + 4]).lower()
        if "source_variant" in window and "envelope_label" in window:
            score += 25
        elif "source_variant" in window or "envelope_label" in window:
            score += 10
    return score


def make_helper_module() -> str:
    locked_json = json.dumps(LOCKED_ROUTE_MAP, indent=4, sort_keys=True)
    floors_json = json.dumps(CONTRACT_ROW_FLOORS, indent=4, sort_keys=True)
    return f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
26EO-LITE exact-pair route helper.

Drop this next to the next real phase or import it from outputs_basic32.
The contract key is always: (envelope_label, source_variant).
Never validate the source_variant globally without its envelope_label.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, List

LOCKED_ROUTE_FINGERPRINT = {EXPECTED_ROUTE_FINGERPRINT!r}
CONTRACT_NAME = "26EO_label_aware_exact_pair_live_guard"
CAPTURE_TARGET = {CAPTURE_TARGET!r}
STRICT_TANGENT_TARGET = {STRICT_TANGENT_TARGET!r}
RELAXED_TANGENT_TARGET = {RELAXED_TANGENT_TARGET!r}

LOCKED_ROUTE_MAP: Dict[str, str] = {locked_json}

CONTRACT_ROW_FLOORS: Dict[str, Dict[str, float]] = {floors_json}


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
        raise KeyError(f"No locked 26EO route for envelope_label={{envelope_label!r}}")
    return ""


def apply_to_record(
    record: Mapping[str, Any],
    *,
    label_key: str = "envelope_label",
    variant_key: str = "source_variant",
    strict: bool = True,
) -> Dict[str, Any]:
    row = dict(record)
    label = normalize_label(row.get(label_key, ""))
    old_variant = row.get(variant_key)
    new_variant = select_source_variant(label, default=old_variant, strict=strict)
    row[label_key] = label
    row[variant_key] = new_variant
    row["locked_route_selected"] = label in LOCKED_ROUTE_MAP
    row["locked_route_fingerprint"] = LOCKED_ROUTE_FINGERPRINT
    row["locked_route_contract"] = CONTRACT_NAME
    row["locked_route_key"] = "envelope_label+source_variant"
    return row


def apply_to_records(
    records: Iterable[Mapping[str, Any]],
    *,
    label_key: str = "envelope_label",
    variant_key: str = "source_variant",
    strict: bool = True,
) -> List[Dict[str, Any]]:
    return [apply_to_record(r, label_key=label_key, variant_key=variant_key, strict=strict) for r in records]


def exact_pair_mask(df: Any, *, label_col: str = "envelope_label", variant_col: str = "source_variant") -> Any:
    mask = None
    for label, variant in LOCKED_ROUTE_MAP.items():
        m = (df[label_col].astype(str) == label) & (df[variant_col].astype(str) == variant)
        mask = m if mask is None else (mask | m)
    return mask


def filter_exact_pair_dataframe(df: Any, *, label_col: str = "envelope_label", variant_col: str = "source_variant") -> Any:
    return df[exact_pair_mask(df, label_col=label_col, variant_col=variant_col)].copy()


def validate_exact_pairs(
    records: Iterable[Mapping[str, Any]],
    *,
    label_key: str = "envelope_label",
    variant_key: str = "source_variant",
    require_all_labels: bool = True,
) -> Dict[str, Any]:
    present = set()
    mismatches = []
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


def validate_contract_rows(
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


def make_selftest(helper_path: Path) -> str:
    return f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import importlib.util
import json
import sys
from pathlib import Path

HELPER = Path({str(helper_path)!r})
spec = importlib.util.spec_from_file_location("phase26eo_helper_under_test", HELPER)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mod)

records = []
for label, variant in mod.LOCKED_ROUTE_MAP.items():
    records.append({{"envelope_label": label, "source_variant": "WRONG_DEFAULT"}})

selected = mod.apply_to_records(records, strict=True)
exact = mod.validate_exact_pairs(selected)
metric = mod.validate_contract_rows([
    {{"envelope_label": label, "min_capture": vals["min_capture"], "max_tangent": vals["max_tangent"]}}
    for label, vals in mod.CONTRACT_ROW_FLOORS.items()
])

ok = bool(exact["contract_pass"] and metric["metrics_pass"] and mod.LOCKED_ROUTE_FINGERPRINT == {EXPECTED_ROUTE_FINGERPRINT!r})
print(json.dumps({{"EO_SELFTEST_PASS": ok, "exact": exact, "metric": metric}}, indent=2))
if not ok:
    sys.exit(1)
'''


def make_import_scaffold(helper_filename: str = HELPER_MODULE_NAME) -> str:
    return f'''
{IMPORT_SENTINEL}
# This scaffold is intentionally label-aware. The contract key is:
#     envelope_label + source_variant
# Never validate by source_variant alone.
try:
    from {Path(helper_filename).stem} import (
        LOCKED_ROUTE_FINGERPRINT as PHASE26EO_LOCKED_ROUTE_FINGERPRINT,
        select_source_variant as phase26eo_select_source_variant,
        apply_to_record as phase26eo_apply_to_record,
        validate_exact_pairs as phase26eo_validate_exact_pairs,
    )
except Exception:
    # If the helper has not been copied beside this phase yet, keep the phase importable.
    PHASE26EO_LOCKED_ROUTE_FINGERPRINT = {EXPECTED_ROUTE_FINGERPRINT!r}
    def phase26eo_select_source_variant(envelope_label, default=None, *, strict=True):
        if strict:
            raise
        return "" if default is None else str(default)
    def phase26eo_apply_to_record(record, **kwargs):
        return dict(record)
    def phase26eo_validate_exact_pairs(records, **kwargs):
        return {{"contract_pass": False, "reason": "26EO helper import failed"}}
{IMPORT_SENTINEL_END}
'''.strip() + "\n\n"


def locate_insert_position(text: str) -> int:
    """Return character offset after module docstring/import block."""
    # Preserve shebang/coding/docstring and future imports.
    lines = text.splitlines(keepends=True)
    idx = 0
    # shebang / coding
    while idx < len(lines) and (lines[idx].startswith("#!") or "coding" in lines[idx][:80] or not lines[idx].strip()):
        idx += 1
    # module docstring
    if idx < len(lines) and re.match(r'\s*[rRuUbBfF]*["\']{3}', lines[idx]):
        quote = '"""' if '"""' in lines[idx] else "'''"
        idx += 1
        while idx < len(lines) and quote not in lines[idx]:
            idx += 1
        if idx < len(lines):
            idx += 1
    # blank lines + future/imports
    while idx < len(lines):
        stripped = lines[idx].strip()
        if not stripped or stripped.startswith("from __future__") or stripped.startswith("import ") or stripped.startswith("from "):
            idx += 1
            continue
        break
    return sum(len(line) for line in lines[:idx])


def apply_scaffold_to_target(target: Path, helper_path: Path, outputs: Path) -> Tuple[bool, Optional[str], str]:
    if not target.exists():
        return False, None, f"target does not exist: {target}"
    text = read_text(target)
    if IMPORT_SENTINEL in text:
        return False, None, "target already contains 26EO scaffold"

    # Copy helper beside the target so simple import works in normal script execution.
    helper_dest = target.parent / HELPER_MODULE_NAME
    if helper_dest.resolve() != helper_path.resolve():
        shutil.copy2(helper_path, helper_dest)

    backup = target.with_suffix(target.suffix + f".phase26eo_backup_{sha16_text(text)}")
    shutil.copy2(target, backup)

    insert_at = locate_insert_position(text)
    scaffold = make_import_scaffold(HELPER_MODULE_NAME)
    new_text = text[:insert_at] + "\n" + scaffold + text[insert_at:]
    write_text(target, new_text)
    return True, str(backup), f"inserted scaffold and copied helper to {helper_dest}"


def write_ranked_callsites(path: Path, candidates: Sequence[CandidateSite]) -> None:
    fieldnames = list(asdict(candidates[0]).keys()) if candidates else ["file", "path", "score"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cand in candidates:
            writer.writerow(asdict(cand))


def make_patch_plan(candidates: Sequence[CandidateSite], helper_path: Path, target: Optional[Path]) -> str:
    target_txt = str(target) if target else "<choose one of the ranked callsites below>"
    top = candidates[:12]
    rows = "\n".join(
        f"| {i+1} | `{c.file}` | {c.score} | {c.likely_boundary_score} | {c.exists} | {c.already_guarded} | `{c.line_hits}` |"
        for i, c in enumerate(top)
    )
    return f'''# 26EO-LITE live guarded transplant patch plan

## Goal

Move from analysis-only route selection into a real guarded phase without repeating the
26EK mistake. The route must be selected and validated by exact pair:

```text
envelope_label + source_variant
```

not by `source_variant` alone.

## Helper written

`{helper_path}`

Copy or import this helper at the phase boundary that chooses candidate variants.

## Target

`{target_txt}`

## Required insertion

Add this near the route/candidate-selection boundary:

```python
source_variant = phase26eo_select_source_variant(
    envelope_label,
    default=source_variant,
    strict=False,
)
```

For sections where only the locked route should be legal, use:

```python
source_variant = phase26eo_select_source_variant(envelope_label, strict=True)
```

For row dictionaries or exported tables, stamp the route like this:

```python
row = phase26eo_apply_to_record(row, label_key="envelope_label", variant_key="source_variant", strict=False)
```

## Acceptance contract

- `locked_route_fingerprint == {EXPECTED_ROUTE_FINGERPRINT}`
- all seven labels selected by `envelope_label`
- exact-pair rows pass `min_capture >= {CAPTURE_TARGET}`
- exact-pair rows pass `max_tangent <= {STRICT_TANGENT_TARGET}`
- global source-only alias risk is ignored as a diagnostic, not treated as failure

## Ranked callsites

| rank | file | EN score | local boundary score | exists | already guarded | line hits |
|---:|---|---:|---:|---:|---:|---|
{rows}

## After manual transplant

Run:

```powershell
python E:\\BBIT\\outputs_basic32\\{SELFTEST_NAME}
python E:\\BBIT\\outputs_basic32\\phase26en_lite_live_guard_selftest.py
```

Then run the next real phase and verify any exported table contains:

```text
locked_route_fingerprint
locked_route_contract
envelope_label
source_variant
```
'''


def maybe_plot(outputs: Path, candidates: Sequence[CandidateSite]) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
        if not candidates:
            return None
        top = candidates[:20]
        labels = [c.file for c in reversed(top)]
        scores = [c.score + c.likely_boundary_score for c in reversed(top)]
        plt.figure(figsize=(14, max(7, 0.35 * len(labels))))
        plt.barh(labels, scores)
        plt.title("26EO-LITE guarded transplant callsite ranking")
        plt.xlabel("EN score + local boundary score")
        plt.tight_layout()
        out = outputs / "phase26eo_lite_callsite_ranking.png"
        plt.savefig(out, dpi=140)
        plt.close()
        return str(out)
    except Exception:
        return None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=f"{PHASE}: {TITLE}")
    ap.add_argument("--outputs", default=str(DEFAULT_OUTPUTS), help="outputs_basic32 directory")
    ap.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT), help="bbit_geomlang source directory")
    ap.add_argument("--target", default="", help="single target phase file to scaffold or plan")
    ap.add_argument("--apply", action="store_true", help="insert safe import/helper scaffold into --target and back up target first")
    ap.add_argument("--no-plot", action="store_true", help="skip PNG plot generation")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    outputs, source_root = resolve_outputs_and_root(args)
    ensure_dir(outputs)

    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] outputs: {outputs}")
    print(f"[{PHASE}] source_root: {source_root}")

    en_summary = load_json_if_exists(outputs / "phase26en_lite_summary.json")
    live_guard_ready = bool(en_summary.get("LIVE_GUARD_READY", False))

    guard_selftest_path = outputs / "phase26en_lite_live_guard_selftest.py"
    guard_selftest_pass, guard_selftest_out = run_selftest(guard_selftest_path)
    print(f"[{PHASE}] EN guard selftest pass: {guard_selftest_pass}")

    helper_path = outputs / HELPER_MODULE_NAME
    selftest_path = outputs / SELFTEST_NAME
    write_text(helper_path, make_helper_module())
    write_text(selftest_path, make_selftest(helper_path))

    eo_selftest_pass, eo_selftest_out = run_selftest(selftest_path)
    print(f"[{PHASE}] EO helper selftest pass: {eo_selftest_pass}")

    candidates = load_callsite_candidates(outputs, source_root)
    ranked_csv = outputs / CALLSITE_OUT_NAME
    if candidates:
        write_ranked_callsites(ranked_csv, candidates)
    else:
        write_text(ranked_csv, "file,path,score\n")

    target = Path(args.target) if args.target else None
    if target and not target.is_absolute():
        target = source_root / target

    patch_plan_path = outputs / PATCH_PLAN_NAME
    patch_json_path = outputs / PATCH_JSON_NAME
    write_text(patch_plan_path, make_patch_plan(candidates, helper_path, target))

    scaffold_applied = False
    scaffold_backup: Optional[str] = None
    apply_message = "dry-run only"
    if args.apply:
        if not target:
            apply_message = "--apply requested but --target was not provided"
            print(f"[{PHASE}] {apply_message}")
        else:
            scaffold_applied, scaffold_backup, apply_message = apply_scaffold_to_target(target, helper_path, outputs)
            print(f"[{PHASE}] apply scaffold: {scaffold_applied} :: {apply_message}")

    top_candidates = [asdict(c) for c in candidates[:10]]
    plot_path = None if args.no_plot else maybe_plot(outputs, candidates)

    contract_exact_pair_pass = bool(eo_selftest_pass and guard_selftest_pass and EXPECTED_ROUTE_FINGERPRINT == en_summary.get("route_fingerprint", EXPECTED_ROUTE_FINGERPRINT))

    patch_json = {
        "phase": PHASE,
        "title": TITLE,
        "contract_key": ["envelope_label", "source_variant"],
        "do_not_use": "source_variant-only global validation",
        "route_fingerprint": EXPECTED_ROUTE_FINGERPRINT,
        "targets": {
            "requested_target": str(target) if target else None,
            "apply": bool(args.apply),
            "scaffold_applied": scaffold_applied,
            "scaffold_backup": scaffold_backup,
            "apply_message": apply_message,
        },
        "files": {
            "helper": str(helper_path),
            "selftest": str(selftest_path),
            "patch_plan": str(patch_plan_path),
            "ranked_callsites": str(ranked_csv),
            "plot": plot_path,
        },
        "locked_route_map": LOCKED_ROUTE_MAP,
        "contract_row_floors": CONTRACT_ROW_FLOORS,
        "top_candidates": top_candidates,
        "guard_selftest_pass": guard_selftest_pass,
        "guard_selftest_tail": guard_selftest_out[-1500:],
        "eo_selftest_pass": eo_selftest_pass,
        "eo_selftest_tail": eo_selftest_out[-1500:],
    }
    write_json(patch_json_path, patch_json)

    result = EOResult(
        phase=PHASE,
        title=TITLE,
        outputs=str(outputs),
        source_root=str(source_root),
        live_guard_ready=live_guard_ready,
        guard_selftest_pass=guard_selftest_pass,
        helper_written=str(helper_path),
        selftest_written=str(selftest_path),
        patch_plan_written=str(patch_plan_path),
        patch_json_written=str(patch_json_path),
        ranked_callsites_written=str(ranked_csv),
        target=str(target) if target else None,
        apply=bool(args.apply),
        scaffold_applied=scaffold_applied,
        scaffold_backup=scaffold_backup,
        contract_exact_pair_pass=contract_exact_pair_pass,
        route_fingerprint=EXPECTED_ROUTE_FINGERPRINT,
        top_candidates=top_candidates[:5],
        next_action=(
            "Choose the real next phase target, run 26EO with --target for a patch plan, then use --apply only for the scaffold. "
            "Manually replace the source_variant selection boundary with phase26eo_select_source_variant(envelope_label, default=old_variant, strict=False)."
        ),
    )

    summary_path = outputs / SUMMARY_NAME
    write_json(summary_path, asdict(result))

    print(f"[{PHASE}] LIVE_GUARD_READY={live_guard_ready}")
    print(f"[{PHASE}] CONTRACT_EXACT_PAIR_PASS={contract_exact_pair_pass}")
    print(f"[{PHASE}] wrote helper: {helper_path}")
    print(f"[{PHASE}] wrote selftest: {selftest_path}")
    print(f"[{PHASE}] wrote patch plan: {patch_plan_path}")
    print(f"[{PHASE}] wrote ranked callsites: {ranked_csv}")
    if plot_path:
        print(f"[{PHASE}] wrote plot: {plot_path}")
    print(f"[{PHASE}] wrote outputs to: {outputs}")

    if not contract_exact_pair_pass:
        print(f"[{PHASE}] WARNING: selftest chain did not fully pass. Do not transplant until fixed.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# --- 26EQ SHADOW PATCH NOTES ---
# This shadow copy only proves importability/syntax of the exact-pair guard.
# Manual/live transplant still requires placing the selection-boundary snippet
# exactly where source_variant is selected for each envelope_label.
# See phase26eq_lite_shadow_patch_bundle.md.
