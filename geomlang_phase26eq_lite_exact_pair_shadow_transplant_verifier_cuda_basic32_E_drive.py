#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
26EQ-LITE exact-pair shadow transplant verifier / guarded diff bundle.

Purpose
-------
26EK proved the dangerous failure mode: validating a source_variant globally can
fail even when the exact contract row is safe. 26EL/EM/EN/EO/EP converged on the
correct transplant rule:

    contract key = envelope_label + source_variant

26EQ does NOT mutate live phase files. It builds a shadow patch bundle for the
highest-ranked callsites, injects the exact-pair guard/import block into shadow
copies, syntax-checks those shadow copies, validates the locked contract rows,
and writes a copy-paste patch bundle for the one place where the real phase must
be changed: the candidate/source-variant selection boundary.

Run from:
    E:\BBIT

Example:
    python bbit_geomlang/geomlang_phase26eq_lite_exact_pair_shadow_transplant_verifier_cuda_basic32_E_drive.py

Outputs are written to:
    E:\BBIT\outputs_basic32
"""
from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PHASE = "26EQ-LITE"
TITLE = "Exact-pair shadow transplant verifier / guarded diff bundle"

ROOT = Path(r"E:\BBIT")
SOURCE_ROOT = ROOT / "bbit_geomlang"
OUT = ROOT / "outputs_basic32"
SHADOW_DIR = OUT / "phase26eq_lite_shadow_patches"

HELPER_NAME = "phase26eo_lite_exact_pair_route_helper.py"
HELPER_PATH = OUT / HELPER_NAME
EP_RANKED = OUT / "phase26ep_lite_ranked_callsites.csv"
EO_RANKED = OUT / "phase26eo_lite_ranked_callsites.csv"
CONTRACT_ROWS_CANDIDATES = [
    OUT / "phase26em_lite_contract_row_smoke.csv",
    OUT / "phase26el_lite_contract_row_smoke.csv",
    OUT / "phase26ep_lite_contract_rows.csv",
    OUT / "phase26ej_lite_label_protection_table.csv",
]

CAPTURE_TARGET = 0.285
STRICT_TANGENT_TARGET = 2.1
RELAXED_TANGENT_TARGET = 2.3
TOP_N = 12

GUARD_SENTINEL_BEGIN = "# --- 26EQ exact-pair locked-route guard BEGIN ---"
GUARD_SENTINEL_END = "# --- 26EQ exact-pair locked-route guard END ---"

GUARD_BLOCK = r'''# --- 26EQ exact-pair locked-route guard BEGIN ---
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
'''

SELECTION_SNIPPET = r'''# --- 26EQ candidate/source selection boundary patch BEGIN ---
# Force the locked exact-pair source for this envelope label.
# IMPORTANT: this must happen at the point where source_variant is chosen for a label.
source_variant = phase26eq_select_source_variant(
    envelope_label,
    default=source_variant,
    strict=False,
)
# --- 26EQ candidate/source selection boundary patch END ---
'''

ROW_STAMP_SNIPPET = r'''# --- 26EQ result-row contract stamp BEGIN ---
# Stamp exported rows so downstream smoke tests can prove the exact-pair contract was used.
row = phase26eq_apply_to_record(
    row,
    label_key="envelope_label",
    variant_key="source_variant",
    strict=False,
)
# --- 26EQ result-row contract stamp END ---
'''

RUNTIME_GATE_CODE = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
26EQ runtime exact-pair route gate.

Import this from a live phase after phase26eo_lite_exact_pair_route_helper.py has
been generated. This module is intentionally tiny: it does not validate source
variants globally. It only forces/validates the locked route by exact pair:
(envelope_label, source_variant).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, List

_OUT = Path(r"E:\BBIT\outputs_basic32")
if str(_OUT) not in sys.path:
    sys.path.insert(0, str(_OUT))

from phase26eo_lite_exact_pair_route_helper import (  # noqa: E402
    LOCKED_ROUTE_FINGERPRINT,
    CONTRACT_NAME,
    LOCKED_ROUTE_MAP,
    CAPTURE_TARGET,
    STRICT_TANGENT_TARGET,
    RELAXED_TANGENT_TARGET,
    select_source_variant,
    apply_to_record,
    apply_to_records,
    validate_exact_pairs,
    validate_contract_rows,
    locked_labels,
    is_locked_label,
)


def force_variant(envelope_label: Any, source_variant: Optional[Any] = None, *, strict: bool = False) -> str:
    return select_source_variant(envelope_label, default=source_variant, strict=strict)


def stamp_row(row: Mapping[str, Any], *, strict: bool = False) -> Dict[str, Any]:
    return apply_to_record(row, strict=strict)


def stamp_rows(rows: Iterable[Mapping[str, Any]], *, strict: bool = False) -> List[Dict[str, Any]]:
    return apply_to_records(rows, strict=strict)


def assert_contract_rows(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    result = validate_contract_rows(rows)
    if not result.get("metrics_pass"):
        raise AssertionError(f"26EQ exact-pair contract row failure: {result}")
    return result
'''


@dataclass
class Candidate:
    file: str
    score: float
    line_hits: str = ""
    guarded: bool = False
    analysis_like: bool = False
    source: str = ""
    exists: bool = False
    syntax_ok: Optional[bool] = None
    shadow_file: str = ""
    inserted_guard: bool = False
    candidate_assignments: int = 0
    result_row_sites: int = 0
    notes: str = ""


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SHADOW_DIR.mkdir(parents=True, exist_ok=True)


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def load_helper_module() -> Any:
    if not HELPER_PATH.exists():
        raise FileNotFoundError(f"Missing helper: {HELPER_PATH}. Run 26EO or 26EP first.")
    spec = importlib.util.spec_from_file_location("phase26eo_lite_exact_pair_route_helper", str(HELPER_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper spec from {HELPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def write_csv_dicts(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    if fieldnames is None:
        keys: List[str] = []
        seen = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    keys.append(k)
                    seen.add(k)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def score_source_text(text: str) -> Tuple[int, List[int]]:
    weighted_terms = {
        "envelope_label": 12,
        "source_variant": 12,
        "select_source": 10,
        "select_variant": 10,
        "candidate": 5,
        "route": 5,
        "variant": 4,
        "label": 4,
        "pareto": 3,
        "locked": 3,
        "contract": 3,
        "capture": 2,
        "tangent": 2,
    }
    lines = text.splitlines()
    hits: List[int] = []
    score = 0
    for idx, line in enumerate(lines, start=1):
        lower = line.lower()
        line_score = 0
        for term, weight in weighted_terms.items():
            if term in lower:
                line_score += weight
        if line_score:
            hits.append(idx)
            score += line_score
    return score, hits[:25]


def scan_fallback_candidates() -> List[Candidate]:
    candidates: List[Candidate] = []
    if not SOURCE_ROOT.exists():
        return candidates
    for path in SOURCE_ROOT.glob("geomlang_phase26*.py"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        score, hits = score_source_text(text)
        if score <= 0:
            continue
        candidates.append(Candidate(
            file=path.name,
            score=float(score),
            line_hits=";".join(str(h) for h in hits),
            source="fallback_scan",
            exists=True,
            guarded=("phase26eq_select_source_variant" in text or "phase26ep_select_source_variant" in text or "phase26eo_lite_exact_pair_route_helper" in text),
            analysis_like=("matplotlib" in text or "plt." in text or "pandas" in text or "DataFrame" in text),
        ))
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def load_ranked_candidates() -> List[Candidate]:
    rows = read_csv_dicts(EP_RANKED) or read_csv_dicts(EO_RANKED)
    candidates: List[Candidate] = []
    for row in rows:
        file_name = row.get("file") or row.get("path") or row.get("filename") or ""
        if not file_name:
            continue
        score_raw = row.get("score") or row.get("relevance_score") or row.get("route_boundary_relevance_score") or "0"
        try:
            score = float(score_raw)
        except Exception:
            score = 0.0
        path = SOURCE_ROOT / file_name
        text = ""
        exists = path.exists()
        if exists:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                text = ""
        candidates.append(Candidate(
            file=file_name,
            score=score,
            line_hits=row.get("line_hits", row.get("hits", "")),
            guarded=str(row.get("guarded", "")).lower() == "true" or ("phase26eq_select_source_variant" in text) or ("phase26ep_select_source_variant" in text),
            analysis_like=str(row.get("analysis_like", "")).lower() == "true" or ("matplotlib" in text or "plt." in text),
            source=EP_RANKED.name if EP_RANKED.exists() else EO_RANKED.name,
            exists=exists,
        ))
    if not candidates:
        candidates = scan_fallback_candidates()
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:TOP_N]


def insertion_index_after_future_imports(text: str) -> int:
    """Return character index after module docstring and __future__ imports."""
    try:
        tree = ast.parse(text)
        lines = text.splitlines(keepends=True)
        insert_line = 0
        body = list(tree.body)
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
            insert_line = getattr(body[0], "end_lineno", body[0].lineno)
        for node in body[1 if insert_line else 0:]:
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                insert_line = max(insert_line, getattr(node, "end_lineno", node.lineno))
            elif isinstance(node, (ast.Import, ast.ImportFrom)) and insert_line == 0:
                # no docstring/future; insert at very top before normal imports
                break
            elif insert_line:
                break
        return sum(len(x) for x in lines[:insert_line])
    except Exception:
        m = re.search(r"^(from __future__ import .*\n)+", text, flags=re.MULTILINE)
        if m:
            return m.end()
        return 0


def count_candidate_boundaries(text: str) -> Tuple[int, int]:
    candidate_assignments = len(re.findall(r"\bsource_variant\s*=", text))
    result_row_sites = len(re.findall(r"\b(row|record|result)\s*=\s*\{", text)) + len(re.findall(r"\.append\(\s*row\s*\)", text))
    return candidate_assignments, result_row_sites


def build_shadow_patch(path: Path) -> Tuple[str, bool, str, Optional[bool]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    inserted = False
    if GUARD_SENTINEL_BEGIN not in text and "phase26eq_select_source_variant" not in text:
        idx = insertion_index_after_future_imports(text)
        prefix = text[:idx]
        suffix = text[idx:]
        sep1 = "" if prefix.endswith("\n") or not prefix else "\n"
        sep2 = "" if suffix.startswith("\n") else "\n"
        text = prefix + sep1 + GUARD_BLOCK + sep2 + suffix
        inserted = True
    banner = (
        "\n\n# --- 26EQ SHADOW PATCH NOTES ---\n"
        "# This shadow copy only proves importability/syntax of the exact-pair guard.\n"
        "# Manual/live transplant still requires placing the selection-boundary snippet\n"
        "# exactly where source_variant is selected for each envelope_label.\n"
        "# See phase26eq_lite_shadow_patch_bundle.md.\n"
    )
    if "26EQ SHADOW PATCH NOTES" not in text:
        text += banner
    syntax_ok: Optional[bool]
    try:
        ast.parse(text)
        syntax_ok = True
    except SyntaxError:
        syntax_ok = False
    return text, inserted, sha16(text), syntax_ok


def validate_contract(helper: Any) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    selected_path = None
    for p in CONTRACT_ROWS_CANDIDATES:
        tmp = read_csv_dicts(p)
        if tmp:
            rows = tmp
            selected_path = p
            break
    if not rows:
        # Fall back to helper's embedded floors.
        for label, variant in helper.LOCKED_ROUTE_MAP.items():
            floors = helper.CONTRACT_ROW_FLOORS[label]
            rows.append({
                "envelope_label": label,
                "source_variant": variant,
                "min_capture": floors["min_capture"],
                "max_tangent": floors["max_tangent"],
            })
        selected_path = Path("helper.CONTRACT_ROW_FLOORS")

    norm_rows: List[Dict[str, Any]] = []
    for r in rows:
        label = r.get("envelope_label", "")
        min_capture = r.get("min_capture", r.get("worst_capture", r.get("capture", "nan")))
        max_tangent = r.get("max_tangent", r.get("worst_tangent", r.get("tangent", "nan")))
        norm_rows.append({
            "envelope_label": label,
            "source_variant": r.get("source_variant", helper.select_source_variant(label, strict=False)),
            "min_capture": float(min_capture),
            "max_tangent": float(max_tangent),
        })
    exact_result = helper.validate_exact_pairs(norm_rows, require_all_labels=True)
    metric_result = helper.validate_contract_rows(norm_rows)
    return {
        "rows_source": str(selected_path),
        "rows": norm_rows,
        "exact_pair_pass": bool(exact_result.get("contract_pass")),
        "metric_pass": bool(metric_result.get("metrics_pass")),
        "exact_result": exact_result,
        "metric_result": metric_result,
    }


def write_runtime_gate() -> Path:
    path = OUT / "phase26eq_lite_runtime_exact_pair_gate.py"
    path.write_text(RUNTIME_GATE_CODE, encoding="utf-8")
    return path


def write_selftest() -> Path:
    code = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

OUT = Path(r"E:\BBIT\outputs_basic32")
if str(OUT) not in sys.path:
    sys.path.insert(0, str(OUT))

from phase26eq_lite_runtime_exact_pair_gate import (  # noqa: E402
    LOCKED_ROUTE_FINGERPRINT,
    LOCKED_ROUTE_MAP,
    CAPTURE_TARGET,
    STRICT_TANGENT_TARGET,
    force_variant,
    stamp_row,
    validate_exact_pairs,
    validate_contract_rows,
)


def main() -> int:
    rows = []
    for label, variant in LOCKED_ROUTE_MAP.items():
        picked = force_variant(label, "bad_default", strict=True)
        assert picked == variant, (label, picked, variant)
        rows.append(stamp_row({
            "envelope_label": label,
            "source_variant": "bad_default",
            "min_capture": CAPTURE_TARGET + 0.001,
            "max_tangent": STRICT_TANGENT_TARGET - 0.001,
        }, strict=True))
    exact = validate_exact_pairs(rows)
    metrics = validate_contract_rows(rows)
    assert exact["contract_pass"], exact
    assert metrics["metrics_pass"], metrics
    assert LOCKED_ROUTE_FINGERPRINT == "98ebdcbb8e995bc1"
    print("[26EQ-LITE] runtime exact-pair gate selftest PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    path = OUT / "phase26eq_lite_runtime_exact_pair_gate_selftest.py"
    path.write_text(code, encoding="utf-8")
    return path


def make_plot(candidates: Sequence[Candidate]) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return None
    if not candidates:
        return None
    labels = [c.file for c in candidates][::-1]
    vals = [c.score for c in candidates][::-1]
    fig_h = max(6.0, 0.42 * len(labels) + 2.0)
    plt.figure(figsize=(14, fig_h))
    plt.barh(labels, vals)
    plt.xlabel("shadow-transplant relevance score")
    plt.title("26EQ-LITE exact-pair shadow transplant targets")
    plt.tight_layout()
    path = OUT / "phase26eq_lite_shadow_transplant_targets.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def write_bundle(candidates: Sequence[Candidate], contract: Mapping[str, Any], runtime_gate: Path, selftest: Path) -> Path:
    path = OUT / "phase26eq_lite_shadow_patch_bundle.md"
    rows = contract.get("rows", [])
    lines: List[str] = []
    lines.append("26EQ-LITE exact-pair shadow transplant bundle")
    lines.append("================================================")
    lines.append("")
    lines.append("Status: shadow bundle generated. Live files were not modified.")
    lines.append("")
    lines.append("Contract rule:")
    lines.append("")
    lines.append("```text")
    lines.append("contract key = envelope_label + source_variant")
    lines.append("Never validate or select source_variant globally without envelope_label.")
    lines.append("```")
    lines.append("")
    lines.append(f"Runtime gate: `{runtime_gate}`")
    lines.append(f"Runtime selftest: `{selftest}`")
    lines.append(f"Contract rows source: `{contract.get('rows_source')}`")
    lines.append(f"Exact-pair pass: `{contract.get('exact_pair_pass')}`")
    lines.append(f"Metric pass: `{contract.get('metric_pass')}`")
    lines.append("")
    lines.append("Locked contract rows:")
    lines.append("")
    lines.append("| envelope_label | source_variant | min_capture | max_tangent |")
    lines.append("|---|---|---:|---:|")
    for r in rows:
        lines.append(f"| {r.get('envelope_label')} | `{r.get('source_variant')}` | {float(r.get('min_capture')):.6f} | {float(r.get('max_tangent')):.6f} |")
    lines.append("")
    lines.append("Import guard block to paste after `from __future__` imports:")
    lines.append("")
    lines.append("```python")
    lines.append(GUARD_BLOCK.rstrip())
    lines.append("```")
    lines.append("")
    lines.append("Candidate/source-variant selection boundary patch:")
    lines.append("")
    lines.append("```python")
    lines.append(SELECTION_SNIPPET.rstrip())
    lines.append("```")
    lines.append("")
    lines.append("Result-row stamping patch:")
    lines.append("")
    lines.append("```python")
    lines.append(ROW_STAMP_SNIPPET.rstrip())
    lines.append("```")
    lines.append("")
    lines.append("Top shadow-patched files:")
    lines.append("")
    lines.append("| rank | file | score | syntax_ok | guard_inserted | source_variant assignments | row sites | shadow file |")
    lines.append("|---:|---|---:|---|---|---:|---:|---|")
    for i, c in enumerate(candidates, start=1):
        lines.append(
            f"| {i} | `{c.file}` | {c.score:.0f} | {c.syntax_ok} | {c.inserted_guard} | "
            f"{c.candidate_assignments} | {c.result_row_sites} | `{c.shadow_file}` |"
        )
    lines.append("")
    lines.append("Acceptance after live transplant:")
    lines.append("")
    lines.append("```text")
    lines.append("fingerprint == 98ebdcbb8e995bc1")
    lines.append("min_capture >= 0.285")
    lines.append("max_tangent <= 2.1")
    lines.append("all 7 envelope labels present")
    lines.append("source_variant is selected/validated only with envelope_label")
    lines.append("```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] outputs: {OUT}")
    print(f"[{PHASE}] source_root: {SOURCE_ROOT}")
    ensure_dirs()

    helper = load_helper_module()
    contract = validate_contract(helper)
    runtime_gate = write_runtime_gate()
    selftest = write_selftest()

    candidates = load_ranked_candidates()
    processed: List[Candidate] = []
    for cand in candidates:
        src = SOURCE_ROOT / cand.file
        cand.exists = src.exists()
        if not src.exists():
            cand.notes = "source file not found on this machine"
            processed.append(cand)
            continue
        try:
            original = src.read_text(encoding="utf-8", errors="replace")
            cand.candidate_assignments, cand.result_row_sites = count_candidate_boundaries(original)
            shadow_text, inserted, text_hash, syntax_ok = build_shadow_patch(src)
            cand.inserted_guard = inserted
            cand.syntax_ok = syntax_ok
            shadow_name = src.stem + "__26eq_shadow__" + text_hash + src.suffix
            shadow_path = SHADOW_DIR / shadow_name
            shadow_path.write_text(shadow_text, encoding="utf-8")
            cand.shadow_file = str(shadow_path)
            cand.notes = "shadow patched only; live file untouched"
        except Exception as exc:
            cand.notes = f"shadow patch failed: {exc}"
            cand.syntax_ok = False
        processed.append(cand)

    plot_path = make_plot(processed)
    top_csv = OUT / "phase26eq_lite_shadow_patch_targets.csv"
    write_csv_dicts(top_csv, [asdict(c) for c in processed])

    manifest = {
        "phase": PHASE,
        "title": TITLE,
        "contract_shadow_ready": bool(contract.get("exact_pair_pass") and contract.get("metric_pass") and all(c.syntax_ok is not False for c in processed if c.exists)),
        "helper": str(HELPER_PATH),
        "runtime_gate": str(runtime_gate),
        "runtime_selftest": str(selftest),
        "shadow_dir": str(SHADOW_DIR),
        "top_targets_csv": str(top_csv),
        "plot": str(plot_path) if plot_path else "",
        "contract": contract,
        "targets": [asdict(c) for c in processed],
        "selection_snippet": SELECTION_SNIPPET,
        "row_stamp_snippet": ROW_STAMP_SNIPPET,
    }
    manifest_path = OUT / "phase26eq_lite_shadow_transplant_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    bundle = write_bundle(processed, contract, runtime_gate, selftest)
    summary = {
        "phase": PHASE,
        "CONTRACT_SHADOW_READY": manifest["contract_shadow_ready"],
        "exact_pair_pass": contract.get("exact_pair_pass"),
        "metric_pass": contract.get("metric_pass"),
        "target_count": len(processed),
        "shadow_syntax_failures": [c.file for c in processed if c.syntax_ok is False],
        "top_target": processed[0].file if processed else None,
        "runtime_gate": str(runtime_gate),
        "bundle": str(bundle),
        "manifest": str(manifest_path),
    }
    summary_path = OUT / "phase26eq_lite_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[{PHASE}] CONTRACT_SHADOW_READY={summary['CONTRACT_SHADOW_READY']}")
    print(f"[{PHASE}] exact_pair_pass={contract.get('exact_pair_pass')} metric_pass={contract.get('metric_pass')}")
    print(f"[{PHASE}] top shadow targets:")
    for i, c in enumerate(processed[:10], start=1):
        print(f"  {i:2d}. {c.file} score={c.score:.0f} syntax_ok={c.syntax_ok} assigns={c.candidate_assignments} rows={c.result_row_sites}")
    print(f"[{PHASE}] wrote runtime gate: {runtime_gate}")
    print(f"[{PHASE}] wrote selftest: {selftest}")
    print(f"[{PHASE}] wrote bundle: {bundle}")
    print(f"[{PHASE}] wrote manifest: {manifest_path}")
    print(f"[{PHASE}] wrote outputs to: {OUT}")
    return 0 if summary["CONTRACT_SHADOW_READY"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
