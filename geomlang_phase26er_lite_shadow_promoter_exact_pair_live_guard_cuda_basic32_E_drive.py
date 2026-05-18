#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
26ER-LITE Shadow promoter / exact-pair live guard promotion gate.

Purpose
-------
26EQ proved the shadow transplant bundle is contract-ready. 26ER is the next
bridge: it ranks the shadow-patched targets for safe promotion, verifies every
shadow file still contains the exact-pair guard, writes an atomic promotion
script, and can optionally promote ONE target with a timestamped backup.

Default mode is non-destructive. It writes:
  - phase26er_lite_promotion_candidates.csv
  - phase26er_lite_shadow_promotion_manifest.json
  - phase26er_lite_shadow_promotion_plan.md
  - phase26er_lite_apply_top_shadow_patch.ps1
  - phase26er_lite_promotion_candidates.png

Optional live promotion:
  python bbit_geomlang/geomlang_phase26er_lite_shadow_promoter_exact_pair_live_guard_cuda_basic32_E_drive.py --apply --target auto

The promoted replacement is copied from the 26EQ shadow patch, never generated
from scratch. The original live file is backed up before overwrite.
"""
from __future__ import annotations

import argparse
import ast
import csv
import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PHASE = "26ER-LITE"
TITLE = "Shadow promoter / exact-pair live guard promotion gate"

CAPTURE_TARGET = 0.285
STRICT_TANGENT_TARGET = 2.10
RELAXED_TANGENT_TARGET = 2.30
EXPECTED_FINGERPRINT = "98ebdcbb8e995bc1"

DEFAULT_OUTPUTS = Path(r"E:\BBIT\outputs_basic32")
DEFAULT_SOURCE_ROOT = Path(r"E:\BBIT\bbit_geomlang")

REQUIRED_GUARD_TOKENS = [
    "LOCKED_ROUTE_MAP",
    "CONTRACT_ROW_FLOORS",
    "validate_exact_pairs",
    "validate_contract_rows",
    "envelope_label",
    "source_variant",
]

RISK_PATTERNS = [
    r"source_variant\s*==",
    r"\[\s*['\"]source_variant['\"]\s*\]",
    r"\.source_variant\b",
    r"groupby\s*\([^\)]*source_variant",
    r"drop_duplicates\s*\([^\)]*source_variant",
]

SAFE_PAIR_PATTERNS = [
    r"envelope_label.*source_variant",
    r"source_variant.*envelope_label",
    r"exact_pair",
    r"LOCKED_ROUTE_MAP",
    r"validate_exact_pairs",
    r"select_source_variant",
    r"apply_to_record",
]


def now_stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def sha16_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def sha16_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)


def syntax_ok(path: Path) -> Tuple[bool, str]:
    try:
        ast.parse(read_text(path), filename=str(path))
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, repr(e)


def count_regexes(text: str, patterns: Iterable[str]) -> int:
    total = 0
    for pat in patterns:
        total += len(re.findall(pat, text, flags=re.IGNORECASE | re.DOTALL))
    return total


def guard_token_result(text: str) -> Dict[str, Any]:
    missing = [tok for tok in REQUIRED_GUARD_TOKENS if tok not in text]
    return {
        "guard_tokens_present": len(missing) == 0,
        "missing_guard_tokens": missing,
        "guard_token_count": len(REQUIRED_GUARD_TOKENS) - len(missing),
    }


def line_hit_count(value: Any) -> int:
    if value is None:
        return 0
    s = str(value)
    if not s.strip():
        return 0
    return len([x for x in re.split(r"[;,\s]+", s) if x.strip().isdigit()])


def find_outputs_and_source(args: argparse.Namespace) -> Tuple[Path, Path]:
    outputs = Path(args.outputs) if args.outputs else DEFAULT_OUTPUTS
    source_root = Path(args.source_root) if args.source_root else DEFAULT_SOURCE_ROOT
    return outputs, source_root


def load_eq_manifest(outputs: Path) -> Dict[str, Any]:
    manifest_path = outputs / "phase26eq_lite_shadow_transplant_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing 26EQ manifest: {manifest_path}")
    return load_json(manifest_path)


def target_source_path(source_root: Path, target_file: str) -> Path:
    p = Path(target_file)
    if p.is_absolute():
        return p
    return source_root / p.name


def score_target(target: Dict[str, Any], source_root: Path) -> Dict[str, Any]:
    file_name = str(target.get("file", ""))
    src_path = target_source_path(source_root, file_name)
    shadow_path = Path(str(target.get("shadow_file", "")))

    row: Dict[str, Any] = {
        "file": Path(file_name).name,
        "source_path": str(src_path),
        "shadow_file": str(shadow_path),
        "eq_score": float(target.get("score", 0.0) or 0.0),
        "line_hit_count": line_hit_count(target.get("line_hits")),
        "candidate_assignments": int(float(target.get("candidate_assignments", 0) or 0)),
        "result_row_sites": int(float(target.get("result_row_sites", 0) or 0)),
        "eq_inserted_guard": bool(target.get("inserted_guard", False)),
        "source_exists": src_path.exists(),
        "shadow_exists": shadow_path.exists(),
    }

    if not row["shadow_exists"]:
        row.update({
            "shadow_syntax_ok": False,
            "syntax_error": "missing shadow file",
            "guard_tokens_present": False,
            "missing_guard_tokens": REQUIRED_GUARD_TOKENS,
            "shadow_hash": "",
            "source_hash": sha16_file(src_path) if src_path.exists() else "",
            "risk_token_count": 999,
            "safe_pair_token_count": 0,
            "already_promoted": False,
        })
    else:
        shadow_text = read_text(shadow_path)
        ok, err = syntax_ok(shadow_path)
        guard = guard_token_result(shadow_text)
        src_hash = sha16_file(src_path) if src_path.exists() else ""
        shadow_hash = sha16_text(shadow_text)
        row.update({
            "shadow_syntax_ok": ok,
            "syntax_error": err,
            **guard,
            "shadow_hash": shadow_hash,
            "source_hash": src_hash,
            "risk_token_count": count_regexes(shadow_text, RISK_PATTERNS),
            "safe_pair_token_count": count_regexes(shadow_text, SAFE_PAIR_PATTERNS),
            "already_promoted": bool(src_hash and src_hash == shadow_hash),
        })

    # Promotion score: prefer high 26EQ relevance, real source, syntax-safe guard,
    # and files that actually have candidate assignment/result-row sites.
    structural_weight = 40 * row["candidate_assignments"] + 55 * row["result_row_sites"]
    guard_bonus = 250 if row.get("guard_tokens_present") else -500
    syntax_bonus = 200 if row.get("shadow_syntax_ok") else -1000
    exists_bonus = 100 if row.get("source_exists") and row.get("shadow_exists") else -250
    risk_penalty = 6 * int(row.get("risk_token_count", 0))
    already_penalty = 1000 if row.get("already_promoted") else 0
    row["promotion_score"] = (
        row["eq_score"] + structural_weight + guard_bonus + syntax_bonus + exists_bonus - risk_penalty - already_penalty
    )
    row["promotion_ready"] = bool(
        row.get("source_exists")
        and row.get("shadow_exists")
        and row.get("shadow_syntax_ok")
        and row.get("guard_tokens_present")
        and not row.get("already_promoted")
    )
    return row


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        write_text(path, "")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def make_plot(path: Path, rows: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    top = rows[:20]
    labels = [r["file"] for r in top][::-1]
    vals = [float(r["promotion_score"]) for r in top][::-1]
    fig_h = max(7, 0.42 * len(labels) + 2)
    plt.figure(figsize=(14, fig_h))
    plt.barh(labels, vals)
    plt.title("26ER-LITE shadow promotion candidates")
    plt.xlabel("promotion score = EQ relevance + guard safety + local boundary score")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def ps_quote(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def write_apply_script(path: Path, selected: Dict[str, Any], source_root: Path) -> None:
    src = Path(selected["source_path"])
    shadow = Path(selected["shadow_file"])
    backup = src.with_name(src.name + f".26erbak_{now_stamp()}")
    text = f"""# 26ER-LITE apply top shadow patch
# Generated non-destructively. Review before running.
$ErrorActionPreference = 'Stop'
$Source = {ps_quote(src)}
$Shadow = {ps_quote(shadow)}
$Backup = {ps_quote(backup)}

Write-Host '[26ER-LITE] source:' $Source
Write-Host '[26ER-LITE] shadow:' $Shadow
Write-Host '[26ER-LITE] backup:' $Backup

if (!(Test-Path $Source)) {{ throw "Missing source file: $Source" }}
if (!(Test-Path $Shadow)) {{ throw "Missing shadow file: $Shadow" }}
Copy-Item -LiteralPath $Source -Destination $Backup -Force
Copy-Item -LiteralPath $Shadow -Destination $Source -Force
Write-Host '[26ER-LITE] promoted shadow patch. Original backed up.'
Write-Host '[26ER-LITE] next: run the target script selftest / previous phase rerun.'
"""
    write_text(path, text)


def write_plan(path: Path, rows: List[Dict[str, Any]], selected: Optional[Dict[str, Any]], manifest: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# 26ER-LITE Shadow Promotion Plan")
    lines.append("")
    lines.append("Status: **PROMOTION_PLAN_READY**" if selected else "Status: **NO_PROMOTION_TARGET_READY**")
    lines.append("")
    lines.append("26ER keeps the same rule discovered in EK/EL/EM/EN/EO/EQ: never validate or select a `source_variant` globally. The only safe contract key is `(envelope_label, source_variant)`.")
    lines.append("")
    lines.append("## Selected promotion target")
    lines.append("")
    if selected:
        lines.append(f"- File: `{selected['file']}`")
        lines.append(f"- Promotion score: `{selected['promotion_score']:.3f}`")
        lines.append(f"- Source: `{selected['source_path']}`")
        lines.append(f"- Shadow: `{selected['shadow_file']}`")
        lines.append(f"- Shadow hash: `{selected['shadow_hash']}`")
        lines.append(f"- Guard tokens present: `{selected['guard_tokens_present']}`")
        lines.append(f"- Syntax OK: `{selected['shadow_syntax_ok']}`")
    else:
        lines.append("No candidate passed source/shadow/syntax/guard readiness checks.")
    lines.append("")
    lines.append("## Why this is still guarded")
    lines.append("")
    lines.append("The selected shadow file is not trusted because its `source_variant` appears in the pool. It is trusted only if the exact-pair guard confirms the label and variant together. This prevents the false alarm seen in EK where global source-variant smoke failed because every locked variant had unsafe aliases outside its assigned label.")
    lines.append("")
    lines.append("## Next command")
    lines.append("")
    lines.append("Non-destructive verification:")
    lines.append("")
    lines.append("```powershell")
    lines.append(r"python bbit_geomlang/geomlang_phase26er_lite_shadow_promoter_exact_pair_live_guard_cuda_basic32_E_drive.py")
    lines.append("```")
    lines.append("")
    lines.append("Optional promotion of the selected target:")
    lines.append("")
    lines.append("```powershell")
    lines.append(r"python bbit_geomlang/geomlang_phase26er_lite_shadow_promoter_exact_pair_live_guard_cuda_basic32_E_drive.py --apply --target auto")
    lines.append("```")
    lines.append("")
    lines.append("## Top candidates")
    lines.append("")
    lines.append("| rank | file | promotion_score | ready | assigns | row sites | risk tokens |")
    lines.append("|---:|---|---:|---|---:|---:|---:|")
    for i, r in enumerate(rows[:12], 1):
        lines.append(
            f"| {i} | `{r['file']}` | {float(r['promotion_score']):.1f} | {r['promotion_ready']} | {r['candidate_assignments']} | {r['result_row_sites']} | {r['risk_token_count']} |"
        )
    write_text(path, "\n".join(lines) + "\n")


def pick_target(rows: List[Dict[str, Any]], target: str) -> Optional[Dict[str, Any]]:
    ready = [r for r in rows if r.get("promotion_ready")]
    if not ready:
        return None
    if target.lower() == "auto":
        return ready[0]
    for r in ready:
        if target.lower() in r["file"].lower():
            return r
    raise ValueError(f"Requested target {target!r} did not match any promotion-ready candidate")


def apply_shadow(selected: Dict[str, Any]) -> Dict[str, Any]:
    src = Path(selected["source_path"])
    shadow = Path(selected["shadow_file"])
    if not src.exists():
        raise FileNotFoundError(f"Missing source: {src}")
    if not shadow.exists():
        raise FileNotFoundError(f"Missing shadow: {shadow}")
    ok, err = syntax_ok(shadow)
    if not ok:
        raise RuntimeError(f"Refusing to apply syntax-bad shadow: {err}")
    backup = src.with_name(src.name + f".26erbak_{now_stamp()}")
    shutil.copy2(src, backup)
    shutil.copy2(shadow, src)
    return {
        "applied": True,
        "source": str(src),
        "shadow": str(shadow),
        "backup": str(backup),
        "new_source_hash": sha16_file(src),
        "backup_hash": sha16_file(backup),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=f"{PHASE}: {TITLE}")
    ap.add_argument("--outputs", default=str(DEFAULT_OUTPUTS), help="outputs_basic32 directory")
    ap.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT), help="bbit_geomlang source directory")
    ap.add_argument("--target", default="auto", help="promotion target substring, or auto")
    ap.add_argument("--apply", action="store_true", help="actually copy selected 26EQ shadow file over the live source, with backup")
    args = ap.parse_args(argv)

    outputs, source_root = find_outputs_and_source(args)
    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] outputs: {outputs}")
    print(f"[{PHASE}] source_root: {source_root}")

    eq_manifest = load_eq_manifest(outputs)
    eq_ready = bool(eq_manifest.get("contract_shadow_ready"))
    contract = eq_manifest.get("contract", {}) or {}
    exact_pair_pass = bool(contract.get("exact_pair_pass", eq_manifest.get("exact_pair_pass", False)))
    metric_pass = bool(contract.get("metric_pass", eq_manifest.get("metric_pass", False)))
    fingerprint = str((contract.get("exact_result", {}) or {}).get("route_fingerprint", EXPECTED_FINGERPRINT))

    raw_targets = list(eq_manifest.get("targets", []))
    rows = [score_target(t, source_root) for t in raw_targets]
    rows.sort(key=lambda r: float(r.get("promotion_score", 0.0)), reverse=True)
    selected = pick_target(rows, args.target)

    ready_count = sum(1 for r in rows if r.get("promotion_ready"))
    syntax_failures = [r for r in rows if r.get("shadow_exists") and not r.get("shadow_syntax_ok")]
    guard_failures = [r for r in rows if r.get("shadow_exists") and not r.get("guard_tokens_present")]
    promotion_plan_ready = bool(eq_ready and exact_pair_pass and metric_pass and selected and not syntax_failures and not guard_failures)

    csv_path = outputs / "phase26er_lite_promotion_candidates.csv"
    manifest_path = outputs / "phase26er_lite_shadow_promotion_manifest.json"
    plan_path = outputs / "phase26er_lite_shadow_promotion_plan.md"
    ps1_path = outputs / "phase26er_lite_apply_top_shadow_patch.ps1"
    plot_path = outputs / "phase26er_lite_promotion_candidates.png"

    write_csv(csv_path, rows)
    make_plot(plot_path, rows)
    write_plan(plan_path, rows, selected, eq_manifest)
    if selected:
        write_apply_script(ps1_path, selected, source_root)

    applied_result: Dict[str, Any] = {"applied": False}
    if args.apply:
        if not promotion_plan_ready:
            raise RuntimeError("Refusing --apply because promotion_plan_ready is false")
        assert selected is not None
        applied_result = apply_shadow(selected)

    out_manifest = {
        "phase": PHASE,
        "title": TITLE,
        "PROMOTION_PLAN_READY": promotion_plan_ready,
        "APPLIED_LIVE_PATCH": bool(applied_result.get("applied")),
        "checks": {
            "eq_contract_shadow_ready": eq_ready,
            "exact_pair_pass": exact_pair_pass,
            "metric_pass": metric_pass,
            "fingerprint_matches_expected": fingerprint == EXPECTED_FINGERPRINT,
            "ready_candidates": ready_count,
            "shadow_syntax_failures": len(syntax_failures),
            "guard_token_failures": len(guard_failures),
        },
        "selected": selected,
        "applied_result": applied_result,
        "outputs": {
            "promotion_candidates_csv": str(csv_path),
            "promotion_plan_md": str(plan_path),
            "apply_script_ps1": str(ps1_path),
            "plot": str(plot_path),
        },
    }
    dump_json(manifest_path, out_manifest)

    print(f"[{PHASE}] PROMOTION_PLAN_READY={promotion_plan_ready}")
    print(f"[{PHASE}] ready candidates: {ready_count}/{len(rows)}")
    if selected:
        print(f"[{PHASE}] selected target: {selected['file']} score={selected['promotion_score']:.3f}")
        print(f"[{PHASE}] source: {selected['source_path']}")
        print(f"[{PHASE}] shadow: {selected['shadow_file']}")
    else:
        print(f"[{PHASE}] selected target: <none>")
    if args.apply:
        print(f"[{PHASE}] APPLIED_LIVE_PATCH={bool(applied_result.get('applied'))}")
        print(f"[{PHASE}] backup: {applied_result.get('backup')}")
    print(f"[{PHASE}] wrote candidates: {csv_path}")
    print(f"[{PHASE}] wrote plan: {plan_path}")
    print(f"[{PHASE}] wrote apply script: {ps1_path}")
    print(f"[{PHASE}] wrote manifest: {manifest_path}")
    print(f"[{PHASE}] wrote outputs to: {outputs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
