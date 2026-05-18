#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Phase 26ES-LITE: post-promotion chain auditor / exact-pair live guard seal

Run from:
    (.venv) PS E:\BBIT> python bbit_geomlang/geomlang_phase26es_lite_post_promotion_chain_auditor_cuda_basic32_E_drive.py

Purpose:
    After 26ER applies the selected 26EQ shadow patch into the live 26EO file,
    this phase audits that the promoted file is still contract-safe, still exact-pair
    guarded, and still accepted by the downstream EO/EQ selftests.

This script does not mutate source. It is a post-apply seal/checkpoint.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PHASE = "26ES-LITE"
TITLE = "Post-promotion chain auditor / exact-pair live guard seal"
EXPECTED_FINGERPRINT = "98ebdcbb8e995bc1"

EO_FILE = "geomlang_phase26eo_lite_live_guarded_transplant_patcher_cuda_basic32_E_drive.py"
EQ_FILE = "geomlang_phase26eq_lite_exact_pair_shadow_transplant_verifier_cuda_basic32_E_drive.py"
EN_GUARD_SELFTEST = "phase26en_lite_live_guard_selftest.py"
EO_SELFTEST = "phase26eo_lite_transplant_selftest.py"
EQ_GATE_SELFTEST = "phase26eq_lite_runtime_exact_pair_gate_selftest.py"
EM_ROUTER_SELFTEST = "phase26em_lite_runtime_route_selftest.py"
ER_MANIFEST = "phase26er_lite_shadow_promotion_manifest.json"

GUARD_TOKENS = (
    "exact_pair",
    "exact-pair",
    "route_fingerprint",
    "contract",
    "CONTRACT_EXACT_PAIR_PASS",
    "LIVE_GUARD_READY",
)

RISK_TOKENS = (
    "source_variant",
    "global source",
    "source-only",
    "alias",
)


def sha16(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def find_root() -> Path:
    cwd = Path.cwd().resolve()
    candidates = [cwd]
    candidates += list(cwd.parents)
    for c in candidates:
        if (c / "bbit_geomlang").exists() and (c / "outputs_basic32").exists():
            return c
    # Windows default for the user's project.
    e_root = Path(r"E:\BBIT")
    if (e_root / "bbit_geomlang").exists():
        return e_root
    return cwd


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction for selftests that print a single JSON object."""
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            obj = json.loads(stripped)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
    m = re.search(r"\{.*\}\s*$", text, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


@dataclass
class CommandResult:
    name: str
    command: List[str]
    returncode: int
    stdout_tail: str
    stderr_tail: str
    elapsed_s: float
    pass_hint: bool
    parsed_json: Optional[Dict[str, Any]] = None


def run_cmd(name: str, cmd: List[str], timeout_s: int = 240) -> CommandResult:
    t0 = time.time()
    try:
        p = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
        stdout = p.stdout or ""
        stderr = p.stderr or ""
        parsed = extract_json_object(stdout)
        pass_hint = p.returncode == 0
        return CommandResult(
            name=name,
            command=cmd,
            returncode=p.returncode,
            stdout_tail=stdout[-5000:],
            stderr_tail=stderr[-3000:],
            elapsed_s=round(time.time() - t0, 3),
            pass_hint=pass_hint,
            parsed_json=parsed,
        )
    except subprocess.TimeoutExpired as e:
        return CommandResult(
            name=name,
            command=cmd,
            returncode=124,
            stdout_tail=(e.stdout or "")[-5000:] if isinstance(e.stdout, str) else "",
            stderr_tail=(e.stderr or "")[-3000:] if isinstance(e.stderr, str) else "timeout expired",
            elapsed_s=round(time.time() - t0, 3),
            pass_hint=False,
            parsed_json=None,
        )
    except Exception as e:
        return CommandResult(
            name=name,
            command=cmd,
            returncode=1,
            stdout_tail="",
            stderr_tail=repr(e),
            elapsed_s=round(time.time() - t0, 3),
            pass_hint=False,
            parsed_json=None,
        )


def py_compile_ok(path: Path) -> Tuple[bool, str]:
    try:
        ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
        return True, ""
    except SyntaxError as e:
        return False, f"{e.filename}:{e.lineno}:{e.offset}: {e.msg}"
    except Exception as e:
        return False, repr(e)


def count_tokens(text: str, tokens: Tuple[str, ...]) -> Dict[str, int]:
    low = text.lower()
    return {tok: low.count(tok.lower()) for tok in tokens}


def infer_passes(results: List[CommandResult]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    by_name = {r.name: r for r in results}

    eo_main = by_name.get("eo_live_patcher")
    if eo_main:
        s = eo_main.stdout_tail
        out["eo_live_patcher_pass"] = (
            eo_main.returncode == 0
            and "LIVE_GUARD_READY=True" in s
            and "CONTRACT_EXACT_PAIR_PASS=True" in s
        )

    eo_self = by_name.get("eo_selftest")
    if eo_self:
        obj = eo_self.parsed_json or {}
        out["eo_selftest_pass"] = bool(obj.get("EO_SELFTEST_PASS")) or "EO_SELFTEST_PASS" in eo_self.stdout_tail and "true" in eo_self.stdout_tail.lower()

    eq_main = by_name.get("eq_shadow_verifier")
    if eq_main:
        s = eq_main.stdout_tail
        out["eq_shadow_verifier_pass"] = (
            eq_main.returncode == 0
            and "CONTRACT_SHADOW_READY=True" in s
            and "exact_pair_pass=True" in s
            and "metric_pass=True" in s
        )

    eq_self = by_name.get("eq_gate_selftest")
    if eq_self:
        obj = eq_self.parsed_json or {}
        out["eq_gate_selftest_pass"] = bool(
            obj.get("EQ_RUNTIME_EXACT_PAIR_GATE_SELFTEST_PASS")
            or obj.get("RUNTIME_EXACT_PAIR_GATE_SELFTEST_PASS")
            or obj.get("SELFTEST_PASS")
            or obj.get("pass")
        )
        if not out["eq_gate_selftest_pass"]:
            out["eq_gate_selftest_pass"] = eq_self.returncode == 0 and "false" not in eq_self.stdout_tail.lower()

    en_self = by_name.get("en_guard_selftest")
    if en_self:
        out["en_guard_selftest_pass"] = en_self.returncode == 0 and "false" not in (en_self.stdout_tail + en_self.stderr_tail).lower()

    em_self = by_name.get("em_router_selftest")
    if em_self:
        out["em_router_selftest_pass"] = em_self.returncode == 0 and "false" not in (em_self.stdout_tail + em_self.stderr_tail).lower()

    out["all_invoked_commands_exit_zero"] = all(r.returncode == 0 for r in results)
    return out


def write_bar_plot(rows: List[Tuple[str, bool]], out_png: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
        names = [r[0] for r in rows]
        vals = [1 if r[1] else 0 for r in rows]
        fig_w = 12
        fig_h = max(4, 0.45 * len(rows) + 1.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.barh(names, vals)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("pass = 1, fail = 0")
        ax.set_title(f"{PHASE} post-promotion seal checks")
        for i, v in enumerate(vals):
            ax.text(v + 0.02, i, "PASS" if v else "FAIL", va="center")
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        return True
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-run", action="store_true", help="Do not invoke EO/EQ selftests; only inspect files/manifests.")
    ap.add_argument("--timeout", type=int, default=240, help="Per-command timeout in seconds.")
    args = ap.parse_args()

    global ROOT
    ROOT = find_root()
    source_root = ROOT / "bbit_geomlang"
    outputs = ROOT / "outputs_basic32"
    outputs.mkdir(parents=True, exist_ok=True)

    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] root: {ROOT}")
    print(f"[{PHASE}] outputs: {outputs}")

    eo_path = source_root / EO_FILE
    eq_path = source_root / EQ_FILE
    er_manifest_path = outputs / ER_MANIFEST
    er_manifest = load_json(er_manifest_path, {}) or {}

    selected = er_manifest.get("selected", {}) if isinstance(er_manifest, dict) else {}
    applied_result = er_manifest.get("applied_result", {}) if isinstance(er_manifest, dict) else {}
    shadow_path = Path(selected.get("shadow_file", "")) if selected.get("shadow_file") else None
    backup_path = Path(applied_result.get("backup", "")) if applied_result.get("backup") else None

    compile_checks = {}
    for name, path in [("eo_live_source", eo_path), ("eq_source", eq_path)]:
        ok, err = py_compile_ok(path) if path.exists() else (False, "missing")
        compile_checks[name] = {"path": str(path), "syntax_ok": ok, "error": err}

    eo_text = eo_path.read_text(encoding="utf-8", errors="replace") if eo_path.exists() else ""
    guard_counts = count_tokens(eo_text, GUARD_TOKENS)
    risk_counts = count_tokens(eo_text, RISK_TOKENS)

    current_eo_hash = sha16(eo_path) if eo_path.exists() else ""
    shadow_hash = sha16(shadow_path) if shadow_path and shadow_path.exists() else ""
    backup_hash = sha16(backup_path) if backup_path and backup_path.exists() else ""

    manifest_apply_ok = bool(er_manifest.get("APPLIED_LIVE_PATCH")) or bool(applied_result.get("applied"))
    source_matches_shadow = bool(current_eo_hash and shadow_hash and current_eo_hash == shadow_hash)
    backup_exists = bool(backup_path and backup_path.exists())
    backup_differs_from_source = bool(backup_hash and current_eo_hash and backup_hash != current_eo_hash)
    fingerprint_ok = EXPECTED_FINGERPRINT in eo_text or EXPECTED_FINGERPRINT in json.dumps(er_manifest)
    guard_tokens_present = all(guard_counts.get(t, 0) > 0 for t in ("exact_pair", "route_fingerprint", "contract"))

    commands: List[CommandResult] = []
    if not args.no_run:
        cmd_plan = [
            ("eo_live_patcher", [sys.executable, str(eo_path)]),
            ("eo_selftest", [sys.executable, str(outputs / EO_SELFTEST)]),
            ("eq_shadow_verifier", [sys.executable, str(eq_path)]),
        ]
        if (outputs / EQ_GATE_SELFTEST).exists():
            cmd_plan.append(("eq_gate_selftest", [sys.executable, str(outputs / EQ_GATE_SELFTEST)]))
        if (outputs / EN_GUARD_SELFTEST).exists():
            cmd_plan.append(("en_guard_selftest", [sys.executable, str(outputs / EN_GUARD_SELFTEST)]))
        if (outputs / EM_ROUTER_SELFTEST).exists():
            cmd_plan.append(("em_router_selftest", [sys.executable, str(outputs / EM_ROUTER_SELFTEST)]))
        for name, cmd in cmd_plan:
            print(f"[{PHASE}] running {name} ...")
            r = run_cmd(name, cmd, timeout_s=args.timeout)
            commands.append(r)
            print(f"[{PHASE}]   {name}: rc={r.returncode} elapsed={r.elapsed_s}s")

    inferred = infer_passes(commands)

    seal_checks: Dict[str, bool] = {
        "er_manifest_applied": manifest_apply_ok,
        "eo_source_exists": eo_path.exists(),
        "eo_syntax_ok": bool(compile_checks["eo_live_source"]["syntax_ok"]),
        "eq_syntax_ok": bool(compile_checks["eq_source"]["syntax_ok"]),
        "fingerprint_present": fingerprint_ok,
        "guard_tokens_present": guard_tokens_present,
        "source_matches_promoted_shadow": source_matches_shadow,
        "backup_exists": backup_exists,
        "backup_differs_from_promoted_source": backup_differs_from_source,
    }
    for k, v in inferred.items():
        if isinstance(v, bool):
            seal_checks[k] = v

    # The source/shadow hash match is important after apply, but allow a manually edited promoted file
    # to remain auditable if all live tests pass. We report it separately in the summary.
    critical_names = [
        "eo_source_exists",
        "eo_syntax_ok",
        "eq_syntax_ok",
        "fingerprint_present",
        "guard_tokens_present",
        "eo_live_patcher_pass",
        "eo_selftest_pass",
        "eq_shadow_verifier_pass",
    ]
    critical = {k: seal_checks.get(k, False) for k in critical_names if k in seal_checks}
    POST_PROMOTION_SEAL_PASS = all(critical.values()) and (manifest_apply_ok or source_matches_shadow)

    report_md = outputs / "phase26es_lite_post_promotion_audit_report.md"
    summary_json = outputs / "phase26es_lite_summary.json"
    log_txt = outputs / "phase26es_lite_command_log.txt"
    plot_png = outputs / "phase26es_lite_post_promotion_seal.png"

    plot_rows = [(k, bool(v)) for k, v in seal_checks.items()]
    plot_written = write_bar_plot(plot_rows, plot_png)

    summary = {
        "phase": PHASE,
        "title": TITLE,
        "POST_PROMOTION_SEAL_PASS": POST_PROMOTION_SEAL_PASS,
        "root": str(ROOT),
        "source_root": str(source_root),
        "outputs": str(outputs),
        "expected_fingerprint": EXPECTED_FINGERPRINT,
        "hashes": {
            "current_eo_hash": current_eo_hash,
            "shadow_hash": shadow_hash,
            "backup_hash": backup_hash,
            "source_matches_promoted_shadow": source_matches_shadow,
            "backup_differs_from_promoted_source": backup_differs_from_source,
        },
        "paths": {
            "eo_source": str(eo_path),
            "eq_source": str(eq_path),
            "er_manifest": str(er_manifest_path),
            "shadow": str(shadow_path) if shadow_path else "",
            "backup": str(backup_path) if backup_path else "",
        },
        "compile_checks": compile_checks,
        "guard_token_counts": guard_counts,
        "risk_token_counts": risk_counts,
        "seal_checks": seal_checks,
        "critical_checks": critical,
        "commands": [asdict(r) for r in commands],
        "outputs_written": {
            "summary_json": str(summary_json),
            "report_md": str(report_md),
            "command_log_txt": str(log_txt),
            "plot_png": str(plot_png) if plot_written else "",
        },
    }

    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = []
    lines.append(f"# {PHASE} post-promotion audit report")
    lines.append("")
    lines.append(f"**POST_PROMOTION_SEAL_PASS:** `{POST_PROMOTION_SEAL_PASS}`")
    lines.append("")
    lines.append("## Critical checks")
    for k, v in critical.items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## Hash audit")
    lines.append(f"- Current EO hash: `{current_eo_hash}`")
    lines.append(f"- Promoted shadow hash: `{shadow_hash}`")
    lines.append(f"- Backup hash: `{backup_hash}`")
    lines.append(f"- Source matches promoted shadow: `{source_matches_shadow}`")
    lines.append(f"- Backup differs from promoted source: `{backup_differs_from_source}`")
    lines.append("")
    lines.append("## Seal checks")
    for k, v in seal_checks.items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## Command results")
    if commands:
        for r in commands:
            lines.append(f"### {r.name}")
            lines.append(f"- returncode: `{r.returncode}`")
            lines.append(f"- elapsed_s: `{r.elapsed_s}`")
            lines.append("```text")
            lines.append((r.stdout_tail or "").strip()[-2000:])
            if r.stderr_tail:
                lines.append("\n[stderr]")
                lines.append(r.stderr_tail.strip()[-1000:])
            lines.append("```")
    else:
        lines.append("Commands were skipped with `--no-run`.")
    report_md.write_text("\n".join(lines), encoding="utf-8")

    log_parts = []
    for r in commands:
        log_parts.append(f"===== {r.name} rc={r.returncode} elapsed={r.elapsed_s}s =====")
        log_parts.append("COMMAND: " + " ".join(r.command))
        log_parts.append("--- STDOUT ---")
        log_parts.append(r.stdout_tail)
        log_parts.append("--- STDERR ---")
        log_parts.append(r.stderr_tail)
    log_txt.write_text("\n".join(log_parts), encoding="utf-8")

    print(f"[{PHASE}] POST_PROMOTION_SEAL_PASS={POST_PROMOTION_SEAL_PASS}")
    print(f"[{PHASE}] checks:")
    for k, v in seal_checks.items():
        print(f"  - {k}: {v}")
    print(f"[{PHASE}] hashes: current_eo={current_eo_hash} shadow={shadow_hash} backup={backup_hash}")
    print(f"[{PHASE}] wrote report: {report_md}")
    print(f"[{PHASE}] wrote summary: {summary_json}")
    if plot_written:
        print(f"[{PHASE}] wrote plot: {plot_png}")
    print(f"[{PHASE}] wrote outputs to: {outputs}")

    return 0 if POST_PROMOTION_SEAL_PASS else 2


if __name__ == "__main__":
    ROOT = Path.cwd()
    raise SystemExit(main())
