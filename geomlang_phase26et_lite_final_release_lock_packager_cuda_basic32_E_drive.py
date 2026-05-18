#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Phase 26ET-LITE: final release lock packager / exact-pair route deployment bundle

Run from:
    (.venv) PS E:\BBIT> python bbit_geomlang/geomlang_phase26et_lite_final_release_lock_packager_cuda_basic32_E_drive.py

Optional:
    (.venv) PS E:\BBIT> python bbit_geomlang/geomlang_phase26et_lite_final_release_lock_packager_cuda_basic32_E_drive.py --no-run
    (.venv) PS E:\BBIT> python bbit_geomlang/geomlang_phase26et_lite_final_release_lock_packager_cuda_basic32_E_drive.py --zip

Purpose:
    26ES proved the promoted 26EO live source is sealed after the ER shadow promotion.
    26ET freezes that state into a deployable release bundle:
      - verifies the 26ES seal and reruns the live guard chain unless --no-run is used
      - hashes the promoted live source, exact-pair helpers, guards, selectors, contracts, and manifests
      - copies the minimum transplant/runtime files into a release bundle directory
      - writes a final release manifest, README, PowerShell smoke runner, rollback note, and optional zip

This script does not modify bbit_geomlang source. It only writes under outputs_basic32.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PHASE = "26ET-LITE"
TITLE = "Final release lock packager / exact-pair route deployment bundle"
EXPECTED_FINGERPRINT = "98ebdcbb8e995bc1"

# Live source files that define or verify the current promoted chain.
SOURCE_FILES = [
    "geomlang_phase26eo_lite_live_guarded_transplant_patcher_cuda_basic32_E_drive.py",
    "geomlang_phase26eq_lite_exact_pair_shadow_transplant_verifier_cuda_basic32_E_drive.py",
    "geomlang_phase26er_lite_shadow_promoter_exact_pair_live_guard_cuda_basic32_E_drive.py",
    "geomlang_phase26es_lite_post_promotion_chain_auditor_cuda_basic32_E_drive.py",
]

# Output/runtime files created by the previous phases and needed for transplant/use/audit.
OUTPUT_FILES = [
    "phase26em_lite_runtime_route_router.py",
    "phase26em_lite_runtime_route_selftest.py",
    "phase26em_lite_summary.json",
    "phase26em_lite_transplant_manifest.json",
    "phase26en_lite_live_contract_guard.py",
    "phase26en_lite_live_guard_selftest.py",
    "phase26en_lite_live_transplant_checklist.md",
    "phase26en_lite_live_transplant_manifest.json",
    "phase26eo_lite_exact_pair_route_helper.py",
    "phase26eo_lite_transplant_selftest.py",
    "phase26eo_lite_transplant_patch_plan.md",
    "phase26eo_lite_transplant_patch_plan.json",
    "phase26eo_lite_summary.json",
    "phase26eq_lite_runtime_exact_pair_gate.py",
    "phase26eq_lite_runtime_exact_pair_gate_selftest.py",
    "phase26eq_lite_shadow_patch_bundle.md",
    "phase26eq_lite_shadow_transplant_manifest.json",
    "phase26eq_lite_summary.json",
    "phase26er_lite_shadow_promotion_manifest.json",
    "phase26er_lite_shadow_promotion_manifest(1).json",
    "phase26er_lite_shadow_promotion_plan.md",
    "phase26er_lite_shadow_promotion_plan(1).md",
    "phase26es_lite_post_promotion_audit_report.md",
    "phase26es_lite_summary.json",
    "phase26es_lite_command_log.txt",
]

REQUIRED_OUTPUT_FILES = [
    "phase26em_lite_runtime_route_router.py",
    "phase26en_lite_live_contract_guard.py",
    "phase26eo_lite_exact_pair_route_helper.py",
    "phase26eo_lite_transplant_selftest.py",
    "phase26eq_lite_runtime_exact_pair_gate.py",
    "phase26eq_lite_runtime_exact_pair_gate_selftest.py",
    "phase26es_lite_summary.json",
]

SELFTEST_COMMANDS = [
    ("eo_live_patcher", ["{py}", "{src}/geomlang_phase26eo_lite_live_guarded_transplant_patcher_cuda_basic32_E_drive.py"]),
    ("eo_selftest", ["{py}", "{out}/phase26eo_lite_transplant_selftest.py"]),
    ("eq_shadow_verifier", ["{py}", "{src}/geomlang_phase26eq_lite_exact_pair_shadow_transplant_verifier_cuda_basic32_E_drive.py"]),
    ("eq_gate_selftest", ["{py}", "{out}/phase26eq_lite_runtime_exact_pair_gate_selftest.py"]),
    ("en_guard_selftest", ["{py}", "{out}/phase26en_lite_live_guard_selftest.py"]),
    ("em_router_selftest", ["{py}", "{out}/phase26em_lite_runtime_route_selftest.py"]),
]


@dataclass
class FileRecord:
    role: str
    name: str
    source_path: str
    bundle_path: str
    exists: bool
    bytes: int
    sha256: str
    sha16: str
    syntax_ok: Optional[bool]
    syntax_error: str
    fingerprint_present: bool


@dataclass
class CommandRecord:
    name: str
    command: List[str]
    returncode: int
    elapsed_s: float
    pass_hint: bool
    stdout_tail: str
    stderr_tail: str


def find_root() -> Path:
    cwd = Path.cwd().resolve()
    for c in [cwd, *cwd.parents]:
        if (c / "bbit_geomlang").exists() and (c / "outputs_basic32").exists():
            return c
    e_root = Path(r"E:\BBIT")
    if (e_root / "bbit_geomlang").exists():
        return e_root
    return cwd


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(read_text(path))
    except Exception:
        return default


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def syntax_check(path: Path) -> Tuple[Optional[bool], str]:
    if path.suffix.lower() != ".py":
        return None, ""
    try:
        ast.parse(read_text(path), filename=str(path))
        return True, ""
    except SyntaxError as e:
        return False, f"{e.filename}:{e.lineno}:{e.offset}: {e.msg}"
    except Exception as e:
        return False, repr(e)


def copy_and_record(role: str, src: Path, dst_root: Path, rel_dir: str) -> FileRecord:
    dst_dir = dst_root / rel_dir
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    exists = src.exists()
    syntax_ok: Optional[bool] = None
    syntax_error = ""
    digest = ""
    size = 0
    fp = False
    if exists:
        shutil.copy2(src, dst)
        size = dst.stat().st_size
        digest = sha256_file(dst)
        syntax_ok, syntax_error = syntax_check(dst)
        try:
            fp = EXPECTED_FINGERPRINT in read_text(dst)
        except Exception:
            fp = False
    return FileRecord(
        role=role,
        name=src.name,
        source_path=str(src),
        bundle_path=str(dst),
        exists=exists,
        bytes=size,
        sha256=digest,
        sha16=digest[:16] if digest else "",
        syntax_ok=syntax_ok,
        syntax_error=syntax_error,
        fingerprint_present=fp,
    )


def render_cmd(template: List[str], py: str, src: Path, out: Path) -> List[str]:
    return [p.replace("{py}", py).replace("{src}", str(src)).replace("{out}", str(out)) for p in template]


def run_command(name: str, cmd: List[str], cwd: Path, timeout_s: int) -> CommandRecord:
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, timeout=timeout_s)
        stdout = p.stdout or ""
        stderr = p.stderr or ""
        low = (stdout + "\n" + stderr).lower()
        pass_hint = p.returncode == 0 and "false" not in low and "traceback" not in low
        # Main phase runners print explicit positive contract markers; accept those even with warnings.
        if name == "eo_live_patcher":
            pass_hint = p.returncode == 0 and "LIVE_GUARD_READY=True" in stdout and "CONTRACT_EXACT_PAIR_PASS=True" in stdout
        elif name == "eq_shadow_verifier":
            pass_hint = p.returncode == 0 and "CONTRACT_SHADOW_READY=True" in stdout and "exact_pair_pass=True" in stdout and "metric_pass=True" in stdout
        return CommandRecord(name, cmd, p.returncode, round(time.time() - t0, 3), pass_hint, stdout[-5000:], stderr[-3000:])
    except subprocess.TimeoutExpired as e:
        return CommandRecord(name, cmd, 124, round(time.time() - t0, 3), False, str(e.stdout or "")[-5000:], "timeout expired")
    except Exception as e:
        return CommandRecord(name, cmd, 1, round(time.time() - t0, 3), False, "", repr(e))


def write_csv(records: List[FileRecord], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()) if records else ["empty"])
        w.writeheader()
        for r in records:
            w.writerow(asdict(r))


def write_plot(checks: Dict[str, bool], out_png: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
        rows = list(checks.items())
        names = [k for k, _ in rows]
        vals = [1 if v else 0 for _, v in rows]
        fig_h = max(5, 0.42 * len(rows) + 1.5)
        fig, ax = plt.subplots(figsize=(12, fig_h))
        ax.barh(names, vals)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("pass = 1, fail = 0")
        ax.set_title(f"{PHASE} final release lock checks")
        for i, v in enumerate(vals):
            ax.text(v + 0.02, i, "PASS" if v else "FAIL", va="center")
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        return True
    except Exception:
        return False


def zip_dir(src_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in src_dir.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(src_dir.parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-run", action="store_true", help="Skip rerunning EO/EQ/EN/EM selftests.")
    ap.add_argument("--timeout", type=int, default=240, help="Per-command timeout seconds.")
    ap.add_argument("--zip", action="store_true", help="Also write a zip archive of the ET release bundle.")
    args = ap.parse_args()

    root = find_root()
    source_root = root / "bbit_geomlang"
    outputs = root / "outputs_basic32"
    release_dir = outputs / "phase26et_lite_release_bundle"
    source_bundle = release_dir / "bbit_geomlang"
    output_bundle = release_dir / "outputs_basic32"
    release_dir.mkdir(parents=True, exist_ok=True)
    source_bundle.mkdir(parents=True, exist_ok=True)
    output_bundle.mkdir(parents=True, exist_ok=True)

    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] root: {root}")
    print(f"[{PHASE}] outputs: {outputs}")
    print(f"[{PHASE}] release_dir: {release_dir}")

    es_summary_path = outputs / "phase26es_lite_summary.json"
    es_summary = load_json(es_summary_path, {}) or {}
    es_pass = bool(es_summary.get("POST_PROMOTION_SEAL_PASS"))

    command_records: List[CommandRecord] = []
    if not args.no_run:
        for name, templ in SELFTEST_COMMANDS:
            cmd = render_cmd(templ, sys.executable, source_root, outputs)
            if not Path(cmd[1]).exists():
                command_records.append(CommandRecord(name, cmd, 127, 0.0, False, "", "target script missing"))
                continue
            print(f"[{PHASE}] running {name} ...")
            rec = run_command(name, cmd, root, args.timeout)
            command_records.append(rec)
            print(f"[{PHASE}]   {name}: rc={rec.returncode} pass={rec.pass_hint} elapsed={rec.elapsed_s}s")

    records: List[FileRecord] = []
    for name in SOURCE_FILES:
        records.append(copy_and_record("source", source_root / name, release_dir, "bbit_geomlang"))
    for name in OUTPUT_FILES:
        src = outputs / name
        if src.exists() or name in REQUIRED_OUTPUT_FILES:
            records.append(copy_and_record("output", src, release_dir, "outputs_basic32"))

    required_sources_exist = all((source_root / n).exists() for n in SOURCE_FILES[:2])
    required_outputs_exist = all((outputs / n).exists() for n in REQUIRED_OUTPUT_FILES)
    all_python_syntax_ok = all(r.syntax_ok is not False for r in records)
    fingerprint_found_anywhere = any(r.fingerprint_present for r in records) or EXPECTED_FINGERPRINT in json.dumps(es_summary)
    command_chain_pass = True if args.no_run else all(r.pass_hint for r in command_records)
    bundle_has_runtime_router = (output_bundle / "phase26em_lite_runtime_route_router.py").exists()
    bundle_has_exact_pair_gate = (output_bundle / "phase26eq_lite_runtime_exact_pair_gate.py").exists()
    bundle_has_live_guard = (output_bundle / "phase26en_lite_live_contract_guard.py").exists()
    bundle_has_eo_helper = (output_bundle / "phase26eo_lite_exact_pair_route_helper.py").exists()

    release_checks: Dict[str, bool] = {
        "es_post_promotion_seal_pass": es_pass,
        "required_sources_exist": required_sources_exist,
        "required_outputs_exist": required_outputs_exist,
        "all_python_syntax_ok": all_python_syntax_ok,
        "fingerprint_found": fingerprint_found_anywhere,
        "command_chain_pass": command_chain_pass,
        "bundle_has_runtime_router": bundle_has_runtime_router,
        "bundle_has_exact_pair_gate": bundle_has_exact_pair_gate,
        "bundle_has_live_guard": bundle_has_live_guard,
        "bundle_has_eo_helper": bundle_has_eo_helper,
    }

    RELEASE_LOCK_PASS = all(release_checks.values())

    files_csv = outputs / "phase26et_lite_release_files.csv"
    if records:
        write_csv(records, files_csv)

    command_log = outputs / "phase26et_lite_command_log.txt"
    command_log.write_text(
        "\n".join(
            [
                f"===== {r.name} rc={r.returncode} pass={r.pass_hint} elapsed={r.elapsed_s}s =====\n"
                f"COMMAND: {' '.join(r.command)}\n--- STDOUT ---\n{r.stdout_tail}\n--- STDERR ---\n{r.stderr_tail}\n"
                for r in command_records
            ]
        ),
        encoding="utf-8",
    )

    smoke_ps1 = release_dir / "run_phase26et_release_smoke.ps1"
    smoke_ps1.write_text(
        "\n".join([
            "$ErrorActionPreference = 'Stop'",
            "$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)",
            "Write-Host '[26ET-LITE] Release smoke using root:' $Root",
            "python (Join-Path $Root 'outputs_basic32/phase26eo_lite_transplant_selftest.py')",
            "python (Join-Path $Root 'outputs_basic32/phase26eq_lite_runtime_exact_pair_gate_selftest.py')",
            "python (Join-Path $Root 'outputs_basic32/phase26en_lite_live_guard_selftest.py')",
            "python (Join-Path $Root 'outputs_basic32/phase26em_lite_runtime_route_selftest.py')",
            "Write-Host '[26ET-LITE] Release smoke completed.'",
            "",
        ]),
        encoding="utf-8",
    )

    rollback_note = release_dir / "ROLLBACK_NOTE.md"
    # Pull backup from ES summary if available.
    backup_path = (((es_summary.get("paths") or {}).get("backup")) if isinstance(es_summary, dict) else "") or ""
    rollback_note.write_text(
        "\n".join([
            f"# {PHASE} rollback note",
            "",
            "This ET bundle is a release snapshot. It does not automatically restore source files.",
            "",
            f"Last known ER/ES backup path: `{backup_path}`" if backup_path else "No backup path was found in the ES summary.",
            "",
            "To roll back manually, copy the backup file over the promoted EO source file, then rerun:",
            "",
            "```powershell",
            "python bbit_geomlang/geomlang_phase26es_lite_post_promotion_chain_auditor_cuda_basic32_E_drive.py",
            "```",
            "",
        ]),
        encoding="utf-8",
    )

    readme = release_dir / "README_PHASE26ET_LITE.md"
    readme.write_text(
        "\n".join([
            f"# {PHASE} release bundle",
            "",
            f"**RELEASE_LOCK_PASS:** `{RELEASE_LOCK_PASS}`",
            f"**Route fingerprint:** `{EXPECTED_FINGERPRINT}`",
            "",
            "## What this bundle contains",
            "",
            "- Promoted live EO guarded transplant source.",
            "- EQ exact-pair runtime gate and verifier artifacts.",
            "- EN live contract guard artifacts.",
            "- EM runtime route router artifacts.",
            "- ES post-promotion audit proof files.",
            "- File hash manifest and smoke runner.",
            "",
            "## Smoke test",
            "",
            "From `E:\\BBIT`:",
            "",
            "```powershell",
            "python bbit_geomlang/geomlang_phase26et_lite_final_release_lock_packager_cuda_basic32_E_drive.py",
            "```",
            "",
            "Or from inside the copied release bundle, run:",
            "",
            "```powershell",
            ".\\run_phase26et_release_smoke.ps1",
            "```",
            "",
            "## Contract meaning",
            "",
            "The release is valid only when exact `(envelope_label, source_variant)` rows are used. Source-only/global alias smoke is intentionally not a deployment contract.",
            "",
        ]),
        encoding="utf-8",
    )

    plot_png = outputs / "phase26et_lite_release_lock_checks.png"
    plot_written = write_plot(release_checks, plot_png)

    manifest_path = outputs / "phase26et_lite_release_manifest.json"
    summary_path = outputs / "phase26et_lite_summary.json"
    report_path = outputs / "phase26et_lite_release_report.md"
    bundle_manifest = release_dir / "phase26et_lite_release_manifest.json"

    zip_path = outputs / "phase26et_lite_release_bundle.zip"
    zip_written = False
    if args.zip:
        zip_dir(release_dir, zip_path)
        zip_written = zip_path.exists()

    manifest = {
        "phase": PHASE,
        "title": TITLE,
        "RELEASE_LOCK_PASS": RELEASE_LOCK_PASS,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": str(root),
        "source_root": str(source_root),
        "outputs": str(outputs),
        "release_dir": str(release_dir),
        "expected_fingerprint": EXPECTED_FINGERPRINT,
        "release_checks": release_checks,
        "es_summary": {
            "path": str(es_summary_path),
            "POST_PROMOTION_SEAL_PASS": es_pass,
            "hashes": es_summary.get("hashes", {}) if isinstance(es_summary, dict) else {},
            "paths": es_summary.get("paths", {}) if isinstance(es_summary, dict) else {},
        },
        "files": [asdict(r) for r in records],
        "commands": [asdict(r) for r in command_records],
        "outputs_written": {
            "release_dir": str(release_dir),
            "release_manifest": str(manifest_path),
            "bundle_manifest": str(bundle_manifest),
            "summary": str(summary_path),
            "report": str(report_path),
            "files_csv": str(files_csv),
            "command_log": str(command_log),
            "plot": str(plot_png) if plot_written else "",
            "zip": str(zip_path) if zip_written else "",
        },
    }

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    bundle_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report_lines = [
        f"# {PHASE} final release report",
        "",
        f"**RELEASE_LOCK_PASS:** `{RELEASE_LOCK_PASS}`",
        f"**Route fingerprint:** `{EXPECTED_FINGERPRINT}`",
        "",
        "## Release checks",
    ]
    for k, v in release_checks.items():
        report_lines.append(f"- `{k}`: `{v}`")
    report_lines += ["", "## Key files"]
    for r in records:
        status = "OK" if r.exists and r.syntax_ok is not False else "CHECK"
        report_lines.append(f"- `{status}` `{r.role}/{r.name}` sha16=`{r.sha16}` bytes=`{r.bytes}`")
    report_lines += ["", "## Command smoke"]
    if command_records:
        for r in command_records:
            report_lines.append(f"- `{r.name}`: rc=`{r.returncode}` pass=`{r.pass_hint}` elapsed=`{r.elapsed_s}s`")
    else:
        report_lines.append("- Commands skipped with `--no-run`.")
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"[{PHASE}] RELEASE_LOCK_PASS={RELEASE_LOCK_PASS}")
    print(f"[{PHASE}] checks:")
    for k, v in release_checks.items():
        print(f"  - {k}: {v}")
    print(f"[{PHASE}] copied files: {sum(1 for r in records if r.exists)}/{len(records)}")
    print(f"[{PHASE}] wrote release dir: {release_dir}")
    print(f"[{PHASE}] wrote manifest: {manifest_path}")
    print(f"[{PHASE}] wrote report: {report_path}")
    if plot_written:
        print(f"[{PHASE}] wrote plot: {plot_png}")
    if zip_written:
        print(f"[{PHASE}] wrote zip: {zip_path}")
    print(f"[{PHASE}] wrote outputs to: {outputs}")

    return 0 if RELEASE_LOCK_PASS else 2


if __name__ == "__main__":
    raise SystemExit(main())
