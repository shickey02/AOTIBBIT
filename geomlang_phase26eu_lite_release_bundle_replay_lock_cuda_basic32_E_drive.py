#!/usr/bin/env python3
r"""
26EU-LITE — Release bundle replay lock / portable seal verifier

Purpose
-------
EU is the final "can I trust the ET release bundle?" stage.

It does not discover a new route. It takes the 26ET release bundle and proves that:
  1. ET says the release lock passed.
  2. The release bundle directory exists.
  3. Every manifest-listed bundled file still exists.
  4. Every manifest-listed bundled file still has the same sha256 hash.
  5. Every bundled Python file compiles.
  6. The core exact-pair guard artifacts are present inside the bundle.
  7. The live post-promotion command chain can still be replayed from E:\BBIT.
  8. A portable checksum catalog + replay README + optional zip archive are emitted.

This is intentionally conservative. EU should be treated as a release seal, not a search phase.
If EU passes, the locked exact-pair route transplant is sealed and packaged.

Run from:
    (.venv) PS E:\BBIT> python bbit_geomlang/geomlang_phase26eu_lite_release_bundle_replay_lock_cuda_basic32_E_drive.py

Optional:
    (.venv) PS E:\BBIT> python bbit_geomlang/geomlang_phase26eu_lite_release_bundle_replay_lock_cuda_basic32_E_drive.py --zip
    (.venv) PS E:\BBIT> python bbit_geomlang/geomlang_phase26eu_lite_release_bundle_replay_lock_cuda_basic32_E_drive.py --skip-live-replay
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PHASE = "26EU-LITE"
TITLE = "Release bundle replay lock / portable seal verifier"
EXPECTED_FINGERPRINT = "98ebdcbb8e995bc1"

ROOT = Path(r"E:\BBIT")
SOURCE_ROOT = ROOT / "bbit_geomlang"
OUTPUTS = ROOT / "outputs_basic32"
ET_MANIFEST = OUTPUTS / "phase26et_lite_release_manifest.json"
ET_SUMMARY = OUTPUTS / "phase26et_lite_summary.json"
ES_SUMMARY = OUTPUTS / "phase26es_lite_summary.json"

DEFAULT_RELEASE_DIR = OUTPUTS / "phase26et_lite_release_bundle"
EU_REPLAY_DIR = OUTPUTS / "phase26eu_lite_release_replay"

CORE_BUNDLE_RELATIVE = [
    Path("bbit_geomlang/geomlang_phase26eo_lite_live_guarded_transplant_patcher_cuda_basic32_E_drive.py"),
    Path("bbit_geomlang/geomlang_phase26eq_lite_exact_pair_shadow_transplant_verifier_cuda_basic32_E_drive.py"),
    Path("bbit_geomlang/geomlang_phase26er_lite_shadow_promoter_exact_pair_live_guard_cuda_basic32_E_drive.py"),
    Path("bbit_geomlang/geomlang_phase26es_lite_post_promotion_chain_auditor_cuda_basic32_E_drive.py"),
    Path("bbit_geomlang/geomlang_phase26et_lite_final_release_lock_packager_cuda_basic32_E_drive.py"),
    Path("outputs_basic32/phase26em_lite_runtime_route_router.py"),
    Path("outputs_basic32/phase26eq_lite_runtime_exact_pair_gate.py"),
    Path("outputs_basic32/phase26en_lite_live_contract_guard.py"),
    Path("outputs_basic32/phase26eo_lite_exact_pair_route_helper.py"),
    Path("outputs_basic32/phase26eo_lite_transplant_selftest.py"),
    Path("outputs_basic32/phase26eq_lite_runtime_exact_pair_gate_selftest.py"),
    Path("outputs_basic32/phase26en_lite_live_guard_selftest.py"),
    Path("outputs_basic32/phase26em_lite_runtime_route_selftest.py"),
]

LIVE_REPLAY_COMMANDS = [
    ("eo_live_patcher", [sys.executable, str(SOURCE_ROOT / "geomlang_phase26eo_lite_live_guarded_transplant_patcher_cuda_basic32_E_drive.py")]),
    ("eo_selftest", [sys.executable, str(OUTPUTS / "phase26eo_lite_transplant_selftest.py")]),
    ("eq_shadow_verifier", [sys.executable, str(SOURCE_ROOT / "geomlang_phase26eq_lite_exact_pair_shadow_transplant_verifier_cuda_basic32_E_drive.py")]),
    ("eq_gate_selftest", [sys.executable, str(OUTPUTS / "phase26eq_lite_runtime_exact_pair_gate_selftest.py")]),
    ("en_guard_selftest", [sys.executable, str(OUTPUTS / "phase26en_lite_live_guard_selftest.py")]),
    ("em_router_selftest", [sys.executable, str(OUTPUTS / "phase26em_lite_runtime_route_selftest.py")]),
]

PASS_TOKENS = {
    "eo_live_patcher": ["LIVE_GUARD_READY=True", "CONTRACT_EXACT_PAIR_PASS=True"],
    "eo_selftest": ["EO_SELFTEST_PASS", "true"],
    "eq_shadow_verifier": ["CONTRACT_SHADOW_READY=True", "exact_pair_pass=True", "metric_pass=True"],
    "eq_gate_selftest": ["EQ_GATE_SELFTEST_PASS", "true"],
    "en_guard_selftest": ["EN_GUARD_SELFTEST_PASS", "true"],
    "em_router_selftest": ["EM_ROUTER_SELFTEST_PASS", "true"],
}


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(read_text(path))
    except Exception:
        return default


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False), encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha16_file(path: Path) -> str:
    return sha256_file(path)[:16]


def syntax_check(path: Path) -> Tuple[bool, str]:
    try:
        text = read_text(path)
        ast.parse(text, filename=str(path))
        return True, ""
    except SyntaxError as e:
        return False, f"{e.__class__.__name__}: {e.msg} at line {e.lineno}:{e.offset}"
    except Exception as e:
        return False, f"{e.__class__.__name__}: {e}"


def contains(path: Path, token: str) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        return token in read_text(path)
    except Exception:
        return False


def normalize_path_string(value: str) -> Path:
    # Windows paths are expected on the user's machine. Path() handles them there.
    return Path(value)


@dataclass
class Check:
    name: str
    pass_: bool
    detail: Any = ""

    def row(self) -> Dict[str, Any]:
        return {"check": self.name, "pass": bool(self.pass_), "detail": self.detail}


@dataclass
class CommandResult:
    name: str
    command: List[str]
    returncode: int
    elapsed_sec: float
    pass_: bool
    stdout_tail: str
    stderr_tail: str

    def row(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "returncode": self.returncode,
            "elapsed_sec": round(self.elapsed_sec, 3),
            "pass": self.pass_,
            "command": " ".join(self.command),
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
        }


def tail(text: str, chars: int = 4000) -> str:
    text = text or ""
    return text[-chars:]


def run_command(name: str, command: List[str], cwd: Path, timeout: int) -> CommandResult:
    t0 = time.time()
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        elapsed = time.time() - t0
        out = proc.stdout or ""
        err = proc.stderr or ""
        token_pass = all(tok in (out + "\n" + err) for tok in PASS_TOKENS.get(name, []))
        passed = proc.returncode == 0 and token_pass
        return CommandResult(name, command, proc.returncode, elapsed, passed, tail(out), tail(err))
    except subprocess.TimeoutExpired as e:
        elapsed = time.time() - t0
        return CommandResult(name, command, 124, elapsed, False, tail(e.stdout or ""), tail(e.stderr or f"TIMEOUT after {timeout}s"))
    except Exception as e:
        elapsed = time.time() - t0
        return CommandResult(name, command, 125, elapsed, False, "", f"{e.__class__.__name__}: {e}")


def manifest_files(et_manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    files = et_manifest.get("files") or []
    if not isinstance(files, list):
        return []
    return [f for f in files if isinstance(f, dict)]


def bundle_file_path(file_record: Dict[str, Any]) -> Optional[Path]:
    p = file_record.get("bundle_path")
    if not p:
        return None
    return normalize_path_string(str(p))


def audit_bundle_files(et_manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rec in manifest_files(et_manifest):
        bp = bundle_file_path(rec)
        exists = bool(bp and bp.exists() and bp.is_file())
        actual_sha256 = sha256_file(bp) if exists and bp else ""
        expected_sha256 = str(rec.get("sha256") or "")
        hash_match = exists and actual_sha256 == expected_sha256
        syntax_ok = True
        syntax_error = ""
        if exists and bp and bp.suffix.lower() == ".py":
            syntax_ok, syntax_error = syntax_check(bp)
        rows.append({
            "role": rec.get("role", ""),
            "name": rec.get("name", ""),
            "bundle_path": str(bp) if bp else "",
            "exists": exists,
            "expected_sha256": expected_sha256,
            "actual_sha256": actual_sha256,
            "hash_match": hash_match,
            "syntax_ok": syntax_ok,
            "syntax_error": syntax_error,
            "bytes": bp.stat().st_size if exists and bp else 0,
        })
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def copy_replay_snapshot(release_dir: Path, replay_dir: Path) -> Tuple[bool, str, int]:
    if not release_dir.exists():
        return False, f"release_dir missing: {release_dir}", 0
    if replay_dir.exists():
        shutil.rmtree(replay_dir)
    shutil.copytree(release_dir, replay_dir)
    count = sum(1 for p in replay_dir.rglob("*") if p.is_file())
    return True, str(replay_dir), count


def zip_directory(src_dir: Path, zip_path: Path) -> Tuple[bool, str, int]:
    if not src_dir.exists():
        return False, f"missing: {src_dir}", 0
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    n = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in src_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(src_dir.parent))
                n += 1
    return True, str(zip_path), n


def plot_checks(path: Path, checks: List[Check]) -> None:
    try:
        import matplotlib.pyplot as plt
        labels = [c.name for c in checks]
        vals = [1 if c.pass_ else 0 for c in checks]
        height = max(6, 0.42 * len(labels) + 1.5)
        fig, ax = plt.subplots(figsize=(14, height))
        y = list(range(len(labels)))
        ax.barh(y, vals)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("pass = 1, fail = 0")
        ax.set_title("26EU-LITE release replay lock checks")
        for yi, v in zip(y, vals):
            ax.text(1.02, yi, "PASS" if v else "FAIL", va="center")
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception as e:
        write_text(path.with_suffix(".plot_error.txt"), f"plot failed: {e}\n")


def build_readme(summary: Dict[str, Any], command_results: List[CommandResult]) -> str:
    checks = summary.get("checks", {})
    lines = []
    lines.append("# 26EU-LITE Release Replay Lock")
    lines.append("")
    lines.append(f"Created UTC: `{summary.get('created_utc')}`")
    lines.append(f"Replay lock pass: `{summary.get('RELEASE_REPLAY_LOCK_PASS')}`")
    lines.append(f"Expected fingerprint: `{summary.get('expected_fingerprint')}`")
    lines.append(f"Release bundle: `{summary.get('release_dir')}`")
    lines.append(f"Replay snapshot: `{summary.get('replay_dir')}`")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    for k, v in checks.items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## Live replay commands")
    lines.append("")
    if not command_results:
        lines.append("Live replay was skipped.")
    else:
        for r in command_results:
            lines.append(f"- `{r.name}`: pass=`{r.pass_}`, rc=`{r.returncode}`, elapsed=`{r.elapsed_sec:.3f}s`")
    lines.append("")
    lines.append("## What this means")
    lines.append("")
    lines.append("EU confirms that the ET bundle is sealed against drift: every manifest hash matches, every bundled Python file compiles, the exact-pair guard artifacts are present, and the live post-promotion command chain still passes unless skipped.")
    lines.append("")
    lines.append("## Restore / replay notes")
    lines.append("")
    lines.append("Keep the release bundle directory together. The most important deployable pieces are the runtime router, exact-pair gate, live guard, and EO helper inside `outputs_basic32`, plus the promoted EO/EQ/ER/ES/ET scripts inside `bbit_geomlang`.")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=f"{PHASE}: {TITLE}")
    ap.add_argument("--root", default=str(ROOT), help="BBIT root. Default: E:\\BBIT")
    ap.add_argument("--release-dir", default=None, help="Override ET release bundle directory")
    ap.add_argument("--skip-live-replay", action="store_true", help="Skip live command chain replay")
    ap.add_argument("--timeout", type=int, default=120, help="Per-command timeout in seconds")
    ap.add_argument("--zip", action="store_true", help="Write a zip archive of the ET release bundle")
    args = ap.parse_args(argv)

    root = Path(args.root)
    source_root = root / "bbit_geomlang"
    outputs = root / "outputs_basic32"
    et_manifest_path = outputs / "phase26et_lite_release_manifest.json"
    et_summary_path = outputs / "phase26et_lite_summary.json"
    es_summary_path = outputs / "phase26es_lite_summary.json"

    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] root: {root}")
    print(f"[{PHASE}] outputs: {outputs}")

    et_manifest = read_json(et_manifest_path, {}) or {}
    et_summary = read_json(et_summary_path, {}) or {}
    es_summary = read_json(es_summary_path, {}) or {}

    release_dir = Path(args.release_dir) if args.release_dir else Path(et_manifest.get("release_dir") or DEFAULT_RELEASE_DIR)
    replay_dir = outputs / "phase26eu_lite_release_replay"
    print(f"[{PHASE}] release_dir: {release_dir}")

    file_rows = audit_bundle_files(et_manifest)
    snapshot_ok, snapshot_detail, snapshot_count = copy_replay_snapshot(release_dir, replay_dir)

    # Rebind live replay commands to the possibly-overridden root.
    live_commands = [
        ("eo_live_patcher", [sys.executable, str(source_root / "geomlang_phase26eo_lite_live_guarded_transplant_patcher_cuda_basic32_E_drive.py")]),
        ("eo_selftest", [sys.executable, str(outputs / "phase26eo_lite_transplant_selftest.py")]),
        ("eq_shadow_verifier", [sys.executable, str(source_root / "geomlang_phase26eq_lite_exact_pair_shadow_transplant_verifier_cuda_basic32_E_drive.py")]),
        ("eq_gate_selftest", [sys.executable, str(outputs / "phase26eq_lite_runtime_exact_pair_gate_selftest.py")]),
        ("en_guard_selftest", [sys.executable, str(outputs / "phase26en_lite_live_guard_selftest.py")]),
        ("em_router_selftest", [sys.executable, str(outputs / "phase26em_lite_runtime_route_selftest.py")]),
    ]

    command_results: List[CommandResult] = []
    if not args.skip_live_replay:
        for name, cmd in live_commands:
            print(f"[{PHASE}] running {name} ...")
            r = run_command(name, cmd, cwd=root, timeout=args.timeout)
            command_results.append(r)
            print(f"[{PHASE}]   {name}: rc={r.returncode} pass={r.pass_} elapsed={r.elapsed_sec:.3f}s")
    else:
        print(f"[{PHASE}] live replay skipped by --skip-live-replay")

    zip_ok = False
    zip_detail = "not requested"
    zip_count = 0
    zip_path = outputs / "phase26eu_lite_release_bundle.zip"
    if args.zip:
        zip_ok, zip_detail, zip_count = zip_directory(release_dir, zip_path)
        print(f"[{PHASE}] zip: {zip_detail} files={zip_count}")

    checks: List[Check] = []
    checks.append(Check("et_manifest_exists", et_manifest_path.exists(), str(et_manifest_path)))
    checks.append(Check("et_release_lock_pass", bool(et_manifest.get("RELEASE_LOCK_PASS") is True or et_summary.get("RELEASE_LOCK_PASS") is True), et_manifest.get("RELEASE_LOCK_PASS")))
    checks.append(Check("es_post_promotion_seal_pass", bool(es_summary.get("POST_PROMOTION_SEAL_PASS") is True or (et_manifest.get("es_summary") or {}).get("POST_PROMOTION_SEAL_PASS") is True), es_summary.get("POST_PROMOTION_SEAL_PASS")))
    checks.append(Check("release_dir_exists", release_dir.exists(), str(release_dir)))
    checks.append(Check("release_snapshot_copied", snapshot_ok, {"detail": snapshot_detail, "files": snapshot_count}))
    checks.append(Check("manifest_has_files", len(file_rows) > 0, len(file_rows)))
    checks.append(Check("all_bundle_files_exist", bool(file_rows) and all(r["exists"] for r in file_rows), sum(1 for r in file_rows if r["exists"])))
    checks.append(Check("all_bundle_hashes_match_manifest", bool(file_rows) and all(r["hash_match"] for r in file_rows), sum(1 for r in file_rows if r["hash_match"])))
    py_rows = [r for r in file_rows if str(r.get("bundle_path", "")).lower().endswith(".py")]
    checks.append(Check("all_bundled_python_syntax_ok", bool(py_rows) and all(r["syntax_ok"] for r in py_rows), sum(1 for r in py_rows if r["syntax_ok"])))
    checks.append(Check("fingerprint_found_in_manifest", EXPECTED_FINGERPRINT in json.dumps(et_manifest), EXPECTED_FINGERPRINT))
    checks.append(Check("bundle_has_core_exact_pair_artifacts", all((release_dir / rel).exists() for rel in CORE_BUNDLE_RELATIVE), [str(rel) for rel in CORE_BUNDLE_RELATIVE if not (release_dir / rel).exists()]))
    checks.append(Check("live_replay_commands_pass", args.skip_live_replay or (bool(command_results) and all(r.pass_ for r in command_results)), "skipped" if args.skip_live_replay else [r.row() for r in command_results]))
    checks.append(Check("zip_written_if_requested", (not args.zip) or zip_ok, zip_detail))

    release_replay_lock_pass = all(c.pass_ for c in checks)
    print(f"[{PHASE}] RELEASE_REPLAY_LOCK_PASS={release_replay_lock_pass}")
    print(f"[{PHASE}] checks:")
    for c in checks:
        print(f"  - {c.name}: {c.pass_}")

    out_prefix = outputs / "phase26eu_lite"
    files_csv = out_prefix.with_name("phase26eu_lite_bundle_file_audit.csv")
    command_csv = out_prefix.with_name("phase26eu_lite_command_replay.csv")
    checks_csv = out_prefix.with_name("phase26eu_lite_release_replay_checks.csv")
    summary_json = out_prefix.with_name("phase26eu_lite_summary.json")
    manifest_json = out_prefix.with_name("phase26eu_lite_release_replay_manifest.json")
    report_md = out_prefix.with_name("phase26eu_lite_release_replay_report.md")
    plot_png = out_prefix.with_name("phase26eu_lite_release_replay_lock_checks.png")

    write_csv(files_csv, file_rows)
    write_csv(command_csv, [r.row() for r in command_results])
    write_csv(checks_csv, [c.row() for c in checks])

    summary = {
        "phase": PHASE,
        "title": TITLE,
        "RELEASE_REPLAY_LOCK_PASS": release_replay_lock_pass,
        "created_utc": now_utc(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "root": str(root),
        "source_root": str(source_root),
        "outputs": str(outputs),
        "release_dir": str(release_dir),
        "replay_dir": str(replay_dir),
        "expected_fingerprint": EXPECTED_FINGERPRINT,
        "checks": {c.name: c.pass_ for c in checks},
        "check_details": [c.row() for c in checks],
        "file_audit": {
            "manifest_files": len(file_rows),
            "files_exist": sum(1 for r in file_rows if r["exists"]),
            "hash_matches": sum(1 for r in file_rows if r["hash_match"]),
            "python_files": len(py_rows),
            "python_syntax_ok": sum(1 for r in py_rows if r["syntax_ok"]),
        },
        "command_replay": [r.row() for r in command_results],
        "zip": {
            "requested": bool(args.zip),
            "ok": zip_ok,
            "path": str(zip_path) if args.zip else "",
            "files": zip_count,
            "detail": zip_detail,
        },
        "inputs": {
            "et_manifest": str(et_manifest_path),
            "et_summary": str(et_summary_path),
            "es_summary": str(es_summary_path),
        },
        "outputs_written": {
            "bundle_file_audit_csv": str(files_csv),
            "command_replay_csv": str(command_csv),
            "checks_csv": str(checks_csv),
            "summary_json": str(summary_json),
            "manifest_json": str(manifest_json),
            "report_md": str(report_md),
            "plot_png": str(plot_png),
        },
    }
    write_json(summary_json, summary)
    write_json(manifest_json, summary)
    write_text(report_md, build_readme(summary, command_results))
    plot_checks(plot_png, checks)

    print(f"[{PHASE}] wrote file audit: {files_csv}")
    print(f"[{PHASE}] wrote command replay: {command_csv}")
    print(f"[{PHASE}] wrote manifest: {manifest_json}")
    print(f"[{PHASE}] wrote report: {report_md}")
    print(f"[{PHASE}] wrote plot: {plot_png}")
    print(f"[{PHASE}] wrote outputs to: {outputs}")
    return 0 if release_replay_lock_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
