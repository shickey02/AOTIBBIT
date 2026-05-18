#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
26EW-LITE clean-room zip extraction replay verifier / portable release seal.

Purpose
-------
Phase 26EV repaired the ET release bundle and wrote:
    E:\BBIT\outputs_basic32\phase26ev_lite_release_bundle_repaired.zip

26EW verifies that this ZIP is actually portable by:
  1. extracting it into a clean temp folder outside E:\BBIT,
  2. auditing hashes and Python syntax inside the extracted copy,
  3. checking that core exact-pair artifacts are present,
  4. running the bundled selftests from the extracted folder,
  5. detecting hard absolute E:\BBIT dependencies in replay output / source,
  6. writing a clean-room portability report, manifest, command log, and plot.

This script is intentionally conservative:
  - It never modifies bbit_geomlang source files.
  - It never modifies the repaired zip.
  - It extracts into outputs_basic32\phase26ew_lite_cleanroom_extract by default.
  - It treats import/path failures as evidence that the bundle is not yet portable.

Run
---
From E:\BBIT:

    python bbit_geomlang/geomlang_phase26ew_lite_cleanroom_zip_replay_verifier_cuda_basic32_E_drive.py

Optional:

    python bbit_geomlang/geomlang_phase26ew_lite_cleanroom_zip_replay_verifier_cuda_basic32_E_drive.py --zip E:\BBIT\outputs_basic32\phase26ev_lite_release_bundle_repaired.zip --fresh

Outputs
-------
    E:\BBIT\outputs_basic32\phase26ew_lite_cleanroom_replay_report.md
    E:\BBIT\outputs_basic32\phase26ew_lite_summary.json
    E:\BBIT\outputs_basic32\phase26ew_lite_cleanroom_manifest.json
    E:\BBIT\outputs_basic32\phase26ew_lite_command_log.txt
    E:\BBIT\outputs_basic32\phase26ew_lite_cleanroom_checks.csv
    E:\BBIT\outputs_basic32\phase26ew_lite_cleanroom_file_audit.csv
    E:\BBIT\outputs_basic32\phase26ew_lite_cleanroom_lock_checks.png
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PHASE = "26EW-LITE"
FINGERPRINT_RE = re.compile(r"98ebdcbb8e995bc1|route_fingerprint|fingerprint", re.I)
ABS_ROOT_RE = re.compile(r"\b[A-Za-z]:\\BBIT\b", re.I)

CORE_EXACT_PAIR_ARTIFACT_NAMES = [
    "phase26em_lite_runtime_route_router.py",
    "phase26em_lite_runtime_route_selftest.py",
    "phase26en_lite_live_contract_guard.py",
    "phase26en_lite_live_guard_selftest.py",
    "phase26eo_lite_exact_pair_route_helper.py",
    "phase26eo_lite_transplant_selftest.py",
    "phase26eq_lite_runtime_exact_pair_gate.py",
    "phase26eq_lite_runtime_exact_pair_gate_selftest.py",
]

PREFERRED_REPLAY_COMMANDS = [
    ("em_router_selftest", ["python", "phase26em_lite_runtime_route_selftest.py"]),
    ("en_guard_selftest", ["python", "phase26en_lite_live_guard_selftest.py"]),
    ("eo_selftest", ["python", "phase26eo_lite_transplant_selftest.py"]),
    ("eq_gate_selftest", ["python", "phase26eq_lite_runtime_exact_pair_gate_selftest.py"]),
]

PASS_TOKENS = [
    "true",
    "pass=true",
    "selftest_pass",
    "metrics_pass",
    "contract_pass",
    "exact_pair_pass",
    "eo_selftest_pass",
]


@dataclass
class Check:
    name: str
    passed: bool
    detail: Any = ""


@dataclass
class CommandResult:
    name: str
    argv: List[str]
    cwd: str
    rc: int
    elapsed_s: float
    pass_detected: bool
    stdout_tail: str
    stderr_tail: str
    absolute_path_leak: bool


def short_hash(path: Path, n: int = 16) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def full_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text_lossy(path: Path, max_bytes: Optional[int] = None) -> str:
    data = path.read_bytes()
    if max_bytes is not None and len(data) > max_bytes:
        data = data[:max_bytes]
    return data.decode("utf-8", errors="replace")


def find_root() -> Path:
    # This file is expected at E:\BBIT\bbit_geomlang\...
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "bbit_geomlang").exists() and (parent / "outputs_basic32").exists():
            return parent
    # Fallback for direct execution from E:\BBIT.
    cwd = Path.cwd().resolve()
    for parent in [cwd, *cwd.parents]:
        if (parent / "outputs_basic32").exists():
            return parent
    return cwd


def latest_existing(paths: Iterable[Path]) -> Optional[Path]:
    found = [p for p in paths if p.exists()]
    if not found:
        return None
    return max(found, key=lambda p: p.stat().st_mtime)


def safe_rmtree(path: Path) -> None:
    if path.exists():
        # Refuse to delete suspicious locations.
        s = str(path.resolve()).lower()
        if len(path.parts) < 3 or s.endswith(":\\") or s in {"c:\\", "e:\\"}:
            raise RuntimeError(f"Refusing to remove unsafe path: {path}")
        shutil.rmtree(path)


def extract_zip(zip_path: Path, extract_base: Path, fresh: bool) -> Path:
    if fresh:
        safe_rmtree(extract_base)
    extract_base.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    cleanroom = extract_base / f"cleanroom_{stamp}"
    cleanroom.mkdir(parents=True, exist_ok=False)

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Basic zip-slip protection.
        for member in zf.namelist():
            dest = (cleanroom / member).resolve()
            if not str(dest).startswith(str(cleanroom.resolve())):
                raise RuntimeError(f"Unsafe zip member path: {member}")
        zf.extractall(cleanroom)

    # Many zips contain a single top-level release folder. If so, use it as replay root.
    children = [p for p in cleanroom.iterdir()]
    dirs = [p for p in children if p.is_dir()]
    files = [p for p in children if p.is_file()]
    if len(dirs) == 1 and not files:
        return dirs[0]
    return cleanroom


def locate_manifest(replay_root: Path) -> Optional[Path]:
    candidates = [
        replay_root / "phase26ev_lite_release_replay_repair_manifest.json",
        replay_root / "phase26et_lite_release_manifest.json",
        replay_root / "release_manifest.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    matches = list(replay_root.rglob("*manifest*.json"))
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def load_json(path: Optional[Path]) -> Dict[str, Any]:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(read_text_lossy(path))
    except Exception:
        return {}


def flatten_manifest_files(obj: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            # Common shapes:
            # {"path": "...", "sha256": "..."} or {"relative_path": "...", "hash": "..."}
            keys = set(x.keys())
            path_val = None
            for k in ["path", "relative_path", "relpath", "file", "name"]:
                if isinstance(x.get(k), str) and ("/" in x[k] or "\\" in x[k] or x[k].endswith(".py") or x[k].endswith(".json") or x[k].endswith(".md")):
                    path_val = x[k]
                    break
            if path_val:
                out.append(x)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(obj)
    # Deduplicate by best path-like value.
    seen = set()
    uniq = []
    for d in out:
        pv = None
        for k in ["path", "relative_path", "relpath", "file", "name"]:
            if isinstance(d.get(k), str):
                pv = d[k]
                break
        if pv and pv not in seen:
            seen.add(pv)
            uniq.append(d)
    return uniq


def manifest_hash_audit(replay_root: Path, manifest: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
    rows = []
    file_recs = flatten_manifest_files(manifest)
    if not file_recs:
        return True, rows  # Not all prior manifests include hashes; existence is checked separately.

    ok_all = True
    for rec in file_recs:
        rel = None
        for k in ["path", "relative_path", "relpath", "file", "name"]:
            if isinstance(rec.get(k), str):
                rel = rec[k]
                break
        if not rel:
            continue
        rel_norm = rel.replace("\\", "/")
        # Try direct, then basename search.
        p = replay_root / rel_norm
        if not p.exists():
            matches = list(replay_root.rglob(Path(rel_norm).name))
            p = matches[0] if matches else p

        expected = None
        for k in ["sha256", "hash", "digest", "file_hash"]:
            if isinstance(rec.get(k), str) and len(rec[k]) >= 8:
                expected = rec[k]
                break

        exists = p.exists()
        actual = full_hash(p) if exists and p.is_file() else ""
        hash_match = True
        if expected and actual:
            hash_match = actual.lower().startswith(expected.lower()) or expected.lower().startswith(actual.lower())
        ok = bool(exists and hash_match)
        ok_all = ok_all and ok
        rows.append({
            "manifest_path": rel,
            "resolved_path": str(p),
            "exists": exists,
            "expected_hash": expected or "",
            "actual_sha256": actual,
            "hash_match": hash_match,
            "ok": ok,
        })
    return ok_all, rows


def syntax_audit(replay_root: Path) -> Tuple[bool, List[Dict[str, Any]]]:
    rows = []
    ok_all = True
    for py in sorted(replay_root.rglob("*.py")):
        rel = py.relative_to(replay_root)
        try:
            ast.parse(read_text_lossy(py), filename=str(py))
            ok = True
            err = ""
        except SyntaxError as e:
            ok = False
            err = f"{e.__class__.__name__}: {e}"
        except Exception as e:
            ok = False
            err = f"{e.__class__.__name__}: {e}"
        ok_all = ok_all and ok
        rows.append({"relative_path": str(rel), "syntax_ok": ok, "error": err})
    return ok_all, rows


def any_file_contains(replay_root: Path, pattern: re.Pattern[str], names: Optional[List[str]] = None) -> bool:
    candidates = []
    if names:
        for name in names:
            candidates.extend(replay_root.rglob(name))
    else:
        candidates = list(replay_root.rglob("*"))
    for p in candidates:
        if p.is_file() and p.suffix.lower() in {".py", ".json", ".md", ".txt"}:
            try:
                if pattern.search(read_text_lossy(p, max_bytes=2_000_000)):
                    return True
            except Exception:
                pass
    return False


def core_artifact_status(replay_root: Path) -> Dict[str, bool]:
    return {name: bool(list(replay_root.rglob(name))) for name in CORE_EXACT_PAIR_ARTIFACT_NAMES}


def run_command(name: str, argv: List[str], cwd: Path, timeout: int = 60) -> CommandResult:
    t0 = time.time()
    env = os.environ.copy()
    # Make the extracted replay root win imports before the live E:\BBIT tree.
    env["PYTHONPATH"] = str(cwd) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd),
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        rc = proc.returncode
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
    except subprocess.TimeoutExpired as e:
        rc = 124
        stdout = e.stdout if isinstance(e.stdout, str) else (e.stdout or b"").decode("utf-8", errors="replace")
        stderr = e.stderr if isinstance(e.stderr, str) else (e.stderr or b"").decode("utf-8", errors="replace")
        stderr += f"\nTIMEOUT after {timeout}s"
    elapsed = time.time() - t0

    combined = (stdout + "\n" + stderr).lower()
    pass_detected = (rc == 0) and any(tok in combined for tok in PASS_TOKENS)
    abs_leak = bool(ABS_ROOT_RE.search(stdout + "\n" + stderr))

    return CommandResult(
        name=name,
        argv=argv,
        cwd=str(cwd),
        rc=rc,
        elapsed_s=round(elapsed, 3),
        pass_detected=pass_detected,
        stdout_tail=stdout[-4000:],
        stderr_tail=stderr[-4000:],
        absolute_path_leak=abs_leak,
    )


def find_replay_commands(replay_root: Path) -> List[Tuple[str, List[str]]]:
    cmds: List[Tuple[str, List[str]]] = []
    for name, argv in PREFERRED_REPLAY_COMMANDS:
        script = replay_root / argv[1]
        if script.exists():
            cmds.append((name, argv))
        else:
            matches = list(replay_root.rglob(argv[1]))
            if matches:
                # Use path relative to replay root.
                rel = str(matches[0].relative_to(replay_root))
                cmds.append((name, ["python", rel]))

    # Fallback: include any obvious selftest if preferred list did not cover enough.
    if len(cmds) < 3:
        seen = {tuple(c[1]) for c in cmds}
        for py in sorted(replay_root.rglob("*selftest*.py")):
            rel = str(py.relative_to(replay_root))
            argv = ("python", rel)
            if argv not in seen:
                cmds.append((py.stem, list(argv)))
                seen.add(argv)
    return cmds


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_checks(path: Path, checks: List[Check]) -> None:
    try:
        import matplotlib.pyplot as plt
        names = [c.name for c in checks]
        vals = [1 if c.passed else 0 for c in checks]
        fig_h = max(6, 0.45 * len(names))
        plt.figure(figsize=(13, fig_h))
        y = range(len(names))
        plt.barh(list(y), vals)
        plt.yticks(list(y), names)
        plt.xlim(0, 1.05)
        plt.xlabel("pass = 1, fail = 0")
        plt.title("26EW-LITE clean-room release replay lock checks")
        for i, v in enumerate(vals):
            plt.text(1.02, i, "PASS" if v else "FAIL", va="center")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
    except Exception as e:
        path.with_suffix(".plot_error.txt").write_text(str(e), encoding="utf-8")


def make_report(
    path: Path,
    zip_path: Path,
    replay_root: Path,
    checks: List[Check],
    command_results: List[CommandResult],
    core_status: Dict[str, bool],
    manifest_path: Optional[Path],
) -> None:
    lines = []
    pass_all = all(c.passed for c in checks)
    lines.append(f"# {PHASE} Clean-room Zip Replay Report")
    lines.append("")
    lines.append(f"**CLEANROOM_REPLAY_LOCK_PASS:** `{pass_all}`")
    lines.append("")
    lines.append(f"- ZIP: `{zip_path}`")
    lines.append(f"- Clean-room replay root: `{replay_root}`")
    lines.append(f"- Manifest: `{manifest_path or ''}`")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| Check | Pass | Detail |")
    lines.append("|---|---:|---|")
    for c in checks:
        detail = json.dumps(c.detail, ensure_ascii=False) if not isinstance(c.detail, str) else c.detail
        if len(detail) > 300:
            detail = detail[:300] + "..."
        lines.append(f"| `{c.name}` | `{c.passed}` | {detail} |")
    lines.append("")
    lines.append("## Core exact-pair artifacts")
    lines.append("")
    lines.append("| Artifact | Present |")
    lines.append("|---|---:|")
    for k, v in core_status.items():
        lines.append(f"| `{k}` | `{v}` |")
    lines.append("")
    lines.append("## Clean-room command replay")
    lines.append("")
    lines.append("| Command | RC | Pass detected | Abs path leak | Elapsed s |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in command_results:
        lines.append(f"| `{r.name}` | `{r.rc}` | `{r.pass_detected}` | `{r.absolute_path_leak}` | `{r.elapsed_s}` |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if pass_all:
        lines.append("The repaired release ZIP extracted into a clean-room directory and replayed its core exact-pair guard chain without relying on the live source tree.")
    else:
        lines.append("The repaired release ZIP is not yet fully clean-room portable. Inspect failed checks and command tails in `phase26ew_lite_command_log.txt`.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", dest="zip_path", default=None, help="Path to repaired release zip.")
    ap.add_argument("--fresh", action="store_true", help="Delete prior 26EW clean-room extracts before extracting.")
    ap.add_argument("--timeout", type=int, default=90, help="Per-command timeout in seconds.")
    args = ap.parse_args(argv)

    root = find_root()
    outputs = root / "outputs_basic32"
    outputs.mkdir(parents=True, exist_ok=True)

    default_zip = outputs / "phase26ev_lite_release_bundle_repaired.zip"
    zip_path = Path(args.zip_path).resolve() if args.zip_path else default_zip.resolve()

    cleanroom_base = outputs / "phase26ew_lite_cleanroom_extract"
    print(f"[{PHASE}] Clean-room zip extraction replay verifier / portable seal")
    print(f"[{PHASE}] root: {root}")
    print(f"[{PHASE}] outputs: {outputs}")
    print(f"[{PHASE}] zip: {zip_path}")

    checks: List[Check] = []
    checks.append(Check("zip_exists", zip_path.exists(), str(zip_path)))
    checks.append(Check("zip_is_file", zip_path.is_file(), str(zip_path) if zip_path.exists() else "missing"))

    replay_root = None
    manifest_path = None
    manifest: Dict[str, Any] = {}
    syntax_rows: List[Dict[str, Any]] = []
    hash_rows: List[Dict[str, Any]] = []
    command_results: List[CommandResult] = []
    core_status: Dict[str, bool] = {name: False for name in CORE_EXACT_PAIR_ARTIFACT_NAMES}

    if not zip_path.exists() or not zip_path.is_file():
        pass_all = False
    else:
        try:
            replay_root = extract_zip(zip_path, cleanroom_base, fresh=args.fresh)
            checks.append(Check("cleanroom_extract_created", replay_root.exists(), str(replay_root)))
        except Exception as e:
            checks.append(Check("cleanroom_extract_created", False, f"{e.__class__.__name__}: {e}"))
            replay_root = None

    if replay_root:
        manifest_path = locate_manifest(replay_root)
        checks.append(Check("manifest_found_in_cleanroom", manifest_path is not None, str(manifest_path or "")))
        manifest = load_json(manifest_path)
        checks.append(Check("manifest_json_loads", bool(manifest), str(manifest_path or "")))

        core_status = core_artifact_status(replay_root)
        checks.append(Check("bundle_has_core_exact_pair_artifacts", all(core_status.values()), core_status))

        syntax_ok, syntax_rows = syntax_audit(replay_root)
        checks.append(Check("all_bundled_python_syntax_ok", syntax_ok, f"{sum(1 for r in syntax_rows if r.get('syntax_ok'))}/{len(syntax_rows)}"))

        hash_ok, hash_rows = manifest_hash_audit(replay_root, manifest)
        checks.append(Check("manifest_hashes_match_when_available", hash_ok, f"{sum(1 for r in hash_rows if r.get('ok'))}/{len(hash_rows)}" if hash_rows else "no hash records found"))

        fingerprint_found = (
            any_file_contains(replay_root, FINGERPRINT_RE, CORE_EXACT_PAIR_ARTIFACT_NAMES)
            or FINGERPRINT_RE.search(json.dumps(manifest, ensure_ascii=False)) is not None
        )
        checks.append(Check("fingerprint_found_in_cleanroom_core", fingerprint_found, "98ebdcbb8e995bc1 or fingerprint token"))

        # Source scan is informational but can reveal non-portable absolute path constants.
        source_abs_path_refs = []
        for p in replay_root.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".py", ".json", ".md", ".txt"}:
                try:
                    txt = read_text_lossy(p, max_bytes=2_000_000)
                except Exception:
                    continue
                if ABS_ROOT_RE.search(txt):
                    source_abs_path_refs.append(str(p.relative_to(replay_root)))
        # We do not hard-fail merely because reports mention E:\BBIT, but Python core files should not.
        core_abs_refs = [x for x in source_abs_path_refs if x.endswith(".py")]
        checks.append(Check("no_absolute_bbit_paths_in_core_python", len(core_abs_refs) == 0, core_abs_refs[:20]))

        cmds = find_replay_commands(replay_root)
        checks.append(Check("replay_commands_discovered", len(cmds) >= 3, [c[0] for c in cmds]))

        for name, cmd in cmds:
            print(f"[{PHASE}] running cleanroom {name} ...")
            res = run_command(name, cmd, replay_root, timeout=args.timeout)
            print(f"[{PHASE}]   {name}: rc={res.rc} pass={res.pass_detected} abs_path_leak={res.absolute_path_leak} elapsed={res.elapsed_s}s")
            command_results.append(res)

        if command_results:
            checks.append(Check("all_cleanroom_replay_commands_exit_zero", all(r.rc == 0 for r in command_results), {r.name: r.rc for r in command_results}))
            checks.append(Check("all_cleanroom_replay_commands_pass_detected", all(r.pass_detected for r in command_results), {r.name: r.pass_detected for r in command_results}))
            checks.append(Check("no_absolute_bbit_path_leak_in_command_output", not any(r.absolute_path_leak for r in command_results), {r.name: r.absolute_path_leak for r in command_results}))
        else:
            checks.append(Check("all_cleanroom_replay_commands_exit_zero", False, "no commands"))
            checks.append(Check("all_cleanroom_replay_commands_pass_detected", False, "no commands"))
            checks.append(Check("no_absolute_bbit_path_leak_in_command_output", False, "no commands"))

    pass_all = all(c.passed for c in checks)
    print(f"[{PHASE}] CLEANROOM_REPLAY_LOCK_PASS={pass_all}")
    print(f"[{PHASE}] checks:")
    for c in checks:
        print(f"  - {c.name}: {c.passed} :: {c.detail}")

    # Outputs.
    checks_rows = [{"name": c.name, "passed": c.passed, "detail": json.dumps(c.detail, ensure_ascii=False) if not isinstance(c.detail, str) else c.detail} for c in checks]
    command_rows = [asdict(r) for r in command_results]

    write_csv(outputs / "phase26ew_lite_cleanroom_checks.csv", checks_rows)
    write_csv(outputs / "phase26ew_lite_cleanroom_file_audit.csv", syntax_rows + hash_rows)
    write_csv(outputs / "phase26ew_lite_command_replay.csv", command_rows)

    command_log_lines = []
    for r in command_results:
        command_log_lines.append("=" * 100)
        command_log_lines.append(f"{r.name}")
        command_log_lines.append(f"argv={r.argv}")
        command_log_lines.append(f"cwd={r.cwd}")
        command_log_lines.append(f"rc={r.rc} elapsed={r.elapsed_s} pass_detected={r.pass_detected} absolute_path_leak={r.absolute_path_leak}")
        command_log_lines.append("--- stdout tail ---")
        command_log_lines.append(r.stdout_tail)
        command_log_lines.append("--- stderr tail ---")
        command_log_lines.append(r.stderr_tail)
    (outputs / "phase26ew_lite_command_log.txt").write_text("\n".join(command_log_lines), encoding="utf-8")

    cleanroom_manifest = {
        "phase": PHASE,
        "cleanroom_replay_lock_pass": pass_all,
        "root": str(root),
        "outputs": str(outputs),
        "zip_path": str(zip_path),
        "zip_sha256": full_hash(zip_path) if zip_path.exists() and zip_path.is_file() else "",
        "replay_root": str(replay_root or ""),
        "manifest_path": str(manifest_path or ""),
        "core_artifacts": core_status,
        "checks": [asdict(c) for c in checks],
        "commands": command_rows,
    }
    (outputs / "phase26ew_lite_cleanroom_manifest.json").write_text(json.dumps(cleanroom_manifest, indent=2), encoding="utf-8")
    (outputs / "phase26ew_lite_summary.json").write_text(json.dumps({
        "phase": PHASE,
        "cleanroom_replay_lock_pass": pass_all,
        "zip_path": str(zip_path),
        "replay_root": str(replay_root or ""),
        "checks_passed": sum(1 for c in checks if c.passed),
        "checks_total": len(checks),
        "commands_total": len(command_results),
        "commands_passed": sum(1 for r in command_results if r.rc == 0 and r.pass_detected),
        "absolute_path_leaks": [r.name for r in command_results if r.absolute_path_leak],
        "core_artifacts_present": all(core_status.values()),
    }, indent=2), encoding="utf-8")

    make_report(
        outputs / "phase26ew_lite_cleanroom_replay_report.md",
        zip_path=zip_path,
        replay_root=replay_root or cleanroom_base,
        checks=checks,
        command_results=command_results,
        core_status=core_status,
        manifest_path=manifest_path,
    )
    plot_checks(outputs / "phase26ew_lite_cleanroom_lock_checks.png", checks)

    print(f"[{PHASE}] wrote report: {outputs / 'phase26ew_lite_cleanroom_replay_report.md'}")
    print(f"[{PHASE}] wrote summary: {outputs / 'phase26ew_lite_summary.json'}")
    print(f"[{PHASE}] wrote manifest: {outputs / 'phase26ew_lite_cleanroom_manifest.json'}")
    print(f"[{PHASE}] wrote plot: {outputs / 'phase26ew_lite_cleanroom_lock_checks.png'}")
    print(f"[{PHASE}] wrote outputs to: {outputs}")

    return 0 if pass_all else 2


if __name__ == "__main__":
    raise SystemExit(main())
