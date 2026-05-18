#!/usr/bin/env python3
"""
26FC-LITE Self-manifest seal finalizer / standalone clean-room lock closer

Purpose
-------
26FB proved the portable bundle is functionally clean-room replayable, but the
lock stayed false because the final manifest listed itself and therefore could
not have a stable hash after the manifest was rewritten.

26FC treats that as a self-referential manifest edge, not a route/guard failure:
  * extracts the 26FB zip into a fresh clean-room
  * verifies core exact-pair artifacts and standalone selftests
  * rewrites a final manifest with explicit self_manifest_policy
  * verifies all manifest hashes except the manifest file itself
  * replays the four standalone tests in the extracted clean-room
  * writes a final sealed zip

Expected terminal goal:
  FINAL_SELF_MANIFEST_SEAL_LOCK_PASS=True
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

PHASE = "26FC-LITE"
FINGERPRINT = "98ebdcbb8e995bc1"
DEFAULT_ROOT = Path(r"E:\BBIT")
DEFAULT_OUTPUTS = DEFAULT_ROOT / "outputs_basic32"
DEFAULT_INPUT_ZIP = DEFAULT_OUTPUTS / "phase26fb_lite_final_portable_release_bundle.zip"
FINAL_ZIP_NAME = "phase26fc_lite_final_self_manifest_sealed_release_bundle.zip"
FINAL_MANIFEST_NAME = "phase26fc_lite_final_self_manifest_seal_manifest.json"

CORE_PY = [
    "phase26em_lite_runtime_route_router.py",
    "phase26em_lite_runtime_route_selftest.py",
    "phase26en_lite_live_contract_guard.py",
    "phase26en_lite_live_guard_selftest.py",
    "phase26eo_lite_exact_pair_route_helper.py",
    "phase26eo_lite_transplant_selftest.py",
    "phase26eq_lite_runtime_exact_pair_gate.py",
    "phase26eq_lite_runtime_exact_pair_gate_selftest.py",
]
CORE_DATA = [
    "phase26ej_lite_integration_contract.json",
    "phase26em_lite_contract_row_smoke.csv",
    "phase26em_lite_transplant_manifest.json",
    "phase26en_lite_live_transplant_manifest.json",
    "phase26eq_lite_shadow_transplant_manifest.json",
]
STANDALONE_TESTS = [
    "phase26fb_standalone_contract_selftest.py",
    "phase26fb_standalone_router_selftest.py",
    "phase26fb_standalone_exact_pair_gate_selftest.py",
    "phase26fb_standalone_manifest_selftest.py",
]
REPLAY_COMMANDS = [
    ("fb_contract_selftest", ["python", "outputs_basic32/phase26fb_standalone_contract_selftest.py"], "26FB_STANDALONE_CONTRACT_SELFTEST_PASS"),
    ("fb_router_selftest", ["python", "outputs_basic32/phase26fb_standalone_router_selftest.py"], "26FB_STANDALONE_ROUTER_SELFTEST_PASS"),
    ("fb_exact_pair_gate_selftest", ["python", "outputs_basic32/phase26fb_standalone_exact_pair_gate_selftest.py"], "26FB_STANDALONE_EXACT_PAIR_GATE_SELFTEST_PASS"),
    ("fb_manifest_selftest", ["python", "outputs_basic32/phase26fb_standalone_manifest_selftest.py"], "26FB_STANDALONE_MANIFEST_SELFTEST_PASS"),
]
ABS_PATH_PAT = re.compile(r"[A-Za-z]:\\\\BBIT|[A-Za-z]:\\BBIT")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def short_hash(path: Path) -> str:
    return sha256_file(path)[:16]


def safe_rel(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def syntax_ok(path: Path) -> Tuple[bool, str]:
    try:
        ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
        return True, ""
    except SyntaxError as e:
        return False, f"{e.filename}:{e.lineno}:{e.msg}"


def has_abs_bbit(path: Path) -> bool:
    txt = path.read_text(encoding="utf-8", errors="replace")
    return bool(ABS_PATH_PAT.search(txt))


def contains_fingerprint(path: Path) -> bool:
    return FINGERPRINT in path.read_text(encoding="utf-8", errors="replace")


def find_bundle_root(extract_dir: Path) -> Path:
    # Prefer a folder containing outputs_basic32; otherwise use extract root.
    candidates = [extract_dir]
    candidates.extend([p for p in extract_dir.rglob("*") if p.is_dir() and p.name.lower() != "__pycache__"])
    for c in candidates:
        if (c / "outputs_basic32").is_dir():
            return c
    return extract_dir


def ensure_outputs(bundle_root: Path) -> Path:
    out = bundle_root / "outputs_basic32"
    out.mkdir(parents=True, exist_ok=True)
    return out


def locate_file(bundle_root: Path, name: str) -> Path | None:
    direct = bundle_root / "outputs_basic32" / name
    if direct.exists():
        return direct
    direct2 = bundle_root / name
    if direct2.exists():
        return direct2
    matches = [p for p in bundle_root.rglob(name) if p.is_file()]
    return matches[0] if matches else None


def normalize_into_outputs(bundle_root: Path) -> List[str]:
    out = ensure_outputs(bundle_root)
    moved: List[str] = []
    for name in CORE_PY + CORE_DATA + STANDALONE_TESTS:
        p = locate_file(bundle_root, name)
        dst = out / name
        if p and p.resolve() != dst.resolve():
            shutil.copy2(p, dst)
            moved.append(name)
    return moved


def build_manifest(bundle_root: Path, final_manifest_rel: str) -> Dict[str, Any]:
    files: List[Dict[str, Any]] = []
    for p in sorted([x for x in bundle_root.rglob("*") if x.is_file()]):
        rel = safe_rel(p, bundle_root)
        if "__pycache__/" in rel or rel.endswith(".pyc"):
            continue
        if rel == final_manifest_rel:
            continue
        files.append({
            "path": rel,
            "size": p.stat().st_size,
            "sha256": sha256_file(p),
        })
    return {
        "phase": PHASE,
        "created_utc": utc_iso(),
        "fingerprint": FINGERPRINT,
        "self_manifest_policy": {
            "path": final_manifest_rel,
            "hash_mode": "excluded_from_own_file_hash_list",
            "reason": "A manifest cannot contain its own final stable hash after it is written. All other files are hash-locked.",
        },
        "files": files,
    }


def verify_manifest(bundle_root: Path, manifest: Dict[str, Any]) -> Tuple[bool, int, int, List[str]]:
    checked = 0
    matched = 0
    bad: List[str] = []
    for item in manifest.get("files", []):
        rel = item.get("path")
        exp = item.get("sha256")
        p = bundle_root / rel
        checked += 1
        if not p.exists():
            bad.append(f"missing {rel}")
            continue
        got = sha256_file(p)
        if got != exp:
            bad.append(f"hash mismatch {rel}")
            continue
        matched += 1
    return checked == matched and not bad, checked, matched, bad


@dataclass
class ReplayResult:
    name: str
    argv: List[str]
    rc: int
    elapsed_s: float
    pass_detected: bool
    abs_path_leak: bool
    stdout_tail: str
    stderr_tail: str


def run_replay(bundle_root: Path) -> List[ReplayResult]:
    results: List[ReplayResult] = []
    env = os.environ.copy()
    env["PYTHONPATH"] = str(bundle_root / "outputs_basic32")
    for name, argv, pass_token in REPLAY_COMMANDS:
        t0 = time.time()
        cp = subprocess.run(argv, cwd=str(bundle_root), text=True, capture_output=True, env=env)
        elapsed = round(time.time() - t0, 3)
        output = (cp.stdout or "") + "\n" + (cp.stderr or "")
        results.append(ReplayResult(
            name=name,
            argv=argv,
            rc=cp.returncode,
            elapsed_s=elapsed,
            pass_detected=(pass_token in output),
            abs_path_leak=bool(ABS_PATH_PAT.search(output)),
            stdout_tail=(cp.stdout or "")[-1000:],
            stderr_tail=(cp.stderr or "")[-1000:],
        ))
    return results


def zip_dir(src: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted([x for x in src.rglob("*") if x.is_file()]):
            if "__pycache__" in p.parts or p.suffix == ".pyc":
                continue
            zf.write(p, safe_rel(p, src))


def make_plot(path: Path, checks: Dict[str, bool]) -> None:
    try:
        import matplotlib.pyplot as plt
        names = list(checks.keys())[::-1]
        vals = [1 if checks[n] else 0 for n in names]
        fig_h = max(5.0, 0.46 * len(names))
        fig, ax = plt.subplots(figsize=(14, fig_h))
        ax.barh(names, vals)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("pass=1 fail=0")
        ax.set_title("26FC-LITE self-manifest sealed clean-room replay")
        for y, v in enumerate(vals):
            ax.text(1.02, y, "PASS" if v else "FAIL", va="center")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--outputs", default=None)
    ap.add_argument("--zip", dest="zip_path", default=str(DEFAULT_INPUT_ZIP))
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--write-final-zip", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    outputs = Path(args.outputs) if args.outputs else root / "outputs_basic32"
    in_zip = Path(args.zip_path)
    final_zip = outputs / FINAL_ZIP_NAME
    clean_parent = outputs / "phase26fc_lite_cleanroom_extract"
    cleanroom = clean_parent / f"cleanroom_{utc_stamp()}"

    print(f"[{PHASE}] Self-manifest seal finalizer / standalone clean-room lock closer")
    print(f"[{PHASE}] root: {root}")
    print(f"[{PHASE}] outputs: {outputs}")
    print(f"[{PHASE}] input zip: {in_zip}")

    outputs.mkdir(parents=True, exist_ok=True)
    if args.fresh and clean_parent.exists():
        shutil.rmtree(clean_parent)
    cleanroom.mkdir(parents=True, exist_ok=True)

    checks: Dict[str, bool] = {}
    details: Dict[str, Any] = {"cleanroom": str(cleanroom), "input_zip": str(in_zip), "final_zip": str(final_zip)}

    checks["zip_exists"] = in_zip.exists()
    checks["zip_is_file"] = in_zip.is_file()
    if not checks["zip_is_file"]:
        summary = {"phase": PHASE, "FINAL_SELF_MANIFEST_SEAL_LOCK_PASS": False, "checks": checks, "details": details}
        write_json(outputs / "phase26fc_lite_summary.json", summary)
        print(f"[{PHASE}] FINAL_SELF_MANIFEST_SEAL_LOCK_PASS=False")
        return 1

    with zipfile.ZipFile(in_zip, "r") as zf:
        zf.extractall(cleanroom)
    bundle_root = find_bundle_root(cleanroom)
    bundle_outputs = ensure_outputs(bundle_root)
    details["bundle_root"] = str(bundle_root)
    details["bundle_outputs"] = str(bundle_outputs)
    details["normalized_into_outputs"] = normalize_into_outputs(bundle_root)

    core_py_present = {n: (bundle_outputs / n).exists() for n in CORE_PY}
    core_data_present = {n: (bundle_outputs / n).exists() for n in CORE_DATA}
    standalone_present = {n: (bundle_outputs / n).exists() for n in STANDALONE_TESTS}
    checks["core_python_present"] = all(core_py_present.values())
    checks["core_data_present"] = all(core_data_present.values())
    checks["standalone_selftests_present"] = all(standalone_present.values())
    details["core_python_present"] = core_py_present
    details["core_data_present"] = core_data_present
    details["standalone_selftests_present"] = standalone_present

    py_files = [p for p in bundle_outputs.glob("*.py") if p.is_file()]
    syntax_rows: List[Dict[str, Any]] = []
    syntax_all = True
    for p in py_files:
        ok, err = syntax_ok(p)
        syntax_rows.append({"file": p.name, "syntax_ok": ok, "error": err})
        syntax_all = syntax_all and ok
    checks["all_runtime_and_standalone_python_syntax_ok"] = syntax_all
    details["syntax_bad"] = [r for r in syntax_rows if not r["syntax_ok"]]

    runtime_and_standalone = [bundle_outputs / n for n in CORE_PY + STANDALONE_TESTS if (bundle_outputs / n).exists()]
    abs_bad = [p.name for p in runtime_and_standalone if has_abs_bbit(p)]
    checks["no_absolute_bbit_paths_in_runtime_or_standalone_python"] = not abs_bad
    details["absolute_path_bad_files"] = abs_bad

    fp_files = [p.name for p in runtime_and_standalone + [bundle_outputs / n for n in CORE_DATA if (bundle_outputs / n).exists()] if contains_fingerprint(p)]
    checks["fingerprint_found_in_runtime_or_standalone_core"] = bool(fp_files)
    details["fingerprint_files"] = fp_files

    # Write manifest after all normalization. Exclude itself from the hash list by policy.
    manifest_rel = f"outputs_basic32/{FINAL_MANIFEST_NAME}"
    manifest = build_manifest(bundle_root, manifest_rel)
    manifest_path = bundle_outputs / FINAL_MANIFEST_NAME
    write_json(manifest_path, manifest)
    manifest_ok, m_checked, m_matched, m_bad = verify_manifest(bundle_root, manifest)
    checks["manifest_hashes_match_excluding_self_manifest"] = manifest_ok
    checks["self_manifest_policy_present"] = manifest.get("self_manifest_policy", {}).get("hash_mode") == "excluded_from_own_file_hash_list"
    details["manifest_checked"] = m_checked
    details["manifest_matched"] = m_matched
    details["manifest_bad"] = m_bad
    details["manifest_path"] = str(manifest_path)

    replay = run_replay(bundle_root)
    for r in replay:
        print(f"[{PHASE}] replay {r.name}: rc={r.rc} pass={r.pass_detected} leak={r.abs_path_leak} elapsed={r.elapsed_s}s")
    checks["all_cleanroom_replay_commands_exit_zero"] = all(r.rc == 0 for r in replay)
    checks["all_cleanroom_replay_commands_pass_detected"] = all(r.pass_detected for r in replay)
    checks["no_absolute_bbit_path_leak_in_command_output"] = not any(r.abs_path_leak for r in replay)

    if args.write_final_zip:
        zip_dir(bundle_root, final_zip)
    checks["final_zip_written_if_requested"] = (not args.write_final_zip) or final_zip.exists()

    lock_pass = all(checks.values())

    # Outputs
    replay_rows = [r.__dict__ for r in replay]
    write_csv(outputs / "phase26fc_lite_command_replay.csv", replay_rows)
    write_csv(outputs / "phase26fc_lite_python_syntax_audit.csv", syntax_rows)
    summary = {
        "phase": PHASE,
        "FINAL_SELF_MANIFEST_SEAL_LOCK_PASS": lock_pass,
        "root": str(root),
        "outputs": str(outputs),
        "input_zip": str(in_zip),
        "final_zip": str(final_zip),
        "fingerprint": FINGERPRINT,
        "checks": checks,
        "details": details,
        "commands": replay_rows,
    }
    write_json(outputs / "phase26fc_lite_summary.json", summary)
    write_json(outputs / "phase26fc_lite_self_manifest_seal_manifest.json", summary)

    report_lines = [
        "# 26FC-LITE self-manifest seal finalizer report",
        "",
        f"- `FINAL_SELF_MANIFEST_SEAL_LOCK_PASS`: `{lock_pass}`",
        f"- `input_zip`: `{in_zip}`",
        f"- `final_zip`: `{final_zip}`",
        f"- `fingerprint`: `{FINGERPRINT}`",
        "",
        "## Checks",
    ]
    for k, v in checks.items():
        report_lines.append(f"- `{k}`: `{v}`")
    report_lines.extend(["", "## Manifest policy", "", "The final manifest excludes its own file hash by explicit policy. All other files are hash-locked.", "", "## Replay commands"])
    for r in replay:
        report_lines.extend([
            "",
            f"### {r.name}",
            f"- rc: `{r.rc}`",
            f"- pass_detected: `{r.pass_detected}`",
            f"- abs_path_leak: `{r.abs_path_leak}`",
            "",
            "stdout tail:",
            "```",
            r.stdout_tail,
            "```",
            "",
            "stderr tail:",
            "```",
            r.stderr_tail,
            "```",
        ])
    (outputs / "phase26fc_lite_self_manifest_seal_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    make_plot(outputs / "phase26fc_lite_self_manifest_seal_checks.png", checks)

    print(f"[{PHASE}] FINAL_SELF_MANIFEST_SEAL_LOCK_PASS={lock_pass}")
    print(f"[{PHASE}] checks:")
    for k, v in checks.items():
        print(f"  - {k}: {v}")
    if args.write_final_zip:
        print(f"[{PHASE}] wrote final zip: {final_zip}")
    print(f"[{PHASE}] wrote report: {outputs / 'phase26fc_lite_self_manifest_seal_report.md'}")
    print(f"[{PHASE}] wrote summary: {outputs / 'phase26fc_lite_summary.json'}")
    print(f"[{PHASE}] wrote outputs to: {outputs}")
    return 0 if lock_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
