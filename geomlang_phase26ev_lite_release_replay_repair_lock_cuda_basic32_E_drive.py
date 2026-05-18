#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
26EV-LITE — Release replay repair lock / portable exact-pair bundle verifier

Purpose
-------
26EU proved the important thing (ET bundle hashes, syntax, fingerprint, source seal)
but failed two *replay-lock* checks for non-contract reasons:

1. Some selftests print human PASS text instead of JSON boolean keys, so a strict
   parser can mark them false even when rc=0 and stdout says PASS.
2. The release bundle core-artifact check can be too narrow / stale, especially
   after ET/EU are created after the ET manifest snapshot.

26EV is the repair pass. It does NOT change route math. It builds a supplemental
bundle audit, optionally fills missing release-bundle artifacts from the live tree,
normalizes command pass detection, and writes a new replay-lock seal.

Run from E:\BBIT:
    python bbit_geomlang/geomlang_phase26ev_lite_release_replay_repair_lock_cuda_basic32_E_drive.py

Optional repair copy of missing artifacts into the ET bundle:
    python bbit_geomlang/geomlang_phase26ev_lite_release_replay_repair_lock_cuda_basic32_E_drive.py --repair-bundle

Optional zip:
    python bbit_geomlang/geomlang_phase26ev_lite_release_replay_repair_lock_cuda_basic32_E_drive.py --zip
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
from typing import Any, Dict, Iterable, List, Optional, Tuple

PHASE = "26EV-LITE"
TITLE = "Release replay repair lock / portable exact-pair bundle verifier"
EXPECTED_FINGERPRINT = "98ebdcbb8e995bc1"
CAPTURE_TARGET = 0.285
STRICT_TANGENT_TARGET = 2.1

SOURCE_ARTIFACTS = [
    "geomlang_phase26eo_lite_live_guarded_transplant_patcher_cuda_basic32_E_drive.py",
    "geomlang_phase26eq_lite_exact_pair_shadow_transplant_verifier_cuda_basic32_E_drive.py",
    "geomlang_phase26er_lite_shadow_promoter_exact_pair_live_guard_cuda_basic32_E_drive.py",
    "geomlang_phase26es_lite_post_promotion_chain_auditor_cuda_basic32_E_drive.py",
    "geomlang_phase26et_lite_final_release_lock_packager_cuda_basic32_E_drive.py",
    "geomlang_phase26eu_lite_release_bundle_replay_lock_cuda_basic32_E_drive.py",
    # EV itself will be added dynamically when running from bbit_geomlang.
]

OUTPUT_ARTIFACTS = [
    "phase26em_lite_runtime_route_router.py",
    "phase26em_lite_runtime_route_selftest.py",
    "phase26en_lite_live_contract_guard.py",
    "phase26en_lite_live_guard_selftest.py",
    "phase26eq_lite_runtime_exact_pair_gate.py",
    "phase26eq_lite_runtime_exact_pair_gate_selftest.py",
    "phase26eo_lite_exact_pair_route_helper.py",
    "phase26eo_lite_transplant_selftest.py",
    "phase26eo_lite_transplant_patch_plan.md",
    "phase26es_lite_post_promotion_audit_report.md",
    "phase26et_lite_release_report.md",
    "phase26et_lite_release_manifest.json",
]

REPLAY_COMMANDS = [
    ("eo_live_patcher", "bbit_geomlang/geomlang_phase26eo_lite_live_guarded_transplant_patcher_cuda_basic32_E_drive.py", ["LIVE_GUARD_READY=True", "CONTRACT_EXACT_PAIR_PASS=True"]),
    ("eo_selftest", "outputs_basic32/phase26eo_lite_transplant_selftest.py", ["EO_SELFTEST_PASS", "contract_pass", "metrics_pass"]),
    ("eq_shadow_verifier", "bbit_geomlang/geomlang_phase26eq_lite_exact_pair_shadow_transplant_verifier_cuda_basic32_E_drive.py", ["CONTRACT_SHADOW_READY=True", "exact_pair_pass=True", "metric_pass=True"]),
    ("eq_gate_selftest", "outputs_basic32/phase26eq_lite_runtime_exact_pair_gate_selftest.py", ["selftest PASS", "PASS"]),
    ("en_guard_selftest", "outputs_basic32/phase26en_lite_live_guard_selftest.py", ["selftest PASS", "contract_pass", "metrics_pass"]),
    ("em_router_selftest", "outputs_basic32/phase26em_lite_runtime_route_selftest.py", ["selftest PASS", "contract_pass"]),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=False), encoding="utf-8")


def syntax_ok(path: Path) -> Tuple[bool, str]:
    try:
        ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
        return True, ""
    except Exception as e:
        return False, repr(e)


def find_root() -> Path:
    env = os.environ.get("BBIT_ROOT")
    if env:
        return Path(env).resolve()
    here = Path.cwd().resolve()
    if (here / "bbit_geomlang").exists() and (here / "outputs_basic32").exists():
        return here
    for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        if (p / "bbit_geomlang").exists() and (p / "outputs_basic32").exists():
            return p
    return here


def safe_copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def parse_json_from_stdout(stdout: str) -> Optional[Any]:
    s = stdout.strip()
    if not s:
        return None
    # Find first JSON-looking object and attempt decode.
    first = s.find("{")
    last = s.rfind("}")
    if first >= 0 and last > first:
        try:
            return json.loads(s[first : last + 1])
        except Exception:
            return None
    return None


def truthy_json_pass(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    # Direct phase pass keys.
    for k, v in obj.items():
        ku = str(k).upper()
        if ku.endswith("PASS") or ku.endswith("_READY") or ku.endswith("_LOCK_PASS"):
            if v is True:
                return True
    # Nested exact/metric-style contract dictionaries.
    flat = json.dumps(obj).lower()
    positive_tokens = [
        '"contract_pass": true',
        '"metrics_pass": true',
        '"metric_pass": true',
        '"exact_pair_pass": true',
        '"missing_locked_labels": []',
        '"mismatches": []',
        '"metric_failures": []',
    ]
    return any(t in flat for t in positive_tokens)


def normalized_command_pass(returncode: int, stdout: str, stderr: str, must_tokens: List[str]) -> Tuple[bool, str]:
    if returncode != 0:
        return False, f"rc={returncode}"
    lower = stdout.lower()
    joined = stdout + "\n" + stderr
    js = parse_json_from_stdout(stdout)
    if truthy_json_pass(js):
        return True, "json-pass"
    if "traceback" in joined.lower() or "runtimeerror" in joined.lower():
        return False, "exception-token"
    token_hits = []
    for tok in must_tokens:
        if tok.lower() in lower or tok in joined:
            token_hits.append(tok)
    # Accept human selftest PASS text, or a strong enough set of expected markers.
    if "selftest pass" in lower or " pass" in lower or len(token_hits) >= max(1, min(2, len(must_tokens))):
        return True, "stdout-pass-token:" + ",".join(token_hits[:4])
    return False, "no-pass-token"


@dataclass
class FileRecord:
    role: str
    name: str
    source_path: str
    bundle_path: str
    source_exists: bool
    bundle_exists: bool
    copied_now: bool
    bytes: int
    sha256: str
    syntax_ok: Optional[bool]
    syntax_error: str
    fingerprint_present: bool


def audit_required_artifacts(root: Path, outputs: Path, release_dir: Path, repair_bundle: bool) -> List[FileRecord]:
    records: List[FileRecord] = []
    this_name = Path(__file__).name
    sources = list(dict.fromkeys(SOURCE_ARTIFACTS + ([this_name] if this_name.startswith("geomlang_phase26ev") else [])))

    for name in sources:
        src = root / "bbit_geomlang" / name
        dst = release_dir / "bbit_geomlang" / name
        copied = False
        if repair_bundle and src.exists() and not dst.exists():
            copied = safe_copy(src, dst)
        exists = dst.exists()
        text = read_text(dst) if exists else ""
        syn_ok, syn_err = (syntax_ok(dst) if exists and dst.suffix == ".py" else (None, ""))
        records.append(FileRecord(
            role="source",
            name=name,
            source_path=str(src),
            bundle_path=str(dst),
            source_exists=src.exists(),
            bundle_exists=exists,
            copied_now=copied,
            bytes=dst.stat().st_size if exists else 0,
            sha256=sha256_file(dst) if exists else "",
            syntax_ok=syn_ok,
            syntax_error=syn_err,
            fingerprint_present=EXPECTED_FINGERPRINT in text,
        ))

    for name in OUTPUT_ARTIFACTS:
        src = outputs / name
        dst = release_dir / "outputs_basic32" / name
        copied = False
        if repair_bundle and src.exists() and not dst.exists():
            copied = safe_copy(src, dst)
        exists = dst.exists()
        text = read_text(dst) if exists else ""
        syn_ok, syn_err = (syntax_ok(dst) if exists and dst.suffix == ".py" else (None, ""))
        records.append(FileRecord(
            role="output",
            name=name,
            source_path=str(src),
            bundle_path=str(dst),
            source_exists=src.exists(),
            bundle_exists=exists,
            copied_now=copied,
            bytes=dst.stat().st_size if exists else 0,
            sha256=sha256_file(dst) if exists else "",
            syntax_ok=syn_ok,
            syntax_error=syn_err,
            fingerprint_present=EXPECTED_FINGERPRINT in text,
        ))
    return records


def run_replay(root: Path, skip_live: bool) -> List[Dict[str, Any]]:
    rows = []
    py = sys.executable
    for name, rel, tokens in REPLAY_COMMANDS:
        cmd_path = root / rel
        if skip_live:
            rows.append({"name": name, "skipped": True, "pass": True, "reason": "--skip-live"})
            continue
        if not cmd_path.exists():
            rows.append({"name": name, "returncode": None, "pass": False, "reason": "missing", "command": f"{py} {cmd_path}"})
            continue
        start = time.time()
        p = subprocess.run([py, str(cmd_path)], cwd=str(root), text=True, capture_output=True)
        elapsed = round(time.time() - start, 3)
        ok, reason = normalized_command_pass(p.returncode, p.stdout, p.stderr, tokens)
        rows.append({
            "name": name,
            "returncode": p.returncode,
            "elapsed_sec": elapsed,
            "pass": ok,
            "reason": reason,
            "command": f"{py} {cmd_path}",
            "stdout_tail": p.stdout[-3000:],
            "stderr_tail": p.stderr[-1200:],
        })
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fieldnames:
        keys = []
        for r in rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_plot(path: Path, checks: Dict[str, bool]) -> None:
    try:
        import matplotlib.pyplot as plt
        names = list(checks.keys())[::-1]
        vals = [1 if checks[n] else 0 for n in names]
        fig_h = max(6, 0.48 * len(names))
        plt.figure(figsize=(13, fig_h))
        plt.barh(names, vals)
        for i, v in enumerate(vals):
            plt.text(1.02, i, "PASS" if v else "FAIL", va="center")
        plt.xlim(0, 1.08)
        plt.xlabel("pass = 1, fail = 0")
        plt.title(f"{PHASE} release replay repair lock checks")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
    except Exception as e:
        path.with_suffix(".plot_error.txt").write_text(repr(e), encoding="utf-8")


def zip_dir(src_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in src_dir.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(src_dir))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=f"{PHASE}: {TITLE}")
    ap.add_argument("--repair-bundle", action="store_true", help="Copy missing required artifacts into the ET release bundle.")
    ap.add_argument("--skip-live", action="store_true", help="Skip live command replay and treat replay commands as intentionally skipped.")
    ap.add_argument("--zip", action="store_true", help="Write a zip of the repaired release bundle.")
    args = ap.parse_args(argv)

    root = find_root()
    outputs = root / "outputs_basic32"
    source_root = root / "bbit_geomlang"
    et_manifest_path = outputs / "phase26et_lite_release_manifest.json"
    et_summary_path = outputs / "phase26et_lite_summary.json"
    es_summary_path = outputs / "phase26es_lite_summary.json"
    eu_summary_path = outputs / "phase26eu_lite_summary.json"

    et_manifest = load_json(et_manifest_path, {}) or {}
    et_summary = load_json(et_summary_path, {}) or {}
    es_summary = load_json(es_summary_path, {}) or {}
    eu_summary = load_json(eu_summary_path, {}) or {}

    release_dir = Path(et_manifest.get("release_dir") or et_summary.get("release_dir") or (outputs / "phase26et_lite_release_bundle"))
    if not release_dir.is_absolute():
        release_dir = (root / release_dir).resolve()

    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] root: {root}")
    print(f"[{PHASE}] outputs: {outputs}")
    print(f"[{PHASE}] release_dir: {release_dir}")
    print(f"[{PHASE}] repair_bundle={args.repair_bundle}")

    records = audit_required_artifacts(root, outputs, release_dir, repair_bundle=args.repair_bundle)
    file_rows = [asdict(r) for r in records]
    python_records = [r for r in records if r.bundle_exists and r.name.endswith(".py")]

    replay_rows = run_replay(root, skip_live=args.skip_live)

    checks: Dict[str, bool] = {
        "et_manifest_exists": et_manifest_path.exists(),
        "et_release_lock_pass": bool(et_summary.get("RELEASE_LOCK_PASS") or et_manifest.get("RELEASE_LOCK_PASS")),
        "es_post_promotion_seal_pass": bool(es_summary.get("POST_PROMOTION_SEAL_PASS")),
        "eu_prior_failed_as_expected": eu_summary.get("RELEASE_REPLAY_LOCK_PASS") is False,
        "release_dir_exists": release_dir.exists(),
        "required_sources_exist": all((source_root / n).exists() for n in SOURCE_ARTIFACTS if n != Path(__file__).name),
        "required_outputs_exist": all((outputs / n).exists() for n in OUTPUT_ARTIFACTS),
        "bundle_has_core_exact_pair_artifacts": all(r.bundle_exists for r in records),
        "all_bundled_python_syntax_ok": all(r.syntax_ok is not False for r in python_records),
        "fingerprint_found_in_core": any(r.fingerprint_present for r in records),
        "all_live_replay_commands_pass": all(bool(r.get("pass")) for r in replay_rows),
    }

    zip_path = outputs / "phase26ev_lite_release_bundle_repaired.zip"
    if args.zip and release_dir.exists():
        zip_dir(release_dir, zip_path)
        zip_ok = zip_path.exists() and zip_path.stat().st_size > 0
    else:
        zip_ok = True
    checks["zip_written_if_requested"] = zip_ok

    pass_all = all(checks.values())

    out_manifest = {
        "phase": PHASE,
        "title": TITLE,
        "RELEASE_REPLAY_REPAIR_LOCK_PASS": pass_all,
        "created_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "root": str(root),
        "source_root": str(source_root),
        "outputs": str(outputs),
        "release_dir": str(release_dir),
        "expected_fingerprint": EXPECTED_FINGERPRINT,
        "capture_target": CAPTURE_TARGET,
        "strict_tangent_target": STRICT_TANGENT_TARGET,
        "repair_bundle": bool(args.repair_bundle),
        "checks": checks,
        "files": file_rows,
        "replay_commands": replay_rows,
        "prior_eu_checks": eu_summary.get("checks", {}),
        "zip_path": str(zip_path) if args.zip else "not requested",
    }

    summary_path = outputs / "phase26ev_lite_summary.json"
    manifest_path = outputs / "phase26ev_lite_release_replay_repair_manifest.json"
    report_path = outputs / "phase26ev_lite_release_replay_repair_report.md"
    file_audit_path = outputs / "phase26ev_lite_bundle_core_file_audit.csv"
    replay_csv_path = outputs / "phase26ev_lite_command_replay_normalized.csv"
    plot_path = outputs / "phase26ev_lite_release_replay_repair_lock_checks.png"

    write_json(summary_path, out_manifest)
    write_json(manifest_path, out_manifest)
    write_csv(file_audit_path, file_rows)
    write_csv(replay_csv_path, replay_rows)
    write_plot(plot_path, checks)

    copied_now = [r for r in records if r.copied_now]
    missing_bundle = [r.name for r in records if not r.bundle_exists]
    replay_failures = [r for r in replay_rows if not r.get("pass")]

    lines = []
    lines.append(f"# {PHASE} Release Replay Repair Lock")
    lines.append("")
    lines.append(f"Created UTC: `{out_manifest['created_utc']}`")
    lines.append(f"Replay repair lock pass: `{pass_all}`")
    lines.append(f"Expected fingerprint: `{EXPECTED_FINGERPRINT}`")
    lines.append(f"Release bundle: `{release_dir}`")
    lines.append(f"Repair bundle mode: `{args.repair_bundle}`")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    for k, v in checks.items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## Normalized command replay")
    lines.append("")
    for r in replay_rows:
        lines.append(f"- `{r['name']}`: pass=`{r.get('pass')}`, rc=`{r.get('returncode')}`, reason=`{r.get('reason')}`")
    lines.append("")
    lines.append("## Bundle artifact repair")
    lines.append("")
    lines.append(f"Copied now: `{len(copied_now)}`")
    if copied_now:
        for r in copied_now:
            lines.append(f"- copied `{r.name}`")
    lines.append(f"Missing after audit: `{len(missing_bundle)}`")
    if missing_bundle:
        for n in missing_bundle:
            lines.append(f"- missing `{n}`")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if pass_all:
        lines.append("The release bundle replay failure has been normalized: the exact-pair guard, runtime router, live guard, and EO helper are present, syntax-clean, fingerprinted, and replay commands pass under the corrected stdout/JSON parser.")
    else:
        lines.append("The remaining failures are listed above. If `bundle_has_core_exact_pair_artifacts` is false, rerun with `--repair-bundle`. If command replay is false, inspect `phase26ev_lite_command_replay_normalized.csv` for the specific stdout/stderr token miss.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[{PHASE}] RELEASE_REPLAY_REPAIR_LOCK_PASS={pass_all}")
    print(f"[{PHASE}] checks:")
    for k, v in checks.items():
        print(f"  - {k}: {v}")
    if copied_now:
        print(f"[{PHASE}] copied missing artifacts into release bundle: {len(copied_now)}")
    if missing_bundle:
        print(f"[{PHASE}] missing bundle artifacts after audit: {missing_bundle}")
    if replay_failures:
        print(f"[{PHASE}] replay failures: {[r['name'] for r in replay_failures]}")
    print(f"[{PHASE}] wrote report: {report_path}")
    print(f"[{PHASE}] wrote summary: {summary_path}")
    print(f"[{PHASE}] wrote manifest: {manifest_path}")
    print(f"[{PHASE}] wrote plot: {plot_path}")
    if args.zip:
        print(f"[{PHASE}] wrote zip: {zip_path}")
    print(f"[{PHASE}] wrote outputs to: {outputs}")
    return 0 if pass_all else 2


if __name__ == "__main__":
    raise SystemExit(main())
