#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
26EZ-LITE portable core path resealer / clean-room replay finalizer.

Purpose
-------
26EY proved the zip is almost portable, but two runtime-core files still carried
absolute E:\BBIT path assumptions:

  - phase26eq_lite_runtime_exact_pair_gate.py
  - phase26eq_lite_runtime_exact_pair_gate_selftest.py

and the EO selftest still imported its helper from E:\BBIT\outputs_basic32.
That made the clean-room replay fail even though the exact-pair contract itself
was valid.

26EZ extracts the latest bundle, rewrites the runtime core so every helper is
resolved from the local extracted bundle directory, reseals the manifest with
fresh hashes, writes a final zip, extracts that zip into a clean room, and runs
only the portable replay chain from inside the clean-room folder.

Run from E:\BBIT:

  python bbit_geomlang/geomlang_phase26ez_lite_portable_core_path_resealer_cuda_basic32_E_drive.py --fresh --write-final-zip

Optional explicit input:

  python bbit_geomlang/geomlang_phase26ez_lite_portable_core_path_resealer_cuda_basic32_E_drive.py --zip E:\BBIT\outputs_basic32\phase26ey_lite_final_portable_release_bundle.zip --fresh --write-final-zip
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

FINGERPRINT = "98ebdcbb8e995bc1"
PHASE = "26EZ-LITE"

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
    "phase26em_lite_contract_row_smoke.csv",
    "phase26ej_lite_integration_contract.json",
    "phase26em_lite_transplant_manifest.json",
    "phase26en_lite_live_transplant_manifest.json",
    "phase26eq_lite_shadow_transplant_manifest.json",
]

REPLAY_COMMANDS = [
    ("em_router_selftest", "phase26em_lite_runtime_route_selftest.py", "EM_ROUTER_SELFTEST_PASS", "runtime route selftest PASS"),
    ("en_guard_selftest", "phase26en_lite_live_guard_selftest.py", "EN_LIVE_GUARD_SELFTEST_PASS", "live contract guard selftest PASS"),
    ("eo_selftest", "phase26eo_lite_transplant_selftest.py", "EO_SELFTEST_PASS", None),
    ("eq_gate_selftest", "phase26eq_lite_runtime_exact_pair_gate_selftest.py", "runtime exact-pair gate selftest PASS", None),
]

ABS_MARKERS = [
    "E:\\BBIT",
    "E:/BBIT",
    "\\BBIT\\",
    "/BBIT/",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def short_hash(path: Path) -> str:
    return sha256_file(path)[:16]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def syntax_ok(path: Path) -> Tuple[bool, str]:
    try:
        ast.parse(read_text(path), filename=str(path))
        return True, ""
    except SyntaxError as exc:
        return False, f"{exc.__class__.__name__}: {exc}"


def has_abs_bbit_marker(text: str) -> bool:
    return any(m in text for m in ABS_MARKERS)


def has_abs_bbit_path_in_output(text: str) -> bool:
    lowered = text.replace("/", "\\")
    return "E:\\BBIT" in lowered or "\\BBIT\\" in lowered


def root_from_script() -> Path:
    here = Path(__file__).resolve()
    if here.parent.name == "bbit_geomlang":
        return here.parent.parent
    if (Path.cwd() / "bbit_geomlang").exists():
        return Path.cwd().resolve()
    return here.parent.resolve()


def default_zip(outputs: Path) -> Path:
    candidates = [
        outputs / "phase26ey_lite_final_portable_release_bundle.zip",
        outputs / "phase26ev_lite_release_bundle_repaired.zip",
        outputs / "phase26et_lite_release_bundle.zip",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def extract_zip(zip_path: Path, stage_dir: Path, *, fresh: bool) -> None:
    if fresh:
        safe_rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(stage_dir)


def flatten_if_single_bundle_dir(stage_dir: Path) -> Path:
    children = [p for p in stage_dir.iterdir() if p.name not in {"__MACOSX"}]
    dirs = [p for p in children if p.is_dir()]
    files = [p for p in children if p.is_file()]
    if not files and len(dirs) == 1:
        return dirs[0]
    return stage_dir


def find_file(base: Path, name: str) -> Optional[Path]:
    p = base / name
    if p.exists():
        return p
    hits = list(base.rglob(name))
    return hits[0] if hits else None


def copy_missing_from_outputs(base: Path, outputs: Path, names: Iterable[str]) -> List[str]:
    copied: List[str] = []
    for name in names:
        if find_file(base, name):
            continue
        src = outputs / name
        if src.exists():
            shutil.copy2(src, base / name)
            copied.append(name)
    return copied


def patch_eq_gate(path: Path) -> bool:
    old = read_text(path)
    new = old
    new = new.replace(
        '_OUT = Path(r"E:\\BBIT\\outputs_basic32")\nif str(_OUT) not in sys.path:\n    sys.path.insert(0, str(_OUT))',
        '_OUT = Path(__file__).resolve().parent\nif str(_OUT) not in sys.path:\n    sys.path.insert(0, str(_OUT))',
    )
    new = new.replace(
        'OUT = Path(r"E:\\BBIT\\outputs_basic32")\nif str(OUT) not in sys.path:\n    sys.path.insert(0, str(OUT))',
        'OUT = Path(__file__).resolve().parent\nif str(OUT) not in sys.path:\n    sys.path.insert(0, str(OUT))',
    )
    if new != old:
        write_text(path, new)
        return True
    return False


def patch_eo_selftest(path: Path) -> bool:
    old = read_text(path)
    new = old.replace(
        'HELPER = Path(\'E:\\\\BBIT\\\\outputs_basic32\\\\phase26eo_lite_exact_pair_route_helper.py\')',
        'HELPER = Path(__file__).resolve().parent / "phase26eo_lite_exact_pair_route_helper.py"',
    )
    new = new.replace(
        'HELPER = Path("E:\\\\BBIT\\\\outputs_basic32\\\\phase26eo_lite_exact_pair_route_helper.py")',
        'HELPER = Path(__file__).resolve().parent / "phase26eo_lite_exact_pair_route_helper.py"',
    )
    if new != old:
        write_text(path, new)
        return True
    return False


def patch_runtime_core(base: Path) -> List[str]:
    patched: List[str] = []
    for name in ["phase26eq_lite_runtime_exact_pair_gate.py", "phase26eq_lite_runtime_exact_pair_gate_selftest.py"]:
        p = find_file(base, name)
        if p and patch_eq_gate(p):
            patched.append(name)
    p = find_file(base, "phase26eo_lite_transplant_selftest.py")
    if p and patch_eo_selftest(p):
        patched.append("phase26eo_lite_transplant_selftest.py")
    return patched


def files_for_manifest(base: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in sorted([x for x in base.rglob("*") if x.is_file()]):
        rel = p.relative_to(base).as_posix()
        if rel.endswith(".zip"):
            continue
        rows.append({
            "path": rel,
            "name": p.name,
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        })
    return rows


def write_resealed_manifest(base: Path, source_zip: Path, patched: List[str], copied: List[str]) -> Path:
    manifest = {
        "phase": PHASE,
        "title": "Final portable exact-pair release bundle reseal",
        "created_utc_unix": int(time.time()),
        "source_zip": source_zip.name,
        "route_fingerprint": FINGERPRINT,
        "portable_contract": "exact-pair helper resolution must be local to extracted bundle",
        "patched_files": patched,
        "copied_missing_artifacts": copied,
        "core_python_files": CORE_PY,
        "core_data_files": CORE_DATA,
        "files": files_for_manifest(base),
    }
    path = base / "phase26ez_lite_final_portable_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def verify_manifest_hashes(manifest_path: Path, base: Path) -> Tuple[bool, str]:
    data = json.loads(read_text(manifest_path))
    total = 0
    ok = 0
    bad: List[str] = []
    for row in data.get("files", []):
        rel = row.get("path") or row.get("relative_path") or row.get("name")
        expected = row.get("sha256") or row.get("hash")
        if not rel or not expected:
            continue
        total += 1
        p = base / str(rel)
        if p.exists() and sha256_file(p) == expected:
            ok += 1
        else:
            bad.append(str(rel))
    return (total > 0 and total == ok), f"{ok}/{total}" + (f" bad={bad[:5]}" if bad else "")


def zip_dir(src_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted([x for x in src_dir.rglob("*") if x.is_file()]):
            zf.write(p, p.relative_to(src_dir).as_posix())


@dataclass
class CommandResult:
    name: str
    script: str
    rc: int
    elapsed: float
    pass_detected: bool
    leak: bool
    stdout_tail: str
    stderr_tail: str


def run_replay(base: Path) -> List[CommandResult]:
    results: List[CommandResult] = []
    env = os.environ.copy()
    env["PYTHONPATH"] = str(base) + os.pathsep + env.get("PYTHONPATH", "")
    for name, script, token_a, token_b in REPLAY_COMMANDS:
        p = find_file(base, script)
        if not p:
            results.append(CommandResult(name, script, 999, 0.0, False, False, "missing", ""))
            continue
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, str(p)],
            cwd=str(base),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        elapsed = time.time() - t0
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        pass_detected = bool((token_a and token_a in combined) or (token_b and token_b in combined))
        # EO prints JSON with a boolean; treat that separately.
        if name == "eo_selftest" and '"EO_SELFTEST_PASS": true' in combined:
            pass_detected = True
        if name == "em_router_selftest" and "EM_RUNTIME_ROUTE_SELFTEST_PASS" in combined:
            pass_detected = True
        if name == "en_guard_selftest" and "EN_LIVE_GUARD_SELFTEST_PASS" in combined:
            pass_detected = True
        results.append(CommandResult(
            name=name,
            script=script,
            rc=proc.returncode,
            elapsed=elapsed,
            pass_detected=pass_detected,
            leak=has_abs_bbit_path_in_output(combined),
            stdout_tail=(proc.stdout or "")[-2000:],
            stderr_tail=(proc.stderr or "")[-2000:],
        ))
    return results


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_plot(path: Path, checks: Dict[str, Tuple[bool, str]]) -> None:
    try:
        import matplotlib.pyplot as plt
        labels = list(checks.keys())[::-1]
        vals = [1 if checks[k][0] else 0 for k in labels]
        fig_h = max(6, 0.46 * len(labels))
        fig, ax = plt.subplots(figsize=(14, fig_h))
        ax.barh(labels, vals)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("pass = 1, fail = 0")
        ax.set_title("26EZ-LITE final portable clean-room replay checks")
        for i, v in enumerate(vals):
            ax.text(1.02, i, "PASS" if v else "FAIL", va="center")
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
    except Exception:
        pass


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", dest="zip_path", default=None, help="Input release zip. Defaults to EY, then EV, then ET zip in outputs_basic32.")
    ap.add_argument("--fresh", action="store_true", help="Remove prior staging/clean-room folders before extracting.")
    ap.add_argument("--write-final-zip", action="store_true", help="Write phase26ez_lite_final_portable_release_bundle.zip.")
    args = ap.parse_args(argv)

    root = root_from_script()
    outputs = root / "outputs_basic32"
    outputs.mkdir(parents=True, exist_ok=True)
    zip_path = Path(args.zip_path).resolve() if args.zip_path else default_zip(outputs).resolve()
    stage_parent = outputs / "phase26ez_lite_stage"
    stage = stage_parent / "bundle_stage"
    clean_parent = outputs / "phase26ez_lite_cleanroom_extract"
    cleanroom = clean_parent / f"cleanroom_{time.strftime('%Y%m%d_%H%M%S')}"
    final_zip = outputs / "phase26ez_lite_final_portable_release_bundle.zip"

    print(f"[{PHASE}] Final portable exact-pair core path resealer")
    print(f"[{PHASE}] root: {root}")
    print(f"[{PHASE}] outputs: {outputs}")
    print(f"[{PHASE}] input zip: {zip_path}")

    checks: Dict[str, Tuple[bool, str]] = {}
    command_results: List[CommandResult] = []
    patched: List[str] = []
    copied: List[str] = []
    manifest_path: Optional[Path] = None

    checks["zip_exists"] = (zip_path.exists(), str(zip_path))
    checks["zip_is_file"] = (zip_path.is_file(), str(zip_path) if zip_path.exists() else "missing")

    if zip_path.is_file():
        extract_zip(zip_path, stage, fresh=True)
        bundle = flatten_if_single_bundle_dir(stage)
        copied = copy_missing_from_outputs(bundle, outputs, CORE_PY + CORE_DATA)
        patched = patch_runtime_core(bundle)

        core_present = {name: bool(find_file(bundle, name)) for name in CORE_PY}
        data_present = {name: bool(find_file(bundle, name)) for name in CORE_DATA}
        checks["core_python_present"] = (all(core_present.values()), json.dumps(core_present, sort_keys=True))
        checks["core_data_present"] = (all(data_present.values()), json.dumps(data_present, sort_keys=True))
        checks["runtime_core_path_patches_applied_or_unneeded"] = (all(bool(find_file(bundle, n)) for n in CORE_PY), ", ".join(patched) if patched else "no text changes needed")

        syntax_rows: List[Dict[str, Any]] = []
        syntax_all = True
        for name in CORE_PY:
            p = find_file(bundle, name)
            if not p:
                syntax_all = False
                syntax_rows.append({"file": name, "syntax_ok": False, "detail": "missing"})
                continue
            ok, detail = syntax_ok(p)
            syntax_all = syntax_all and ok
            syntax_rows.append({"file": name, "syntax_ok": ok, "detail": detail})
        checks["all_runtime_core_python_syntax_ok"] = (syntax_all, f"{sum(1 for r in syntax_rows if r['syntax_ok'])}/{len(syntax_rows)}")

        leak_files = []
        for name in CORE_PY:
            p = find_file(bundle, name)
            if p and has_abs_bbit_marker(read_text(p)):
                leak_files.append(name)
        checks["no_absolute_bbit_paths_in_runtime_core_python"] = (not leak_files, str(leak_files))

        core_text = "\n".join(read_text(find_file(bundle, n)) for n in CORE_PY if find_file(bundle, n))
        checks["fingerprint_found_in_runtime_core"] = (FINGERPRINT in core_text, FINGERPRINT)

        manifest_path = write_resealed_manifest(bundle, zip_path, patched, copied)
        mh_ok, mh_detail = verify_manifest_hashes(manifest_path, bundle)
        checks["manifest_hashes_match_after_reseal"] = (mh_ok, mh_detail)

        if args.write_final_zip:
            zip_dir(bundle, final_zip)
        checks["final_zip_written_if_requested"] = ((not args.write_final_zip) or final_zip.exists(), str(final_zip) if args.write_final_zip else "not requested")

        # Clean-room replay from the newly resealed zip if requested, otherwise from staged bundle.
        replay_zip = final_zip if args.write_final_zip else zip_path
        extract_zip(replay_zip, cleanroom, fresh=True)
        clean_bundle = flatten_if_single_bundle_dir(cleanroom)
        checks["cleanroom_extract_created"] = (clean_bundle.exists(), str(clean_bundle))

        clean_syntax_all = True
        for name in CORE_PY:
            p = find_file(clean_bundle, name)
            ok = bool(p and syntax_ok(p)[0])
            clean_syntax_all = clean_syntax_all and ok
        checks["all_cleanroom_runtime_core_python_syntax_ok"] = (clean_syntax_all, "")

        clean_leaks = []
        for name in CORE_PY:
            p = find_file(clean_bundle, name)
            if p and has_abs_bbit_marker(read_text(p)):
                clean_leaks.append(name)
        checks["no_absolute_bbit_paths_in_cleanroom_runtime_core_python"] = (not clean_leaks, str(clean_leaks))

        command_results = run_replay(clean_bundle)
        for r in command_results:
            print(f"[{PHASE}] replay {r.name}: rc={r.rc} pass={r.pass_detected} leak={r.leak} elapsed={r.elapsed:.3f}s")
        checks["all_cleanroom_replay_commands_exit_zero"] = (all(r.rc == 0 for r in command_results), str({r.name: r.rc for r in command_results}))
        checks["all_cleanroom_replay_commands_pass_detected"] = (all(r.pass_detected for r in command_results), str({r.name: r.pass_detected for r in command_results}))
        checks["no_absolute_bbit_path_leak_in_command_output"] = (not any(r.leak for r in command_results), str({r.name: r.leak for r in command_results}))

        write_csv(outputs / "phase26ez_lite_runtime_core_syntax.csv", syntax_rows)
        write_csv(outputs / "phase26ez_lite_command_replay.csv", [r.__dict__ for r in command_results])

    lock_pass = all(ok for ok, _ in checks.values())
    summary = {
        "phase": PHASE,
        "FINAL_PORTABLE_CLEANROOM_LOCK_PASS": lock_pass,
        "root": str(root),
        "outputs": str(outputs),
        "input_zip": str(zip_path),
        "final_zip": str(final_zip) if args.write_final_zip else None,
        "route_fingerprint": FINGERPRINT,
        "patched_files": patched,
        "copied_missing_artifacts": copied,
        "checks": {k: {"pass": v[0], "detail": v[1]} for k, v in checks.items()},
        "command_replay": [r.__dict__ for r in command_results],
    }
    (outputs / "phase26ez_lite_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = [
        f"# {PHASE} final portable clean-room replay report",
        "",
        f"FINAL_PORTABLE_CLEANROOM_LOCK_PASS={lock_pass}",
        "",
        f"input_zip: `{zip_path}`",
        f"final_zip: `{final_zip if args.write_final_zip else 'not requested'}`",
        f"fingerprint: `{FINGERPRINT}`",
        "",
        "## Checks",
        "",
    ]
    for k, (ok, detail) in checks.items():
        report.append(f"- {'PASS' if ok else 'FAIL'} `{k}` — {detail}")
    report += ["", "## Replayed commands", ""]
    for r in command_results:
        report.append(f"- `{r.name}` rc={r.rc} pass={r.pass_detected} leak={r.leak} elapsed={r.elapsed:.3f}s")
    (outputs / "phase26ez_lite_final_portable_cleanroom_report.md").write_text("\n".join(report), encoding="utf-8")

    write_plot(outputs / "phase26ez_lite_final_portable_cleanroom_lock_checks.png", checks)

    print(f"[{PHASE}] FINAL_PORTABLE_CLEANROOM_LOCK_PASS={lock_pass}")
    print(f"[{PHASE}] checks:")
    for k, (ok, detail) in checks.items():
        print(f"  - {k}: {ok} :: {detail}")
    print(f"[{PHASE}] wrote summary: {outputs / 'phase26ez_lite_summary.json'}")
    print(f"[{PHASE}] wrote report: {outputs / 'phase26ez_lite_final_portable_cleanroom_report.md'}")
    if args.write_final_zip:
        print(f"[{PHASE}] wrote final zip: {final_zip}")
    print(f"[{PHASE}] wrote outputs to: {outputs}")
    return 0 if lock_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
