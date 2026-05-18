#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
26EY-LITE final portable bundle normalizer / manifest resealer

Purpose
-------
Takes the repaired 26EV/26EX release bundle, stages it in a clean folder,
repairs missing exact-pair runtime artifacts, removes machine-local path leaks
from the runtime core where safe, rebuilds the manifest from the *actual staged
files*, writes a final portable zip, then extracts that final zip into a fresh
clean-room and replays the exact-pair runtime selftests.

Default Windows usage:
    python bbit_geomlang/geomlang_phase26ey_lite_final_portable_bundle_resealer_cuda_basic32_E_drive.py --fresh --write-final-zip

Optional:
    python bbit_geomlang/geomlang_phase26ey_lite_final_portable_bundle_resealer_cuda_basic32_E_drive.py --zip E:\\BBIT\\outputs_basic32\\phase26ev_lite_release_bundle_repaired.zip --fresh --write-final-zip

This script is deliberately conservative: it does not change your live source
files. It only writes a resealed portable bundle under outputs_basic32.
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

PHASE = "26EY-LITE"
FINGERPRINT = "98ebdcbb8e995bc1"
FINAL_ZIP_NAME = "phase26ey_lite_final_portable_release_bundle.zip"
STAGE_DIR_NAME = "phase26ey_lite_reseal_stage"
CLEANROOM_DIR_NAME = "phase26ey_lite_cleanroom_extract"

CORE_RUNTIME_FILES = [
    "phase26em_lite_runtime_route_router.py",
    "phase26em_lite_runtime_route_selftest.py",
    "phase26en_lite_live_contract_guard.py",
    "phase26en_lite_live_guard_selftest.py",
    "phase26eo_lite_exact_pair_route_helper.py",
    "phase26eo_lite_transplant_selftest.py",
    "phase26eq_lite_runtime_exact_pair_gate.py",
    "phase26eq_lite_runtime_exact_pair_gate_selftest.py",
]

CORE_DATA_FILES = [
    "phase26em_lite_contract_row_smoke.csv",
    "phase26ej_lite_integration_contract.json",
    "phase26em_lite_transplant_manifest.json",
    "phase26en_lite_live_transplant_manifest.json",
    "phase26eq_lite_shadow_transplant_manifest.json",
]

REPLAY_COMMANDS = [
    ("em_router_selftest", "phase26em_lite_runtime_route_selftest.py", ["EM_ROUTER_SELFTEST_PASS", "ROUTER_SELFTEST_PASS", '"pass": true', "true"]),
    ("en_guard_selftest", "phase26en_lite_live_guard_selftest.py", ["EN_GUARD_SELFTEST_PASS", "LIVE_GUARD_SELFTEST_PASS", '"pass": true', "true"]),
    ("eo_selftest", "phase26eo_lite_transplant_selftest.py", ["EO_SELFTEST_PASS", '"EO_SELFTEST_PASS": true']),
    ("eq_gate_selftest", "phase26eq_lite_runtime_exact_pair_gate_selftest.py", ["EQ_GATE_SELFTEST_PASS", "RUNTIME_EXACT_PAIR_GATE_SELFTEST_PASS", '"pass": true', "true"]),
]

ABS_PATH_PATTERNS = [
    re.compile(r"E:\\\\BBIT", re.IGNORECASE),
    re.compile(r"E:/BBIT", re.IGNORECASE),
    re.compile(r"E:\\BBIT", re.IGNORECASE),
]

# Path-looking strings that are allowed in docs/reports are not allowed in these runtime core files.
RUNTIME_CORE_SCAN_EXTS = {".py"}


@dataclass
class CommandResult:
    name: str
    script: str
    returncode: int
    pass_detected: bool
    abs_path_leak: bool
    elapsed_s: float
    stdout_tail: str
    stderr_tail: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def short_hash(path: Path) -> str:
    return sha256_file(path)[:16]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for k in row.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def tail_text(s: str, n: int = 4000) -> str:
    s = s or ""
    return s[-n:]


def has_abs_bbit_path(text: str) -> bool:
    return any(p.search(text or "") for p in ABS_PATH_PATTERNS)


def scrub_abs_bbit_text(text: str) -> Tuple[str, int]:
    """Replace local-machine BBIT path literals with a neutral token.

    This is intentionally simple and only used on staged portable files. If code
    relied on the literal path, the clean-room replay would fail, so replay is
    the guardrail.
    """
    count = 0
    out = text
    replacements = [
        (r"E:\\BBIT", "<BBIT_ROOT>"),
        (r"E:/BBIT", "<BBIT_ROOT>"),
        (r"E:\\\\BBIT", "<BBIT_ROOT>"),
    ]
    for old, new in replacements:
        out2 = re.sub(re.escape(old), new, out, flags=re.IGNORECASE)
        if out2 != out:
            count += out.count(old)
            out = out2
    # Also catch longer escaped fragments in docstrings/comments.
    out2 = re.sub(r"E:\\\\BBIT", "<BBIT_ROOT>", out, flags=re.IGNORECASE)
    if out2 != out:
        count += 1
        out = out2
    return out, count


def discover_root(script_path: Path) -> Path:
    # Typical live path: E:/BBIT/bbit_geomlang/this_script.py -> root E:/BBIT
    p = script_path.resolve()
    if p.parent.name.lower() == "bbit_geomlang":
        return p.parent.parent
    cwd = Path.cwd().resolve()
    if (cwd / "outputs_basic32").exists() and (cwd / "bbit_geomlang").exists():
        return cwd
    if cwd.name.lower() == "bbit_geomlang" and (cwd.parent / "outputs_basic32").exists():
        return cwd.parent
    return cwd


def safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def copytree_contents(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def extract_zip(zip_path: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            # Prevent zip-slip
            out_path = (dst / member.filename).resolve()
            if not str(out_path).startswith(str(dst.resolve())):
                raise RuntimeError(f"Unsafe zip member path: {member.filename}")
            zf.extract(member, dst)


def locate_bundle_root(extract_dir: Path) -> Path:
    # If files are directly in extraction root, use it. Otherwise pick first dir
    # that contains at least two core runtime files.
    if sum((extract_dir / f).exists() for f in CORE_RUNTIME_FILES) >= 2:
        return extract_dir
    best = extract_dir
    best_score = -1
    for d in [extract_dir] + [p for p in extract_dir.rglob("*") if p.is_dir()]:
        score = sum((d / f).exists() for f in CORE_RUNTIME_FILES)
        if score > best_score:
            best_score = score
            best = d
    return best


def ensure_core_artifacts(stage: Path, outputs: Path, source_root: Path) -> List[str]:
    copied: List[str] = []
    for name in CORE_RUNTIME_FILES + CORE_DATA_FILES:
        dst = stage / name
        if dst.exists():
            continue
        candidates = [outputs / name, source_root / name]
        found = next((c for c in candidates if c.exists()), None)
        if found:
            shutil.copy2(found, dst)
            copied.append(name)
    return copied


def syntax_check_python_files(root: Path) -> Tuple[bool, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    ok = True
    for py in sorted(root.rglob("*.py")):
        rel = py.relative_to(root).as_posix()
        try:
            src = py.read_text(encoding="utf-8")
            ast.parse(src, filename=rel)
            rows.append({"file": rel, "syntax_ok": True, "error": ""})
        except Exception as e:
            ok = False
            rows.append({"file": rel, "syntax_ok": False, "error": repr(e)})
    return ok, rows


def scrub_runtime_core(stage: Path) -> Tuple[List[Dict[str, Any]], bool, List[str]]:
    rows: List[Dict[str, Any]] = []
    remaining: List[str] = []
    for name in CORE_RUNTIME_FILES:
        path = stage / name
        if not path.exists():
            rows.append({"file": name, "exists": False, "scrubbed_replacements": 0, "abs_path_remaining": False})
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        before_leak = has_abs_bbit_path(text)
        new_text, repl = scrub_abs_bbit_text(text)
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
        after_text = path.read_text(encoding="utf-8", errors="replace")
        after_leak = has_abs_bbit_path(after_text)
        if after_leak:
            remaining.append(name)
        rows.append({
            "file": name,
            "exists": True,
            "had_abs_path_before": before_leak,
            "scrubbed_replacements": repl,
            "abs_path_remaining": after_leak,
        })
    return rows, not remaining, remaining


def scan_runtime_core_paths(stage: Path) -> Tuple[bool, List[str]]:
    leaks: List[str] = []
    for name in CORE_RUNTIME_FILES:
        path = stage / name
        if path.exists() and has_abs_bbit_path(path.read_text(encoding="utf-8", errors="replace")):
            leaks.append(name)
    return not leaks, leaks


def build_manifest(stage: Path, root: Path, source_zip: Optional[Path], copied: List[str], scrub_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    files: List[Dict[str, Any]] = []
    for p in sorted(stage.rglob("*")):
        if p.is_file():
            rel = p.relative_to(stage).as_posix()
            files.append({
                "path": rel,
                "size": p.stat().st_size,
                "sha256": sha256_file(p),
                "kind": "python" if p.suffix == ".py" else p.suffix.lstrip(".") or "file",
            })
    return {
        "phase": PHASE,
        "title": "Final portable exact-pair bundle manifest reseal",
        "created_utc_epoch": int(time.time()),
        "route_fingerprint": FINGERPRINT,
        "source_zip": str(source_zip) if source_zip else None,
        "file_count": len(files),
        "files": files,
        "core_runtime_files": CORE_RUNTIME_FILES,
        "core_data_files": CORE_DATA_FILES,
        "copied_missing_artifacts": copied,
        "scrubbed_runtime_core": scrub_rows,
        "notes": [
            "Manifest hashes are computed after bundle repair/scrub, not copied from older ET/EV manifests.",
            "Clean-room replay must pass from extracted zip contents.",
        ],
    }


def verify_manifest_hashes(stage: Path, manifest: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    ok = True
    for rec in manifest.get("files", []):
        rel = rec.get("path", "")
        p = stage / rel
        exists = p.exists()
        got = sha256_file(p) if exists else ""
        exp = rec.get("sha256", "")
        match = exists and got == exp
        if not match:
            ok = False
        rows.append({"path": rel, "exists": exists, "expected_sha256": exp, "actual_sha256": got, "match": match})
    return ok, rows


def write_zip_from_dir(src_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(src_dir.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(src_dir).as_posix())


def detect_pass(output: str, tokens: List[str]) -> bool:
    low = output.lower()
    if "false" in low and any(key in low for key in ["pass=false", '"pass": false', "selftest_pass=false", "lock_pass=false"]):
        return False
    return any(tok.lower() in low for tok in tokens)


def run_replay(cleanroom: Path) -> List[CommandResult]:
    results: List[CommandResult] = []
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cleanroom) + os.pathsep + env.get("PYTHONPATH", "")
    for name, script, tokens in REPLAY_COMMANDS:
        script_path = cleanroom / script
        start = time.time()
        if not script_path.exists():
            results.append(CommandResult(name, script, 127, False, False, 0.0, "", f"missing script: {script}"))
            continue
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(cleanroom),
            env=env,
            text=True,
            capture_output=True,
            timeout=120,
        )
        elapsed = time.time() - start
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        results.append(CommandResult(
            name=name,
            script=script,
            returncode=proc.returncode,
            pass_detected=(proc.returncode == 0 and detect_pass(combined, tokens)),
            abs_path_leak=has_abs_bbit_path(combined),
            elapsed_s=round(elapsed, 3),
            stdout_tail=tail_text(proc.stdout),
            stderr_tail=tail_text(proc.stderr),
        ))
    return results


def make_plot(check_rows: List[Dict[str, Any]], out_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        labels = [r["check"] for r in check_rows]
        vals = [1 if r["pass"] else 0 for r in check_rows]
        fig_h = max(6, 0.42 * len(labels))
        fig, ax = plt.subplots(figsize=(13, fig_h))
        y = list(range(len(labels)))
        ax.barh(y, vals)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("pass = 1, fail = 0")
        ax.set_title("26EY-LITE final portable reseal checks")
        for yi, v in zip(y, vals):
            ax.text(1.02, yi, "PASS" if v else "FAIL", va="center")
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=140)
        plt.close(fig)
    except Exception as e:
        out_png.with_suffix(".plot_error.txt").write_text(repr(e), encoding="utf-8")


def write_report(path: Path, summary: Dict[str, Any], cmd_results: List[CommandResult], check_rows: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append(f"# {PHASE} final portable bundle reseal report")
    lines.append("")
    lines.append(f"**FINAL_PORTABLE_RESEAL_LOCK_PASS:** `{summary['FINAL_PORTABLE_RESEAL_LOCK_PASS']}`")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| check | pass | detail |")
    lines.append("|---|---:|---|")
    for r in check_rows:
        lines.append(f"| `{r['check']}` | `{r['pass']}` | {str(r.get('detail','')).replace('|','/')} |")
    lines.append("")
    lines.append("## Clean-room replay")
    lines.append("")
    lines.append("| command | rc | pass_detected | abs_path_leak | elapsed_s |")
    lines.append("|---|---:|---:|---:|---:|")
    for c in cmd_results:
        lines.append(f"| `{c.name}` | {c.returncode} | `{c.pass_detected}` | `{c.abs_path_leak}` | {c.elapsed_s} |")
    lines.append("")
    lines.append("## Final zip")
    lines.append("")
    lines.append(f"`{summary.get('final_zip')}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=f"{PHASE} final portable bundle normalizer / manifest resealer")
    ap.add_argument("--zip", dest="zip_path", default=None, help="Input repaired release zip. Defaults to outputs_basic32/phase26ev_lite_release_bundle_repaired.zip")
    ap.add_argument("--fresh", action="store_true", help="Delete prior EY stage/cleanroom folders before running")
    ap.add_argument("--write-final-zip", action="store_true", help="Write the final resealed portable zip")
    ap.add_argument("--no-scrub", action="store_true", help="Do not scrub absolute BBIT paths from staged runtime core")
    args = ap.parse_args(argv)

    script_path = Path(__file__)
    root = discover_root(script_path)
    source_root = root / "bbit_geomlang"
    outputs = root / "outputs_basic32"
    outputs.mkdir(parents=True, exist_ok=True)

    input_zip = Path(args.zip_path).resolve() if args.zip_path else outputs / "phase26ev_lite_release_bundle_repaired.zip"
    stage_parent = outputs / STAGE_DIR_NAME
    clean_parent = outputs / CLEANROOM_DIR_NAME
    stamp = time.strftime("%Y%m%d_%H%M%S")
    stage = stage_parent / f"stage_{stamp}"
    cleanroom = clean_parent / f"cleanroom_{stamp}"
    final_zip = outputs / FINAL_ZIP_NAME

    print(f"[{PHASE}] Final portable bundle normalizer / manifest resealer")
    print(f"[{PHASE}] root: {root}")
    print(f"[{PHASE}] outputs: {outputs}")
    print(f"[{PHASE}] input zip: {input_zip}")

    if args.fresh:
        safe_rmtree(stage_parent)
        safe_rmtree(clean_parent)

    checks: Dict[str, Tuple[bool, Any]] = {}
    checks["zip_exists"] = (input_zip.exists(), str(input_zip))
    checks["zip_is_file"] = (input_zip.is_file(), str(input_zip) if input_zip.exists() else "missing")

    copied: List[str] = []
    scrub_rows: List[Dict[str, Any]] = []
    syntax_rows: List[Dict[str, Any]] = []
    manifest_hash_rows: List[Dict[str, Any]] = []
    cmd_results: List[CommandResult] = []
    manifest: Dict[str, Any] = {}

    if input_zip.exists() and input_zip.is_file():
        raw_extract = stage / "_raw_extract"
        extract_zip(input_zip, raw_extract)
        bundle_root = locate_bundle_root(raw_extract)
        stage.mkdir(parents=True, exist_ok=True)
        # Copy actual bundle contents to stage root.
        copytree_contents(bundle_root, stage)
        if (stage / "_raw_extract").exists():
            shutil.rmtree(stage / "_raw_extract")

        copied = ensure_core_artifacts(stage, outputs, source_root)
        checks["core_artifacts_present_after_repair"] = (all((stage / f).exists() for f in CORE_RUNTIME_FILES), {f: (stage / f).exists() for f in CORE_RUNTIME_FILES})
        checks["core_data_present_after_repair"] = (any((stage / f).exists() for f in CORE_DATA_FILES), {f: (stage / f).exists() for f in CORE_DATA_FILES})

        if args.no_scrub:
            scrub_rows = []
        else:
            scrub_rows, scrub_ok, scrub_remaining = scrub_runtime_core(stage)
            checks["runtime_core_abs_paths_scrubbed"] = (scrub_ok, scrub_remaining)

        syntax_ok, syntax_rows = syntax_check_python_files(stage)
        checks["all_staged_python_syntax_ok"] = (syntax_ok, f"{sum(1 for r in syntax_rows if r['syntax_ok'])}/{len(syntax_rows)}")

        runtime_path_ok, runtime_leaks = scan_runtime_core_paths(stage)
        checks["no_absolute_bbit_paths_in_runtime_core_python"] = (runtime_path_ok, runtime_leaks)

        manifest = build_manifest(stage, root, input_zip, copied, scrub_rows)
        # Write manifest into stage, then rebuild including the manifest itself.
        write_json(stage / "phase26ey_lite_final_portable_manifest.json", manifest)
        manifest = build_manifest(stage, root, input_zip, copied, scrub_rows)
        write_json(stage / "phase26ey_lite_final_portable_manifest.json", manifest)

        manifest_hash_ok, manifest_hash_rows = verify_manifest_hashes(stage, manifest)
        checks["manifest_hashes_match_after_reseal"] = (manifest_hash_ok, f"{sum(1 for r in manifest_hash_rows if r['match'])}/{len(manifest_hash_rows)}")
        checks["fingerprint_found_in_stage_core"] = (any(FINGERPRINT in p.read_text(encoding="utf-8", errors="ignore") for p in stage.rglob("*.py")), FINGERPRINT)

        if args.write_final_zip:
            write_zip_from_dir(stage, final_zip)
        checks["final_zip_written_if_requested"] = ((not args.write_final_zip) or final_zip.exists(), str(final_zip))

        if final_zip.exists():
            extract_zip(final_zip, cleanroom)
            cr_root = locate_bundle_root(cleanroom)
            checks["cleanroom_extract_created"] = (cr_root.exists(), str(cr_root))
            cr_syntax_ok, cr_syntax_rows = syntax_check_python_files(cr_root)
            checks["all_cleanroom_python_syntax_ok"] = (cr_syntax_ok, f"{sum(1 for r in cr_syntax_rows if r['syntax_ok'])}/{len(cr_syntax_rows)}")
            cr_path_ok, cr_path_leaks = scan_runtime_core_paths(cr_root)
            checks["no_absolute_bbit_paths_in_cleanroom_runtime_core_python"] = (cr_path_ok, cr_path_leaks)
            cmd_results = run_replay(cr_root)
            checks["all_cleanroom_replay_commands_exit_zero"] = (all(c.returncode == 0 for c in cmd_results), {c.name: c.returncode for c in cmd_results})
            checks["all_cleanroom_replay_commands_pass_detected"] = (all(c.pass_detected for c in cmd_results), {c.name: c.pass_detected for c in cmd_results})
            checks["no_absolute_bbit_path_leak_in_command_output"] = (not any(c.abs_path_leak for c in cmd_results), {c.name: c.abs_path_leak for c in cmd_results})
        else:
            checks["cleanroom_extract_created"] = (False, "final zip not written")

    final_pass = all(v[0] for v in checks.values())

    check_rows = [{"check": k, "pass": bool(v[0]), "detail": v[1]} for k, v in checks.items()]
    summary = {
        "phase": PHASE,
        "FINAL_PORTABLE_RESEAL_LOCK_PASS": final_pass,
        "root": str(root),
        "outputs": str(outputs),
        "input_zip": str(input_zip),
        "stage": str(stage),
        "cleanroom": str(cleanroom),
        "final_zip": str(final_zip) if final_zip.exists() else None,
        "route_fingerprint": FINGERPRINT,
        "copied_missing_artifacts": copied,
        "checks": {k: {"pass": bool(v[0]), "detail": v[1]} for k, v in checks.items()},
        "command_results": [asdict(c) for c in cmd_results],
    }

    write_json(outputs / "phase26ey_lite_summary.json", summary)
    write_json(outputs / "phase26ey_lite_final_portable_manifest.json", manifest if manifest else {})
    write_csv(outputs / "phase26ey_lite_reseal_checks.csv", check_rows, ["check", "pass", "detail"])
    write_csv(outputs / "phase26ey_lite_manifest_hash_audit.csv", manifest_hash_rows)
    write_csv(outputs / "phase26ey_lite_runtime_scrub_audit.csv", scrub_rows)
    write_csv(outputs / "phase26ey_lite_syntax_audit.csv", syntax_rows)
    write_csv(outputs / "phase26ey_lite_command_replay.csv", [asdict(c) for c in cmd_results])
    write_report(outputs / "phase26ey_lite_final_portable_reseal_report.md", summary, cmd_results, check_rows)
    make_plot(check_rows, outputs / "phase26ey_lite_final_portable_reseal_checks.png")

    print(f"[{PHASE}] FINAL_PORTABLE_RESEAL_LOCK_PASS={final_pass}")
    print(f"[{PHASE}] checks:")
    for k, (ok, detail) in checks.items():
        print(f"  - {k}: {ok} :: {detail}")
    for c in cmd_results:
        print(f"[{PHASE}] replay {c.name}: rc={c.returncode} pass={c.pass_detected} leak={c.abs_path_leak} elapsed={c.elapsed_s}s")
    if final_zip.exists():
        print(f"[{PHASE}] wrote final zip: {final_zip}")
    print(f"[{PHASE}] wrote report: {outputs / 'phase26ey_lite_final_portable_reseal_report.md'}")
    print(f"[{PHASE}] wrote summary: {outputs / 'phase26ey_lite_summary.json'}")
    print(f"[{PHASE}] wrote outputs to: {outputs}")
    return 0 if final_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
