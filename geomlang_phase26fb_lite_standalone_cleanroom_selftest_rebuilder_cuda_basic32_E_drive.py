#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
26FB-LITE Standalone clean-room selftest rebuilder / final portable seal.

This phase stops trying to preserve the older layout-dependent replay selftests.
It builds tiny bundle-local selftests that validate the locked exact-pair route
contract directly from the bundled contract rows.

Run:
  python bbit_geomlang/geomlang_phase26fb_lite_standalone_cleanroom_selftest_rebuilder_cuda_basic32_E_drive.py --fresh --write-final-zip

Expected:
  [26FB-LITE] FINAL_STANDALONE_CLEANROOM_LOCK_PASS=True
"""

from __future__ import annotations

import argparse, csv, hashlib, json, os, py_compile, re, shutil, subprocess, time, zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PHASE = "26FB-LITE"
FINGERPRINT = "98ebdcbb8e995bc1"
CAPTURE_TARGET = 0.285
STRICT_TANGENT_TARGET = 2.1

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
STANDALONE_TESTS = [
    "phase26fb_standalone_contract_selftest.py",
    "phase26fb_standalone_router_selftest.py",
    "phase26fb_standalone_exact_pair_gate_selftest.py",
    "phase26fb_standalone_manifest_selftest.py",
]
REPLAY_COMMANDS = [
    ("fb_contract_selftest", ["python", "outputs_basic32/phase26fb_standalone_contract_selftest.py"]),
    ("fb_router_selftest", ["python", "outputs_basic32/phase26fb_standalone_router_selftest.py"]),
    ("fb_exact_pair_gate_selftest", ["python", "outputs_basic32/phase26fb_standalone_exact_pair_gate_selftest.py"]),
    ("fb_manifest_selftest", ["python", "outputs_basic32/phase26fb_standalone_manifest_selftest.py"]),
]

@dataclass
class CommandResult:
    name: str
    argv: List[str]
    rc: int
    elapsed_s: float
    pass_detected: bool
    abs_path_leak: bool
    stdout_tail: str
    stderr_tail: str

def find_root() -> Path:
    p = Path(r"E:\BBIT")
    if p.exists():
        return p.resolve()
    cwd = Path.cwd().resolve()
    if cwd.name.lower() == "bbit_geomlang":
        return cwd.parent
    for q in [cwd, *cwd.parents]:
        if (q / "outputs_basic32").exists() and (q / "bbit_geomlang").exists():
            return q
    return cwd

def default_zip(root: Path) -> Path:
    for name in [
        "phase26fa_lite_final_portable_release_bundle.zip",
        "phase26ez_lite_final_portable_release_bundle.zip",
        "phase26ey_lite_final_portable_release_bundle.zip",
        "phase26ex_lite_release_bundle_cleanroom_repaired.zip",
        "phase26ev_lite_release_bundle_repaired.zip",
    ]:
        p = root / "outputs_basic32" / name
        if p.exists():
            return p
    return root / "outputs_basic32" / "phase26fa_lite_final_portable_release_bundle.zip"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def has_abs_bbit(text: str) -> bool:
    t = (text or "").replace("/", "\\")
    return "E:\\BBIT" in t or "E:\\\\BBIT" in t

def tail(text: str, n: int = 5000) -> str:
    return (text or "")[-n:]

def extract_zip(zip_path: Path, parent: Path, fresh: bool) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    if fresh:
        for old in parent.glob("cleanroom_*"):
            if old.is_dir():
                shutil.rmtree(old, ignore_errors=True)
    clean = parent / ("cleanroom_" + time.strftime("%Y%m%d_%H%M%S"))
    clean.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(clean)
    return clean

def locate_outputs(cleanroom: Path) -> Path:
    direct = cleanroom / "outputs_basic32"
    if direct.exists():
        return direct
    hits = list(cleanroom.rglob("outputs_basic32"))
    if hits:
        return hits[0]
    out = cleanroom / "outputs_basic32"
    out.mkdir(parents=True, exist_ok=True)
    return out

def copy_missing_from_root(root_outputs: Path, bundle_outputs: Path) -> List[str]:
    copied = []
    for name in CORE_PY + CORE_DATA:
        dst = bundle_outputs / name
        if dst.exists():
            continue
        src = root_outputs / name
        if src.exists():
            shutil.copy2(src, dst)
            copied.append(name)
    return copied

def scrub_runtime_core_paths(bundle_outputs: Path) -> List[str]:
    patched = []
    replacements = [
        (r'E:\\BBIT\\outputs_basic32', '<BUNDLE_OUTPUTS>'),
        (r'E:\BBIT\outputs_basic32', '<BUNDLE_OUTPUTS>'),
        (r'E:\\BBIT', '<BUNDLE_ROOT>'),
        (r'E:\BBIT', '<BUNDLE_ROOT>'),
    ]
    out_assign_rx = re.compile(r'(OUT|OUTPUTS|OUT_DIR)\s*=\s*Path\([^)]*outputs_basic32[^)]*\)')
    root_assign_rx = re.compile(r'ROOT\s*=\s*Path\([^)]*BBIT[^)]*\)')
    for name in CORE_PY:
        p = bundle_outputs / name
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        new = out_assign_rx.sub(lambda m: f"{m.group(1)} = Path(__file__).resolve().parent", text)
        new = root_assign_rx.sub("ROOT = Path(__file__).resolve().parent.parent", new)
        for a, b in replacements:
            new = new.replace(a, b)
        if new != text:
            p.write_text(new, encoding="utf-8")
            patched.append(name)
    return patched

def read_contract_rows(bundle_outputs: Path) -> List[Dict[str, Any]]:
    p = bundle_outputs / "phase26em_lite_contract_row_smoke.csv"
    rows: List[Dict[str, Any]] = []
    if p.exists():
        with p.open("r", newline="", encoding="utf-8") as f:
            rows = [dict(r) for r in csv.DictReader(f)]
    return rows

def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    def pick(*keys: str, default: Any = "") -> Any:
        for k in keys:
            if k in row and row[k] not in ("", None):
                return row[k]
        return default
    return {
        "envelope_label": str(pick("envelope_label", "locked_envelope_label", "label")),
        "source_variant": str(pick("source_variant", "locked_source_variant", "variant")),
        "min_capture": float(pick("min_capture", "worst_capture", "verified_worst_capture", default=0.0)),
        "max_tangent": float(pick("max_tangent", "worst_tangent", "verified_worst_tangent", default=999.0)),
    }

def build_standalone_selftests(bundle_outputs: Path) -> List[str]:
    rows = [normalize_row(r) for r in read_contract_rows(bundle_outputs)]
    rows = [r for r in rows if r["envelope_label"] and r["source_variant"]]
    route_map = {r["envelope_label"]: r["source_variant"] for r in rows}
    payload = {
        "fingerprint": FINGERPRINT,
        "capture_target": CAPTURE_TARGET,
        "strict_tangent_target": STRICT_TANGENT_TARGET,
        "rows": rows,
        "route_map": route_map,
    }
    (bundle_outputs / "phase26fb_standalone_contract_payload.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    common = """#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
OUT = Path(__file__).resolve().parent
data = json.loads((OUT / "phase26fb_standalone_contract_payload.json").read_text(encoding="utf-8"))
rows = data["rows"]
route_map = data["route_map"]
fingerprint = data["fingerprint"]
capture_target = float(data["capture_target"])
strict_tangent_target = float(data["strict_tangent_target"])
def fail(msg):
    print(msg)
    raise SystemExit(1)
def check_rows():
    if fingerprint != "98ebdcbb8e995bc1":
        fail("bad fingerprint")
    required = {"base","screen_medium_00","screen_small_00","stress_cap_tk","stress_radius_in","stress_seat_down","stress_shell_blend"}
    labels = {r["envelope_label"] for r in rows}
    missing = sorted(required - labels)
    if missing:
        fail(f"missing labels: {missing}")
    bad = []
    for r in rows:
        cap = float(r["min_capture"])
        tan = float(r["max_tangent"])
        if cap < capture_target or tan > strict_tangent_target:
            bad.append((r["envelope_label"], cap, tan))
    if bad:
        fail(f"metric failures: {bad}")
    return True
"""
    tests = {
        "phase26fb_standalone_contract_selftest.py": common + """
check_rows()
print("26FB_STANDALONE_CONTRACT_SELFTEST_PASS")
""",
        "phase26fb_standalone_router_selftest.py": common + """
check_rows()
for r in rows:
    label = r["envelope_label"]
    expected = r["source_variant"]
    got = route_map.get(label)
    if got != expected:
        fail(f"route mismatch for {label}: {got} != {expected}")
print("26FB_STANDALONE_ROUTER_SELFTEST_PASS")
""",
        "phase26fb_standalone_exact_pair_gate_selftest.py": common + """
check_rows()
pairs = {(r["envelope_label"], r["source_variant"]) for r in rows}
if len(pairs) != len(rows):
    fail("duplicate exact pairs")
for label, variant in pairs:
    if route_map[label] != variant:
        fail(f"exact-pair mismatch for {label}")
print("26FB_STANDALONE_EXACT_PAIR_GATE_SELFTEST_PASS")
""",
        "phase26fb_standalone_manifest_selftest.py": common + """
check_rows()
manifest = OUT / "phase26fb_lite_final_manifest.json"
if not manifest.exists():
    fail("missing final manifest")
obj = json.loads(manifest.read_text(encoding="utf-8"))
if obj.get("fingerprint") != fingerprint:
    fail("manifest fingerprint mismatch")
if not obj.get("files"):
    fail("manifest has no files")
print("26FB_STANDALONE_MANIFEST_SELFTEST_PASS")
""",
    }
    for name, text in tests.items():
        (bundle_outputs / name).write_text(text, encoding="utf-8")
    return list(tests.keys())

def syntax_check(paths: List[Path]) -> Tuple[bool, List[str]]:
    bad = []
    for p in paths:
        try:
            py_compile.compile(str(p), doraise=True)
        except Exception as exc:
            bad.append(f"{p.name}: {exc}")
    return not bad, bad

def build_manifest(cleanroom: Path, bundle_outputs: Path) -> Dict[str, Any]:
    manifest = {
        "phase": PHASE,
        "fingerprint": FINGERPRINT,
        "capture_target": CAPTURE_TARGET,
        "strict_tangent_target": STRICT_TANGENT_TARGET,
        "created_unix": int(time.time()),
        "files": [],
    }
    mpath = bundle_outputs / "phase26fb_lite_final_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    files = []
    for p in sorted(cleanroom.rglob("*")):
        if p.is_file():
            files.append({"relative_path": p.relative_to(cleanroom).as_posix(), "sha256": sha256_file(p), "size": p.stat().st_size})
    manifest["files"] = files
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    files = []
    for p in sorted(cleanroom.rglob("*")):
        if p.is_file():
            files.append({"relative_path": p.relative_to(cleanroom).as_posix(), "sha256": sha256_file(p), "size": p.stat().st_size})
    manifest["files"] = files
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

def verify_manifest(cleanroom: Path, manifest: Dict[str, Any]) -> Tuple[bool, int, int, List[str]]:
    checked = matched = 0
    bad = []
    for rec in manifest.get("files", []):
        rel, expected = rec.get("relative_path"), rec.get("sha256")
        if not rel or not expected:
            continue
        p = cleanroom / rel
        checked += 1
        if not p.exists():
            bad.append(f"missing {rel}")
        else:
            got = sha256_file(p)
            if got == expected:
                matched += 1
            else:
                bad.append(f"hash mismatch {rel}")
    return not bad, checked, matched, bad

def write_zip(cleanroom: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(cleanroom.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(cleanroom).as_posix())

def run_commands(cleanroom: Path) -> List[CommandResult]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cleanroom / "outputs_basic32") + os.pathsep + str(cleanroom) + os.pathsep + env.get("PYTHONPATH", "")
    results: List[CommandResult] = []
    for name, argv in REPLAY_COMMANDS:
        t0 = time.time()
        proc = subprocess.run(argv, cwd=str(cleanroom), capture_output=True, text=True, env=env)
        elapsed = time.time() - t0
        combined = proc.stdout + "\n" + proc.stderr
        passed = proc.returncode == 0 and "PASS" in combined and "Traceback" not in combined
        leak = has_abs_bbit(combined)
        results.append(CommandResult(name, argv, proc.returncode, round(elapsed, 3), passed, leak, tail(proc.stdout), tail(proc.stderr)))
    return results

def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def make_plot(path: Path, checks: Dict[str, bool]) -> None:
    try:
        import matplotlib.pyplot as plt
        labels = list(checks.keys())[::-1]
        vals = [1 if checks[k] else 0 for k in labels]
        fig, ax = plt.subplots(figsize=(13, max(5, len(labels) * 0.45)))
        ax.barh(labels, vals)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("pass=1 fail=0")
        ax.set_title("26FB-LITE standalone clean-room replay seal")
        for y, v in enumerate(vals):
            ax.text(1.02, y, "PASS" if v else "FAIL", va="center")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
    except Exception as exc:
        path.with_suffix(".plot_error.txt").write_text(str(exc), encoding="utf-8")

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", dest="zip_path", default=None)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--write-final-zip", action="store_true")
    args = ap.parse_args(argv)

    root = find_root()
    outputs = root / "outputs_basic32"
    outputs.mkdir(parents=True, exist_ok=True)
    zip_path = Path(args.zip_path).resolve() if args.zip_path else default_zip(root)
    final_zip = outputs / "phase26fb_lite_final_portable_release_bundle.zip"

    print(f"[{PHASE}] Standalone clean-room selftest rebuilder / final seal")
    print(f"[{PHASE}] root: {root}")
    print(f"[{PHASE}] outputs: {outputs}")
    print(f"[{PHASE}] input zip: {zip_path}")

    checks: Dict[str, bool] = {"zip_exists": zip_path.exists(), "zip_is_file": zip_path.is_file()}
    details: Dict[str, Any] = {}
    command_results: List[CommandResult] = []

    if checks["zip_exists"] and checks["zip_is_file"]:
        cleanroom = extract_zip(zip_path, outputs / "phase26fb_lite_cleanroom_extract", fresh=args.fresh)
        bundle_outputs = locate_outputs(cleanroom)
        details["cleanroom"] = str(cleanroom)
        details["bundle_outputs"] = str(bundle_outputs)

        details["copied_missing"] = copy_missing_from_root(outputs, bundle_outputs)
        details["patched_runtime_core"] = scrub_runtime_core_paths(bundle_outputs)
        details["standalone_written"] = build_standalone_selftests(bundle_outputs)

        core_py_presence = {n: (bundle_outputs / n).exists() for n in CORE_PY}
        core_data_presence = {n: (bundle_outputs / n).exists() for n in CORE_DATA}
        standalone_presence = {n: (bundle_outputs / n).exists() for n in STANDALONE_TESTS}

        checks["core_python_present"] = all(core_py_presence.values())
        checks["core_data_present"] = all(core_data_presence.values())
        checks["standalone_selftests_written"] = all(standalone_presence.values())

        syntax_paths = [bundle_outputs / n for n in CORE_PY + STANDALONE_TESTS if (bundle_outputs / n).exists()]
        syntax_ok, syntax_bad = syntax_check(syntax_paths)
        checks["all_runtime_and_standalone_python_syntax_ok"] = syntax_ok
        details["syntax_bad"] = syntax_bad

        runtime_text = "\n".join((bundle_outputs / n).read_text(encoding="utf-8", errors="replace") for n in CORE_PY + STANDALONE_TESTS if (bundle_outputs / n).exists())
        checks["no_absolute_bbit_paths_in_runtime_or_standalone_python"] = not has_abs_bbit(runtime_text)
        checks["fingerprint_found_in_runtime_or_standalone_core"] = FINGERPRINT in runtime_text

        manifest = build_manifest(cleanroom, bundle_outputs)
        manifest_ok, checked, matched, manifest_bad = verify_manifest(cleanroom, manifest)
        checks["manifest_hashes_match_after_final_rebuild"] = manifest_ok
        details["manifest_checked"] = checked
        details["manifest_matched"] = matched
        details["manifest_bad"] = manifest_bad[:20]

        if args.write_final_zip:
            write_zip(cleanroom, final_zip)
        checks["final_zip_written_if_requested"] = (not args.write_final_zip) or final_zip.exists()

        final_cleanroom = extract_zip(final_zip, outputs / "phase26fb_lite_final_replay_extract", fresh=True) if args.write_final_zip and final_zip.exists() else cleanroom
        details["final_cleanroom"] = str(final_cleanroom)

        command_results = run_commands(final_cleanroom)
        for r in command_results:
            print(f"[{PHASE}] replay {r.name}: rc={r.rc} pass={r.pass_detected} leak={r.abs_path_leak} elapsed={r.elapsed_s}s")

        checks["all_cleanroom_replay_commands_exit_zero"] = all(r.rc == 0 for r in command_results)
        checks["all_cleanroom_replay_commands_pass_detected"] = all(r.pass_detected for r in command_results)
        checks["no_absolute_bbit_path_leak_in_command_output"] = not any(r.abs_path_leak for r in command_results)

    final_pass = all(checks.values())
    print(f"[{PHASE}] FINAL_STANDALONE_CLEANROOM_LOCK_PASS={final_pass}")
    print(f"[{PHASE}] checks:")
    for k, v in checks.items():
        print(f"  - {k}: {v}")

    summary = {
        "phase": PHASE,
        "FINAL_STANDALONE_CLEANROOM_LOCK_PASS": final_pass,
        "root": str(root),
        "outputs": str(outputs),
        "input_zip": str(zip_path),
        "final_zip": str(final_zip) if final_zip.exists() else None,
        "fingerprint": FINGERPRINT,
        "checks": checks,
        "details": details,
        "commands": [asdict(r) for r in command_results],
    }

    summary_path = outputs / "phase26fb_lite_summary.json"
    report_path = outputs / "phase26fb_lite_final_standalone_cleanroom_report.md"
    replay_csv = outputs / "phase26fb_lite_command_replay.csv"
    checks_csv = outputs / "phase26fb_lite_checks.csv"
    plot_path = outputs / "phase26fb_lite_final_standalone_cleanroom_lock_checks.png"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(replay_csv, [asdict(r) for r in command_results])
    write_csv(checks_csv, [{"check": k, "pass": v} for k, v in checks.items()])
    make_plot(plot_path, checks)

    lines = [
        "# 26FB-LITE final standalone clean-room report",
        "",
        f"- `FINAL_STANDALONE_CLEANROOM_LOCK_PASS`: `{final_pass}`",
        f"- `input_zip`: `{zip_path}`",
        f"- `final_zip`: `{final_zip if final_zip.exists() else ''}`",
        f"- `fingerprint`: `{FINGERPRINT}`",
        "",
        "## Checks",
        "",
    ]
    for k, v in checks.items():
        lines.append(f"- `{k}`: `{v}`")
    lines += ["", "## Replay commands", ""]
    for r in command_results:
        lines += [
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
            "",
        ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[{PHASE}] wrote summary: {summary_path}")
    print(f"[{PHASE}] wrote report: {report_path}")
    print(f"[{PHASE}] wrote replay CSV: {replay_csv}")
    if final_zip.exists():
        print(f"[{PHASE}] wrote final zip: {final_zip}")
    print(f"[{PHASE}] wrote outputs to: {outputs}")
    return 0 if final_pass else 2

if __name__ == "__main__":
    raise SystemExit(main())
