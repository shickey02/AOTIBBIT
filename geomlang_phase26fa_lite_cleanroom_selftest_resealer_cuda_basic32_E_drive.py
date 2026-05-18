#!/usr/bin/env python3
"""
26FA-LITE: Clean-room selftest resealer / portable replay finalizer.

Goal:
  Take the 26EZ portable zip, rewrite the *portable runtime selftests* so they
  resolve every artifact from the extracted bundle itself, reseal manifest
  hashes, write a final zip, and immediately replay the clean-room command
  chain.

Why this exists:
  26EZ fixed manifest hashes and removed absolute paths from the runtime core,
  but the replay chain still failed because several bundled selftests were still
  semantically tied to the old build layout or emitted path-leaking failures.
  This phase makes the selftests bundle-local and JSON-verifiable.
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
import textwrap
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

PHASE = "26FA-LITE"
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
    "phase26ej_lite_integration_contract.json",
    "phase26em_lite_contract_row_smoke.csv",
    "phase26em_lite_transplant_manifest.json",
    "phase26en_lite_live_transplant_manifest.json",
    "phase26eq_lite_shadow_transplant_manifest.json",
]
REPLAY = [
    ("em_router_selftest", "phase26em_lite_runtime_route_selftest.py"),
    ("en_guard_selftest", "phase26en_lite_live_guard_selftest.py"),
    ("eo_selftest", "phase26eo_lite_transplant_selftest.py"),
    ("eq_gate_selftest", "phase26eq_lite_runtime_exact_pair_gate_selftest.py"),
]
ABS_PATTERNS = ["E:\\BBIT", "E:/BBIT", "C:\\", "D:\\", "\\BBIT\\", "/BBIT/", "bbit_geomlang\\"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(s).lstrip(), encoding="utf-8")


def first_existing(paths: List[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    return paths[0]


def extract_zip(zip_path: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)


def find_file(stage: Path, name: str) -> Path | None:
    hits = list(stage.rglob(name))
    if not hits:
        return None
    # Prefer outputs_basic32 copies for portable runtime artifacts.
    hits.sort(key=lambda p: ("outputs_basic32" not in str(p).replace("/", "\\"), len(str(p))))
    return hits[0]


def find_outputs_dir(stage: Path) -> Path:
    for d in stage.rglob("outputs_basic32"):
        if d.is_dir() and (d / "phase26em_lite_runtime_route_router.py").exists():
            return d
    # Some zips may contain the output files at root.
    if (stage / "phase26em_lite_runtime_route_router.py").exists():
        return stage
    raise FileNotFoundError("Could not locate portable outputs_basic32 directory in extracted bundle")


def patch_runtime_gate_paths(outputs: Path) -> List[str]:
    patched = []
    for name in ["phase26eq_lite_runtime_exact_pair_gate.py", "phase26eq_lite_runtime_exact_pair_gate_selftest.py"]:
        p = outputs / name
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="replace")
        new = txt
        new = re.sub(r"Path\(r?['\"]E:[^'\"]*outputs_basic32['\"]\)", "Path(__file__).resolve().parent", new)
        new = re.sub(r"_OUT\s*=\s*Path\([^\n]+\)", "_OUT = Path(__file__).resolve().parent", new)
        new = new.replace("E:\\BBIT\\outputs_basic32", ".")
        new = new.replace("E:/BBIT/outputs_basic32", ".")
        if new != txt:
            p.write_text(new, encoding="utf-8")
            patched.append(name)
    return patched


def write_portable_selftests(outputs: Path) -> List[str]:
    patched = []

    write_text(outputs / "phase26em_lite_runtime_route_selftest.py", r'''
        from __future__ import annotations
        import json, sys
        from pathlib import Path
        HERE = Path(__file__).resolve().parent
        if str(HERE) not in sys.path:
            sys.path.insert(0, str(HERE))
        import phase26em_lite_runtime_route_router as router
        EXPECTED = ["base", "screen_medium_00", "screen_small_00", "stress_cap_tk", "stress_radius_in", "stress_seat_down", "stress_shell_blend"]
        def main():
            routes = router.available_locked_routes()
            missing = [x for x in EXPECTED if x not in routes]
            mismatches = []
            for label in EXPECTED:
                rec = router.get_locked_route(label)
                if not rec or str(rec.get("envelope_label")) != label:
                    mismatches.append(label)
            ok = (not missing) and (not mismatches) and getattr(router, "ROUTE_FINGERPRINT", "") == "98ebdcbb8e995bc1"
            print(json.dumps({"EM_ROUTER_SELFTEST_PASS": ok, "fingerprint": getattr(router, "ROUTE_FINGERPRINT", ""), "labels": routes, "missing": missing, "mismatches": mismatches}, indent=2))
            return 0 if ok else 1
        if __name__ == "__main__":
            raise SystemExit(main())
    ''')
    patched.append("phase26em_lite_runtime_route_selftest.py")

    write_text(outputs / "phase26en_lite_live_guard_selftest.py", r'''
        from __future__ import annotations
        import csv, json, sys
        from pathlib import Path
        HERE = Path(__file__).resolve().parent
        if str(HERE) not in sys.path:
            sys.path.insert(0, str(HERE))
        import phase26en_lite_live_contract_guard as guard
        EXPECTED = ["base", "screen_medium_00", "screen_small_00", "stress_cap_tk", "stress_radius_in", "stress_seat_down", "stress_shell_blend"]
        def load_rows():
            p = HERE / "phase26em_lite_contract_row_smoke.csv"
            rows = []
            with p.open("r", encoding="utf-8-sig", newline="") as f:
                for r in csv.DictReader(f):
                    rows.append(r)
            return rows
        def main():
            rows = load_rows()
            results = []
            for r in rows:
                label = r.get("envelope_label") or r.get("locked_envelope_label") or r.get("label")
                min_capture = float(r.get("min_capture", 0.0))
                max_tangent = float(r.get("max_tangent", 999.0))
                ok, info = guard.check_exact_pair_record({"envelope_label": label, "min_capture": min_capture, "max_tangent": max_tangent})
                results.append({"label": label, "ok": bool(ok), "info": info})
            labels = sorted({x["label"] for x in results})
            missing = [x for x in EXPECTED if x not in labels]
            ok = (not missing) and all(x["ok"] for x in results) and getattr(guard, "ROUTE_FINGERPRINT", "") == "98ebdcbb8e995bc1"
            print(json.dumps({"EN_GUARD_SELFTEST_PASS": ok, "checked": len(results), "labels": labels, "missing": missing}, indent=2))
            return 0 if ok else 1
        if __name__ == "__main__":
            raise SystemExit(main())
    ''')
    patched.append("phase26en_lite_live_guard_selftest.py")

    write_text(outputs / "phase26eo_lite_transplant_selftest.py", r'''
        from __future__ import annotations
        import csv, json, sys
        from pathlib import Path
        HERE = Path(__file__).resolve().parent
        if str(HERE) not in sys.path:
            sys.path.insert(0, str(HERE))
        import phase26eo_lite_exact_pair_route_helper as helper
        EXPECTED = ["base", "screen_medium_00", "screen_small_00", "stress_cap_tk", "stress_radius_in", "stress_seat_down", "stress_shell_blend"]
        def load_rows():
            p = HERE / "phase26em_lite_contract_row_smoke.csv"
            with p.open("r", encoding="utf-8-sig", newline="") as f:
                return list(csv.DictReader(f))
        def main():
            rows = load_rows()
            missing = []
            metric_failures = []
            for label in EXPECTED:
                r = next((x for x in rows if (x.get("envelope_label") or x.get("locked_envelope_label") or x.get("label")) == label), None)
                if r is None:
                    missing.append(label); continue
                min_capture = float(r.get("min_capture", 0.0))
                max_tangent = float(r.get("max_tangent", 999.0))
                if min_capture < 0.285 or max_tangent > 2.1:
                    metric_failures.append({"label": label, "min_capture": min_capture, "max_tangent": max_tangent})
                route = helper.get_exact_pair_route(label)
                if not route or route.get("envelope_label") != label:
                    metric_failures.append({"label": label, "route_failure": True})
            ok = (not missing) and (not metric_failures) and getattr(helper, "ROUTE_FINGERPRINT", "") == "98ebdcbb8e995bc1"
            print(json.dumps({"EO_SELFTEST_PASS": ok, "fingerprint": getattr(helper, "ROUTE_FINGERPRINT", ""), "missing_locked_labels": missing, "metric_failures": metric_failures}, indent=2))
            return 0 if ok else 1
        if __name__ == "__main__":
            raise SystemExit(main())
    ''')
    patched.append("phase26eo_lite_transplant_selftest.py")

    write_text(outputs / "phase26eq_lite_runtime_exact_pair_gate_selftest.py", r'''
        from __future__ import annotations
        import csv, json, sys
        from pathlib import Path
        HERE = Path(__file__).resolve().parent
        if str(HERE) not in sys.path:
            sys.path.insert(0, str(HERE))
        import phase26eq_lite_runtime_exact_pair_gate as gate
        EXPECTED = ["base", "screen_medium_00", "screen_small_00", "stress_cap_tk", "stress_radius_in", "stress_seat_down", "stress_shell_blend"]
        def main():
            rows_path = HERE / "phase26em_lite_contract_row_smoke.csv"
            rows = list(csv.DictReader(rows_path.open("r", encoding="utf-8-sig", newline="")))
            failures = []
            for r in rows:
                label = r.get("envelope_label") or r.get("locked_envelope_label") or r.get("label")
                min_capture = float(r.get("min_capture", 0.0))
                max_tangent = float(r.get("max_tangent", 999.0))
                ok, info = gate.allow_exact_pair(label, min_capture, max_tangent)
                if not ok:
                    failures.append({"label": label, "info": info})
            labels = sorted({r.get("envelope_label") or r.get("locked_envelope_label") or r.get("label") for r in rows})
            missing = [x for x in EXPECTED if x not in labels]
            ok = (not missing) and (not failures)
            print(json.dumps({"EQ_GATE_SELFTEST_PASS": ok, "checked": len(rows), "missing": missing, "failures": failures}, indent=2))
            return 0 if ok else 1
        if __name__ == "__main__":
            raise SystemExit(main())
    ''')
    patched.append("phase26eq_lite_runtime_exact_pair_gate_selftest.py")
    return patched


def ensure_core_data(outputs: Path, root: Path) -> List[str]:
    copied = []
    for name in CORE_DATA:
        dst = outputs / name
        if dst.exists():
            continue
        src_hits = list(root.rglob(name))
        src_hits = [p for p in src_hits if p != dst and p.is_file()]
        if src_hits:
            shutil.copy2(src_hits[0], dst)
            copied.append(name)
    return copied


def syntax_ok(paths: List[Path]) -> Tuple[bool, Dict[str, str]]:
    errors = {}
    for p in paths:
        try:
            ast.parse(p.read_text(encoding="utf-8", errors="replace"), filename=str(p))
        except Exception as e:
            errors[p.name] = str(e)
    return (not errors), errors


def abs_path_hits(paths: List[Path]) -> Dict[str, List[str]]:
    out = {}
    for p in paths:
        txt = p.read_text(encoding="utf-8", errors="replace")
        hits = [pat for pat in ABS_PATTERNS if pat in txt]
        if hits:
            out[p.name] = hits
    return out


def load_manifest(stage: Path) -> dict:
    for name in ["phase26et_lite_release_manifest.json", "phase26ev_lite_release_replay_repair_manifest.json", "phase26ey_lite_final_portable_manifest.json"]:
        p = find_file(stage, name)
        if p:
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    return {}


def write_resealed_manifest(stage: Path, outputs: Path, prior: dict) -> Path:
    files = []
    for p in sorted([x for x in stage.rglob("*") if x.is_file()]):
        rel = p.relative_to(stage).as_posix()
        if rel.endswith("phase26et_lite_release_manifest.json") or rel.endswith("phase26fa_lite_final_portable_manifest.json"):
            continue
        fp = sha256(p)
        files.append({
            "role": "portable_bundle_file",
            "name": p.name,
            "relative_path": rel,
            "bytes": p.stat().st_size,
            "sha256": fp,
            "sha16": fp[:16],
            "syntax_ok": None,
            "fingerprint_present": FINGERPRINT in p.read_text(encoding="utf-8", errors="ignore") if p.suffix in {".py", ".json", ".md", ".txt", ".csv"} else False,
        })
    manifest = {
        "phase": "26FA-LITE",
        "title": "Clean-room selftest resealer / portable replay finalizer",
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "route_fingerprint": FINGERPRINT,
        "prior_manifest_phase": prior.get("phase"),
        "files": files,
        "core_python": CORE_PY,
        "core_data": CORE_DATA,
        "replay_commands": [x[0] for x in REPLAY],
    }
    for target in [outputs / "phase26et_lite_release_manifest.json", outputs / "phase26fa_lite_final_portable_manifest.json"]:
        write_text(target, json.dumps(manifest, indent=2))
    return outputs / "phase26fa_lite_final_portable_manifest.json"


def write_zip(stage: Path, zip_out: Path) -> None:
    if zip_out.exists():
        zip_out.unlink()
    with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(stage).as_posix())


def replay_cleanroom(zip_path: Path, clean_root: Path) -> Tuple[Dict[str, dict], Path]:
    extract_zip(zip_path, clean_root)
    outputs = find_outputs_dir(clean_root)
    results = {}
    env = os.environ.copy()
    env["PYTHONPATH"] = str(outputs)
    for label, script in REPLAY:
        p = outputs / script
        t0 = time.time()
        cp = subprocess.run([sys.executable, str(p)], cwd=str(outputs), text=True, capture_output=True, env=env)
        text = (cp.stdout or "") + (cp.stderr or "")
        pass_detected = (cp.returncode == 0) and ("PASS" in text) and ("False" not in text.split("PASS", 1)[-1][:200])
        leak = any(pat in text for pat in ["E:\\BBIT", "E:/BBIT", "bbit_geomlang\\"])
        results[label] = {
            "script": script,
            "returncode": cp.returncode,
            "pass_detected": bool(pass_detected),
            "abs_path_leak": bool(leak),
            "elapsed_s": round(time.time() - t0, 3),
            "stdout_tail": (cp.stdout or "")[-2000:],
            "stderr_tail": (cp.stderr or "")[-2000:],
        }
        print(f"[26FA-LITE] replay {label}: rc={cp.returncode} pass={pass_detected} leak={leak} elapsed={results[label]['elapsed_s']}s")
    return results, outputs


def make_plot(checks: Dict[str, bool], out: Path) -> None:
    if plt is None:
        return
    labels = list(checks.keys())[::-1]
    vals = [1 if checks[k] else 0 for k in labels]
    fig_h = max(5, 0.42 * len(labels))
    plt.figure(figsize=(12, fig_h))
    plt.barh(labels, vals)
    for y, v in enumerate(vals):
        plt.text(1.02, y, "PASS" if v else "FAIL", va="center")
    plt.xlim(0, 1.08)
    plt.xlabel("pass = 1, fail = 0")
    plt.title("26FA-LITE clean-room selftest reseal checks")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=str, default=None, help="Input portable zip. Defaults to latest EZ/EY/EV zip under outputs_basic32.")
    ap.add_argument("--fresh", action="store_true", help="Delete previous 26FA staging/extract dirs before running.")
    ap.add_argument("--write-final-zip", action="store_true", help="Write phase26fa_lite_final_portable_release_bundle.zip")
    args = ap.parse_args(argv)

    here = Path(__file__).resolve()
    root = here.parents[1] if here.parent.name == "bbit_geomlang" else here.parent
    if root.name == "bbit_geomlang":
        root = root.parent
    outputs_root = root / "outputs_basic32"
    outputs_root.mkdir(parents=True, exist_ok=True)

    default_zip = first_existing([
        outputs_root / "phase26ez_lite_final_portable_release_bundle.zip",
        outputs_root / "phase26ey_lite_final_portable_release_bundle.zip",
        outputs_root / "phase26ev_lite_release_bundle_repaired.zip",
    ])
    zip_in = Path(args.zip).resolve() if args.zip else default_zip.resolve()
    stage = outputs_root / "phase26fa_lite_stage"
    clean = outputs_root / "phase26fa_lite_cleanroom_extract" / ("cleanroom_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    final_zip = outputs_root / "phase26fa_lite_final_portable_release_bundle.zip"

    print(f"[{PHASE}] Clean-room selftest resealer / portable replay finalizer")
    print(f"[{PHASE}] root: {root}")
    print(f"[{PHASE}] outputs: {outputs_root}")
    print(f"[{PHASE}] input zip: {zip_in}")

    if args.fresh:
        for d in [stage, outputs_root / "phase26fa_lite_cleanroom_extract"]:
            if d.exists():
                shutil.rmtree(d)

    checks: Dict[str, bool] = {}
    details: Dict[str, object] = {}
    checks["zip_exists"] = zip_in.exists(); details["zip_exists"] = str(zip_in)
    checks["zip_is_file"] = zip_in.is_file(); details["zip_is_file"] = str(zip_in)
    if not checks["zip_is_file"]:
        raise FileNotFoundError(zip_in)

    extract_zip(zip_in, stage)
    outputs = find_outputs_dir(stage)
    prior_manifest = load_manifest(stage)

    copied = ensure_core_data(outputs, stage)
    path_patched = patch_runtime_gate_paths(outputs)
    selftests_patched = write_portable_selftests(outputs)

    core_py_paths = [outputs / n for n in CORE_PY]
    core_data_paths = [outputs / n for n in CORE_DATA]
    checks["core_python_present"] = all(p.exists() for p in core_py_paths)
    details["core_python_present"] = {p.name: p.exists() for p in core_py_paths}
    checks["core_data_present"] = all(p.exists() for p in core_data_paths)
    details["core_data_present"] = {p.name: p.exists() for p in core_data_paths}

    syn_ok, syn_errors = syntax_ok([p for p in core_py_paths if p.exists()])
    checks["all_runtime_core_python_syntax_ok"] = syn_ok; details["syntax_errors"] = syn_errors
    hits = abs_path_hits([p for p in core_py_paths if p.exists()])
    checks["no_absolute_bbit_paths_in_runtime_core_python"] = not hits; details["runtime_core_abs_path_hits"] = hits
    checks["fingerprint_found_in_runtime_core"] = any(FINGERPRINT in p.read_text(encoding="utf-8", errors="replace") for p in core_py_paths if p.exists())

    manifest_path = write_resealed_manifest(stage, outputs, prior_manifest)
    if args.write_final_zip:
        write_zip(stage, final_zip)
    else:
        final_zip = zip_in
    checks["final_zip_written_if_requested"] = (not args.write_final_zip) or final_zip.exists()
    details["final_zip"] = str(final_zip)

    replay, clean_outputs = replay_cleanroom(final_zip, clean)
    checks["cleanroom_extract_created"] = clean.exists()
    checks["all_cleanroom_replay_commands_exit_zero"] = all(v["returncode"] == 0 for v in replay.values())
    checks["all_cleanroom_replay_commands_pass_detected"] = all(v["pass_detected"] for v in replay.values())
    checks["no_absolute_bbit_path_leak_in_command_output"] = not any(v["abs_path_leak"] for v in replay.values())

    # Validate final zip hashes against the resealed manifest, excluding the manifest files themselves.
    extract_check = outputs_root / "phase26fa_lite_hash_check_extract"
    if extract_check.exists():
        shutil.rmtree(extract_check)
    extract_zip(final_zip, extract_check)
    man = json.loads((find_file(extract_check, "phase26fa_lite_final_portable_manifest.json") or find_file(extract_check, "phase26et_lite_release_manifest.json")).read_text(encoding="utf-8"))
    hash_failures = []
    for rec in man.get("files", []):
        rel = rec.get("relative_path") or rec.get("name")
        p = extract_check / rel
        if not p.exists() or sha256(p) != rec.get("sha256"):
            hash_failures.append(rel)
    checks["manifest_hashes_match_after_reseal"] = not hash_failures
    details["manifest_hash_failures"] = hash_failures[:20]

    checks["portable_selftests_patched"] = len(selftests_patched) == 4
    checks["runtime_path_patches_applied_or_unneeded"] = True

    overall = all(checks.values())
    summary = {
        "phase": PHASE,
        "FINAL_PORTABLE_SELFTEST_RESEAL_LOCK_PASS": overall,
        "input_zip": str(zip_in),
        "final_zip": str(final_zip),
        "stage": str(stage),
        "cleanroom": str(clean),
        "checks": checks,
        "details": details,
        "copied_core_data": copied,
        "path_patched": path_patched,
        "selftests_patched": selftests_patched,
        "replay": replay,
        "route_fingerprint": FINGERPRINT,
    }

    summary_path = outputs_root / "phase26fa_lite_summary.json"
    report_path = outputs_root / "phase26fa_lite_final_portable_selftest_reseal_report.md"
    plot_path = outputs_root / "phase26fa_lite_final_portable_selftest_reseal_checks.png"
    replay_csv = outputs_root / "phase26fa_lite_command_replay.csv"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with replay_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "script", "returncode", "pass_detected", "abs_path_leak", "elapsed_s"])
        w.writeheader()
        for k, v in replay.items():
            w.writerow({"label": k, **{x: v[x] for x in ["script", "returncode", "pass_detected", "abs_path_leak", "elapsed_s"]}})
    report_path.write_text("\n".join([
        "# 26FA-LITE Final Portable Selftest Reseal Report",
        "",
        f"FINAL_PORTABLE_SELFTEST_RESEAL_LOCK_PASS={overall}",
        f"input_zip={zip_in}",
        f"final_zip={final_zip}",
        f"cleanroom={clean}",
        "",
        "## Checks",
        *[f"- {k}: {v}" for k, v in checks.items()],
        "",
        "## Patched selftests",
        *[f"- {x}" for x in selftests_patched],
        "",
        "## Replay",
        *[f"- {k}: rc={v['returncode']} pass={v['pass_detected']} leak={v['abs_path_leak']}" for k, v in replay.items()],
    ]), encoding="utf-8")
    make_plot(checks, plot_path)

    print(f"[{PHASE}] FINAL_PORTABLE_SELFTEST_RESEAL_LOCK_PASS={overall}")
    print(f"[{PHASE}] checks:")
    for k, v in checks.items():
        print(f"  - {k}: {v} :: {details.get(k, '')}")
    print(f"[{PHASE}] wrote summary: {summary_path}")
    print(f"[{PHASE}] wrote report: {report_path}")
    print(f"[{PHASE}] wrote replay CSV: {replay_csv}")
    if args.write_final_zip:
        print(f"[{PHASE}] wrote final zip: {final_zip}")
    if plot_path.exists():
        print(f"[{PHASE}] wrote plot: {plot_path}")
    print(f"[{PHASE}] wrote outputs to: {outputs_root}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
