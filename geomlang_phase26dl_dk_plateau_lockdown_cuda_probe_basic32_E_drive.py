#!/usr/bin/env python3
"""
Phase 26DL — DK plateau lockdown + CUDA readiness probe.

What this phase is for
----------------------
26DK found a very broad, nearly-flat elite plateau rather than a sharp single
winner. 26DL stops expanding the search and instead asks:

1) Which DK winner family survives a compact re-evaluation?
2) Are radius/seat/blend/cap/shell/tangent-kill genuinely important, or are
   some dimensions effectively neutral once the bowl is in the right regime?
3) Can we make the next step cheaper by moving only the expensive rollout/grid
   loops to CUDA later?

This script intentionally keeps the case count modest. It prefers DK's CSV if
it is present, deduplicates the top DK parameter profiles, adds a small set of
controlled perturbations around the plateau center, then emits concise plots.
"""
from __future__ import annotations

import csv
import glob
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)

PHASE = "26DL"
TITLE = "DK plateau lockdown and CUDA-readiness probe"
DESCRIPTION = (
    "Re-evaluates the compact DK elite plateau, deduplicates the winning "
    "profiles, probes only local one-step perturbations, and reports whether "
    "the search is now dominated by a stable ridge rather than a sharp optimum."
)
ARTIFACT_PREFIX = "phase26dl"

ROOT = Path(r"E:\BBIT\outputs_basic32")
SRC = Path(r"E:\BBIT\bbit_geomlang")
if not ROOT.exists():
    ROOT = Path.cwd()
if not SRC.exists():
    SRC = Path.cwd()

# Prefer DK if it exists locally; otherwise fall back through DJ/DI.
PREV_GLOBS = [
    "geomlang_phase26dk_*basic32_E_drive.py",
    "geomlang_phase26dj_*basic32_E_drive.py",
    "geomlang_phase26di_*basic32_E_drive.py",
]

TRACKED = [
    "BOWL_RADIUS_FRAC",
    "BOWL_SEAT_AXIS_GAIN",
    "BOWL_DIRECTIONAL_BLEND",
    "BOWL_NORM_CAP_MULT",
    "BOWL_SHELL_RADIAL_GAIN",
    "BOWL_TANGENT_KILL",
]

# DK's visible plateau center from the user's 26DK run.
PLATEAU_CENTER = {
    "BOWL_RADIUS_FRAC": 0.66,
    "BOWL_SEAT_AXIS_GAIN": 0.35,
    "BOWL_DIRECTIONAL_BLEND": 1.00,
    "BOWL_NORM_CAP_MULT": 1.00,
    "BOWL_SHELL_RADIAL_GAIN": 0.10,
    "BOWL_TANGENT_KILL": 0.90,
}

# Small perturbation grid, not a full cartesian explosion.
LOCAL_VALUES = {
    "BOWL_RADIUS_FRAC": [0.62, 0.64, 0.66, 0.68, 0.70],
    "BOWL_SEAT_AXIS_GAIN": [0.25, 0.30, 0.35, 0.375, 0.40],
    "BOWL_DIRECTIONAL_BLEND": [0.96, 0.98, 1.00, 1.02, 1.04],
    "BOWL_NORM_CAP_MULT": [1.00, 1.02, 1.04, 1.06, 1.08],
    "BOWL_SHELL_RADIAL_GAIN": [0.08, 0.10, 0.12],
    "BOWL_TANGENT_KILL": [0.84, 0.88, 0.90, 0.92, 0.96],
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def find_previous() -> Tuple[Any, Path]:
    candidates: List[Path] = []
    for pat in PREV_GLOBS:
        candidates.extend(Path(p).resolve() for p in glob.glob(str(SRC / pat)))
        candidates.extend(Path(p).resolve() for p in glob.glob(str(Path.cwd() / pat)))
    # Remove this file and duplicates, newest preferred inside each phase preference order.
    seen = set()
    ordered: List[Path] = []
    for pat in PREV_GLOBS:
        phase_hits = [p for p in candidates if Path(p).match(pat)]
        phase_hits = sorted(set(phase_hits), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
        for p in phase_hits:
            if p.name not in seen and p.name != Path(__file__).name:
                seen.add(p.name)
                ordered.append(p)
    if not ordered:
        raise FileNotFoundError("Could not find a prior 26DK/26DJ/26DI script to import.")

    skipped: List[Path] = []
    for path in ordered:
        mod = load_module(path, f"phase26dl_prev_{path.stem}")
        base = getattr(mod, "BASE", mod)
        has_runtime = hasattr(mod, "build_basins") or all(
            hasattr(base, name)
            for name in [
                "load_model",
                "build_synthetic_dataset",
                "encode_dataset",
                "evaluate_grid",
                "pick_best_config",
            ]
        )
        if has_runtime:
            if skipped:
                print(
                    f"[{PHASE}] Skipped wrapper-only helpers: "
                    + ", ".join(p.name for p in skipped)
                )
            print(f"[{PHASE}] Loaded previous helpers from: {path}")
            return mod, path
        skipped.append(path)

    # Last resort: return the preferred wrapper so the eventual error names the
    # missing API from the module the user most likely expected us to reuse.
    mod = load_module(ordered[0], f"phase26dl_prev_{ordered[0].stem}_fallback")
    print(f"[{PHASE}] Loaded previous helpers from: {ordered[0]}")
    return mod, ordered[0]


PREV, PREV_PATH = find_previous()
BASE = getattr(PREV, "BASE", PREV)
MODS = []
for m in [BASE, PREV, getattr(PREV, "PREV", None)]:
    if m is not None and m not in MODS:
        MODS.append(m)


def patch_meta() -> None:
    outs = {
        "ROOT": ROOT,
        "PHASE": PHASE,
        "TITLE": TITLE,
        "DESCRIPTION": DESCRIPTION,
        "ARTIFACT_PREFIX": ARTIFACT_PREFIX,
        "OUT_CAPTURE": ROOT / f"{ARTIFACT_PREFIX}_capture_curves.png",
        "OUT_HEAT": ROOT / f"{ARTIFACT_PREFIX}_force_heatmaps.png",
        "OUT_ROLLOUT": ROOT / f"{ARTIFACT_PREFIX}_rollout_examples.png",
        "OUT_FIELD": ROOT / f"{ARTIFACT_PREFIX}_vector_field_examples.png",
        "OUT_SUMMARY": ROOT / f"{ARTIFACT_PREFIX}_summary.json",
    }
    for m in MODS:
        for k, v in outs.items():
            try:
                setattr(m, k, v)
            except Exception:
                pass


patch_meta()


def baseline_constants() -> Dict[str, float]:
    vals: Dict[str, float] = {}
    prev_base = getattr(PREV, "BASELINE", None)
    if not isinstance(prev_base, dict):
        prev_base = getattr(PREV, "BASELINE_CONSTANTS", {})
    for name in TRACKED:
        if isinstance(prev_base, dict) and name in prev_base:
            vals[name] = float(prev_base[name])
            continue
        for m in MODS:
            if hasattr(m, name):
                vals[name] = float(getattr(m, name))
                break
    return vals


BASELINE = baseline_constants()


def clean_overrides(ov: Dict[str, Any]) -> Dict[str, float]:
    out = {}
    for k, v in ov.items():
        if k in BASELINE and v not in [None, ""]:
            try:
                vv = float(v)
                if math.isfinite(vv):
                    out[k] = vv
            except Exception:
                pass
    return out


def apply(overrides: Dict[str, Any]) -> None:
    merged = dict(BASELINE)
    merged.update(clean_overrides(overrides))
    for m in MODS:
        for k, v in merged.items():
            if hasattr(m, k):
                try:
                    setattr(m, k, float(v))
                except Exception:
                    pass
        # Preserve any wrapper phase's current vector field implementation.
        if hasattr(PREV, "evaluate_field"):
            try:
                m.evaluate_field = PREV.evaluate_field
            except Exception:
                pass


def num(x: Any, d: float = 0.0) -> float:
    try:
        v = float(x)
        return d if math.isnan(v) or math.isinf(v) else v
    except Exception:
        return d


def score_from_best(best: Dict[str, Any]) -> float:
    # Reuse prior scoring if available so the scale remains comparable.
    if hasattr(PREV, "score"):
        try:
            return float(PREV.score(best))
        except Exception:
            pass
    for k in ["robust_score", "composite_score", "score", "best_score", "mean_score", "objective"]:
        if k in best:
            return num(best[k], -1e9)
    return (
        5.0 * num(best.get("capture_rate"))
        + 2.0 * num(best.get("distance_progress"))
        + 0.5 * num(best.get("signed_alignment"))
        - 0.25 * num(best.get("tangent_ratio"))
        - 0.05 * num(best.get("mean_endpoint_dist", best.get("endpoint_dist")))
    )


def compact(best: Dict[str, Any]) -> Dict[str, Any]:
    if hasattr(PREV, "compact"):
        try:
            return dict(PREV.compact(best))
        except Exception:
            pass
    keys = [
        "sigma_mult", "strength", "capture_rate", "distance_progress",
        "signed_alignment", "tangent_ratio", "mean_endpoint_dist",
        "endpoint_dist", "composite_score", "score", "best_score", "mean_score",
        "objective", "robust_score",
    ]
    out = {k: best.get(k) for k in keys if k in best}
    out["dl_scalar_score"] = score_from_best(best)
    return out


def build_basins() -> Tuple[Any, Any, Any]:
    if hasattr(PREV, "build_basins"):
        return PREV.build_basins()
    model = BASE.load_model()
    imgs, rel_ids, scale_ids, sr_ids, sb_ids = BASE.build_synthetic_dataset()
    z = BASE.encode_dataset(model, imgs).astype(np.float32)
    if z.shape[1] > BASE.PCA_DIM:
        z2 = BASE.PCA(n_components=BASE.PCA_DIM, random_state=BASE.SEED).fit_transform(z).astype(np.float32)
    else:
        z2 = z.copy()
    border_idx = BASE.select_border_particles(z2)
    basins = BASE.build_pair_basins(z2, rel_ids, border_idx)
    if not basins:
        raise RuntimeError("No basins constructed.")
    return z2, rel_ids, basins


def eval_case(name: str, overrides: Dict[str, Any], basins: Any) -> Dict[str, Any]:
    ov = clean_overrides(overrides)
    apply(ov)
    t0 = time.time()
    rows = BASE.evaluate_grid(basins)
    best = BASE.pick_best_config(rows)
    elapsed = time.time() - t0
    sc = score_from_best(best)
    return {
        "case": name,
        "overrides": ov,
        "rows": rows,
        "best": best,
        "best_summary": compact(best),
        "score": sc,
        "elapsed_sec": elapsed,
    }


def read_dk_csv_candidates(limit: int = 24) -> List[Tuple[str, Dict[str, float]]]:
    paths = [ROOT / "phase26dk_case_results.csv", Path.cwd() / "phase26dk_case_results.csv"]
    path = next((p for p in paths if p.exists()), None)
    if path is None:
        return []
    out: List[Tuple[str, Dict[str, float]]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    def row_score(r: Dict[str, str]) -> float:
        for k in ["robust_score", "score", "mean_score", "composite_score"]:
            if k in r and r[k] not in [None, ""]:
                return num(r[k], -1e9)
        return -1e9
    rows.sort(key=row_score, reverse=True)
    seen = set()
    for r in rows:
        ov = clean_overrides(r)
        sig = tuple((k, round(v, 6)) for k, v in sorted(ov.items()))
        if ov and sig not in seen:
            seen.add(sig)
            out.append((f"dk_csv_{len(out)+1:02d}_{r.get('case','elite')}", ov))
        if len(out) >= limit:
            break
    print(f"[{PHASE}] Loaded {len(out)} deduped candidates from {path}")
    return out


def add_unique(out: List[Tuple[str, Dict[str, float]]], name: str, ov: Dict[str, Any]) -> None:
    cov = clean_overrides(ov)
    sig = tuple((k, round(v, 6)) for k, v in sorted(cov.items()))
    existing = {tuple((k, round(v, 6)) for k, v in sorted(o.items())) for _, o in out}
    if sig not in existing:
        out.append((name, cov))


def build_cases() -> List[Tuple[str, Dict[str, float]]]:
    cases: List[Tuple[str, Dict[str, float]]] = []

    for name, ov in read_dk_csv_candidates(limit=18):
        add_unique(cases, name, ov)

    # Explicit center and DK visible winner variants.
    add_unique(cases, "dl_plateau_center_r066_seat035_blend100_cap100_shell010_tk090", PLATEAU_CENTER)
    for radius in [0.62, 0.66, 0.70]:
        c = dict(PLATEAU_CENTER, BOWL_RADIUS_FRAC=radius)
        add_unique(cases, f"dl_radius_check_{radius:.2f}", c)
    for seat in [0.25, 0.35, 0.40]:
        c = dict(PLATEAU_CENTER, BOWL_SEAT_AXIS_GAIN=seat)
        add_unique(cases, f"dl_seat_check_{seat:.3f}", c)
    for blend in [0.92, 0.96, 1.00, 1.04]:
        c = dict(PLATEAU_CENTER, BOWL_DIRECTIONAL_BLEND=blend)
        add_unique(cases, f"dl_blend_check_{blend:.2f}", c)
    for cap in [1.00, 1.02, 1.04, 1.06, 1.08]:
        c = dict(PLATEAU_CENTER, BOWL_NORM_CAP_MULT=cap)
        add_unique(cases, f"dl_cap_check_{cap:.2f}", c)
    for shell in [0.08, 0.10, 0.12]:
        c = dict(PLATEAU_CENTER, BOWL_SHELL_RADIAL_GAIN=shell)
        add_unique(cases, f"dl_shell_check_{shell:.2f}", c)
    for tk in [0.84, 0.88, 0.90, 0.92, 0.96]:
        c = dict(PLATEAU_CENTER, BOWL_TANGENT_KILL=tk)
        add_unique(cases, f"dl_tk_check_{tk:.2f}", c)

    # A tiny interaction set that tests the actual DK ridge signal without exploding runtime.
    for radius, seat, cap in [
        (0.62, 0.25, 1.00),
        (0.62, 0.35, 1.00),
        (0.66, 0.35, 1.00),
        (0.66, 0.35, 1.04),
        (0.70, 0.35, 1.00),
        (0.70, 0.40, 1.00),
    ]:
        c = dict(PLATEAU_CENTER, BOWL_RADIUS_FRAC=radius, BOWL_SEAT_AXIS_GAIN=seat, BOWL_NORM_CAP_MULT=cap)
        add_unique(cases, f"dl_r{radius:.2f}_seat{seat:.3f}_cap{cap:.2f}", c)

    return cases


def write_csv(path: Path, results: List[Dict[str, Any]], base_score: float) -> None:
    consts = sorted({k for r in results for k in r["overrides"]})
    metrics = sorted({k for r in results for k in r["best_summary"]})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["case", "score", "delta_vs_baseline", "elapsed_sec"] + consts + metrics,
            extrasaction="ignore",
        )
        w.writeheader()
        for r in results:
            row = {
                "case": r["case"],
                "score": r["score"],
                "delta_vs_baseline": r["score"] - base_score,
                "elapsed_sec": r.get("elapsed_sec", 0.0),
            }
            row.update(r["overrides"])
            row.update(r["best_summary"])
            w.writerow(row)


def plot_scores(results: List[Dict[str, Any]], base_score: float) -> None:
    plt = BASE.plt
    ordered = sorted(results, key=lambda r: r["score"])
    plt.figure(figsize=(13, max(6, 0.27 * len(ordered))))
    plt.barh([r["case"] for r in ordered], [r["score"] for r in ordered], label="score")
    plt.axvline(base_score, linewidth=1.0, label="baseline")
    plt.title(f"{PHASE} DK plateau lockdown scores")
    plt.xlabel("DL scalar score")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(ROOT / f"{ARTIFACT_PREFIX}_top_scores.png", dpi=160)
    plt.close()


def plot_one_factor(results: List[Dict[str, Any]]) -> None:
    plt = BASE.plt
    center = clean_overrides(PLATEAU_CENTER)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.ravel()
    for ax, key in zip(axes, TRACKED):
        xs, ys = [], []
        for r in results:
            ov = r["overrides"]
            if key not in ov:
                continue
            ok = True
            for k, cv in center.items():
                if k == key:
                    continue
                if k in ov and abs(float(ov[k]) - float(cv)) > 1e-9:
                    ok = False
                    break
            if ok:
                xs.append(float(ov[key]))
                ys.append(float(r["score"]))
        if xs:
            order = np.argsort(xs)
            ax.plot(np.array(xs)[order], np.array(ys)[order], marker="o")
        ax.set_title(key.replace("BOWL_", "").lower())
        ax.set_xlabel(key)
        ax.set_ylabel("score")
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"{PHASE} one-factor plateau slices", fontsize=16)
    fig.tight_layout()
    fig.savefig(ROOT / f"{ARTIFACT_PREFIX}_parameter_slices.png", dpi=160)
    plt.close(fig)


def cuda_probe() -> Dict[str, Any]:
    info: Dict[str, Any] = {"torch_imported": False, "cuda_available": False}
    try:
        import torch  # type: ignore
        info["torch_imported"] = True
        info["torch_version"] = getattr(torch, "__version__", "unknown")
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["cuda_device_count"] = int(torch.cuda.device_count())
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            # Tiny sanity timing; this is not used for the phase result.
            a = torch.randn(256, 256, device="cuda")
            b = torch.randn(256, 256, device="cuda")
            torch.cuda.synchronize()
            t0 = time.time()
            c = a @ b
            torch.cuda.synchronize()
            info["tiny_matmul_ms"] = (time.time() - t0) * 1000.0
            info["tiny_checksum"] = float(c[0, 0].detach().cpu())
    except Exception as e:
        info["error"] = repr(e)
    return info


def emit_inherited_plots(z2: Any, rel_ids: Any, basins: Any, best_result: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    apply(best_result["overrides"])
    rows = BASE.evaluate_grid(basins)
    best = BASE.pick_best_config(rows)
    for fname in ["plot_capture_curves", "plot_heatmaps", "plot_rollout_examples", "plot_vector_fields"]:
        try:
            fn = getattr(BASE, fname)
            if fname == "plot_capture_curves":
                fn(rows)
            elif fname == "plot_heatmaps":
                fn(rows, basins)
            elif fname == "plot_rollout_examples":
                fn(z2, rel_ids, basins, best)
            elif fname == "plot_vector_fields":
                fn(z2, basins, best)
        except Exception as e:
            msg = f"{fname}: {e}"
            errors.append(msg)
            print(f"[{PHASE}] plot warning: {msg}")
    return errors


def main() -> Dict[str, Any]:
    print(f"[{PHASE}] {TITLE}")
    print(f"[{PHASE}] Output root: {ROOT}")
    print(f"[{PHASE}] Baseline constants detected:")
    for k, v in sorted(BASELINE.items()):
        print(f"  {k} = {v}")

    cuda = cuda_probe()
    print(f"[{PHASE}] CUDA probe: {json.dumps(cuda, indent=2)}")

    z2, rel_ids, basins = build_basins()
    print(f"[{PHASE}] Built basins: {len(basins)}")

    results: List[Dict[str, Any]] = []
    baseline = eval_case("26_base_current_constants", {}, basins)
    results.append(baseline)
    base_score = baseline["score"]
    print(f"[{PHASE}] baseline score={base_score:.9g}")

    cases = build_cases()
    print(f"[{PHASE}] Evaluating {len(cases)} lockdown cases.")
    for i, (name, ov) in enumerate(cases, start=1):
        r = eval_case(name, ov, basins)
        results.append(r)
        print(
            f"[{PHASE}] {i:03d}/{len(cases):03d} {name:<68} "
            f"score={r['score']:.9g} delta={r['score']-base_score:+.9g} "
            f"time={r['elapsed_sec']:.2f}s"
        )

    best = max(results, key=lambda r: r["score"])
    ordered = sorted(results, key=lambda r: r["score"], reverse=True)
    top_scores = [r["score"] for r in ordered[: min(10, len(ordered))]]
    plateau_width = max(top_scores) - min(top_scores) if top_scores else 0.0

    print(f"[{PHASE}] Best case: {best['case']} score={best['score']:.9g}")
    print(f"[{PHASE}] Top-10 plateau width: {plateau_width:.9g}")
    print(f"[{PHASE}] Best overrides:")
    for k, v in sorted(best["overrides"].items()):
        print(f"  {k} = {v}")

    write_csv(ROOT / f"{ARTIFACT_PREFIX}_case_results.csv", results, base_score)
    plot_scores(results, base_score)
    plot_one_factor(results)
    plot_errors = emit_inherited_plots(z2, rel_ids, basins, best)

    summary = {
        "phase": PHASE,
        "title": TITLE,
        "description": DESCRIPTION,
        "previous_script": str(PREV_PATH),
        "n_basins": len(basins),
        "baseline_constants": BASELINE,
        "plateau_center": clean_overrides(PLATEAU_CENTER),
        "cuda_probe": cuda,
        "n_cases_evaluated": len(results),
        "baseline_score": base_score,
        "best_case": {
            "case": best["case"],
            "score": best["score"],
            "delta_vs_baseline": best["score"] - base_score,
            "elapsed_sec": best.get("elapsed_sec", 0.0),
            "overrides": best["overrides"],
            "best_summary": best["best_summary"],
        },
        "top_10_cases": [
            {
                "case": r["case"],
                "score": r["score"],
                "overrides": r["overrides"],
                "best_summary": r["best_summary"],
            }
            for r in ordered[:10]
        ],
        "top_10_plateau_width": plateau_width,
        "interpretation_hint": (
            "If the top_10_plateau_width is tiny and cap=1.00 remains dominant, "
            "treat DK/DL as a stable ridge and stop broad scalar searching. "
            "Next step should be either CUDA-vectorized rollouts or qualitative "
            "latent-endpoint diagnostics, not another huge CPU lattice."
        ),
        "plot_errors": plot_errors,
        "artifacts": [
            f"{ARTIFACT_PREFIX}_summary.json",
            f"{ARTIFACT_PREFIX}_case_results.csv",
            f"{ARTIFACT_PREFIX}_top_scores.png",
            f"{ARTIFACT_PREFIX}_parameter_slices.png",
            f"{ARTIFACT_PREFIX}_capture_curves.png",
            f"{ARTIFACT_PREFIX}_force_heatmaps.png",
            f"{ARTIFACT_PREFIX}_rollout_examples.png",
            f"{ARTIFACT_PREFIX}_vector_field_examples.png",
        ],
    }
    with (ROOT / f"{ARTIFACT_PREFIX}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    apply({})
    return summary


if __name__ == "__main__":
    main()
