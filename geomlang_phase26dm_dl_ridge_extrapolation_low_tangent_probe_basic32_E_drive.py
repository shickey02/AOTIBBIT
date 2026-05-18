#!/usr/bin/env python3
"""
Phase 26DM — DL ridge extrapolation + low-tangent kill probe.

What this phase is for
----------------------
26DL showed that the DK plateau is not perfectly flat: the one-factor slices
improved along a ridge: radius/seat/blend/cap/shell upward, tangent-kill downward.
26DM asks:

1) Does the DL ridge continue improving if we step just beyond DL's best
   visible values?
2) Is low tangent-kill genuinely causal, or did it only win as a single-factor
   check?
3) Can we identify a smaller DM family to freeze before CUDA/vectorized rollouts?

This script keeps the case count modest. It reads DL/DK CSV candidates when
present, adds ridge extrapolations around the DL visible winner, then emits
concise plots and a short JSON summary.
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

PHASE = "26DM"
TITLE = "DL ridge extrapolation and low-tangent kill probe"
DESCRIPTION = (
    "Follows the DL ridge signal with compact extrapolations: larger radius/seat/"
    "blend/cap/shell and lower tangent-kill. Reports whether the ridge keeps "
    "improving or folds back into the DK/DL plateau."
)
ARTIFACT_PREFIX = "phase26dm"

ROOT = Path(r"E:\BBIT\outputs_basic32")
SRC = Path(r"E:\BBIT\bbit_geomlang")
if not ROOT.exists():
    ROOT = Path.cwd()
if not SRC.exists():
    SRC = Path.cwd()

# Prefer DK if it exists locally; otherwise fall back through DJ/DI.
PREV_GLOBS = [
    "geomlang_phase26dl_*basic32_E_drive.py",
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
PARAM_KEYS = TRACKED

# DL visible ridge center from the user's 26DL run.
# DL one-factor slices suggested improvement at radius=.70, seat=.40, blend=1.04,
# cap=1.08, shell=.12, and tangent_kill=.84. DM centers there and probes around it.
PLATEAU_CENTER = {
    "BOWL_RADIUS_FRAC": 0.70,
    "BOWL_SEAT_AXIS_GAIN": 0.40,
    "BOWL_DIRECTIONAL_BLEND": 1.04,
    "BOWL_NORM_CAP_MULT": 1.08,
    "BOWL_SHELL_RADIAL_GAIN": 0.12,
    "BOWL_TANGENT_KILL": 0.84,
}

# Small perturbation grid, not a full cartesian explosion.
LOCAL_VALUES = {
    "BOWL_RADIUS_FRAC": [0.66, 0.70, 0.74, 0.78],
    "BOWL_SEAT_AXIS_GAIN": [0.35, 0.40, 0.45, 0.50],
    "BOWL_DIRECTIONAL_BLEND": [1.00, 1.04, 1.08, 1.12],
    "BOWL_NORM_CAP_MULT": [1.04, 1.08, 1.12, 1.16],
    "BOWL_SHELL_RADIAL_GAIN": [0.10, 0.12, 0.14, 0.16],
    "BOWL_TANGENT_KILL": [0.72, 0.76, 0.80, 0.84, 0.88],
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
    mod = load_module(ordered[0], f"phase26dl_prev_{ordered[0].stem}")
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
    prev_base = getattr(PREV, "BASELINE", {})
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


def read_prior_csv_candidates(limit: int = 30) -> List[Tuple[str, Dict[str, float]]]:
    paths: List[Path] = []
    for stem in ["phase26dl_case_results.csv", "phase26dk_case_results.csv"]:
        paths.extend([ROOT / stem, Path.cwd() / stem])
    existing = [p for p in paths if p.exists()]
    if not existing:
        return []

    rows: List[Dict[str, str]] = []
    for path in existing:
        with path.open("r", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rr = dict(r)
                rr["_source_csv"] = path.name
                rows.append(rr)

    rows.sort(key=lambda r: float(r.get("score", "-999")), reverse=True)
    out: List[Tuple[str, Dict[str, float]]] = []
    seen = set()
    for r in rows:
        ov: Dict[str, float] = {}
        ok = True
        for k in PARAM_KEYS:
            if k not in r or r[k] == "":
                ok = False
                break
            ov[k] = float(r[k])
        if not ok:
            continue
        sig = tuple(round(ov[k], 6) for k in PARAM_KEYS)
        if sig in seen:
            continue
        seen.add(sig)
        src = r.get("_source_csv", "csv").replace("_case_results.csv", "")
        out.append((f"prior_csv_{len(out)+1:02d}_{src}_{r.get('case','elite')}", ov))
        if len(out) >= limit:
            break
    print(f"[{PHASE}] Loaded {len(out)} deduped candidates from {len(existing)} prior CSV file(s)")
    return out

def add_unique(out: List[Tuple[str, Dict[str, float]]], name: str, ov: Dict[str, Any]) -> None:
    cov = clean_overrides(ov)
    sig = tuple((k, round(v, 6)) for k, v in sorted(cov.items()))
    existing = {tuple((k, round(v, 6)) for k, v in sorted(o.items())) for _, o in out}
    if sig not in existing:
        out.append((name, cov))


def build_cases() -> List[Tuple[str, Dict[str, float]]]:
    cases: List[Tuple[str, Dict[str, float]]] = []

    for name, ov in read_prior_csv_candidates(limit=20):
        add_unique(cases, name, ov)

    add_unique(cases, "dm_ridge_center_r070_seat040_blend104_cap108_shell012_tk084", PLATEAU_CENTER)

    # One-factor ridge extrapolations around the DL-visible endpoint.
    for radius in LOCAL_VALUES["BOWL_RADIUS_FRAC"]:
        add_unique(cases, f"dm_radius_ridge_{radius:.2f}", dict(PLATEAU_CENTER, BOWL_RADIUS_FRAC=radius))
    for seat in LOCAL_VALUES["BOWL_SEAT_AXIS_GAIN"]:
        add_unique(cases, f"dm_seat_ridge_{seat:.3f}", dict(PLATEAU_CENTER, BOWL_SEAT_AXIS_GAIN=seat))
    for blend in LOCAL_VALUES["BOWL_DIRECTIONAL_BLEND"]:
        add_unique(cases, f"dm_blend_ridge_{blend:.2f}", dict(PLATEAU_CENTER, BOWL_DIRECTIONAL_BLEND=blend))
    for cap in LOCAL_VALUES["BOWL_NORM_CAP_MULT"]:
        add_unique(cases, f"dm_cap_ridge_{cap:.2f}", dict(PLATEAU_CENTER, BOWL_NORM_CAP_MULT=cap))
    for shell in LOCAL_VALUES["BOWL_SHELL_RADIAL_GAIN"]:
        add_unique(cases, f"dm_shell_ridge_{shell:.2f}", dict(PLATEAU_CENTER, BOWL_SHELL_RADIAL_GAIN=shell))
    for tk in LOCAL_VALUES["BOWL_TANGENT_KILL"]:
        add_unique(cases, f"dm_tk_low_probe_{tk:.2f}", dict(PLATEAU_CENTER, BOWL_TANGENT_KILL=tk))

    # Tiny interaction probes: follow the ridge together, then test overreach/foldback.
    interaction_profiles = [
        (0.66, 0.35, 1.00, 1.00, 0.10, 0.80),
        (0.66, 0.35, 1.04, 1.08, 0.12, 0.80),
        (0.70, 0.40, 1.04, 1.08, 0.12, 0.84),
        (0.70, 0.40, 1.04, 1.08, 0.12, 0.80),
        (0.70, 0.40, 1.08, 1.12, 0.14, 0.80),
        (0.74, 0.40, 1.04, 1.08, 0.12, 0.80),
        (0.74, 0.45, 1.08, 1.12, 0.14, 0.80),
        (0.74, 0.45, 1.08, 1.12, 0.14, 0.76),
        (0.78, 0.45, 1.08, 1.12, 0.14, 0.76),
        (0.78, 0.50, 1.12, 1.16, 0.16, 0.76),
        # foldback checks
        (0.78, 0.50, 1.12, 1.16, 0.16, 0.72),
        (0.82, 0.50, 1.12, 1.16, 0.16, 0.72),
        (0.74, 0.45, 1.16, 1.20, 0.18, 0.72),
    ]
    for radius, seat, blend, cap, shell, tk in interaction_profiles:
        ov = {
            "BOWL_RADIUS_FRAC": radius,
            "BOWL_SEAT_AXIS_GAIN": seat,
            "BOWL_DIRECTIONAL_BLEND": blend,
            "BOWL_NORM_CAP_MULT": cap,
            "BOWL_SHELL_RADIAL_GAIN": shell,
            "BOWL_TANGENT_KILL": tk,
        }
        add_unique(
            cases,
            f"dm_interact_r{radius:.2f}_seat{seat:.2f}_b{blend:.2f}_cap{cap:.2f}_sh{shell:.2f}_tk{tk:.2f}",
            ov,
        )

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
    plt.title(f"{PHASE} DL ridge extrapolation scores")
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
    fig.suptitle(f"{PHASE} ridge extrapolation one-factor slices", fontsize=16)
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
            "If DM keeps improving at the outward ridge edge, the basin parameters were not "
            "locked yet and 26DN should do a very small second extrapolation or CUDA batch. "
            "If DM folds back, freeze the best DM family and move to endpoint/latent "
            "quality diagnostics instead of more scalar sweeps."
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
