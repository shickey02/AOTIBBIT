# geomlang_phase26dn_elite_stability_low_tangent_audit_basic32_E_drive.py
# Phase 26DN — elite stability + low-tangent confirmation audit
#
# What this phase does:
#   1) Loads the exact 26DM evaluator, so the field/rollout mechanics stay comparable.
#   2) Pulls the best 26DM candidates from phase26dm_case_results.csv when available.
#   3) Adds hand-picked DN candidates around the two important 26DM discoveries:
#        - the high-score ridge around r=.66 / seat=.35 / blend=1 / cap=1 / shell=.12 / tk=.90
#        - the promising interaction ridge around r=.78 / seat=.50 / blend=1.12 / cap=1.16 / shell=.16 / tk=.76
#   4) Re-ranks with a DN score that rewards capture/progress/alignment but explicitly penalizes
#      the tangent/radial ratio so we do not accidentally select a bridge that wins by sideways drift.
#   5) Performs a local jitter audit around the top cases to check whether the candidate is stable
#      or just a single-parameter accident.
#
# Run from E:\BBIT:
#   python bbit_geomlang/geomlang_phase26dn_elite_stability_low_tangent_audit_basic32_E_drive.py

from __future__ import annotations

import importlib.util
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PHASE = "26DN"
TITLE = "elite stability + low-tangent confirmation audit"

ROOT = Path(r"E:/BBIT") if Path(r"E:/BBIT").exists() else Path.cwd()
SCRIPT_DIR = ROOT / "bbit_geomlang"
OUT_DIR = ROOT / "outputs_basic32"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DM_PATH = SCRIPT_DIR / "geomlang_phase26dm_dl_ridge_extrapolation_low_tangent_probe_basic32_E_drive.py"
if not DM_PATH.exists():
    # fallback for running from the same folder as the script
    DM_PATH = Path(__file__).with_name("geomlang_phase26dm_dl_ridge_extrapolation_low_tangent_probe_basic32_E_drive.py")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


DM = load_module(DM_PATH, "phase26dm")
BASE = DM.BASE
print(f"[{PHASE}] Loaded Phase 26DM helpers from: {DM_PATH}")

# Evaluation grid inherited from 26DM for comparability.
SIGMA_MULTS = list(getattr(DM, "SIGMA_MULTS", [0.50, 0.85, 1.20, 1.80, 2.60, 3.80]))
STRENGTHS = list(getattr(DM, "STRENGTHS", [0.00, 0.20, 0.45, 0.90, 1.50, 2.40]))
BASINS_CACHE: Optional[Tuple[Any, Any, Any]] = None

# DN jitter grid. These are deliberately small because this is not a broad search;
# it is a stability audit around the 26DM ridge.
JITTERS = [
    {"dr": 0.00, "dseat": 0.00, "dblend": 0.00, "dcap": 0.00, "dshell": 0.00, "dtk": 0.00},
    {"dr": -0.02, "dseat": 0.00, "dblend": 0.00, "dcap": 0.00, "dshell": 0.00, "dtk": 0.00},
    {"dr": +0.02, "dseat": 0.00, "dblend": 0.00, "dcap": 0.00, "dshell": 0.00, "dtk": 0.00},
    {"dr": 0.00, "dseat": -0.05, "dblend": 0.00, "dcap": 0.00, "dshell": 0.00, "dtk": 0.00},
    {"dr": 0.00, "dseat": +0.05, "dblend": 0.00, "dcap": 0.00, "dshell": 0.00, "dtk": 0.00},
    {"dr": 0.00, "dseat": 0.00, "dblend": -0.04, "dcap": 0.00, "dshell": 0.00, "dtk": 0.00},
    {"dr": 0.00, "dseat": 0.00, "dblend": +0.04, "dcap": 0.00, "dshell": 0.00, "dtk": 0.00},
    {"dr": 0.00, "dseat": 0.00, "dblend": 0.00, "dcap": -0.04, "dshell": 0.00, "dtk": 0.00},
    {"dr": 0.00, "dseat": 0.00, "dblend": 0.00, "dcap": +0.04, "dshell": 0.00, "dtk": 0.00},
    {"dr": 0.00, "dseat": 0.00, "dblend": 0.00, "dcap": 0.00, "dshell": -0.02, "dtk": 0.00},
    {"dr": 0.00, "dseat": 0.00, "dblend": 0.00, "dcap": 0.00, "dshell": +0.02, "dtk": 0.00},
    {"dr": 0.00, "dseat": 0.00, "dblend": 0.00, "dcap": 0.00, "dshell": 0.00, "dtk": -0.04},
    {"dr": 0.00, "dseat": 0.00, "dblend": 0.00, "dcap": 0.00, "dshell": 0.00, "dtk": +0.04},
]


@dataclass(frozen=True)
class Params:
    radius_frac: float
    seat_axis_gain: float
    directional_blend: float
    norm_cap_mult: float
    shell_radial_gain: float
    tangent_kill: float

    def clipped(self) -> "Params":
        return Params(
            radius_frac=float(np.clip(self.radius_frac, 0.58, 0.86)),
            seat_axis_gain=float(np.clip(self.seat_axis_gain, 0.25, 0.60)),
            directional_blend=float(np.clip(self.directional_blend, 0.88, 1.20)),
            norm_cap_mult=float(np.clip(self.norm_cap_mult, 0.92, 1.24)),
            shell_radial_gain=float(np.clip(self.shell_radial_gain, 0.06, 0.20)),
            tangent_kill=float(np.clip(self.tangent_kill, 0.68, 0.96)),
        )


def params_to_env(p: Params) -> Dict[str, float]:
    return {
        "BOWL_RADIUS_FRAC": p.radius_frac,
        "BOWL_SEAT_AXIS_GAIN": p.seat_axis_gain,
        "BOWL_DIRECTIONAL_BLEND": p.directional_blend,
        "BOWL_NORM_CAP_MULT": p.norm_cap_mult,
        "BOWL_SHELL_RADIAL_GAIN": p.shell_radial_gain,
        "BOWL_TANGENT_KILL": p.tangent_kill,
    }


def apply_env(p: Params) -> None:
    for k, v in params_to_env(p).items():
        os.environ[k] = f"{v:.10g}"


def unset_env() -> None:
    for k in params_to_env(Params(0.7, 0.4, 1.0, 1.08, 0.12, 0.8)).keys():
        os.environ.pop(k, None)


def get_basins() -> Tuple[Any, Any, Any]:
    global BASINS_CACHE
    if BASINS_CACHE is None:
        BASINS_CACHE = DM.build_basins()
    return BASINS_CACHE


def set_eval_grid() -> None:
    for module in [BASE, DM, getattr(DM, "PREV", None), *list(getattr(DM, "MODS", []))]:
        if module is None:
            continue
        try:
            module.SIGMA_MULTS = list(SIGMA_MULTS)
            module.STRENGTHS = list(STRENGTHS)
            if hasattr(module, "RBF_SIGMA_MULTS"):
                module.RBF_SIGMA_MULTS = list(SIGMA_MULTS)
            if hasattr(module, "FORCE_STRENGTHS"):
                module.FORCE_STRENGTHS = list(STRENGTHS)
        except Exception:
            pass


def env_case_name(prefix: str, p: Params) -> str:
    return (
        f"{prefix}_r{p.radius_frac:.2f}_seat{p.seat_axis_gain:.2f}_"
        f"b{p.directional_blend:.2f}_cap{p.norm_cap_mult:.2f}_"
        f"sh{p.shell_radial_gain:.2f}_tk{p.tangent_kill:.2f}"
    )


def dn_scalar_score(best: Dict[str, float]) -> float:
    """DN score: reward capture + radial progress + alignment; punish tangent drift.

    26DM's score usefully found energetic ridges. DN now asks a narrower question:
    which ridge remains good when we care about not slipping sideways?
    """
    cap = float(best.get("capture_rate", 0.0))
    align = max(0.0, float(best.get("signed_alignment", best.get("alignment", 0.0))))
    prog = max(0.0, float(best.get("distance_progress", 0.0)))
    disp = max(0.0, float(best.get("mean_displacement", best.get("displacement", 0.0))))
    tan = float(best.get("tangent_ratio", 999.0))
    sm = float(best.get("sigma_mult", 0.0))
    strength = float(best.get("strength", 0.0))

    # Normalize common observed scales from the 26DL/DM runs.
    cap_term = cap / 0.30
    align_term = align / 0.70
    prog_term = prog / 0.0024
    disp_term = disp / 0.0070

    # Tangent ratios below ~10 are excellent, 10-30 tolerable, >40 increasingly suspect.
    tan_penalty = 1.0 / (1.0 + max(0.0, tan - 8.0) / 28.0)

    # Avoid over-selecting trivial weak fields, but do not force maximum strength.
    strength_bonus = 0.90 + 0.10 * min(1.0, strength / 0.45)

    # Mild preference for broad kernels that had good capture in DM, as long as tangent is low.
    sigma_bonus = 1.03 if sm >= 2.60 else 1.00

    raw = (0.38 * cap_term) + (0.22 * align_term) + (0.25 * prog_term) + (0.15 * disp_term)
    return float(raw * tan_penalty * strength_bonus * sigma_bonus)


def best_from_rows(rows: List[Dict]) -> Dict:
    best = None
    for r in rows:
        rr = dict(r)
        rr["dn_score"] = dn_scalar_score(rr)
        if best is None or rr["dn_score"] > best["dn_score"]:
            best = rr
    assert best is not None
    return best


def evaluate_one(case: str, p: Params) -> Dict:
    _z2, _rel_ids, basins = get_basins()
    set_eval_grid()
    apply_env(p)
    t0 = time.time()
    try:
        DM.apply(params_to_env(p))
        rows = BASE.evaluate_grid(basins)
    finally:
        DM.apply({})
        unset_env()
    best = best_from_rows(rows)
    out = {
        "case": case,
        "elapsed_sec": time.time() - t0,
        **params_to_env(p),
        "dn_score": float(best["dn_score"]),
        "capture_rate": float(best.get("capture_rate", 0.0)),
        "distance_progress": float(best.get("distance_progress", 0.0)),
        "signed_alignment": float(best.get("signed_alignment", best.get("alignment", 0.0))),
        "mean_displacement": float(best.get("mean_displacement", best.get("displacement", 0.0))),
        "tangent_ratio": float(best.get("tangent_ratio", np.nan)),
        "sigma_mult": float(best.get("sigma_mult", np.nan)),
        "strength": float(best.get("strength", np.nan)),
    }
    return out


def jitter_params(p: Params, j: Dict[str, float]) -> Params:
    return Params(
        p.radius_frac + j["dr"],
        p.seat_axis_gain + j["dseat"],
        p.directional_blend + j["dblend"],
        p.norm_cap_mult + j["dcap"],
        p.shell_radial_gain + j["dshell"],
        p.tangent_kill + j["dtk"],
    ).clipped()


def read_dm_candidates(limit: int = 16) -> List[Tuple[str, Params]]:
    csv_path = OUT_DIR / "phase26dm_case_results.csv"
    if not csv_path.exists():
        csv_path = ROOT / "phase26dm_case_results.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    if "score" in df.columns:
        df = df.sort_values("score", ascending=False)
    rows = []
    for _, r in df.head(limit).iterrows():
        p = Params(
            float(r["BOWL_RADIUS_FRAC"]),
            float(r["BOWL_SEAT_AXIS_GAIN"]),
            float(r["BOWL_DIRECTIONAL_BLEND"]),
            float(r["BOWL_NORM_CAP_MULT"]),
            float(r["BOWL_SHELL_RADIAL_GAIN"]),
            float(r["BOWL_TANGENT_KILL"]),
        ).clipped()
        rows.append((f"dm_top_{len(rows)+1:02d}_{str(r['case'])[:48]}", p))
    return rows


def unique_cases(cases: Iterable[Tuple[str, Params]]) -> List[Tuple[str, Params]]:
    seen = set()
    out = []
    for name, p in cases:
        key = tuple(round(v, 4) for v in asdict(p).values())
        if key in seen:
            continue
        seen.add(key)
        out.append((name, p))
    return out


def build_cases() -> List[Tuple[str, Params]]:
    cases: List[Tuple[str, Params]] = []
    cases.extend(read_dm_candidates(limit=18))

    # Exact/near-exact 26DM discoveries.
    cases.extend([
        ("dn_dm_best_stab_exact", Params(0.66, 0.35, 1.00, 1.00, 0.12, 0.90)),
        ("dn_dm_second_interact_exact", Params(0.78, 0.50, 1.12, 1.16, 0.16, 0.76)),
        ("dn_low_tangent_shell_016", Params(0.66, 0.35, 1.00, 1.00, 0.16, 0.90)),
        ("dn_low_tangent_tk092", Params(0.66, 0.35, 1.00, 1.00, 0.12, 0.92)),
        ("dn_low_tangent_tk094", Params(0.66, 0.35, 1.00, 1.00, 0.12, 0.94)),
        ("dn_interact_tk080", Params(0.78, 0.50, 1.12, 1.16, 0.16, 0.80)),
        ("dn_interact_tk084", Params(0.78, 0.50, 1.12, 1.16, 0.16, 0.84)),
        ("dn_interact_shell014_tk080", Params(0.78, 0.50, 1.12, 1.16, 0.14, 0.80)),
        ("dn_middle_stable", Params(0.70, 0.40, 1.04, 1.08, 0.12, 0.84)),
        ("dn_middle_shell", Params(0.74, 0.45, 1.08, 1.12, 0.14, 0.80)),
        ("dn_low_tangent_blend104", Params(0.66, 0.35, 1.04, 1.00, 0.12, 0.90)),
        ("dn_low_tangent_cap104", Params(0.66, 0.35, 1.00, 1.04, 0.12, 0.90)),
    ])
    return unique_cases(cases)


def audit_top(base_results: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    rows = []
    top = base_results.sort_values("dn_score", ascending=False).head(top_n)
    for _, row in top.iterrows():
        base_p = Params(
            row.BOWL_RADIUS_FRAC,
            row.BOWL_SEAT_AXIS_GAIN,
            row.BOWL_DIRECTIONAL_BLEND,
            row.BOWL_NORM_CAP_MULT,
            row.BOWL_SHELL_RADIAL_GAIN,
            row.BOWL_TANGENT_KILL,
        )
        for ji, j in enumerate(JITTERS):
            p = jitter_params(base_p, j)
            name = f"{row.case}__jit{ji:02d}"
            res = evaluate_one(name, p)
            res["parent_case"] = row.case
            res["jitter_index"] = ji
            rows.append(res)
            print(
                f"[{PHASE}] jitter {row.case[:42]:42s} #{ji:02d} "
                f"dn={res['dn_score']:.3f} cap={res['capture_rate']:.3f} "
                f"align={res['signed_alignment']:.3f} prog={res['distance_progress']:.5f} "
                f"tan={res['tangent_ratio']:.2f} sm={res['sigma_mult']:.2f} st={res['strength']:.2f}"
            )
    return pd.DataFrame(rows)


def summarize_stability(jitter_df: pd.DataFrame) -> pd.DataFrame:
    if jitter_df.empty:
        return pd.DataFrame()
    g = jitter_df.groupby("parent_case", as_index=False)
    s = g.agg(
        dn_mean=("dn_score", "mean"),
        dn_median=("dn_score", "median"),
        dn_min=("dn_score", "min"),
        dn_max=("dn_score", "max"),
        cap_mean=("capture_rate", "mean"),
        cap_min=("capture_rate", "min"),
        tan_mean=("tangent_ratio", "mean"),
        tan_max=("tangent_ratio", "max"),
        align_mean=("signed_alignment", "mean"),
        progress_mean=("distance_progress", "mean"),
    )
    # Conservative final score: robust mean with a hard floor term and tangent safety.
    s["stability_score"] = (
        0.50 * s["dn_mean"]
        + 0.30 * s["dn_median"]
        + 0.20 * s["dn_min"]
    ) / (1.0 + np.maximum(0.0, s["tan_max"] - 30.0) / 60.0)
    return s.sort_values("stability_score", ascending=False)


def save_top_plot(df: pd.DataFrame, path: Path, score_col: str, title: str) -> None:
    d = df.sort_values(score_col, ascending=True).tail(32)
    plt.figure(figsize=(11, max(7, 0.30 * len(d))))
    plt.barh(d["case"] if "case" in d.columns else d["parent_case"], d[score_col])
    plt.title(title)
    plt.xlabel(score_col)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_scatter(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(9, 7))
    sc = plt.scatter(df["capture_rate"], df["tangent_ratio"], c=df["dn_score"], s=70)
    plt.colorbar(sc, label="DN score")
    plt.xlabel("capture_rate")
    plt.ylabel("tangent/radial ratio")
    plt.title(f"{PHASE} capture vs tangent drift")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_stability_plot(stab: pd.DataFrame, path: Path) -> None:
    if stab.empty:
        return
    d = stab.sort_values("stability_score", ascending=True).tail(16)
    y = np.arange(len(d))
    plt.figure(figsize=(12, max(6, 0.38 * len(d))))
    plt.barh(y - 0.18, d["stability_score"], height=0.36, label="stability_score")
    plt.barh(y + 0.18, d["dn_min"], height=0.36, label="worst jitter DN")
    plt.yticks(y, d["parent_case"])
    plt.xlabel("score")
    plt.title(f"{PHASE} jitter stability audit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    print(json.dumps({
        "phase": PHASE,
        "title": TITLE,
        "sigma_mults": SIGMA_MULTS,
        "strengths": STRENGTHS,
        "root": str(ROOT),
        "out_dir": str(OUT_DIR),
        "note": "DN re-ranks DM ridges with explicit tangent/radial penalty, then jitters the top cases.",
    }, indent=2))

    cases = build_cases()
    print(f"[{PHASE}] Evaluating {len(cases)} base cases...")

    base_rows = []
    for i, (name, p) in enumerate(cases, 1):
        res = evaluate_one(name, p)
        base_rows.append(res)
        print(
            f"[{PHASE}] base {i:02d}/{len(cases):02d} {name[:58]:58s} "
            f"dn={res['dn_score']:.3f} cap={res['capture_rate']:.3f} "
            f"align={res['signed_alignment']:.3f} prog={res['distance_progress']:.5f} "
            f"tan={res['tangent_ratio']:.2f} sm={res['sigma_mult']:.2f} st={res['strength']:.2f}"
        )

    base_df = pd.DataFrame(base_rows).sort_values("dn_score", ascending=False)
    base_csv = OUT_DIR / "phase26dn_base_results.csv"
    base_df.to_csv(base_csv, index=False)

    print(f"[{PHASE}] Running jitter audit on top base cases...")
    jitter_df = audit_top(base_df, top_n=8)
    jitter_csv = OUT_DIR / "phase26dn_jitter_results.csv"
    jitter_df.to_csv(jitter_csv, index=False)

    stability_df = summarize_stability(jitter_df)
    stability_csv = OUT_DIR / "phase26dn_stability_summary.csv"
    stability_df.to_csv(stability_csv, index=False)

    save_top_plot(base_df, OUT_DIR / "phase26dn_base_top_scores.png", "dn_score", f"{PHASE} base DN low-tangent scores")
    save_scatter(base_df, OUT_DIR / "phase26dn_capture_vs_tangent.png")
    save_stability_plot(stability_df, OUT_DIR / "phase26dn_stability_scores.png")

    summary = {
        "phase": PHASE,
        "title": TITLE,
        "n_base_cases": int(len(base_df)),
        "n_jitter_cases": int(len(jitter_df)),
        "best_base": base_df.head(8).to_dict(orient="records"),
        "best_stability": stability_df.head(8).to_dict(orient="records") if not stability_df.empty else [],
        "outputs": {
            "base_csv": str(base_csv),
            "jitter_csv": str(jitter_csv),
            "stability_csv": str(stability_csv),
            "base_plot": str(OUT_DIR / "phase26dn_base_top_scores.png"),
            "scatter_plot": str(OUT_DIR / "phase26dn_capture_vs_tangent.png"),
            "stability_plot": str(OUT_DIR / "phase26dn_stability_scores.png"),
        },
    }
    summary_path = OUT_DIR / "phase26dn_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n[{PHASE}] DONE")
    print(f"[{PHASE}] Wrote: {base_csv}")
    print(f"[{PHASE}] Wrote: {jitter_csv}")
    print(f"[{PHASE}] Wrote: {stability_csv}")
    print(f"[{PHASE}] Wrote: {summary_path}")
    print("\nTop base cases:")
    print(base_df[["case", "dn_score", "capture_rate", "signed_alignment", "distance_progress", "tangent_ratio", "sigma_mult", "strength"]].head(12).to_string(index=False))
    if not stability_df.empty:
        print("\nTop stability cases:")
        print(stability_df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
