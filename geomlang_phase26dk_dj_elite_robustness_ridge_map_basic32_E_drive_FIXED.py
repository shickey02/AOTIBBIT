# geomlang_phase26dk_dj_elite_robustness_ridge_map_basic32_E_drive.py
"""
Phase 26DK — DJ elite robustness ridge-map

Purpose
-------
26DJ found a new best local-composition candidate:

    radius=0.66
    tangent_kill=0.90
    directional_blend=1.00
    shell_radial_gain=0.10
    norm_cap_mult=1.04
    seat_axis_gain=0.35

But the DJ table also showed that neighboring winners can come from slightly
different regimes, especially:

    - r ≈ 0.62–0.70
    - seat ≈ 0.25–0.40
    - blend ≈ 0.88–1.04
    - cap ≈ 1.00–1.08
    - shell ≈ 0.08–0.12
    - tk ≈ 0.84–0.96

DK therefore does NOT merely push the best score. It asks:

    1. Is the DJ best case stable under small perturbations?
    2. Is there a real ridge around r=0.66/cap=1.04/blend=1.00/seat=0.35?
    3. Can we find a slightly lower peak but better robust candidate?
    4. Which parameters are actually carrying the improvement?

Outputs
-------
E:/BBIT/outputs_basic32/phase26dk_case_results.csv
E:/BBIT/outputs_basic32/phase26dk_summary.json
E:/BBIT/outputs_basic32/phase26dk_top_scores.png
E:/BBIT/outputs_basic32/phase26dk_ridge_heatmaps.png
E:/BBIT/outputs_basic32/phase26dk_parameter_slices.png

Notes
-----
This script is designed to reuse the Phase 26DI/DJ harness instead of rewriting
the full BBIT rollout stack. It tries the most likely exported evaluator names.
If your local 26DJ/26DI file used a different function name, the adapter will
print all available callable names so we can patch DK quickly.
"""

from __future__ import annotations

import copy
import csv
import importlib.util
import inspect
import itertools
import json
import math
import os
import random
import sys
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Paths
# =============================================================================

PHASE = "26DK"
TITLE = "DJ elite robustness ridge-map"

ROOT = Path(r"E:\BBIT")
SCRIPT_DIR = ROOT / "bbit_geomlang"
OUT_DIR = ROOT / "outputs_basic32"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DJ_PATH = SCRIPT_DIR / "geomlang_phase26dj_di_elite_local_composition_basic32_E_drive.py"
DI_PATH = SCRIPT_DIR / "geomlang_phase26di_dh_winner_combination_lattice_basic32_E_drive.py"

DJ_SUMMARY_PATH = OUT_DIR / "phase26dj_summary.json"
DJ_CASE_CSV_PATH = OUT_DIR / "phase26dj_case_results.csv"

OUT_CSV = OUT_DIR / "phase26dk_case_results.csv"
OUT_JSON = OUT_DIR / "phase26dk_summary.json"
OUT_TOP_PNG = OUT_DIR / "phase26dk_top_scores.png"
OUT_HEATMAP_PNG = OUT_DIR / "phase26dk_ridge_heatmaps.png"
OUT_SLICES_PNG = OUT_DIR / "phase26dk_parameter_slices.png"


# =============================================================================
# DJ best seed and DK search space
# =============================================================================

BASELINE_CONSTANTS = {
    "BOWL_DIRECTIONAL_BLEND": 0.62,
    "BOWL_FORWARD_COMMIT_GAIN": 0.28,
    "BOWL_FORWARD_COMMIT_RING_GAIN": 0.18,
    "BOWL_GATE_FLOOR": 0.18,
    "BOWL_NORM_CAP_MULT": 1.04,
    "BOWL_PROGRESS_SHARPNESS": 9.5,
    "BOWL_PULL_GAIN": 0.14,
    "BOWL_RADIAL_SINK_GAIN": 0.14,
    "BOWL_RADIUS_FRAC": 0.78,
    "BOWL_RING_FRAC": 0.76,
    "BOWL_SEAT_AXIS_GAIN": 0.30,
    "BOWL_SHELL_RADIAL_GAIN": 0.06,
    "BOWL_TANGENT_KILL": 0.20,
    "BOWL_TANGENT_RATIO_CENTER": 0.88,
}

DJ_BEST = {
    "BOWL_RADIUS_FRAC": 0.66,
    "BOWL_TANGENT_KILL": 0.90,
    "BOWL_DIRECTIONAL_BLEND": 1.00,
    "BOWL_SHELL_RADIAL_GAIN": 0.10,
    "BOWL_NORM_CAP_MULT": 1.04,
    "BOWL_SEAT_AXIS_GAIN": 0.35,
}

# These are deliberately local around the DJ ridge.
RADIUS_VALUES = [0.62, 0.64, 0.65, 0.66, 0.67, 0.68, 0.70]
SEAT_VALUES = [0.25, 0.30, 0.325, 0.35, 0.375, 0.40]
BLEND_VALUES = [0.88, 0.92, 0.96, 0.98, 1.00, 1.02, 1.04]
CAP_VALUES = [1.00, 1.02, 1.04, 1.06, 1.08]
SHELL_VALUES = [0.08, 0.10, 0.12]
TK_VALUES = [0.84, 0.88, 0.90, 0.92, 0.96]

# Runtime parameters.
# Keep this manageable first. If DK finds a clean ridge, DL can do the expensive repeat pass.
MAX_CASES = 260
TOP_N_PRINT = 40
TOP_N_PLOT = 80

# Optional robustness repeats.
# If the imported harness is deterministic, these simply repeat the same value.
# If it samples rollout starts internally, this gives us a stability estimate.
N_REPEATS_PER_CASE = 3
RANDOM_SEEDS = [11, 17, 23]


# =============================================================================
# Utility
# =============================================================================

def load_module(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"Cannot find module: {path}")

    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load spec for: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_csv_dicts(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def flatten_metric_value(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float, np.floating)):
        if math.isfinite(float(x)):
            return float(x)
        return None
    if isinstance(x, str):
        try:
            v = float(x)
            if math.isfinite(v):
                return v
        except Exception:
            return None
    return None


def safe_mean(xs: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if not vals:
        return None
    return float(np.mean(vals))


def safe_std(xs: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    if len(vals) < 2:
        return 0.0 if len(vals) == 1 else None
    return float(np.std(vals, ddof=1))


def robust_score(mean_score: float, std_score: float, n: int) -> float:
    """
    Conservative lower-confidence-ish score.

    We do not over-formalize this because the repeats are not necessarily
    independent statistical trials. This is a practical penalty:
        mean - 0.75 * std

    The aim is to demote spiky one-off winners without hiding strong peaks.
    """
    if n <= 1:
        return mean_score
    return mean_score - 0.75 * std_score


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# =============================================================================
# Harness adapter
# =============================================================================

class HarnessAdapter:
    """
    Adapter around the existing DI/DJ evaluator.

    This avoids duplicating the entire Phase 26 capture system.
    """

    POSSIBLE_EVALUATOR_NAMES = [
        "evaluate_case",
        "eval_case",
        "evaluate_candidate",
        "eval_candidate",
        "score_case",
        "score_candidate",
        "run_case",
        "run_candidate",
        "evaluate_overrides",
        "eval_overrides",
    ]

    POSSIBLE_CONTEXT_NAMES = [
        "build_context",
        "make_context",
        "load_context",
        "prepare_context",
        "init_context",
        "load_runtime_context",
        "setup_runtime",
    ]

    def __init__(self):
        print(f"[{PHASE}] Loading DJ harness from: {DJ_PATH}")
        try:
            self.mod = load_module(DJ_PATH, "phase26dj_harness")
            self.source_path = DJ_PATH
        except Exception as e_dj:
            print(f"[{PHASE}] Could not import DJ harness. Falling back to DI.")
            print(f"[{PHASE}] DJ import error: {repr(e_dj)}")
            print(f"[{PHASE}] Loading DI harness from: {DI_PATH}")
            self.mod = load_module(DI_PATH, "phase26di_harness")
            self.source_path = DI_PATH

        self.context = self._try_make_context()
        self.evaluator = self._find_evaluator()

        print(f"[{PHASE}] Harness source: {self.source_path}")
        print(f"[{PHASE}] Evaluator: {self.evaluator.__name__}")
        if self.context is not None:
            print(f"[{PHASE}] Context object created: {type(self.context).__name__}")
        else:
            print(f"[{PHASE}] No explicit context object found; using module globals.")

    def _try_make_context(self) -> Any:
        """
        Build the runtime object required by the imported evaluator.

        Important DK patch:
        DJ exposes eval_case(name, overrides, basins), but it does not expose a
        generic build_context() wrapper. Its basins are created in DJ.main() via
        DI.build_basins(). The original DK adapter therefore found DJ.eval_case
        correctly but had no `basins` argument to pass, causing the harness call
        to fail before any ridge-map cases could run.

        This method now tries explicit basin builders after the generic context
        names. If build_basins() returns the common tuple (z2, rel_ids, basins),
        we keep only the third item as the evaluator context.
        """
        for name in self.POSSIBLE_CONTEXT_NAMES:
            fn = getattr(self.mod, name, None)
            if callable(fn):
                try:
                    print(f"[{PHASE}] Trying context builder: {name}()")
                    return fn()
                except TypeError:
                    # Try common no-heavy-plot / quiet signatures.
                    try:
                        return fn(plot=False)
                    except Exception:
                        pass
                except Exception:
                    print(f"[{PHASE}] Context builder {name} failed:")
                    traceback.print_exc(limit=2)

        # Explicit DJ/DI basin fallback. DJ imports the DI harness as `DI`, and
        # DJ.main() itself calls `DI.build_basins()`. Recreate that here instead
        # of calling DJ.main(), which would rerun the whole DJ phase and make
        # plots/files unnecessarily.
        basin_builders = []
        if callable(getattr(self.mod, "build_basins", None)):
            basin_builders.append(("module.build_basins", getattr(self.mod, "build_basins")))

        di_mod = getattr(self.mod, "DI", None)
        if di_mod is not None and callable(getattr(di_mod, "build_basins", None)):
            basin_builders.append(("module.DI.build_basins", getattr(di_mod, "build_basins")))

        for label, fn in basin_builders:
            try:
                print(f"[{PHASE}] Trying basin builder: {label}()")
                built = fn()
                if isinstance(built, tuple) and len(built) >= 3:
                    print(f"[{PHASE}] Built basins from {label}: tuple len={len(built)}; using item[2]")
                    return built[2]
                print(f"[{PHASE}] Built context from {label}: {type(built).__name__}")
                return built
            except Exception:
                print(f"[{PHASE}] Basin builder {label} failed:")
                traceback.print_exc(limit=2)

        return None

    def _find_evaluator(self) -> Callable:
        for name in self.POSSIBLE_EVALUATOR_NAMES:
            fn = getattr(self.mod, name, None)
            if callable(fn):
                return fn

        print(f"\n[{PHASE}] ERROR: Could not find an evaluator in imported module.")
        print(f"[{PHASE}] Available callables:")
        for k, v in sorted(vars(self.mod).items()):
            if callable(v) and not k.startswith("_"):
                try:
                    sig = str(inspect.signature(v))
                except Exception:
                    sig = "(signature unavailable)"
                print(f"  - {k}{sig}")
        raise RuntimeError("No compatible evaluator found. Paste the callable list back and I will patch DK.")

    def evaluate(self, case_name: str, overrides: Dict[str, float], seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            set_global_seed(seed)

        fn = self.evaluator
        sig = inspect.signature(fn)
        params = sig.parameters

        # Try keyword-driven call first.
        kwargs = {}
        for p in params:
            pl = p.lower()
            if pl in ["case", "case_name", "name", "label"]:
                kwargs[p] = case_name
            elif pl in ["overrides", "override", "constants", "const_overrides", "params"]:
                kwargs[p] = overrides
            elif pl in ["context", "ctx", "runtime", "state", "basins", "basin", "basin_set"]:
                kwargs[p] = self.context
            elif pl in ["seed", "random_seed"]:
                kwargs[p] = seed
            elif pl in ["plot", "make_plots", "save_plots"]:
                kwargs[p] = False
            elif pl in ["quiet", "verbose"]:
                kwargs[p] = True if pl == "quiet" else False

        try:
            if kwargs:
                result = fn(**kwargs)
            else:
                result = fn(case_name, overrides)
            return normalize_result(case_name, overrides, result)
        except TypeError as e_kw:
            # Try common positional variants.
            attempts = []
            if self.context is not None:
                attempts.extend([
                    (case_name, overrides, self.context),
                    (overrides, self.context),
                    (self.context, case_name, overrides),
                    (self.context, overrides),
                ])
            attempts.extend([
                (case_name, overrides),
                (overrides,),
            ])

            last_err = e_kw
            for args in attempts:
                try:
                    result = fn(*args)
                    return normalize_result(case_name, overrides, result)
                except TypeError as e:
                    last_err = e
                except Exception:
                    raise

            print(f"\n[{PHASE}] Evaluator signature could not be called.")
            print(f"[{PHASE}] Function: {fn.__name__}{sig}")
            print(f"[{PHASE}] Keyword attempt: {kwargs}")
            raise last_err


def normalize_result(case_name: str, overrides: Dict[str, float], result: Any) -> Dict[str, Any]:
    """
    Accepts the different result shapes the previous phase scripts may return.
    """

    out = {
        "case": case_name,
        "overrides": copy.deepcopy(overrides),
        "raw_result_type": type(result).__name__,
    }

    if isinstance(result, dict):
        out.update(result)

        # Common nested summary format.
        if "best_summary" in result and isinstance(result["best_summary"], dict):
            for k, v in result["best_summary"].items():
                out.setdefault(k, v)

        # Common score aliases.
        for key in [
            "dk_scalar_score",
            "dj_scalar_score",
            "di_scalar_score",
            "scalar_score",
            "score",
            "best_score",
        ]:
            if key in result:
                out["score"] = flatten_metric_value(result[key])
                break

    elif isinstance(result, (tuple, list)):
        # Common pattern: (score, summary)
        if len(result) >= 1:
            out["score"] = flatten_metric_value(result[0])
        if len(result) >= 2 and isinstance(result[1], dict):
            out.update(result[1])
            if "best_summary" in result[1] and isinstance(result[1]["best_summary"], dict):
                for k, v in result[1]["best_summary"].items():
                    out.setdefault(k, v)

    else:
        out["score"] = flatten_metric_value(result)

    # Normalize score from any aliases if not found.
    if out.get("score") is None:
        for key in [
            "dk_scalar_score",
            "dj_scalar_score",
            "di_scalar_score",
            "scalar_score",
            "best_score",
        ]:
            if key in out:
                out["score"] = flatten_metric_value(out[key])
                break

    # Normalize key metrics.
    for key in ["capture_rate", "distance_progress", "tangent_ratio", "sigma_mult", "strength"]:
        if key in out:
            out[key] = flatten_metric_value(out[key])

    if out.get("score") is None:
        raise RuntimeError(
            f"Evaluator returned a result, but no score could be extracted for {case_name}. "
            f"Result type={type(result).__name__}, result={repr(result)[:1000]}"
        )

    return out


# =============================================================================
# Candidate generation
# =============================================================================

@dataclass
class Candidate:
    case: str
    overrides: Dict[str, float]
    family: str


def fmt(x: float) -> str:
    return f"{x:.3f}".rstrip("0").rstrip(".")


def make_candidate_name(prefix: str, overrides: Dict[str, float]) -> str:
    r = overrides["BOWL_RADIUS_FRAC"]
    seat = overrides["BOWL_SEAT_AXIS_GAIN"]
    blend = overrides["BOWL_DIRECTIONAL_BLEND"]
    cap = overrides["BOWL_NORM_CAP_MULT"]
    shell = overrides["BOWL_SHELL_RADIAL_GAIN"]
    tk = overrides["BOWL_TANGENT_KILL"]
    return (
        f"{prefix}_r{r:.3f}_seat{seat:.3f}_blend{blend:.3f}_"
        f"cap{cap:.3f}_shell{shell:.3f}_tk{tk:.3f}"
    ).replace(".", "p")


def add_candidate(cands: List[Candidate], seen: set, family: str, overrides: Dict[str, float], prefix: str):
    key = tuple(sorted((k, round(float(v), 6)) for k, v in overrides.items()))
    if key in seen:
        return
    seen.add(key)
    cands.append(Candidate(
        case=make_candidate_name(prefix, overrides),
        overrides=copy.deepcopy(overrides),
        family=family,
    ))


def build_candidates() -> List[Candidate]:
    """
    Build a focused set instead of a huge full factorial grid.

    Structure:
    1. Include exact DJ best.
    2. One-axis slices around DJ best.
    3. Two-axis radius/blend/seat/cap ridge.
    4. Shell/tangent-kill stabilizer probes.
    5. Known DJ neighboring elites.
    """

    cands: List[Candidate] = []
    seen = set()

    # 1. Exact best.
    add_candidate(cands, seen, "dj_exact_best", DJ_BEST, "dk_exact")

    # 2. One-axis slices.
    for r in RADIUS_VALUES:
        o = copy.deepcopy(DJ_BEST)
        o["BOWL_RADIUS_FRAC"] = r
        add_candidate(cands, seen, "radius_slice", o, "dk_rslice")

    for seat in SEAT_VALUES:
        o = copy.deepcopy(DJ_BEST)
        o["BOWL_SEAT_AXIS_GAIN"] = seat
        add_candidate(cands, seen, "seat_slice", o, "dk_seatslice")

    for blend in BLEND_VALUES:
        o = copy.deepcopy(DJ_BEST)
        o["BOWL_DIRECTIONAL_BLEND"] = blend
        add_candidate(cands, seen, "blend_slice", o, "dk_blendslice")

    for cap in CAP_VALUES:
        o = copy.deepcopy(DJ_BEST)
        o["BOWL_NORM_CAP_MULT"] = cap
        add_candidate(cands, seen, "cap_slice", o, "dk_capslice")

    for shell in SHELL_VALUES:
        o = copy.deepcopy(DJ_BEST)
        o["BOWL_SHELL_RADIAL_GAIN"] = shell
        add_candidate(cands, seen, "shell_slice", o, "dk_shellslice")

    for tk in TK_VALUES:
        o = copy.deepcopy(DJ_BEST)
        o["BOWL_TANGENT_KILL"] = tk
        add_candidate(cands, seen, "tk_slice", o, "dk_tkslice")

    # 3. Main ridge: radius x seat x blend, keeping cap/shell/tk at DJ best.
    for r, seat, blend in itertools.product(
        [0.64, 0.65, 0.66, 0.67, 0.68, 0.70],
        [0.30, 0.325, 0.35, 0.375],
        [0.96, 0.98, 1.00, 1.02],
    ):
        o = copy.deepcopy(DJ_BEST)
        o["BOWL_RADIUS_FRAC"] = r
        o["BOWL_SEAT_AXIS_GAIN"] = seat
        o["BOWL_DIRECTIONAL_BLEND"] = blend
        add_candidate(cands, seen, "radius_seat_blend_ridge", o, "dk_rsb")

    # 4. Cap/blend coupling around the exact best.
    for cap, blend in itertools.product([1.00, 1.02, 1.04, 1.06, 1.08], [0.96, 0.98, 1.00, 1.02, 1.04]):
        o = copy.deepcopy(DJ_BEST)
        o["BOWL_NORM_CAP_MULT"] = cap
        o["BOWL_DIRECTIONAL_BLEND"] = blend
        add_candidate(cands, seen, "cap_blend_coupling", o, "dk_capblend")

    # 5. Stabilizer probes: shell/tk around best and around r=.70 seat=.35.
    anchors = [
        ("best_anchor", DJ_BEST),
        ("r70_anchor", {
            "BOWL_RADIUS_FRAC": 0.70,
            "BOWL_TANGENT_KILL": 0.90,
            "BOWL_DIRECTIONAL_BLEND": 1.00,
            "BOWL_SHELL_RADIAL_GAIN": 0.10,
            "BOWL_NORM_CAP_MULT": 1.00,
            "BOWL_SEAT_AXIS_GAIN": 0.35,
        }),
        ("r62_capture_anchor", {
            "BOWL_RADIUS_FRAC": 0.62,
            "BOWL_TANGENT_KILL": 0.90,
            "BOWL_DIRECTIONAL_BLEND": 0.92,
            "BOWL_SHELL_RADIAL_GAIN": 0.10,
            "BOWL_NORM_CAP_MULT": 1.00,
            "BOWL_SEAT_AXIS_GAIN": 0.25,
        }),
    ]

    for anchor_name, anchor in anchors:
        for shell, tk, cap in itertools.product([0.08, 0.10, 0.12], [0.84, 0.88, 0.90, 0.92, 0.96], [1.00, 1.04, 1.08]):
            o = copy.deepcopy(anchor)
            o["BOWL_SHELL_RADIAL_GAIN"] = shell
            o["BOWL_TANGENT_KILL"] = tk
            o["BOWL_NORM_CAP_MULT"] = cap
            add_candidate(cands, seen, f"stabilizer_{anchor_name}", o, "dk_stab")

    # Deterministic priority: keep all slices, then sample ridge if too large.
    if len(cands) > MAX_CASES:
        required_fams = {
            "dj_exact_best",
            "radius_slice",
            "seat_slice",
            "blend_slice",
            "cap_slice",
            "shell_slice",
            "tk_slice",
        }
        required = [c for c in cands if c.family in required_fams]
        optional = [c for c in cands if c.family not in required_fams]

        rng = random.Random(2604)
        rng.shuffle(optional)
        cands = required + optional[: max(0, MAX_CASES - len(required))]

    return cands


# =============================================================================
# Scoring pass
# =============================================================================

def summarize_repeats(case: Candidate, repeat_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = [flatten_metric_value(r.get("score")) for r in repeat_results]
    captures = [flatten_metric_value(r.get("capture_rate")) for r in repeat_results]
    progresses = [flatten_metric_value(r.get("distance_progress")) for r in repeat_results]
    tangents = [flatten_metric_value(r.get("tangent_ratio")) for r in repeat_results]

    mean_score = safe_mean(scores)
    std_score = safe_std(scores)
    n_ok = sum(s is not None for s in scores)

    if mean_score is None:
        raise RuntimeError(f"No valid scores for case {case.case}")

    std_score = 0.0 if std_score is None else std_score

    out = {
        "phase": PHASE,
        "case": case.case,
        "family": case.family,
        "score_mean": mean_score,
        "score_std": std_score,
        "score_min": float(np.min([s for s in scores if s is not None])),
        "score_max": float(np.max([s for s in scores if s is not None])),
        "score_robust": robust_score(mean_score, std_score, n_ok),
        "n_repeats": n_ok,

        "capture_rate_mean": safe_mean(captures),
        "capture_rate_std": safe_std(captures),
        "distance_progress_mean": safe_mean(progresses),
        "distance_progress_std": safe_std(progresses),
        "tangent_ratio_mean": safe_mean(tangents),
        "tangent_ratio_std": safe_std(tangents),

        "BOWL_RADIUS_FRAC": case.overrides["BOWL_RADIUS_FRAC"],
        "BOWL_TANGENT_KILL": case.overrides["BOWL_TANGENT_KILL"],
        "BOWL_DIRECTIONAL_BLEND": case.overrides["BOWL_DIRECTIONAL_BLEND"],
        "BOWL_SHELL_RADIAL_GAIN": case.overrides["BOWL_SHELL_RADIAL_GAIN"],
        "BOWL_NORM_CAP_MULT": case.overrides["BOWL_NORM_CAP_MULT"],
        "BOWL_SEAT_AXIS_GAIN": case.overrides["BOWL_SEAT_AXIS_GAIN"],

        "overrides_json": json.dumps(case.overrides, sort_keys=True),
        "repeat_results_json": json.dumps(repeat_results, default=str),
    }

    # Carry through best sigma/strength by mean-winning repeat.
    best_i = int(np.argmax([s if s is not None else -1e9 for s in scores]))
    best_rep = repeat_results[best_i]
    out["best_repeat_seed"] = best_rep.get("seed")
    out["best_repeat_score"] = flatten_metric_value(best_rep.get("score"))
    out["best_sigma_mult"] = flatten_metric_value(best_rep.get("sigma_mult"))
    out["best_strength"] = flatten_metric_value(best_rep.get("strength"))

    return out


def run_cases() -> List[Dict[str, Any]]:
    adapter = HarnessAdapter()
    candidates = build_candidates()

    print(f"\n[{PHASE}] {TITLE}")
    print(f"[{PHASE}] Candidate count: {len(candidates)}")
    print(f"[{PHASE}] Repeats per case: {N_REPEATS_PER_CASE}")
    print(f"[{PHASE}] Seeds: {RANDOM_SEEDS}")
    print("")

    rows: List[Dict[str, Any]] = []

    for i, cand in enumerate(candidates, 1):
        repeat_results = []

        for j in range(N_REPEATS_PER_CASE):
            seed = RANDOM_SEEDS[j % len(RANDOM_SEEDS)]
            try:
                result = adapter.evaluate(cand.case, cand.overrides, seed=seed)
                result["seed"] = seed
                repeat_results.append(result)
            except Exception as e:
                print(f"[{PHASE}] ERROR evaluating {cand.case} seed={seed}: {repr(e)}")
                traceback.print_exc(limit=3)

        if not repeat_results:
            print(f"[{PHASE}] {i:04d}/{len(candidates)} {cand.case:72s} FAILED")
            continue

        row = summarize_repeats(cand, repeat_results)
        rows.append(row)

        print(
            f"[{PHASE}] {i:04d}/{len(candidates)} "
            f"{cand.case[:72]:72s} "
            f"mean={row['score_mean']:.8f} "
            f"robust={row['score_robust']:.8f} "
            f"std={row['score_std']:.5f} "
            f"cap={row['capture_rate_mean'] if row['capture_rate_mean'] is not None else float('nan'):.4f} "
            f"tan={row['tangent_ratio_mean'] if row['tangent_ratio_mean'] is not None else float('nan'):.3f}"
        )

    rows.sort(key=lambda r: r["score_robust"], reverse=True)
    return rows


# =============================================================================
# Output
# =============================================================================

def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def make_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    dj_summary = read_json(DJ_SUMMARY_PATH)

    best = rows[0] if rows else None
    top_15 = rows[:15]

    summary = {
        "phase": PHASE,
        "title": TITLE,
        "description": (
            "Robustness and ridge-map pass around the 26DJ elite local-composition family. "
            "Ranks candidates by mean score penalized by repeat instability."
        ),
        "n_cases_evaluated": len(rows),
        "n_repeats_per_case": N_REPEATS_PER_CASE,
        "random_seeds": RANDOM_SEEDS,
        "dj_best_seed": DJ_BEST,
        "max_cases": MAX_CASES,
        "best_case": compact_row(best) if best else None,
        "top_15_cases": [compact_row(r) for r in top_15],
        "family_summary": family_summary(rows),
        "parameter_summary": parameter_summary(rows),
        "paths": {
            "csv": str(OUT_CSV),
            "summary_json": str(OUT_JSON),
            "top_scores_png": str(OUT_TOP_PNG),
            "ridge_heatmaps_png": str(OUT_HEATMAP_PNG),
            "parameter_slices_png": str(OUT_SLICES_PNG),
        },
    }

    if dj_summary:
        summary["phase26dj_best_case"] = dj_summary.get("best_case")
        summary["phase26dj_baseline_score"] = dj_summary.get("baseline_score")

    return summary


def compact_row(r: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if r is None:
        return None

    keys = [
        "case",
        "family",
        "score_mean",
        "score_std",
        "score_robust",
        "score_min",
        "score_max",
        "capture_rate_mean",
        "distance_progress_mean",
        "tangent_ratio_mean",
        "best_sigma_mult",
        "best_strength",
        "BOWL_RADIUS_FRAC",
        "BOWL_TANGENT_KILL",
        "BOWL_DIRECTIONAL_BLEND",
        "BOWL_SHELL_RADIAL_GAIN",
        "BOWL_NORM_CAP_MULT",
        "BOWL_SEAT_AXIS_GAIN",
    ]
    return {k: r.get(k) for k in keys}


def family_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {}
    fams = sorted(set(r["family"] for r in rows))
    for fam in fams:
        xs = [r for r in rows if r["family"] == fam]
        xs_sorted = sorted(xs, key=lambda r: r["score_robust"], reverse=True)
        out[fam] = {
            "n": len(xs),
            "best": compact_row(xs_sorted[0]),
            "mean_robust": float(np.mean([r["score_robust"] for r in xs])),
            "mean_score": float(np.mean([r["score_mean"] for r in xs])),
        }
    return out


def parameter_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    params = [
        "BOWL_RADIUS_FRAC",
        "BOWL_TANGENT_KILL",
        "BOWL_DIRECTIONAL_BLEND",
        "BOWL_SHELL_RADIAL_GAIN",
        "BOWL_NORM_CAP_MULT",
        "BOWL_SEAT_AXIS_GAIN",
    ]

    out = {}
    for p in params:
        values = sorted(set(round(float(r[p]), 6) for r in rows))
        vals = []
        for v in values:
            xs = [r for r in rows if round(float(r[p]), 6) == v]
            vals.append({
                "value": v,
                "n": len(xs),
                "mean_robust": float(np.mean([r["score_robust"] for r in xs])),
                "best_robust": float(np.max([r["score_robust"] for r in xs])),
                "best_case": max(xs, key=lambda r: r["score_robust"])["case"],
            })
        out[p] = vals
    return out


# =============================================================================
# Plots
# =============================================================================

def plot_top_scores(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    top = rows[: min(TOP_N_PLOT, len(rows))]
    labels = [r["case"] for r in top][::-1]
    vals = [r["score_robust"] for r in top][::-1]
    means = [r["score_mean"] for r in top][::-1]

    plt.figure(figsize=(12, max(8, 0.22 * len(top))))
    y = np.arange(len(top))
    plt.barh(y, vals, label="robust score")
    plt.plot(means, y, "o", markersize=3, label="mean score")
    plt.yticks(y, labels, fontsize=7)
    plt.xlabel("DK score")
    plt.title(f"{PHASE} robust elite scores — top {len(top)}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_TOP_PNG, dpi=170)
    plt.close()


def pivot_best(
    rows: List[Dict[str, Any]],
    x_key: str,
    y_key: str,
    value_key: str = "score_robust",
    fixed: Optional[Dict[str, float]] = None,
) -> Tuple[List[float], List[float], np.ndarray]:
    fixed = fixed or {}

    filtered = []
    for r in rows:
        ok = True
        for k, v in fixed.items():
            if abs(float(r[k]) - float(v)) > 1e-9:
                ok = False
                break
        if ok:
            filtered.append(r)

    xs = sorted(set(float(r[x_key]) for r in filtered))
    ys = sorted(set(float(r[y_key]) for r in filtered))

    mat = np.full((len(ys), len(xs)), np.nan, dtype=float)

    for yi, yv in enumerate(ys):
        for xi, xv in enumerate(xs):
            cell = [r for r in filtered if float(r[x_key]) == xv and float(r[y_key]) == yv]
            if cell:
                mat[yi, xi] = max(float(r[value_key]) for r in cell)

    return xs, ys, mat


def draw_heat(ax, xs, ys, mat, title, xlabel, ylabel):
    im = ax.imshow(mat, aspect="auto", origin="lower")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(xs)))
    ax.set_xticklabels([fmt(x) for x in xs], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(ys)))
    ax.set_yticklabels([fmt(y) for y in ys])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_ridge_heatmaps(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    xs, ys, mat = pivot_best(
        rows,
        "BOWL_RADIUS_FRAC",
        "BOWL_SEAT_AXIS_GAIN",
        fixed={
            "BOWL_DIRECTIONAL_BLEND": 1.00,
            "BOWL_NORM_CAP_MULT": 1.04,
            "BOWL_SHELL_RADIAL_GAIN": 0.10,
            "BOWL_TANGENT_KILL": 0.90,
        },
    )
    draw_heat(
        axes[0, 0],
        xs, ys, mat,
        "Best robust score: radius x seat\nfixed blend=1.00 cap=1.04 shell=.10 tk=.90",
        "BOWL_RADIUS_FRAC",
        "BOWL_SEAT_AXIS_GAIN",
    )

    xs, ys, mat = pivot_best(
        rows,
        "BOWL_DIRECTIONAL_BLEND",
        "BOWL_NORM_CAP_MULT",
        fixed={
            "BOWL_RADIUS_FRAC": 0.66,
            "BOWL_SEAT_AXIS_GAIN": 0.35,
            "BOWL_SHELL_RADIAL_GAIN": 0.10,
            "BOWL_TANGENT_KILL": 0.90,
        },
    )
    draw_heat(
        axes[0, 1],
        xs, ys, mat,
        "Best robust score: blend x cap\nfixed r=.66 seat=.35 shell=.10 tk=.90",
        "BOWL_DIRECTIONAL_BLEND",
        "BOWL_NORM_CAP_MULT",
    )

    xs, ys, mat = pivot_best(
        rows,
        "BOWL_TANGENT_KILL",
        "BOWL_SHELL_RADIAL_GAIN",
        fixed={
            "BOWL_RADIUS_FRAC": 0.66,
            "BOWL_SEAT_AXIS_GAIN": 0.35,
            "BOWL_DIRECTIONAL_BLEND": 1.00,
            "BOWL_NORM_CAP_MULT": 1.04,
        },
    )
    draw_heat(
        axes[1, 0],
        xs, ys, mat,
        "Best robust score: tangent kill x shell\nfixed r=.66 seat=.35 blend=1.00 cap=1.04",
        "BOWL_TANGENT_KILL",
        "BOWL_SHELL_RADIAL_GAIN",
    )

    xs, ys, mat = pivot_best(
        rows,
        "BOWL_DIRECTIONAL_BLEND",
        "BOWL_SEAT_AXIS_GAIN",
        fixed={
            "BOWL_RADIUS_FRAC": 0.66,
            "BOWL_NORM_CAP_MULT": 1.04,
            "BOWL_SHELL_RADIAL_GAIN": 0.10,
            "BOWL_TANGENT_KILL": 0.90,
        },
    )
    draw_heat(
        axes[1, 1],
        xs, ys, mat,
        "Best robust score: blend x seat\nfixed r=.66 cap=1.04 shell=.10 tk=.90",
        "BOWL_DIRECTIONAL_BLEND",
        "BOWL_SEAT_AXIS_GAIN",
    )

    fig.suptitle(f"{PHASE} ridge heatmaps", fontsize=18)
    plt.tight_layout()
    plt.savefig(OUT_HEATMAP_PNG, dpi=170)
    plt.close()


def plot_parameter_slices(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    params = [
        ("BOWL_RADIUS_FRAC", "radius"),
        ("BOWL_SEAT_AXIS_GAIN", "seat axis gain"),
        ("BOWL_DIRECTIONAL_BLEND", "directional blend"),
        ("BOWL_NORM_CAP_MULT", "norm cap mult"),
        ("BOWL_SHELL_RADIAL_GAIN", "shell radial gain"),
        ("BOWL_TANGENT_KILL", "tangent kill"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    for ax, (p, title) in zip(axes, params):
        values = sorted(set(float(r[p]) for r in rows))
        mean_vals = []
        best_vals = []

        for v in values:
            xs = [r for r in rows if float(r[p]) == v]
            mean_vals.append(float(np.mean([r["score_robust"] for r in xs])))
            best_vals.append(float(np.max([r["score_robust"] for r in xs])))

        ax.plot(values, mean_vals, marker="o", label="mean robust")
        ax.plot(values, best_vals, marker="o", label="best robust")
        ax.set_title(title)
        ax.set_xlabel(p)
        ax.set_ylabel("score")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle(f"{PHASE} parameter slices", fontsize=18)
    plt.tight_layout()
    plt.savefig(OUT_SLICES_PNG, dpi=170)
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 88)
    print(f"{PHASE} — {TITLE}")
    print("=" * 88)

    if DJ_SUMMARY_PATH.exists():
        dj = read_json(DJ_SUMMARY_PATH)
        if dj and dj.get("best_case"):
            print(f"[{PHASE}] Loaded previous DJ summary.")
            print(f"[{PHASE}] DJ best case: {dj['best_case'].get('case')}")
            print(f"[{PHASE}] DJ best score: {dj['best_case'].get('score')}")
    else:
        print(f"[{PHASE}] No previous DJ summary found at {DJ_SUMMARY_PATH}; continuing anyway.")

    rows = run_cases()

    if not rows:
        raise RuntimeError("No successful DK evaluations.")

    write_csv(rows, OUT_CSV)
    summary = make_summary(rows)
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_top_scores(rows)
    plot_ridge_heatmaps(rows)
    plot_parameter_slices(rows)

    print("\n" + "=" * 88)
    print(f"[{PHASE}] COMPLETE")
    print("=" * 88)
    print(f"[{PHASE}] Wrote CSV:     {OUT_CSV}")
    print(f"[{PHASE}] Wrote summary: {OUT_JSON}")
    print(f"[{PHASE}] Wrote plot:    {OUT_TOP_PNG}")
    print(f"[{PHASE}] Wrote plot:    {OUT_HEATMAP_PNG}")
    print(f"[{PHASE}] Wrote plot:    {OUT_SLICES_PNG}")

    print(f"\n[{PHASE}] Top {min(TOP_N_PRINT, len(rows))} by robust score:")
    for i, r in enumerate(rows[:TOP_N_PRINT], 1):
        print(
            f"  {i:02d}. {r['case'][:70]:70s} "
            f"robust={r['score_robust']:.8f} "
            f"mean={r['score_mean']:.8f} "
            f"std={r['score_std']:.5f} "
            f"cap={r['capture_rate_mean'] if r['capture_rate_mean'] is not None else float('nan'):.4f} "
            f"tan={r['tangent_ratio_mean'] if r['tangent_ratio_mean'] is not None else float('nan'):.3f} "
            f"r={r['BOWL_RADIUS_FRAC']:.3f} "
            f"seat={r['BOWL_SEAT_AXIS_GAIN']:.3f} "
            f"blend={r['BOWL_DIRECTIONAL_BLEND']:.3f} "
            f"capmult={r['BOWL_NORM_CAP_MULT']:.3f} "
            f"shell={r['BOWL_SHELL_RADIAL_GAIN']:.3f} "
            f"tk={r['BOWL_TANGENT_KILL']:.3f}"
        )

    best = rows[0]
    print(f"\n[{PHASE}] Best robust candidate:")
    print(json.dumps(compact_row(best), indent=2))


if __name__ == "__main__":
    main()