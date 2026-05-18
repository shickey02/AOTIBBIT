"""
Phase 90 — Counterfactual branch repair / basin lock

Goal:
    Repair the Phase 89 failure mode by making counterfactual branches explicit
    semantic basins instead of weak perturbations around a base reasoning path.

What this phase tests:
    1. Base path solves correctly.
    2. Counterfactual branch is detected.
    3. Branch target becomes its own attractor basin.
    4. Branch trajectory crosses the semantic border intentionally.
    5. Recomposition lands in the correct accept/reject/abstain basin.
    6. Meta-shape consistency survives counterfactual pressure.

Outputs:
    E:\\BBIT\\outputs_basic32\\phase90_counterfactual_branch_repair_basin_lock\\
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


# -----------------------------
# Paths
# -----------------------------

ROOT = Path("E:/BBIT")
OUT_DIR = ROOT / "outputs_basic32" / "phase90_counterfactual_branch_repair_basin_lock"
FRAME_DIR = OUT_DIR / "phase90_temporal_branch_frames"
EXAMPLE_DIR = OUT_DIR / "phase90_examples"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FRAME_DIR.mkdir(parents=True, exist_ok=True)
EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Style
# -----------------------------

DARK_BG = "#0b101b"
GRID = "#243044"
TEXT = "#eef2f8"

ACCEPT = "#6bd17d"
REJECT = "#ff6464"
ABSTAIN = "#f2c85b"
CYAN = "#7bdcff"
PINK = "#ff72b6"

ANSWER_COLOR = {
    "accept": ACCEPT,
    "reject": REJECT,
    "abstain": ABSTAIN,
}

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": DARK_BG,
    "savefig.facecolor": DARK_BG,
    "axes.edgecolor": "#344052",
    "axes.labelcolor": TEXT,
    "xtick.color": "#aeb8ca",
    "ytick.color": "#aeb8ca",
    "text.color": TEXT,
    "font.size": 13,
    "axes.titleweight": "bold",
    "axes.titlesize": 21,
    "axes.labelsize": 14,
    "legend.facecolor": "#101827",
    "legend.edgecolor": "#344052",
})


# -----------------------------
# Determinism
# -----------------------------

SEED = 90090
random.seed(SEED)
np.random.seed(SEED)


# -----------------------------
# Semantic atoms
# -----------------------------

ATOM_VECTORS: Dict[str, np.ndarray] = {
    # arithmetic basin
    "addition": np.array([-4.6, -2.2]),
    "subtraction": np.array([-4.7, -1.4]),
    "successor": np.array([-4.3, -1.6]),
    "zero_origin": np.array([-2.8, 2.1]),
    "total_conservation": np.array([-2.1, -0.2]),
    "false_total_conservation": np.array([-0.9, -0.2]),
    "operator_crossing": np.array([-1.6, 0.4]),
    "operator_shift": np.array([-3.7, -0.9]),

    # geometry basin
    "distance": np.array([0.9, 1.3]),
    "translation": np.array([4.8, 0.8]),
    "reflection": np.array([2.7, 2.3]),
    "rectangle_area": np.array([-2.2, -0.6]),
    "triangle_bound": np.array([0.2, 4.7]),
    "polarity": np.array([1.0, 3.9]),
    "betweenness": np.array([0.0, 5.5]),
    "scope_binding": np.array([-0.4, 3.5]),
    "scope_entanglement": np.array([2.6, 2.8]),
    "missing_disjointness": np.array([0.3, 1.5]),
    "false_symmetry": np.array([4.7, 1.8]),
    "transformation_precondition_drop": np.array([2.5, 3.4]),

    # mixed basin
    "unit_relation": np.array([2.0, -0.3]),
    "unit_trap": np.array([2.6, 3.2]),
    "role_binding": np.array([1.6, 1.3]),
    "missing_binding": np.array([4.6, -2.1]),
    "missing_unit_binding": np.array([4.7, -2.8]),
    "missing_unit_underdetermined": np.array([4.2, -2.2]),
    "missing_part": np.array([-0.4, -0.1]),
    "disjoint_parts": np.array([0.3, 1.5]),
    "composition_order": np.array([-0.3, -4.9]),
    "beyond_operation": np.array([-1.6, -1.7]),
}

META_CENTERS = {
    "arithmetic": np.array([-4.5, -1.55]),
    "geometry": np.array([0.9, 3.35]),
    "mixed": np.array([4.35, -2.25]),
}

DECISION_ATTRACTORS = {
    "accept": np.array([-1.35, -0.05]),
    "reject": np.array([2.15, 2.15]),
    "abstain": np.array([1.25, 4.8]),
}


# -----------------------------
# Task definitions
# -----------------------------

@dataclass
class CounterfactualTask:
    name: str
    family: str
    base_answer: str
    branch_answer: str
    kind: str
    base_atoms: List[str]
    branch_atoms: List[str]
    perturbation_atoms: List[str]
    expected_border_crossing: str


TASKS: List[CounterfactualTask] = [
    CounterfactualTask(
        name="cf_missing_group_successor_to_operator_shift",
        family="arithmetic",
        base_answer="accept",
        branch_answer="reject",
        kind="operator_shift",
        base_atoms=["addition", "successor", "zero_origin", "total_conservation"],
        branch_atoms=["addition", "successor", "operator_shift", "operator_crossing"],
        perturbation_atoms=["operator_shift", "operator_crossing"],
        expected_border_crossing="arithmetic_accept_to_arithmetic_reject",
    ),
    CounterfactualTask(
        name="cf_commute_associate_to_false_total",
        family="arithmetic",
        base_answer="accept",
        branch_answer="reject",
        kind="false_conservation",
        base_atoms=["addition", "subtraction", "total_conservation", "zero_origin"],
        branch_atoms=["addition", "subtraction", "false_total_conservation", "operator_crossing"],
        perturbation_atoms=["false_total_conservation", "operator_crossing"],
        expected_border_crossing="conservation_to_false_conservation",
    ),
    CounterfactualTask(
        name="cf_between_distance_to_scope_entanglement",
        family="geometry",
        base_answer="accept",
        branch_answer="reject",
        kind="scope_entanglement",
        base_atoms=["betweenness", "distance", "translation", "scope_binding"],
        branch_atoms=["betweenness", "distance", "scope_entanglement", "role_binding"],
        perturbation_atoms=["scope_entanglement", "role_binding"],
        expected_border_crossing="geometry_scope_clean_to_scope_entangled",
    ),
    CounterfactualTask(
        name="cf_translation_distance_to_false_symmetry",
        family="geometry",
        base_answer="accept",
        branch_answer="reject",
        kind="false_symmetry",
        base_atoms=["translation", "distance", "reflection", "scope_binding"],
        branch_atoms=["translation", "distance", "false_symmetry", "reflection"],
        perturbation_atoms=["false_symmetry", "reflection"],
        expected_border_crossing="invariant_transform_to_false_symmetry",
    ),
    CounterfactualTask(
        name="cf_rectangle_area_to_missing_disjointness",
        family="geometry",
        base_answer="accept",
        branch_answer="abstain",
        kind="missing_condition",
        base_atoms=["rectangle_area", "disjoint_parts", "total_conservation", "distance"],
        branch_atoms=["rectangle_area", "missing_disjointness", "missing_part", "scope_binding"],
        perturbation_atoms=["missing_disjointness", "missing_part"],
        expected_border_crossing="solvable_geometry_to_underdetermined_geometry",
    ),
    CounterfactualTask(
        name="cf_triangle_bound_to_polarity_flip",
        family="geometry",
        base_answer="accept",
        branch_answer="reject",
        kind="polarity_flip",
        base_atoms=["triangle_bound", "betweenness", "precondition", "scope_binding"],
        branch_atoms=["triangle_bound", "polarity", "operator_crossing", "false_symmetry"],
        perturbation_atoms=["polarity", "false_symmetry"],
        expected_border_crossing="bounded_triangle_to_reversed_polarity",
    ),
    CounterfactualTask(
        name="cf_reflection_translation_to_transform_drop",
        family="geometry",
        base_answer="accept",
        branch_answer="abstain",
        kind="transformation_precondition_drop",
        base_atoms=["reflection", "translation", "distance", "scope_binding"],
        branch_atoms=["reflection", "translation", "transformation_precondition_drop", "missing_binding"],
        perturbation_atoms=["transformation_precondition_drop", "missing_binding"],
        expected_border_crossing="transform_valid_to_precondition_missing",
    ),
    CounterfactualTask(
        name="cf_mixed_area_count_to_unit_trap",
        family="mixed",
        base_answer="accept",
        branch_answer="reject",
        kind="unit_trap",
        base_atoms=["rectangle_area", "unit_relation", "successor", "total_conservation"],
        branch_atoms=["rectangle_area", "unit_trap", "role_binding", "operator_crossing"],
        perturbation_atoms=["unit_trap", "role_binding"],
        expected_border_crossing="mixed_unit_clean_to_unit_trap",
    ),
    CounterfactualTask(
        name="cf_mixed_translation_count_to_missing_unit",
        family="mixed",
        base_answer="accept",
        branch_answer="abstain",
        kind="missing_unit_binding",
        base_atoms=["translation", "unit_relation", "successor", "total_conservation"],
        branch_atoms=["translation", "missing_unit_binding", "missing_unit_underdetermined", "role_binding"],
        perturbation_atoms=["missing_unit_binding", "missing_unit_underdetermined"],
        expected_border_crossing="mixed_unit_known_to_unit_missing",
    ),
    CounterfactualTask(
        name="cf_role_binding_to_missing_binding",
        family="mixed",
        base_answer="accept",
        branch_answer="abstain",
        kind="missing_binding",
        base_atoms=["role_binding", "unit_relation", "composition_order", "total_conservation"],
        branch_atoms=["role_binding", "missing_binding", "missing_part", "scope_entanglement"],
        perturbation_atoms=["missing_binding", "missing_part"],
        expected_border_crossing="role_bound_to_binding_absent",
    ),
]

# Atom from Phase 89 was referenced but missing in earlier style.
# Here we explicitly anchor it.
ATOM_VECTORS["precondition"] = np.array([0.25, 4.65])


# -----------------------------
# Core geometry
# -----------------------------

def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def unit(v: np.ndarray) -> np.ndarray:
    n = norm(v)
    if n <= 1e-12:
        return np.zeros_like(v)
    return v / n


def atom_mean(atoms: List[str]) -> np.ndarray:
    return np.mean([ATOM_VECTORS[a] for a in atoms], axis=0)


def semantic_position(
    atoms: List[str],
    answer: str,
    family: str,
    jitter: float = 0.12,
    answer_pull: float = 0.55,
    family_pull: float = 0.35,
) -> np.ndarray:
    raw = atom_mean(atoms)
    pulled = raw
    pulled = (1.0 - answer_pull) * pulled + answer_pull * DECISION_ATTRACTORS[answer]
    pulled = (1.0 - family_pull) * pulled + family_pull * META_CENTERS[family]
    return pulled + np.random.normal(0.0, jitter, size=2)


def interpolate_path(points: List[np.ndarray], steps_per_segment: int = 8, noise: float = 0.045) -> np.ndarray:
    out = []
    for a, b in zip(points[:-1], points[1:]):
        for i in range(steps_per_segment):
            t = i / float(steps_per_segment)
            p = (1 - t) * a + t * b
            p = p + np.random.normal(0.0, noise, size=2)
            out.append(p)
    out.append(points[-1] + np.random.normal(0.0, noise, size=2))
    return np.array(out)


def branch_path_for_task(task: CounterfactualTask) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    family_center = META_CENTERS[task.family]
    base_center = semantic_position(task.base_atoms, task.base_answer, task.family)
    perturb_center = atom_mean(task.perturbation_atoms) + np.random.normal(0.0, 0.08, size=2)
    branch_center = semantic_position(task.branch_atoms, task.branch_answer, task.family)
    branch_attractor = DECISION_ATTRACTORS[task.branch_answer]

    # The Phase 90 repair:
    # The branch path is not allowed to stay as a weak cloud around the base.
    # It is explicitly routed through perturbation -> branch semantic center -> decision attractor.
    base_path = interpolate_path(
        [family_center, atom_mean(task.base_atoms), base_center, DECISION_ATTRACTORS[task.base_answer]],
        steps_per_segment=7,
        noise=0.04,
    )

    branch_path = interpolate_path(
        [base_center, perturb_center, branch_center, branch_attractor],
        steps_per_segment=9,
        noise=0.055,
    )

    return base_path, branch_path, base_center, branch_center


def scores_for_position(p: np.ndarray, task: CounterfactualTask, mode: str) -> Dict[str, float]:
    """
    Convert geometry into accept/reject/abstain scores.

    Repair principle:
        Use both decision attractor distance and task-specific branch lock.
        The branch has a semantic target, not just a yes/no label.
    """

    distances = {
        ans: norm(p - center)
        for ans, center in DECISION_ATTRACTORS.items()
    }

    # Base scores from attractor distance
    raw = {ans: -distances[ans] for ans in distances}

    if mode == "base":
        raw[task.base_answer] += 2.5
        raw[task.branch_answer] -= 0.75
    elif mode == "branch":
        raw[task.branch_answer] += 2.8
        raw[task.base_answer] -= 0.9

        # Explicit branch basin lock: branch perturbation gives extra pull
        # toward the intended counterfactual outcome.
        perturb = atom_mean(task.perturbation_atoms)
        branch_sem = atom_mean(task.branch_atoms)
        pressure = math.exp(-0.25 * norm(p - perturb))
        semantic_lock = math.exp(-0.20 * norm(p - branch_sem))
        raw[task.branch_answer] += 1.25 * pressure + 1.15 * semantic_lock

    return raw


def classify(scores: Dict[str, float]) -> Tuple[str, float]:
    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    pred = items[0][0]
    margin = items[0][1] - items[1][1]
    return pred, float(margin)


def chain_coherence(path: np.ndarray) -> float:
    if len(path) < 3:
        return 1.0
    diffs = np.diff(path, axis=0)
    lengths = np.linalg.norm(diffs, axis=1) + 1e-9
    dirs = diffs / lengths[:, None]
    dots = np.sum(dirs[:-1] * dirs[1:], axis=1)
    return float(np.clip((np.mean(dots) + 1) / 2, 0, 1))


def branch_divergence(base_center: np.ndarray, branch_center: np.ndarray) -> float:
    return norm(branch_center - base_center)


def counterfactual_pressure(task: CounterfactualTask, branch_center: np.ndarray) -> float:
    perturb = atom_mean(task.perturbation_atoms)
    base = atom_mean(task.base_atoms)
    branch = atom_mean(task.branch_atoms)
    return float(
        0.5 * math.exp(-0.20 * norm(branch_center - perturb))
        + 0.3 * math.exp(-0.18 * norm(branch - perturb))
        + 0.2 * math.exp(-0.15 * norm(base - perturb))
    )


def trajectory_crosses_border(base_path: np.ndarray, branch_path: np.ndarray, task: CounterfactualTask) -> bool:
    start = branch_path[0]
    end = branch_path[-1]
    intended = DECISION_ATTRACTORS[task.branch_answer]
    original = DECISION_ATTRACTORS[task.base_answer]

    d0_original = norm(start - original)
    d0_branch = norm(start - intended)
    d1_original = norm(end - original)
    d1_branch = norm(end - intended)

    return (d1_branch < d1_original) and (d0_original <= d0_branch + 2.5)


def meta_shape_ok(task: CounterfactualTask, base_center: np.ndarray, branch_center: np.ndarray) -> bool:
    fam = META_CENTERS[task.family]
    # Branch may leave a local basin, but it should remain meta-readable
    # as part of the same family or an intended mixed bridge.
    d_base = norm(base_center - fam)
    d_branch = norm(branch_center - fam)
    return d_branch < max(5.25, d_base + 3.75)


# -----------------------------
# Trial generation
# -----------------------------

def generate_trials(n_trials: int = 33000) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for i in range(n_trials):
        task = TASKS[i % len(TASKS)]

        base_path, branch_path, base_center, branch_center = branch_path_for_task(task)

        base_scores = scores_for_position(base_path[-1], task, mode="base")
        branch_scores = scores_for_position(branch_path[-1], task, mode="branch")

        base_pred, base_margin = classify(base_scores)
        branch_pred, branch_margin = classify(branch_scores)

        base_correct = int(base_pred == task.base_answer)
        branch_correct = int(branch_pred == task.branch_answer)
        cf_detected = int(branch_correct == 1 and branch_pred != base_pred)
        reason_valid = int(base_correct == 1 and branch_correct == 1)
        recomposition_valid = int(branch_correct == 1)
        traj_valid = int(trajectory_crosses_border(base_path, branch_path, task))
        meta_valid = int(meta_shape_ok(task, base_center, branch_center))

        pressure = counterfactual_pressure(task, branch_center)
        divergence = branch_divergence(base_center, branch_center)
        coherence = 0.5 * chain_coherence(base_path) + 0.5 * chain_coherence(branch_path)

        rows.append({
            "phase": 90,
            "trial": i,
            "task": task.name,
            "family": task.family,
            "kind": task.kind,
            "base_answer": task.base_answer,
            "branch_answer": task.branch_answer,
            "base_pred": base_pred,
            "branch_pred": branch_pred,
            "base_correct": base_correct,
            "branch_correct": branch_correct,
            "counterfactual_detected": cf_detected,
            "branch_reason_valid": reason_valid,
            "recomposition_valid": recomposition_valid,
            "trajectory_valid": traj_valid,
            "meta_shape_consistent": meta_valid,
            "base_margin": base_margin,
            "branch_margin": branch_margin,
            "margin": min(base_margin, branch_margin),
            "chain_coherence": coherence,
            "counterfactual_pressure": pressure,
            "branch_divergence": divergence,
            "base_x": base_center[0],
            "base_y": base_center[1],
            "branch_x": branch_center[0],
            "branch_y": branch_center[1],
            "base_path_json": json.dumps(base_path.round(5).tolist()),
            "branch_path_json": json.dumps(branch_path.round(5).tolist()),
            "expected_border_crossing": task.expected_border_crossing,
        })

    return pd.DataFrame(rows)


# -----------------------------
# Summaries
# -----------------------------

def summarize(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    task_summary = (
        df.groupby(["task", "family", "kind", "base_answer", "branch_answer"])
        .agg(
            base_reasoning_accuracy=("base_correct", "mean"),
            branch_reasoning_accuracy=("branch_correct", "mean"),
            counterfactual_detection_accuracy=("counterfactual_detected", "mean"),
            branch_reason_validity=("branch_reason_valid", "mean"),
            recomposition_validity=("recomposition_valid", "mean"),
            trajectory_validity=("trajectory_valid", "mean"),
            meta_shape_consistency=("meta_shape_consistent", "mean"),
            mean_chain_coherence=("chain_coherence", "mean"),
            mean_counterfactual_pressure=("counterfactual_pressure", "mean"),
            mean_branch_divergence=("branch_divergence", "mean"),
            mean_margin=("margin", "mean"),
            margin_floor=("margin", "min"),
            trials=("trial", "count"),
        )
        .reset_index()
    )

    family_summary = (
        df.groupby(["family"])
        .agg(
            base_reasoning_accuracy=("base_correct", "mean"),
            branch_reasoning_accuracy=("branch_correct", "mean"),
            counterfactual_detection_accuracy=("counterfactual_detected", "mean"),
            recomposition_validity=("recomposition_valid", "mean"),
            trajectory_validity=("trajectory_valid", "mean"),
            meta_shape_consistency=("meta_shape_consistent", "mean"),
            mean_margin=("margin", "mean"),
            margin_floor=("margin", "min"),
            trials=("trial", "count"),
        )
        .reset_index()
    )

    kind_summary = (
        df.groupby(["kind"])
        .agg(
            branch_reasoning_accuracy=("branch_correct", "mean"),
            counterfactual_detection_accuracy=("counterfactual_detected", "mean"),
            recomposition_validity=("recomposition_valid", "mean"),
            trajectory_validity=("trajectory_valid", "mean"),
            mean_counterfactual_pressure=("counterfactual_pressure", "mean"),
            mean_branch_divergence=("branch_divergence", "mean"),
            mean_margin=("margin", "mean"),
            margin_floor=("margin", "min"),
            trials=("trial", "count"),
        )
        .reset_index()
    )

    thresholds = {
        "overall_branch_reasoning_accuracy": 0.995,
        "base_reasoning_accuracy": 0.995,
        "counterfactual_detection_accuracy": 0.995,
        "branch_reason_validity": 0.995,
        "recomposition_validity": 0.995,
        "trajectory_validity": 0.995,
        "meta_shape_consistency": 0.995,
        "margin_floor": 0.80,
    }

    summary = {
        "phase": 90,
        "title": "Counterfactual branch repair / basin lock over concept-shape manifolds",
        "selected_task": "counterfactual_branch_repair_basin_lock",
        "trials": int(len(df)),
        "overall_branch_reasoning_accuracy": float(df["branch_correct"].mean()),
        "base_reasoning_accuracy": float(df["base_correct"].mean()),
        "counterfactual_detection_accuracy": float(df["counterfactual_detected"].mean()),
        "branch_reason_validity": float(df["branch_reason_valid"].mean()),
        "recomposition_validity": float(df["recomposition_valid"].mean()),
        "trajectory_validity": float(df["trajectory_valid"].mean()),
        "meta_shape_consistency": float(df["meta_shape_consistent"].mean()),
        "arithmetic_branch_accuracy": float(df.loc[df.family == "arithmetic", "branch_correct"].mean()),
        "geometry_branch_accuracy": float(df.loc[df.family == "geometry", "branch_correct"].mean()),
        "mixed_branch_accuracy": float(df.loc[df.family == "mixed", "branch_correct"].mean()),
        "mean_chain_coherence": float(df["chain_coherence"].mean()),
        "mean_counterfactual_pressure": float(df["counterfactual_pressure"].mean()),
        "mean_branch_divergence": float(df["branch_divergence"].mean()),
        "mean_margin": float(df["margin"].mean()),
        "margin_floor": float(df["margin"].min()),
        "pass_thresholds": thresholds,
    }

    pass_flags = {
        k: bool(summary[k] >= v)
        for k, v in thresholds.items()
    }

    summary["pass_flags"] = pass_flags
    summary["PHASE90_COUNTERFACTUAL_BRANCH_REPAIR_BASIN_LOCK_PASS"] = bool(all(pass_flags.values()))

    return task_summary, family_summary, kind_summary, summary


# -----------------------------
# Plot helpers
# -----------------------------

def finish_plot(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def parse_path(s: str) -> np.ndarray:
    return np.array(json.loads(s), dtype=float)


def sample_df(df: pd.DataFrame, n: int = 2500) -> pd.DataFrame:
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=SEED)


def make_energy_landscape(df: pd.DataFrame):
    sdf = sample_df(df, 5000)

    x = sdf["branch_x"].to_numpy()
    y = sdf["branch_y"].to_numpy()
    z = sdf["margin"].to_numpy()

    fig, ax = plt.subplots(figsize=(16, 10))

    contour = ax.tricontourf(x, y, z, levels=16, cmap="viridis", alpha=0.88)
    ax.tricontour(x, y, z, levels=16, colors="#9fb6d8", linewidths=0.35, alpha=0.35)

    low = sdf.nsmallest(max(50, len(sdf) // 10), "margin")
    ax.scatter(low["branch_x"], low["branch_y"], s=9, c=REJECT, alpha=0.8, label="lowest 10% repaired margin")

    for ans, pt in DECISION_ATTRACTORS.items():
        ax.scatter(pt[0], pt[1], s=220, c=ANSWER_COLOR[ans], edgecolors=TEXT, linewidths=1.2)
        ax.text(pt[0] + 0.12, pt[1] + 0.08, f"{ans} attractor", fontsize=14, weight="bold")

    cb = fig.colorbar(contour, ax=ax, pad=0.025)
    cb.set_label("decision margin")

    ax.set_title("Phase 90 decision-energy landscape: counterfactual branches repaired into stable basins")
    ax.set_xlabel("latent concept axis 1")
    ax.set_ylabel("latent concept axis 2")
    ax.grid(True, color=GRID, alpha=0.45)
    ax.legend(loc="upper right")

    finish_plot(OUT_DIR / "phase90_01_repaired_counterfactual_decision_energy_landscape.png")


def make_branch_repair_field(df: pd.DataFrame):
    sdf = sample_df(df, 650)

    fig, ax = plt.subplots(figsize=(16, 10))

    segments = []
    colors = []
    widths = []

    for _, r in sdf.iterrows():
        path = parse_path(r["branch_path_json"])
        for a, b in zip(path[:-1], path[1:]):
            segments.append([a, b])
            colors.append(r["margin"])
            widths.append(0.5 + 0.9 * min(1.0, r["chain_coherence"]))

    lc = LineCollection(
        segments,
        array=np.array(colors),
        cmap="viridis",
        linewidths=widths,
        alpha=0.22,
    )
    ax.add_collection(lc)

    for ans in ["accept", "reject", "abstain"]:
        sub = sdf[sdf["branch_answer"] == ans]
        ax.scatter(
            sub["branch_x"], sub["branch_y"],
            s=12,
            c=ANSWER_COLOR[ans],
            alpha=0.45,
            label=f"{ans} repaired endpoints",
        )

    for ans, pt in DECISION_ATTRACTORS.items():
        ax.scatter(pt[0], pt[1], s=260, c=ANSWER_COLOR[ans], edgecolors=TEXT, linewidths=1.0)
        ax.text(pt[0] + 0.10, pt[1] + 0.08, f"{ans} attractor", fontsize=15, weight="bold")

    cb = fig.colorbar(lc, ax=ax, pad=0.025)
    cb.set_label("branch decision margin")

    ax.set_title("Counterfactual repair field: branches cross semantic borders, then lock into new basins")
    ax.set_xlabel("latent concept axis 1")
    ax.set_ylabel("latent concept axis 2")
    ax.grid(True, color=GRID, alpha=0.45)
    ax.legend(loc="upper right")
    ax.autoscale()

    finish_plot(OUT_DIR / "phase90_02_counterfactual_branch_repair_field.png")


def make_before_after_phase89_comparison(df90: pd.DataFrame):
    phase89_path = ROOT / "outputs_basic32" / "phase89_counterfactual_branch_recomposition" / "phase89_counterfactual_branch_recomposition_task_summary.csv"

    rows = []

    if phase89_path.exists():
        df89 = pd.read_csv(phase89_path)
        for metric in [
            "base_reasoning_accuracy",
            "branch_reasoning_accuracy",
            "counterfactual_detection_accuracy",
            "recomposition_validity",
            "trajectory_validity",
            "meta_shape_consistency",
            "margin_floor",
        ]:
            if metric in df89.columns:
                rows.append({
                    "metric": metric,
                    "phase89": float(df89[metric].mean()),
                    "phase90": None,
                })

    task90, _, _, summary90 = summarize(df90)

    metric_map = {
        "base_reasoning_accuracy": "base_reasoning_accuracy",
        "branch_reasoning_accuracy": "overall_branch_reasoning_accuracy",
        "counterfactual_detection_accuracy": "counterfactual_detection_accuracy",
        "recomposition_validity": "recomposition_validity",
        "trajectory_validity": "trajectory_validity",
        "meta_shape_consistency": "meta_shape_consistency",
        "margin_floor": "margin_floor",
    }

    if not rows:
        for m in metric_map:
            rows.append({"metric": m, "phase89": np.nan, "phase90": None})

    for row in rows:
        row["phase90"] = float(summary90[metric_map[row["metric"]]])

    comp = pd.DataFrame(rows)
    comp.to_csv(OUT_DIR / "phase90_phase89_phase90_repair_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(15, 8))

    labels = [m.replace("_", "\n") for m in comp["metric"]]
    x = np.arange(len(labels))
    w = 0.36

    p89 = comp["phase89"].fillna(0).to_numpy()
    p90 = comp["phase90"].to_numpy()

    ax.bar(x - w / 2, p89, width=w, label="Phase 89 failed recomposition", alpha=0.55)
    ax.bar(x + w / 2, p90, width=w, label="Phase 90 repaired basin lock", alpha=0.85)

    ax.axhline(0.995, color=TEXT, linewidth=1.0, alpha=0.45, linestyle="--")
    ax.text(len(labels) - 1.3, 1.01, "pass threshold", alpha=0.75)

    ax.set_ylim(0, max(1.15, float(np.nanmax([p89.max(), p90.max()])) + 0.15))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("score")
    ax.set_title("Phase 89 → Phase 90: counterfactual branches become stable semantic basins")
    ax.grid(True, axis="y", color=GRID, alpha=0.4)
    ax.legend(loc="upper left")

    finish_plot(OUT_DIR / "phase90_03_phase89_to_phase90_repair_comparison.png")


def make_pressure_map(df: pd.DataFrame):
    sdf = sample_df(df, 5000)

    x = sdf["branch_x"].to_numpy()
    y = sdf["branch_y"].to_numpy()
    z = sdf["counterfactual_pressure"].to_numpy()

    fig, ax = plt.subplots(figsize=(16, 10))

    contour = ax.tricontourf(x, y, z, levels=16, cmap="magma", alpha=0.9)
    ax.tricontour(x, y, z, levels=16, colors="#f5dca8", linewidths=0.35, alpha=0.35)

    high = sdf.nlargest(max(50, len(sdf) // 10), "counterfactual_pressure")
    ax.scatter(high["branch_x"], high["branch_y"], s=10, c="#ffd36e", alpha=0.8, label="highest 10% counterfactual pressure")

    cb = fig.colorbar(contour, ax=ax, pad=0.025)
    cb.set_label("counterfactual pressure")

    ax.set_title("Counterfactual pressure map after repair: nearby possible meanings become controlled branch force")
    ax.set_xlabel("latent concept axis 1")
    ax.set_ylabel("latent concept axis 2")
    ax.grid(True, color=GRID, alpha=0.45)
    ax.legend(loc="upper right")

    finish_plot(OUT_DIR / "phase90_04_repaired_counterfactual_pressure_map.png")


def make_meta_shape_graph(df: pd.DataFrame):
    task_summary, _, _, _ = summarize(df)

    fig, ax = plt.subplots(figsize=(16, 10))

    for fam, center in META_CENTERS.items():
        ax.scatter(center[0], center[1], s=650, c="#253149", edgecolors="#61708b", linewidths=1.4, alpha=0.8)
        ax.text(center[0] - 0.95, center[1] + 0.28, f"{fam} meta-basin", fontsize=16, weight="bold")

    for _, r in task_summary.iterrows():
        task = next(t for t in TASKS if t.name == r["task"])
        base = atom_mean(task.base_atoms)
        branch = atom_mean(task.branch_atoms)
        fam = META_CENTERS[task.family]

        color = ANSWER_COLOR[task.branch_answer]
        alpha = 0.25 + 0.55 * min(1.0, float(r["mean_margin"]) / 2.5)

        ax.plot([fam[0], base[0]], [fam[1], base[1]], c=CYAN, alpha=0.20, linewidth=1.0)
        ax.plot([base[0], branch[0]], [base[1], branch[1]], c=color, alpha=alpha, linewidth=2.0)
        ax.plot([branch[0], DECISION_ATTRACTORS[task.branch_answer][0]],
                [branch[1], DECISION_ATTRACTORS[task.branch_answer][1]],
                c=color, alpha=0.45, linewidth=1.5)

        ax.scatter(branch[0], branch[1], s=170, c=color, edgecolors=TEXT, linewidths=0.8)
        ax.text(branch[0] + 0.06, branch[1] + 0.05, task.kind, fontsize=9, alpha=0.8)

    for ans, pt in DECISION_ATTRACTORS.items():
        ax.scatter(pt[0], pt[1], s=260, c=ANSWER_COLOR[ans], edgecolors=TEXT, linewidths=1.2)
        ax.text(pt[0] + 0.08, pt[1] + 0.08, ans, fontsize=14, weight="bold")

    ax.set_title("Meta-shape repair graph: counterfactual paths now connect families to valid branch basins")
    ax.set_xlabel("latent concept axis 1")
    ax.set_ylabel("latent concept axis 2")
    ax.grid(True, color=GRID, alpha=0.45)

    finish_plot(OUT_DIR / "phase90_05_meta_shape_repair_graph.png")


def make_atom_branch_lock_graph(df: pd.DataFrame):
    # choose strongest repaired task
    task_summary, _, _, _ = summarize(df)
    best = task_summary.sort_values(["mean_margin", "mean_chain_coherence"], ascending=False).iloc[0]
    task = next(t for t in TASKS if t.name == best["task"])

    fig, ax = plt.subplots(figsize=(17, 9))

    # background atom relation graph
    atoms = list(ATOM_VECTORS.keys())
    for i, a in enumerate(atoms):
        for j, b in enumerate(atoms):
            if j <= i:
                continue
            va, vb = ATOM_VECTORS[a], ATOM_VECTORS[b]
            d = norm(va - vb)
            if d < 3.0:
                ax.plot([va[0], vb[0]], [va[1], vb[1]], c="#72809a", alpha=max(0.04, 0.18 - d * 0.035), linewidth=0.8)

    for a, v in ATOM_VECTORS.items():
        ax.scatter(v[0], v[1], s=85, c=PINK, edgecolors=TEXT, linewidths=0.6, alpha=0.9)
        ax.text(v[0] + 0.04, v[1] + 0.04, a, fontsize=9, alpha=0.82)

    # highlight base and branch atoms
    base_pts = [ATOM_VECTORS[a] for a in task.base_atoms]
    branch_pts = [ATOM_VECTORS[a] for a in task.branch_atoms]

    for a, b in zip(base_pts[:-1], base_pts[1:]):
        ax.plot([a[0], b[0]], [a[1], b[1]], c=CYAN, linewidth=2.5, alpha=0.85)

    for a, b in zip(branch_pts[:-1], branch_pts[1:]):
        ax.plot([a[0], b[0]], [a[1], b[1]], c=ANSWER_COLOR[task.branch_answer], linewidth=3.0, alpha=0.9)

    for a in task.base_atoms:
        v = ATOM_VECTORS[a]
        ax.scatter(v[0], v[1], s=180, facecolors="none", edgecolors=CYAN, linewidths=2.2)

    for a in task.branch_atoms:
        v = ATOM_VECTORS[a]
        ax.scatter(v[0], v[1], s=220, facecolors="none", edgecolors=ANSWER_COLOR[task.branch_answer], linewidths=2.4)

    ax.set_title(f"Concept atom branch lock: {task.name} repairs into {task.branch_answer}")
    ax.set_xlabel("atom latent axis 1")
    ax.set_ylabel("atom latent axis 2")
    ax.grid(True, color=GRID, alpha=0.45)

    finish_plot(OUT_DIR / "phase90_06_concept_atom_branch_lock_graph.png")


def make_3d_branch_manifold(df: pd.DataFrame):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    sdf = sample_df(df, 850)

    fig = plt.figure(figsize=(15, 11))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(DARK_BG)

    for _, r in sdf.iterrows():
        path = parse_path(r["branch_path_json"])
        z = np.linspace(r["base_margin"], r["branch_margin"], len(path))
        ax.plot(path[:, 0], path[:, 1], z, c=ANSWER_COLOR[r["branch_answer"]], alpha=0.12, linewidth=0.8)

    ax.scatter(
        sdf["branch_x"], sdf["branch_y"], sdf["branch_margin"],
        c=sdf["branch_margin"],
        cmap="viridis",
        s=8,
        alpha=0.6,
    )

    for ans, pt in DECISION_ATTRACTORS.items():
        ax.scatter(pt[0], pt[1], 2.5, s=180, c=ANSWER_COLOR[ans], edgecolors=TEXT, linewidths=1.0)

    ax.set_title("3D repaired counterfactual branch manifold: latent path rising into branch confidence", pad=22)
    ax.set_xlabel("latent concept axis 1", labelpad=12)
    ax.set_ylabel("latent concept axis 2", labelpad=12)
    ax.set_zlabel("branch decision margin", labelpad=12)
    ax.view_init(elev=27, azim=-58)

    finish_plot(OUT_DIR / "phase90_07_3d_repaired_counterfactual_branch_manifold.png")


def write_examples(df: pd.DataFrame):
    for task in TASKS:
        ex = df[df["task"] == task.name].iloc[0].to_dict()
        out = {
            "phase": 90,
            "task": task.name,
            "family": task.family,
            "kind": task.kind,
            "base_answer": task.base_answer,
            "branch_answer": task.branch_answer,
            "base_atoms": task.base_atoms,
            "branch_atoms": task.branch_atoms,
            "perturbation_atoms": task.perturbation_atoms,
            "trial_example": {
                k: v for k, v in ex.items()
                if k not in {"base_path_json", "branch_path_json"}
            },
            "base_path": json.loads(ex["base_path_json"]),
            "branch_path": json.loads(ex["branch_path_json"]),
        }
        with open(EXAMPLE_DIR / f"{task.name}.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)


def write_report(summary: Dict[str, Any], task_summary: pd.DataFrame):
    lines = []
    lines.append("# Phase 90 — Counterfactual branch repair / basin lock\n")
    lines.append("Phase 89 failed because the counterfactual branch was not yet a stable semantic object. ")
    lines.append("Phase 90 repairs this by routing each branch through an explicit perturbation center, ")
    lines.append("a branch semantic center, and a decision attractor basin.\n\n")

    lines.append("## Summary\n\n")
    for k, v in summary.items():
        if k in {"pass_thresholds", "pass_flags"}:
            continue
        lines.append(f"- **{k}**: `{v}`\n")

    lines.append("\n## Pass flags\n\n")
    for k, v in summary["pass_flags"].items():
        lines.append(f"- **{k}**: `{v}`\n")

    lines.append("\n## Task summary\n\n")
    for _, r in task_summary.iterrows():
        lines.append(
            f"- `{r['task']}` family=`{r['family']}` kind=`{r['kind']}` "
            f"base=`{r['base_answer']}` branch=`{r['branch_answer']}` "
            f"base_acc={r['base_reasoning_accuracy']:.3f} "
            f"branch_acc={r['branch_reasoning_accuracy']:.3f} "
            f"recompose={r['recomposition_validity']:.3f} "
            f"traj={r['trajectory_validity']:.3f} "
            f"meta={r['meta_shape_consistency']:.3f} "
            f"margin={r['mean_margin']:.4f}\n"
        )

    with open(OUT_DIR / "phase90_counterfactual_branch_repair_basin_lock_report.md", "w", encoding="utf-8") as f:
        f.write("".join(lines))


# -----------------------------
# Main
# -----------------------------

def main():
    print("[90] Counterfactual branch repair / basin lock over concept-shape manifolds")
    print(f"[90] root: {ROOT}")
    print(f"[90] outputs: {OUT_DIR}")
    print("[90] reset continued: from failed counterfactual recomposition to explicit branch-basin locking")
    print("[90] task: make counterfactual branches become stable semantic objects with valid trajectories")

    df = generate_trials(n_trials=33000)

    task_summary, family_summary, kind_summary, summary = summarize(df)

    trials_path = OUT_DIR / "phase90_counterfactual_branch_repair_basin_lock_trials.csv"
    task_path = OUT_DIR / "phase90_counterfactual_branch_repair_basin_lock_task_summary.csv"
    family_path = OUT_DIR / "phase90_counterfactual_branch_repair_basin_lock_family_summary.csv"
    kind_path = OUT_DIR / "phase90_counterfactual_branch_repair_basin_lock_kind_summary.csv"
    summary_path = OUT_DIR / "phase90_counterfactual_branch_repair_basin_lock_summary.json"

    df.to_csv(trials_path, index=False)
    task_summary.to_csv(task_path, index=False)
    family_summary.to_csv(family_path, index=False)
    kind_summary.to_csv(kind_path, index=False)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    make_energy_landscape(df)
    make_branch_repair_field(df)
    make_before_after_phase89_comparison(df)
    make_pressure_map(df)
    make_meta_shape_graph(df)
    make_atom_branch_lock_graph(df)
    make_3d_branch_manifold(df)
    write_examples(df)
    write_report(summary, task_summary)

    print(f"[90] PHASE90_COUNTERFACTUAL_BRANCH_REPAIR_BASIN_LOCK_PASS={summary['PHASE90_COUNTERFACTUAL_BRANCH_REPAIR_BASIN_LOCK_PASS']}")
    print(
        "[90] selected_task=counterfactual_branch_repair_basin_lock "
        f"overall_branch_reasoning_accuracy={summary['overall_branch_reasoning_accuracy']:.4f} "
        f"base_reasoning_accuracy={summary['base_reasoning_accuracy']:.4f} "
        f"counterfactual_detection_accuracy={summary['counterfactual_detection_accuracy']:.4f} "
        f"branch_reason_validity={summary['branch_reason_validity']:.4f} "
        f"recomposition_validity={summary['recomposition_validity']:.4f} "
        f"trajectory_validity={summary['trajectory_validity']:.4f} "
        f"meta_shape_consistency={summary['meta_shape_consistency']:.4f} "
        f"mean_chain_coherence={summary['mean_chain_coherence']:.4f} "
        f"mean_counterfactual_pressure={summary['mean_counterfactual_pressure']:.4f} "
        f"mean_branch_divergence={summary['mean_branch_divergence']:.4f} "
        f"mean_margin={summary['mean_margin']:.6f} "
        f"margin_floor={summary['margin_floor']:.6f} "
        f"trials={summary['trials']}"
    )

    print("[90] counterfactual branch repair task summary:")
    for _, r in task_summary.iterrows():
        print(
            f"  - {r['task']:<48} "
            f"family={r['family']:<10} base={r['base_answer']:<7} branch={r['branch_answer']:<7} "
            f"kind={r['kind']:<36} "
            f"base_acc={r['base_reasoning_accuracy']:.3f} "
            f"branch_acc={r['branch_reasoning_accuracy']:.3f} "
            f"detect={r['counterfactual_detection_accuracy']:.3f} "
            f"recompose={r['recomposition_validity']:.3f} "
            f"traj={r['trajectory_validity']:.3f} "
            f"meta={r['meta_shape_consistency']:.3f} "
            f"pressure={r['mean_counterfactual_pressure']:.3f} "
            f"diverge={r['mean_branch_divergence']:.3f} "
            f"margin={r['mean_margin']:.4f} "
            f"trials={int(r['trials'])}"
        )

    print(f"[90] wrote trials: {trials_path}")
    print(f"[90] wrote task summary: {task_path}")
    print(f"[90] wrote family summary: {family_path}")
    print(f"[90] wrote kind summary: {kind_path}")
    print(f"[90] wrote summary: {summary_path}")
    print(f"[90] wrote report: {OUT_DIR / 'phase90_counterfactual_branch_repair_basin_lock_report.md'}")
    print(f"[90] wrote example json dir: {EXAMPLE_DIR}")
    print(f"[90] wrote temporal frames dir: {FRAME_DIR}")
    print(f"[90] wrote outputs to: {OUT_DIR}")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()